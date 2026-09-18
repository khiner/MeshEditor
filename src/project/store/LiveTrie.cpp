#include "project/store/LiveTrie.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <format>
#include <unordered_set>

namespace store {
namespace {
// Process-wide bytes held by nodes and child arrays.
std::atomic<uint64_t> NodeBytes{};
Node **AllocChildren() {
    NodeBytes.fetch_add(Fanout * sizeof(Node *), std::memory_order_relaxed);
    return new Node *[Fanout] {};
}

Node *Alloc(LiveTrie &trie, NodeKind kind) {
    NodeBytes.fetch_add(sizeof(Node), std::memory_order_relaxed);
    auto *n = new Node{1, kind, {}, nullptr, {}};
    ++trie.S.Nodes;
    if (kind == NodeKind::Aliased) ++trie.S.AliasedNodes;
    if (kind == NodeKind::Interior) {
        n->Children = AllocChildren();
    }
    return n;
}

void FreeChildren(Node **a) {
    NodeBytes.fetch_sub(Fanout * sizeof(Node *), std::memory_order_relaxed);
    delete[] a;
}

void ReleaseNode(LiveTrie &trie, Node *n) {
    if (--n->Refs) return;
    --trie.S.Nodes;
    switch (n->Kind) {
        case NodeKind::Aliased: --trie.S.AliasedNodes; break;
        case NodeKind::Absent: break;
        case NodeKind::Owned:
            --trie.S.OwnedSlots;
            trie.S.OwnedBytes -= n->Value.OwnedBytes();
            FreeBlob(n->Value);
            break;
        case NodeKind::Interior:
            for (uint32_t i = 0; i < Fanout; ++i) ReleaseNode(trie, n->Children[i]);
            FreeChildren(n->Children);
            break;
    }
    NodeBytes.fetch_sub(sizeof(Node), std::memory_order_relaxed);
    delete n;
}

// Retain adopt children or allocate aliased children, preserving the value for every referencing version.
void ToInterior(LiveTrie &trie, Node *n, Node *const *adopt = nullptr) {
    assert(n->Kind == NodeKind::Aliased);
    --trie.S.AliasedNodes;
    n->Kind = NodeKind::Interior;
    n->Children = AllocChildren();
    for (uint32_t i = 0; i < Fanout; ++i) {
        if (adopt) {
            n->Children[i] = adopt[i];
            ++adopt[i]->Refs;
        } else {
            n->Children[i] = Alloc(trie, NodeKind::Aliased);
        }
    }
}

// Use the settled hash of the slot, or hash the raw bytes of a value the slot holds in default state.
Hash128 ContentHash(const LiveTrie &trie, uint64_t slot, const Blob &value) {
    if (slot < trie.SlotHashes.size()) {
        const auto &e = trie.SlotHashes[slot];
        if (e.State == LiveTrie::SlotState::Value && !e.Dirty) return e.H;
    }
    assert(!value.Destroy && "native values require a settled hash");
    return HashBytes(value.View());
}

void CaptureInto(LiveTrie &trie, Node *n, uint64_t slot, std::optional<Blob> value) {
    assert(n->Kind == NodeKind::Aliased);
    --trie.S.AliasedNodes;
    if (value) {
        n->Kind = NodeKind::Owned;
        n->Value = *value;
        n->Hash = ContentHash(trie, slot, n->Value);
        ++trie.S.OwnedSlots;
        trie.S.OwnedBytes += n->Value.OwnedBytes();
    } else {
        n->Kind = NodeKind::Absent;
    }
}

LiveTrie::SlotHash &SlotHashAt(LiveTrie &trie, uint64_t slot) {
    if (slot >= trie.SlotHashes.size()) trie.SlotHashes.resize(slot + 1);
    return trie.SlotHashes[slot];
}

void MarkDirtySlot(LiveTrie &trie, uint64_t slot) {
    auto &e = SlotHashAt(trie, slot);
    if (!e.Dirty) {
        e.Dirty = true;
        trie.DirtySlots.push_back(slot);
    }
}

// Return the writable node, capturing leaves in [first, last] with fetch(slot).
// The caller releases its reference to n when the result differs.
Node *WriteRec(LiveTrie &trie, Node *n, uint32_t level, uint64_t base, uint64_t first, uint64_t last, bool owned, auto &&fetch) {
    const bool uniq = owned && n->Refs == 1;
    if (uniq && n->Kind == NodeKind::Aliased) return n;

    if (level == 0) {
        assert(n->Kind == NodeKind::Aliased && "the present reaches only aliased leaves");
        CaptureInto(trie, n, base, fetch(base));
        return Alloc(trie, NodeKind::Aliased);
    }

    if (n->Kind == NodeKind::Aliased) ToInterior(trie, n);
    assert(n->Kind == NodeKind::Interior);
    const auto span = SlotSpan(level - 1);
    Node *p = n;
    if (!uniq) {
        p = Alloc(trie, NodeKind::Interior);
        for (uint32_t i = 0; i < Fanout; ++i) {
            p->Children[i] = n->Children[i];
            ++n->Children[i]->Refs;
        }
    }
    const uint32_t lo = uint32_t((std::max(first, base) - base) / span);
    const uint32_t hi = uint32_t((std::min(last, base + SlotSpan(level) - 1) - base) / span);
    for (uint32_t i = lo; i <= hi; ++i) {
        auto *child = n->Children[i];
        auto *replacement = WriteRec(trie, child, level - 1, base + i * span, first, last, uniq, fetch);
        if (replacement != child) {
            ReleaseNode(trie, p->Children[i]);
            p->Children[i] = replacement;
        }
    }
    return p;
}

void CaptureImpl(LiveTrie &trie, uint64_t first, uint64_t count, auto &&fetch) {
    if (count == 0) return;
    assert(!trie.Busy && "capture between a plan and its commit");
    assert(first + count <= SlotSpan(trie.Levels));
    auto *root = WriteRec(trie, trie.Root, trie.Levels, 0, first, first + count - 1, true, fetch);
    if (root != trie.Root) {
        ReleaseNode(trie, trie.Root);
        trie.Root = root;
    }
}

void DirtyManifest(LiveTrie &trie, uint32_t level, uint64_t index) {
    auto &m = trie.Manifest[level];
    if (index >= m.Nodes.size()) m.Nodes.resize(index + 1);
    if (m.Nodes[index].Dirty) return;
    m.Nodes[index].Dirty = true;
    m.Dirty.push_back(index);
}

void RehashSlot(LiveTrie &trie, uint64_t slot, Hash128 incoming, bool incoming_default) {
    auto &e = SlotHashAt(trie, slot);
    if (incoming_default ? e.State == LiveTrie::SlotState::Value : e.State != LiveTrie::SlotState::Value || e.H != incoming)
        DirtyManifest(trie, 0, slot / Fanout);
    if (e.State == LiveTrie::SlotState::Value) {
        const Term t{slot, e.H};
        trie.Lane0 -= t.L0;
        trie.Lane1 -= t.L1;
    }
    if (incoming_default) {
        e = {{}, LiveTrie::SlotState::Default, false};
    } else {
        e = {incoming, LiveTrie::SlotState::Value, false};
        const Term t{slot, incoming};
        trie.Lane0 += t.L0;
        trie.Lane1 += t.L1;
    }
}

// Collect the leaves that differ between the present and the target version, moving owned values into the plan.
void PlanRec(LiveTrie &trie, Node *p, Node *t, uint32_t level, uint64_t base, RestorePlan &plan) {
    if (p == t) return;
    if (level == 0) {
        assert(p->Kind == NodeKind::Aliased);
        if (t->Kind == NodeKind::Aliased) return;
        SlotChange c{.Slot = base};
        if (t->Kind == NodeKind::Owned) {
            --trie.S.OwnedSlots;
            trie.S.OwnedBytes -= t->Value.OwnedBytes();
            c.Incoming = std::exchange(t->Value, {});
            const auto *e = base < trie.SlotHashes.size() ? &trie.SlotHashes[base] : nullptr;
            c.MaybeEqual = !(e && e->State == LiveTrie::SlotState::Value && !e->Dirty && !(e->H == t->Hash));
            c.Default = trie.PageBytes && (c.Incoming.Size != trie.PageBytes || IsZero(c.Incoming.View()));
        } else {
            c.Erase = true;
        }
        plan.Changes.push_back(c);
        return;
    }
    if (t->Kind == NodeKind::Aliased) return;
    assert(t->Kind == NodeKind::Interior);
    const bool aliased = p->Kind == NodeKind::Aliased;
    const auto span = SlotSpan(level - 1);
    for (uint32_t i = 0; i < Fanout; ++i) {
        auto *child = aliased ? p : p->Children[i];
        if (child != t->Children[i]) PlanRec(trie, child, t->Children[i], level - 1, base + i * span, plan);
    }
}

// Swap applied values and return the target node for the present version.
Node *CommitRec(LiveTrie &trie, Node *p, Node *t, uint32_t level, uint64_t base, bool owned, RestorePlan &plan, size_t &next) {
    if (p == t) return t;
    const bool uniq = owned && p->Refs == 1;

    if (level == 0) {
        assert(p->Kind == NodeKind::Aliased);
        if (t->Kind == NodeKind::Aliased) return t; // Both alias live data.
        auto &c = plan.Changes[next++];
        assert(c.Slot == base && "plan applied out of order");
        const auto slot = base;
        NodeKind old_kind;
        Blob old_value{};
        Hash128 old_hash{};
        if (t->Kind == NodeKind::Owned) {
            if (c.Unchanged) {
                old_value = c.Incoming;
                old_hash = t->Hash;
                old_kind = NodeKind::Owned;
            } else {
                old_value = c.Old;
                old_kind = c.WasPresent ? NodeKind::Owned : NodeKind::Absent;
                if (c.WasPresent) old_hash = ContentHash(trie, slot, old_value);
                RehashSlot(trie, slot, t->Hash, c.Default);
            }
        } else if (c.Unchanged) {
            old_kind = NodeKind::Absent;
        } else {
            assert(c.WasPresent);
            old_value = c.Old;
            old_kind = NodeKind::Owned;
            old_hash = ContentHash(trie, slot, old_value);
            RehashSlot(trie, slot, {}, true);
        }
        t->Kind = NodeKind::Aliased;
        ++trie.S.AliasedNodes;
        if (!uniq) {
            --trie.S.AliasedNodes;
            p->Kind = old_kind;
            p->Value = old_value;
            p->Hash = old_hash;
            if (old_kind == NodeKind::Owned) {
                ++trie.S.OwnedSlots;
                trie.S.OwnedBytes += old_value.OwnedBytes();
            }
        } else if (old_kind == NodeKind::Owned) {
            FreeBlob(old_value);
        }
        if (!c.Unchanged && trie.CollectChanged) trie.ChangedSlots.push_back(slot);
        return t;
    }

    if (t->Kind == NodeKind::Aliased) {
        // Preserve reachability of aliased nodes from the present root.
        if (p->Kind == NodeKind::Interior) ToInterior(trie, t, p->Children);
        return t;
    }
    assert(t->Kind == NodeKind::Interior);
    if (p->Kind == NodeKind::Aliased) ToInterior(trie, p);
    assert(p->Kind == NodeKind::Interior);
    const auto span = SlotSpan(level - 1);
    for (uint32_t i = 0; i < Fanout; ++i) {
        if (p->Children[i] != t->Children[i]) CommitRec(trie, p->Children[i], t->Children[i], level - 1, base + i * span, uniq, plan, next);
    }
    return t;
}

void MaterializeRec(Node *n, uint32_t level, uint64_t base, uint64_t limit, std::vector<MaterializedRun> &out) {
    if (base >= limit) return;
    switch (n->Kind) {
        case NodeKind::Aliased: out.push_back({base, std::min(limit, base + SlotSpan(level)) - base, nullptr}); return;
        case NodeKind::Owned: out.push_back({base, 1, &n->Value}); return;
        case NodeKind::Absent: return;
        case NodeKind::Interior: {
            const auto span = SlotSpan(level - 1);
            for (uint32_t i = 0; i < Fanout; ++i) MaterializeRec(n->Children[i], level - 1, base + i * span, limit, out);
            return;
        }
    }
}

bool CheckRec(Node *n, uint32_t level, std::string &why) {
    if (n->Refs == 0) {
        why = "node with zero refs reachable";
        return false;
    }
    switch (n->Kind) {
        case NodeKind::Aliased: return true;
        case NodeKind::Owned:
        case NodeKind::Absent:
            why = "present reaches a history leaf";
            return false;
        case NodeKind::Interior:
            if (level == 0) {
                why = "interior at level 0";
                return false;
            }
            for (uint32_t i = 0; i < Fanout; ++i)
                if (!CheckRec(n->Children[i], level - 1, why)) return false;
            return true;
    }
    return false;
}

void CollectAliased(Node *n, std::unordered_set<const Node *> &out) {
    if (n->Kind == NodeKind::Aliased) out.insert(n);
    else if (n->Kind == NodeKind::Interior)
        for (uint32_t i = 0; i < Fanout; ++i) CollectAliased(n->Children[i], out);
}

bool CheckVersionRec(Node *n, const std::unordered_set<const Node *> &present_aliased, std::string &why) {
    if (n->Kind == NodeKind::Aliased) {
        if (!present_aliased.contains(n)) {
            why = "pinned version holds an aliased node the present does not";
            return false;
        }
        return true;
    }
    if (n->Kind == NodeKind::Interior)
        for (uint32_t i = 0; i < Fanout; ++i)
            if (!CheckVersionRec(n->Children[i], present_aliased, why)) return false;
    return true;
}
} // namespace

uint64_t SharedNodeBytes() { return NodeBytes.load(std::memory_order_relaxed); }

LiveTrie::LiveTrie(uint32_t levels, uint32_t page_bytes)
    : Levels(levels), PageBytes(page_bytes), Root(Alloc(*this, NodeKind::Aliased)), Manifest(levels) {}

LiveTrie::~LiveTrie() { ReleaseNode(*this, Root); }

void LiveTrie::Write(uint64_t first, uint64_t count, std::span<const std::byte> contents) {
    CaptureImpl(*this, first, count, [&](uint64_t slot) -> std::optional<Blob> {
        const auto end = (slot + 1) * PageBytes;
        if (end > contents.size()) return std::nullopt;
        return CopyBlob(contents.subspan(slot * PageBytes, PageBytes));
    });
    // Mark hashes dirty after capturing values with their pre-write hashes.
    MarkDirty(first, count);
}

bool LiveTrie::Uncaptured(uint64_t slot) const {
    assert(slot < SlotSpan(Levels));
    bool uniq = true;
    const Node *n = Root;
    for (uint32_t level = Levels;; --level) {
        uniq = uniq && n->Refs == 1;
        if (n->Kind == NodeKind::Aliased) return !uniq;
        assert(n->Kind == NodeKind::Interior && level > 0);
        n = n->Children[(slot / SlotSpan(level - 1)) % Fanout];
    }
}

void LiveTrie::Capture(uint64_t slot, std::optional<Blob> value) {
    CaptureImpl(*this, slot, 1, [&](uint64_t) { return std::exchange(value, std::nullopt); });
    if (value) FreeBlob(*value);
}

void LiveTrie::MarkDirty(uint64_t first, uint64_t count) {
    for (uint64_t s = first, last = first + count; s < last; ++s) MarkDirtySlot(*this, s);
}

void LiveTrie::Rehash(uint64_t slot, std::optional<std::span<const std::byte>> bytes) {
    auto &e = SlotHashAt(*this, slot);
    e.Dirty = false;
    if (!bytes || (PageBytes && IsZero(*bytes))) RehashSlot(*this, slot, {}, true);
    else RehashSlot(*this, slot, HashBytes(*bytes), false);
}

void LiveTrie::SettleFlat(std::span<const std::byte> contents) {
    for (const auto slot : DirtySlots) {
        const auto end = (slot + 1) * PageBytes;
        if (end > contents.size()) Rehash(slot, std::nullopt);
        else Rehash(slot, contents.subspan(slot * PageBytes, PageBytes));
    }
    DirtySlots.clear();
}

Stamp LiveTrie::CurrentStamp(uint64_t length) const {
    assert(DirtySlots.empty() && "stamp requires settled hashes");
    return {{Lane0, Lane1}, length};
}

Version LiveTrie::Pin(uint64_t length) {
    const auto s = CurrentStamp(length);
    ++Root->Refs;
    return {Root, s};
}

void LiveTrie::Release(Version &v) {
    if (v.Root) ReleaseNode(*this, v.Root);
    v = {};
}

RestorePlan LiveTrie::PlanRestore(const Version &v) {
    assert(!Busy && "restore reentry into a trie mid-restore or mid-load");
    assert(DirtySlots.empty() && "restore requires settled hashes");
    Busy = true;
    RestorePlan plan{.Length = v.S.Length, .Target = v.Root, .Hash = v.S.H};
    PlanRec(*this, Root, v.Root, Levels, 0, plan);
    return plan;
}

bool LiveTrie::CommitRestore(RestorePlan &&plan) {
    assert(Busy);
    size_t next = 0;
    auto *root = CommitRec(*this, Root, plan.Target, Levels, 0, true, plan, next);
    assert(next == plan.Changes.size() && "plan applied incompletely");
    ++root->Refs;
    ReleaseNode(*this, Root);
    Root = root;
    Busy = false;
    return Hash128{Lane0, Lane1} == plan.Hash;
}

RestorePlan LiveTrie::PlanLoad(uint64_t length, std::span<const std::pair<uint64_t, Hash128>> changes, const std::unordered_map<Hash128, std::vector<std::byte>, Hash128Hasher> &leaves) {
    assert(!Busy && "load reentry into a trie mid-restore or mid-load");
    assert(DirtySlots.empty() && "load requires settled hashes");
    Busy = true;
    RestorePlan plan{.Length = length};
    plan.Changes.reserve(changes.size());
    for (const auto &[slot, hash] : changes) {
        const bool erase = hash == Hash128{};
        plan.Changes.push_back({.Slot = slot, .Incoming = erase ? Blob{} : CopyBlob(leaves.at(hash)), .Erase = erase});
    }
    return plan;
}

void LiveTrie::CommitLoad(RestorePlan &&plan) {
    assert(Busy);
    Busy = false;
    for (auto &c : plan.Changes) {
        if (Uncaptured(c.Slot)) Capture(c.Slot, c.WasPresent ? std::optional{c.Old} : std::nullopt);
        else if (c.WasPresent) FreeBlob(c.Old);
        if (CollectChanged) ChangedSlots.push_back(c.Slot);
    }
    for (const auto &c : plan.Changes) MarkDirtySlot(*this, c.Slot);
}

std::vector<MaterializedRun> LiveTrie::Materialize(const Version &v) const {
    std::vector<MaterializedRun> out;
    MaterializeRec(v.Root, Levels, 0, SlotsFor(v.S.Length), out);
    return out;
}

ManifestChildren LiveTrie::ChildrenAt(uint32_t level, uint64_t index, uint64_t length) const {
    ManifestChildren children{};
    const auto limit = SlotsFor(length);
    for (uint64_t digit = 0; digit < Fanout; ++digit) {
        const auto child = index * Fanout + digit;
        if (level == 0) {
            if (child < limit && child < SlotHashes.size() && SlotHashes[child].State == SlotState::Value)
                children[digit] = SlotHashes[child].H;
        } else if (child < Manifest[level - 1].Nodes.size()) {
            children[digit] = Manifest[level - 1].Nodes[child].Hash;
        }
    }
    return children;
}

Hash128 LiveTrie::ManifestRoot(uint64_t length) {
    assert(DirtySlots.empty() && "manifest requires settled hashes");
    const auto slots = SlotsFor(length);
    // Recompute boundary manifests when the live length changes.
    if (slots != ManifestSlots) {
        const auto end = std::min<uint64_t>(std::max(slots, ManifestSlots), SlotHashes.size());
        for (auto first = std::min(slots, ManifestSlots) / Fanout; first * Fanout < end; ++first) DirtyManifest(*this, 0, first);
        ManifestSlots = slots;
    }
    for (uint32_t level = 0; level < Levels; ++level) {
        auto &m = Manifest[level];
        for (const auto index : m.Dirty) {
            auto &node = m.Nodes[index];
            node.Dirty = false;
            const auto hash = ManifestRecord{level, ChildrenAt(level, index, length)}.Hash();
            if (node.Hash == hash) continue;
            node.Hash = hash;
            if (level + 1 < Levels) DirtyManifest(*this, level + 1, index / Fanout);
        }
        m.Dirty.clear();
    }
    return Manifest.back().Nodes.empty() ? Hash128{} : Manifest.back().Nodes[0].Hash;
}

bool LiveTrie::CheckHashes(std::string &why, std::span<const std::pair<uint64_t, Hash128>> live) const {
    if (!DirtySlots.empty()) {
        why = "unsettled dirty slots";
        return false;
    }
    uint64_t l0 = 0, l1 = 0;
    for (const auto &[s, h] : live) {
        const Term t{s, h};
        l0 += t.L0;
        l1 += t.L1;
        if (s >= SlotHashes.size() || SlotHashes[s].State != SlotState::Value || !(SlotHashes[s].H == h)) {
            why = std::format("slot {} does not match its settled hash (written without capture?)", s);
            return false;
        }
    }
    if (l0 != Lane0 || l1 != Lane1) {
        why = "state hash lanes disagree with live content";
        return false;
    }
    return true;
}

bool LiveTrie::Check(std::string &why, const std::vector<Version> &all_versions) const {
    if (!CheckRec(Root, Levels, why)) return false;
    std::unordered_set<const Node *> aliased;
    CollectAliased(Root, aliased);
    for (const auto &v : all_versions)
        if (v.Root && !CheckVersionRec(v.Root, aliased, why)) return false;
    return true;
}
} // namespace store
