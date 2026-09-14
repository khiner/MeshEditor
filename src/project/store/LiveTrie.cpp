#include "project/store/LiveTrie.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <format>
#include <memory_resource>

namespace store {
namespace {
// Account for the process-wide pool, including its cached blocks.
struct NodeMemory : std::pmr::memory_resource {
    std::atomic<uint64_t> Bytes{};
    void *do_allocate(size_t bytes, size_t alignment) override {
        auto *p = std::pmr::new_delete_resource()->allocate(bytes, alignment);
        Bytes.fetch_add(bytes, std::memory_order_relaxed);
        return p;
    }
    void do_deallocate(void *p, size_t bytes, size_t alignment) override {
        Bytes.fetch_sub(bytes, std::memory_order_relaxed);
        std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
    }
    bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override { return this == &other; }
};
struct NodeAllocator {
    NodeMemory Memory;
    std::pmr::synchronized_pool_resource Pool{{64, Fanout * sizeof(Node *)}, &Memory};
};
NodeAllocator &Allocator() {
    static NodeAllocator allocator;
    return allocator;
}
std::pmr::synchronized_pool_resource &Nodes() { return Allocator().Pool; }
Node **AllocChildren() {
    return ::new (Nodes().allocate(Fanout * sizeof(Node *), alignof(Node *))) Node *[Fanout] {};
}

Node *Alloc(LiveTrie &trie, NodeKind kind) {
    auto *n = static_cast<Node *>(Nodes().allocate(sizeof(Node), alignof(Node)));
    std::construct_at(n, Node{1, kind, {}, nullptr, {}});
    ++trie.S.Nodes;
    if (kind == NodeKind::Aliased) ++trie.S.AliasedNodes;
    if (kind == NodeKind::Interior) {
        n->Children = AllocChildren();
    }
    return n;
}

void FreeChildren(Node **a) {
    Nodes().deallocate(a, Fanout * sizeof(Node *), alignof(Node *));
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
    Nodes().deallocate(n, sizeof(Node), alignof(Node));
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

// Use the cached hash until the slot is marked dirty, then hash the captured value.
Hash128 ContentHash(const LiveTrie &trie, uint64_t slot, const Blob &value) {
    if (slot < trie.SlotHashes.size()) {
        const auto &e = trie.SlotHashes[slot];
        if (e.State == LiveTrie::SlotState::Value && !e.Dirty) return e.H;
    }
    return HashBytes(store::Encoded(trie.L, value));
}

void CaptureInto(LiveTrie &trie, Node *n, uint64_t slot) {
    assert(n->Kind == NodeKind::Aliased);
    --trie.S.AliasedNodes;
    if (trie.L.Present(slot)) {
        n->Kind = NodeKind::Owned;
        n->Value = store::Capture(trie.L, slot);
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

void MarkDirty(LiveTrie &trie, uint64_t slot) {
    auto &e = SlotHashAt(trie, slot);
    if (!e.Dirty) {
        e.Dirty = true;
        trie.DirtySlots.push_back(slot);
    }
}

// Return the writable node.
// The caller releases its reference to n when the result differs.
Node *WriteRec(LiveTrie &trie, Node *n, uint32_t level, uint64_t base, uint64_t first, uint64_t last, bool owned) {
    const bool uniq = owned && n->Refs == 1;
    if (uniq && n->Kind == NodeKind::Aliased) return n;

    if (level == 0) {
        assert(n->Kind == NodeKind::Aliased && "the present reaches only aliased leaves");
        CaptureInto(trie, n, base);
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
        auto *replacement = WriteRec(trie, child, level - 1, base + i * span, first, last, uniq);
        if (replacement != child) {
            ReleaseNode(trie, p->Children[i]);
            p->Children[i] = replacement;
        }
    }
    return p;
}

void WriteImpl(LiveTrie &trie, uint64_t first, uint64_t count) {
    if (count == 0) return;
    assert(first + count <= SlotSpan(trie.Levels));
    auto *root = WriteRec(trie, trie.Root, trie.Levels, 0, first, first + count - 1, true);
    if (root != trie.Root) {
        ReleaseNode(trie, trie.Root);
        trie.Root = root;
    }
    // Mark hashes dirty after capturing values with their pre-write hashes.
    for (uint64_t s = first, last = first + count; s < last; ++s) MarkDirty(trie, s);
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

// Swap differing values and return the target node for the present version.
Node *RestoreRec(LiveTrie &trie, Node *p, Node *t, uint32_t level, uint64_t base, bool owned) {
    if (p == t) return t;
    const bool uniq = owned && p->Refs == 1;

    if (level == 0) {
        assert(p->Kind == NodeKind::Aliased);
        if (t->Kind == NodeKind::Aliased) return t; // Both alias live data.
        const auto slot = base;
        NodeKind old_kind;
        Blob old_value{};
        Hash128 old_hash{};
        bool changed = true;
        if (t->Kind == NodeKind::Owned) {
            --trie.S.OwnedSlots;
            trie.S.OwnedBytes -= t->Value.OwnedBytes();
            // Matching hashes still require byte and entity-generation comparisons.
            const auto *e = slot < trie.SlotHashes.size() ? &trie.SlotHashes[slot] : nullptr;
            const bool maybe_equal = !(e && e->State == LiveTrie::SlotState::Value && !e->Dirty && !(e->H == t->Hash));
            if (maybe_equal && store::Equals(trie.L, slot, t->Value)) {
                old_value = t->Value;
                old_hash = t->Hash;
                old_kind = NodeKind::Owned;
                changed = false;
            } else {
                const bool incoming_default = store::DefaultValue(trie.L, t->Value);
                bool was_present = false;
                old_value = trie.L.Replace(slot, t->Value, was_present);
                old_kind = was_present ? NodeKind::Owned : NodeKind::Absent;
                if (was_present) old_hash = ContentHash(trie, slot, old_value);
                RehashSlot(trie, slot, t->Hash, incoming_default);
            }
            t->Value = {};
        } else {
            if (trie.L.Present(slot)) {
                old_value = trie.L.Erase(slot);
                old_kind = NodeKind::Owned;
                old_hash = ContentHash(trie, slot, old_value);
                RehashSlot(trie, slot, {}, true);
            } else {
                old_kind = NodeKind::Absent;
                changed = false;
            }
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
        if (changed && trie.CollectChanged) trie.ChangedSlots.push_back(slot);
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
        if (p->Children[i] != t->Children[i]) RestoreRec(trie, p->Children[i], t->Children[i], level - 1, base + i * span, uniq);
    }
    return t;
}

bool DiffersRec(const LiveTrie &trie, Node *p, Node *t, uint32_t level, uint64_t base) {
    if (p == t) return false;
    if (level == 0) {
        switch (t->Kind) {
            case NodeKind::Aliased: return false;
            case NodeKind::Absent: return trie.L.Present(base);
            case NodeKind::Owned: return !store::Equals(trie.L, base, t->Value);
            case NodeKind::Interior: return true;
        }
    }
    if (t->Kind == NodeKind::Aliased) return false;
    const auto span = SlotSpan(level - 1);
    for (uint32_t i = 0; i < Fanout; ++i) {
        auto *child = p->Kind == NodeKind::Aliased ? p : p->Children[i];
        if (child != t->Children[i] && DiffersRec(trie, child, t->Children[i], level - 1, base + i * span)) return true;
    }
    return false;
}

void MaterializeRec(const LiveTrie &trie, Node *n, uint32_t level, uint64_t base, uint64_t limit, const std::function<void(uint64_t, std::span<const std::byte>)> &visit) {
    if (base >= limit) return;
    switch (n->Kind) {
        case NodeKind::Aliased:
            for (uint64_t s = base, end = std::min(limit, base + SlotSpan(level)); s < end; ++s)
                if (trie.L.Present(s)) visit(s, trie.L.Read(s));
            return;
        case NodeKind::Owned: visit(base, store::Encoded(trie.L, n->Value)); return;
        case NodeKind::Absent: return;
        case NodeKind::Interior: {
            const auto span = SlotSpan(level - 1);
            for (uint32_t i = 0; i < Fanout; ++i) MaterializeRec(trie, n->Children[i], level - 1, base + i * span, limit, visit);
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

uint64_t SharedNodePoolBytes() { return Allocator().Memory.Bytes.load(std::memory_order_relaxed); }

LiveTrie::LiveTrie(Live live, uint32_t levels, uint32_t slot_bytes)
    : L(std::move(live)), Levels(levels), BytesPerSlot(slot_bytes), Root(Alloc(*this, NodeKind::Aliased)), Manifest(levels) {}

LiveTrie::~LiveTrie() { ReleaseNode(*this, Root); }

void LiveTrie::SettleHashes() {
    for (const auto slot : DirtySlots) {
        auto &e = SlotHashes[slot];
        if (!e.Dirty) continue; // Skip duplicate dirty indices.
        e.Dirty = false;
        if (store::DefaultAt(L, slot)) RehashSlot(*this, slot, {}, true);
        else RehashSlot(*this, slot, HashBytes(L.Read(slot)), false);
    }
    DirtySlots.clear();
}

Stamp LiveTrie::CurrentStamp() {
    SettleHashes();
    return {{Lane0, Lane1}, L.Length()};
}

void LiveTrie::Write(uint64_t first, uint64_t count) {
    assert(!Busy && "write reentry into a trie mid-restore or mid-load");
    assert(!ExternalWritesForbidden && "LiveTrie::Write forbidden during track restoration and AfterTracks");
    WriteImpl(*this, first, count);
}

Version LiveTrie::Pin() {
    const auto s = CurrentStamp();
    ++Root->Refs;
    return {Root, s};
}

void LiveTrie::Release(Version &v) {
    if (v.Root) ReleaseNode(*this, v.Root);
    v = {};
}

bool LiveTrie::Restore(const Version &v) {
    assert(!Busy && "restore reentry into a trie mid-restore or mid-load");
    Busy = true;
    SettleHashes(); // Update hashes before capturing outgoing values.
    auto *root = RestoreRec(*this, Root, v.Root, Levels, 0, true);
    ++root->Refs;
    ReleaseNode(*this, Root);
    Root = root;
    if (L.SetLength) L.SetLength(v.S.Length);
    Busy = false;
    return Hash128{Lane0, Lane1} == v.S.H;
}

bool LiveTrie::Differs(const Version &v) const {
    if (L.Length() != v.S.Length) return true;
    return DiffersRec(*this, Root, v.Root, Levels, 0);
}

void LiveTrie::Materialize(const Version &v, const std::function<void(uint64_t, std::span<const std::byte>)> &visit) const {
    MaterializeRec(*this, v.Root, Levels, 0, store::SlotsFor(L, v.S.Length), visit);
}

void LiveTrie::LoadChanges(uint64_t length, std::span<const std::pair<uint64_t, Hash128>> changes, const std::unordered_map<Hash128, std::vector<std::byte>, Hash128Hasher> &leaves) {
    assert(!Busy && "load reentry into a trie mid-restore or mid-load");
    Busy = true;
    SettleHashes();
    const auto apply = [&](uint64_t slot, Blob incoming, bool erase = false) {
        WriteImpl(*this, slot, 1);
        bool was_present;
        auto old = erase ? L.Erase(slot) : L.Replace(slot, incoming, was_present);
        FreeBlob(old);
        if (CollectChanged) ChangedSlots.push_back(slot);
    };
    for (const auto &[slot, hash] : changes) {
        const bool erase = hash == Hash128{};
        apply(slot, erase ? Blob{} : CopyBlob(leaves.at(hash)), erase);
    }
    if (L.ForEachNonReusable) L.ForEachNonReusable([&](uint64_t slot) {
        // Changed slots already have replacements queued.
        if (!SlotHashes[slot].Dirty) apply(slot, store::Capture(L, slot));
    });
    if (L.SetLength) L.SetLength(length);
    Busy = false;
}

ManifestChildren LiveTrie::ChildrenAt(uint32_t level, uint64_t index) const {
    ManifestChildren children{};
    const auto limit = store::SlotsFor(L, L.Length());
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

Hash128 LiveTrie::ManifestRoot() {
    SettleHashes();
    const auto slots = store::SlotsFor(L, L.Length());
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
            const auto hash = ManifestRecord{level, ChildrenAt(level, index)}.Hash();
            if (node.Hash == hash) continue;
            node.Hash = hash;
            if (level + 1 < Levels) DirtyManifest(*this, level + 1, index / Fanout);
        }
        m.Dirty.clear();
    }
    return Manifest.back().Nodes.empty() ? Hash128{} : Manifest.back().Nodes[0].Hash;
}

uint64_t LiveTrie::ManifestBytes() const {
    uint64_t bytes = Manifest.capacity() * sizeof(ManifestLevel);
    for (const auto &m : Manifest) bytes += m.Nodes.capacity() * sizeof(ManifestNode) + m.Dirty.capacity() * sizeof(uint64_t);
    return bytes;
}

bool LiveTrie::CheckHashes(std::string &why) const {
    if (!DirtySlots.empty()) {
        why = "unsettled dirty slots";
        return false;
    }
    uint64_t l0 = 0, l1 = 0;
    uint64_t bad = ~0ull;
    store::ForEachPresent(L, [&](uint64_t s) {
        if (store::DefaultAt(L, s)) return;
        const auto h = HashBytes(L.Read(s));
        const Term t{s, h};
        l0 += t.L0;
        l1 += t.L1;
        if (bad == ~0ull && (s >= SlotHashes.size() || SlotHashes[s].State != SlotState::Value || !(SlotHashes[s].H == h))) bad = s;
    });
    if (bad != ~0ull) {
        why = std::format("slot {} does not match its settled hash (written without Write?)", bad);
        return false;
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
