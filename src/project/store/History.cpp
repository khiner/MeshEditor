#include "project/store/History.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace store {
namespace {
constexpr size_t FileTree = 2;
constexpr const char *TreeLogName = "tree.log";
// Content record format: [hash][u32 size][payload].
constexpr uint64_t ContentHeader = sizeof(Hash128) + sizeof(uint32_t);

template<typename T> void Put(std::vector<std::byte> &out, const T &v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto *p = reinterpret_cast<const std::byte *>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}
bool ReadFile(const std::filesystem::path &path, std::vector<std::byte> &out) {
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    if (!in) return false;
    const auto size = in.tellg();
    if (size < 0) return false;
    out.resize(size_t(size));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(out.data()), size);
    return bool(in) || size == 0;
}
template<typename T> bool Take(std::span<const std::byte> &in, T &v) {
    if (in.size() < sizeof(T)) return false;
    std::memcpy(&v, in.data(), sizeof(T));
    in = in.subspan(sizeof(T));
    return true;
}

void PutState(std::vector<std::byte> &out, const std::vector<Stamp> &stamps, const std::vector<Hash128> &roots) {
    for (size_t i = 0; i < stamps.size(); ++i) {
        Put(out, stamps[i].H);
        Put(out, stamps[i].Length);
        Put(out, roots[i]);
    }
}

bool TakeState(std::span<const std::byte> &in, HistoryNode &node, size_t tracks) {
    node.Stamps.resize(tracks);
    node.Roots.resize(tracks);
    for (size_t i = 0; i < tracks; ++i)
        if (!Take(in, node.Stamps[i].H) || !Take(in, node.Stamps[i].Length) || !Take(in, node.Roots[i])) return false;
    return true;
}

std::vector<std::byte> NodePayload(const HistoryNode &node) {
    std::vector<std::byte> payload;
    Put(payload, uint32_t(node.Action.size()));
    payload.insert(payload.end(), node.Action.begin(), node.Action.end());
    Put(payload, uint32_t(node.Label.size()));
    const auto *label = reinterpret_cast<const std::byte *>(node.Label.data());
    payload.insert(payload.end(), label, label + node.Label.size());
    PutState(payload, node.Stamps, node.Roots);
    return payload;
}
struct LoadPlan {
    std::vector<std::vector<std::pair<uint64_t, Hash128>>> Changes;
    std::unordered_map<Hash128, std::vector<std::byte>, Hash128Hasher> Leaves;
};

// Disable public writes through AfterTracks, then enable them for AfterRestore.
struct PipelineScope {
    History &Hist;
    explicit PipelineScope(History &h) : Hist(h) {
        for (auto &t : Hist.Tracks) t.Trie->ExternalWritesForbidden = true;
    }
    ~PipelineScope() {
        for (auto &t : Hist.Tracks) t.Trie->ExternalWritesForbidden = false;
    }
};

std::string ReplayStep(History &history, const HistoryNode &node) {
    history.Callbacks.Replay(node.Action);
    for (size_t i = 0; i < history.Tracks.size(); ++i) {
        if (history.Tracks[i].Trie->CurrentStamp() != node.Stamps[i]) return history.Tracks[i].Name;
    }
    return {};
}

bool Matches(const History &history, const Snapshot &s) {
    for (size_t i = 0; i < history.Tracks.size(); ++i)
        if (history.Tracks[i].Trie->Differs(s.Versions[i])) return false;
    return true;
}

std::vector<std::byte> MaterializeSnapshot(const History &history, const Snapshot &s) {
    // Per-track format: [u64 length][u64 count]([u64 slot][u32 size][bytes])*.
    std::vector<std::byte> out;
    for (const auto i : history.Order) {
        const auto &v = s.Versions[i];
        Put(out, v.S.Length);
        const auto count_pos = out.size();
        Put(out, uint64_t{0});
        uint64_t count = 0;
        history.Tracks[i].Trie->Materialize(v, [&](uint64_t slot, std::span<const std::byte> bytes) {
            Put(out, slot);
            Put(out, uint32_t(bytes.size()));
            out.insert(out.end(), bytes.begin(), bytes.end());
            ++count;
        });
        std::memcpy(out.data() + count_pos, &count, sizeof(count));
    }
    return out;
}

uint64_t OwnedBytes(const History &history) {
    uint64_t total = 0;
    for (const auto &t : history.Tracks) total += t.Trie->Stats().OwnedBytes;
    return total;
}

std::vector<Stamp> CurrentStamps(History &history) {
    std::vector<Stamp> out;
    out.reserve(history.Tracks.size());
    for (auto &t : history.Tracks) out.push_back(t.Trie->CurrentStamp());
    return out;
}

std::vector<Version> AllVersions(const History &history, size_t track) {
    std::vector<Version> out;
    for (const auto &n : history.Nodes)
        if (n.Hot) out.push_back(n.Hot->Versions[track]);
    return out;
}

bool CheckHashes(History &history, std::string &why) {
    history.SettleHashes();
    for (const auto &t : history.Tracks) {
        if (!t.Trie->CheckHashes(why)) {
            why = t.Name + ": " + why;
            return false;
        }
    }
    return true;
}

void RunPipeline(History &history, auto &&per_track) {
    if (history.Callbacks.BeforeRestore) history.Callbacks.BeforeRestore();
    {
        PipelineScope scope{history};
        for (const auto i : history.Order) per_track(i);
        if (history.Callbacks.AfterTracks) history.Callbacks.AfterTracks();
    }
    if (history.Callbacks.AfterRestore) history.Callbacks.AfterRestore();
}

void AddIntegrityError(History &history, std::string what) {
    if (!history.IntegrityError.empty()) history.IntegrityError += ", ";
    history.IntegrityError += std::move(what);
}

void Adopt(History &history, const Snapshot &s) {
    for (const auto i : history.Order) {
        if (!history.Tracks[i].Trie->Restore(s.Versions[i])) {
            AddIntegrityError(history, history.Tracks[i].Name + ": equal-state adoption failed");
            assert(false && "equal-state adoption failed");
        }
    }
}

void ReleaseNode(History &history, int node) {
    auto &n = history.Nodes[node];
    if (!n.Hot) return;
    history.Release(*n.Hot);
    n.Hot.reset();
}

void SetPresent(History &history, int node) {
    history.Present = node;
    history.Nodes[node].LastVisited = ++history.VisitCounter;
}

int TreeDistance(const History &history, int a, int b) {
    int x = a, y = b;
    while (history.Nodes[x].Depth > history.Nodes[y].Depth) x = history.Nodes[x].Parent;
    while (history.Nodes[y].Depth > history.Nodes[x].Depth) y = history.Nodes[y].Parent;
    while (x != y) {
        x = history.Nodes[x].Parent;
        y = history.Nodes[y].Parent;
    }
    return history.Nodes[a].Depth + history.Nodes[b].Depth - 2 * history.Nodes[x].Depth;
}

bool CheckIO(History &history, std::string error) {
    if (error.empty()) return true;
    if (history.IntegrityError.empty()) history.IntegrityError = std::move(error);
    return false;
}

std::vector<std::byte> Descriptor(const History &history) {
    std::vector<std::byte> out;
    Put(out, uint64_t{0x45524f5453545350}); // PSTSTORE
    Put(out, uint32_t{4}); // store format revision
    Put(out, history.SchemaRevision);
    Put(out, uint32_t(history.Tracks.size()));
    for (const auto &t : history.Tracks) {
        Put(out, uint32_t(t.Name.size()));
        for (char c : t.Name) Put(out, c);
        Put(out, int32_t(t.Phase));
        Put(out, t.Trie->Levels);
        Put(out, uint64_t(t.Trie->BytesPerSlot));
    }
    return out;
}

bool ReadExtent(std::ifstream &in, History::Extent e, Hash128 hash, std::vector<std::byte> &out) {
    out.resize(e.Size);
    in.seekg(std::streamoff(e.Offset));
    in.read(reinterpret_cast<char *>(out.data()), std::streamsize(out.size()));
    return bool(in) && HashBytes(out) == hash;
}

bool ReadManifestRecord(const History &history, std::ifstream &in, Hash128 ref, uint8_t &level, ManifestChildren &children) {
    const auto it = history.NodeLog.Idx.find(ref);
    if (it == history.NodeLog.Idx.end()) return false;
    std::vector<std::byte> payload;
    if (!ReadExtent(in, it->second, ref, payload)) return false;
    std::span<const std::byte> p{payload};
    uint16_t count;
    if (!Take(p, level) || !Take(p, count) || count > Fanout) return false;
    children.fill({});
    uint8_t previous{};
    for (uint16_t k = 0; k < count; ++k) {
        uint8_t digit;
        Hash128 child;
        if (!Take(p, digit) || !Take(p, child) || digit >= Fanout || (k && digit <= previous) || child == Hash128{}) return false;
        children[digit] = child;
        previous = digit;
    }
    return p.empty();
}

// Index records through the first framing or hash error.
bool ScanContentLog(History &history, History::ContentLog &log) {
    log.Size = 0;
    std::ifstream in{history.Dir / log.Name, std::ios::binary | std::ios::ate};
    if (!in) return false;
    const auto end = in.tellg();
    if (end < 0) return false;
    in.seekg(0);
    std::vector<std::byte> bytes;
    while (uint64_t(end) - log.Size >= ContentHeader) {
        Hash128 hash;
        uint32_t psize;
        if (!in.read(reinterpret_cast<char *>(&hash), sizeof(hash)) || !in.read(reinterpret_cast<char *>(&psize), sizeof(psize))) break;
        if (psize > uint64_t(end) - log.Size - ContentHeader) break;
        bytes.resize(psize);
        if (!in.read(reinterpret_cast<char *>(bytes.data()), psize) || HashBytes(bytes) != hash) break;
        log.Idx.emplace(hash, History::Extent{log.Size + ContentHeader, psize});
        log.Size += ContentHeader + psize;
    }
    return true;
}

bool ReadProject(History &history, const std::filesystem::path &dir, uint64_t &tree_size) {
    std::vector<std::byte> bytes;
    if (!ReadFile(dir / TreeLogName, bytes)) return false;
    const auto descriptor = Descriptor(history);
    if (bytes.size() < descriptor.size() || !std::equal(descriptor.begin(), descriptor.end(), bytes.begin())) return false;
    history.Dir = dir;
    if (!ScanContentLog(history, history.LeafLog) || !ScanContentLog(history, history.NodeLog)) return false;
    // Cache subtree validity to check each reachable record once.
    std::unordered_map<Hash128, bool, Hash128Hasher> resolvable_memo;
    std::ifstream nodes_in{dir / history.NodeLog.Name, std::ios::binary};
    const auto resolvable = [&](this auto &&self, Hash128 ref) -> bool {
        if (ref == Hash128{}) return true;
        if (const auto it = resolvable_memo.find(ref); it != resolvable_memo.end()) return it->second;
        auto &memo = resolvable_memo[ref] = false;
        uint8_t lv;
        ManifestChildren children;
        if (!ReadManifestRecord(history, nodes_in, ref, lv, children)) return false;
        for (const auto child : children)
            if (child != Hash128{} && (lv > 0 ? !self(child) : !history.LeafLog.Idx.contains(child))) return false;
        return memo = true;
    };
    // Node IDs follow record order, so discard all records after the first invalid record.
    tree_size = descriptor.size();
    auto rest = std::span<const std::byte>{bytes}.subspan(tree_size);
    while (!rest.empty()) {
        uint32_t len;
        RecordKind kind;
        int32_t parent;
        if (!Take(rest, len) || rest.size() < len || len < 5 || !Take(rest, kind) || !Take(rest, parent)) break;
        auto payload = rest.subspan(0, len - 5);
        rest = rest.subspan(len - 5);
        if (kind == RecordKind::Root || kind == RecordKind::Action) {
            uint32_t asz;
            if (!Take(payload, asz) || payload.size() < asz) break;
            HistoryNode n;
            n.Action.assign(payload.begin(), payload.begin() + asz);
            payload = payload.subspan(asz);
            uint32_t label_size;
            if (!Take(payload, label_size) || payload.size() < label_size) break;
            n.Label.assign(reinterpret_cast<const char *>(payload.data()), label_size);
            payload = payload.subspan(label_size);
            if (!TakeState(payload, n, history.Tracks.size())) break;
            if (kind == RecordKind::Action && (parent < 0 || parent >= int(history.Nodes.size()))) break;
            if (!std::ranges::all_of(n.Roots, resolvable)) break;
            std::optional<HistoryNode> saved;
            if (kind == RecordKind::Root && !payload.empty()) {
                saved.emplace();
                saved->Label = "Saved";
                saved->Children.push_back(1);
                if (!TakeState(payload, *saved, history.Tracks.size()) || !std::ranges::all_of(saved->Roots, resolvable)) break;
            }
            if (!payload.empty()) break;
            if (kind == RecordKind::Root) {
                history.Nodes.clear();
                if (saved) {
                    history.Nodes.push_back(std::move(*saved));
                    n.Parent = 0;
                    n.Depth = 1;
                }
            } else {
                n.Parent = parent;
                n.Depth = history.Nodes[parent].Depth + 1;
                n.ReplayBaseline = false;
                history.Nodes[parent].Children.push_back(int(history.Nodes.size()));
            }
            history.Nodes.push_back(std::move(n));
            history.Present = int(history.Nodes.size()) - 1;
        } else if (kind == RecordKind::Navigate && parent >= 0 && parent < int(history.Nodes.size())) {
            history.Present = parent;
        }
        tree_size += 4 + len;
    }
    // Exclude manifests with missing descendants from deduplication after a partial write.
    std::erase_if(history.NodeLog.Idx, [&](const auto &entry) { return !resolvable(entry.first); });
    return !history.Nodes.empty();
}

bool AdoptProject(History &history, History &candidate) {
    if (!CheckIO(history, candidate.Log.Stop())) return false;
    const auto &dir = candidate.Dir;
    if (!CheckIO(history, history.Log.Start({dir / history.LeafLog.Name, dir / history.NodeLog.Name, dir / TreeLogName}, false))) return false;
    for (int i = 0; i < int(history.Nodes.size()); ++i) ReleaseNode(history, i);
    history.Nodes = std::move(candidate.Nodes);
    history.LeafLog = std::move(candidate.LeafLog);
    history.NodeLog = std::move(candidate.NodeLog);
    history.Dir = std::move(candidate.Dir);
    history.Present = candidate.Present;
    history.PeakPendingBytes = std::max(candidate.PeakPendingBytes, candidate.Log.PendingMemory().second);
    ++history.Revision;
    history.VisitCounter = 0;
    history.IntegrityError.clear();
    return true;
}

// Validate against current live data, including uncommitted edits, before mutating any track.
bool PrepareLoad(History &history, int node, LoadPlan &plan) {
    const auto &n = history.Nodes[node];
    std::ifstream leaves{history.Dir / history.LeafLog.Name, std::ios::binary}, nodes_in{history.Dir / history.NodeLog.Name, std::ios::binary};
    if (!leaves || !nodes_in) return false;
    plan.Changes.resize(history.Tracks.size());
    for (size_t i = 0; i < history.Tracks.size(); ++i) {
        auto *trie = history.Tracks[i].Trie;
        const auto limit = trie->SlotsFor(n.Stamps[i].Length);
        if (limit > SlotSpan(trie->Levels)) return false;
        const auto live_root = trie->ManifestRoot();
        const auto live_stamp = trie->CurrentStamp();
        const auto live_limit = trie->SlotsFor(live_stamp.Length);
        auto state = live_stamp.H;
        const auto diff = [&](this auto &&self, Hash128 live, Hash128 target, uint32_t level, uint64_t index) -> bool {
            // Validate target slots beyond the new length even when subtree hashes match.
            if (live == target && (target == Hash128{} || limit >= live_limit || (index + 1) * SlotSpan(level) <= limit)) return true;
            if (level) {
                const auto before = trie->ChildrenAt(level - 1, index);
                ManifestChildren after{};
                if (target != Hash128{}) {
                    uint8_t recorded_level;
                    if (!ReadManifestRecord(history, nodes_in, target, recorded_level, after) || recorded_level != level - 1) return false;
                }
                for (uint64_t digit = 0; digit < Fanout; ++digit)
                    if (!self(before[digit], after[digit], level - 1, index * Fanout + digit)) return false;
                return true;
            }
            if (target != Hash128{} && index >= limit) return false;
            if (live != Hash128{}) {
                const Term term{index, live};
                state.A -= term.L0;
                state.B -= term.L1;
            }
            if (target != Hash128{}) {
                const Term term{index, target};
                state.A += term.L0;
                state.B += term.L1;
                const auto it = history.LeafLog.Idx.find(target);
                if (it == history.LeafLog.Idx.end() || (trie->BytesPerSlot && it->second.Size != trie->BytesPerSlot)) return false;
                const auto [leaf, inserted] = plan.Leaves.try_emplace(target);
                if (inserted && !ReadExtent(leaves, it->second, target, leaf->second)) return false;
            }
            plan.Changes[i].push_back({index, target});
            return true;
        };
        if (!diff(live_root, n.Roots[i], trie->Levels, 0) || state != n.Stamps[i].H) return false;
    }
    return true;
}

void ApplyLoad(History &history, int node, const LoadPlan &plan) {
    const auto &n = history.Nodes[node];
    RunPipeline(history, [&](size_t i) {
        history.Tracks[i].Trie->LoadChanges(n.Stamps[i].Length, plan.Changes[i], plan.Leaves);
    });
    for (size_t i = 0; i < history.Tracks.size(); ++i) {
        if (history.Tracks[i].Trie->CurrentStamp() == n.Stamps[i]) continue;
        AddIntegrityError(history, history.Tracks[i].Name + ": cold load verification failed");
        std::fprintf(stderr, "[history] %s: cold load verification failed\n", history.Tracks[i].Name.c_str());
        assert(false && "cold load verification failed: loaded state does not hash to the recorded stamp");
    }
}

void EnqueuePending(History &history) {
    uint64_t staged = history.Log.PendingMemory().first;
    for (const auto &pending : history.Pending) staged += pending.capacity();
    history.PeakPendingBytes = std::max(history.PeakPendingBytes, staged);
    for (size_t file = 0; file < history.Pending.size(); ++file) {
        if (history.Pending[file].empty()) continue;
        history.Log.Append(file, std::exchange(history.Pending[file], {}));
    }
}

void AppendContent(History &history, History::ContentLog &log, Hash128 hash, std::span<const std::byte> payload) {
    if (!log.Idx.try_emplace(hash, History::Extent{log.Size + ContentHeader, uint32_t(payload.size())}).second) return;
    auto &pending = history.Pending[log.File];
    // Limit each append buffer to roughly 1 MiB, allowing larger individual records.
    if (!pending.empty() && pending.size() + ContentHeader + payload.size() > (1u << 20)) EnqueuePending(history);
    Put(pending, hash);
    Put(pending, uint32_t(payload.size()));
    pending.insert(pending.end(), payload.begin(), payload.end());
    log.Size += ContentHeader + payload.size();
}

Hash128 BuildManifest(History &history, size_t track) {
    auto *trie = history.Tracks[track].Trie;
    const auto root = trie->ManifestRoot();
    // Indexed roots have indexed descendants.
    const auto persist = [&](this auto &&self, Hash128 hash, uint32_t level, uint64_t index) -> void {
        if (hash == Hash128{} || history.NodeLog.Idx.contains(hash)) return;
        const auto children = trie->ChildrenAt(level, index);
        for (uint64_t digit = 0; digit < Fanout; ++digit) {
            const auto child = children[digit];
            if (child == Hash128{}) continue;
            if (level) self(child, level - 1, index * Fanout + digit);
            else if (!history.LeafLog.Idx.contains(child)) {
                AppendContent(history, history.LeafLog, child, trie->L.Read(index * Fanout + digit));
            }
        }
        AppendContent(history, history.NodeLog, hash, ManifestRecord{level, children}.View());
    };
    persist(root, trie->Levels - 1, 0);
    return root;
}

void AppendTreeRecord(History &history, RecordKind kind, int parent, const std::vector<std::byte> &payload) {
    assert(history.Log.Started() && "the tree log requires Begin or Open");
    // Tree record format: [u32 len][u8 kind][i32 parent][payload].
    auto &record = history.Pending[FileTree];
    Put(record, uint32_t(1 + 4 + payload.size()));
    Put(record, uint8_t(kind));
    Put(record, int32_t(parent));
    record.insert(record.end(), payload.begin(), payload.end());
    EnqueuePending(history);
}

void PersistNode(History &history, RecordKind kind, int node) {
    assert(history.Log.Started() && "commit requires Begin or Open");
    auto &n = history.Nodes[node];
    const auto *parent = n.Parent >= 0 ? &history.Nodes[n.Parent] : nullptr;
    n.Roots.clear();
    for (size_t i = 0; i < history.Tracks.size(); ++i) {
        if (parent && parent->Roots.size() == history.Tracks.size() && parent->Stamps[i] == n.Stamps[i]) n.Roots.push_back(parent->Roots[i]);
        else n.Roots.push_back(BuildManifest(history, i));
    }
    AppendTreeRecord(history, kind, n.Parent, NodePayload(n));
}

bool LoadNode(History &history, int node) {
    EnqueuePending(history);
    if (!CheckIO(history, history.Log.FlushAndWait())) return false;
    LoadPlan plan;
    if (!PrepareLoad(history, node, plan)) return CheckIO(history, "cold load contains missing or corrupt records");
    ApplyLoad(history, node, plan);
    return true;
}
} // namespace

void History::Track(LiveTrie &t, std::string name, int phase) {
    Tracks.push_back({&t, std::move(name), phase});
    // Keep serialized track indices in registration order when sorting restoration order.
    Order.insert(std::ranges::upper_bound(Order, phase, {}, [&](size_t i) { return Tracks[i].Phase; }), Tracks.size() - 1);
}

Snapshot History::Pin() {
    Snapshot s;
    for (auto &t : Tracks) s.Versions.push_back(t.Trie->Pin());
    return s;
}

void History::SettleHashes() {
    for (auto &t : Tracks) t.Trie->SettleHashes();
}

std::string History::TakeIntegrityError() {
    auto error = std::exchange(IntegrityError, {});
    return error.empty() ? Log.Error() : error;
}

void History::Restore(const Snapshot &s) {
    RunPipeline(*this, [&](size_t i) {
        if (!Tracks[i].Trie->Restore(s.Versions[i])) {
            AddIntegrityError(*this, Tracks[i].Name + ": restored state hash mismatch");
            assert(false && "restored state hash does not match pinned hash");
        }
    });
}

void History::Release(Snapshot &snapshot) {
    for (size_t i = 0; i < Tracks.size(); ++i) Tracks[i].Trie->Release(snapshot.Versions[i]);
    snapshot.Versions.clear();
}

bool History::Begin(const std::filesystem::path &dir) {
    assert(Dir.empty() || Dir != dir);
    if (!CheckIO(*this, Log.FlushAndWait())) return false;
    History candidate;
    candidate.Tracks = Tracks;
    candidate.Order = Order;
    candidate.SchemaRevision = SchemaRevision;
    candidate.Dir = dir;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return CheckIO(*this, "cannot create " + dir.string() + ": " + ec.message());
    if (!CheckIO(*this, candidate.Log.Start({dir / LeafLog.Name, dir / NodeLog.Name, dir / TreeLogName}, true))) return false;
    candidate.Pending[FileTree] = Descriptor(*this);
    HistoryNode root;
    root.Label = "Root";
    root.Hot = Pin();
    root.Stamps = CurrentStamps(*this);
    candidate.Nodes.push_back(std::move(root));
    SetPresent(candidate, 0);
    PersistNode(candidate, RecordKind::Root, 0);
    if (!AdoptProject(*this, candidate)) return false;
    SetPresent(*this, 0);
    return true;
}

bool History::Close() {
    const bool saved = CheckIO(*this, Log.Stop());
    for (int i = 0; i < int(Nodes.size()); ++i) ReleaseNode(*this, i);
    Nodes.clear();
    ++Revision;
    Present = -1;
    VisitCounter = 0;
    Dir.clear();
    LeafLog.Idx.clear();
    NodeLog.Idx.clear();
    LeafLog.Size = NodeLog.Size = 0;
    for (auto &p : Pending) p.clear();
    return saved;
}

bool History::Relocate(const std::filesystem::path &dir) {
    if (!CheckIO(*this, Log.FlushAndWait())) return false;
    if (!CheckIO(*this, Log.Start({dir / LeafLog.Name, dir / NodeLog.Name, dir / TreeLogName}, false))) return false;
    Dir = dir;
    return true;
}

int History::Commit(std::string label, std::vector<std::byte> action) {
    assert(Present >= 0);
    const auto stamps = CurrentStamps(*this);
    if (stamps == Nodes[Present].Stamps) {
        assert(Matches(*this, *Nodes[Present].Hot) && "state hash matches the present but bytes differ");
        Adopt(*this, *Nodes[Present].Hot); // Release redundant copies for equal state.
        return Present;
    }
    const auto adopt = [&](int node) {
        auto &n = Nodes[node];
        if (n.Hot) {
            assert(Matches(*this, *n.Hot) && "stamp matches but bytes differ");
            Adopt(*this, *n.Hot);
        } else {
            n.Hot = Pin();
        }
        SetPresent(*this, node);
        AppendTreeRecord(*this, RecordKind::Navigate, node, {});
        return node;
    };
    for (const int child : Nodes[Present].Children)
        if (Nodes[child].Stamps == stamps) return adopt(child);
    if (const int parent = Nodes[Present].Parent; parent >= 0 && Nodes[parent].Stamps == stamps) return adopt(parent);
    const int id = int(Nodes.size());
    HistoryNode n;
    n.Parent = Present;
    n.ReplayBaseline = false;
    n.Action = std::move(action);
    n.Label = std::move(label);
    n.Depth = Nodes[Present].Depth + 1;
    n.Hot = Pin();
    n.Stamps = stamps;
    Nodes.push_back(std::move(n));
    Nodes[Present].Children.push_back(id);
    ++Revision;
    SetPresent(*this, id);
    PersistNode(*this, RecordKind::Action, id);
    return id;
}

void History::Navigate(int node) {
    if (node == Present || node < 0 || node >= int(Nodes.size())) return;
    if (Nodes[node].Hot) {
        Restore(*Nodes[node].Hot);
    } else if (LoadNode(*this, node)) {
        Nodes[node].Hot = Pin();
    } else {
        return;
    }
    SetPresent(*this, node);
    AppendTreeRecord(*this, RecordKind::Navigate, node, {});
}

void History::Revert() {
    assert(Present >= 0 && Nodes[Present].Hot);
    Restore(*Nodes[Present].Hot);
}

bool History::Save() {
    AppendTreeRecord(*this, RecordKind::Navigate, Present, {});
    return CheckIO(*this, Log.FlushAndWait());
}

bool History::Clear(const HistoryPosition *saved) {
    if (!CheckIO(*this, Log.FlushAndWait())) return false;
    auto stamps = CurrentStamps(*this);
    std::vector<HistoryNode> retained;
    if (saved && saved->Stamps != stamps) {
        auto &root = retained.emplace_back();
        root.Label = "Saved";
        root.Children.push_back(1);
        root.Stamps = saved->Stamps;
        root.Roots = saved->Roots;
    }
    const int current = int(retained.size());
    auto &live = retained.emplace_back();
    live.Label = current ? "Current" : "Root";
    live.Parent = current - 1;
    live.Depth = current;
    live.Hot = Pin();
    live.Stamps = std::move(stamps);
    for (size_t i = 0; i < Tracks.size(); ++i) live.Roots.push_back(BuildManifest(*this, i));
    auto payload = NodePayload(live);
    if (current) PutState(payload, saved->Stamps, saved->Roots);
    AppendTreeRecord(*this, RecordKind::Root, -1, payload);
    if (!CheckIO(*this, Log.FlushAndWait())) {
        Release(*live.Hot);
        return false;
    }
    for (int i = 0; i < int(Nodes.size()); ++i) ReleaseNode(*this, i);
    Nodes = std::move(retained);
    VisitCounter = 0;
    ++Revision;
    SetPresent(*this, current);
    return true;
}

void History::Evict(uint64_t cap) {
    if (OwnedBytes(*this) <= cap) return;
    std::vector<int> protected_nodes;
    for (int n = Present, k = 0; n >= 0 && k <= UndoWindow; n = Nodes[n].Parent, ++k) {
        protected_nodes.push_back(n);
        if (Nodes[n].ReplayBaseline) break;
    }
    std::vector<int> victims;
    for (int i = 0; i < int(Nodes.size()); ++i) {
        if (!Nodes[i].Hot) continue;
        if (std::find(protected_nodes.begin(), protected_nodes.end(), i) != protected_nodes.end()) continue;
        victims.push_back(i);
    }
    if (victims.empty() || !CheckIO(*this, Log.FlushAndWait())) return;
    std::vector<int> distance(Nodes.size());
    for (const int v : victims) distance[v] = TreeDistance(*this, v, Present);
    // Evict farthest nodes first, breaking ties by least recent visit.
    std::sort(victims.begin(), victims.end(), [&](int a, int b) {
        if (distance[a] != distance[b]) return distance[a] > distance[b];
        return Nodes[a].LastVisited < Nodes[b].LastVisited;
    });
    for (const int victim : victims) {
        if (OwnedBytes(*this) <= cap) return;
        ReleaseNode(*this, victim);
    }
}

std::vector<std::byte> History::Materialize(int node) const {
    assert(node >= 0 && Nodes[node].Hot);
    return MaterializeSnapshot(*this, *Nodes[node].Hot);
}

std::vector<std::byte> History::MaterializeLive() {
    auto live = Pin();
    auto bytes = MaterializeSnapshot(*this, live);
    Release(live);
    return bytes;
}

std::string History::Replay(int node) {
    if (node < 0 || node >= int(Nodes.size())) return "bad node";
    auto original = Pin();
    std::vector<int> path;
    int baseline = node;
    while (!Nodes[baseline].ReplayBaseline) {
        path.push_back(baseline);
        baseline = Nodes[baseline].Parent;
    }
    std::string diff;
    if (Nodes[baseline].Hot) Restore(*Nodes[baseline].Hot);
    else if (!LoadNode(*this, baseline)) diff = "baseline unloadable";
    try {
        if (diff.empty()) {
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                diff = ReplayStep(*this, Nodes[*it]);
                if (!diff.empty()) break;
            }
        }
    } catch (const std::exception &error) {
        diff = error.what();
    }
    if (diff.empty()) {
        if (Nodes[node].Hot) Adopt(*this, *Nodes[node].Hot);
        else Nodes[node].Hot = Pin();
        SetPresent(*this, node);
        AppendTreeRecord(*this, RecordKind::Navigate, node, {});
    } else Restore(original);
    Release(original);
    return diff;
}

std::string History::ValidateReplay(int node) {
    if (node < 0 || node >= int(Nodes.size())) return "bad node";
    if (Nodes[node].ReplayBaseline) return {};
    assert(Present >= 0 && Nodes[Present].Hot);
    const int orig = Present;
    const auto &n = Nodes[node];
    if (Nodes[n.Parent].Hot) Restore(*Nodes[n.Parent].Hot);
    else if (!LoadNode(*this, n.Parent)) return "parent unloadable";
    const auto diff = ReplayStep(*this, n);
    Restore(*Nodes[orig].Hot);
    return diff;
}

std::string History::DiffImage(const std::vector<std::byte> &a_bytes, const std::vector<std::byte> &b_bytes) const {
    if (a_bytes == b_bytes) return {};
    std::span<const std::byte> a{a_bytes}, b{b_bytes};
    for (const auto i : Order) {
        const auto &t = Tracks[i];
        uint64_t la, ca, lb, cb;
        Take(a, la);
        Take(a, ca);
        Take(b, lb);
        Take(b, cb);
        const auto skip = [](std::span<const std::byte> &in, uint64_t count) {
            const auto *start = in.data();
            for (uint64_t i = 0; i < count; ++i) {
                uint64_t slot;
                uint32_t size;
                Take(in, slot);
                Take(in, size);
                in = in.subspan(size);
            }
            return std::span<const std::byte>{start, in.data()};
        };
        const auto sa = skip(a, ca), sb = skip(b, cb);
        if (la != lb || ca != cb || !std::equal(sa.begin(), sa.end(), sb.begin(), sb.end())) return t.Name;
    }
    return "(unknown)";
}

HistoryStats History::Stats() const {
    HistoryStats s;
    s.MetadataBytes = Nodes.capacity() * sizeof(HistoryNode);
    for (const auto &t : Tracks) {
        const auto &ts = t.Trie->Stats();
        s.OwnedBytes += ts.OwnedBytes;
        s.Nodes += ts.Nodes;
        s.AliasedNodes += ts.AliasedNodes;
        s.SlabBytes += t.Trie->SlabBytes();
        s.HashBytes += t.Trie->HashStorageBytes();
        s.ManifestBytes += t.Trie->ManifestBytes();
        s.MetadataBytes += t.Trie->ChangedSlots.capacity() * sizeof(uint64_t);
    }
    for (const auto *log : {&LeafLog, &NodeLog})
        s.MetadataBytes += log->Idx.bucket_count() * sizeof(void *) + log->Idx.size() * (sizeof(decltype(log->Idx)::value_type) + 2 * sizeof(void *));
    for (const auto &n : Nodes) {
        s.MetadataBytes += n.Children.capacity() * sizeof(int) + n.Action.capacity() + n.Label.capacity() + 1 + n.Stamps.capacity() * sizeof(Stamp) + n.Roots.capacity() * sizeof(Hash128);
        if (n.Hot) {
            s.MetadataBytes += n.Hot->Versions.capacity() * sizeof(Version);
            ++s.HotNodes;
        } else ++s.ColdNodes;
    }
    const auto [pending, peak] = Log.PendingMemory();
    s.PendingBytes = pending;
    s.PeakPendingBytes = std::max(peak, PeakPendingBytes);
    for (const auto &p : Pending) s.PendingBytes += p.capacity();
    return s;
}

bool History::Check(std::string &why) const {
    for (size_t i = 0; i < Tracks.size(); ++i) {
        const auto versions = AllVersions(*this, i);
        if (!Tracks[i].Trie->Check(why, versions)) {
            why = Tracks[i].Name + ": " + why;
            return false;
        }
    }
    return true;
}

bool History::Audit(std::string &why) { return Check(why) && CheckHashes(*this, why); }

int History::FindPosition(const HistoryPosition &position) const {
    const auto matches = [&](const HistoryNode &node) { return node.Stamps == position.Stamps && node.Roots == position.Roots; };
    if (position.Node >= 0 && position.Node < int(Nodes.size()) && matches(Nodes[position.Node])) return position.Node;
    const auto it = std::ranges::find_if(Nodes, matches);
    return it == Nodes.end() ? -1 : int(it - Nodes.begin());
}

bool History::Open(const std::filesystem::path &dir, const HistoryPosition *position) {
    if (!CheckIO(*this, Log.FlushAndWait())) return false;
    History candidate;
    candidate.Tracks = Tracks;
    candidate.SchemaRevision = SchemaRevision;
    uint64_t tree_size{};
    LoadPlan plan;
    if (!ReadProject(candidate, dir, tree_size))
        return CheckIO(*this, "cannot open project: incompatible format or missing or corrupt records");
    if (position) candidate.Present = candidate.FindPosition(*position);
    if (candidate.Present < 0 || !PrepareLoad(candidate, candidate.Present, plan))
        return CheckIO(*this, "cannot restore project position: missing or corrupt records");
    // Truncate incomplete records after validating the target state.
    for (const auto &[name, size] : {std::pair{LeafLog.Name, candidate.LeafLog.Size}, {NodeLog.Name, candidate.NodeLog.Size}, {TreeLogName, tree_size}}) {
        std::error_code ec;
        std::filesystem::resize_file(dir / name, size, ec);
        if (ec) return CheckIO(*this, "cannot truncate " + (dir / name).string() + ": " + ec.message());
    }
    if (!AdoptProject(*this, candidate)) return false;
    ApplyLoad(*this, candidate.Present, plan);
    Nodes[candidate.Present].Hot = Pin();
    SetPresent(*this, candidate.Present);
    return true;
}
} // namespace store
