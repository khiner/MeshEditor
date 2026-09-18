#include "project/store/History.h"

#include "project/ComponentPool.h"
#include "project/store/Pages.h"
#include "project/store/Records.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>

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

// Run fn on the owner of a track.
template<typename H> decltype(auto) With(H &history, const History::Tracked &t, auto &&fn) {
    switch (t.K) {
        case History::Kind::Pages: return fn(*history.PageTracks[t.Index]);
        case History::Kind::Records: return fn(*history.RecordTracks[t.Index]);
        case History::Kind::Pool: return fn(*history.PoolTracks[t.Index]);
    }
    std::unreachable();
}
template<typename H> LiveTrie &TrieOf(H &history, const History::Tracked &t) {
    return With(history, t, [](auto &owner) -> LiveTrie & { return owner.Trie; });
}
Stamp CurrentStamp(History &history, size_t track) {
    return With(history, history.Tracks[track], [](auto &owner) {
        owner.Settle();
        return owner.Trie.CurrentStamp(owner.Length());
    });
}
Version PinTrack(History &history, size_t track) {
    return With(history, history.Tracks[track], [](auto &owner) {
        owner.Settle();
        return owner.Trie.Pin(owner.Length());
    });
}
bool RestoreTrack(History &history, size_t track, const Version &v) {
    return With(history, history.Tracks[track], [&](auto &owner) { return owner.Restore(v); });
}
void LoadTrack(History &history, size_t track, uint64_t length, const LoadPlan &plan) {
    With(history, history.Tracks[track], [&](auto &owner) { owner.Load(length, plan.Changes[track], plan.Leaves); });
}
// Serialize present slots of a version as [u64 length][u64 count]([u64 slot][u32 size][bytes])*.
void MaterializeTrack(History &history, size_t track, const Version &v, std::vector<std::byte> &out) {
    Put(out, v.S.Length);
    const auto count_pos = out.size();
    Put(out, uint64_t{0});
    uint64_t count = 0;
    const auto put = [&](uint64_t slot, std::span<const std::byte> bytes) {
        Put(out, slot);
        Put(out, uint32_t(bytes.size()));
        out.insert(out.end(), bytes.begin(), bytes.end());
        ++count;
    };
    With(history, history.Tracks[track], [&](auto &owner) {
        for (const auto &run : owner.Trie.Materialize(v)) {
            if (run.Owned) put(run.Slot, owner.Encode(*run.Owned));
            else {
                for (uint64_t s = run.Slot, end = run.Slot + run.Count; s < end; ++s)
                    if (owner.Present(s)) put(s, owner.Read(s));
            }
        }
    });
    std::memcpy(out.data() + count_pos, &count, sizeof(count));
}
bool CheckTrackHashes(History &history, size_t track, std::string &why) {
    return With(history, history.Tracks[track], [&](auto &owner) {
        owner.Settle();
        std::vector<std::pair<uint64_t, Hash128>> live;
        for (uint64_t s = 0, n = owner.Trie.SlotsFor(owner.Length()); s < n; ++s) {
            if (!owner.Present(s)) continue;
            const auto bytes = owner.Read(s);
            if (owner.Trie.PageBytes && IsZero(bytes)) continue;
            live.emplace_back(s, HashBytes(bytes));
        }
        return owner.Trie.CheckHashes(why, live);
    });
}

// Disable track writes through AfterTracks, then enable them for AfterRestore.
struct PipelineScope {
    History &Hist;
    explicit PipelineScope(History &h) : Hist(h) {
        for (auto &t : Hist.Tracks) TrieOf(Hist, t).ExternalWritesForbidden = true;
    }
    ~PipelineScope() {
        for (auto &t : Hist.Tracks) TrieOf(Hist, t).ExternalWritesForbidden = false;
    }
};

std::string ReplayStep(History &history, const HistoryNode &node) {
    history.Callbacks.Replay(node.Action);
    for (size_t i = 0; i < history.Tracks.size(); ++i) {
        if (CurrentStamp(history, i) != node.Stamps[i]) return history.Tracks[i].Name;
    }
    return {};
}

std::vector<std::byte> MaterializeSnapshot(History &history, const Snapshot &s) {
    std::vector<std::byte> out;
    for (const auto i : history.Order) MaterializeTrack(history, i, s.Versions[i], out);
    return out;
}

uint64_t OwnedBytes(const History &history) {
    uint64_t total = 0;
    for (const auto &t : history.Tracks) total += TrieOf(history, t).Stats().OwnedBytes;
    return total;
}

std::vector<Stamp> CurrentStamps(History &history) {
    std::vector<Stamp> out;
    out.reserve(history.Tracks.size());
    for (size_t i = 0; i < history.Tracks.size(); ++i) out.push_back(CurrentStamp(history, i));
    return out;
}

std::vector<Version> AllVersions(const History &history, size_t track) {
    std::vector<Version> out;
    for (const auto &n : history.Nodes)
        if (n.Hot) out.push_back(n.Hot->Versions[track]);
    return out;
}

bool CheckHashes(History &history, std::string &why) {
    for (size_t i = 0; i < history.Tracks.size(); ++i) {
        if (!CheckTrackHashes(history, i, why)) {
            why = history.Tracks[i].Name + ": " + why;
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
        if (!RestoreTrack(history, i, s.Versions[i])) {
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

std::filesystem::path LogPath(const std::filesystem::path &dir, size_t file) {
    constexpr const char *Names[]{"leaves.log", "nodes.log", TreeLogName};
    return dir / Names[file];
}

// Report the first failed stream.
std::string StreamError(History &history) {
    for (size_t i = 0; i < history.Streams.size(); ++i)
        if (history.Streams[i].is_open() && !history.Streams[i]) return "cannot write " + LogPath(history.Dir, i).string();
    return {};
}

// Flush every open stream to the OS and report the first failure.
std::string Flush(History &history) {
    for (auto &stream : history.Streams)
        if (stream.is_open()) stream.flush();
    return StreamError(history);
}

constexpr size_t StreamBufferBytes = 1u << 20;
std::string OpenStreams(std::array<std::ofstream, 3> &streams, std::array<std::vector<char>, 3> &buffers, const std::filesystem::path &dir, bool truncate) {
    const auto mode = std::ios::binary | (truncate ? std::ios::trunc : std::ios::app);
    for (size_t i = 0; i < streams.size(); ++i) {
        buffers[i].resize(StreamBufferBytes);
        streams[i].rdbuf()->pubsetbuf(buffers[i].data(), std::streamsize(buffers[i].size()));
        streams[i].open(LogPath(dir, i), mode);
        if (!streams[i]) return "cannot open " + LogPath(dir, i).string();
    }
    return {};
}

std::string CloseStreams(History &history) {
    std::string error;
    for (size_t i = 0; i < history.Streams.size(); ++i) {
        auto &stream = history.Streams[i];
        if (!stream.is_open()) continue;
        stream.close();
        if (!stream && error.empty()) error = "cannot close " + LogPath(history.Dir, i).string();
        stream.clear();
    }
    return error;
}

void Append(History &history, size_t file, std::span<const std::byte> bytes) {
    history.Streams[file].write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
}

std::vector<std::byte> Descriptor(const History &history) {
    std::vector<std::byte> out;
    Put(out, uint64_t{0x45524f5453545350}); // PSTSTORE
    Put(out, uint32_t{4}); // store format revision
    Put(out, history.SchemaRevision);
    Put(out, uint32_t(history.Tracks.size()));
    for (const auto &t : history.Tracks) {
        const auto &trie = TrieOf(history, t);
        Put(out, uint32_t(t.Name.size()));
        for (char c : t.Name) Put(out, c);
        Put(out, int32_t(t.Phase));
        Put(out, trie.Levels);
        Put(out, uint64_t(trie.PageBytes));
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
            HistoryNode n{.Action = {payload.begin(), payload.begin() + asz}};
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

// Take over the candidate's tree and open streams.
bool AdoptProject(History &history, History &candidate) {
    if (!CheckIO(history, Flush(candidate))) return false;
    CloseStreams(history);
    for (int i = 0; i < int(history.Nodes.size()); ++i) ReleaseNode(history, i);
    history.Nodes = std::move(candidate.Nodes);
    history.LeafLog = std::move(candidate.LeafLog);
    history.NodeLog = std::move(candidate.NodeLog);
    history.Dir = std::move(candidate.Dir);
    history.Streams = std::move(candidate.Streams);
    history.StreamBuffers = std::move(candidate.StreamBuffers);
    history.Present = candidate.Present;
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
        auto &trie = TrieOf(history, history.Tracks[i]);
        const auto limit = trie.SlotsFor(n.Stamps[i].Length);
        if (limit > SlotSpan(trie.Levels)) return false;
        const auto live_stamp = CurrentStamp(history, i);
        const auto live_root = trie.ManifestRoot(live_stamp.Length);
        const auto live_limit = trie.SlotsFor(live_stamp.Length);
        auto state = live_stamp.H;
        const auto diff = [&](this auto &&self, Hash128 live, Hash128 target, uint32_t level, uint64_t index) -> bool {
            // Validate target slots beyond the new length even when subtree hashes match.
            if (live == target && (target == Hash128{} || limit >= live_limit || (index + 1) * SlotSpan(level) <= limit)) return true;
            if (level) {
                const auto before = trie.ChildrenAt(level - 1, index, live_stamp.Length);
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
                if (it == history.LeafLog.Idx.end() || (trie.PageBytes && it->second.Size != trie.PageBytes)) return false;
                const auto [leaf, inserted] = plan.Leaves.try_emplace(target);
                if (inserted && !ReadExtent(leaves, it->second, target, leaf->second)) return false;
            }
            plan.Changes[i].push_back({index, target});
            return true;
        };
        if (!diff(live_root, n.Roots[i], trie.Levels, 0) || state != n.Stamps[i].H) return false;
    }
    return true;
}

void ApplyLoad(History &history, int node, const LoadPlan &plan) {
    const auto &n = history.Nodes[node];
    RunPipeline(history, [&](size_t i) { LoadTrack(history, i, n.Stamps[i].Length, plan); });
    for (size_t i = 0; i < history.Tracks.size(); ++i) {
        if (CurrentStamp(history, i) == n.Stamps[i]) continue;
        AddIntegrityError(history, history.Tracks[i].Name + ": cold load verification failed");
        std::fprintf(stderr, "[history] %s: cold load verification failed\n", history.Tracks[i].Name.c_str());
        assert(false && "cold load verification failed: loaded state does not hash to the recorded stamp");
    }
}

void AppendContent(History &history, History::ContentLog &log, Hash128 hash, std::span<const std::byte> payload) {
    if (!log.Idx.try_emplace(hash, History::Extent{log.Size + ContentHeader, uint32_t(payload.size())}).second) return;
    std::vector<std::byte> header;
    Put(header, hash);
    Put(header, uint32_t(payload.size()));
    Append(history, log.File, header);
    Append(history, log.File, payload);
    log.Size += ContentHeader + payload.size();
}

Hash128 BuildManifest(History &history, size_t track) {
    return With(history, history.Tracks[track], [&](auto &owner) {
        owner.Settle();
        const auto length = owner.Length();
        auto &trie = owner.Trie;
        const auto root = trie.ManifestRoot(length);
        // Indexed roots have indexed descendants.
        const auto persist = [&](this auto &&self, Hash128 hash, uint32_t level, uint64_t index) -> void {
            if (hash == Hash128{} || history.NodeLog.Idx.contains(hash)) return;
            const auto children = trie.ChildrenAt(level, index, length);
            for (uint64_t digit = 0; digit < Fanout; ++digit) {
                const auto child = children[digit];
                if (child == Hash128{}) continue;
                if (level) self(child, level - 1, index * Fanout + digit);
                else if (!history.LeafLog.Idx.contains(child)) {
                    AppendContent(history, history.LeafLog, child, owner.Read(index * Fanout + digit));
                }
            }
            AppendContent(history, history.NodeLog, hash, ManifestRecord{level, children}.View());
        };
        persist(root, trie.Levels - 1, 0);
        return root;
    });
}

void AppendTreeRecord(History &history, RecordKind kind, int parent, const std::vector<std::byte> &payload) {
    assert(history.Streams[FileTree].is_open() && "the tree log requires Begin or Open");
    // Tree record format: [u32 len][u8 kind][i32 parent][payload].
    std::vector<std::byte> record;
    Put(record, uint32_t(1 + 4 + payload.size()));
    Put(record, uint8_t(kind));
    Put(record, int32_t(parent));
    record.insert(record.end(), payload.begin(), payload.end());
    Append(history, FileTree, record);
}

void PersistNode(History &history, RecordKind kind, int node) {
    assert(history.Streams[FileTree].is_open() && "commit requires Begin or Open");
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
    if (!CheckIO(history, Flush(history))) return false;
    LoadPlan plan;
    if (!PrepareLoad(history, node, plan)) return CheckIO(history, "cold load contains missing or corrupt records");
    ApplyLoad(history, node, plan);
    return true;
}

void AddTrack(History &history, std::string name, int phase, History::Kind kind, size_t index) {
    history.Tracks.push_back({std::move(name), phase, kind, index});
    // Keep serialized track indices in registration order when sorting restoration order.
    history.Order.insert(std::ranges::upper_bound(history.Order, phase, {}, [&](size_t i) { return history.Tracks[i].Phase; }), history.Tracks.size() - 1);
}
} // namespace

void History::Track(Pages &p, std::string name, int phase) {
    PageTracks.push_back(&p);
    AddTrack(*this, std::move(name), phase, Kind::Pages, PageTracks.size() - 1);
}
void History::Track(Records &r, std::string name, int phase) {
    RecordTracks.push_back(&r);
    AddTrack(*this, std::move(name), phase, Kind::Records, RecordTracks.size() - 1);
}
void History::Track(project::ComponentPool &p, std::string name, int phase) {
    PoolTracks.push_back(&p);
    AddTrack(*this, std::move(name), phase, Kind::Pool, PoolTracks.size() - 1);
}

Snapshot History::Pin() {
    Snapshot s;
    for (size_t i = 0; i < Tracks.size(); ++i) s.Versions.push_back(PinTrack(*this, i));
    return s;
}

void History::SettleHashes() {
    for (const auto &t : Tracks) With(*this, t, [](auto &owner) { owner.Settle(); });
}

std::string History::TakeIntegrityError() {
    auto error = std::exchange(IntegrityError, {});
    return error.empty() ? StreamError(*this) : error;
}

void History::Restore(const Snapshot &s) {
    RunPipeline(*this, [&](size_t i) {
        if (!RestoreTrack(*this, i, s.Versions[i])) {
            AddIntegrityError(*this, Tracks[i].Name + ": restored state hash mismatch");
            std::fprintf(stderr, "[history] %s: restored state hash mismatch\n", Tracks[i].Name.c_str());
            assert(false && "restored state hash does not match pinned hash");
        }
    });
}

void History::Release(Snapshot &snapshot) {
    for (size_t i = 0; i < Tracks.size(); ++i) TrieOf(*this, Tracks[i]).Release(snapshot.Versions[i]);
    snapshot.Versions.clear();
}

bool History::Begin(const std::filesystem::path &dir) {
    assert(Dir.empty() || Dir != dir);
    if (!CheckIO(*this, Flush(*this))) return false;
    History candidate{.SchemaRevision = SchemaRevision, .Tracks = Tracks, .PageTracks = PageTracks, .RecordTracks = RecordTracks, .PoolTracks = PoolTracks, .Order = Order, .Dir = dir};
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return CheckIO(*this, "cannot create " + dir.string() + ": " + ec.message());
    if (!CheckIO(*this, OpenStreams(candidate.Streams, candidate.StreamBuffers, dir, true))) return false;
    Append(candidate, FileTree, Descriptor(*this));
    HistoryNode root{.Label = "Root", .Hot = Pin(), .Stamps = CurrentStamps(*this)};
    candidate.Nodes.push_back(std::move(root));
    SetPresent(candidate, 0);
    PersistNode(candidate, RecordKind::Root, 0);
    if (!AdoptProject(*this, candidate)) return false;
    SetPresent(*this, 0);
    return true;
}

bool History::Close() {
    const bool saved = CheckIO(*this, Flush(*this)) && CheckIO(*this, CloseStreams(*this));
    for (int i = 0; i < int(Nodes.size()); ++i) ReleaseNode(*this, i);
    Nodes.clear();
    ++Revision;
    Present = -1;
    VisitCounter = 0;
    Dir.clear();
    LeafLog.Idx.clear();
    NodeLog.Idx.clear();
    LeafLog.Size = NodeLog.Size = 0;
    return saved;
}

bool History::Relocate(const std::filesystem::path &dir) {
    if (!CheckIO(*this, Flush(*this))) return false;
    std::array<std::vector<char>, 3> buffers;
    std::array<std::ofstream, 3> streams;
    if (!CheckIO(*this, OpenStreams(streams, buffers, dir, false))) return false;
    if (!CheckIO(*this, CloseStreams(*this))) return false;
    Streams = std::move(streams);
    StreamBuffers = std::move(buffers);
    Dir = dir;
    return true;
}

int History::Commit(std::string label, std::vector<std::byte> action) {
    assert(Present >= 0);
    const auto stamps = CurrentStamps(*this);
    if (stamps == Nodes[Present].Stamps) {
        Adopt(*this, *Nodes[Present].Hot); // Release redundant copies for equal state.
        return Present;
    }
    const auto adopt = [&](int node) {
        auto &n = Nodes[node];
        if (n.Hot) Adopt(*this, *n.Hot);
        else n.Hot = Pin();
        SetPresent(*this, node);
        AppendTreeRecord(*this, RecordKind::Navigate, node, {});
        return node;
    };
    for (const int child : Nodes[Present].Children)
        if (Nodes[child].Stamps == stamps) return adopt(child);
    if (const int parent = Nodes[Present].Parent; parent >= 0 && Nodes[parent].Stamps == stamps) return adopt(parent);
    const int id = int(Nodes.size());
    HistoryNode n{.Parent = Present, .Action = std::move(action), .Label = std::move(label), .Depth = Nodes[Present].Depth + 1, .ReplayBaseline = false, .Hot = Pin(), .Stamps = stamps};
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
    return CheckIO(*this, Flush(*this));
}

bool History::Clear(const HistoryPosition *saved) {
    if (!CheckIO(*this, Flush(*this))) return false;
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
    if (!CheckIO(*this, Flush(*this))) {
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
    if (victims.empty() || !CheckIO(*this, Flush(*this))) return;
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

HistoryStats History::Stats() const {
    HistoryStats s{.OwnedBytes = OwnedBytes(*this), .SharedNodeBytes = SharedNodePoolBytes()};
    for (const auto &n : Nodes) {
        if (n.Hot) ++s.HotNodes;
        else ++s.ColdNodes;
    }
    return s;
}

bool History::Check(std::string &why) const {
    for (size_t i = 0; i < Tracks.size(); ++i) {
        const auto versions = AllVersions(*this, i);
        if (!TrieOf(*this, Tracks[i]).Check(why, versions)) {
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
    if (!CheckIO(*this, Flush(*this))) return false;
    History candidate{.SchemaRevision = SchemaRevision, .Tracks = Tracks, .PageTracks = PageTracks, .RecordTracks = RecordTracks, .PoolTracks = PoolTracks};
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
    if (!CheckIO(*this, OpenStreams(candidate.Streams, candidate.StreamBuffers, dir, false))) return false;
    if (!AdoptProject(*this, candidate)) return false;
    ApplyLoad(*this, candidate.Present, plan);
    Nodes[candidate.Present].Hot = Pin();
    SetPresent(*this, candidate.Present);
    return true;
}
} // namespace store
