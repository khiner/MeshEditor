#include "project/store/History.h"

#include "project/ComponentPool.h"
#include "project/WorkspacePool.h"
#include "project/store/Pages.h"
#include "project/store/Records.h"

#include <print>
#include <unordered_set>

namespace store {
namespace {
constexpr size_t FileTree = 2;
constexpr uint64_t TreeMagic = 0x45524f5453545350; // PSTSTORE
constexpr uint64_t ActionsMagic = 0x534e544341545350; // PSTACTNS
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
bool Take(std::span<const std::byte> &in, auto &v) {
    if (in.size() < sizeof(v)) return false;
    std::memcpy(&v, in.data(), sizeof(v));
    in = in.subspan(sizeof(v));
    return true;
}

void Put(std::vector<std::byte> &out, const std::optional<uint32_t> &node) {
    Put(out, uint8_t(node.has_value()));
    if (node) Put(out, *node);
}
bool Take(std::span<const std::byte> &in, std::optional<uint32_t> &node) {
    uint8_t present;
    if (!Take(in, present) || present > 1) return false;
    node.reset();
    if (present) {
        uint32_t id;
        if (!Take(in, id)) return false;
        node = id;
    }
    return true;
}

void PutState(std::vector<std::byte> &out, std::span<const Stamp> stamps, std::span<const Hash128> roots) {
    for (size_t i = 0; i < stamps.size(); ++i) {
        Put(out, stamps[i].H);
        Put(out, stamps[i].Length);
        Put(out, roots[i]);
    }
}

bool TakeState(std::span<const std::byte> &in, HistoryNode &node, size_t tracks) {
    node.Stamps.resize(tracks);
    node.Roots.resize(tracks);
    for (size_t i = 0; i < tracks; ++i) {
        if (!Take(in, node.Stamps[i].H) || !Take(in, node.Stamps[i].Length) || !Take(in, node.Roots[i])) return false;
    }
    return true;
}

// Node actions format: [u32 size][actions][u32 size][label].
void PutActions(const History &history, std::vector<std::byte> &out, const HistoryNode &node, std::span<const uint32_t> remap = {}) {
    const auto actions = history.ActionsOf(node);
    const auto workspaces = history.WorkspacesOf(node);
    Put(out, uint32_t(actions.size()));
    out.insert(out.end(), actions.begin(), actions.end());
    Put(out, uint32_t(workspaces.size()));
    for (auto id : workspaces) Put(out, remap.empty() ? id : remap[id]);
    Put(out, uint32_t(node.Label.size()));
    const auto *label = reinterpret_cast<const std::byte *>(node.Label.data());
    out.insert(out.end(), label, label + node.Label.size());
}

bool TakeActions(History &history, std::span<const std::byte> &in, HistoryNode &node) {
    uint32_t actions_size, label_size;
    if (!Take(in, actions_size) || in.size() < actions_size) return false;
    node.Actions = {history.ActionBytes.size(), actions_size};
    history.ActionBytes.insert(history.ActionBytes.end(), in.begin(), in.begin() + actions_size);
    in = in.subspan(actions_size);
    uint32_t contexts;
    if (!Take(in, contexts) || contexts > in.size() / sizeof(uint32_t)) return false;
    node.Workspaces = {history.WorkspaceRefs.size(), contexts};
    for (uint32_t i = 0; i < contexts; ++i) {
        uint32_t id;
        if (!Take(in, id)) return false;
        history.WorkspaceRefs.push_back(id);
    }
    if (!Take(in, label_size) || in.size() < label_size) return false;
    node.Label.assign(reinterpret_cast<const char *>(in.data()), label_size);
    in = in.subspan(label_size);
    return true;
}

// A node without state records only its actions.
std::vector<std::byte> NodePayload(const History &history, const HistoryNode &node) {
    std::vector<std::byte> payload;
    PutActions(history, payload, node);
    PutState(payload, node.Stamps, node.Roots);
    return payload;
}
struct LoadPlan {
    std::vector<std::vector<std::pair<uint64_t, Hash128>>> Changes;
    std::unordered_map<Hash128, std::vector<std::byte>, Hash128Hasher> Leaves;
};

decltype(auto) With(auto &history, const History::Tracked &t, auto &&fn) {
    switch (t.K) {
        case History::Kind::Pages: return fn(*history.PageTracks[t.Index]);
        case History::Kind::Records: return fn(*history.RecordTracks[t.Index]);
        case History::Kind::Pool: return fn(*history.PoolTracks[t.Index]);
        case History::Kind::Workspace: return fn(*history.Workspace);
    }
    std::unreachable();
}
const History::Tracked &TrackAt(const History &history, size_t track) {
    static const History::Tracked workspace{"workspace", 0, History::Kind::Workspace, 0};
    return track == history.Tracks.size() ? workspace : history.Tracks[track];
}
LiveTrie &TrieOf(auto &history, const History::Tracked &t) {
    return With(history, t, [](auto &owner) -> LiveTrie & { return owner.Trie; });
}
Stamp CurrentStamp(History &history, size_t track) {
    return With(history, TrackAt(history, track), [](auto &owner) {
        owner.Settle();
        return owner.Trie.CurrentStamp(owner.Length());
    });
}
Version PinTrack(History &history, size_t track) {
    return With(history, TrackAt(history, track), [](auto &owner) {
        owner.Settle();
        return owner.Trie.Pin(owner.Length());
    });
}
bool RestoreTrack(History &history, size_t track, const Version &v) {
    return With(history, TrackAt(history, track), [&](auto &owner) { return owner.Restore(v); });
}
void LoadTrack(History &history, size_t track, uint64_t length, const LoadPlan &plan) {
    With(history, TrackAt(history, track), [&](auto &owner) { owner.Load(length, plan.Changes[track], plan.Leaves); });
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
    With(history, TrackAt(history, track), [&](auto &owner) {
        for (const auto &run : owner.Trie.Materialize(v)) {
            if (run.Owned) put(run.Slot, owner.Encode(*run.Owned));
            else {
                for (uint64_t s = run.Slot, end = run.Slot + run.Count; s < end; ++s) {
                    if (owner.Present(s)) put(s, owner.Read(s));
                }
            }
        }
    });
    std::memcpy(out.data() + count_pos, &count, sizeof(count));
}
bool CheckTrackHashes(History &history, size_t track, std::string &why) {
    return With(history, TrackAt(history, track), [&](auto &owner) {
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
        for (const auto &t : Hist.Tracks) TrieOf(Hist, t).ExternalWritesForbidden = true;
    }
    ~PipelineScope() {
        for (const auto &t : Hist.Tracks) TrieOf(Hist, t).ExternalWritesForbidden = false;
    }
};

// Replays a node from its parent's state, returning an action error or the first track whose stamp differs.
std::string ReplayStep(History &history, const HistoryNode &node) {
    const auto replayed = history.Callbacks.Replay(history, history.ActionsOf(node), history.WorkspacesOf(node));
    if (!replayed) return replayed.error();
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
    if (history.Workspace) total += history.Workspace->Trie.Stats().OwnedBytes;
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
    for (const auto &n : history.Nodes) {
        if (n.Hot) out.push_back(n.Hot->Versions[track]);
    }
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

std::expected<void, std::string> RunPipeline(History &history, auto &&per_track) {
    if (history.Callbacks.BeforeRestore) history.Callbacks.BeforeRestore();
    {
        const PipelineScope scope{history};
        for (const auto i : history.Order) per_track(i);
        if (history.Callbacks.AfterTracks) history.Callbacks.AfterTracks();
    }
    if (history.Callbacks.AfterRestore) return history.Callbacks.AfterRestore();
    return {};
}

void AddIntegrityError(History &history, std::string_view what) {
    if (!history.IntegrityError.empty()) history.IntegrityError += ", ";
    history.IntegrityError += what;
}

void Adopt(History &history, const Snapshot &s) {
    for (const auto i : history.Order) {
        if (!RestoreTrack(history, i, s.Versions[i])) {
            AddIntegrityError(history, history.Tracks[i].Name + ": equal-state adoption failed");
            assert(false && "equal-state adoption failed");
        }
    }
}

void ReleaseNode(History &history, uint32_t node) {
    auto &n = history.Nodes[node];
    if (!n.Hot) return;
    history.Release(*n.Hot);
    n.Hot.reset();
}

void SelectLineage(History &history, uint32_t node) {
    for (auto child = node; history.Nodes[child].Parent;) {
        const auto parent = *history.Nodes[child].Parent;
        history.Nodes[parent].RedoChild = child;
        child = parent;
    }
}

void SetPresent(History &history, uint32_t node) {
    history.Present = node;
    SelectLineage(history, node);
    history.Nodes[node].LastVisited = ++history.VisitCounter;
}

// Release descendant snapshots and detach those nodes from the history tree.
void DropDescendants(History &history, uint32_t node) {
    auto pending = history.Nodes[node].Children;
    history.Nodes[node].Children.clear();
    history.Nodes[node].RedoChild.reset();
    while (!pending.empty()) {
        auto &n = history.Nodes[pending.back()];
        pending.pop_back();
        pending.append_range(n.Children);
        n.Children.clear();
        n.RedoChild.reset();
        for (auto id : history.WorkspacesOf(n)) history.ReleaseWorkspace(id);
        n.Workspaces = {};
        n.Actions = {};
        if (n.Hot) history.Release(*n.Hot);
        n.Hot.reset();
    }
}

uint32_t TreeDistance(const History &history, uint32_t a, uint32_t b) {
    auto x = a, y = b;
    while (history.Nodes[x].Depth > history.Nodes[y].Depth) x = *history.Nodes[x].Parent;
    while (history.Nodes[y].Depth > history.Nodes[x].Depth) y = *history.Nodes[y].Parent;
    while (x != y) {
        x = *history.Nodes[x].Parent;
        y = *history.Nodes[y].Parent;
    }
    return history.Nodes[a].Depth + history.Nodes[b].Depth - 2 * history.Nodes[x].Depth;
}

bool CheckIO(History &history, std::string error) {
    if (error.empty()) return true;
    if (history.IntegrityError.empty()) history.IntegrityError = std::move(error);
    return false;
}

std::filesystem::path LogPath(const std::filesystem::path &dir, size_t file) {
    constexpr const char *Names[]{"leaves.log", "nodes.log", History::TreeLogName};
    return dir / Names[file];
}

std::string StreamError(History &history) {
    for (size_t i = 0; i < history.Streams.size(); ++i) {
        if (history.Streams[i].is_open() && !history.Streams[i]) return "cannot write " + LogPath(history.Dir, i).string();
    }
    return {};
}

// Flush every open stream to the OS and report the first failure.
std::string Flush(History &history) {
    for (auto &stream : history.Streams) {
        if (stream.is_open()) stream.flush();
    }
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

// Read the tree log, keeping only the records whose roots resolve in the content logs.
bool ReadProject(History &history, const std::filesystem::path &dir, uint64_t &tree_size) {
    std::vector<std::byte> bytes;
    if (!ReadFile(dir / History::TreeLogName, bytes)) return false;
    auto rest = std::span<const std::byte>{bytes};
    uint64_t magic;
    if (!Take(rest, magic) || magic != TreeMagic) return false;
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
        for (const auto child : children) {
            if (child != Hash128{} && (lv > 0 ? !self(child) : !history.LeafLog.Idx.contains(child))) return false;
        }
        return memo = true;
    };
    // Node IDs follow record order, so discard all records after the first invalid record.
    tree_size = sizeof(TreeMagic);
    while (!rest.empty()) {
        uint32_t len;
        RecordKind kind;
        std::optional<uint32_t> parent;
        if (!Take(rest, len) || rest.size() < len) break;
        auto payload = rest.first(len);
        rest = rest.subspan(len);
        if (!Take(payload, kind) || !Take(payload, parent)) break;
        if (kind == RecordKind::Workspace) {
            WorkspaceVersion version;
            if (!parent || !Take(payload, version.S) || !Take(payload, version.Root) || !payload.empty() || !resolvable(version.Root)) break;
            if (*parent > history.Workspaces.size()) break;
            if (*parent == history.Workspaces.size()) history.Workspaces.emplace_back();
            history.Workspaces[*parent] = std::move(version);
            history.WrittenWorkspaces = uint32_t(history.Workspaces.size());
        } else if (kind == RecordKind::Kept) {
            Hash128 state;
            uint32_t count;
            if (parent || !Take(payload, state) || !Take(payload, count) || payload.size() != size_t(count) * (sizeof(uint32_t) + sizeof(Hash128))) break;
            KeptState kept;
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t track;
                Hash128 root;
                Take(payload, track);
                Take(payload, root);
                kept.Tracks.push_back(track);
                kept.Roots.push_back(root);
            }
            if (std::ranges::any_of(kept.Tracks, [&](uint32_t track) { return track >= history.Tracks.size(); })) break;
            // A kept state whose content a partial write lost stays unresolved.
            if (std::ranges::all_of(kept.Roots, resolvable)) history.Kept.insert_or_assign(state, std::move(kept));
        } else if (kind == RecordKind::State) {
            HistoryNode state;
            if (!parent || *parent >= history.Nodes.size() || !TakeState(payload, state, history.Tracks.size()) || !payload.empty() || !std::ranges::all_of(state.Roots, resolvable)) break;
            history.Nodes[*parent].Stamps = std::move(state.Stamps);
            history.Nodes[*parent].Roots = std::move(state.Roots);
        } else if (kind != RecordKind::Navigate) {
            HistoryNode n;
            if (!TakeActions(history, payload, n) || (!payload.empty() && !TakeState(payload, n, history.Tracks.size()))) break;
            if (kind == RecordKind::Root ? parent.has_value() : !parent || *parent >= history.Nodes.size()) break;
            if (!std::ranges::all_of(n.Roots, resolvable) || std::ranges::any_of(history.WorkspacesOf(n), [&](auto id) { return id >= history.Workspaces.size(); })) break;
            for (auto id : history.WorkspacesOf(n)) history.RetainWorkspace(id);
            std::optional<HistoryNode> saved;
            if (kind == RecordKind::Root && !payload.empty()) {
                saved.emplace();
                saved->Label = "Saved";
                saved->Children.push_back(1);
                if (!TakeState(payload, *saved, history.Tracks.size()) || !std::ranges::all_of(saved->Roots, resolvable)) break;
            }
            if (!payload.empty()) break;
            if (kind == RecordKind::Replace) {
                DropDescendants(history, *parent);
                auto &target = history.Nodes[*parent];
                for (auto id : history.WorkspacesOf(target)) history.ReleaseWorkspace(id);
                target.Workspaces = std::move(n.Workspaces);
                target.Actions = std::move(n.Actions);
                target.Label = std::move(n.Label);
                target.Stamps = std::move(n.Stamps);
                target.Roots = std::move(n.Roots);
                history.Present = parent;
                SelectLineage(history, *parent);
                tree_size += 4 + len;
                continue;
            }
            if (kind == RecordKind::Root) {
                for (const auto &old : history.Nodes)
                    for (auto id : history.WorkspacesOf(old)) history.ReleaseWorkspace(id);
                history.Nodes.clear();
                if (saved) {
                    history.Nodes.push_back(std::move(*saved));
                    n.Parent = 0;
                    n.Depth = 1;
                }
            } else {
                n.Parent = parent;
                n.Depth = history.Nodes[*parent].Depth + 1;
                n.ReplayBaseline = false;
                history.Nodes[*parent].Children.push_back(uint32_t(history.Nodes.size()));
            }
            history.Nodes.push_back(std::move(n));
            history.Present = uint32_t(history.Nodes.size() - 1);
            SelectLineage(history, *history.Present);
        } else if (parent && *parent < history.Nodes.size() && payload.empty()) {
            history.Present = parent;
            SelectLineage(history, *parent);
        } else {
            break;
        }
        tree_size += 4 + len;
    }
    history.CompactActions();
    // Exclude manifests with missing descendants from deduplication after a partial write.
    std::erase_if(history.NodeLog.Idx, [&](const auto &entry) { return !resolvable(entry.first); });
    return !history.Nodes.empty();
}

// Build the tree of an actions log, whose nodes have no state and whose root is the only baseline.
// Actions log format: [u64 magic][u32 node count][u32 present], then per node its optional parent and actions.
bool ReadActions(History &history, const std::filesystem::path &dir) {
    std::vector<std::byte> bytes;
    if (!ReadFile(dir / History::ActionsLogName, bytes)) return false;
    std::span<const std::byte> in{bytes};
    uint64_t magic;
    uint32_t count;
    uint32_t present;
    if (!Take(in, magic) || magic != ActionsMagic || !Take(in, count) || !Take(in, present) || present >= count) return false;
    uint32_t versions;
    if (!Take(in, versions) || versions > in.size() / (sizeof(Stamp) + sizeof(Hash128))) return false;
    history.Workspaces.resize(versions);
    for (auto &v : history.Workspaces)
        if (!Take(in, v.S) || !Take(in, v.Root)) return false;
    for (auto i = uint32_t{0}; i < count; ++i) {
        std::optional<uint32_t> parent;
        HistoryNode n;
        if (!Take(in, parent) || (i ? !parent || *parent >= i : parent.has_value()) || !TakeActions(history, in, n)) return false;
        if (i) {
            n.Parent = parent;
            n.Depth = history.Nodes[*parent].Depth + 1;
            n.ReplayBaseline = false;
            history.Nodes[*parent].Children.push_back(i);
            history.Nodes[*parent].RedoChild = i;
        }
        for (auto id : history.WorkspacesOf(n)) {
            if (id >= history.Workspaces.size()) return false;
            history.RetainWorkspace(id);
        }
        history.Nodes.push_back(std::move(n));
    }
    history.Dir = dir;
    history.Present = present;
    return in.empty();
}

bool AdoptProject(History &history, History &candidate) {
    if (!CheckIO(history, Flush(candidate))) return false;
    CloseStreams(history);
    for (auto i = uint32_t{0}; i < history.Nodes.size(); ++i) ReleaseNode(history, i);
    for (auto &v : history.Workspaces)
        if (v.Hot.Root) history.Workspace->Trie.Release(v.Hot);
    history.Workspaces = std::move(candidate.Workspaces);
    history.WrittenWorkspaces = candidate.WrittenWorkspaces;
    history.ActionBytes = std::move(candidate.ActionBytes);
    history.WorkspaceRefs = std::move(candidate.WorkspaceRefs);
    history.Nodes = std::move(candidate.Nodes);
    history.LeafLog = std::move(candidate.LeafLog);
    history.NodeLog = std::move(candidate.NodeLog);
    history.Kept = std::move(candidate.Kept);
    history.Dir = std::move(candidate.Dir);
    history.Streams = std::move(candidate.Streams);
    history.StreamBuffers = std::move(candidate.StreamBuffers);
    history.Present = candidate.Present;
    ++history.Revision;
    history.VisitCounter = 0;
    history.IntegrityError.clear();
    return true;
}

// Visit changed leaves, validating that the stored tree fits its target length.
bool DiffManifest(History &history, std::ifstream &nodes, size_t track, Hash128 root, uint64_t limit, auto &&visit) {
    auto &trie = TrieOf(history, TrackAt(history, track));
    if (limit > SlotSpan(trie.Levels)) return false;
    const auto live = CurrentStamp(history, track);
    const auto live_limit = trie.SlotsFor(live.Length);
    const auto diff = [&](this auto &&self, Hash128 before, Hash128 after, uint32_t level, uint64_t index) -> bool {
        if (before == after && (after == Hash128{} || limit >= live_limit || (index + 1) * SlotSpan(level) <= limit)) return true;
        if (!level) return (after == Hash128{} || index < limit) && visit(index, before, after);
        const auto old_children = trie.ChildrenAt(level - 1, index, live.Length);
        ManifestChildren children{};
        if (after != Hash128{}) {
            uint8_t recorded;
            if (!ReadManifestRecord(history, nodes, after, recorded, children) || recorded != level - 1) return false;
        }
        for (uint64_t digit = 0; digit < Fanout; ++digit)
            if (!self(old_children[digit], children[digit], level - 1, index * Fanout + digit)) return false;
        return true;
    };
    return diff(trie.ManifestRoot(live.Length), root, trie.Levels, 0);
}

// Validate hashes and content before mutating a document or workspace track.
bool PrepareTrackLoad(StateReader &reader, size_t track, Stamp target, Hash128 root, std::vector<std::pair<uint64_t, Hash128>> &changes, std::unordered_map<Hash128, std::vector<std::byte>, Hash128Hasher> &leaves) {
    auto &history = reader.Hist;
    auto &trie = TrieOf(history, TrackAt(history, track));
    auto state = CurrentStamp(history, track).H;
    const bool valid = reader.Open() && DiffManifest(history, reader.Nodes, track, root, trie.SlotsFor(target.Length), [&](uint64_t slot, Hash128 before, Hash128 after) {
                           if (before != Hash128{}) {
                               const Term term{slot, before};
                               state.A -= term.L0;
                               state.B -= term.L1;
                           }
                           if (after != Hash128{}) {
                               const Term term{slot, after};
                               state.A += term.L0;
                               state.B += term.L1;
                               const auto it = history.LeafLog.Idx.find(after);
                               if (it == history.LeafLog.Idx.end() || (trie.PageBytes && it->second.Size != trie.PageBytes)) return false;
                               const auto [leaf, inserted] = leaves.try_emplace(after);
                               if (inserted && !ReadExtent(reader.Leaves, it->second, after, leaf->second)) return false;
                           }
                           changes.emplace_back(slot, after);
                           return true;
                       });
    return valid && state == target.H;
}

bool PrepareLoad(History &history, uint32_t node, LoadPlan &plan) {
    const auto &n = history.Nodes[node];
    StateReader reader{history};
    plan.Changes.resize(history.Tracks.size());
    for (size_t i = 0; i < history.Tracks.size(); ++i)
        if (!PrepareTrackLoad(reader, i, n.Stamps[i], n.Roots[i], plan.Changes[i], plan.Leaves)) return false;
    return true;
}

std::expected<void, std::string> ApplyLoad(History &history, uint32_t node, const LoadPlan &plan) {
    const auto &n = history.Nodes[node];
    const auto restored = RunPipeline(history, [&](size_t i) { LoadTrack(history, i, n.Stamps[i].Length, plan); });
    if (!restored) return restored;
    for (size_t i = 0; i < history.Tracks.size(); ++i) {
        if (CurrentStamp(history, i) == n.Stamps[i]) continue;
        AddIntegrityError(history, history.Tracks[i].Name + ": cold load verification failed");
        std::println(stderr, "[history] {}: cold load verification failed", history.Tracks[i].Name);
        assert(false && "cold load verification failed: loaded state does not hash to the recorded stamp");
    }
    return {};
}

void AppendContent(History &history, History::ContentLog &log, Hash128 hash, std::span<const std::byte> payload) {
    if (!log.Idx.try_emplace(hash, History::Extent{log.Size + ContentHeader, uint32_t(payload.size())}).second) return;
    std::array<std::byte, ContentHeader> header;
    const auto size = uint32_t(payload.size());
    std::memcpy(header.data(), &hash, sizeof(hash));
    std::memcpy(header.data() + sizeof(hash), &size, sizeof(size));
    Append(history, log.File, header);
    Append(history, log.File, payload);
    log.Size += ContentHeader + payload.size();
}

Hash128 BuildManifest(History &history, size_t track) {
    return With(history, TrackAt(history, track), [&](auto &owner) {
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
                else if (!history.LeafLog.Idx.contains(child)) AppendContent(history, history.LeafLog, child, owner.Read(index * Fanout + digit));
            }
            AppendContent(history, history.NodeLog, hash, ManifestRecord{level, children}.View());
        };
        persist(root, trie.Levels - 1, 0);
        return root;
    });
}

// Persist a pinned version's content and return its manifest root.
// The version's owned leaves and the manifest nodes above them join the content logs, and the subtrees it shares with live content persist from live data.
Hash128 PersistVersion(History &history, size_t track, const Version &v) {
    return With(history, TrackAt(history, track), [&](auto &owner) {
        owner.Settle();
        auto &trie = owner.Trie;
        const auto live_length = owner.Length();
        trie.ManifestRoot(live_length);
        const auto limit = trie.SlotsFor(v.S.Length), live_limit = trie.SlotsFor(live_length);
        const auto record = [&](uint32_t level, const ManifestChildren &children) {
            const ManifestRecord manifest{level, children};
            const auto hash = manifest.Hash();
            if (hash != Hash128{} && !history.NodeLog.Idx.contains(hash)) AppendContent(history, history.NodeLog, hash, manifest.View());
            return hash;
        };
        // The live slot's hash, which the version shares, or none beyond either length.
        const auto live_slot = [&](uint64_t slot) {
            if (slot >= limit || slot >= live_limit || slot >= trie.SlotHashes.size() || trie.SlotHashes[slot].State != LiveTrie::SlotState::Value) return Hash128{};
            const auto hash = trie.SlotHashes[slot].H;
            if (!history.LeafLog.Idx.contains(hash)) AppendContent(history, history.LeafLog, hash, owner.Read(slot));
            return hash;
        };
        // Indexed roots have indexed descendants.
        const auto persist_live = [&](this auto &&self, Hash128 hash, uint32_t level, uint64_t index) -> void {
            if (hash == Hash128{} || history.NodeLog.Idx.contains(hash)) return;
            const auto children = trie.ChildrenAt(level, index, live_length);
            for (uint64_t digit = 0; digit < Fanout; ++digit) {
                const auto child = children[digit];
                if (child == Hash128{}) continue;
                if (level) self(child, level - 1, index * Fanout + digit);
                else if (!history.LeafLog.Idx.contains(child)) AppendContent(history, history.LeafLog, child, owner.Read(index * Fanout + digit));
            }
            AppendContent(history, history.NodeLog, hash, ManifestRecord{level, children}.View());
        };
        // The manifest hash of the live slots under a trie level, bounded by the version's length.
        // A subtree within both lengths, or any subtree while the lengths agree, is the live manifest node.
        const auto live_subtree = [&](this auto &&self, uint32_t level, uint64_t base) -> Hash128 {
            if (base >= limit) return {};
            if (level == 0) return live_slot(base);
            const auto span = SlotSpan(level);
            if (limit == live_limit || (base + span <= limit && base + span <= live_limit)) {
                const auto &nodes = trie.Manifest[level - 1].Nodes;
                const auto hash = base / span < nodes.size() ? nodes[base / span].Hash : Hash128{};
                persist_live(hash, level - 1, base / span);
                return hash;
            }
            ManifestChildren children{};
            for (uint64_t digit = 0; digit < Fanout; ++digit) children[digit] = self(level - 1, base + digit * SlotSpan(level - 1));
            return record(level - 1, children);
        };
        const auto walk = [&](this auto &&self, const Node *n, uint32_t level, uint64_t base) -> Hash128 {
            if (base >= limit) return {};
            switch (n->Kind) {
                case NodeKind::Aliased: return live_subtree(level, base);
                case NodeKind::Absent: return {};
                case NodeKind::Owned: {
                    if (n->Default) return {};
                    if (!history.LeafLog.Idx.contains(n->Hash)) AppendContent(history, history.LeafLog, n->Hash, owner.Encode(n->Value));
                    return n->Hash;
                }
                case NodeKind::Interior: {
                    ManifestChildren children{};
                    for (uint64_t digit = 0; digit < Fanout; ++digit) children[digit] = self(n->Children[digit], level - 1, base + digit * SlotSpan(level - 1));
                    return record(level - 1, children);
                }
            }
            std::unreachable();
        };
        return walk(v.Root, trie.Levels, 0);
    });
}

void AppendTreeRecord(History &history, RecordKind kind, std::optional<uint32_t> parent, std::span<const std::byte> payload) {
    assert(history.Streams[FileTree].is_open() && "the tree log requires Begin or Open");
    // Tree record format: [u32 len][u8 kind][u8 has parent][u32 parent if present][payload].
    std::vector<std::byte> record;
    Put(record, uint32_t(2 + (parent ? sizeof(uint32_t) : 0) + payload.size()));
    Put(record, uint8_t(kind));
    Put(record, parent);
    record.insert(record.end(), payload.begin(), payload.end());
    Append(history, FileTree, record);
}

// Persist live content and return manifest roots, reusing parent roots for unchanged tracks.
std::vector<Hash128> PersistContent(History &history, const HistoryNode &n) {
    const auto *parent = n.Parent ? &history.Nodes[*n.Parent] : nullptr;
    std::vector<Hash128> roots;
    for (size_t i = 0; i < history.Tracks.size(); ++i) {
        if (parent && parent->Roots.size() == history.Tracks.size() && parent->Stamps[i] == n.Stamps[i]) roots.push_back(parent->Roots[i]);
        else roots.push_back(BuildManifest(history, i));
    }
    return roots;
}

void PersistNode(History &history, RecordKind kind, uint32_t node) {
    assert(history.Streams[FileTree].is_open() && "commit requires Begin or Open");
    auto &n = history.Nodes[node];
    n.Roots = PersistContent(history, n);
    AppendTreeRecord(history, kind, kind == RecordKind::Replace ? node : n.Parent, NodePayload(history, n));
}

std::expected<void, std::string> RestoreNode(History &history, uint32_t node) {
    if (history.Nodes[node].Hot) return history.Restore(*history.Nodes[node].Hot);
    const auto failed = [&](std::string error) -> std::expected<void, std::string> {
        CheckIO(history, std::move(error));
        return std::unexpected{history.TakeIntegrityError()};
    };
    if (const auto error = Flush(history); !error.empty()) return failed(error);
    LoadPlan plan;
    if (!PrepareLoad(history, node, plan)) return failed("cold load contains missing or corrupt records");
    const auto restored = ApplyLoad(history, node, plan);
    return restored ? restored : failed(restored.error());
}

// The nearest node at or above `node` that has state.
std::optional<uint32_t> StateOrigin(const History &history, uint32_t node) {
    std::optional<uint32_t> origin = node;
    while (origin && history.Nodes[*origin].Stamps.empty()) origin = history.Nodes[*origin].Parent;
    return origin;
}

// Replay each node below `origin` down to `node` from `origin`'s live state, or from the empty state when it is absent.
// Persist, pin and record the content of each replayed node.
std::expected<void, std::string> ReplayPath(History &history, std::optional<uint32_t> origin, uint32_t node) {
    std::vector<uint32_t> path;
    for (auto id = std::optional{node}; id != origin; id = history.Nodes[*id].Parent) path.push_back(*id);
    std::ranges::reverse(path);
    if (!origin) history.Callbacks.Reset(history.Dir);
    for (const auto id : path) {
        auto &n = history.Nodes[id];
        const auto replayed = history.Callbacks.Replay(history, history.ActionsOf(n), history.WorkspacesOf(n));
        if (!replayed) return replayed;
        n.Stamps = CurrentStamps(history);
        n.Roots = PersistContent(history, n);
        n.Hot = history.Pin();
        std::vector<std::byte> state;
        PutState(state, n.Stamps, n.Roots);
        AppendTreeRecord(history, RecordKind::State, id, state);
    }
    return {};
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
    const auto error = std::exchange(IntegrityError, {});
    return error.empty() ? StreamError(*this) : error;
}

std::expected<void, std::string> History::Restore(const Snapshot &s) {
    const auto restored = RunPipeline(*this, [&](size_t i) {
        if (!RestoreTrack(*this, i, s.Versions[i])) {
            AddIntegrityError(*this, Tracks[i].Name + ": restored state hash mismatch");
            std::println(stderr, "[history] {}: restored state hash mismatch", Tracks[i].Name);
            assert(false && "restored state hash does not match pinned hash");
        }
    });
    if (!restored) AddIntegrityError(*this, restored.error());
    return restored;
}

void History::Release(Snapshot &snapshot) {
    for (size_t i = 0; i < Tracks.size(); ++i) TrieOf(*this, Tracks[i]).Release(snapshot.Versions[i]);
    snapshot.Versions.clear();
}

std::vector<Hash128> History::RootsOf(Hash128 state, std::span<const size_t> tracks) const {
    if (const auto it = Kept.find(state); it != Kept.end()) {
        std::vector<Hash128> roots;
        for (const auto i : tracks) {
            const auto found = std::ranges::find(it->second.Tracks, uint32_t(i));
            if (found == it->second.Tracks.end()) return {};
            roots.push_back(it->second.Roots[size_t(found - it->second.Tracks.begin())]);
        }
        return roots;
    }
    std::vector<Stamp> stamps(tracks.size());
    for (const auto &n : Nodes) {
        if (n.Stamps.size() != Tracks.size() || n.Roots.size() != Tracks.size()) continue;
        for (size_t k = 0; k < tracks.size(); ++k) stamps[k] = n.Stamps[tracks[k]];
        if (StateHash(stamps) != state) continue;
        std::vector<Hash128> roots;
        for (const auto i : tracks) roots.push_back(n.Roots[i]);
        return roots;
    }
    return {};
}

namespace {
void RecordKept(History &history, Hash128 state, std::span<const size_t> tracks, std::vector<Hash128> roots) {
    KeptState kept;
    std::vector<std::byte> payload;
    Put(payload, state);
    Put(payload, uint32_t(tracks.size()));
    for (size_t k = 0; k < tracks.size(); ++k) {
        kept.Tracks.push_back(uint32_t(tracks[k]));
        Put(payload, uint32_t(tracks[k]));
        Put(payload, roots[k]);
    }
    kept.Roots = std::move(roots);
    AppendTreeRecord(history, RecordKind::Kept, std::nullopt, payload);
    history.Kept.emplace(state, std::move(kept));
}
} // namespace

void History::Keep(Hash128 state, std::span<const size_t> tracks) {
    if (Kept.contains(state)) return;
    if (auto roots = RootsOf(state, tracks); !roots.empty()) RecordKept(*this, state, tracks, std::move(roots));
}

void History::Keep(Hash128 state, std::span<const size_t> tracks, std::span<const Version> versions) {
    if (Kept.contains(state) || !Streams[FileTree].is_open()) return;
    std::vector<Hash128> roots;
    roots.reserve(tracks.size());
    for (size_t k = 0; k < tracks.size(); ++k) roots.push_back(PersistVersion(*this, tracks[k], versions[k]));
    RecordKept(*this, state, tracks, std::move(roots));
}

void History::StoreActions(HistoryNode &node, std::span<const std::byte> actions, std::span<const uint32_t> workspaces) {
    node.Actions = {ActionBytes.size(), actions.size()};
    node.Workspaces = {WorkspaceRefs.size(), workspaces.size()};
    ActionBytes.insert(ActionBytes.end(), actions.begin(), actions.end());
    WorkspaceRefs.insert(WorkspaceRefs.end(), workspaces.begin(), workspaces.end());
}
void History::CompactActions() {
    auto bytes = std::exchange(ActionBytes, {});
    auto refs = std::exchange(WorkspaceRefs, {});
    for (auto &node : Nodes) StoreActions(node, std::span{bytes}.subspan(node.Actions.First, node.Actions.Count), std::span{refs}.subspan(node.Workspaces.First, node.Workspaces.Count));
}

uint32_t History::CaptureWorkspace() {
    Workspace->Settle();
    const auto stamp = Workspace->Trie.CurrentStamp(Workspace->Length());
    if (!Workspaces.empty() && Workspaces.back().S == stamp) {
        if (!Workspaces.back().Hot.Root) Workspaces.back().Hot = Workspace->Pin();
        ++Workspaces.back().Refs;
        return uint32_t(Workspaces.size() - 1);
    }
    Workspaces.push_back({.S = stamp, .Hot = Workspace->Pin(), .Refs = 1});
    return uint32_t(Workspaces.size() - 1);
}
void History::RetainWorkspace(uint32_t id) { ++Workspaces.at(id).Refs; }
void History::ReleaseWorkspace(uint32_t id) {
    auto &v = Workspaces.at(id);
    if (--v.Refs == 0 && v.Hot.Root) Workspace->Trie.Release(v.Hot);
    while (Workspaces.size() > WrittenWorkspaces && !Workspaces.back().Refs) Workspaces.pop_back();
}
void History::PersistWorkspaces() {
    while (WrittenWorkspaces < Workspaces.size()) {
        auto &v = Workspaces[WrittenWorkspaces];
        if (v.Hot.Root) v.Root = PersistVersion(*this, Tracks.size(), v.Hot);
        std::vector<std::byte> bytes;
        Put(bytes, v.S);
        Put(bytes, v.Root);
        AppendTreeRecord(*this, RecordKind::Workspace, WrittenWorkspaces++, bytes);
    }
}
std::expected<void, std::string> History::RestoreWorkspace(uint32_t id) {
    if (!Workspace || id >= Workspaces.size()) return std::unexpected{"Missing action workspace"};
    auto &v = Workspaces[id];
    Workspace->Settle();
    if (v.Hot.Root) {
        if (!Workspace->Restore(v.Hot)) return std::unexpected{"Workspace restore mismatch"};
    } else {
        StateReader reader{*this};
        std::vector<std::pair<uint64_t, Hash128>> changes;
        std::unordered_map<Hash128, std::vector<std::byte>, Hash128Hasher> leaves;
        if (!PrepareTrackLoad(reader, Tracks.size(), v.S, v.Root, changes, leaves)) return std::unexpected{"Missing or corrupt workspace pages"};
        Workspace->Load(v.S.Length, changes, leaves);
        Workspace->Settle();
        if (Workspace->Trie.CurrentStamp(Workspace->Length()) != v.S) return std::unexpected{"Workspace load mismatch"};
        v.Hot = Workspace->Pin();
    }
    return {};
}

bool History::Begin(const std::filesystem::path &dir, std::vector<std::byte> actions, std::vector<uint32_t> workspaces) {
    assert(Dir.empty() || Dir != dir);
    const auto fail = [&] {
        for (auto id : workspaces) ReleaseWorkspace(id);
        return false;
    };
    if (!CheckIO(*this, Flush(*this))) return fail();
    History candidate{.Workspace = Workspace, .Tracks = Tracks, .PageTracks = PageTracks, .RecordTracks = RecordTracks, .PoolTracks = PoolTracks, .Order = Order, .Dir = dir};
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        CheckIO(*this, "cannot create " + dir.string() + ": " + ec.message());
        return fail();
    }
    if (!CheckIO(*this, OpenStreams(candidate.Streams, candidate.StreamBuffers, dir, true))) return fail();
    Append(candidate, FileTree, std::as_bytes(std::span{&TreeMagic, 1u}));
    std::unordered_map<uint32_t, uint32_t> remapped;
    for (auto &id : workspaces) {
        const auto original = id;
        auto [it, added] = remapped.try_emplace(id, uint32_t(candidate.Workspaces.size()));
        if (added) {
            auto v = Workspaces.at(id);
            if (!v.Hot.Root) throw std::logic_error("New history requires hot initial workspace");
            ++v.Hot.Root->Refs;
            v.Refs = 0;
            candidate.Workspaces.push_back(v);
        }
        id = it->second;
        candidate.RetainWorkspace(id);
        ReleaseWorkspace(original);
    }
    candidate.PersistWorkspaces();
    HistoryNode root{.Label = "Root", .Hot = Pin(), .Stamps = CurrentStamps(*this)};
    candidate.StoreActions(root, actions, workspaces);
    candidate.Nodes.push_back(std::move(root));
    SetPresent(candidate, 0);
    PersistNode(candidate, RecordKind::Root, 0);
    if (!AdoptProject(*this, candidate)) return false;
    SetPresent(*this, 0);
    return true;
}

bool History::Close() {
    const bool saved = CheckIO(*this, Flush(*this)) && CheckIO(*this, CloseStreams(*this));
    for (auto i = uint32_t{0}; i < Nodes.size(); ++i) ReleaseNode(*this, i);
    Nodes.clear();
    ActionBytes.clear();
    WorkspaceRefs.clear();
    for (auto &v : Workspaces)
        if (v.Hot.Root) Workspace->Trie.Release(v.Hot);
    Workspaces.clear();
    WrittenWorkspaces = 0;
    ++Revision;
    Present.reset();
    VisitCounter = 0;
    Dir.clear();
    LeafLog.Idx.clear();
    NodeLog.Idx.clear();
    LeafLog.Size = NodeLog.Size = 0;
    Kept.clear();
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

uint32_t History::Commit(std::string label, std::vector<std::byte> actions, std::vector<uint32_t> workspaces) {
    assert(Present);
    const auto stamps = CurrentStamps(*this);
    if (stamps == Nodes[*Present].Stamps) {
        for (auto id : workspaces) ReleaseWorkspace(id);
        Adopt(*this, *Nodes[*Present].Hot); // Release redundant copies for equal state.
        return *Present;
    }
    const auto adopt = [&](uint32_t node) {
        auto &n = Nodes[node];
        if (n.Hot) Adopt(*this, *n.Hot);
        else n.Hot = Pin();
        SetPresent(*this, node);
        for (auto id : workspaces) ReleaseWorkspace(id);
        AppendTreeRecord(*this, RecordKind::Navigate, node, {});
        return node;
    };
    for (const auto child : Nodes[*Present].Children) {
        if (Nodes[child].Stamps == stamps) return adopt(child);
    }
    if (const auto parent = Nodes[*Present].Parent; parent && Nodes[*parent].Stamps == stamps) return adopt(*parent);
    const auto id = uint32_t(Nodes.size());
    HistoryNode n{.Parent = Present, .Label = std::move(label), .Depth = Nodes[*Present].Depth + 1, .ReplayBaseline = false, .Hot = Pin(), .Stamps = stamps};
    StoreActions(n, actions, workspaces);
    Nodes.push_back(std::move(n));
    Nodes[*Present].Children.push_back(id);
    ++Revision;
    SetPresent(*this, id);
    PersistWorkspaces();
    PersistNode(*this, RecordKind::Action, id);
    return id;
}

uint32_t History::Replace(uint32_t node, std::vector<std::byte> actions, std::vector<uint32_t> workspaces) {
    assert(node < Nodes.size() && Nodes[node].Parent);
    DropDescendants(*this, node);
    auto &n = Nodes[node];
    if (n.Hot) Release(*n.Hot);
    for (auto id : WorkspacesOf(n)) ReleaseWorkspace(id);
    StoreActions(n, actions, workspaces);
    PersistWorkspaces();
    n.Stamps = CurrentStamps(*this);
    n.Hot = Pin();
    ++Revision;
    SetPresent(*this, node);
    CompactActions();
    PersistNode(*this, RecordKind::Replace, node);
    return node;
}

void History::Navigate(uint32_t node) {
    if (node == Present || node >= Nodes.size()) return;
    auto original = Pin();
    const auto origin = StateOrigin(*this, node);
    std::expected<void, std::string> restored;
    if (origin) {
        restored = RestoreNode(*this, *origin);
        if (restored && !Nodes[*origin].Hot) Nodes[*origin].Hot = Pin();
    }
    if (restored) restored = ReplayPath(*this, origin, node);
    if (restored) {
        SetPresent(*this, node);
        AppendTreeRecord(*this, RecordKind::Navigate, node, {});
    } else {
        AddIntegrityError(*this, restored.error());
        (void)Restore(original);
    }
    Release(original);
}

void History::Revert() {
    assert(Present && Nodes[*Present].Hot);
    (void)Restore(*Nodes[*Present].Hot);
}

bool History::Save() {
    if (Callbacks.BeforeSave) Callbacks.BeforeSave();
    if (Present) AppendTreeRecord(*this, RecordKind::Navigate, Present, {});
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
    const auto current = uint32_t(retained.size());
    auto &live = retained.emplace_back();
    live.Label = current ? "Current" : "Root";
    if (current) live.Parent = 0;
    live.Depth = current;
    live.Hot = Pin();
    live.Stamps = std::move(stamps);
    for (size_t i = 0; i < Tracks.size(); ++i) live.Roots.push_back(BuildManifest(*this, i));
    auto payload = NodePayload(*this, live);
    if (current) PutState(payload, saved->Stamps, saved->Roots);
    AppendTreeRecord(*this, RecordKind::Root, std::nullopt, payload);
    if (!CheckIO(*this, Flush(*this))) {
        Release(*live.Hot);
        return false;
    }
    for (const auto &n : Nodes)
        for (auto id : WorkspacesOf(n)) ReleaseWorkspace(id);
    for (auto i = uint32_t{0}; i < Nodes.size(); ++i) ReleaseNode(*this, i);
    Nodes = std::move(retained);
    ActionBytes.clear();
    WorkspaceRefs.clear();
    VisitCounter = 0;
    ++Revision;
    SetPresent(*this, current);
    return true;
}

void History::Evict(uint64_t cap) {
    if (Workspace && OwnedBytes(*this) > cap) {
        PersistWorkspaces();
        if (!CheckIO(*this, Flush(*this))) return;
        for (auto &v : Workspaces) {
            if (OwnedBytes(*this) <= cap) break;
            Workspace->Trie.Release(v.Hot);
        }
    }
    if (!Present || OwnedBytes(*this) <= cap) return;
    std::vector<uint32_t> protected_nodes;
    auto node = Present;
    for (auto k = uint32_t{0}; node && k <= UndoWindow; node = Nodes[*node].Parent, ++k) {
        protected_nodes.push_back(*node);
        if (Nodes[*node].ReplayBaseline) break;
    }
    std::vector<uint32_t> victims;
    for (auto i = uint32_t{0}; i < Nodes.size(); ++i) {
        if (!Nodes[i].Hot) continue;
        if (std::ranges::contains(protected_nodes, i)) continue;
        victims.push_back(i);
    }
    if (victims.empty() || !CheckIO(*this, Flush(*this))) return;
    std::vector<uint32_t> distance(Nodes.size());
    for (const auto v : victims) distance[v] = TreeDistance(*this, v, *Present);
    // Evict farthest nodes first, breaking ties by least recent visit.
    std::ranges::sort(victims, [&](auto a, auto b) {
        if (distance[a] != distance[b]) return distance[a] > distance[b];
        return Nodes[a].LastVisited < Nodes[b].LastVisited;
    });
    for (const auto victim : victims) {
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

std::string History::Replay() {
    if (!Present) return "no present node";
    auto original = Pin();
    std::vector<uint32_t> path;
    auto baseline = *Present;
    while (!Nodes[baseline].ReplayBaseline) {
        path.push_back(baseline);
        baseline = *Nodes[baseline].Parent;
    }
    std::string diff;
    if (const auto restored = RestoreNode(*this, baseline); !restored) diff = restored.error();
    for (auto it = path.rbegin(); diff.empty() && it != path.rend(); ++it) diff = ReplayStep(*this, Nodes[*it]);
    if (diff.empty()) {
        if (Nodes[*Present].Hot) Adopt(*this, *Nodes[*Present].Hot);
        else Nodes[*Present].Hot = Pin();
        SetPresent(*this, *Present);
    } else {
        (void)Restore(original);
    }
    Release(original);
    return diff;
}

std::string History::ValidateReplay() {
    if (!Present) return "no present node";
    if (Nodes[*Present].ReplayBaseline) return {};
    assert(Nodes[*Present].Hot);
    const auto &n = Nodes[*Present];
    std::string diff;
    if (const auto restored = RestoreNode(*this, *n.Parent); !restored) diff = restored.error();
    if (diff.empty()) diff = ReplayStep(*this, n);
    (void)Restore(*n.Hot);
    return diff;
}

HistoryStats History::Stats() const {
    HistoryStats s{.OwnedBytes = OwnedBytes(*this), .SharedNodeBytes = SharedNodeBytes()};
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

bool History::Audit(std::string &why) {
    if (!Check(why) || !CheckHashes(*this, why)) return false;
    if (Workspace) {
        std::vector<Version> versions;
        for (const auto &v : Workspaces)
            if (v.Hot.Root) versions.push_back(v.Hot);
        if (!Workspace->Trie.Check(why, versions) || !CheckTrackHashes(*this, Tracks.size(), why)) return false;
    }
    return true;
}

std::optional<uint32_t> History::FindPosition(const HistoryPosition &position) const {
    const auto matches = [&](const HistoryNode &node) { return node.Stamps == position.Stamps && node.Roots == position.Roots; };
    if (position.Node && *position.Node < Nodes.size() && matches(Nodes[*position.Node])) return position.Node;
    const auto it = std::ranges::find_if(Nodes, matches);
    if (it == Nodes.end()) return std::nullopt;
    return uint32_t(it - Nodes.begin());
}

bool History::Open(const std::filesystem::path &dir, const HistoryPosition *position) {
    if (!CheckIO(*this, Flush(*this))) return false;
    History candidate{.Workspace = Workspace, .Tracks = Tracks, .PageTracks = PageTracks, .RecordTracks = RecordTracks, .PoolTracks = PoolTracks, .Order = Order, .Callbacks = Callbacks};
    // Reconstruct states by replay when content logs are absent.
    std::error_code ec;
    const bool pages = std::filesystem::exists(dir / TreeLogName, ec);
    uint64_t tree_size{};
    if (pages ? !ReadProject(candidate, dir, tree_size) : !ReadActions(candidate, dir)) {
        return CheckIO(*this, "cannot open project: invalid format or missing or corrupt records");
    }
    if (!pages && candidate.Workspace) {
        if (!ScanContentLog(candidate, candidate.LeafLog) || !ScanContentLog(candidate, candidate.NodeLog)) return false;
    }
    if (position) candidate.Present = candidate.FindPosition(*position);
    // The present node replays from its nearest ancestor with state, which loads from the content logs.
    const auto origin = candidate.Present ? StateOrigin(candidate, *candidate.Present) : std::nullopt;
    LoadPlan plan;
    if (!candidate.Present || (origin && !PrepareLoad(candidate, *origin, plan))) {
        return CheckIO(*this, "cannot restore project position: missing or corrupt records");
    }
    if (pages) {
        // Truncate incomplete records after validating the target state.
        for (const auto &[name, size] : {std::pair{LeafLog.Name, candidate.LeafLog.Size}, {NodeLog.Name, candidate.NodeLog.Size}, {TreeLogName, tree_size}}) {
            std::filesystem::resize_file(dir / name, size, ec);
            if (ec) return CheckIO(*this, "cannot truncate " + (dir / name).string() + ": " + ec.message());
        }
    }
    if (!CheckIO(*this, OpenStreams(candidate.Streams, candidate.StreamBuffers, dir, false))) return false;
    if (!pages) {
        // The actions log becomes a tree log of nodes without state.
        Append(candidate, FileTree, std::as_bytes(std::span{&TreeMagic, 1u}));
        candidate.PersistWorkspaces();
        for (auto i = uint32_t{0}; i < candidate.Nodes.size(); ++i) AppendTreeRecord(candidate, i ? RecordKind::Action : RecordKind::Root, candidate.Nodes[i].Parent, NodePayload(candidate, candidate.Nodes[i]));
        AppendTreeRecord(candidate, RecordKind::Navigate, candidate.Present, {});
    }
    auto original = Pin();
    std::expected<void, std::string> restored;
    if (origin) {
        restored = ApplyLoad(candidate, *origin, plan);
        if (restored) candidate.Nodes[*origin].Hot = candidate.Pin();
    }
    if (restored) restored = ReplayPath(candidate, origin, *candidate.Present);
    if (restored) SetPresent(candidate, *candidate.Present);
    const bool adopted = restored && AdoptProject(*this, candidate);
    if (!adopted) {
        if (!restored) AddIntegrityError(*this, restored.error());
        if (Callbacks.Reset) Callbacks.Reset(Dir);
        (void)Restore(original);
    }
    Release(original);
    if (!adopted) return false;
    if (!pages) std::filesystem::remove(dir / ActionsLogName, ec);
    return true;
}

Hash128 StateHash(std::span<const Stamp> stamps) { return HashBytes(std::as_bytes(stamps)); }

StateReader::StateReader(History &history) : Hist(history) {
    if (Hist.Dir.empty() || !CheckIO(Hist, Flush(Hist))) return;
    Leaves.open(Hist.Dir / Hist.LeafLog.Name, std::ios::binary);
    Nodes.open(Hist.Dir / Hist.NodeLog.Name, std::ios::binary);
}

std::optional<std::vector<std::byte>> StateReader::Read(size_t track, Hash128 root, uint64_t slot) {
    const auto &trie = TrieOf(Hist, TrackAt(Hist, track));
    auto hash = root;
    for (auto level = trie.Levels; level-- > 0 && hash != Hash128{};) {
        uint8_t recorded;
        ManifestChildren children;
        if (!ReadManifestRecord(Hist, Nodes, hash, recorded, children) || recorded != level) return std::nullopt;
        hash = children[(slot / SlotSpan(level)) % Fanout];
    }
    std::vector<std::byte> bytes;
    if (hash == Hash128{}) return bytes;
    const auto it = Hist.LeafLog.Idx.find(hash);
    if (it == Hist.LeafLog.Idx.end() || !ReadExtent(Leaves, it->second, hash, bytes)) return std::nullopt;
    return bytes;
}

std::optional<std::vector<uint64_t>> StateReader::DifferingSlots(size_t track, Hash128 root) {
    std::vector<uint64_t> slots;
    const auto &trie = TrieOf(Hist, TrackAt(Hist, track));
    if (!DiffManifest(Hist, Nodes, track, root, SlotSpan(trie.Levels), [&](uint64_t slot, Hash128, Hash128) {
            slots.push_back(slot);
            return true;
        })) return std::nullopt;
    return slots;
}

std::expected<std::vector<uint32_t>, std::string> History::WriteActions(const std::filesystem::path &path) const {
    if (Nodes.empty() || Nodes[0].Actions.empty()) return std::unexpected{"the history starts from a cleared state that no recorded actions rebuild"};
    std::vector<uint32_t> kept{0};
    for (size_t i = 0; i < kept.size(); ++i) kept.append_range(Nodes[kept[i]].Children);
    std::ranges::sort(kept);
    std::vector<std::optional<uint32_t>> ids(Nodes.size());
    for (auto i = uint32_t{0}; i < kept.size(); ++i) ids[kept[i]] = i;
    if (!Present || !ids[*Present]) return std::unexpected{"the current history node is not reachable"};
    std::vector<std::byte> out;
    Put(out, ActionsMagic);
    Put(out, uint32_t(kept.size()));
    Put(out, *ids[*Present]);
    std::vector<uint32_t> workspace_ids(Workspaces.size(), UINT32_MAX), used;
    for (auto id : kept)
        for (auto w : WorkspacesOf(Nodes[id])) {
            if (workspace_ids[w] != UINT32_MAX) continue;
            workspace_ids[w] = uint32_t(used.size());
            used.push_back(w);
        }
    Put(out, uint32_t(used.size()));
    for (auto w : used) {
        Put(out, Workspaces[w].S);
        Put(out, Workspaces[w].Root);
    }
    for (const auto id : kept) {
        Put(out, Nodes[id].Parent ? ids[*Nodes[id].Parent] : std::nullopt);
        PutActions(*this, out, Nodes[id], workspace_ids);
    }
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    file.write(reinterpret_cast<const char *>(out.data()), std::streamsize(out.size()));
    file.close();
    if (!file) return std::unexpected{"cannot write " + path.string()};
    if (Workspace) {
        if (path.parent_path() == Dir) return std::unexpected{"Action export needs a separate directory"};
        std::ifstream nodes{Dir / NodeLog.Name, std::ios::binary}, leaves{Dir / LeafLog.Name, std::ios::binary};
        std::ofstream nodes_out{path.parent_path() / NodeLog.Name, std::ios::binary}, leaves_out{path.parent_path() / LeafLog.Name, std::ios::binary};
        std::array<std::unordered_set<Hash128, Hash128Hasher>, 2> copied;
        const auto copy = [&](this auto &&self, Hash128 hash, bool node) -> bool {
            if (hash == Hash128{} || !copied[node].insert(hash).second) return true;
            const auto &log = node ? NodeLog : LeafLog;
            const auto found = log.Idx.find(hash);
            std::vector<std::byte> bytes;
            if (found == log.Idx.end() || !ReadExtent(node ? nodes : leaves, found->second, hash, bytes)) return false;
            if (node) {
                uint8_t level;
                ManifestChildren children;
                if (!ReadManifestRecord(*this, nodes, hash, level, children)) return false;
                for (auto child : children)
                    if (!self(child, level != 0)) return false;
            }
            auto &output = node ? nodes_out : leaves_out;
            const auto size = uint32_t(bytes.size());
            output.write(reinterpret_cast<const char *>(&hash), sizeof(hash));
            output.write(reinterpret_cast<const char *>(&size), sizeof(size));
            output.write(reinterpret_cast<const char *>(bytes.data()), size);
            return bool(output);
        };
        for (auto w : used)
            if (!copy(Workspaces[w].Root, true)) return std::unexpected{"Cannot export workspace pages"};
        nodes_out.close();
        leaves_out.close();
        if (!nodes_out || !leaves_out) return std::unexpected{"Cannot write workspace pages"};
    }
    return kept;
}
} // namespace store
