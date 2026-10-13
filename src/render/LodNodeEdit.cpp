#include "render/LodNodeEdit.h"
#include "Profile.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "render/ClusterLod.h"
#include "state/Scene.h"
#include <map>
#include <numeric>
#include <optional>
#include <set>

namespace {
constexpr uint32_t MaxLeafRecords{2u * ClusterLodSpanLeafRecords}, MaxNodeChildren{2u * ClusterLodSpanNodeWidth};

// Divides `count` items into the fewest runs of at most `width`, whose lengths differ by at most one.
std::vector<Range> EvenRuns(uint32_t count, uint32_t width) {
    const uint64_t run_count = (count + width - 1u) / width;
    std::vector<Range> runs;
    runs.reserve(run_count);
    for (uint64_t i = 0u; i < run_count; ++i) runs.push_back({uint32_t(count * i / run_count), uint32_t(count * (i + 1u) / run_count - count * i / run_count)});
    return runs;
}

// The owner's traversal nodes, their parents and their leaves' member runs, read and written by handle.
// A leaf lists its clusters in ascending handle order.
struct OwnerNodes {
    RenderArenas &Buffers;
    MeshStore::Record &Owner;

    ElementAddress Address(uint32_t node) const { return Buffers.LodNodes.Address(Owner.NodeData, ElementHandle{node}); }
    const LodNode &Get(uint32_t node) const { return Buffers.LodNodes.Get({Address(node).Value(), 1u})[0]; }
    LodNode &Edit(uint32_t node) { return Buffers.LodNodes.GetMutable({Address(node).Value(), 1u})[0]; }
    uint32_t Parent(uint32_t node) const { return Buffers.LodParents.Get({Address(node).Value(), 1u})[0]; }
    void SetParent(uint32_t node, uint32_t parent) { Buffers.LodParents.GetMutable({Address(node).Value(), 1u})[0] = parent; }
    // The leaf's members at indices [first, end).
    std::vector<uint32_t> Members(const LodNode &leaf, uint32_t first = 0u, uint32_t end = InvalidOffset) const {
        std::vector<uint32_t> members;
        if (leaf.Members == InvalidOffset) return members;
        end = std::min(end, leaf.MeshletCount);
        const auto entries = Buffers.LodMembers.Entries(Owner.MemberData);
        const auto values = Buffers.LodMembers.Buffer.GetSpan<uint32_t>();
        members.reserve(end > first ? end - first : 0u);
        for (uint32_t i = first; i < end; ++i) members.push_back(values[Resolve<RenderBlockSize>(entries, ElementHandle{leaf.Members + i}).Value()]);
        return members;
    }
    // The index of the leaf's first member not below the cluster, with that member, or InvalidOffset past the last member.
    std::pair<uint32_t, uint32_t> LowerBound(const LodNode &leaf, uint32_t cluster) const {
        if (leaf.Members == InvalidOffset) return {0u, InvalidOffset};
        const auto entries = Buffers.LodMembers.Entries(Owner.MemberData);
        const auto values = Buffers.LodMembers.Buffer.GetSpan<uint32_t>();
        const auto member = [&](uint32_t i) { return values[Resolve<RenderBlockSize>(entries, ElementHandle{leaf.Members + i}).Value()]; };
        uint32_t lo = 0u, hi = leaf.MeshletCount;
        while (lo < hi) {
            const auto mid = (lo + hi) / 2u;
            if (member(mid) < cluster) lo = mid + 1u;
            else hi = mid;
        }
        return {lo, lo < leaf.MeshletCount ? member(lo) : InvalidOffset};
    }
    // The headroom members after the leaf's run, counted up to `limit`.
    uint32_t Headroom(const LodNode &leaf, uint32_t limit = InvalidOffset) const {
        if (leaf.Members == InvalidOffset) return 0u;
        const auto &members = Buffers.LodMembers;
        const auto entries = members.Entries(Owner.MemberData);
        const auto values = members.Buffer.GetSpan<uint32_t>();
        uint32_t headroom = 0u;
        for (ElementHandle h{leaf.Members + leaf.MeshletCount}; headroom < limit; ++h.Value, ++headroom) {
            if (!members.Contains(entries, h) || values[Resolve<RenderBlockSize>(entries, h).Value()] != InvalidOffset) break;
        }
        return headroom;
    }
    // Releases the leaf's run and its headroom.
    void Unlist(const LodNode &leaf) {
        if (leaf.Members == InvalidOffset) return;
        std::vector<uint32_t> handles(leaf.MeshletCount + Headroom(leaf));
        std::iota(handles.begin(), handles.end(), leaf.Members);
        Buffers.LodMembers.Erase(Owner.MemberData, handles);
    }
};

// A node value to place, with the node it moves from, or InvalidOffset for a new node.
struct NodeEntry {
    LodNode Value;
    uint32_t Former{InvalidOffset};
};

// Places span-tree nodes in new runs of handles, and each placed node's children or members point back at it.
struct SpanPlacement {
    OwnerNodes &Nodes;
    std::vector<uint32_t> Placed, Formers;
    // Each placed leaf's members with their new leaf, written together once every node is placed.
    std::vector<std::pair<uint32_t, uint32_t>> Leaves;

    // Writes the entries to one new run under `parent`, and returns the run's handles.
    Range Place(std::span<const NodeEntry> entries, uint32_t parent) {
        auto &buffers = Nodes.Buffers;
        const auto run = buffers.Insert(RenderSet::Nodes, Nodes.Owner.NodeData, uint32_t(entries.size()));
        for (uint32_t i = 0u; i < entries.size(); ++i) {
            const auto id = run.Handle.Value + i;
            const auto &[value, former] = entries[i];
            buffers.LodNodes.GetMutable({(run.Address + i).Value(), 1u})[0] = value;
            buffers.LodParents.GetMutable({(run.Address + i).Value(), 1u})[0] = parent;
            if (value.ChildCount) {
                for (uint32_t c = 0u; c < value.ChildCount; ++c) Nodes.SetParent(value.ChildOffset + c, id);
            } else {
                for (const auto cluster : Nodes.Members(value)) Leaves.emplace_back(cluster, id);
            }
            Placed.push_back(id);
            if (former != InvalidOffset) Formers.push_back(former);
        }
        return {run.Handle.Value, run.Count};
    }

    // Gives `node` the entries as its children.
    // A node past twice the build's width keeps its first even run and returns the others as siblings its parent takes after it.
    // A root instead adds levels below itself until it holds every entry.
    std::vector<LodNode> Adopt(uint32_t node, std::vector<NodeEntry> entries) {
        if (entries.size() > MaxNodeChildren && Nodes.Parent(node) != InvalidOffset) {
            const auto range = Place(entries, node);
            std::vector<LodNode> siblings;
            for (const auto run : EvenRuns(uint32_t(entries.size()), ClusterLodSpanNodeWidth)) {
                siblings.push_back({.ChildOffset = range.Offset + run.Offset, .ChildCount = run.Count});
            }
            auto &value = Nodes.Edit(node);
            value.ChildOffset = siblings.front().ChildOffset;
            value.ChildCount = siblings.front().ChildCount;
            siblings.erase(siblings.begin());
            return siblings;
        }
        while (entries.size() > MaxNodeChildren) {
            const auto range = Place(entries, InvalidOffset);
            std::vector<NodeEntry> level;
            for (const auto run : EvenRuns(uint32_t(entries.size()), ClusterLodSpanNodeWidth)) {
                level.push_back({.Value = {.ChildOffset = range.Offset + run.Offset, .ChildCount = run.Count}});
            }
            entries = std::move(level);
        }
        const auto range = Place(entries, node);
        auto &value = Nodes.Edit(node);
        value.ChildOffset = range.Offset;
        value.ChildCount = uint32_t(entries.size());
        value.Members = InvalidOffset;
        return {};
    }

    // Writes the members' leaves and releases the moved nodes' former handles.
    // The refit covers every placed node in place of the handle it moved from.
    void Commit(std::set<uint32_t> &affected) {
        auto &buffers = Nodes.Buffers;
        const auto meshlets = buffers.Meshlets.Entries(Nodes.Owner.MeshletData);
        std::vector<uint32_t> addresses;
        addresses.reserve(Leaves.size());
        for (const auto &[cluster, leaf] : Leaves) addresses.push_back(Resolve<RenderBlockSize>(meshlets, ElementHandle{cluster}).Value());
        buffers.MeshletLodLeaves.Buffer.CaptureWriteElements(addresses, sizeof(uint32_t));
        auto *leaves = reinterpret_cast<uint32_t *>(buffers.MeshletLodLeaves.Buffer.Contents().data());
        for (uint32_t i = 0u; i < Leaves.size(); ++i) leaves[addresses[i]] = Leaves[i].second;
        std::ranges::sort(Formers);
        buffers.LodNodes.Erase(Nodes.Owner.NodeData, Formers);
        for (const auto former : Formers) affected.erase(former);
        affected.insert(Placed.begin(), Placed.end());
    }
};

// Splits each overfull leaf's final members into even runs of at most one leaf span in cluster order, and removes each emptied leaf from its parent's run.
// An ancestor that outgrows twice the node width splits the same way, and one left without children leaves its parent in turn.
// A primitive's root stays, and a root left without children becomes an empty leaf.
void RestructureLeaves(OwnerNodes &nodes, const std::map<uint32_t, std::vector<uint32_t>> &overfull, std::span<const uint32_t> emptied, std::set<uint32_t> &affected) {
    const profile::CpuScope scope{"LodNodeSplit"};
    SpanPlacement placement{nodes};
    // Each parent's edited children, which add siblings right after themselves or, with none, leave.
    std::map<uint32_t, std::map<uint32_t, std::optional<std::vector<LodNode>>>> changes;
    for (const auto &[leaf, all] : overfull) {
        nodes.Unlist(nodes.Get(leaf));
        std::vector<NodeEntry> pieces;
        for (const auto run : EvenRuns(uint32_t(all.size()), ClusterLodSpanLeafRecords)) {
            pieces.push_back({.Value = {.MeshletCount = run.Count, .Members = nodes.Buffers.ListMembers(nodes.Owner.MemberData, std::span{all}.subspan(run.Offset, run.Count))}});
        }
        auto &value = nodes.Edit(leaf);
        value.Members = pieces.front().Value.Members;
        value.MeshletCount = pieces.front().Value.MeshletCount;
        const auto parent = nodes.Parent(leaf);
        // A root leaf becomes the parent of its runs.
        if (parent == InvalidOffset) {
            placement.Adopt(leaf, std::move(pieces));
            continue;
        }
        auto &siblings = changes[parent][leaf].emplace();
        for (uint32_t r = 1u; r < pieces.size(); ++r) siblings.push_back(pieces[r].Value);
    }
    for (const auto leaf : emptied) changes[nodes.Parent(leaf)][leaf];
    while (!changes.empty()) {
        decltype(changes) next;
        for (const auto &[node, children] : changes) {
            const auto value = nodes.Get(node);
            std::vector<NodeEntry> entries;
            for (uint32_t c = 0u; c < value.ChildCount; ++c) {
                const auto child = value.ChildOffset + c;
                const auto found = children.find(child);
                if (found != children.end() && !found->second) {
                    placement.Formers.push_back(child);
                    continue;
                }
                entries.push_back({.Value = nodes.Get(child), .Former = child});
                if (found != children.end()) {
                    for (const auto &sibling : *found->second) entries.push_back({.Value = sibling});
                }
            }
            const auto parent = nodes.Parent(node);
            if (entries.empty() && parent != InvalidOffset) next[parent][node];
            else if (entries.empty()) nodes.Edit(node) = {.Members = InvalidOffset};
            else if (auto siblings = placement.Adopt(node, std::move(entries)); !siblings.empty()) next[parent][node].emplace(std::move(siblings));
        }
        changes = std::move(next);
    }
    placement.Commit(affected);
}
} // namespace

LodNodeRefit EditLodNodes(state::Scene &r, mtl::ComputeChain &chain, MeshStore::Record &owner, std::span<const uint32_t> removed, std::span<const LodClusterRun> added, std::span<const uint32_t> touched) {
    if (removed.empty() && added.empty() && touched.empty()) return {};
    const profile::CpuScope scope{"LodNodeEdit"};
    auto &buffers = r.Context.get<MeshStore>().Render();
    if (!owner.MeshletData || !owner.NodeData) throw std::invalid_argument("LOD membership edit requires a live render owner.");
    OwnerNodes nodes{buffers, owner};
    // The visits insert nothing, so the owner's meshlet and node tables stay in place while they read them.
    const auto meshlet_table = buffers.Meshlets.Entries(owner.MeshletData), node_table = buffers.LodNodes.Entries(owner.NodeData);
    // A traversal leaf that holds the cluster exactly when it is not being added.
    const auto holds = [&](uint32_t node, uint32_t cluster, bool adding) {
        if (!buffers.LodNodes.Contains(node_table, ElementHandle{node})) return false;
        const auto &value = buffers.LodNodes.Get({Resolve<RenderBlockSize>(node_table, ElementHandle{node}).Value(), 1u})[0];
        return !value.ChildCount && (nodes.LowerBound(value, cluster).second == cluster) != adding;
    };
    // The clusters each edited leaf gains and loses.
    std::map<uint32_t, std::array<std::vector<uint32_t>, 2>> leaves;
    std::set<uint32_t> affected, primitives, roots, pinned;
    // Consecutive clusters of one primitive record its nodes once, and consecutive clusters of one leaf record its path once.
    uint32_t last_primitive = InvalidOffset, last_leaf = InvalidOffset;
    PrimitiveRecord primitive{};
    // Side zero gains the cluster and side one loses it.
    const auto visit = [&](uint32_t cluster, uint32_t side, bool edited, uint32_t primitive_id, bool finest) {
        const bool adding = edited && side == 0u;
        if (!buffers.Meshlets.Contains(meshlet_table, ElementHandle{cluster})) throw std::invalid_argument("LOD membership edit references a foreign cluster.");
        if (primitive_id != last_primitive) {
            if (!buffers.Primitives.Contains(owner.PrimitiveData, ElementHandle{primitive_id})) throw std::invalid_argument("LOD membership edit references a foreign primitive.");
            primitive = buffers.Primitives.Get({buffers.Primitives.Address(owner.PrimitiveData, ElementHandle{primitive_id}).Value(), 1u})[0];
            primitives.insert(primitive_id);
            roots.insert(primitive.LodRootNode);
            pinned.insert(primitive.LodFinestNode);
            affected.insert(primitive.LodFinestNode);
            last_primitive = primitive_id;
            last_leaf = InvalidOffset;
        }
        const auto leaf = buffers.MeshletLodLeaves.Get({Resolve<RenderBlockSize>(meshlet_table, ElementHandle{cluster}).Value(), 1u})[0];
        if (!holds(leaf, cluster, adding)) throw std::invalid_argument("LOD membership edit disagrees with its traversal leaf.");
        if (edited) leaves[leaf][side].push_back(cluster);
        if (finest) {
            if (!holds(primitive.LodFinestNode, cluster, adding)) throw std::invalid_argument("LOD membership edit disagrees with its finest node.");
            if (edited && primitive.LodFinestNode != leaf) leaves[primitive.LodFinestNode][side].push_back(cluster);
        }
        if (leaf == last_leaf) return;
        last_leaf = leaf;
        // A node already affected had its whole path to the root validated and recorded.
        for (uint32_t node = leaf, depth = 0u; affected.insert(node).second; ++depth) {
            const auto parent = nodes.Parent(node);
            if (parent == InvalidOffset) {
                if (node != primitive.LodRootNode) throw std::invalid_argument("LOD membership path does not reach its primitive root.");
                break;
            }
            if (!buffers.LodNodes.Contains(owner.NodeData, ElementHandle{parent}) || depth > 32u) throw std::invalid_argument("LOD membership path has an invalid parent.");
            const auto &value = nodes.Get(parent);
            if (node < value.ChildOffset || uint64_t(node) >= uint64_t(value.ChildOffset) + value.ChildCount) {
                throw std::invalid_argument("LOD membership parent does not contain its child.");
            }
            node = parent;
        }
    };
    // Removed and touched clusters have published records.
    const auto visit_recorded = [&](uint32_t cluster, uint32_t side, bool edited) {
        if (!buffers.Meshlets.Contains(meshlet_table, ElementHandle{cluster})) throw std::invalid_argument("LOD membership edit references a foreign cluster.");
        const auto &record = buffers.Meshlets.Get({Resolve<RenderBlockSize>(meshlet_table, ElementHandle{cluster}).Value(), 1u})[0];
        visit(cluster, side, edited, record.Primitive, record.RefinedGroup == InvalidOffset);
    };
    for (const auto cluster : removed) visit_recorded(cluster, 1u, true);
    for (const auto &run : added) {
        for (uint32_t i = 0u; i < run.Count; ++i) visit(run.First + i, 0u, true, run.Primitive, run.Finest);
    }
    for (const auto cluster : touched) visit_recorded(cluster, 0u, false);

    std::map<uint32_t, std::vector<uint32_t>> overfull;
    std::vector<uint32_t> emptied;
    for (auto &[node, members] : leaves) {
        const auto value = nodes.Get(node);
        auto &[gains, losses] = members;
        std::ranges::sort(gains);
        std::ranges::sort(losses);
        // The members before the first loss and before the first gain's place stay.
        const auto first = std::min(losses.empty() ? value.MeshletCount : nodes.LowerBound(value, losses.front()).first, gains.empty() ? value.MeshletCount : nodes.LowerBound(value, gains.front()).first);
        auto tail = nodes.Members(value, first);
        std::erase_if(tail, [&](uint32_t cluster) { return std::ranges::binary_search(losses, cluster); });
        const auto kept = std::ssize(tail);
        tail.insert(tail.end(), gains.begin(), gains.end());
        std::inplace_merge(tail.begin(), tail.begin() + kept, tail.end());
        if (std::ranges::adjacent_find(tail) != tail.end() || first + tail.size() + losses.size() != value.MeshletCount + gains.size()) throw std::logic_error("LOD membership edit repeats cluster identities.");
        const auto count = first + uint32_t(tail.size());
        // An overfull leaf takes its final members directly as even runs.
        if (count > MaxLeafRecords && !pinned.contains(node)) {
            auto all = nodes.Members(value, 0u, first);
            all.append_range(tail);
            overfull.emplace(node, std::move(all));
            continue;
        }
        if (!count && !pinned.contains(node) && !roots.contains(node)) emptied.push_back(node);
        // The run grows or shrinks in place when its headroom holds the members, writing only from `first`, and the members it vacates become headroom.
        auto updated = value;
        if (value.Members != InvalidOffset && count && (count <= value.MeshletCount || nodes.Headroom(value, count - value.MeshletCount) == count - value.MeshletCount)) {
            const auto entries = buffers.LodMembers.Entries(owner.MemberData);
            const auto end = std::max(count, value.MeshletCount);
            for (uint32_t i = first; i < end;) {
                const auto handle = value.Members + i;
                const auto run = std::min(end - i, RenderBlockSize - handle % RenderBlockSize);
                const auto values = buffers.LodMembers.GetMutable({Resolve<RenderBlockSize>(entries, ElementHandle{handle}).Value(), run});
                for (uint32_t j = 0u; j < run; ++j) values[j] = i + j < count ? tail[i + j - first] : InvalidOffset;
                i += run;
            }
        } else {
            auto all = nodes.Members(value, 0u, first);
            all.append_range(tail);
            nodes.Unlist(value);
            updated.Members = buffers.ListMembers(owner.MemberData, all);
        }
        updated.MeshletCount = count;
        nodes.Edit(node) = updated;
    }
    if (!overfull.empty() || !emptied.empty()) RestructureLeaves(nodes, overfull, emptied, affected);

    // Deeper nodes refit first, since their parents read their bounds and counts.
    std::map<uint32_t, std::vector<uint32_t>, std::greater<>> levels;
    uint32_t tree_depth = 0u;
    for (const auto node : affected) {
        uint32_t depth = 0u;
        for (auto id = node; nodes.Parent(id) != InvalidOffset && depth <= 32u; id = nodes.Parent(id)) ++depth;
        if (depth > 32u || (depth ? pinned.contains(node) : !roots.contains(node) && !pinned.contains(node))) {
            throw std::invalid_argument("LOD refit has a detached node or a parented pinned node.");
        }
        levels[depth].push_back(node);
        tree_depth = std::max(tree_depth, depth);
    }
    // A root that split adds a level the traversal descends.
    owner.LodDepth = std::max(owner.LodDepth, tree_depth);
    std::vector<uint32_t> jobs;
    std::vector<std::pair<uint32_t, Range>> batches;
    for (const auto &[depth, level] : levels) {
        const auto first = uint32_t(jobs.size());
        for (const auto id : level) {
            auto node = nodes.Get(id);
            uint64_t count = 0u;
            if (node.ChildCount) {
                for (uint32_t c = 0u; c < node.ChildCount; ++c) count += nodes.Get(node.ChildOffset + c).MeshletCount;
            } else {
                count = node.MeshletCount;
            }
            if (count > UINT32_MAX) throw std::length_error("LOD node membership exceeds its count.");
            node.MeshletCount = uint32_t(count);
            // A pinned finest node keeps every finest member at an infinite error, and an empty node encloses nothing.
            if (pinned.contains(id)) node.Error = std::numeric_limits<float>::infinity();
            else if (!count) node = {.FirstMeshlet = node.FirstMeshlet, .ChildOffset = node.ChildOffset, .ChildCount = node.ChildCount, .Members = node.Members};
            else jobs.push_back(id);
            nodes.Edit(id) = node;
        }
        if (jobs.size() > first) batches.push_back({depth, {first, uint32_t(jobs.size()) - first}});
    }
    for (const auto id : primitives) {
        auto &value = buffers.Primitives.GetMutable({buffers.Primitives.Address(owner.PrimitiveData, ElementHandle{id}).Value(), 1u})[0];
        value.MeshletCount = nodes.Get(value.LodRootNode).MeshletCount;
        value.Level0Count = nodes.Get(value.LodFinestNode).MeshletCount;
    }
    if (jobs.empty()) return {};
    const auto job_words = chain.Scratch.Allocate(std::span<const uint32_t>{jobs});
    for (auto &[depth, batch] : batches) batch.Offset += job_words.Offset;
    const LodNodeRefitPushConstants pc{
        .Jobs = {chain.Scratch.Buffer.Slot, job_words.Offset},
        .Nodes = buffers.LodNodes.Slots(),
        .Meshlets = buffers.Meshlets.Slots(),
        .Groups = buffers.ClusterGroups.Slots(),
        .Members = buffers.LodMembers.Slots(),
        .NodeSet = owner.NodeData.Index,
        .MeshletSet = owner.MeshletData.Index,
        .GroupSet = owner.GroupData.Index,
        .MemberSet = owner.MemberData.Index,
        .NodeSlot = buffers.LodNodes.Buffer.Slot,
        .ParentSlot = buffers.LodParents.Buffer.Slot,
        .MeshletSlot = buffers.Meshlets.Buffer.Slot,
        .GroupSlot = buffers.ClusterGroups.Buffer.Slot,
        .MemberSlot = buffers.LodMembers.Buffer.Slot,
        .ErrorSlot = chain.Scratch.Buffer.Slot,
    };
    return {pc, std::move(batches)};
}

void RecordLodNodeRefits(state::Scene &r, mtl::ComputeChain &chain, std::span<const LodNodeRefit> refits) {
    uint32_t deepest = 0u;
    for (const auto &refit : refits) {
        for (const auto &[depth, batch] : refit.Depths) deepest = std::max(deepest, depth);
    }
    const auto &pipeline = GetMeshPipelines(r)[MeshPass::LodNodeRefit];
    for (uint32_t depth = deepest + 1u; depth--;) {
        chain.Concurrent([&] {
            for (const auto &refit : refits) {
                for (const auto &[at, batch] : refit.Depths) {
                    if (at != depth) continue;
                    auto pc = refit.Pc;
                    pc.Jobs.Offset = batch.Offset;
                    pc.Count = batch.Count;
                    chain.Groups(pipeline, pc, pc.Count, 128u);
                }
            }
        });
    }
}
