#include "render/LodNodeEdit.h"
#include "Profile.h"
#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"
#include "render/GpuBuffers.h"
#include "state/Scene.h"
#include <map>
#include <optional>
#include <set>

namespace {
constexpr uint32_t MaxLeafRecords{2u * ClusterLodSpanLeafRecords}, MaxNodeChildren{2u * ClusterLodSpanNodeWidth};

// Divides `count` items into the fewest runs of at most `width`, whose lengths differ by at most one.
std::vector<Range> EvenRuns(uint32_t count, uint32_t width) {
    const uint64_t run_count = (count + width - 1u) / width;
    std::vector<Range> runs;
    for (uint64_t i = 0u; i < run_count; ++i) runs.push_back({uint32_t(count * i / run_count), uint32_t(count * (i + 1u) / run_count - count * i / run_count)});
    return runs;
}

// A node value to place, with the node it moves from, or InvalidOffset for a new node.
struct NodeEntry {
    LodNode Value;
    uint32_t Former{InvalidOffset};
};

// Places span-tree nodes in new contiguous runs, and each placed node's children or members point back at it.
struct SpanPlacement {
    GpuBuffers &Buffers;
    std::vector<uint32_t> Placed, Formers;
    // Each placed leaf's members with their new leaf, written together once every node is placed.
    std::vector<std::pair<uint32_t, uint32_t>> Leaves;

    // Writes the entries to one new contiguous run under `parent`.
    Range Place(std::span<const NodeEntry> entries, uint32_t parent) {
        const auto range = Buffers.LodNodes.Allocate(uint32_t(entries.size()));
        Buffers.LodParents.Mirror(range);
        for (uint32_t i = 0u; i < entries.size(); ++i) {
            const auto id = range.Offset + i;
            const auto &[value, former] = entries[i];
            Buffers.LodNodes.GetMutable({id, 1u})[0] = value;
            Buffers.LodParents.GetMutable({id, 1u})[0] = parent;
            if (value.ChildCount) std::ranges::fill(Buffers.LodParents.GetMutable({value.ChildOffset, value.ChildCount}), id);
            else Buffers.ActiveMeshlets.ForEach(value.MeshletRoot, [&](uint32_t cluster) { Leaves.emplace_back(cluster, id); });
            Placed.push_back(id);
            if (former != InvalidOffset) Formers.push_back(former);
        }
        return range;
    }

    // Gives `node` the entries as its children.
    // A node past twice the build's width keeps its first even run and returns the others as siblings its parent takes after it.
    // A root instead adds levels below itself until it holds every entry.
    std::vector<LodNode> Adopt(uint32_t node, std::vector<NodeEntry> entries) {
        if (entries.size() > MaxNodeChildren && Buffers.LodParents.Get({node, 1u})[0] != InvalidOffset) {
            const auto range = Place(entries, node);
            std::vector<LodNode> siblings;
            for (const auto run : EvenRuns(uint32_t(entries.size()), ClusterLodSpanNodeWidth)) {
                siblings.push_back({.ChildOffset = range.Offset + run.Offset, .ChildCount = run.Count});
            }
            auto &value = Buffers.LodNodes.GetMutable({node, 1u})[0];
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
        auto &value = Buffers.LodNodes.GetMutable({node, 1u})[0];
        value.ChildOffset = range.Offset;
        value.ChildCount = uint32_t(entries.size());
        value.MeshletRoot = InvalidOffset;
        return {};
    }

    // Writes the members' leaves, publishes node ownership, and releases the moved nodes' former ids.
    // The refit covers every placed node in place of the id it moved from.
    void Commit(MeshBuffers &owner, std::set<uint32_t> &affected) {
        std::vector<uint32_t> clusters;
        for (const auto &[cluster, leaf] : Leaves) clusters.push_back(cluster);
        Buffers.MeshletLodLeaves.Buffer.CaptureWriteElements(clusters, sizeof(uint32_t));
        auto *leaves = reinterpret_cast<uint32_t *>(Buffers.MeshletLodLeaves.Buffer.Contents().data());
        for (const auto &[cluster, leaf] : Leaves) leaves[cluster] = leaf;
        MeshletIndexEdit edit{.Root = owner.NodeRoot, .Added = Placed, .Removed = Formers};
        Buffers.ActiveMeshlets.Update(std::span{&edit, 1u});
        owner.NodeRoot = edit.Root;
        for (const auto former : Formers) {
            Buffers.LodNodes.Release({former, 1u});
            affected.erase(former);
        }
        affected.insert(Placed.begin(), Placed.end());
    }
};

// The members each edited leaf gains and loses.
using LeafEdits = std::map<uint32_t, std::array<std::vector<uint32_t>, 2>>;

// Splits each overfull leaf's final members into even runs of at most one leaf span in cluster order, and removes each emptied leaf from its parent's run.
// An ancestor that outgrows twice the node width splits the same way, and one left without children leaves its parent in turn.
// A primitive's root stays, and a root left without children becomes an empty leaf.
void RestructureLeaves(GpuBuffers &buffers, MeshBuffers &owner, const LeafEdits &leaves, std::span<const uint32_t> overfull, std::span<const uint32_t> emptied,
                       std::set<uint32_t> &affected) {
    const profile::CpuScope scope{"LodNodeSplit"};
    auto &index = buffers.ActiveMeshlets;
    std::vector<std::vector<uint32_t>> members(overfull.size());
    std::vector<std::vector<Range>> runs(overfull.size());
    std::vector<MeshletIndexEdit> edits;
    for (uint32_t i = 0u; i < overfull.size(); ++i) {
        const auto root = buffers.LodNodes.Get({overfull[i], 1u})[0].MeshletRoot;
        const auto &[gains, losses] = leaves.at(overfull[i]);
        auto lost = losses;
        std::ranges::sort(lost);
        auto &all = members[i];
        index.ForEach(root, [&](uint32_t cluster) { if (!std::ranges::binary_search(lost, cluster)) all.push_back(cluster); });
        const auto kept = std::ssize(all);
        all.insert(all.end(), gains.begin(), gains.end());
        std::sort(all.begin() + kept, all.end());
        std::inplace_merge(all.begin(), all.begin() + kept, all.end());
        index.Release(root);
        runs[i] = EvenRuns(uint32_t(all.size()), ClusterLodSpanLeafRecords);
        for (const auto run : runs[i]) edits.push_back({.Added = std::span{all}.subspan(run.Offset, run.Count)});
    }
    index.Update(edits);
    SpanPlacement placement{buffers};
    // Each parent's edited children, which add siblings right after themselves or, with none, leave.
    std::map<uint32_t, std::map<uint32_t, std::optional<std::vector<LodNode>>>> changes;
    for (uint32_t i = 0u, e = 0u; i < overfull.size(); ++i) {
        std::vector<NodeEntry> pieces;
        for (const auto run : runs[i]) {
            if (index.Count(edits[e].Root) != run.Count) throw std::logic_error("LOD membership edit repeats cluster identities.");
            pieces.push_back({.Value = {.MeshletRoot = edits[e++].Root}});
        }
        const auto leaf = overfull[i];
        buffers.LodNodes.GetMutable({leaf, 1u})[0].MeshletRoot = pieces.front().Value.MeshletRoot;
        const auto parent = buffers.LodParents.Get({leaf, 1u})[0];
        // A root leaf becomes the parent of its runs.
        if (parent == InvalidOffset) {
            placement.Adopt(leaf, std::move(pieces));
            continue;
        }
        auto &siblings = changes[parent][leaf].emplace();
        for (uint32_t r = 1u; r < pieces.size(); ++r) siblings.push_back(pieces[r].Value);
    }
    for (const auto leaf : emptied) {
        index.Release(buffers.LodNodes.Get({leaf, 1u})[0].MeshletRoot);
        changes[buffers.LodParents.Get({leaf, 1u})[0]][leaf];
    }
    while (!changes.empty()) {
        decltype(changes) next;
        for (const auto &[node, children] : changes) {
            const auto value = buffers.LodNodes.Get({node, 1u})[0];
            std::vector<NodeEntry> entries;
            for (uint32_t c = 0u; c < value.ChildCount; ++c) {
                const auto child = value.ChildOffset + c;
                const auto found = children.find(child);
                if (found != children.end() && !found->second) {
                    placement.Formers.push_back(child);
                    continue;
                }
                entries.push_back({.Value = buffers.LodNodes.Get({child, 1u})[0], .Former = child});
                if (found != children.end()) for (const auto &sibling : *found->second) entries.push_back({.Value = sibling});
            }
            const auto parent = buffers.LodParents.Get({node, 1u})[0];
            if (entries.empty() && parent != InvalidOffset) next[parent][node];
            else if (entries.empty()) buffers.LodNodes.GetMutable({node, 1u})[0] = {.MeshletRoot = InvalidOffset};
            else if (auto siblings = placement.Adopt(node, std::move(entries)); !siblings.empty()) next[parent][node].emplace(std::move(siblings));
        }
        changes = std::move(next);
    }
    placement.Commit(owner, affected);
}
} // namespace

LodNodeRefit EditLodNodes(state::Scene &r, mtl::ComputeChain &chain, MeshBuffers &owner, std::span<const uint32_t> removed, std::span<const LodClusterRun> added,
                          std::span<const uint32_t> touched) {
    if (removed.empty() && added.empty() && touched.empty()) return {};
    const profile::CpuScope scope{"LodNodeEdit"};
    auto &buffers = r.Context.get<GpuBuffers>();
    if (owner.MeshletRoot==InvalidOffset || owner.NodeRoot==InvalidOffset) throw std::invalid_argument("LOD membership edit requires a live render owner.");
    const auto &index = buffers.ActiveMeshlets;
    const auto meshlet_capacity = std::min(buffers.Meshlets.Buffer.Count<MeshletRecord>(),buffers.MeshletLodLeaves.Buffer.Count<uint32_t>());
    const auto node_capacity = std::min(buffers.LodNodes.Buffer.Count<LodNode>(),buffers.LodParents.Buffer.Count<uint32_t>());
    const auto primitive_capacity = buffers.Primitives.Buffer.Count<PrimitiveRecord>();
    // A traversal leaf that holds the cluster exactly when it is not being added.
    const auto holds = [&](uint32_t node, uint32_t cluster, bool adding) {
        if (node>=node_capacity || !index.Contains(owner.NodeRoot,node)) return false;
        const auto &value = buffers.LodNodes.Get({node,1u})[0];
        return !value.ChildCount && index.Contains(value.MeshletRoot,cluster) != adding;
    };
    LeafEdits leaves;
    std::set<uint32_t> affected, primitives, roots, pinned;
    // Consecutive clusters of one primitive record its nodes once, and consecutive clusters of one leaf record its path once.
    uint32_t last_primitive = InvalidOffset, last_leaf = InvalidOffset;
    // Side zero gains the cluster and side one loses it.
    const auto visit = [&](uint32_t cluster, uint32_t side, bool edited, uint32_t primitive_id, bool finest) {
        const bool adding = edited && side == 0u;
        if (cluster>=meshlet_capacity || !index.Contains(owner.MeshletRoot,cluster)) throw std::invalid_argument("LOD membership edit references a foreign cluster.");
        if (primitive_id!=last_primitive && (primitive_id>=primitive_capacity || !index.Contains(owner.PrimitiveRoot,primitive_id))) {
            throw std::invalid_argument("LOD membership edit references a foreign primitive.");
        }
        const auto &primitive = buffers.Primitives.Get({primitive_id,1u})[0];
        if (primitive_id!=last_primitive) {
            primitives.insert(primitive_id);
            roots.insert(primitive.LodRootNode);
            pinned.insert(primitive.LodFinestNode);
            affected.insert(primitive.LodFinestNode);
            last_primitive = primitive_id;
            last_leaf = InvalidOffset;
        }
        const auto leaf = buffers.MeshletLodLeaves.Get({cluster,1u})[0];
        if (!holds(leaf,cluster,adding)) throw std::invalid_argument("LOD membership edit disagrees with its traversal leaf.");
        if (edited) leaves[leaf][side].push_back(cluster);
        if (finest) {
            if (!holds(primitive.LodFinestNode,cluster,adding)) throw std::invalid_argument("LOD membership edit disagrees with its finest node.");
            if (edited && primitive.LodFinestNode!=leaf) leaves[primitive.LodFinestNode][side].push_back(cluster);
        }
        if (leaf==last_leaf) return;
        last_leaf = leaf;
        // A node already affected had its whole path to the root validated and recorded.
        for (uint32_t node=leaf, depth=0u; affected.insert(node).second; ++depth) {
            const auto parent = buffers.LodParents.Get({node,1u})[0];
            if (parent==InvalidOffset) {
                if (node!=primitive.LodRootNode) throw std::invalid_argument("LOD membership path does not reach its primitive root.");
                break;
            }
            if (parent>=node_capacity || !index.Contains(owner.NodeRoot,parent) || depth>32u) throw std::invalid_argument("LOD membership path has an invalid parent.");
            const auto &value = buffers.LodNodes.Get({parent,1u})[0];
            if (node<value.ChildOffset || uint64_t(node)>=uint64_t(value.ChildOffset)+value.ChildCount) {
                throw std::invalid_argument("LOD membership parent does not contain its child.");
            }
            node = parent;
        }
    };
    // Removed and touched clusters have published records.
    const auto visit_recorded = [&](uint32_t cluster, uint32_t side, bool edited) {
        if (cluster>=meshlet_capacity) throw std::invalid_argument("LOD membership edit references a foreign cluster.");
        const auto &record = buffers.Meshlets.Get({cluster,1u})[0];
        visit(cluster,side,edited,record.Primitive,record.RefinedGroup==InvalidOffset);
    };
    for (const auto cluster : removed) visit_recorded(cluster,1u,true);
    for (const auto &run : added)
        for (uint32_t i=0u; i<run.Count; ++i) visit(run.First+i,0u,true,run.Primitive,run.Finest);
    for (const auto cluster : touched) visit_recorded(cluster,0u,false);

    std::vector<MeshletIndexEdit> edits;
    std::vector<uint32_t> edited, counts, overfull, emptied;
    for (const auto &[node,members] : leaves) {
        const auto root = buffers.LodNodes.Get({node,1u})[0].MeshletRoot;
        const auto count = uint32_t(index.Count(root)-members[1].size()+members[0].size());
        // An overfull leaf takes its final members directly as even runs.
        if (count>MaxLeafRecords && !pinned.contains(node)) {
            overfull.push_back(node);
            continue;
        }
        if (!count && !pinned.contains(node) && !roots.contains(node)) emptied.push_back(node);
        edited.push_back(node);
        counts.push_back(count);
        edits.push_back({.Root=root,.Added=members[0],.Removed=members[1]});
    }
    buffers.ActiveMeshlets.Update(edits);
    for (uint32_t i=0u; i<edits.size(); ++i) {
        if (index.Count(edits[i].Root)!=counts[i]) throw std::logic_error("LOD membership edit repeats cluster identities.");
        buffers.LodNodes.GetMutable({edited[i],1u})[0].MeshletRoot = edits[i].Root;
    }
    if (!overfull.empty() || !emptied.empty()) RestructureLeaves(buffers,owner,leaves,overfull,emptied,affected);

    // Deeper nodes refit first, since their parents read their bounds and counts.
    std::map<uint32_t,std::vector<uint32_t>,std::greater<>> levels;
    uint32_t tree_depth = 0u;
    for (const auto node : affected) {
        uint32_t depth = 0u;
        for (auto id=node; buffers.LodParents.Get({id,1u})[0]!=InvalidOffset && depth<=32u; id=buffers.LodParents.Get({id,1u})[0]) ++depth;
        if (depth>32u || (depth ? pinned.contains(node) : !roots.contains(node) && !pinned.contains(node))) {
            throw std::invalid_argument("LOD refit has a detached node or a parented pinned node.");
        }
        levels[depth].push_back(node);
        tree_depth = std::max(tree_depth,depth);
    }
    // A root that split adds a level the traversal descends.
    if (tree_depth>buffers.MeshletLodDepth) {
        if (buffers.LodDepthHistory) buffers.LodDepthHistory->Write(0u,1u);
        buffers.MeshletLodDepth = tree_depth;
    }
    std::vector<uint32_t> jobs;
    std::vector<std::pair<uint32_t, Range>> batches;
    for (const auto &[depth,level] : levels) {
        const auto first = uint32_t(jobs.size());
        for (const auto id : level) {
            auto node = buffers.LodNodes.Get({id,1u})[0];
            uint64_t count = 0u;
            if (node.ChildCount) for (const auto &child : buffers.LodNodes.Get({node.ChildOffset,node.ChildCount})) count += child.MeshletCount;
            else count = index.Count(node.MeshletRoot);
            if (count>UINT32_MAX) throw std::length_error("LOD node membership exceeds its count.");
            node.MeshletCount = uint32_t(count);
            // A pinned finest node keeps every finest member at an infinite error, and an empty node encloses nothing.
            if (pinned.contains(id)) node.Error = std::numeric_limits<float>::infinity();
            else if (!count) node = {.FirstMeshlet=node.FirstMeshlet,.ChildOffset=node.ChildOffset,.ChildCount=node.ChildCount,.MeshletRoot=node.MeshletRoot};
            else jobs.push_back(id);
            buffers.LodNodes.GetMutable({id,1u})[0] = node;
        }
        if (jobs.size()>first) batches.push_back({depth,{first,uint32_t(jobs.size())-first}});
    }
    for (const auto id : primitives) {
        auto &primitive = buffers.Primitives.GetMutable({id,1u})[0];
        primitive.MeshletCount = buffers.LodNodes.Get({primitive.LodRootNode,1u})[0].MeshletCount;
        primitive.Level0Count = buffers.LodNodes.Get({primitive.LodFinestNode,1u})[0].MeshletCount;
    }
    if (jobs.empty()) return {};
    const auto job_words=chain.Scratch.Allocate(std::span<const uint32_t>{jobs});
    for (auto &[depth,batch] : batches) batch.Offset+=job_words.Offset;
    const LodNodeRefitPushConstants pc{
        .Jobs={chain.Scratch.Buffer.Slot,job_words.Offset},
        .Nodes=index.Ref(owner.NodeRoot),.Meshlets=index.Ref(owner.MeshletRoot),.Groups=index.Ref(owner.GroupRoot),
        .NodeSlot=buffers.LodNodes.Buffer.Slot,.ParentSlot=buffers.LodParents.Buffer.Slot,
        .MeshletSlot=buffers.Meshlets.Buffer.Slot,.GroupSlot=buffers.ClusterGroups.Buffer.Slot,.ErrorSlot=chain.Scratch.Buffer.Slot,
        .NodeCapacity=std::min(buffers.LodNodes.Buffer.Count<LodNode>(),buffers.LodParents.Buffer.Count<uint32_t>()),.MeshletCapacity=buffers.Meshlets.Buffer.Count<MeshletRecord>(),
        .GroupCapacity=buffers.ClusterGroups.Buffer.Count<ClusterGroup>(),.IndexNodeCapacity=index.Nodes.Buffer.Count<MeshletIndexNode>(),
    };
    return {pc,std::move(batches)};
}

void RecordLodNodeRefits(state::Scene &r, mtl::ComputeChain &chain, std::span<const LodNodeRefit> refits) {
    uint32_t deepest = 0u;
    for (const auto &refit : refits)
        for (const auto &[depth,batch] : refit.Depths) deepest = std::max(deepest,depth);
    const auto &pipeline = GetMeshPipelines(r)[MeshPass::LodNodeRefit];
    for (uint32_t depth = deepest+1u; depth--;) {
        chain.Concurrent([&] {
            for (const auto &refit : refits)
                for (const auto &[at,batch] : refit.Depths) {
                    if (at != depth) continue;
                    auto pc = refit.Pc;
                    pc.Jobs.Offset = batch.Offset;
                    pc.Count = batch.Count;
                    chain.Groups(pipeline,pc,pc.Count,128u);
                }
        });
    }
}
