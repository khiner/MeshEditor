#include "render/ClusterLodRepair.h"
#include "Parallel.h"
#include "Profile.h"
#include "gpu/MeshletGeometryEncoding.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/LodNodeEdit.h"
#include "metal/Dispatch.h"
#include "render/MeshletBuild.h"
#include "render/MeshletIndex.h"
#include "render/MeshletStorage.h"
#include "state/Scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <unordered_map>
#include <unordered_set>

std::vector<uint32_t> ClusterGroupClosure(const GpuBuffers &buffers, std::span<const uint32_t> seeds) {
    std::vector<uint32_t> closure(seeds.begin(), seeds.end());
    std::ranges::sort(closure);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
    std::unordered_set<uint32_t> visited(closure.begin(), closure.end());
    const auto links = buffers.GroupLinks.Buffer.GetSpan<ClusterGroupLinks>();
    const auto ids = buffers.GroupClusterIds.Buffer.GetSpan<uint32_t>();
    const auto records = buffers.Meshlets.Buffer.GetSpan<MeshletRecord>();
    for (size_t i = 0; i < closure.size(); ++i) {
        const auto link = links[closure[i]];
        for (const auto proxy : ids.subspan(link.ProxyOffset, link.ProxyCount)) {
            const auto parent = records[proxy].GroupIndex;
            if (parent != InvalidOffset && visited.insert(parent).second) closure.push_back(parent);
        }
    }
    std::ranges::sort(closure);
    return closure;
}

void ReplaceGroupClusters(GpuBuffers &buffers, std::span<const uint32_t> removed, std::span<const uint32_t> added) {
    std::vector<uint32_t> groups;
    for (const auto list : {removed, added}) {
        for (const auto id : list) {
            const auto &record = buffers.Meshlets.Get({id, 1u})[0];
            if (record.GroupIndex != InvalidOffset) groups.push_back(record.GroupIndex);
            if (record.RefinedGroup != InvalidOffset) groups.push_back(record.RefinedGroup);
        }
    }
    if (groups.empty()) return;
    std::ranges::sort(groups);
    groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
    const std::unordered_set<uint32_t> removed_ids(removed.begin(), removed.end());
    // Runs hold members first and proxies second, matching the two record fields.
    struct Runs {
        std::array<std::vector<uint32_t>, 2> Ids;
        std::array<bool, 2> Changed{};
    };
    std::vector<Runs> runs(groups.size());
    const auto run_of = [&](uint32_t group) -> Runs & { return runs[std::ranges::lower_bound(groups, group) - groups.begin()]; };
    for (uint32_t i = 0; i < groups.size(); ++i) {
        const auto links = buffers.GroupLinks.Get({groups[i], 1u})[0];
        const std::array before{Range{links.MemberOffset, links.MemberCount}, Range{links.ProxyOffset, links.ProxyCount}};
        for (uint32_t side = 0; side < 2u; ++side) {
            for (const auto id : buffers.GroupClusterIds.Get(before[side])) {
                if (removed_ids.contains(id)) runs[i].Changed[side] = true;
                else runs[i].Ids[side].push_back(id);
            }
        }
    }
    for (const auto id : added) {
        const auto &record = buffers.Meshlets.Get({id, 1u})[0];
        const std::array sides{record.GroupIndex, record.RefinedGroup};
        for (uint32_t side = 0; side < 2u; ++side) {
            if (sides[side] == InvalidOffset) continue;
            auto &run = run_of(sides[side]);
            run.Ids[side].push_back(id);
            run.Changed[side] = true;
        }
    }
    // Every rewritten run shares one allocation, so its history capture is a single range.
    uint64_t total = 0u;
    for (const auto &run : runs)
        for (uint32_t side = 0; side < 2u; ++side) if (run.Changed[side]) total += run.Ids[side].size();
    if (total > UINT32_MAX) throw std::length_error("Cluster group links exceed the canonical address domain.");
    const auto allocation = buffers.GroupClusterIds.Allocate(uint32_t(total));
    const auto ids = buffers.GroupClusterIds.GetMutable(allocation);
    buffers.GroupLinks.Buffer.CaptureWriteElements(groups, sizeof(ClusterGroupLinks));
    uint32_t next = 0u;
    for (uint32_t i = 0; i < groups.size(); ++i) {
        auto links = buffers.GroupLinks.Get({groups[i], 1u})[0];
        const std::array offsets{&links.MemberOffset, &links.ProxyOffset};
        const std::array counts{&links.MemberCount, &links.ProxyCount};
        for (uint32_t side = 0; side < 2u; ++side) {
            if (!runs[i].Changed[side]) continue;
            buffers.GroupClusterIds.Release({*offsets[side], *counts[side]});
            std::ranges::copy(runs[i].Ids[side], ids.begin() + next);
            *offsets[side] = allocation.Offset + next;
            *counts[side] = uint32_t(runs[i].Ids[side].size());
            next += *counts[side];
        }
        buffers.GroupLinks.GetMutable({groups[i], 1u})[0] = links;
    }
}

std::vector<uint32_t> InvalidateClusterGroups(state::Scene &r, state::Entity entity, std::span<const uint32_t> seeds) {
    if (seeds.empty()) return {};
    const profile::CpuScope scope{"InvalidateClusterGroups"};
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &owner = MeshBuffersOf(r, entity);
    const auto closure = ClusterGroupClosure(buffers, seeds);
    buffers.ClusterGroups.Buffer.CaptureWriteElements(closure, sizeof(ClusterGroup));
    std::vector<uint32_t> members;
    for (const auto group : closure) {
        buffers.ClusterGroups.GetMutable({group, 1u})[0].Error = INFINITY;
        const auto links = buffers.GroupLinks.Get({group, 1u})[0];
        const auto ids = buffers.GroupClusterIds.Get({links.MemberOffset, links.MemberCount});
        members.insert(members.end(), ids.begin(), ids.end());
    }
    MeshletIndexEdit edit{.Root = owner.DirtyGroupRoot, .Added = closure};
    buffers.ActiveMeshlets.Update(std::span{&edit, 1u});
    owner.DirtyGroupRoot = edit.Root;
    r.Context.get<GpuSceneState>().LodDirty.insert(entity);
    std::ranges::sort(members);
    return members;
}

namespace {
// One primitive's pool: the kept members of its stale groups, each at its group's level.
struct PoolBuild {
    uint32_t Primitive{};
    float Scale{};
    // Members in level order, and each member's level relative to the lowest stale level.
    std::vector<uint32_t> Members, Levels;
    // The lowest cluster that each level's stale groups simplified to before the repair.
    std::vector<uint32_t> Replaced;
    // Each member triangle's canonical vertex corners, in member order.
    std::vector<uvec3> Corners;
    std::vector<ClusterLodSourceCluster> Clusters;
    ClusterLodPrimitive Triangles;
    MeshletBuildInputs Inputs;
    ClusterLodBuild Build;
};

void GatherPool(const GpuBuffers &buffers, const MeshStore &meshes, const Mesh &mesh, PoolBuild &pool) {
    const auto records = buffers.Meshlets.Buffer.GetSpan<MeshletRecord>();
    const auto corners = buffers.MeshletVertexCorners.Buffer.GetSpan<uint32_t>();
    const auto local = buffers.MeshletLocalTriangles.Buffer.GetSpan<uint8_t>();
    std::vector<uint32_t> first_triangles(pool.Members.size() + 1u);
    for (uint32_t i = 0; i < pool.Members.size(); ++i) first_triangles[i + 1u] = first_triangles[i] + records[pool.Members[i]].TriangleCount;
    pool.Corners.resize(first_triangles.back());
    pool.Clusters.resize(pool.Members.size());
    constexpr uint32_t MembersPerBlock{256u};
    ParallelFor((uint32_t(pool.Members.size()) + MembersPerBlock - 1u) / MembersPerBlock, [&](uint32_t block) {
        for (uint32_t i = block * MembersPerBlock; i < std::min(uint32_t(pool.Members.size()), (block + 1u) * MembersPerBlock); ++i) {
            const auto &record = records[pool.Members[i]];
            for (uint32_t t = 0; t < record.TriangleCount; ++t) {
                auto &triangle = pool.Corners[first_triangles[i] + t];
                for (uint32_t c = 0; c < 3u; ++c) {
                    const auto vertex = local[record.LocalTriangleOffset + t * 3u + c] & uint8_t(MeshletGeometryEncoding::LocalIndexMask);
                    triangle[c] = corners[record.VertexOffset + vertex];
                }
            }
            const auto sphere = record.RefinedGroup == InvalidOffset ?
                ClusterGroup{.Center = record.Center, .Radius = record.Radius} :
                buffers.ClusterGroups.Get({record.RefinedGroup, 1u})[0];
            pool.Clusters[i] = {
                .FirstVertex = record.VertexOffset, .VertexCount = record.VertexCount,
                .FirstLocalTriangle = record.LocalTriangleOffset, .TriangleCount = record.TriangleCount,
                .Center = sphere.Center, .Radius = sphere.Radius, .Error = sphere.Error,
                .ConeCullSafe = (record.ConeAxisCutoff >> 24u) != 127u,
            };
        }
    });
    pool.Scale = buffers.Primitives.Get({pool.Primitive, 1u})[0].SimplifyScale;
    pool.Triangles = {
        .TriangleCount = uint32_t(pool.Corners.size()), .ClusterCount = uint32_t(pool.Members.size()),
        .Attributes = buffers.Primitives.Get({pool.Primitive, 1u})[0].LodAttributes,
    };
    pool.Inputs = CaptureMeshletInputs(mesh, meshes, {ElementView<uvec3>{pool.Corners}, {}});
}

// A group of original geometry sits at level zero, and any other group sits one level above the groups its members came from.
// Members of one group share a level, so the first member decides.
uint32_t GroupLevel(const GpuBuffers &buffers, uint32_t group, std::unordered_map<uint32_t, uint32_t> &levels) {
    if (const auto found = levels.find(group); found != levels.end()) return found->second;
    const auto links = buffers.GroupLinks.Get({group, 1u})[0];
    uint32_t level = 0;
    if (links.MemberCount) {
        const auto first = buffers.GroupClusterIds.Get({links.MemberOffset, 1u})[0];
        const auto refined = buffers.Meshlets.Get({first, 1u})[0].RefinedGroup;
        if (refined != InvalidOffset) level = GroupLevel(buffers, refined, levels) + 1u;
    }
    levels.emplace(group, level);
    return level;
}

// Places the pool's rebuilt groups and new clusters, points its members at their new groups, and links every new group.
// Appends the new clusters and groups.
void CommitPool(GpuBuffers &buffers, const PoolBuild &pool, std::vector<uint32_t> &added, std::vector<uint32_t> &new_groups) {
    const auto &build = pool.Build;
    Range groups,vertices,triangles;
    const auto allocation=PublishClusterLodStorage(buffers,build,std::array{pool.Primitive},groups,vertices,triangles);
    const auto group_id=[&](uint32_t group) { return group==ClusterLodInvalid ? InvalidOffset : groups.Offset+group; };
    buffers.Meshlets.Buffer.CaptureWriteElements(pool.Members, sizeof(MeshletRecord));
    auto *meshlets = reinterpret_cast<MeshletRecord *>(buffers.Meshlets.Buffer.Contents().data());
    for (uint32_t i = 0; i < pool.Members.size(); ++i) meshlets[pool.Members[i]].GroupIndex = group_id(build.Level0Groups[i]);
    const auto record_of = [&](uint32_t id) { return id < build.Level0Count() ? pool.Members[id] : allocation.Offset + id - build.Level0Count(); };

    // Groups follow level order, so every new cluster's refined group has its level before any group the cluster joins.
    std::vector<uint32_t> group_levels(build.Groups.size());
    for (uint32_t g = 0; g < build.Groups.size(); ++g) {
        const auto first = build.GroupClusters[build.Groups[g].FirstCluster];
        group_levels[g] = first < build.Level0Count() ? pool.Levels[first] : group_levels[build.Clusters[first - build.Level0Count()].RefinedGroup] + 1u;
    }
    // A new cluster joins the leaf of the lowest cluster its level replaces, or the leaf the level below chose where its level replaces none.
    std::vector<uint32_t> anchors;
    for (uint32_t level = 0, anchor = pool.Members.front(); level < build.LevelCount; ++level) {
        if (level < pool.Replaced.size() && pool.Replaced[level] != InvalidOffset) anchor = pool.Replaced[level];
        anchors.push_back(buffers.MeshletLodLeaves.Get({anchor, 1u})[0]);
    }
    const auto leaves = buffers.MeshletLodLeaves.GetMutable(allocation);
    for (uint32_t c = 0; c < build.Clusters.size(); ++c) leaves[c] = anchors[group_levels[build.Clusters[c].RefinedGroup]];

    const auto links=std::span{reinterpret_cast<ClusterGroupLinks *>(buffers.GroupLinks.Buffer.Contents().data())+groups.Offset,groups.Count};
    auto *ids=reinterpret_cast<uint32_t *>(buffers.GroupClusterIds.Buffer.Contents().data());
    for (uint32_t g=0u;g<build.Groups.size();++g) {
        const auto &group=build.Groups[g];
        auto &link=links[g];
        for (const auto id : std::span{build.GroupClusters}.subspan(group.FirstCluster,group.ClusterCount))
            ids[link.MemberOffset+link.MemberCount++]=record_of(id);
    }
    for (uint32_t i = 0; i < allocation.Count; ++i) added.push_back(allocation.Offset + i);
    for (uint32_t i = 0; i < groups.Count; ++i) new_groups.push_back(groups.Offset + i);
}
} // namespace

void RepairDirtyClusterGroups(state::Scene &r, MeshBuffers &owner) {
    auto &buffers = r.Context.get<GpuBuffers>();
    if (!buffers.ActiveMeshlets.Count(owner.DirtyGroupRoot)) {
        buffers.ActiveMeshlets.Release(owner.DirtyGroupRoot);
        owner.DirtyGroupRoot = InvalidOffset;
        return;
    }
    const profile::CpuScope scope{"ClusterLodRepair"};
    const auto &meshes = r.Context.get<const MeshStore>();
    const Mesh mesh{meshes, owner.StoreId};
    std::vector<uint32_t> seeds;
    buffers.ActiveMeshlets.ForEach(owner.DirtyGroupRoot, [&](uint32_t group) { seeds.push_back(group); });
    std::unordered_map<uint32_t, uint32_t> levels;
    std::map<uint32_t, std::vector<uint32_t>> by_level;
    const auto closure = ClusterGroupClosure(buffers, seeds);
    for (const auto group : closure) by_level[GroupLevel(buffers, group, levels)].push_back(group);
    profile::RecordCounter("LodRepairGroups", closure.size());

    // Every stale group retires with the clusters it simplified to.
    std::vector<uint32_t> removed;
    for (const auto group : closure) {
        const auto links = buffers.GroupLinks.Get({group, 1u})[0];
        const auto proxies = buffers.GroupClusterIds.Get({links.ProxyOffset, links.ProxyCount});
        removed.insert(removed.end(), proxies.begin(), proxies.end());
    }
    // Each primitive's pool re-partitions every level as a full build does, and its kept members join at their groups' levels.
    std::map<uint32_t, PoolBuild> pools;
    {
        const profile::CpuScope stage{"LodRepairGather"};
        const auto records = buffers.Meshlets.Buffer.GetSpan<MeshletRecord>();
        const auto first_level = by_level.begin()->first, level_count = by_level.rbegin()->first - first_level + 1u;
        const auto pool_of = [&](uint32_t id) -> PoolBuild & {
            auto &pool = pools[records[id].Primitive];
            pool.Primitive = records[id].Primitive;
            pool.Replaced.resize(level_count, InvalidOffset);
            return pool;
        };
        // Levels ascend, so each pool's members arrive in level order.
        for (const auto &[level, groups] : by_level) {
            for (const auto group : groups) {
                const auto links = buffers.GroupLinks.Get({group, 1u})[0];
                for (const auto id : buffers.GroupClusterIds.Get({links.ProxyOffset, links.ProxyCount})) {
                    auto &replaced = pool_of(id).Replaced[level - first_level];
                    replaced = std::min(replaced, id);
                }
                // A stale group's error is infinite, so a member simplified from one retires with it.
                for (const auto id : buffers.GroupClusterIds.Get({links.MemberOffset, links.MemberCount})) {
                    const auto refined = records[id].RefinedGroup;
                    if (refined != InvalidOffset && std::isinf(buffers.ClusterGroups.Get({refined, 1u})[0].Error)) continue;
                    auto &pool = pool_of(id);
                    pool.Members.push_back(id);
                    pool.Levels.push_back(level - first_level);
                }
            }
        }
        std::erase_if(pools, [](const auto &entry) { return entry.second.Members.empty(); });
        for (auto &[primitive, pool] : pools) GatherPool(buffers, meshes, mesh, pool);
    }
    {
        const profile::CpuScope stage{"LodRepairSimplify"};
        const auto vertex_corners = buffers.MeshletVertexCorners.Buffer.GetSpan<uint32_t>();
        const auto local_triangles = buffers.MeshletLocalTriangles.Buffer.GetSpan<uint8_t>();
        std::vector<PoolBuild *> work;
        for (auto &[primitive, pool] : pools) work.push_back(&pool);
        ParallelFor(uint32_t(work.size()), [&](uint32_t i) {
            auto &pool = *work[i];
            pool.Build = RebuildClusterLod(ClusterLodMesh{
                .CornerVertices = pool.Inputs.Indices,
                .Positions = &pool.Inputs.Vertices.front().Position.x, .PositionStride = sizeof(Vertex),
                .VertexFirst = pool.Inputs.VertexFirst, .DenseVertices = pool.Inputs.DenseVertices,
                .Normals = pool.Inputs.Normals,
                .Weld = pool.Inputs.Weld,
                .Primitives = std::span{&pool.Triangles, 1u}, .Clusters = pool.Clusters,
                .SourceVertexCorners = vertex_corners, .SourceLocalTriangles = local_triangles,
            }, pool.Levels, pool.Scale);
        });
    }
    std::vector<uint32_t> added, new_groups, touched;
    {
        const profile::CpuScope stage{"LodRepairCommit"};
        for (const auto &[primitive, pool] : pools) {
            CommitPool(buffers, pool, added, new_groups);
            // The kept members joined new groups, so their leaves refit.
            touched.insert(touched.end(), pool.Members.begin(), pool.Members.end());
        }
    }
    const profile::CpuScope stage{"LodRepairPublish"};
    std::ranges::sort(added);
    std::array ownership{
        MeshletIndexEdit{.Root = owner.MeshletRoot, .Added = added},
        MeshletIndexEdit{.Root = owner.GroupRoot, .Added = new_groups, .Removed = closure},
    };
    buffers.ActiveMeshlets.Update(ownership);
    owner.MeshletRoot = ownership[0].Root;
    owner.GroupRoot = ownership[1].Root;
    ++owner.MeshletRevision;
    std::vector<uint32_t> blocks;
    for (const auto id : added) if (blocks.empty() || blocks.back() != id / 256u) blocks.push_back(id / 256u);
    buffers.PosedMeshletBounds.UpdateBlocks(owner.StoreId, owner.MeshletRevision, blocks,
        [&](uint32_t block) { return buffers.ActiveMeshlets.HasBlock(owner.MeshletRoot, block); }, owner.RenderTopology);
    std::vector<LodClusterRun> runs;
    for (const auto id : added) {
        const auto primitive = buffers.Meshlets.Get({id, 1u})[0].Primitive;
        if (!runs.empty() && runs.back().First + runs.back().Count == id && runs.back().Primitive == primitive) ++runs.back().Count;
        else runs.push_back({id, 1u, primitive, false});
    }
    mtl::ComputeChain chain{buffers.Ctx};
    EditLodNodes(r, chain, owner, removed, runs, touched);
    chain.Submit();
    RetireMeshletStorage(r, owner, removed);
    std::vector<Range> runs_released, groups_released;
    for (const auto group : closure) {
        const auto links = buffers.GroupLinks.Get({group, 1u})[0];
        runs_released.insert(runs_released.end(), {Range{links.MemberOffset, links.MemberCount}, Range{links.ProxyOffset, links.ProxyCount}});
        groups_released.push_back({group, 1u});
    }
    buffers.GroupClusterIds.Release(std::move(runs_released));
    buffers.ClusterGroups.Release(std::move(groups_released));
    buffers.ActiveMeshlets.Release(owner.DirtyGroupRoot);
    owner.DirtyGroupRoot = InvalidOffset;
    buffers.PreludeStale = true;
}
