#include "render/ClusterLodRepair.h"
#include "Parallel.h"
#include "Profile.h"
#include "gpu/MeshletGeometryEncoding.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "render/GpuBufferOps.h"
#include "render/GpuSceneState.h"
#include "render/LodNodeEdit.h"
#include "render/MeshletBuild.h"
#include "render/MeshletStorage.h"
#include "state/Scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <unordered_map>
#include <unordered_set>

std::vector<uint32_t> ClusterGroupClosure(const RenderArenas &buffers, const MeshStore::Record &owner, std::span<const uint32_t> seeds) {
    std::vector<uint32_t> closure(seeds.begin(), seeds.end());
    std::ranges::sort(closure);
    closure.erase(std::ranges::unique(closure).begin(), closure.end());
    std::unordered_set<uint32_t> visited(closure.begin(), closure.end());
    const auto groups = buffers.ClusterGroups.Entries(owner.GroupData), meshlets = buffers.Meshlets.Entries(owner.MeshletData);
    const auto links = buffers.GroupLinks.Buffer.GetSpan<ClusterGroupLinks>();
    const auto ids = buffers.GroupClusterIds.Get(owner.GroupClusters.Values);
    const auto records = buffers.Meshlets.Buffer.GetSpan<MeshletRecord>();
    for (size_t i = 0; i < closure.size(); ++i) {
        const auto link = links[Resolve<RenderBlockSize>(groups, ElementHandle{closure[i]}).Value()];
        for (const auto proxy : ids.subspan(link.ProxyOffset, link.ProxyCount)) {
            const auto parent = records[Resolve<RenderBlockSize>(meshlets, ElementHandle{proxy}).Value()].GroupIndex;
            if (parent != InvalidOffset && visited.insert(parent).second) closure.push_back(parent);
        }
    }
    std::ranges::sort(closure);
    return closure;
}

void ReplaceGroupClusters(MeshStore &meshes, MeshStore::Record &owner, std::span<const uint32_t> removed, std::span<const uint32_t> added) {
    auto &buffers = meshes.Render();
    const auto meshlets = buffers.Meshlets.Entries(owner.MeshletData), group_entries = buffers.ClusterGroups.Entries(owner.GroupData);
    const auto record_of = [&](uint32_t id) -> const MeshletRecord & { return buffers.Meshlets.Get({Resolve<RenderBlockSize>(meshlets, ElementHandle{id}).Value(), 1u})[0]; };
    std::vector<uint32_t> groups;
    for (const auto list : {removed, added}) {
        for (const auto id : list) {
            const auto &record = record_of(id);
            if (record.GroupIndex != InvalidOffset) groups.push_back(record.GroupIndex);
            if (record.RefinedGroup != InvalidOffset) groups.push_back(record.RefinedGroup);
        }
    }
    if (groups.empty()) return;
    std::ranges::sort(groups);
    groups.erase(std::ranges::unique(groups).begin(), groups.end());
    const std::unordered_set<uint32_t> removed_ids(removed.begin(), removed.end());
    // Runs hold members first and proxies second, matching the two record fields.
    struct Runs {
        std::array<std::vector<uint32_t>, 2> Ids;
        std::array<bool, 2> Changed{};
    };
    std::vector<Runs> runs(groups.size());
    std::vector<uint32_t> addresses;
    addresses.reserve(groups.size());
    for (const auto group : groups) addresses.push_back(Resolve<RenderBlockSize>(group_entries, ElementHandle{group}).Value());
    const auto run_of = [&](uint32_t group) -> Runs & { return runs[std::ranges::lower_bound(groups, group) - groups.begin()]; };
    const auto group_ids = buffers.GroupClusterIds.Get(owner.GroupClusters.Values);
    for (uint32_t i = 0; i < groups.size(); ++i) {
        const auto links = buffers.GroupLinks.Get({addresses[i], 1u})[0];
        const std::array before{Range{links.MemberOffset, links.MemberCount}, Range{links.ProxyOffset, links.ProxyCount}};
        for (uint32_t side = 0; side < 2u; ++side) {
            for (const auto id : group_ids.subspan(before[side].Offset, before[side].Count)) {
                if (removed_ids.contains(id)) runs[i].Changed[side] = true;
                else runs[i].Ids[side].push_back(id);
            }
        }
    }
    for (const auto id : added) {
        const auto &record = record_of(id);
        const std::array sides{record.GroupIndex, record.RefinedGroup};
        for (uint32_t side = 0; side < 2u; ++side) {
            if (sides[side] == InvalidOffset) continue;
            auto &run = run_of(sides[side]);
            run.Ids[side].push_back(id);
            run.Changed[side] = true;
        }
    }
    // The replaced runs leave the owner's range, and every rewritten run shares one appended run, so its history capture is a single range.
    uint64_t total = 0u;
    std::vector<Range> replaced;
    for (uint32_t i = 0; i < groups.size(); ++i) {
        const auto links = buffers.GroupLinks.Get({addresses[i], 1u})[0];
        const std::array before{Range{links.MemberOffset, links.MemberCount}, Range{links.ProxyOffset, links.ProxyCount}};
        for (uint32_t side = 0; side < 2u; ++side) {
            if (runs[i].Changed[side]) {
                total += runs[i].Ids[side].size();
                replaced.push_back(before[side]);
            }
        }
    }
    if (total > UINT32_MAX) throw std::length_error("Cluster group links exceed the canonical address domain.");
    meshes.ReleasePayloadRuns(owner, RenderPayload::GroupClusters, std::move(replaced));
    const auto first = meshes.AppendPayload(owner, RenderPayload::GroupClusters, uint32_t(total));
    const auto ids = buffers.GroupClusterIds.GetMutable({owner.GroupClusters.Values.Offset + first, uint32_t(total)});
    buffers.GroupLinks.Buffer.CaptureWriteElements(addresses, sizeof(ClusterGroupLinks));
    uint32_t next = 0u;
    for (uint32_t i = 0; i < groups.size(); ++i) {
        auto links = buffers.GroupLinks.Get({addresses[i], 1u})[0];
        const std::array offsets{&links.MemberOffset, &links.ProxyOffset};
        const std::array counts{&links.MemberCount, &links.ProxyCount};
        for (uint32_t side = 0; side < 2u; ++side) {
            if (!runs[i].Changed[side]) continue;
            std::ranges::copy(runs[i].Ids[side], ids.begin() + next);
            *offsets[side] = first + next;
            *counts[side] = uint32_t(runs[i].Ids[side].size());
            next += *counts[side];
        }
        buffers.GroupLinks.GetMutable({addresses[i], 1u})[0] = links;
    }
}

std::vector<std::vector<uint32_t>> InvalidateClusterGroups(state::Scene &r, std::span<const ClusterGroupSeeds> seeds) {
    std::vector<std::vector<uint32_t>> members(seeds.size());
    if (std::ranges::all_of(seeds, [](const auto &seed) { return seed.Groups.empty(); })) return members;
    const profile::CpuScope scope{"InvalidateClusterGroups"};
    auto &buffers = r.Context.get<MeshStore>().Render();
    for (uint32_t i = 0u; i < seeds.size(); ++i) {
        if (seeds[i].Groups.empty()) continue;
        const auto &owner = RecordOf(r, seeds[i].Entity);
        const auto entries = buffers.ClusterGroups.Entries(owner.GroupData);
        const auto closure = ClusterGroupClosure(buffers, owner, seeds[i].Groups);
        std::vector<uint32_t> addresses;
        addresses.reserve(closure.size());
        for (const auto group : closure) addresses.push_back(Resolve<RenderBlockSize>(entries, ElementHandle{group}).Value());
        std::ranges::sort(addresses);
        buffers.ClusterGroups.Buffer.CaptureWriteElements(addresses, sizeof(ClusterGroup));
        for (const auto group : closure) {
            const auto address = Resolve<RenderBlockSize>(entries, ElementHandle{group});
            buffers.ClusterGroups.GetMutable({address.Value(), 1u})[0].Error = INFINITY;
            const auto links = buffers.GroupLinks.Get({address.Value(), 1u})[0];
            const auto ids = buffers.GroupClusterIds.Get(owner.GroupClusters.Values).subspan(links.MemberOffset, links.MemberCount);
            members[i].insert(members[i].end(), ids.begin(), ids.end());
            MaskSet(buffers.DirtyGroups, address);
        }
        std::ranges::sort(members[i]);
        r.emplace_or_replace<ClusterLodDirty>(seeds[i].Entity);
    }
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
    // The owner's payloads the clusters index.
    std::span<const uint32_t> VertexCorners;
    std::span<const MeshletTriangle> LocalTriangles;
    ClusterLodPrimitive Triangles;
    MeshletBuildInputs Inputs;
    ClusterLodBuild Build;
};

// The owner's render tables, resolving its cluster, group and primitive handles to arena addresses, and its payloads.
struct OwnerTables {
    std::span<const uint32_t> Meshlets, Groups, Primitives, VertexCorners, GroupClusters;
    std::span<const MeshletTriangle> LocalTriangles;
    OwnerTables(const RenderArenas &buffers, const MeshStore::Record &owner)
        : Meshlets{buffers.Meshlets.Entries(owner.MeshletData)}, Groups{buffers.ClusterGroups.Entries(owner.GroupData)}, Primitives{buffers.Primitives.Entries(owner.PrimitiveData)},
          VertexCorners{buffers.MeshletVertexCorners.Get(owner.MeshletVertices.Values)}, GroupClusters{buffers.GroupClusterIds.Get(owner.GroupClusters.Values)}, LocalTriangles{buffers.MeshletLocalTriangles.Get(owner.MeshletLocalTriangles.Values)} {}
};

void GatherPool(const RenderArenas &buffers, const OwnerTables &tables, const MeshStore &meshes, const Mesh &mesh, PoolBuild &pool) {
    const auto records = buffers.Meshlets.Buffer.GetSpan<MeshletRecord>();
    const auto record_of = [&](uint32_t id) -> const MeshletRecord & { return records[Resolve<RenderBlockSize>(tables.Meshlets, ElementHandle{id}).Value()]; };
    std::vector<uint32_t> first_triangles(pool.Members.size() + 1u);
    for (uint32_t i = 0; i < pool.Members.size(); ++i) first_triangles[i + 1u] = first_triangles[i] + record_of(pool.Members[i]).TriangleCount;
    pool.Corners.resize(first_triangles.back());
    pool.Clusters.resize(pool.Members.size());
    constexpr uint32_t MembersPerBlock{256u};
    ParallelFor((uint32_t(pool.Members.size()) + MembersPerBlock - 1u) / MembersPerBlock, [&](uint32_t block) {
        for (uint32_t i = block * MembersPerBlock; i < std::min(uint32_t(pool.Members.size()), (block + 1u) * MembersPerBlock); ++i) {
            const auto &record = record_of(pool.Members[i]);
            for (uint32_t t = 0; t < record.TriangleCount; ++t) {
                auto &triangle = pool.Corners[first_triangles[i] + t];
                for (uint32_t c = 0; c < 3u; ++c) {
                    const auto vertex = tables.LocalTriangles[record.LocalTriangleOffset + t][c] & uint8_t(MeshletGeometryEncoding::LocalIndexMask);
                    triangle[c] = tables.VertexCorners[record.VertexOffset + vertex];
                }
            }
            const auto sphere = record.RefinedGroup == InvalidOffset ?
                ClusterGroup{.Center = record.Center, .Radius = record.Radius} :
                buffers.ClusterGroups.Get({Resolve<RenderBlockSize>(tables.Groups, ElementHandle{record.RefinedGroup}).Value(), 1u})[0];
            pool.Clusters[i] = {
                .FirstVertex = record.VertexOffset,
                .VertexCount = record.VertexCount,
                .FirstLocalTriangle = record.LocalTriangleOffset,
                .TriangleCount = record.TriangleCount,
                .Center = sphere.Center,
                .Radius = sphere.Radius,
                .Error = sphere.Error,
                .ConeCullSafe = (record.ConeAxisCutoff >> 24u) != 127u,
            };
        }
    });
    const auto &primitive = buffers.Primitives.Get({Resolve<RenderBlockSize>(tables.Primitives, ElementHandle{pool.Primitive}).Value(), 1u})[0];
    pool.Scale = primitive.SimplifyScale;
    pool.VertexCorners = tables.VertexCorners;
    pool.LocalTriangles = tables.LocalTriangles;
    pool.Triangles = {
        .TriangleCount = uint32_t(pool.Corners.size()),
        .ClusterCount = uint32_t(pool.Members.size()),
        .Attributes = primitive.LodAttributes,
    };
    pool.Inputs = CaptureMeshletInputs(mesh, meshes, {ElementView<uvec3>{pool.Corners}, {}, {}, meshes.GetTriangleCorners(mesh.GetStoreId()).CornerTable});
}

// A group of original geometry sits at level zero, and any other group sits one level above the groups its members came from.
// Members of one group share a level, so the first member decides.
uint32_t GroupLevel(const RenderArenas &buffers, const OwnerTables &tables, uint32_t group, std::unordered_map<uint32_t, uint32_t> &levels) {
    if (const auto found = levels.find(group); found != levels.end()) return found->second;
    const auto links = buffers.GroupLinks.Get({Resolve<RenderBlockSize>(tables.Groups, ElementHandle{group}).Value(), 1u})[0];
    uint32_t level = 0;
    if (links.MemberCount) {
        const auto first = tables.GroupClusters[links.MemberOffset];
        const auto refined = buffers.Meshlets.Get({Resolve<RenderBlockSize>(tables.Meshlets, ElementHandle{first}).Value(), 1u})[0].RefinedGroup;
        if (refined != InvalidOffset) level = GroupLevel(buffers, tables, refined, levels) + 1u;
    }
    levels.emplace(group, level);
    return level;
}

// Places the pool's rebuilt groups and new clusters, points its members at their new groups, and links every new group.
// Appends the new clusters and groups.
void CommitPool(MeshStore &meshes, MeshStore::Record &owner, const PoolBuild &pool, std::vector<uint32_t> &added, std::vector<uint32_t> &new_groups) {
    auto &buffers = meshes.Render();
    const auto &build = pool.Build;
    RenderRun groups;
    const auto allocation = PublishClusterLodStorage(meshes, owner, build, std::array{pool.Primitive}, groups);
    const OwnerTables tables{buffers, owner};
    const auto group_id = [&](uint32_t group) { return group == ClusterLodInvalid ? InvalidOffset : groups.Handle.Value + group; };
    std::vector<uint32_t> members;
    members.reserve(pool.Members.size());
    for (const auto id : pool.Members) members.push_back(Resolve<RenderBlockSize>(tables.Meshlets, ElementHandle{id}).Value());
    buffers.Meshlets.Buffer.CaptureWriteElements(members, sizeof(MeshletRecord));
    auto *meshlets = reinterpret_cast<MeshletRecord *>(buffers.Meshlets.Buffer.Contents().data());
    for (uint32_t i = 0; i < pool.Members.size(); ++i) meshlets[members[i]].GroupIndex = group_id(build.Level0Groups[i]);
    const auto record_of = [&](uint32_t id) { return id < build.Level0Count() ? pool.Members[id] : allocation.Handle.Value + id - build.Level0Count(); };

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
        anchors.push_back(buffers.MeshletLodLeaves.Get({Resolve<RenderBlockSize>(tables.Meshlets, ElementHandle{anchor}).Value(), 1u})[0]);
    }
    const auto leaves = buffers.MeshletLodLeaves.GetMutable({allocation.Address.Value(), allocation.Count});
    for (uint32_t c = 0; c < build.Clusters.size(); ++c) leaves[c] = anchors[group_levels[build.Clusters[c].RefinedGroup]];

    const auto links = std::span{reinterpret_cast<ClusterGroupLinks *>(buffers.GroupLinks.Buffer.Contents().data()) + groups.Address.Value(), groups.Count};
    auto *ids = reinterpret_cast<uint32_t *>(buffers.GroupClusterIds.Buffer.Contents().data()) + owner.GroupClusters.Values.Offset;
    for (uint32_t g = 0u; g < build.Groups.size(); ++g) {
        const auto &group = build.Groups[g];
        auto &link = links[g];
        for (const auto id : std::span{build.GroupClusters}.subspan(group.FirstCluster, group.ClusterCount)) {
            ids[link.MemberOffset + link.MemberCount++] = record_of(id);
        }
    }
    for (uint32_t i = 0; i < allocation.Count; ++i) added.push_back(allocation.Handle.Value + i);
    for (uint32_t i = 0; i < groups.Count; ++i) new_groups.push_back(groups.Handle.Value + i);
}
} // namespace

void RepairDirtyClusterGroups(state::Scene &r, mtl::ComputeChain &chain, std::span<const state::Entity> entities) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &buffers = meshes.Render();
    // One owner's stale closure, the coarse clusters it retires and its pools by primitive.
    struct OwnerRepair {
        state::Entity Entity;
        std::vector<uint32_t> Closure, Removed;
        std::map<uint32_t, PoolBuild> Pools;
    };
    std::vector<OwnerRepair> repairs;
    for (const auto entity : entities) {
        if (meshes.HasDirtyGroups(RecordOf(r, entity))) repairs.push_back({.Entity = entity});
    }
    if (repairs.empty()) return;
    const profile::CpuScope scope{"ClusterLodRepair"};
    {
        const profile::CpuScope stage{"LodRepairGather"};
        const auto records = buffers.Meshlets.Buffer.GetSpan<MeshletRecord>();
        for (auto &repair : repairs) {
            const auto &owner = RecordOf(r, repair.Entity);
            const OwnerTables tables{buffers, owner};
            const auto record_of = [&](uint32_t id) -> const MeshletRecord & { return records[Resolve<RenderBlockSize>(tables.Meshlets, ElementHandle{id}).Value()]; };
            const Mesh mesh{meshes, owner.StoreId};
            // The stale groups are the owner's groups whose dirty bits are set.
            std::vector<uint32_t> seeds;
            buffers.ClusterGroups.ForEach(owner.GroupData, [&](ElementHandle group, ElementAddress address) {
                if (MaskHas(buffers.DirtyGroups, address)) seeds.push_back(group.Value);
            });
            std::unordered_map<uint32_t, uint32_t> levels;
            std::map<uint32_t, std::vector<uint32_t>> by_level;
            repair.Closure = ClusterGroupClosure(buffers, owner, seeds);
            for (const auto group : repair.Closure) by_level[GroupLevel(buffers, tables, group, levels)].push_back(group);
            profile::RecordCounter("LodRepairGroups", repair.Closure.size());
            const auto links_of = [&](uint32_t group) { return buffers.GroupLinks.Get({Resolve<RenderBlockSize>(tables.Groups, ElementHandle{group}).Value(), 1u})[0]; };
            // Every stale group retires with the clusters it simplified to.
            for (const auto group : repair.Closure) {
                const auto links = links_of(group);
                const auto proxies = tables.GroupClusters.subspan(links.ProxyOffset, links.ProxyCount);
                repair.Removed.insert(repair.Removed.end(), proxies.begin(), proxies.end());
            }
            // Each primitive's pool re-partitions every level as a full build does, and its kept members join at their groups' levels.
            auto &pools = repair.Pools;
            const auto first_level = by_level.begin()->first, level_count = by_level.rbegin()->first - first_level + 1u;
            const auto pool_of = [&](uint32_t id) -> PoolBuild & {
                auto &pool = pools[record_of(id).Primitive];
                pool.Primitive = record_of(id).Primitive;
                pool.Replaced.resize(level_count, InvalidOffset);
                return pool;
            };
            // Levels ascend, so each pool's members arrive in level order.
            for (const auto &[level, groups] : by_level) {
                for (const auto group : groups) {
                    const auto links = links_of(group);
                    for (const auto id : tables.GroupClusters.subspan(links.ProxyOffset, links.ProxyCount)) {
                        auto &replaced = pool_of(id).Replaced[level - first_level];
                        replaced = std::min(replaced, id);
                    }
                    // A stale group's error is infinite, so a member simplified from one retires with it.
                    for (const auto id : tables.GroupClusters.subspan(links.MemberOffset, links.MemberCount)) {
                        const auto refined = record_of(id).RefinedGroup;
                        if (refined != InvalidOffset && std::isinf(buffers.ClusterGroups.Get({Resolve<RenderBlockSize>(tables.Groups, ElementHandle{refined}).Value(), 1u})[0].Error)) continue;
                        auto &pool = pool_of(id);
                        pool.Members.push_back(id);
                        pool.Levels.push_back(level - first_level);
                    }
                }
            }
            std::erase_if(pools, [](const auto &entry) { return entry.second.Members.empty(); });
            for (auto &[primitive, pool] : pools) GatherPool(buffers, tables, meshes, mesh, pool);
        }
    }
    {
        const profile::CpuScope stage{"LodRepairSimplify"};
        std::vector<PoolBuild *> work;
        for (auto &repair : repairs) {
            for (auto &[primitive, pool] : repair.Pools) work.push_back(&pool);
        }
        ParallelFor(uint32_t(work.size()), [&](uint32_t i) {
            auto &pool = *work[i];
            pool.Build = RebuildClusterLod(ClusterLodMesh{
                                               .CornerVertices = pool.Inputs.Indices,
                                               .Positions = &pool.Inputs.Vertices.front().Position.x,
                                               .PositionStride = sizeof(Vertex),
                                               .VertexTable = pool.Inputs.VertexTable,
                                               .DenseVertices = pool.Inputs.DenseVertices,
                                               .Normals = pool.Inputs.Normals,
                                               .Weld = pool.Inputs.Weld,
                                               .Primitives = std::span{&pool.Triangles, 1u},
                                               .Clusters = pool.Clusters,
                                               .SourceVertexCorners = pool.VertexCorners,
                                               .SourceLocalTriangles = pool.LocalTriangles,
                                           },
                                           pool.Levels, pool.Scale);
        });
    }
    // Each owner's new clusters, new groups and kept members.
    std::vector<std::vector<uint32_t>> added(repairs.size()), new_groups(repairs.size()), touched(repairs.size());
    {
        const profile::CpuScope stage{"LodRepairCommit"};
        for (uint32_t i = 0u; i < repairs.size(); ++i) {
            auto &owner = EditRecordOf(r, repairs[i].Entity);
            for (const auto &[primitive, pool] : repairs[i].Pools) {
                CommitPool(meshes, owner, pool, added[i], new_groups[i]);
                // The kept members joined new groups, so their leaves refit.
                touched[i].insert(touched[i].end(), pool.Members.begin(), pool.Members.end());
            }
            std::ranges::sort(added[i]);
        }
    }
    const profile::CpuScope stage{"LodRepairPublish"};
    std::vector<LodNodeRefit> refits;
    for (uint32_t i = 0u; i < repairs.size(); ++i) {
        const auto &repair = repairs[i];
        auto &owner = EditRecordOf(r, repair.Entity);
        ++owner.MeshletRevision;
        const auto meshlets = buffers.Meshlets.Entries(owner.MeshletData);
        std::vector<uint32_t> addresses;
        for (const auto id : added[i]) addresses.push_back(Resolve<RenderBlockSize>(meshlets, ElementHandle{id}).Value());
        std::ranges::sort(addresses);
        UpdatePosedMeshletBlocks(r, owner, addresses);
        std::vector<LodClusterRun> runs;
        for (const auto id : added[i]) {
            const auto primitive = buffers.Meshlets.Get({Resolve<RenderBlockSize>(meshlets, ElementHandle{id}).Value(), 1u})[0].Primitive;
            if (!runs.empty() && runs.back().First + runs.back().Count == id && runs.back().Primitive == primitive) ++runs.back().Count;
            else runs.push_back({id, 1u, primitive, false});
        }
        refits.push_back(EditLodNodes(r, owner, repair.Removed, runs, touched[i]));
    }
    RecordLodNodeRefits(r, chain, refits);
    // The retired clusters and stale groups stay live until the recorded refits complete, and their release clears the owner's dirty groups.
    chain.AfterSubmit([&r, repairs = std::move(repairs)] {
        auto &meshes = r.Context.get<MeshStore>();
        auto &buffers = meshes.Render();
        for (const auto &repair : repairs) {
            auto &owner = EditRecordOf(r, repair.Entity);
            RetireMeshletStorage(r, owner, repair.Removed);
            // The stale groups release the runs their links own and their dirty bits.
            const auto entries = buffers.ClusterGroups.Entries(owner.GroupData);
            std::vector<Range> runs;
            for (const auto group : repair.Closure) {
                const auto address = Resolve<RenderBlockSize>(entries, ElementHandle{group});
                const auto links = buffers.GroupLinks.Get({address.Value(), 1u})[0];
                runs.insert(runs.end(), {Range{links.MemberOffset, links.MemberCount}, Range{links.ProxyOffset, links.ProxyCount}});
                MaskClear(buffers.DirtyGroups, address);
            }
            meshes.ReleasePayloadRuns(owner, RenderPayload::GroupClusters, std::move(runs));
            buffers.ClusterGroups.Erase(owner.GroupData, repair.Closure);
        }
    });
}
