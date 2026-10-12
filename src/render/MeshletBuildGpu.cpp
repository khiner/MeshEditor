#include "render/MeshletBuildGpu.h"
#include "Profile.h"

#include "gpu/MeshletBuildJob.h"
#include "gpu/MeshletBuildPushConstants.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchChunks.h"
#include "metal/Dispatch.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBuffers.h"
#include "render/MeshletOwners.h"
#include "render/MeshletSpatial.h"
#include "state/Scene.h"

#include <algorithm>
#include <array>
#include <bit>
#include <format>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace {
constexpr uint32_t TileSize = 256, RecordWords = sizeof(MeshletRecord) / sizeof(uint32_t);
uint32_t Tiles(uint32_t n) { return n / TileSize + (n % TileSize != 0); }
uint32_t BuildTiles(uint32_t n) { return n / MeshletBuildTileElements + (n % MeshletBuildTileElements != 0); }

struct Layout {
    uint64_t Words{};
    uint32_t Take(uint64_t n) {
        if (n > std::numeric_limits<uint32_t>::max() - Words) throw std::length_error("Meshlet build exceeds GPU scratch address space.");
        const auto first = uint32_t(Words);
        Words += n;
        return first;
    }
};

// The element work bounds of one source's build.
struct SourceWork {
    const MeshStore::Record *Record;
    // An unowned canonical source without seeded elements gathers its triangle, edge or point membership.
    bool Gathers;
    uint32_t Endpoints, MaterialBound;
    uint64_t MaterialBlocks;
};

SourceWork WorkOf(const MeshletBuildSource &source) {
    const auto *record = source.Destination->ExtrasFaces.Count ? nullptr : source.Destination;
    const auto *material_owner = source.Owner ? source.Owner : record;
    const auto material_bound = material_owner ? std::max(material_owner->PrimitiveMaterials.Count, 1u) : 1u;
    return {
        .Record = record,
        .Gathers = !source.Owner && record && source.Elements.Storage.Slot == InvalidSlot,
        .Endpoints = source.Topology == 0 ? 3u : source.Topology == 1 ? 2u :
                                                                        1u,
        .MaterialBound = material_bound,
        // A fragment's triangles name only its bound primitive, whose key the host seeds.
        .MaterialBlocks = source.Owner ? 1u : std::min(uint64_t(WorkDomainBlocks(material_bound)), uint64_t(source.ElementCount)),
    };
}

// Lays out the job's build scratch from its element and primitive counts.
void LayoutJob(Layout &layout, MeshletBuildJob &job) {
    const auto n = job.ElementCount, p = job.PrimitiveCount;
    const uint64_t tile_bound = uint64_t(BuildTiles(n)) + p;
    const uint64_t record_bound = std::min(uint64_t(n), uint64_t(n) / 16u + 2u * p + BuildTiles(n));
    if (tile_bound > UINT32_MAX) throw std::length_error("Meshlet tile count exceeds GPU address space.");
    job.TileBound = uint32_t(tile_bound);
    job.RadixPassCount = 8u + (std::bit_width(p ? p - 1u : 0u) + 3u) / 4u;
    job.StatsOffset = layout.Take(16);
    job.KeysOffset = layout.Take(uint64_t(n) * 2u);
    job.OrderOffset = layout.Take(n);
    job.TempOrderOffset = layout.Take(n);
    job.HistogramOffset = layout.Take(uint64_t(job.BlockCount) * 16u);
    job.DigitTotalsOffset = layout.Take(16);
    job.PrimitiveScratchOffset = layout.Take(uint64_t(p) * 8u);
    job.TileScratchOffset = layout.Take(tile_bound * 4u);
    job.TileCountsOffset = layout.Take(tile_bound * 4u);
    job.RecordScratchOffset = layout.Take(record_bound * RecordWords);
    job.VertexScratchOffset = layout.Take(uint64_t(n) * (job.Topology == 0u ? 3u : job.Topology == 1u ? 2u :
                                                                                                        1u));
}

MeshletBuildPushConstants PushConstants(const MeshStore &meshes) {
    const auto &b = meshes.Render();
    return {
        .VertexRefsSlot = b.MeshletVertexCorners.Buffer.Slot,
        .TriangleIdsSlot = b.MeshletTriangleIds.Buffer.Slot,
        .LocalTrianglesSlot = b.MeshletLocalTriangles.Buffer.Slot,
        .MeshletsSlot = b.Meshlets.Buffer.Slot,
        .PrimitivesSlot = b.Primitives.Buffer.Slot,
        .NodesSlot = b.LodNodes.Buffer.Slot,
        .PrimitiveRoutesSlot = b.PrimitiveRoutes.Buffer.Slot,
        .LodLeavesSlot = b.MeshletLodLeaves.Buffer.Slot,
        .LodParentsSlot = b.LodParents.Buffer.Slot,
        .FaceSharpnessSlot = meshes.Slots().FaceSharpness,
    };
}
} // namespace

uint64_t MeshletBuildScratchWords(const MeshStore &meshes, std::span<const MeshletBuildSource> sources) {
    const auto gather = [](const auto &arena, ElementSetRef set) {
        const uint64_t blocks = set ? arena.Set(set).BlockCount : 0u;
        return ElementWorkWords(arena.Capacity(), blocks) + blocks + SortElementWorkWords(WorkCapacity(arena.Capacity(), blocks));
    };
    uint64_t words = 0u;
    for (const auto &source : sources) {
        const auto work = WorkOf(source);
        if (work.Gathers) words += meshes.WithRenderDomain(*work.Record, source.Topology, gather);
        words += ElementWorkWords(work.MaterialBound, work.MaterialBlocks);
        if (!source.Owner) words += SortElementWorkWords(WorkCapacity(work.MaterialBound, work.MaterialBlocks));
    }
    return words;
}

namespace {
// The words one source's build takes: its membership in the chain's scratch and its build scratch with its primitives at their bound.
uint64_t SourceBuildWords(const MeshStore &meshes, const MeshletBuildSource &source) {
    const auto work = WorkOf(source);
    MeshletBuildJob job{.Topology = source.Topology, .ElementCount = source.ElementCount, .PrimitiveCount = source.Owner ? 1u : work.MaterialBound, .BlockCount = Tiles(source.ElementCount)};
    Layout layout;
    LayoutJob(layout, job);
    return MeshletBuildScratchWords(meshes, std::span{&source, 1u}) + layout.Words;
}

void ValidateSources(const MeshStore &meshes, std::span<const MeshletBuildSource> sources) {
    const auto &render = meshes.Render();
    // Validate every borrowed binding before releasing any destination. A
    // fragment must never own or destroy its parent's primitive/LOD records.
    std::unordered_map<const MeshStore::Record *, uint32_t> destinations;
    for (const auto &source : sources) {
        if (!source.Destination || source.Topology > 2u) throw std::invalid_argument("Meshlet build requires a destination and triangle, line or point topology.");
        auto &mask = destinations[source.Destination];
        const uint32_t topology = 1u << source.Topology;
        if ((mask & topology) || (mask && (source.Owner || source.Destination->ExtrasFaces.Count)) || (mask & 8u))
            throw std::invalid_argument("Meshlet build requires distinct topologies per canonical owner and distinct fragment destinations.");
        mask |= topology | (source.Owner ? 8u : 0u);
    }
    for (const auto &source : sources) {
        if (!source.Owner) {
            if (source.Primitive != InvalidOffset || source.Group != InvalidOffset) throw std::invalid_argument("Meshlet fragment binding requires its render owner.");
            continue;
        }
        const auto &owner = *source.Owner;
        const bool triangles = source.Topology == 0u && source.Elements.Storage.Slot != InvalidSlot;
        const bool elements = (source.Topology == 1u || source.Topology == 2u) && source.Elements.Storage.Slot != InvalidSlot && source.Group == InvalidOffset;
        if ((!triangles && !elements) ||
            owner.ExtrasFaces.Count || owner.StoreId != source.Destination->StoreId ||
            !render.Primitives.Contains(owner.PrimitiveData, ElementHandle{source.Primitive}))
            throw std::invalid_argument("Meshlet fragment requires owned canonical element work and primitive.");
        if (destinations.contains(source.Owner)) throw std::invalid_argument("Meshlet fragment owner cannot also be a build destination.");
        const auto &primitive = render.Primitives.Get({render.Primitives.Address(owner.PrimitiveData, ElementHandle{source.Primitive}).Value(), 1u})[0];
        if (primitive.Topology != source.Topology || meshes.PrimitiveRoute(owner, primitive.PrimitiveIndex, source.Topology) != source.Primitive)
            throw std::invalid_argument("Meshlet fragment primitive has no matching source route.");
        if (source.Group != InvalidOffset) {
            if (!render.ClusterGroups.Contains(owner.GroupData, ElementHandle{source.Group})) throw std::invalid_argument("Meshlet fragment group belongs to another owner.");
            const auto links = render.GroupLinks.Get({render.ClusterGroups.Address(owner.GroupData, ElementHandle{source.Group}).Value(), 1u})[0];
            if (!links.MemberCount || uint64_t(links.MemberOffset) + links.MemberCount > owner.GroupClusters.Values.Count) {
                throw std::invalid_argument("Meshlet fragment group has invalid membership.");
            }
            const auto first = render.GroupClusterIds.Get(owner.GroupClusters.Values)[links.MemberOffset];
            if (!render.Meshlets.Contains(owner.MeshletData, ElementHandle{first})) throw std::invalid_argument("Meshlet fragment group references a foreign meshlet.");
            const auto &member = render.Meshlets.Get({render.Meshlets.Address(owner.MeshletData, ElementHandle{first}).Value(), 1u})[0];
            if (member.GroupIndex != source.Group || member.Primitive != source.Primitive) {
                throw std::invalid_argument("Meshlet fragment group disagrees with its primitive.");
            }
        }
    }
}

uint32_t OutputCount(uint64_t count) {
    if (count > UINT32_MAX) throw std::length_error("Meshlet output exceeds arena address space.");
    return uint32_t(count);
}

void BuildChunk(state::Scene &r, mtl::ComputeChain &chain, std::span<MeshletBuildSource> sources) {
    const profile::CpuScope scope{"MeshletBuild"};
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    const auto &pipelines = GetMeshPipelines(r);
    Layout layout;
    auto &membership = chain.Scratch;
    mtl::Buffer scratch{buffers.Ctx, 0u, SlotType::Buffer, mtl::BufferLifetime::Workspace};
    mtl::Buffer job_buffer{buffers.Ctx, 0u, SlotType::Buffer, mtl::BufferLifetime::Workspace};
    mtl::Buffer tile_buffer{buffers.Ctx, 0u, SlotType::Buffer, mtl::BufferLifetime::Workspace};
    std::vector<ElementWork> element_work, gathered_work;
    std::vector<MeshletBuildJob> jobs;
    // All jobs for a destination are adjacent and publish one allocation together.
    std::vector<Range> destinations;
    for (uint32_t i = 0u; i < sources.size(); ++i) {
        if (i == 0u || sources[i].Destination != sources[i - 1u].Destination) destinations.push_back({i, 1u});
        else ++destinations.back().Count;
    }
    // Fragments already know their primitive counts; canonical builds gather them.
    const bool owned = std::ranges::all_of(sources, [](const auto &source) { return source.Owner != nullptr; });
    membership.ReserveAdditional(MeshletBuildScratchWords(meshes, sources));
    {
        std::vector<MeshStore::Record *> released;
        released.reserve(destinations.size());
        for (const auto group : destinations) released.push_back(sources[group.Offset].Destination);
        meshes.ReleaseRender(released);
    }
    for (auto &source : sources) {
        auto &mb = *source.Destination;
        const auto n = source.ElementCount;
        auto elements = source.Elements;
        const auto work = WorkOf(source);
        const auto *record = work.Record;
        ElementWorkSeedJob seed{};
        if (work.Gathers) {
            seed = meshes.WithRenderDomain(*record, source.Topology, [&](const auto &arena, ElementSetRef set) { return PrepareElementMembershipWork(membership, arena, set); });
            element_work.push_back(seed.Work);
            elements = seed.Work;
        }
        const auto materials = source.Owner ?
            SeedElementWorkHandles(membership, work.MaterialBound, std::array{render.Primitives.Get({render.Primitives.Address(source.Owner->PrimitiveData, ElementHandle{source.Primitive}).Value(), 1u})[0].PrimitiveIndex}) :
            AllocateElementWork(membership, work.MaterialBound, work.MaterialBlocks);
        if (!source.Owner) gathered_work.push_back(materials);
        if (uint64_t(n) * work.Endpoints > UINT32_MAX) throw std::length_error("Meshlet output exceeds arena address space.");
        MeshletBuildJob job{
            .Mesh = BuildMeshRecord(buffers, meshes, mb.StoreId),
            .AuxIndices = mb.ExtrasEdges.Count ? SlotOffset{render.ExtrasEdges.Buffer.Slot, mb.ExtrasEdges.Offset} : SlotOffset{},
            .Topology = source.Topology,
            .ElementCount = n,
            .PrimitiveCount = source.Owner && n ? 1u : 0u,
            .BlockCount = Tiles(n),
            .ElementBlockIds = seed.BlockIds,
            .ElementBlockCount = seed.BlockCount,
            .ElementBlocksSlot = seed.BlocksSlot,
            .ElementOwner = seed.Owner,
            .Elements = elements,
            .Materials = materials,
            .ExistingPrimitive = source.Primitive,
            .ExistingGroup = source.Group,
        };
        jobs.push_back(job);
    }
    job_buffer.SetUsedSize(as_bytes(jobs).size());
    job_buffer.Update(as_bytes(jobs));
    // Each stage dispatches all independent jobs together.
    // Tiles carry only job/local-group metadata.
    // Geometry and sort state stay in GPU arenas.
    enum Domain : uint32_t { MembershipDomain,
                             InputDomain,
                             InitDomain,
                             DigitDomain,
                             JobDomain,
                             TileInitDomain,
                             TileDomain,
                             PrimitiveDomain,
                             DomainCount };
    std::array<Range, DomainCount> domains{};
    std::vector<uvec2> tiles;
    const auto domain = [&](Domain which, auto &&count) {
        const auto first = uint32_t(tiles.size());
        for (uint32_t i = 0u; i < jobs.size(); ++i) {
            const auto groups = count(jobs[i]);
            if (uint64_t(tiles.size()) + groups > UINT32_MAX) throw std::length_error("Meshlet dispatch tiles exceed their address space.");
            for (uint32_t group = 0u; group < groups; ++group) tiles.emplace_back(i, group);
        }
        domains[which] = {first, uint32_t(tiles.size()) - first};
    };
    domain(MembershipDomain, [](const auto &job) { return job.ElementBlockCount; });
    domain(InputDomain, [](const auto &job) { return job.BlockCount; });
    tile_buffer.Update(as_bytes(tiles));
    auto pc = PushConstants(meshes);
    pc.TilesSlot = tile_buffer.Slot;
    pc.JobsSlot = job_buffer.Slot;
    pc.ScratchSlot = scratch.Slot;
    const auto dispatch = [&](MeshPass pass, Domain which, uint32_t parameter = 0u) {
        const auto range = domains[which];
        pc.FirstTile = range.Offset;
        pc.PassParameter = parameter;
        chain.Groups(pipelines[pass], pc, range.Count, pass == MeshPass::MeshletBuildClusters ? MeshletBuildClusterThreads : 256u);
    };
    {
        const profile::CpuScope phase{"MeshletGather"};
        dispatch(MeshPass::MeshletBuildElements, MembershipDomain);
        EncodeSortElementWork(r, chain, element_work);
        dispatch(MeshPass::MeshletBuildMaterials, InputDomain);
        EncodeSortElementWork(r, chain, gathered_work);
        if (!owned) chain.Submit();
    }
    for (const auto work : element_work) CheckElementWork(membership, work);
    for (uint32_t i = 0u; i < jobs.size(); ++i) {
        auto &job = jobs[i];
        if (job.ElementBlockIds.Slot != InvalidSlot) {
            job.ElementCount = ElementWorkCount(membership, job.Elements);
            if (job.ElementCount > sources[i].ElementCount) throw std::logic_error("Gathered meshlet membership exceeds its domain count.");
            job.BlockCount = Tiles(job.ElementCount);
        }
        if (!sources[i].Owner) {
            CheckElementWork(membership, job.Materials);
            job.PrimitiveCount = membership.Get({job.Materials.Storage.Offset + 5u, 1u})[0];
        }
        LayoutJob(layout, job);
    }
    uint64_t triangle_ids = 0u, local_triangles = 0u;
    for (const auto &job : jobs) {
        triangle_ids += job.ElementCount;
        if (job.Topology == 0u) local_triangles += job.ElementCount;
    }
    render.MeshletTriangleIds.ReserveAdditional(triangle_ids);
    render.MeshletLocalTriangles.ReserveAdditional(local_triangles);
    // A fragment's payload runs append to the ranges of its render owner, and a canonical destination's fill fresh ranges of exactly their size.
    const auto payload_owner = [&](Range group) -> MeshStore::Record & { return sources[group.Offset].Owner ? *sources[group.Offset].Owner : *sources[group.Offset].Destination; };
    // Each destination's payload runs, as offsets in its payload owner's ranges.
    struct PayloadRuns {
        Range Triangles, LocalTriangles, Vertices;
    };
    std::vector<PayloadRuns> payload_runs(destinations.size());
    // Gives each job the pool positions of its payload runs and captures them for their GPU writes, once the batch appended every run, since growth moves a range.
    const auto place_payloads = [&] {
        for (uint32_t d = 0u; d < destinations.size(); ++d) {
            const auto &owner = payload_owner(destinations[d]);
            for (uint32_t i = destinations[d].Offset; i < destinations[d].Offset + destinations[d].Count; ++i) {
                jobs[i].TriangleAddress = owner.MeshletTriangles.Values.Offset + jobs[i].TriangleOffset;
                jobs[i].LocalTriangleAddress = owner.MeshletLocalTriangles.Values.Offset + jobs[i].LocalTriangleOffset;
                jobs[i].VertexAddress = owner.MeshletVertices.Values.Offset + jobs[i].VertexOffset;
            }
            const auto &runs = payload_runs[d];
            for (const auto [payload, run] : {std::pair{RenderPayload::TriangleIds, runs.Triangles}, std::pair{RenderPayload::LocalTriangles, runs.LocalTriangles}, std::pair{RenderPayload::VertexCorners, runs.Vertices}})
                WithPayload(render, owner, payload, [&](const auto &pool, const PayloadRange &range) { pool.CaptureWrite({range.Values.Offset + run.Offset, run.Count}); });
        }
    };
    for (uint32_t d = 0u; d < destinations.size(); ++d) {
        const auto group = destinations[d];
        auto &mb = *sources[group.Offset].Destination;
        ++mb.MeshletRevision;
        uint64_t elements = 0u, local = 0u;
        for (uint32_t i = group.Offset; i < group.Offset + group.Count; ++i) {
            elements += jobs[i].ElementCount;
            if (jobs[i].Topology == 0u) local += jobs[i].ElementCount;
            if (!sources[i].Owner && jobs[i].ElementCount) mb.RenderTopologies |= 1u << jobs[i].Topology;
        }
        // An empty published mesh still has a completed render binding.
        if (!sources[group.Offset].Owner && !mb.RenderTopologies) mb.RenderTopologies = 1u << sources[group.Offset].Topology;
        auto &runs = payload_runs[d];
        runs.Triangles = {meshes.AppendPayload(payload_owner(group), RenderPayload::TriangleIds, OutputCount(elements)), OutputCount(elements)};
        runs.LocalTriangles = {meshes.AppendPayload(payload_owner(group), RenderPayload::LocalTriangles, OutputCount(local)), OutputCount(local)};
        uint32_t element = runs.Triangles.Offset, triangle = runs.LocalTriangles.Offset;
        for (uint32_t i = group.Offset; i < group.Offset + group.Count; ++i) {
            auto &job = jobs[i];
            job.TriangleOffset = element;
            job.LocalTriangleOffset = triangle;
            element += job.ElementCount;
            if (job.Topology == 0u) triangle += job.ElementCount;
        }
    }
    place_payloads();
    tiles.clear();
    domain(InputDomain, [](const auto &job) { return job.BlockCount; });
    domain(InitDomain, [](const auto &job) { return Tiles(std::max(16u, job.PrimitiveCount)); });
    domain(DigitDomain, [](const auto &job) { return job.BlockCount ? 16u : 0u; });
    domain(JobDomain, [](const auto &) { return 1u; });
    domain(TileInitDomain, [](const auto &job) { return Tiles(job.TileBound); });
    domain(TileDomain, [](const auto &job) { return job.TileBound; });
    domain(PrimitiveDomain, [](const auto &job) { return job.ExistingPrimitive == InvalidOffset ? Tiles(job.PrimitiveCount) : 0u; });
    tile_buffer.Update(as_bytes(tiles));
    uint32_t radix_passes = 0u;
    for (const auto &job : jobs) radix_passes = std::max(radix_passes, job.RadixPassCount);
    scratch.SetUsedSize(layout.Words * sizeof(uint32_t));
    job_buffer.Update(as_bytes(jobs));
    {
        const profile::CpuScope phase{"MeshletConstruct"};
        // The production build is one submission.
        // Profiling splits a build without an owner into its major GPU phases, so a load regression has an owner.
        const auto profile_phase = [&](std::string_view name) {
            if (!profile::Enabled || owned) return;
            const profile::CpuScope timed{name};
            chain.Submit();
        };
        dispatch(MeshPass::MeshletBuildInit, InitDomain);
        dispatch(MeshPass::MeshletBuildBounds, InputDomain);
        dispatch(MeshPass::MeshletBuildKeys, InputDomain);
        profile_phase("MeshletBuildKeys");
        for (uint32_t shift = 0u; shift < radix_passes * 4u; shift += 4u) {
            dispatch(MeshPass::MeshletBuildHistogram, InputDomain, shift);
            dispatch(MeshPass::MeshletBuildHistogramPrefix, DigitDomain, shift);
            dispatch(MeshPass::MeshletBuildScatter, InputDomain, shift);
        }
        profile_phase("MeshletBuildSort");
        dispatch(MeshPass::MeshletBuildSegments, JobDomain);
        dispatch(MeshPass::MeshletBuildTiles, TileInitDomain);
        profile_phase("MeshletBuildTiles");
        dispatch(MeshPass::MeshletBuildClusters, TileDomain);
        profile_phase("MeshletBuildClusters");
        dispatch(MeshPass::MeshletBuildOffsets, JobDomain);
        chain.Submit();
    }
    uint64_t meshlets = 0u, vertices = 0u, primitives = 0u;
    std::vector<uvec2> counts;
    counts.reserve(jobs.size());
    for (uint32_t i = 0; i < jobs.size(); ++i) {
        const auto &job = jobs[i];
        CheckElementWork(membership, job.Materials);
        const auto results = scratch.GetSpan<uint32_t>({job.StatsOffset + 8u, 2u});
        const auto meshlet_count = results[0], vertex_count = results[1];
        uint64_t element_sum = 0u;
        for (uint32_t p = 0u; p < job.PrimitiveCount; ++p) element_sum += scratch.GetSpan<uint32_t>({job.PrimitiveScratchOffset + p * 8u, 1u})[0];
        if (element_sum != job.ElementCount) throw std::runtime_error(std::format("GPU meshlet primitive counts changed during construction: expected {}, got {}, scratch words {}, primitive offset {}.", job.ElementCount, element_sum, layout.Words, job.PrimitiveScratchOffset));
        if (scratch.GetSpan<uint32_t>({job.StatsOffset + 13u, 1u})[0]) throw std::invalid_argument("Meshlet input references a primitive absent from its metadata.");
        if (meshlet_count > job.ElementCount || vertex_count > uint64_t(job.ElementCount) * 3u) throw std::runtime_error("GPU meshlet counts exceed construction bounds.");
        counts.emplace_back(meshlet_count, vertex_count);
        meshlets += meshlet_count;
        vertices += vertex_count;
        if (!sources[i].Owner) primitives += job.PrimitiveCount;
    }
    render.Meshlets.ReserveAdditional(meshlets);
    render.MeshletLodLeaves.ReserveAdditional(meshlets);
    render.MeshletSpatialNodes.ReserveAdditional(meshlets);
    render.MeshletVertexCorners.ReserveAdditional(vertices);
    render.Primitives.ReserveAdditional(primitives);
    render.LodNodes.ReserveAdditional(primitives);
    render.LodParents.ReserveAdditional(primitives);
    const auto capture = [](const mtl::Buffer &buffer, uint64_t stride, RenderRun run) { buffer.CaptureWrite(uint64_t(run.Address.Value()) * stride, uint64_t(run.Count) * stride); };
    std::vector<RenderRun> node_runs(destinations.size());
    for (uint32_t d = 0u; d < destinations.size(); ++d) {
        const auto group = destinations[d];
        auto &mb = *sources[group.Offset].Destination;
        const bool canonical = !sources[group.Offset].Owner;
        uint64_t clusters = 0u, corners = 0u, primitive_count = 0u;
        uint32_t routes = mb.ExtrasFaces.Count ? 1u : std::max(mb.PrimitiveMaterials.Count, 1u);
        for (uint32_t i = group.Offset; i < group.Offset + group.Count; ++i) {
            clusters += counts[i].x;
            corners += counts[i].y;
            if (canonical) {
                primitive_count += jobs[i].PrimitiveCount;
                ForEachWorkElement(membership, jobs[i].Materials, [&](uint32_t key) { routes = std::max(routes, key + 1u); });
            }
        }
        // A fragment's clusters join its owner's meshlet set, and a canonical destination takes fresh dense sets.
        auto &meshlet_set = canonical ? mb.MeshletData : sources[group.Offset].Owner->MeshletData;
        const auto meshlet_run = render.Insert(RenderSet::Meshlets, meshlet_set, OutputCount(clusters));
        mb.Meshlets = {meshlet_run.Handle.Value, meshlet_run.Count};
        mb.Level0Count = mb.Meshlets.Count;
        auto &runs = payload_runs[d];
        runs.Vertices = {meshes.AppendPayload(payload_owner(group), RenderPayload::VertexCorners, OutputCount(corners)), OutputCount(corners)};
        RenderRun primitive_run{}, node_run{};
        if (canonical) {
            primitive_run = render.Insert(RenderSet::Primitives, mb.PrimitiveData, OutputCount(primitive_count));
            node_run = render.Insert(RenderSet::Nodes, mb.NodeData, OutputCount(primitive_count));
            render.MeshRecords.GetMutable({mb.StoreId, 1u})[0] = jobs[group.Offset].Mesh;
            meshes.ReservePrimitiveRoutes(mb, routes);
        }
        node_runs[d] = node_run;
        uint32_t cluster = 0u, vertex = runs.Vertices.Offset, primitive = 0u;
        for (uint32_t i = group.Offset; i < group.Offset + group.Count; ++i) {
            auto &job = jobs[i];
            job.MeshletAddress = (meshlet_run.Address + cluster).Value();
            job.MeshletHandle = meshlet_run.Handle.Value + cluster;
            job.VertexOffset = vertex;
            job.PrimitiveAddress = (primitive_run.Address + primitive).Value();
            job.PrimitiveHandle = primitive_run.Handle.Value + primitive;
            job.NodeAddress = (node_run.Address + primitive).Value();
            job.NodeHandle = node_run.Handle.Value + primitive;
            job.PrimitiveRoutes = mb.PrimitiveRoutes.Offset;
            cluster += counts[i].x;
            vertex += counts[i].y;
            if (canonical) primitive += job.PrimitiveCount;
        }
        capture(render.Meshlets.Buffer, sizeof(MeshletRecord), meshlet_run);
        capture(render.Primitives.Buffer, sizeof(PrimitiveRecord), primitive_run);
        capture(render.LodNodes.Buffer, sizeof(LodNode), node_run);
        capture(render.MeshletLodLeaves.Buffer, sizeof(uint32_t), meshlet_run);
        capture(render.LodParents.Buffer, sizeof(uint32_t), node_run);
    }
    place_payloads();
    job_buffer.Update(as_bytes(jobs));
    {
        const profile::CpuScope phase{"MeshletPublish"};
        dispatch(MeshPass::MeshletBuildEmit, TileDomain);
        dispatch(MeshPass::MeshletBuildPrimitives, PrimitiveDomain);
    }
    // The chain holds every build workspace through the submit that completes its readers.
    chain.Retain(std::move(scratch));
    chain.Retain(std::move(job_buffer));
    chain.Retain(std::move(tile_buffer));
    if (owned) return;
    // A canonical source that is its own render owner publishes the owners of the element blocks its work names.
    chain.Concurrent([&] {
        for (uint32_t i = 0u; i < sources.size(); ++i) {
            const auto &source = sources[i];
            if (source.Owner || source.Destination->ExtrasFaces.Count) continue;
            std::vector<uint32_t> blocks;
            ForEachWorkBlock(membership, jobs[i].Elements, [&](uint32_t local, auto) { blocks.push_back(local); });
            std::ranges::sort(blocks);
            PublishMeshletOwners(r, chain, *source.Destination, source.Topology, std::array{Range{jobs[i].MeshletHandle, counts[i].x}}, blocks);
        }
    });
    chain.Submit();
    const profile::CpuScope membership_scope{"MeshletMembership"};
    // Each populated traversal leaf of a new owner lists its cluster run.
    // Edit fragments join their owner's traversal leaves when the owner adopts them.
    std::vector<MeshStore::Record *> spatial;
    for (uint32_t d = 0u; d < destinations.size(); ++d) {
        const auto &source = sources[destinations[d].Offset];
        if (source.Owner) continue;
        auto &mb = *source.Destination;
        const auto nodes = node_runs[d];
        for (uint32_t n = 0u; n < nodes.Count; ++n) {
            auto node = render.LodNodes.Get({(nodes.Address + n).Value(), 1u})[0];
            if (node.ChildCount || !node.MeshletCount) continue;
            if (node.Members != InvalidOffset) throw std::logic_error("LOD leaf already owns meshlet membership.");
            std::vector<uint32_t> members(node.MeshletCount);
            std::iota(members.begin(), members.end(), node.FirstMeshlet);
            node.Members = render.ListMembers(mb.MemberData, members);
            render.LodNodes.GetMutable({(nodes.Address + n).Value(), 1u})[0] = node;
        }
        if (!mb.ExtrasFaces.Count && (mb.RenderTopologies & 1u)) spatial.push_back(&mb);
    }
    BuildMeshletSpatial(r, spatial);
    buffers.Ctx.ReclaimRetiredBuffers();
}
} // namespace

void BuildGpuMeshlets(state::Scene &r, mtl::ComputeChain &chain, std::span<MeshletBuildSource> sources) {
    if (sources.empty()) return;
    const auto &meshes = r.Context.get<const MeshStore>();
    ValidateSources(meshes, sources);
    // Keep every topology of an owner in the same chunk: releasing or publishing
    // that owner again in a later chunk would discard its earlier outputs.
    struct Group {
        std::vector<MeshletBuildSource> Sources;
        uint64_t Words{};
    };
    std::vector<Group> groups;
    std::unordered_map<const MeshStore::Record *, uint32_t> destinations;
    for (const auto &source : sources) {
        const auto [it, inserted] = destinations.try_emplace(source.Destination, uint32_t(groups.size()));
        if (inserted) groups.emplace_back();
        auto &group = groups[it->second];
        group.Sources.push_back(source);
        group.Words += SourceBuildWords(meshes, source);
    }
    const auto split = ChunkByScratch(uint32_t(groups.size()), ScratchWordBudget, [&](uint32_t i) {
        return uint32_t(std::min<uint64_t>(groups[i].Words, UINT32_MAX));
    });
    std::vector<MeshletBuildSource> chunk_sources;
    for (const auto chunk : split.Chunks) {
        chunk_sources.clear();
        for (uint32_t i = chunk.Offset; i < chunk.Offset + chunk.Count; ++i)
            chunk_sources.insert(chunk_sources.end(), groups[i].Sources.begin(), groups[i].Sources.end());
        BuildChunk(r, chain, chunk_sources);
    }
}

uint32_t EnsureMeshletPrimitive(state::Scene &r, MeshStore::Record &owner, uint32_t source_primitive, uint32_t topology) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    if (owner.ExtrasFaces.Count || topology > 2u) throw std::invalid_argument("Primitive route requires a canonical render owner and topology.");
    if (source_primitive >= std::max(owner.PrimitiveMaterials.Count, 1u)) throw std::out_of_range("Primitive route exceeds the mesh material domain.");
    if (const auto existing = meshes.PrimitiveRoute(owner, source_primitive, topology); existing != InvalidOffset) {
        if (!render.Primitives.Contains(owner.PrimitiveData, ElementHandle{existing})) throw std::logic_error("Primitive route names a foreign render primitive.");
        return existing;
    }
    const auto primitive = render.Insert(RenderSet::Primitives, owner.PrimitiveData, 1u);
    const auto node = render.Insert(RenderSet::Nodes, owner.NodeData, 1u);
    render.Primitives.GetMutable({primitive.Address.Value(), 1u})[0] = {
        .PrimitiveIndex = source_primitive,
        .Topology = topology,
        .PrimitiveMaterialOffset = OffsetOrInvalid(owner.PrimitiveMaterials),
        .LodRootNode = node.Handle.Value,
        .LodFinestNode = node.Handle.Value,
    };
    render.LodNodes.GetMutable({node.Address.Value(), 1u})[0] = {.Error = std::numeric_limits<float>::infinity(), .Members = InvalidOffset};
    render.LodParents.GetMutable({node.Address.Value(), 1u})[0] = InvalidOffset;
    meshes.ReservePrimitiveRoutes(owner, std::max(owner.PrimitiveMaterials.Count, 1u));
    render.PrimitiveRoutes.GetMutable({owner.PrimitiveRoutes.Offset + 3u * source_primitive + topology, 1u})[0] = primitive.Handle.Value;
    ++owner.MeshletRevision;
    return primitive.Handle.Value;
}
