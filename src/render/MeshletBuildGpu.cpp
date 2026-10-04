#include "render/MeshletBuildGpu.h"
#include "Profile.h"

#include "gpu/MeshletBuildJob.h"
#include "gpu/MeshletBuildPushConstants.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/ScratchChunks.h"
#include "metal/Dispatch.h"
#include "render/GpuBuffers.h"
#include "render/MeshBuffers.h"
#include "render/ElementWorkOps.h"
#include "render/MeshletOwners.h"
#include "render/MeshletSpatial.h"
#include "state/Scene.h"

#include <algorithm>
#include <array>
#include <bit>
#include <format>
#include <limits>
#include <stdexcept>
#include <unordered_set>

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

SourceWork WorkOf(const MeshStore &meshes, const MeshletBuildSource &source) {
    const auto *record = source.StoreId == InvalidOffset ? nullptr : &meshes.Get(source.StoreId);
    const auto material_bound = record ? std::max(record->PrimitiveMaterials.Count,1u) : 1u;
    return {
        .Record = record,
        .Gathers = !source.Owner && record && source.Elements.Storage.Slot == InvalidSlot,
        .Endpoints = source.Topology == 0 ? 3u : source.Topology == 1 ? 2u : 1u,
        .MaterialBound = material_bound,
        // A fragment's triangles name only its bound primitive, whose key the host seeds.
        .MaterialBlocks = source.Owner ? 1u : std::min(uint64_t(WorkDomainBlocks(material_bound)),uint64_t(source.ElementCount)),
    };
}

// Lays out the job's build scratch from its element and primitive counts.
void LayoutJob(Layout &layout, MeshletBuildJob &job) {
    const auto n = job.ElementCount, p = job.PrimitiveCount;
    const uint64_t tile_bound = uint64_t(BuildTiles(n)) + p;
    const uint64_t record_bound = std::min(uint64_t(n),uint64_t(n)/16u+2u*p+BuildTiles(n));
    if (tile_bound > UINT32_MAX) throw std::length_error("Meshlet tile count exceeds GPU address space.");
    job.TileBound = uint32_t(tile_bound);
    job.RadixPassCount = 8u+(std::bit_width(p ? p-1u : 0u)+3u)/4u;
    job.StatsOffset = layout.Take(16);
    job.KeysOffset = layout.Take(uint64_t(n)*2u);
    job.OrderOffset = layout.Take(n);
    job.TempOrderOffset = layout.Take(n);
    job.HistogramOffset = layout.Take(uint64_t(job.BlockCount)*16u);
    job.DigitTotalsOffset = layout.Take(16);
    job.PrimitiveScratchOffset = layout.Take(uint64_t(p)*8u);
    job.TileScratchOffset = layout.Take(tile_bound*4u);
    job.TileCountsOffset = layout.Take(tile_bound*4u);
    job.RecordScratchOffset = layout.Take(record_bound*RecordWords);
    job.VertexScratchOffset = layout.Take(uint64_t(n)*(job.Topology == 0u ? 3u : job.Topology == 1u ? 2u : 1u));
}

MeshletBuildPushConstants PushConstants(const GpuBuffers &b, const MeshStore &meshes) {
    return {
        .VertexRefsSlot = b.MeshletVertexCorners.Buffer.Slot,
        .TriangleIdsSlot = b.MeshletTriangleIds.Buffer.Slot,
        .LocalTrianglesSlot = b.MeshletLocalTriangles.Buffer.Slot,
        .MeshletsSlot = b.Meshlets.Buffer.Slot,
        .PrimitivesSlot = b.Primitives.Buffer.Slot,
        .NodesSlot = b.LodNodes.Buffer.Slot,
        .PrimitiveRoutesSlot = b.PrimitiveRoutes.Buffer.Slot,
        .LodLeavesSlot = b.MeshletLodLeaves.Buffer.Slot, .LodParentsSlot = b.LodParents.Buffer.Slot,
        .CornerSectors = meshes.Slots().CornerSector,
        .FaceSharpnessSlot = meshes.Slots().FaceSharpness,
    };
}
} // namespace

uint64_t MeshletBuildScratchWords(const MeshStore &meshes, std::span<const MeshletBuildSource> sources) {
    const auto gather = [](const auto &arena, ElementSetRef set) {
        const uint64_t blocks = set ? arena.Set(set).BlockCount : 0u;
        return ElementWorkWords(arena.Capacity(),blocks) + blocks + SortElementWorkWords(WorkCapacity(arena.Capacity(),blocks));
    };
    uint64_t words = 0u;
    for (const auto &source : sources) {
        const auto work = WorkOf(meshes,source);
        if (work.Gathers) words += source.Topology == 0u ? gather(meshes.Arenas().Triangles,work.Record->TriangleData) :
            source.Topology == 1u ? gather(meshes.Arenas().EdgeHalfedges,work.Record->EdgeData) : gather(meshes.Arenas().Vertices,work.Record->Vertices);
        words += ElementWorkWords(work.MaterialBound,work.MaterialBlocks);
        if (!source.Owner) words += SortElementWorkWords(WorkCapacity(work.MaterialBound,work.MaterialBlocks));
    }
    return words;
}

namespace {
// The words one source's build takes: its membership in the chain's scratch and its build scratch with its primitives at their bound.
uint64_t SourceBuildWords(const MeshStore &meshes, const MeshletBuildSource &source) {
    const auto work = WorkOf(meshes,source);
    MeshletBuildJob job{.Topology = source.Topology, .ElementCount = source.ElementCount,
        .PrimitiveCount = source.Owner ? 1u : work.MaterialBound, .BlockCount = Tiles(source.ElementCount)};
    Layout layout;
    LayoutJob(layout, job);
    return MeshletBuildScratchWords(meshes,std::span{&source,1u}) + layout.Words;
}

void BuildChunk(state::Scene &r, mtl::ComputeChain &chain, std::span<MeshletBuildSource> sources) {
    const profile::CpuScope scope{"MeshletBuild"};
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    Layout layout;
    auto &membership = chain.Scratch;
    mtl::Buffer scratch{buffers.Ctx,0u,SlotType::Buffer,mtl::BufferLifetime::Workspace};
    mtl::Buffer job_buffer{buffers.Ctx,0u,SlotType::Buffer,mtl::BufferLifetime::Workspace};
    mtl::Buffer tile_buffer{buffers.Ctx,0u,SlotType::Buffer,mtl::BufferLifetime::Workspace};
    std::vector<ElementWorkSeedJob> seeds;
    std::vector<ElementWork> element_work, gathered_work;
    std::vector<MeshletBuildJob> jobs;
    // Validate every borrowed binding before releasing any destination. A
    // fragment must never own or destroy its parent's primitive/LOD records.
    std::unordered_set<const MeshBuffers *> destinations;
    // A batch of fragments knows each count its layout needs, so only a batch with a new owner reads its gathered counts.
    const bool owned = std::ranges::all_of(sources,[](const auto &source) { return source.Owner != nullptr; });
    for (const auto &source : sources)
        if (!source.Destination || !destinations.insert(source.Destination).second) {
            throw std::invalid_argument("Meshlet build destinations must be distinct and non-null.");
        }
    for (const auto &source : sources) {
        if (!source.Owner) {
            if (source.Primitive != InvalidOffset || source.Group != InvalidOffset) throw std::invalid_argument("Meshlet fragment binding requires its render owner.");
            continue;
        }
        const auto &owner = *source.Owner;
        const bool triangles=source.Topology==0u && source.Mesh.TriangleSlot!=InvalidSlot &&
            source.Elements.Storage.Slot!=InvalidSlot;
        const bool elements=(source.Topology==1u || source.Topology==2u) && source.Mesh.TriangleSlot==InvalidSlot &&
            source.Elements.Storage.Slot!=InvalidSlot && source.Group==InvalidOffset;
        if ((!triangles && !elements) ||
            source.StoreId == InvalidOffset || owner.StoreId != source.StoreId ||
            !buffers.ActiveMeshlets.Contains(owner.PrimitiveRoot,source.Primitive))
            throw std::invalid_argument("Meshlet fragment requires owned canonical element work and primitive.");
        if (destinations.contains(source.Owner)) throw std::invalid_argument("Meshlet fragment owner cannot also be a build destination.");
        const auto &primitive = buffers.Primitives.Get({source.Primitive,1u})[0];
        if (buffers.PrimitiveRoute(owner,primitive.PrimitiveIndex) != source.Primitive) throw std::invalid_argument("Meshlet fragment primitive has no source route.");
        if (source.Group != InvalidOffset) {
            if (!buffers.ActiveMeshlets.Contains(owner.GroupRoot,source.Group)) throw std::invalid_argument("Meshlet fragment group belongs to another owner.");
            const auto links = buffers.GroupLinks.Get({source.Group,1u})[0];
            if (!links.MemberCount || uint64_t(links.MemberOffset)+links.MemberCount > buffers.GroupClusterIds.Buffer.Count<uint32_t>()) {
                throw std::invalid_argument("Meshlet fragment group has invalid membership.");
            }
            const auto first = buffers.GroupClusterIds.Get({links.MemberOffset,1u})[0];
            if (!buffers.ActiveMeshlets.Contains(owner.MeshletRoot,first)) throw std::invalid_argument("Meshlet fragment group references a foreign meshlet.");
            const auto &member = buffers.Meshlets.Get({first,1u})[0];
            if (member.GroupIndex != source.Group || member.Primitive != source.Primitive) {
                throw std::invalid_argument("Meshlet fragment group disagrees with its primitive.");
            }
        }
    }
    membership.ReserveAdditional(MeshletBuildScratchWords(meshes,sources));
    uint64_t triangle_ids=0u, local_triangles=0u;
    for (const auto &source : sources) {
        triangle_ids+=source.ElementCount;
        if (source.Topology==0u) local_triangles+=uint64_t(source.ElementCount)*3u;
    }
    buffers.MeshletTriangleIds.ReserveAdditional(triangle_ids);
    buffers.MeshletLocalTriangles.ReserveAdditional(local_triangles);
    {
        std::vector<MeshBuffers *> released;
        released.reserve(sources.size());
        for (const auto &source : sources) released.push_back(source.Destination);
        buffers.ReleaseMeshlets(released);
    }
    for (auto &source : sources) {
        auto &mb = *source.Destination;
        mb.StoreId = source.StoreId; ++mb.MeshletRevision;
        if (!source.Owner) mb.RenderTopology=source.Topology;
        const auto n = source.ElementCount;
        auto elements = source.Elements;
        const auto work = WorkOf(meshes,source);
        const auto *record = work.Record;
        if (source.Mesh.TriangleSlot != InvalidSlot && !record) throw std::invalid_argument("Canonical meshlet input requires its store ID.");
        if (work.Gathers) {
            const auto seed = source.Topology == 0u ? PrepareElementMembershipWork(membership,meshes.Arenas().Triangles,record->TriangleData) :
                source.Topology == 1u ? PrepareElementMembershipWork(membership,meshes.Arenas().EdgeHalfedges,record->EdgeData) :
                PrepareElementMembershipWork(membership,meshes.Arenas().Vertices,record->Vertices);
            seeds.push_back(seed); element_work.push_back(seed.Work); elements = seed.Work;
        }
        const auto materials = source.Owner ?
            SeedElementWorkHandles(membership,work.MaterialBound,std::array{buffers.Primitives.Get({source.Primitive,1u})[0].PrimitiveIndex}) :
            AllocateElementWork(membership,work.MaterialBound,work.MaterialBlocks);
        if (!source.Owner) gathered_work.push_back(materials);
        if (uint64_t(n) * work.Endpoints > UINT32_MAX) throw std::length_error("Meshlet output exceeds arena address space.");
        mb.MeshletTriangles = buffers.MeshletTriangleIds.Allocate(n);
        mb.MeshletLocalTriangles = buffers.MeshletLocalTriangles.Allocate(source.Topology == 0 ? n * 3u : 0u);
        if (n) buffers.MeshletTriangleIds.Buffer.CaptureWrite(uint64_t(mb.MeshletTriangles.Offset)*4u,uint64_t(n)*4u);
        if (mb.MeshletLocalTriangles.Count) buffers.MeshletLocalTriangles.Buffer.CaptureWrite(mb.MeshletLocalTriangles.Offset,mb.MeshletLocalTriangles.Count);
        MeshletBuildJob job{
            .Mesh = source.Mesh,
            .AuxIndices = source.AuxIndices,
            .Topology = source.Topology,
            .ElementCount = n,
            .PrimitiveCount = source.Owner && n ? 1u : 0u,
            .BlockCount = Tiles(n),
            .TriangleOffset = mb.MeshletTriangles.Offset,
            .LocalTriangleOffset = mb.MeshletLocalTriangles.Offset,
            .Elements = elements, .Materials = materials,
            .ExistingPrimitive = source.Primitive, .ExistingGroup = source.Group,
        };
        jobs.push_back(job);
    }
    job_buffer.SetUsedSize(as_bytes(jobs).size());
    job_buffer.Update(as_bytes(jobs));
    // Each stage dispatches all independent jobs together.
    // Tiles carry only job/local-group metadata.
    // Geometry and sort state stay in GPU arenas.
    enum Domain : uint32_t { InputDomain, InitDomain, DigitDomain, JobDomain, TileInitDomain, TileDomain, PrimitiveDomain, DomainCount };
    std::array<Range,DomainCount> domains{};
    std::vector<uvec2> tiles;
    const auto domain=[&](Domain which,auto &&count) {
        const auto first=uint32_t(tiles.size());
        for (uint32_t i=0u;i<jobs.size();++i) {
            const auto groups=count(jobs[i]);
            if (uint64_t(tiles.size())+groups>UINT32_MAX) throw std::length_error("Meshlet dispatch tiles exceed their address space.");
            for (uint32_t group=0u;group<groups;++group) tiles.emplace_back(i,group);
        }
        domains[which]={first,uint32_t(tiles.size())-first};
    };
    domain(InputDomain,[](const auto &job) { return job.BlockCount; });
    tile_buffer.Update(as_bytes(tiles));
    auto pc = PushConstants(buffers, meshes);
    pc.TilesSlot=tile_buffer.Slot;
    pc.JobsSlot = job_buffer.Slot;
    pc.ScratchSlot = scratch.Slot;
    const auto dispatch = [&](MeshPass pass, Domain which, uint32_t parameter = 0u) {
        const auto range=domains[which];
        pc.FirstTile=range.Offset;
        pc.PassParameter = parameter;
        chain.Groups(pipelines[pass],pc,range.Count,pass == MeshPass::MeshletBuildClusters ? MeshletBuildClusterThreads : 256u);
    };
    {
        const profile::CpuScope phase{"MeshletGather"};
        EncodeElementMembershipWork(r,chain,seeds);
        EncodeSortElementWork(r,chain,element_work);
        dispatch(MeshPass::MeshletBuildMaterials,InputDomain);
        EncodeSortElementWork(r,chain,gathered_work);
        if (!owned) chain.Submit();
    }
    for (const auto work : element_work) CheckElementWork(membership,work);
    for (uint32_t i = 0u; i < jobs.size(); ++i) {
        auto &job = jobs[i];
        if (!sources[i].Owner) {
            CheckElementWork(membership,job.Materials);
            job.PrimitiveCount = membership.Get({job.Materials.Storage.Offset+5u,1u})[0];
        }
        LayoutJob(layout, job);
    }
    domain(InitDomain,[](const auto &job) { return Tiles(std::max(16u,job.PrimitiveCount)); });
    domain(DigitDomain,[](const auto &job) { return job.BlockCount ? 16u : 0u; });
    domain(JobDomain,[](const auto &) { return 1u; });
    domain(TileInitDomain,[](const auto &job) { return Tiles(job.TileBound); });
    domain(TileDomain,[](const auto &job) { return job.TileBound; });
    domain(PrimitiveDomain,[](const auto &job) { return job.ExistingPrimitive==InvalidOffset ? Tiles(job.PrimitiveCount) : 0u; });
    tile_buffer.Update(as_bytes(tiles));
    uint32_t radix_passes=0u;
    for (const auto &job:jobs) radix_passes=std::max(radix_passes,job.RadixPassCount);
    scratch.SetUsedSize(layout.Words*sizeof(uint32_t));
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
        dispatch(MeshPass::MeshletBuildInit,InitDomain);
        dispatch(MeshPass::MeshletBuildBounds,InputDomain);
        dispatch(MeshPass::MeshletBuildKeys,InputDomain);
        profile_phase("MeshletBuildKeys");
        for (uint32_t shift=0u;shift<radix_passes*4u;shift+=4u) {
            dispatch(MeshPass::MeshletBuildHistogram,InputDomain,shift);
            dispatch(MeshPass::MeshletBuildHistogramPrefix,DigitDomain,shift);
            dispatch(MeshPass::MeshletBuildScatter,InputDomain,shift);
        }
        profile_phase("MeshletBuildSort");
        dispatch(MeshPass::MeshletBuildSegments,JobDomain);
        dispatch(MeshPass::MeshletBuildTiles,TileInitDomain);
        profile_phase("MeshletBuildTiles");
        dispatch(MeshPass::MeshletBuildClusters,TileDomain);
        profile_phase("MeshletBuildClusters");
        dispatch(MeshPass::MeshletBuildOffsets,JobDomain);
        chain.Submit();
    }
    uint64_t meshlets=0u, vertices=0u, primitives=0u, records=0u;
    for (uint32_t i = 0; i < jobs.size(); ++i) {
        const auto results = scratch.GetSpan<uint32_t>({jobs[i].StatsOffset + 8u, 2u});
        meshlets+=results[0];
        vertices+=results[1];
        if (!sources[i].Owner) {
            primitives+=jobs[i].PrimitiveCount;
            ++records;
        }
    }
    buffers.Meshlets.ReserveAdditional(meshlets);
    buffers.MeshletLodLeaves.ReserveAdditional(meshlets);
    buffers.MeshletSpatialNodes.ReserveAdditional(meshlets);
    buffers.MeshletVertexCorners.ReserveAdditional(vertices);
    buffers.Primitives.ReserveAdditional(primitives);
    buffers.LodNodes.ReserveAdditional(primitives);
    buffers.LodParents.ReserveAdditional(primitives);
    buffers.MeshRecords.ReserveAdditional(records);
    for (uint32_t i = 0; i < jobs.size(); ++i) {
        auto &job = jobs[i];
        auto &mb = *sources[i].Destination;
        CheckElementWork(membership,job.Materials);
        const auto results = scratch.GetSpan<uint32_t>({job.StatsOffset + 8u, 2u});
        const auto meshlet_count = results[0], vertex_count = results[1];
        uint64_t element_sum=0u;
        for (uint32_t p=0u;p<job.PrimitiveCount;++p) element_sum+=scratch.GetSpan<uint32_t>({job.PrimitiveScratchOffset+p*8u,1u})[0];
        if (element_sum!=job.ElementCount) throw std::runtime_error(std::format("GPU meshlet primitive counts changed during construction: expected {}, got {}, scratch words {}, primitive offset {}.",job.ElementCount,element_sum,layout.Words,job.PrimitiveScratchOffset));
        if (scratch.GetSpan<uint32_t>({job.StatsOffset+13u,1u})[0]) throw std::invalid_argument("Meshlet input references a primitive absent from its metadata.");
        if (meshlet_count > job.ElementCount || vertex_count > uint64_t(job.ElementCount) * 3u) throw std::runtime_error("GPU meshlet counts exceed construction bounds.");
        mb.Meshlets = buffers.AllocateMeshlets(meshlet_count);
        mb.Level0Count = meshlet_count;
        mb.MeshletVertices = buffers.MeshletVertexCorners.Allocate(vertex_count);
        if (!sources[i].Owner) {
            mb.Primitives = buffers.Primitives.Allocate(job.PrimitiveCount);
            mb.LodNodes = buffers.LodNodes.Allocate(job.PrimitiveCount);
            mb.MeshRecord = buffers.MeshRecords.Allocate(std::span<const MeshRecord>{&sources[i].Mesh, 1});
        }
        buffers.LodParents.Mirror(mb.LodNodes);
        job.VertexOffset = mb.MeshletVertices.Offset;
        job.MeshletOffset = mb.Meshlets.Offset;
        job.PrimitiveOffset = mb.Primitives.Offset;
        job.NodeOffset = mb.LodNodes.Offset;
        // The routes cover every referenced source primitive before the GPU writes them.
        if (!sources[i].Owner) {
            uint32_t routes = sources[i].StoreId == InvalidOffset ? 1u : std::max(meshes.Get(sources[i].StoreId).PrimitiveMaterials.Count, 1u);
            ForEachWorkElement(membership,job.Materials,[&](uint32_t key) { routes = std::max(routes, key + 1u); });
            buffers.ReservePrimitiveRoutes(mb, routes);
        }
        job.PrimitiveRoutes = mb.PrimitiveRoutes.Offset;
    }
    for (const auto &source : sources) {
        const auto &mb = *source.Destination;
        buffers.Meshlets.CaptureWrite(mb.Meshlets); buffers.MeshletTriangleIds.CaptureWrite(mb.MeshletTriangles);
        buffers.MeshletVertexCorners.CaptureWrite(mb.MeshletVertices); buffers.MeshletLocalTriangles.CaptureWrite(mb.MeshletLocalTriangles);
        buffers.Primitives.CaptureWrite(mb.Primitives); buffers.LodNodes.CaptureWrite(mb.LodNodes);
        buffers.MeshletLodLeaves.CaptureWrite(mb.Meshlets); buffers.LodParents.CaptureWrite(mb.LodNodes);
    }
    job_buffer.Update(as_bytes(jobs));
    {
        const profile::CpuScope phase{"MeshletPublish"};
        dispatch(MeshPass::MeshletBuildEmit,TileDomain);
        dispatch(MeshPass::MeshletBuildPrimitives,PrimitiveDomain);
    }
    // The chain holds every build workspace through the submit that completes its readers.
    chain.Retain(std::move(scratch)); chain.Retain(std::move(job_buffer)); chain.Retain(std::move(tile_buffer));
    if (owned) return;
    // A canonical source that is its own render owner publishes the owners of the element blocks its work names.
    chain.Concurrent([&] {
        for (uint32_t i = 0u; i < sources.size(); ++i) {
            const auto &source = sources[i];
            if (source.Owner || source.StoreId == InvalidOffset) continue;
            std::vector<uint32_t> blocks;
            ForEachWorkBlock(membership,jobs[i].Elements,[&](uint32_t block,auto) { blocks.push_back(block); });
            std::ranges::sort(blocks);
            PublishMeshletOwners(r,chain,*source.Destination,std::array{source.Destination->Meshlets},blocks);
        }
    });
    chain.Submit();
    const profile::CpuScope membership_scope{"MeshletMembership"};
    // A new owner publishes its clusters, primitives and nodes, and each populated traversal leaf its cluster run.
    // Edit fragments remain provisional ranges until their canonical owner adopts them.
    std::vector<MeshletIndexEdit> ownership;
    std::vector<uint32_t> owned_sources, leaves;
    for (uint32_t i=0u;i<sources.size();++i) {
        const auto &source=sources[i];
        if (source.Owner) continue;
        owned_sources.push_back(i);
        ownership.push_back({.Insert=source.Destination->Meshlets});
        ownership.push_back({.Insert=source.Destination->Primitives});
        ownership.push_back({.Insert=source.Destination->LodNodes});
    }
    for (const auto i : owned_sources) {
        const auto nodes=sources[i].Destination->LodNodes;
        for (uint32_t n=0u;n<nodes.Count;++n) {
            const auto &node=buffers.LodNodes.Get({nodes.Offset+n,1u})[0];
            if (node.ChildCount || !node.MeshletCount) continue;
            if (node.MeshletRoot!=InvalidOffset) throw std::logic_error("LOD leaf already owns meshlet membership.");
            leaves.push_back(nodes.Offset+n);
            ownership.push_back({.Insert={node.FirstMeshlet,node.MeshletCount}});
        }
    }
    if (!ownership.empty()) buffers.ActiveMeshlets.Update(ownership);
    for (uint32_t j=0u;j<leaves.size();++j) buffers.LodNodes.GetMutable({leaves[j],1u})[0].MeshletRoot=ownership[owned_sources.size()*3u+j].Root;
    std::vector<MeshBuffers *> spatial;
    for (uint32_t j=0u;j<owned_sources.size();++j) {
        const auto i=owned_sources[j];
        sources[i].Destination->MeshletRoot = ownership[j*3u].Root;
        sources[i].Destination->PrimitiveRoot = ownership[j*3u+1u].Root;
        sources[i].Destination->NodeRoot = ownership[j*3u+2u].Root;
        if (sources[i].StoreId!=InvalidOffset && sources[i].Topology==0u) spatial.push_back(sources[i].Destination);
    }
    BuildMeshletSpatial(r,spatial);
    buffers.Ctx.ReclaimRetiredBuffers();
}
} // namespace

void BuildGpuMeshlets(state::Scene &r, mtl::ComputeChain &chain, std::span<MeshletBuildSource> sources) {
    if (sources.empty()) return;
    // Chunks build one after another, so the chain holds one chunk's build scratch at a time.
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto split = ChunkByScratch(uint32_t(sources.size()), ScratchWordBudget, [&](uint32_t i) {
        return uint32_t(std::min<uint64_t>(SourceBuildWords(meshes,sources[i]),UINT32_MAX));
    });
    for (const auto chunk : split.Chunks) BuildChunk(r, chain, sources.subspan(chunk.Offset, chunk.Count));
}

uint32_t EnsureMeshletPrimitive(state::Scene &r, MeshBuffers &owner, uint32_t source_primitive) {
    auto &buffers=r.Context.get<GpuBuffers>();
    const auto &meshes=r.Context.get<const MeshStore>();
    if (owner.StoreId==InvalidOffset || owner.RenderTopology>2u) throw std::invalid_argument("Primitive route requires a canonical render owner.");
    const auto &mesh=meshes.Get(owner.StoreId);
    if (source_primitive>=std::max(mesh.PrimitiveMaterials.Count,1u)) throw std::out_of_range("Primitive route exceeds the mesh material domain.");
    if (const auto existing=buffers.PrimitiveRoute(owner,source_primitive); existing!=InvalidOffset) {
        if (!buffers.ActiveMeshlets.Contains(owner.PrimitiveRoot,existing)) throw std::logic_error("Primitive route names a foreign render primitive.");
        return existing;
    }
    if (!owner.MeshRecord.Count) {
        const auto record=BuildMeshRecord(buffers,owner,meshes,owner.StoreId,owner.RenderTopology==0u,owner.RenderTopology==1u);
        owner.MeshRecord=buffers.MeshRecords.Allocate(std::span{&record,1u});
    }
    const auto primitive=buffers.Primitives.Allocate(1u);
    const auto node=buffers.LodNodes.Allocate(1u);
    buffers.LodParents.Mirror(node);
    buffers.Primitives.GetMutable(primitive)[0]={
        .PrimitiveIndex=source_primitive,
        .PrimitiveMaterialOffset=buffers.MeshRecords.Get(owner.MeshRecord)[0].PrimitiveMaterialOffset,
        .LodRootNode=node.Offset,.LodFinestNode=node.Offset,
    };
    buffers.LodNodes.GetMutable(node)[0]={.Error=std::numeric_limits<float>::infinity(),.MeshletRoot=InvalidOffset};
    buffers.LodParents.GetMutable(node)[0]=InvalidOffset;
    std::array edits{
        MeshletIndexEdit{.Root=owner.PrimitiveRoot,.Insert=primitive},
        MeshletIndexEdit{.Root=owner.NodeRoot,.Insert=node},
    };
    buffers.ActiveMeshlets.Update(edits);
    owner.PrimitiveRoot=edits[0].Root;
    owner.NodeRoot=edits[1].Root;
    buffers.ReservePrimitiveRoutes(owner,std::max(mesh.PrimitiveMaterials.Count,1u));
    buffers.PrimitiveRoutes.GetMutable({owner.PrimitiveRoutes.Offset+source_primitive,1u})[0]=primitive.Offset;
    if (!owner.Primitives.Count) owner.Primitives=primitive;
    if (!owner.LodNodes.Count) owner.LodNodes=node;
    ++owner.MeshletRevision;
    return primitive.Offset;
}
