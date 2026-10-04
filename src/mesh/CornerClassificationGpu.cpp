#include "mesh/MeshStore.h"

#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"
#include "render/ElementWorkOps.h"

MeshStore::CornerClassUpdate MeshStore::EncodeCornerClassification(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, ElementWork vertices,
                                                                   uint32_t vertex_count, uint32_t incoming, bool complete) {
    const auto &record = Records.at(id);
    const bool seed_all = vertices.Storage.Slot == InvalidSlot;
    CornerClassUpdate update{.StoreId = id, .Complete = complete || seed_all};
    if (seed_all) {
        vertex_count = Buffers.Vertices.Count(record.Vertices);
        incoming = Buffers.FaceCorners.Count(record.FaceCorners);
    }
    if (!record.TriangleCount) {
        auto &written = WriteRecord(id);
        for (const auto block : GetBlockList(id, ElementDomain::Halfedge).Blocks) {
            Buffers.CornerSectors.Release(block);
            Buffers.NormalSectors.Release(block);
        }
        written.SectorBlockCount = 0u;
        written.Classification = uint32_t(CornerClassMode::UniformVertex);
        DerivedRecords.at(id).NormalRevision = ++NextNormalRevision;
        return update;
    }
    if (!vertex_count) return update;
    const auto state = chain.Scratch.Allocate(2u);
    std::ranges::fill(chain.Scratch.GetMutable(state),0u);
    auto &pc = update.Pc;
    pc = {
        .Connectivity=GetConnectivityRef(id),.VertexCount=vertex_count,
        .CornerBlocksSlot=Buffers.FaceCorners.Blocks.Buffer.Slot,.CornerOwner=record.FaceCorners.Index,
        .FaceCount=Buffers.FaceTriangles.Count(record.FaceData),
        .EdgeSharpnessSlot=Buffers.EdgeSharpness.Buffer.Slot,.FaceSharpnessSlot=Buffers.FaceSharpness.Buffer.Slot,
        .CornerSectors=Buffers.CornerSectors.Ref(),.State={chain.Scratch.Buffer.Slot,state.Offset},
    };
    update.VertexCount = vertex_count;
    update.Incoming = incoming;
    if (seed_all) {
        // The live masks of the mesh's vertex blocks seed the work on the host.
        const auto blocks = GetBlockList(id, ElementDomain::Vertex).Blocks;
        pc.Vertices = AllocateElementWork(chain.Scratch,Buffers.Vertices.Capacity(),blocks.size());
        auto data = chain.Scratch.GetMutable(WorkStorageRange(pc.Vertices));
        for (const auto block : blocks) {
            const auto &live = Buffers.Vertices.Blocks.Get({block,1u})[0].Live;
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) MarkElementWorkWord(data,pc.Vertices,block*MeshElementBlockWords+w,live[w]);
        }
        FinishElementWork(data,pc.Vertices.Capacity);
    } else pc.Vertices = vertices;
    // Dirty and needed blocks are blocks of the mesh's corner set.
    const auto corner_blocks = Buffers.FaceCorners.Capacity()/MeshElementBlockSize;
    const auto block_bound = std::min(incoming,Buffers.FaceCorners.Set(record.FaceCorners).BlockCount);
    pc.DirtyBlocks = AllocateElementWork(chain.Scratch,corner_blocks,block_bound);
    pc.NeededBlocks = AllocateElementWork(chain.Scratch,corner_blocks,block_bound);
    const auto &pipelines = GetMeshPipelines(r);
    const auto groups = uint32_t((uint64_t(vertex_count)+255u)/256u);
    // The count and the plan read the same fans and write separate results.
    chain.Concurrent([&] {
        chain.Groups(pipelines[MeshPass::CornerClassificationCount],pc,groups);
        chain.Groups(pipelines[MeshPass::CornerClassificationPlan],pc,groups);
    });
    return update;
}

void MeshStore::PlanCornerClassification(state::Scene &r, mtl::ComputeChain &chain, CornerClassUpdate &update) {
    auto &pc = update.Pc;
    if (pc.State.Slot == InvalidSlot) return;
    const auto state = chain.Scratch.Get({pc.State.Offset,2u});
    if (state[0] > update.Incoming) throw std::logic_error("Corner classification exceeds its incoming corner bound.");
    const auto flags = state[1];
    std::vector<uint32_t> needed;
    // The block works finish on the host, in ascending block order.
    for (const auto work : {pc.DirtyBlocks,pc.NeededBlocks}) {
        CheckElementWork(chain.Scratch,work);
        FinishElementWork(chain.Scratch.GetMutable(WorkStorageRange(work)),work.Capacity);
    }
    ForEachWorkElement(chain.Scratch,pc.DirtyBlocks,[&](uint32_t b) { update.Dirty.push_back(b); });
    ForEachWorkElement(chain.Scratch,pc.NeededBlocks,[&](uint32_t b) { needed.push_back(b); });
    // Existing root payloads stay in place. Only newly needed blocks receive
    // defaults, and only changed corner payload pages need write capture.
    const auto added = uint32_t(std::ranges::count_if(needed, [&](uint32_t b) { return !Buffers.CornerSectors.PayloadBlock(b); }));
    Buffers.CornerSectors.Attach(needed,InvalidOffset);
    Buffers.NormalSectors.Attach(needed);
    for (const auto b : update.Dirty) {
        const auto payload = Buffers.CornerSectors.PayloadBlock(b);
        if (payload) Buffers.CornerSectors.Values.Buffer.CaptureWrite(uint64_t(payload-1u)*256u*4u,256u*4u);
    }
    const auto status = chain.Scratch.Allocate(uint32_t(update.Dirty.size()));
    pc.StatusOffset = status.Offset-pc.State.Offset;
    std::ranges::fill(chain.Scratch.GetMutable(status),0u);
    pc.CornerSectors = Buffers.CornerSectors.Ref();
    const auto &pipelines = GetMeshPipelines(r);
    chain.Groups(pipelines[MeshPass::CornerClassificationWrite],pc,uint32_t((uint64_t(update.VertexCount)+255u)/256u));
    chain.Groups(pipelines[MeshPass::CornerClassificationBlocks],pc,uint32_t(update.Dirty.size()));
    auto &record = WriteRecord(update.StoreId);
    // Blocks that can hold roots count as sector blocks before the recorded writes, so a derive recorded after them reads sectors.
    // Finish releases the blocks the writes leave without roots.
    record.SectorBlockCount += added;
    const auto uniform = !(flags&2u) ? uint32_t(CornerClassMode::UniformFace) : !(flags&5u) ? uint32_t(CornerClassMode::UniformVertex) : uint32_t(CornerClassMode::Mixed);
    // A local result can preserve an already-uniform mode. Once a mesh is
    // mixed, unaffected faces keep using their canonical flags and roots.
    record.Classification = update.Complete || uniform == record.Classification ? uniform : uint32_t(CornerClassMode::Mixed);
    DerivedRecords.at(update.StoreId).NormalRevision = ++NextNormalRevision;
}

void MeshStore::FinishCornerClassification(const mtl::ComputeChain &chain, const CornerClassUpdate &update) {
    if (update.Dirty.empty() || update.Pc.State.Slot == InvalidSlot) return;
    auto &record = WriteRecord(update.StoreId);
    const auto present = chain.Scratch.Get({update.Pc.State.Offset+update.Pc.StatusOffset,uint32_t(update.Dirty.size())});
    for (uint32_t i = 0u; i < update.Dirty.size(); ++i) {
        const auto b = update.Dirty[i];
        if (!(present[i]&2u)) Buffers.NormalSectors.Release(b);
        if (present[i]&1u || !Buffers.CornerSectors.PayloadBlock(b)) continue;
        Buffers.CornerSectors.Release(b);
        --record.SectorBlockCount;
    }
}

void MeshStore::UpdateCornerClassification(state::Scene &r, mtl::ComputeChain &chain, std::span<const uint32_t> ids) {
    std::vector<CornerClassUpdate> updates;
    updates.reserve(ids.size());
    chain.Concurrent([&] { for (const auto id : ids) updates.push_back(EncodeCornerClassification(r,chain,id)); });
    chain.Submit();
    for (auto &update : updates) PlanCornerClassification(r,chain,update);
    chain.AfterSubmit([this,&chain,updates=std::move(updates)] {
        for (const auto &update : updates) FinishCornerClassification(chain,update);
    });
}
