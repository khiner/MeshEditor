#include "mesh/MeshStore.h"

#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"
#include "render/ElementWorkOps.h"

MeshStore::CornerClassUpdate MeshStore::EncodeCornerClassification(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, ElementWork vertices, uint32_t vertex_count, uint32_t incoming, bool complete) {
    const auto &record = Records.at(id);
    const bool seed_all = vertices.Storage.Slot == InvalidSlot;
    CornerClassUpdate update{.StoreId = id, .Complete = complete || seed_all};
    if (seed_all) {
        vertex_count = Buffers.Vertices.Count(record.Vertices);
        incoming = Buffers.FaceCorners.Count(record.FaceCorners);
    }
    if (!record.TriangleCount) {
        auto &written = WriteRecord(id);
        Buffers.CornerSectors.Release(written.CornerSectors);
        Buffers.NormalSectors.Release(written.NormalSectors);
        written.Classification = uint32_t(CornerClassMode::UniformVertex);
        DerivedRecords.at(id).NormalRevision = ++NextNormalRevision;
        return update;
    }
    if (!vertex_count) return update;
    const auto state = chain.Scratch.Allocate(2u);
    std::ranges::fill(chain.Scratch.GetMutable(state), 0u);
    auto &pc = update.Pc;
    pc = {
        .Connectivity = GetConnectivityRef(id),
        .VertexCount = vertex_count,
        .FaceCount = Buffers.FaceTriangles.Count(record.FaceData),
        .EdgeSharpnessSlot = Buffers.EdgeSharpness.Buffer.Slot,
        .FaceSharpnessSlot = Buffers.FaceSharpness.Buffer.Slot,
        .CornerSectors = Buffers.CornerSectors.Ref(record.CornerSectors),
        .State = {chain.Scratch.Buffer.Slot, state.Offset},
    };
    update.VertexCount = vertex_count;
    update.Incoming = incoming;
    if (seed_all) {
        // The live masks of the mesh's vertex blocks seed the work on the host.
        const auto blocks = GetBlockList(id, ElementDomain::Vertex).Blocks;
        pc.Vertices = AllocateElementWork(chain.Scratch, Buffers.Vertices.HandleBound(record.Vertices), blocks.size());
        auto data = chain.Scratch.GetMutable(WorkStorageRange(pc.Vertices));
        for (const auto block : blocks) {
            const auto &members = Buffers.Vertices.Blocks.Get({block, 1u})[0];
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) MarkElementWorkWord(data, pc.Vertices, members.Local * MeshElementBlockWords + w, members.Live[w]);
        }
        FinishElementWork(data, pc.Vertices.Capacity);
    } else pc.Vertices = vertices;
    // Dirty and needed blocks are the arena blocks of the mesh's corner set.
    const auto corner_blocks = Buffers.FaceCorners.Capacity() / MeshElementBlockSize;
    const auto block_bound = std::min(incoming, Buffers.FaceCorners.Set(record.FaceCorners).BlockCount);
    pc.DirtyBlocks = AllocateElementWork(chain.Scratch, corner_blocks, block_bound);
    pc.NeededBlocks = AllocateElementWork(chain.Scratch, corner_blocks, block_bound);
    const auto &pipelines = GetMeshPipelines(r);
    const auto groups = uint32_t((uint64_t(vertex_count) + 255u) / 256u);
    // The count and the plan read the same fans and write separate results.
    chain.Concurrent([&] {
        chain.Groups(pipelines[MeshPass::CornerClassificationCount], pc, groups);
        chain.Groups(pipelines[MeshPass::CornerClassificationPlan], pc, groups);
    });
    return update;
}

void MeshStore::PlanCornerClassification(state::Scene &r, mtl::ComputeChain &chain, CornerClassUpdate &update) {
    auto &pc = update.Pc;
    if (pc.State.Slot == InvalidSlot) return;
    const auto state = chain.Scratch.Get({pc.State.Offset, 2u});
    if (state[0] > update.Incoming) throw std::logic_error("Corner classification exceeds its incoming corner bound.");
    const auto flags = state[1];
    // The block works finish on the host, in ascending block order.
    for (const auto work : {pc.DirtyBlocks, pc.NeededBlocks}) {
        CheckElementWork(chain.Scratch, work);
        FinishElementWork(chain.Scratch.GetMutable(WorkStorageRange(work)), work.Capacity);
    }
    bool needed = false;
    ForEachWorkElement(chain.Scratch, pc.NeededBlocks, [&](uint32_t) { needed = true; });
    auto &record = WriteRecord(update.StoreId);
    const auto uniform = !(flags & 2u) ? uint32_t(CornerClassMode::UniformFace) : !(flags & 5u) ? uint32_t(CornerClassMode::UniformVertex) :
                                                                                                  uint32_t(CornerClassMode::Mixed);
    // A local result can preserve an already-uniform mode. Once a mesh is
    // mixed, unaffected faces keep using their canonical flags and roots.
    record.Classification = update.Complete || uniform == record.Classification ? uniform : uint32_t(CornerClassMode::Mixed);
    DerivedRecords.at(update.StoreId).NormalRevision = ++NextNormalRevision;
    // A mixed mesh holds sector roots and their normals from its first needed corner, which starts every corner at the vertex normal, and a uniform mesh holds none.
    if (record.Classification != uint32_t(CornerClassMode::Mixed)) {
        Buffers.CornerSectors.Release(record.CornerSectors);
        Buffers.NormalSectors.Release(record.NormalSectors);
        return;
    }
    if (!record.CornerSectors.Count) {
        if (!needed) return;
        const auto blocks = Buffers.FaceCorners.HandleBound(record.FaceCorners) / MeshElementBlockSize;
        Buffers.CornerSectors.Cover(record.CornerSectors, blocks);
        Buffers.CornerSectors.Clear(record.CornerSectors, {0u, blocks});
        Buffers.NormalSectors.Cover(record.NormalSectors, blocks);
        Buffers.NormalSectors.Clear(record.NormalSectors, {0u, blocks});
    } else {
        // Only the dirty corner blocks' roots change.
        std::vector<uint32_t> written;
        ForEachWorkElement(chain.Scratch, pc.DirtyBlocks, [&](uint32_t b) { written.push_back(record.CornerSectors.Offset + Buffers.FaceCorners.Blocks.Get({b, 1u})[0].Local); });
        Buffers.CornerSectors.Values.Buffer.CaptureWriteElements(written, sizeof(decltype(Buffers.CornerSectors)::Block));
    }
    pc.CornerSectors = Buffers.CornerSectors.Ref(record.CornerSectors);
    chain.Groups(GetMeshPipelines(r)[MeshPass::CornerClassificationWrite], pc, uint32_t((uint64_t(update.VertexCount) + 255u) / 256u));
}

void MeshStore::UpdateCornerClassification(state::Scene &r, mtl::ComputeChain &chain, std::span<const uint32_t> ids) {
    std::vector<CornerClassUpdate> updates;
    updates.reserve(ids.size());
    chain.Concurrent([&] { for (const auto id : ids) updates.push_back(EncodeCornerClassification(r,chain,id)); });
    chain.Submit();
    for (auto &update : updates) PlanCornerClassification(r, chain, update);
}
