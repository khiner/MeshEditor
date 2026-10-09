#include "mesh/TopologyWritePages.h"

#include "mesh/MeshStore.h"
#include "mesh/PageFootprint.h"
#include "state/Scene.h"

namespace {
void AddMaskBlocks(PageFootprint &pages, SelectionMasks masks, std::span<const uint32_t> blocks) {
    std::vector<uint32_t> mask_blocks;
    mask_blocks.reserve(blocks.size());
    for (const auto block : blocks) mask_blocks.push_back(masks.First + block);
    pages.Add(masks.Arena->Buffer, mask_blocks, sizeof(SelectionBlock));
}
} // namespace

void CaptureTopologyEmitWrites(state::Scene &r, const MeshTopologyJob &job, uint32_t triangle_count, std::span<const uint32_t> vertex_blocks, std::span<const uint32_t> face_blocks, const std::optional<GeometryMasks> &selection_output) {
    if (uint64_t(job.DstFaceCount) * 3u > job.DstHalfedgeCount) throw std::invalid_argument("Topology output faces require at least three corners.");
    if (uint64_t(job.DstCornerOffset) + job.DstHalfedgeCount > UINT32_MAX) throw std::length_error("Topology output exceeds its canonical address space.");
    const auto &a = r.Context.get<MeshStore>().Arenas();
    const auto corner_blocks = RunBlocks(job.DstCornerOffset, job.DstHalfedgeCount);
    PageFootprint pages;
    pages.Add(a.Vertices.Buffer, vertex_blocks, BlockBytes<Vertex>);
    if (!job.DstFaceCount) {
        pages.Add(a.BaseVertexNormals.Buffer, vertex_blocks, BlockBytes<vec3>);
        pages.AttributeValues(a.VertexPrimitives, vertex_blocks, false);
    }
    if (selection_output) AddMaskBlocks(pages, selection_output->Vertices, vertex_blocks);
    pages.Add(a.VertexHidden.Buffer, vertex_blocks, sizeof(SelectionBlock));
    if (job.VertexAttributes & MeshAttributeBit_Color0) pages.AttributeValues(a.VertexColors, vertex_blocks, true);
    if (job.HasSkin) pages.AttributeValues(a.Skin, vertex_blocks, true);
    if (job.MorphTargetCount) pages.AttributeValues(a.Morph, vertex_blocks, true, job.MorphTargetCount);
    pages.Add(a.FaceCorners.Buffer, corner_blocks, BlockBytes<uint32_t>);
    pages.Add(a.HalfedgeFaces.Buffer, corner_blocks, BlockBytes<uint32_t>);
    pages.Add(a.OppositeHalfedges.Buffer, corner_blocks, BlockBytes<uint32_t>);
    if (job.CornerAttributes & MeshAttributeBit_Tangent) pages.AttributeValues(a.CornerTangents, corner_blocks, true);
    if (job.CornerAttributes & MeshAttributeBit_Color0) pages.AttributeValues(a.CornerColors, corner_blocks, true);
    for (uint32_t uv = 0u; uv < 4u; ++uv)
        if (job.CornerAttributes & (MeshAttributeBit_TexCoord0 << uv)) pages.AttributeValues(a.CornerUvs[uv], corner_blocks, true);
    pages.Add(a.FaceRanges.Buffer, face_blocks, BlockBytes<uvec2>);
    pages.Add(a.FaceTriangles.Buffer, face_blocks, BlockBytes<uint32_t>);
    pages.Add(a.FaceSharpness.Buffer, face_blocks, BlockBytes<uint8_t>);
    if (selection_output) AddMaskBlocks(pages, selection_output->Faces, face_blocks);
    pages.Add(a.FaceHidden.Buffer, face_blocks, sizeof(SelectionBlock));
    pages.AttributeValues(a.FacePrimitives, face_blocks, true);
    pages.Add(a.Triangles.Buffer, RunBlocks(job.DstTriangleOffset, triangle_count), BlockBytes<uvec3>);
    pages.CaptureWrites();
}

void CaptureTopologyEdgeWrites(state::Scene &r, std::span<const uint32_t> edge_blocks, const std::optional<GeometryMasks> &selection_output) {
    const auto &a = r.Context.get<MeshStore>().Arenas();
    PageFootprint pages;
    pages.Add(a.EdgeHalfedges.Buffer, edge_blocks, BlockBytes<uint32_t>);
    pages.Add(a.EdgeSharpness.Buffer, edge_blocks, BlockBytes<uint8_t>);
    if (selection_output) AddMaskBlocks(pages, selection_output->Edges, edge_blocks);
    pages.Add(a.EdgeHidden.Buffer, edge_blocks, sizeof(SelectionBlock));
    pages.CaptureWrites();
}

void CaptureTopologyNormalWrites(state::Scene &r, const MeshTopologyJob &job, const BufferArena<uint32_t> &retained) {
    if (!(job.CornerAttributes & MeshAttributeBit_Normal)) return;
    const auto &a = r.Context.get<MeshStore>().Arenas();
    PageFootprint pages;
    pages.AttributeValues(a.CustomNormals, RunBlocks(job.DstCornerOffset, job.DstHalfedgeCount), true);
    pages.AttributeValues(a.CustomNormals, WorkBlocks(retained, job.RetainedNormalCorners, job.RetainedNormalCornerCount), false);
    pages.CaptureWrites();
}
