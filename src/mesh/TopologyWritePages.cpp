#include "mesh/TopologyWritePages.h"

#include "mesh/MeshStore.h"
#include "mesh/PageFootprint.h"
#include "state/Registry.h"

namespace {
void AddMaskBlocks(PageFootprint &pages, SelectionMasks masks, std::span<const uint32_t> blocks) {
    std::vector<uint32_t> mask_blocks;
    mask_blocks.reserve(blocks.size());
    for (const auto block : blocks) mask_blocks.push_back(masks.First + block);
    pages.Add(masks.Arena->Buffer, mask_blocks, sizeof(SelectionBlock));
}
} // namespace

void CaptureTopologyEmitWrites(state::Registry &r, uint32_t id, const MeshTopologyJob &job, uint32_t triangle_count, std::span<const uint32_t> vertex_blocks, std::span<const uint32_t> face_blocks, const std::optional<GeometryMasks> &selection_output) {
    if (uint64_t(job.DstFaceCount) * 3u > job.DstHalfedgeCount) throw std::invalid_argument("Topology output faces require at least three corners.");
    if (uint64_t(job.DstCornerOffset) + job.DstHalfedgeCount > UINT32_MAX) throw std::length_error("Topology output exceeds its handle space.");
    const auto &meshes = r.Resources.get<MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const auto corner_blocks = RunBlocks(a.FaceCorners.Entries(record.FaceCorners), job.DstCornerOffset, job.DstHalfedgeCount);
    const auto vertex_members = a.Vertices.Blocks.Buffer.GetSpan<MeshElementBlock>(), corner_members = a.FaceCorners.Blocks.Buffer.GetSpan<MeshElementBlock>();
    PageFootprint pages;
    pages.Add(a.Vertices.Buffer, vertex_blocks, BlockBytes<Vertex>);
    if (!job.DstFaceCount) {
        pages.Add(a.BaseVertexNormals.Buffer, vertex_blocks, BlockBytes<vec3>);
        pages.Layer(a.VertexPrimitives.Values, record.VertexPrimitives, vertex_members, vertex_blocks);
    }
    if (selection_output) AddMaskBlocks(pages, selection_output->Vertices, vertex_blocks);
    pages.Add(a.VertexHidden.Buffer, vertex_blocks, sizeof(SelectionBlock));
    pages.Layer(a.VertexColors.Values, record.VertexColors, vertex_members, vertex_blocks);
    pages.Layer(a.Skin.Values, record.Skin, vertex_members, vertex_blocks);
    pages.Layer(a.Morph.Values, record.Morph, vertex_members, vertex_blocks, record.MorphTargetCount);
    pages.Add(a.FaceCorners.Buffer, corner_blocks, BlockBytes<uint32_t>);
    pages.Add(a.HalfedgeFaces.Buffer, corner_blocks, BlockBytes<uint32_t>);
    pages.Add(a.OppositeHalfedges.Buffer, corner_blocks, BlockBytes<uint32_t>);
    pages.Layer(a.CornerTangents.Values, record.CornerTangents, corner_members, corner_blocks);
    pages.Layer(a.CornerColors.Values, record.CornerColors, corner_members, corner_blocks);
    for (uint32_t uv = 0u; uv < 4u; ++uv) pages.Layer(a.CornerUvs[uv].Values, record.CornerUvs[uv], corner_members, corner_blocks);
    pages.Add(a.FaceRanges.Buffer, face_blocks, BlockBytes<uvec2>);
    pages.Add(a.FaceTriangles.Buffer, face_blocks, BlockBytes<uint32_t>);
    pages.Add(a.FaceSharpness.Buffer, face_blocks, BlockBytes<uint8_t>);
    if (selection_output) AddMaskBlocks(pages, selection_output->Faces, face_blocks);
    pages.Add(a.FaceHidden.Buffer, face_blocks, sizeof(SelectionBlock));
    pages.Layer(a.FacePrimitives.Values, record.FacePrimitives, a.FaceTriangles.Blocks.Buffer.GetSpan<MeshElementBlock>(), face_blocks);
    pages.Add(a.Triangles.Buffer, RunBlocks(a.Triangles.Entries(record.TriangleData), job.DstTriangleOffset, triangle_count), BlockBytes<uvec3>);
    pages.CaptureWrites();
}

void CaptureTopologyEdgeWrites(state::Registry &r, std::span<const uint32_t> edge_blocks, const std::optional<GeometryMasks> &selection_output) {
    const auto &a = r.Resources.get<MeshStore>().Arenas();
    PageFootprint pages;
    pages.Add(a.EdgeHalfedges.Buffer, edge_blocks, BlockBytes<uint32_t>);
    pages.Add(a.EdgeSharpness.Buffer, edge_blocks, BlockBytes<uint8_t>);
    if (selection_output) AddMaskBlocks(pages, selection_output->Edges, edge_blocks);
    pages.Add(a.EdgeHidden.Buffer, edge_blocks, sizeof(SelectionBlock));
    pages.CaptureWrites();
}

void CaptureTopologyNormalWrites(state::Registry &r, uint32_t id, const MeshTopologyJob &job, const BufferArena<uint32_t> &retained) {
    if (!(job.CornerAttributes & MeshAttributeBit_Normal)) return;
    const auto &meshes = r.Resources.get<MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const auto corners = a.FaceCorners.Entries(record.FaceCorners);
    const auto members = a.FaceCorners.Blocks.Buffer.GetSpan<MeshElementBlock>();
    PageFootprint pages;
    pages.Layer(a.CustomNormals.Values, record.CustomNormals, members, RunBlocks(corners, job.DstCornerOffset, job.DstHalfedgeCount));
    pages.Layer(a.CustomNormals.Values, record.CustomNormals, members, WorkBlocks(retained, job.RetainedNormalCorners, job.RetainedNormalCornerCount, corners));
    pages.CaptureWrites();
}
