#include "render/GpuBuffers.h"

#include "gpu/CornerClassEncoding.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"

#include "meshoptimizer.h"

#include <unordered_map>

namespace {
constexpr size_t MeshletMaxVertices{64};
constexpr size_t MeshletMaxTriangles{48};

std::array<uint32_t, 3> CanonicalTriangle(std::array<uint32_t, 3> triangle) {
    const std::array rotations{triangle, std::array{triangle[1], triangle[2], triangle[0]}, std::array{triangle[2], triangle[0], triangle[1]}};
    return *std::ranges::min_element(rotations);
}

struct TriangleHash {
    size_t operator()(const std::array<uint32_t, 3> &v) const {
        size_t seed = v[0];
        seed ^= size_t(v[1]) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        seed ^= size_t(v[2]) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        return seed;
    }
};
} // namespace

void GpuBuffers::RebuildMeshlets(MeshBuffers &buffers, const Mesh &mesh, const MeshStore &meshes) {
    Meshlets.Release(buffers.Meshlets);
    MeshletTriangleIds.Release(buffers.MeshletTriangles);
    Primitives.Release(buffers.Primitives);
    buffers.Primitives = buffers.Meshlets = buffers.MeshletTriangles = {};

    const auto indices = mesh.CreateTriangleIndices();
    if (indices.empty()) return;

    const auto vertices = mesh.GetVerticesSpan();
    std::vector<MeshletRecord> meshlet_records;
    std::vector<PrimitiveRecord> primitive_records;
    std::vector<uint32_t> triangle_ids;

    for (const auto &primitive : meshes.GetPrimitiveTriangleRanges(mesh.GetStoreId())) {
        const auto first_index = size_t(primitive.FirstTriangle) * 3;
        const auto index_count = size_t(primitive.TriangleCount) * 3;
        const auto primitive_indices = std::span{indices}.subspan(first_index, index_count);
        const auto bound = meshopt_buildMeshletsBound(index_count, MeshletMaxVertices, MeshletMaxTriangles);
        std::vector<meshopt_Meshlet> built(bound);
        std::vector<uint32_t> local_vertices(index_count);
        std::vector<uint8_t> local_triangles(index_count);
        const auto meshlet_count = meshopt_buildMeshlets(
            built.data(), local_vertices.data(), local_triangles.data(), primitive_indices.data(), primitive_indices.size(),
            &vertices.front().Position.x, vertices.size(), sizeof(Vertex), MeshletMaxVertices, MeshletMaxTriangles, 0.5f
        );
        built.resize(meshlet_count);

        std::unordered_multimap<std::array<uint32_t, 3>, uint32_t, TriangleHash> source_triangles;
        source_triangles.reserve(primitive.TriangleCount);
        for (uint32_t i = 0; i < primitive.TriangleCount; ++i) {
            const auto offset = first_index + size_t(i) * 3;
            source_triangles.emplace(CanonicalTriangle({indices[offset], indices[offset + 1], indices[offset + 2]}), primitive.FirstTriangle + i);
        }

        const uint32_t first_meshlet = meshlet_records.size();
        for (const auto &meshlet : built) {
            const uint32_t first_triangle_id = triangle_ids.size();
            for (uint32_t t = 0; t < meshlet.triangle_count; ++t) {
                std::array<uint32_t, 3> triangle;
                for (uint32_t c = 0; c < 3; ++c) {
                    const auto local = local_triangles[meshlet.triangle_offset + t * 3 + c];
                    triangle[c] = local_vertices[meshlet.vertex_offset + local];
                }
                const auto key = CanonicalTriangle(triangle);
                const auto source = source_triangles.find(key);
                triangle_ids.push_back(source->second);
                source_triangles.erase(source);
            }
            const auto bounds = meshopt_computeMeshletBounds(
                local_vertices.data() + meshlet.vertex_offset,
                local_triangles.data() + meshlet.triangle_offset,
                meshlet.triangle_count, &vertices.front().Position.x, vertices.size(), sizeof(Vertex)
            );
            meshlet_records.emplace_back(MeshletRecord{
                .TriangleOffset = first_triangle_id,
                .TriangleCount = meshlet.triangle_count,
                .Primitive = uint32_t(primitive_records.size()),
                .Center = {bounds.center[0], bounds.center[1], bounds.center[2]},
                .Radius = bounds.radius,
            });
        }

        DrawData draw{
            .VertexSlot = buffers.Vertices.Slot,
            .IndexSlotOffset = {buffers.FaceIndices.Slot, buffers.FaceIndices.Offset + uint32_t(first_index)},
            .ModelSlot = Instances.TransformBuffer.Slot,
            .ObjectIdSlot = meshes.GetFaceIdRange(mesh.GetStoreId()).Slot,
            .CornerClassOffset = meshes.GetCornerClassOffset(mesh.GetStoreId()),
            .CustomCornerMaskOffset = OffsetOrInvalid(meshes.GetCustomCornerMaskRange(mesh.GetStoreId())),
            .CustomCornerNormalOffset = OffsetOrInvalid(meshes.GetCustomCornerNormalRange(mesh.GetStoreId())),
            .CornerBase = uint32_t(first_index),
            .BaseSeamNormalOffset = OffsetOrInvalid(meshes.GetBaseSeamNormalRange(mesh.GetStoreId())),
            .CornerTangentOffset = OffsetOrInvalid(meshes.GetCornerTangentRange(mesh.GetStoreId())),
            .CornerColorOffset = OffsetOrInvalid(meshes.GetCornerColorRange(mesh.GetStoreId())),
            .FaceIdOffset = meshes.GetFaceIdRange(mesh.GetStoreId()).Offset + primitive.FirstTriangle,
            .BaseFaceNormalOffset = meshes.GetFaceDataRange(mesh.GetStoreId()).Offset,
            .VertexCountOrHeadImageSlot = buffers.Vertices.Count,
            .ElementStateSlotOffset = meshes.GetFaceStateRange(mesh.GetStoreId()),
            .InstanceStateSlot = Instances.StateBuffer.Slot,
            .VertexOffset = buffers.Vertices.Offset,
            .MorphShadingAuthored = meshes.GetMorphShadingAuthored(mesh.GetStoreId()) ? 1u : 0u,
            .PrimitiveMaterialOffset = OffsetOrInvalid(meshes.GetPrimitiveMaterialRange(mesh.GetStoreId())),
            .ElementPrimitiveOffset = OffsetOrInvalid(meshes.GetElementPrimitiveRange(mesh.GetStoreId())),
        };
        if (draw.CornerClassOffset < uint32_t(CornerClassEncoding::UniformFaceOffset)) draw.CornerClassOffset += uint32_t(first_index);
        const auto advance_corner = [first_index](uint32_t &offset) {
            if (offset != InvalidOffset) offset += uint32_t(first_index);
        };
        advance_corner(draw.CornerTangentOffset);
        advance_corner(draw.CornerColorOffset);
        for (uint32_t set = 0; set < draw.CornerUvOffsets.size(); ++set) {
            draw.CornerUvOffsets[set] = OffsetOrInvalid(meshes.GetCornerUvRange(mesh.GetStoreId(), set));
            advance_corner(draw.CornerUvOffsets[set]);
        }
        primitive_records.emplace_back(PrimitiveRecord{
            .Draw = draw,
            .PrimitiveIndex = primitive.PrimitiveIndex,
            .FirstTriangle = primitive.FirstTriangle,
            .MeshletOffset = first_meshlet,
            .MeshletCount = uint32_t(meshlet_records.size()) - first_meshlet,
        });
    }

    buffers.MeshletTriangles = MeshletTriangleIds.Allocate(triangle_ids);
    for (auto &record : meshlet_records) {
        record.TriangleOffset += buffers.MeshletTriangles.Offset;
    }
    buffers.Meshlets = Meshlets.Allocate(meshlet_records);
    for (auto &record : primitive_records) record.MeshletOffset += buffers.Meshlets.Offset;
    buffers.Primitives = Primitives.Allocate(primitive_records);
    for (auto &record : Meshlets.GetMutable(buffers.Meshlets)) record.Primitive += buffers.Primitives.Offset;
}
