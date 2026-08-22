#include "render/GpuBuffers.h"

#include "gpu/CornerClassEncoding.h"
#include "gpu/MeshletGeometryEncoding.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"

#include "meshoptimizer.h"

#include <bit>
#include <unordered_map>

namespace {
constexpr size_t MeshletMaxVertices{64};
constexpr size_t MeshletMaxTriangles{48};

std::array<uint32_t, 3> CanonicalTriangle(std::array<uint32_t, 3> triangle) {
    const std::array rotations{triangle, std::array{triangle[1], triangle[2], triangle[0]}, std::array{triangle[2], triangle[0], triangle[1]}};
    return *std::ranges::min_element(rotations);
}

void HashCombine(size_t &seed, uint32_t value) { seed ^= size_t(value) + 0x9e3779b9u + (seed << 6) + (seed >> 2); }

struct TriangleHash {
    size_t operator()(const std::array<uint32_t, 3> &v) const {
        size_t seed = v[0];
        HashCombine(seed, v[1]);
        HashCombine(seed, v[2]);
        return seed;
    }
};

struct RenderVertexKey {
    // Vertex id, corner class, four UVs, tangent, and color, all compared as raw bits.
    std::array<uint32_t, 18> Words{};
    bool operator==(const RenderVertexKey &) const = default;
};

struct RenderVertexHash {
    size_t operator()(const RenderVertexKey &key) const {
        size_t seed{};
        for (const uint32_t word : key.Words) HashCombine(seed, word);
        return seed;
    }
};

uint32_t FloatBits(float value) { return std::bit_cast<uint32_t>(value); }

// A mixed-normal meshlet stores the never-culls cutoff, so the cone test needs no separate flag.
uint32_t PackCone(const meshopt_Bounds &bounds, bool cone_cull_safe) {
    return uint32_t(uint8_t(bounds.cone_axis_s8[0])) |
        uint32_t(uint8_t(bounds.cone_axis_s8[1])) << 8u |
        uint32_t(uint8_t(bounds.cone_axis_s8[2])) << 16u |
        uint32_t(uint8_t(cone_cull_safe ? bounds.cone_cutoff_s8 : int8_t{127})) << 24u;
}
} // namespace

void GpuBuffers::RebuildMeshlets(MeshBuffers &buffers, const Mesh &mesh, const MeshStore &meshes) {
    Meshlets.Release(buffers.Meshlets);
    MeshletTriangleIds.Release(buffers.MeshletTriangles);
    MeshletVertexCorners.Release(buffers.MeshletVertices);
    MeshletLocalTriangles.Release(buffers.MeshletLocalTriangles);
    Primitives.Release(buffers.Primitives);
    buffers.Primitives = buffers.Meshlets = buffers.MeshletTriangles = buffers.MeshletVertices = buffers.MeshletLocalTriangles = {};

    const auto indices = mesh.CreateTriangleIndices();
    if (indices.empty()) return;

    const auto vertices = mesh.GetVerticesSpan();
    std::vector<MeshletRecord> meshlet_records;
    std::vector<PrimitiveRecord> primitive_records;
    std::vector<uint32_t> triangle_ids, meshlet_vertices;
    std::vector<uint8_t> meshlet_triangles;

    const uint32_t store_id = mesh.GetStoreId();
    const auto corner_classes = meshes.GetCornerClasses(store_id);
    const uint32_t uniform_corner_class = meshes.GetCornerClassOffset(store_id);
    const auto face_ids = meshes.GetTriangleFaceIds(store_id);
    [[maybe_unused]] const auto element_primitives = meshes.GetElementPrimitiveIndices(store_id);
    const bool morph_shading_authored = meshes.GetMorphShadingAuthored(store_id);
    const auto custom_corner_masks = meshes.GetCustomCornerMasks(store_id);
    const auto corner_tangents = meshes.GetCornerTangents(store_id);
    const auto corner_colors = meshes.GetCornerColors(store_id);
    const std::array corner_uvs{
        meshes.GetCornerUvs(store_id, 0),
        meshes.GetCornerUvs(store_id, 1),
        meshes.GetCornerUvs(store_id, 2),
        meshes.GetCornerUvs(store_id, 3),
    };

    const uint32_t uniform_class_word =
        uint32_t(uniform_corner_class == uint32_t(CornerClassEncoding::UniformFaceOffset) ? CornerClass::Face : CornerClass::Vertex)
        << uint32_t(CornerClassEncoding::TagShift);
    const auto corner_class_value = [&](uint32_t global_corner) {
        return corner_classes.empty() ? uniform_class_word : corner_classes[global_corner];
    };
    const auto has_custom_normal = [&](uint32_t global_corner) {
        return !custom_corner_masks.empty() &&
            (custom_corner_masks[global_corner / 32u].x & (1u << (global_corner % 32u))) != 0u;
    };

    std::vector<uint32_t> welded_indices, representative_corners;
    std::vector<std::array<float, 3>> welded_positions;
    std::vector<uint8_t> flat_face_triangles;
    std::unordered_map<RenderVertexKey, uint32_t, RenderVertexHash> welded;
    std::vector<meshopt_Meshlet> built;
    std::vector<uint32_t> local_vertices;
    std::vector<uint8_t> local_triangles;
    std::unordered_multimap<std::array<uint32_t, 3>, uint32_t, TriangleHash> source_triangles;
    for (const auto &primitive : meshes.GetPrimitiveTriangleRanges(mesh.GetStoreId())) {
        const auto first_index = size_t(primitive.FirstTriangle) * 3;
        const auto index_count = size_t(primitive.TriangleCount) * 3;
        const auto primitive_indices = std::span{indices}.subspan(first_index, index_count);

        // Meshlets are built inside one source-primitive range. Coverage classification therefore
        // remains uniform across every resulting meshlet and can be done once in the meshlet cull.
        for (uint32_t triangle = primitive.FirstTriangle; triangle < primitive.FirstTriangle + primitive.TriangleCount; ++triangle) {
            assert(face_ids[triangle] > 0u && face_ids[triangle] <= element_primitives.size());
            assert(element_primitives[face_ids[triangle] - 1u] == primitive.PrimitiveIndex);
        }

        welded_indices.assign(index_count, 0u);
        representative_corners.clear();
        welded_positions.clear();
        welded.clear();
        welded.reserve(index_count);
        flat_face_triangles.assign(primitive.TriangleCount, morph_shading_authored ? 0u : 1u);
        if (!morph_shading_authored) {
            for (uint32_t triangle = 0; triangle < primitive.TriangleCount; ++triangle) {
                for (uint32_t c = 0; c < 3u; ++c) {
                    const uint32_t global_corner = uint32_t(first_index) + triangle * 3u + c;
                    const uint32_t tag = corner_class_value(global_corner) >> uint32_t(CornerClassEncoding::TagShift);
                    if (tag != uint32_t(CornerClass::Face) || has_custom_normal(global_corner)) {
                        flat_face_triangles[triangle] = 0u;
                        break;
                    }
                }
            }
        }
        const auto append_render_vertex = [&](uint32_t corner) {
            const uint32_t render_vertex = uint32_t(representative_corners.size());
            welded_indices[corner] = render_vertex;
            representative_corners.push_back(corner | (flat_face_triangles[corner / 3u] ? uint32_t(MeshletGeometryEncoding::FlatVertexBit) : 0u));
            const vec3 position = vertices[primitive_indices[corner]].Position;
            welded_positions.push_back({position.x, position.y, position.z});
            return render_vertex;
        };
        for (uint32_t corner = 0; corner < index_count; ++corner) {
            const uint32_t global_corner = uint32_t(first_index) + corner;
            // A custom-normal corner's frame depends on its own triangle, so it welds with nothing.
            if (has_custom_normal(global_corner)) {
                append_render_vertex(corner);
                continue;
            }
            RenderVertexKey key;
            auto &words = key.Words;
            words[0] = primitive_indices[corner];
            uint32_t corner_class = corner_class_value(global_corner);
            // All-Face triangles carry their common face normal per primitive. A Face corner in any
            // other triangle still reads the source face normal in the vertex transform, so that
            // otherwise-implicit input remains part of its render-equivalence key.
            if (!flat_face_triangles[corner / 3u] &&
                corner_class >> uint32_t(CornerClassEncoding::TagShift) == uint32_t(CornerClass::Face)) {
                corner_class |= (face_ids[global_corner / 3u] - 1u) & uint32_t(CornerClassEncoding::IndexMask);
            }
            words[1] = corner_class;
            uint32_t word = 2;
            for (const auto uvs : corner_uvs) {
                const vec2 uv = uvs.empty() ? vec2(0) : uvs[global_corner];
                words[word++] = FloatBits(uv.x);
                words[word++] = FloatBits(uv.y);
            }
            const vec4 tangent = corner_tangents.empty() ? vec4(0, 0, 0, 1) : corner_tangents[global_corner];
            words[word++] = FloatBits(tangent.x);
            words[word++] = FloatBits(tangent.y);
            words[word++] = FloatBits(tangent.z);
            words[word++] = FloatBits(tangent.w);
            const vec4 color = corner_colors.empty() ? vec4(1) : corner_colors[global_corner];
            words[word++] = FloatBits(color.x);
            words[word++] = FloatBits(color.y);
            words[word++] = FloatBits(color.z);
            words[word] = FloatBits(color.w);

            if (const auto found = welded.find(key); found != welded.end()) {
                welded_indices[corner] = found->second;
                continue;
            }
            welded.emplace(key, append_render_vertex(corner));
        }

        const auto bound = meshopt_buildMeshletsBound(index_count, MeshletMaxVertices, MeshletMaxTriangles);
        built.resize(bound);
        local_vertices.resize(bound * MeshletMaxVertices);
        local_triangles.resize(bound * MeshletMaxTriangles * 3u);
        const auto meshlet_count = meshopt_buildMeshlets(
            built.data(), local_vertices.data(), local_triangles.data(), welded_indices.data(), welded_indices.size(),
            welded_positions.front().data(), welded_positions.size(), sizeof(welded_positions.front()),
            MeshletMaxVertices, MeshletMaxTriangles, 0.5f
        );
        built.resize(meshlet_count);

        source_triangles.clear();
        source_triangles.reserve(primitive.TriangleCount);
        for (uint32_t i = 0; i < primitive.TriangleCount; ++i) {
            const auto offset = size_t(i) * 3;
            source_triangles.emplace(CanonicalTriangle({welded_indices[offset], welded_indices[offset + 1], welded_indices[offset + 2]}), primitive.FirstTriangle + i);
        }

        const uint32_t first_meshlet = meshlet_records.size();
        for (const auto &meshlet : built) {
            const uint32_t first_triangle_id = triangle_ids.size();
            const uint32_t first_vertex = meshlet_vertices.size();
            const uint32_t first_local_triangle = meshlet_triangles.size();
            for (uint32_t v = 0; v < meshlet.vertex_count; ++v) {
                meshlet_vertices.push_back(representative_corners[local_vertices[meshlet.vertex_offset + v]]);
            }
            bool cone_cull_safe = true;
            for (uint32_t t = 0; t < meshlet.triangle_count; ++t) {
                std::array<uint32_t, 3> triangle;
                std::array<uint8_t, 3> local_triangle;
                for (uint32_t c = 0; c < 3; ++c) {
                    local_triangle[c] = local_triangles[meshlet.triangle_offset + t * 3 + c];
                    triangle[c] = local_vertices[meshlet.vertex_offset + local_triangle[c]];
                }
                const auto key = CanonicalTriangle(triangle);
                const auto source = source_triangles.find(key);
                assert(source != source_triangles.end());
                const uint32_t source_triangle = source->second - primitive.FirstTriangle;
                cone_cull_safe &= flat_face_triangles[source_triangle] != 0u;
                for (uint32_t c = 0; c < 3u; ++c) {
                    meshlet_triangles.push_back(local_triangle[c] | (c == 0u && flat_face_triangles[source_triangle] ? uint8_t(MeshletGeometryEncoding::FlatTriangleBit) : 0u));
                }
                triangle_ids.push_back(source->second);
                source_triangles.erase(source);
            }
            const auto bounds = meshopt_computeMeshletBounds(
                local_vertices.data() + meshlet.vertex_offset,
                local_triangles.data() + meshlet.triangle_offset,
                meshlet.triangle_count, welded_positions.front().data(), welded_positions.size(), sizeof(welded_positions.front())
            );
            meshlet_records.emplace_back(MeshletRecord{
                .TriangleOffset = first_triangle_id,
                .TriangleCount = meshlet.triangle_count,
                .VertexOffset = first_vertex,
                .VertexCount = meshlet.vertex_count,
                .LocalTriangleOffset = first_local_triangle,
                .Primitive = uint32_t(primitive_records.size()),
                .ConeAxisCutoff = PackCone(bounds, cone_cull_safe),
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
    buffers.MeshletVertices = MeshletVertexCorners.Allocate(meshlet_vertices);
    buffers.MeshletLocalTriangles = MeshletLocalTriangles.Allocate(meshlet_triangles);
    for (auto &record : meshlet_records) {
        record.TriangleOffset += buffers.MeshletTriangles.Offset;
        record.VertexOffset += buffers.MeshletVertices.Offset;
        record.LocalTriangleOffset += buffers.MeshletLocalTriangles.Offset;
    }
    buffers.Meshlets = Meshlets.Allocate(meshlet_records);
    for (auto &record : primitive_records) record.MeshletOffset += buffers.Meshlets.Offset;
    buffers.Primitives = Primitives.Allocate(primitive_records);
    for (auto &record : Meshlets.GetMutable(buffers.Meshlets)) record.Primitive += buffers.Primitives.Offset;
}
