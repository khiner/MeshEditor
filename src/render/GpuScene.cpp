#include "render/GpuBuffers.h"

#include "FlatKeyMap.h"
#include "Parallel.h"
#include "gpu/CornerClassEncoding.h"
#include "gpu/MeshPrimitiveTopology.h"
#include "gpu/MeshletGeometryEncoding.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"

#include "meshoptimizer.h"

#include <bit>
#include <limits>
#include <numeric>

namespace {
constexpr size_t MeshletMaxVertices{64};
constexpr size_t MeshletMaxTriangles{48};
// A primitive with more triangles than this is split into spatial chunks that clusterize on their
// own. The split comes from the triangle count alone, so a mesh's meshlets never depend on how many
// cores ran the build.
constexpr uint32_t ChunkTriangles{128u * 1024u};
// The widest weld key: vertex id and corner class, four UV sets, tangents, and colors.
constexpr uint32_t MaxWeldKeyWords{18};

std::array<uint32_t, 3> CanonicalTriangle(std::array<uint32_t, 3> triangle) {
    const std::array rotations{triangle, std::array{triangle[1], triangle[2], triangle[0]}, std::array{triangle[2], triangle[0], triangle[1]}};
    return *std::ranges::min_element(rotations);
}

uint32_t FloatBits(float value) { return std::bit_cast<uint32_t>(value); }

uint32_t PackLocalTriangleOffset(uint32_t offset, MeshPrimitiveTopology topology) {
    assert((offset & ~uint32_t(MeshletGeometryEncoding::LocalTriangleOffsetMask)) == 0u);
    return offset | (uint32_t(topology) << uint32_t(MeshletGeometryEncoding::TopologyShift));
}

// A mixed-normal meshlet stores the never-culls cutoff, so the cone test needs no separate flag.
uint32_t PackCone(const meshopt_Bounds &bounds, bool cone_cull_safe) {
    return uint32_t(uint8_t(bounds.cone_axis_s8[0])) |
        uint32_t(uint8_t(bounds.cone_axis_s8[1])) << 8u |
        uint32_t(uint8_t(bounds.cone_axis_s8[2])) << 16u |
        uint32_t(uint8_t(cone_cull_safe ? bounds.cone_cutoff_s8 : int8_t{127})) << 24u;
}
// One meshlet triangle carries one source triangle id and three local indices, so both arenas are
// sized exactly from the source ranges and written in place. Meshlet and vertex counts emerge only
// as meshopt runs, so those two arenas take their final size after the build.
struct MeshletSink {
    std::span<uint32_t> TriangleIds;
    std::span<uint8_t> LocalTriangles;
    std::vector<MeshletRecord> Records{};
    std::vector<uint32_t> Vertices{};
    std::vector<PrimitiveRecord> Primitives{};
    uint32_t TriangleIdCount{}, LocalTriangleCount{};

    void PushTriangleId(uint32_t id) {
        assert(TriangleIdCount < TriangleIds.size());
        TriangleIds[TriangleIdCount++] = id;
    }
    uint32_t RecordCount() const { return uint32_t(Records.size()); }
    uint32_t VertexCount() const { return uint32_t(Vertices.size()); }
    uint32_t PrimitiveCount() const { return uint32_t(Primitives.size()); }
};

// One spatial chunk of a primitive's triangles, holding the meshlets its own clusterization made.
// Vertex offsets stay relative to the chunk's vertex list until the merge places them.
struct MeshletChunk {
    uint32_t FirstTriangle{}, TriangleCount{}, TriangleIdBase{};
    std::vector<MeshletRecord> Records{};
    std::vector<uint32_t> Vertices{};
};

// Split a triangle range at the median of the widest axis, the axis meshopt's own kd-tree picks, so
// every chunk stays one compact spatial region.
void SplitTriangleChunks(std::span<uint32_t> triangles, std::span<const std::array<float, 3>> centroids, uint32_t first, std::vector<MeshletChunk> &chunks) {
    if (triangles.size() <= ChunkTriangles) {
        chunks.emplace_back(MeshletChunk{.FirstTriangle = first, .TriangleCount = uint32_t(triangles.size())});
        return;
    }
    float mean[3]{}, vars[3]{};
    float count = 1, inverse = 1;
    for (const auto triangle : triangles) {
        const auto &point = centroids[triangle];
        for (uint32_t k = 0; k < 3u; ++k) {
            const float delta = point[k] - mean[k];
            mean[k] += delta * inverse;
            vars[k] += delta * (point[k] - mean[k]);
        }
        count += 1.f;
        inverse = 1.f / count;
    }
    const uint32_t axis = (vars[0] >= vars[1] && vars[0] >= vars[2]) ? 0u : (vars[1] >= vars[2] ? 1u : 2u);
    const auto middle = triangles.size() / 2;
    std::nth_element(triangles.begin(), triangles.begin() + middle, triangles.end(), [&](uint32_t a, uint32_t b) { return centroids[a][axis] < centroids[b][axis]; });
    SplitTriangleChunks(triangles.first(middle), centroids, first, chunks);
    SplitTriangleChunks(triangles.subspan(middle), centroids, first + uint32_t(middle), chunks);
}
} // namespace

// Release a mesh's meshlet ranges and take the two whose size the mesh already fixes: one triangle id
// per meshlet triangle and three local indices with it. Serial, because arena offsets follow call order.
void GpuBuffers::ReserveMeshlets(MeshBuffers &buffers, const Mesh &mesh) {
    Meshlets.Release(buffers.Meshlets);
    MeshletTriangleIds.Release(buffers.MeshletTriangles);
    MeshletVertexCorners.Release(buffers.MeshletVertices);
    MeshletLocalTriangles.Release(buffers.MeshletLocalTriangles);
    Primitives.Release(buffers.Primitives);
    buffers.Primitives = buffers.Meshlets = buffers.MeshletTriangles = buffers.MeshletVertices = buffers.MeshletLocalTriangles = {};

    const bool face_topology = mesh.FaceCount() > 0u;
    const uint32_t element_count = face_topology ? 0u : (mesh.EdgeCount() != 0u ? mesh.EdgeCount() : mesh.VertexCount());
    buffers.MeshletTriangles = MeshletTriangleIds.Allocate(face_topology ? mesh.TriangleIndexCount() / 3u : element_count);
    buffers.MeshletLocalTriangles = MeshletLocalTriangles.Allocate(face_topology ? mesh.TriangleIndexCount() : 0u);
}

// Touches only what belongs to its own mesh: the reserved spans are its own, and the counts that
// emerge as meshopt runs come back for the caller to commit.
MeshletBuild GpuBuffers::BuildMeshlets(const MeshBuffers &buffers, const Mesh &mesh, const MeshStore &meshes) {
    // A triangle mesh's draws index the store's corner array, so the clusterizer reads it there. An
    // n-gon mesh fans into a triangulated buffer of its own, written before this build.
    const auto corners = mesh.CornerVertices();
    const auto indices = corners.size() == mesh.TriangleIndexCount() ? corners : FaceIndexBuffer.Get(buffers.FaceIndices);
    assert(indices.size() == mesh.TriangleIndexCount());

    const auto vertices = mesh.GetVerticesSpan();

    const uint32_t store_id = mesh.GetStoreId();
    const auto corner_classes = meshes.GetCornerClasses(store_id);
    const uint32_t uniform_corner_class = meshes.GetCornerClassOffset(store_id);
    const auto face_ids = meshes.GetTriangleFaceIds(store_id);
    const auto element_primitives = meshes.GetElementPrimitiveIndices(store_id);
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

    // Meshlets translate straight into their arenas, so bound every one of them from the source
    // ranges first. The face-less element grouping is part of that bound.
    const bool face_topology = mesh.FaceCount() > 0u;
    const bool line_topology = !face_topology && mesh.EdgeCount() != 0u;
    // Forward shading keeps six deterministic unshared corners per element. Visibility is
    // position-only, so it safely shares four quad vertices and fits sixteen elements in the
    // triangle entry's 64-vertex output contract.
    constexpr uint32_t ElementsPerMeshlet{16u};
    const uint32_t element_count = face_topology ? 0u : (line_topology ? mesh.EdgeCount() : mesh.VertexCount());
    std::vector<uint32_t> edge_indices(line_topology ? element_count * 2u : 0u);
    std::vector<std::vector<uint32_t>> primitive_elements;
    const auto primitive_materials = meshes.GetPrimitiveMaterialRange(store_id);
    if (!face_topology) {
        if (line_topology) mesh.WriteEdgeIndices(edge_indices);
        primitive_elements.resize(std::max(primitive_materials.Count, 1u));
        for (uint32_t element = 0u; element < element_count; ++element) {
            const uint32_t first_vertex = line_topology ? edge_indices[element * 2u] : element;
            const uint32_t primitive = element_primitives.empty() ? 0u : element_primitives[first_vertex];
            assert(primitive < primitive_elements.size());
            if (line_topology && !element_primitives.empty()) {
                assert(element_primitives[edge_indices[element * 2u + 1u]] == primitive);
            }
            primitive_elements[primitive].push_back(element);
        }
    }

    MeshletSink sink{
        .TriangleIds = MeshletTriangleIds.GetMutable(buffers.MeshletTriangles),
        .LocalTriangles = MeshletLocalTriangles.GetMutable(buffers.MeshletLocalTriangles),
    };

    const uint32_t live_uv_sets = uint32_t(std::ranges::count_if(corner_uvs, [](auto uvs) { return !uvs.empty(); }));
    const uint32_t weld_key_words = 2u + 2u * live_uv_sets + (corner_tangents.empty() ? 0u : 4u) + (corner_colors.empty() ? 0u : 4u);

    std::vector<uint8_t> flat_face_triangles;
    std::vector<uint32_t> chunk_triangles;
    std::vector<MeshletChunk> chunks;
    // One chunk's clusterization. It welds only its own corners and fills only the arena range its
    // triangle count already fixed, so every chunk of a primitive runs at once.
    const auto build_chunk = [&](const PrimitiveTriangleRange &primitive, std::span<const uint32_t> primitive_indices, uint32_t primitive_record, MeshletChunk &chunk) {
        if (chunk.TriangleCount == 0u) return;

        const uint32_t first_index = primitive.FirstTriangle * 3u;
        const uint32_t corner_count = chunk.TriangleCount * 3u;
        const auto chunk_triangle_ids = std::span{chunk_triangles}.subspan(chunk.FirstTriangle, chunk.TriangleCount);

        // Meshlets are built inside one source-primitive range. Coverage classification therefore
        // remains uniform across every resulting meshlet and can be done once in the meshlet cull.
        for (const auto triangle : chunk_triangle_ids) {
            assert(face_ids[primitive.FirstTriangle + triangle] > 0u && face_ids[primitive.FirstTriangle + triangle] <= element_primitives.size());
            assert(element_primitives[face_ids[primitive.FirstTriangle + triangle] - 1u] == primitive.PrimitiveIndex);
        }
        if (!morph_shading_authored) {
            for (const auto triangle : chunk_triangle_ids) {
                for (uint32_t c = 0; c < 3u; ++c) {
                    const uint32_t global_corner = first_index + triangle * 3u + c;
                    const uint32_t tag = corner_class_value(global_corner) >> uint32_t(CornerClassEncoding::TagShift);
                    if (tag != uint32_t(CornerClass::Face) || has_custom_normal(global_corner)) {
                        flat_face_triangles[triangle] = 0u;
                        break;
                    }
                }
            }
        }

        std::vector<uint32_t> welded_indices(corner_count, 0u), representative_corners;
        std::vector<std::array<float, 3>> welded_positions;
        FlatKeyMap welded;
        welded.Reset(weld_key_words, corner_count);
        const auto append_render_vertex = [&](uint32_t corner, uint32_t chunk_corner) {
            const uint32_t render_vertex = uint32_t(representative_corners.size());
            welded_indices[chunk_corner] = render_vertex;
            representative_corners.push_back(corner | (flat_face_triangles[corner / 3u] ? uint32_t(MeshletGeometryEncoding::FlatVertexBit) : 0u));
            const vec3 position = vertices[primitive_indices[corner]].Position;
            welded_positions.push_back({position.x, position.y, position.z});
            return render_vertex;
        };
        for (uint32_t i = 0; i < chunk.TriangleCount; ++i) {
            for (uint32_t c = 0; c < 3u; ++c) {
                const uint32_t corner = chunk_triangle_ids[i] * 3u + c;
                const uint32_t chunk_corner = i * 3u + c;
                const uint32_t global_corner = first_index + corner;
                // A custom-normal corner's frame depends on its own triangle, so it welds with nothing.
                if (has_custom_normal(global_corner)) {
                    append_render_vertex(corner, chunk_corner);
                    continue;
                }
                std::array<uint32_t, MaxWeldKeyWords> words{};
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
                    if (uvs.empty()) continue;
                    const vec2 uv = uvs[global_corner];
                    words[word++] = FloatBits(uv.x);
                    words[word++] = FloatBits(uv.y);
                }
                if (!corner_tangents.empty()) {
                    const vec4 tangent = corner_tangents[global_corner];
                    words[word++] = FloatBits(tangent.x);
                    words[word++] = FloatBits(tangent.y);
                    words[word++] = FloatBits(tangent.z);
                    words[word++] = FloatBits(tangent.w);
                }
                if (!corner_colors.empty()) {
                    const vec4 color = corner_colors[global_corner];
                    words[word++] = FloatBits(color.x);
                    words[word++] = FloatBits(color.y);
                    words[word++] = FloatBits(color.z);
                    words[word++] = FloatBits(color.w);
                }
                assert(word == weld_key_words);

                if (const auto *found = welded.Find(words.data())) {
                    welded_indices[chunk_corner] = *found;
                    continue;
                }
                welded.Insert(words.data(), append_render_vertex(corner, chunk_corner));
            }
        }

        const auto bound = meshopt_buildMeshletsBound(corner_count, MeshletMaxVertices, MeshletMaxTriangles);
        std::vector<meshopt_Meshlet> built(bound);
        std::vector<uint32_t> local_vertices(bound * MeshletMaxVertices);
        std::vector<uint8_t> local_meshlet_triangles(bound * MeshletMaxTriangles * 3u);
        const auto meshlet_count = meshopt_buildMeshlets(
            built.data(), local_vertices.data(), local_meshlet_triangles.data(), welded_indices.data(), welded_indices.size(),
            welded_positions.front().data(), welded_positions.size(), sizeof(welded_positions.front()),
            MeshletMaxVertices, MeshletMaxTriangles, 0.5f
        );
        built.resize(meshlet_count);

        FlatKeyMap source_triangles;
        source_triangles.Reset(3u, chunk.TriangleCount);
        for (uint32_t i = 0; i < chunk.TriangleCount; ++i) {
            const auto offset = size_t(i) * 3;
            const auto source_key = CanonicalTriangle({welded_indices[offset], welded_indices[offset + 1], welded_indices[offset + 2]});
            source_triangles.Insert(source_key.data(), primitive.FirstTriangle + chunk_triangle_ids[i]);
        }

        const auto triangle_ids = sink.TriangleIds.subspan(chunk.TriangleIdBase, chunk.TriangleCount);
        const auto local_triangles = sink.LocalTriangles.subspan(size_t(chunk.TriangleIdBase) * 3, size_t(chunk.TriangleCount) * 3);
        uint32_t triangle_id_count = 0, local_triangle_count = 0;
        for (const auto &meshlet : built) {
            const uint32_t first_triangle_id = triangle_id_count;
            const uint32_t first_local_triangle = local_triangle_count;
            const uint32_t first_vertex = uint32_t(chunk.Vertices.size());
            for (uint32_t v = 0; v < meshlet.vertex_count; ++v) {
                chunk.Vertices.push_back(representative_corners[local_vertices[meshlet.vertex_offset + v]]);
            }
            bool cone_cull_safe = true;
            for (uint32_t t = 0; t < meshlet.triangle_count; ++t) {
                std::array<uint32_t, 3> triangle;
                std::array<uint8_t, 3> local_triangle;
                for (uint32_t c = 0; c < 3; ++c) {
                    local_triangle[c] = local_meshlet_triangles[meshlet.triangle_offset + t * 3 + c];
                    triangle[c] = local_vertices[meshlet.vertex_offset + local_triangle[c]];
                }
                const auto key = CanonicalTriangle(triangle);
                auto *source = source_triangles.Find(key.data());
                assert(source != nullptr);
                const uint32_t source_index = *source;
                const uint32_t source_triangle = source_index - primitive.FirstTriangle;
                *source = FlatKeyMap::Taken;
                cone_cull_safe &= flat_face_triangles[source_triangle] != 0u;
                for (uint32_t c = 0; c < 3u; ++c) {
                    local_triangles[local_triangle_count++] = uint8_t(local_triangle[c] | (c == 0u && flat_face_triangles[source_triangle] ? uint8_t(MeshletGeometryEncoding::FlatTriangleBit) : 0u));
                }
                triangle_ids[triangle_id_count++] = source_index;
            }
            const auto bounds = meshopt_computeMeshletBounds(
                local_vertices.data() + meshlet.vertex_offset,
                local_meshlet_triangles.data() + meshlet.triangle_offset,
                meshlet.triangle_count, welded_positions.front().data(), welded_positions.size(), sizeof(welded_positions.front())
            );
            chunk.Records.emplace_back(MeshletRecord{
                .TriangleOffset = buffers.MeshletTriangles.Offset + chunk.TriangleIdBase + first_triangle_id,
                .TriangleCount = meshlet.triangle_count,
                .VertexOffset = first_vertex,
                .VertexCount = meshlet.vertex_count,
                .LocalTriangleOffset = PackLocalTriangleOffset(buffers.MeshletLocalTriangles.Offset + chunk.TriangleIdBase * 3u + first_local_triangle, MeshPrimitiveTopology::Triangle),
                .Primitive = primitive_record,
                .ConeAxisCutoff = PackCone(bounds, cone_cull_safe),
                .Center = {bounds.center[0], bounds.center[1], bounds.center[2]},
                .Radius = bounds.radius,
            });
        }
        // Every source triangle lands in exactly one meshlet, which is what fixes each chunk's range.
        assert(triangle_id_count == chunk.TriangleCount);
    };

    for (const auto &primitive : meshes.GetPrimitiveTriangleRanges(mesh.GetStoreId())) {
        const auto first_index = size_t(primitive.FirstTriangle) * 3;
        const auto index_count = size_t(primitive.TriangleCount) * 3;
        const auto primitive_indices = std::span{indices}.subspan(first_index, index_count);

        flat_face_triangles.assign(primitive.TriangleCount, morph_shading_authored ? 0u : 1u);
        chunk_triangles.resize(primitive.TriangleCount);
        std::iota(chunk_triangles.begin(), chunk_triangles.end(), 0u);
        chunks.clear();
        if (primitive.TriangleCount <= ChunkTriangles) {
            chunks.emplace_back(MeshletChunk{.TriangleCount = primitive.TriangleCount});
        } else {
            // Splitting reads every centroid once per level, so gather them once into their own array.
            constexpr uint32_t CentroidBlock{16u * 1024u};
            std::vector<std::array<float, 3>> centroids(primitive.TriangleCount);
            ParallelFor((primitive.TriangleCount + CentroidBlock - 1u) / CentroidBlock, [&](uint32_t block) {
                const uint32_t last = std::min((block + 1u) * CentroidBlock, primitive.TriangleCount);
                for (uint32_t triangle = block * CentroidBlock; triangle < last; ++triangle) {
                    const vec3 centroid = (vec3{vertices[primitive_indices[triangle * 3u]].Position} +
                                           vec3{vertices[primitive_indices[triangle * 3u + 1u]].Position} +
                                           vec3{vertices[primitive_indices[triangle * 3u + 2u]].Position}) /
                        3.f;
                    centroids[triangle] = {centroid.x, centroid.y, centroid.z};
                }
            });
            SplitTriangleChunks(chunk_triangles, centroids, 0u, chunks);
        }

        uint32_t triangle_id_base = sink.TriangleIdCount;
        for (auto &chunk : chunks) {
            chunk.TriangleIdBase = triangle_id_base;
            triangle_id_base += chunk.TriangleCount;
        }
        const uint32_t primitive_record = sink.PrimitiveCount();
        ParallelFor(uint32_t(chunks.size()), [&](uint32_t i) { build_chunk(primitive, primitive_indices, primitive_record, chunks[i]); });

        // Chunks merge in split order, so the meshlets a mesh produces never depend on which chunk
        // finished first.
        const uint32_t first_meshlet = sink.RecordCount();
        size_t record_total = 0, vertex_total = 0;
        for (const auto &chunk : chunks) {
            record_total += chunk.Records.size();
            vertex_total += chunk.Vertices.size();
        }
        sink.Records.reserve(sink.Records.size() + record_total);
        sink.Vertices.reserve(sink.Vertices.size() + vertex_total);
        for (auto &chunk : chunks) {
            const uint32_t vertex_base = sink.VertexCount();
            for (auto &record : chunk.Records) record.VertexOffset += vertex_base;
            sink.Records.insert(sink.Records.end(), chunk.Records.begin(), chunk.Records.end());
            sink.Vertices.insert(sink.Vertices.end(), chunk.Vertices.begin(), chunk.Vertices.end());
            sink.TriangleIdCount += chunk.TriangleCount;
            sink.LocalTriangleCount += chunk.TriangleCount * 3u;
            chunk.Records = {};
            chunk.Vertices = {};
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
            .FaceFirstTriangleOffset = meshes.GetFaceDataRange(mesh.GetStoreId()).Offset,
            .VertexEdgeAdjacencyOffset = OffsetOrInvalid(meshes.GetVertexEdgeAdjacencyRange(mesh.GetStoreId())),
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
        sink.Primitives.emplace_back(PrimitiveRecord{
            .Draw = draw,
            .PrimitiveIndex = primitive.PrimitiveIndex,
            .FirstTriangle = primitive.FirstTriangle,
            .MeshletOffset = first_meshlet,
            .MeshletCount = sink.RecordCount() - first_meshlet,
        });
    }

    if (!face_topology) {
        for (uint32_t primitive_index = 0u; primitive_index < primitive_elements.size(); ++primitive_index) {
            const auto &elements = primitive_elements[primitive_index];
            if (elements.empty()) continue;
            const uint32_t first_meshlet = sink.RecordCount();
            for (uint32_t base = 0u; base < elements.size(); base += ElementsPerMeshlet) {
                const uint32_t count = std::min(ElementsPerMeshlet, uint32_t(elements.size()) - base);
                const uint32_t first_element_id = sink.TriangleIdCount;
                const uint32_t first_vertex = sink.VertexCount();
                vec3 lo(std::numeric_limits<float>::max());
                vec3 hi(std::numeric_limits<float>::lowest());
                for (uint32_t i = 0u; i < count; ++i) {
                    const uint32_t element = elements[base + i];
                    sink.PushTriangleId(element);
                    const uint32_t vertex_count = line_topology ? 2u : 1u;
                    for (uint32_t endpoint = 0u; endpoint < vertex_count; ++endpoint) {
                        const uint32_t vertex = line_topology ? edge_indices[element * 2u + endpoint] : element;
                        sink.Vertices.push_back(vertex);
                        const vec3 position = vertices[vertex].Position;
                        lo = glm::min(lo, position);
                        hi = glm::max(hi, position);
                    }
                }
                const vec3 center = (lo + hi) * 0.5f;
                float radius = 0.0f;
                for (uint32_t i = first_vertex; i < sink.VertexCount(); ++i) {
                    radius = std::max(radius, glm::distance(center, vec3(vertices[sink.Vertices[i]].Position)));
                }
                sink.Records.emplace_back(MeshletRecord{
                    .TriangleOffset = buffers.MeshletTriangles.Offset + first_element_id,
                    .TriangleCount = count,
                    .VertexOffset = first_vertex,
                    .VertexCount = count * (line_topology ? 2u : 1u),
                    .LocalTriangleOffset = PackLocalTriangleOffset(buffers.MeshletLocalTriangles.Offset + sink.LocalTriangleCount, line_topology ? MeshPrimitiveTopology::Line : MeshPrimitiveTopology::Point),
                    .Primitive = sink.PrimitiveCount(),
                    .ConeAxisCutoff = uint32_t(uint8_t(127)) << 24u,
                    .Center = center,
                    .Radius = radius,
                });
            }

            DrawData draw{
                .VertexSlot = buffers.Vertices.Slot,
                .IndexSlotOffset = line_topology ? buffers.EdgeIndices : buffers.VertexIndices,
                .ModelSlot = Instances.TransformBuffer.Slot,
                .ObjectIdSlot = InvalidSlot,
                .CornerColorOffset = OffsetOrInvalid(meshes.GetCornerColorRange(store_id)),
                .VertexCountOrHeadImageSlot = buffers.Vertices.Count,
                .InstanceStateSlot = Instances.StateBuffer.Slot,
                .VertexOffset = buffers.Vertices.Offset,
                .PrimitiveMaterialOffset = OffsetOrInvalid(primitive_materials),
                .ElementPrimitiveOffset = OffsetOrInvalid(meshes.GetElementPrimitiveRange(store_id)),
            };
            sink.Primitives.emplace_back(PrimitiveRecord{
                .Draw = draw,
                .PrimitiveIndex = primitive_index,
                .FirstTriangle = 0u,
                .MeshletOffset = first_meshlet,
                .MeshletCount = sink.RecordCount() - first_meshlet,
            });
        }
    }

    return {
        .Records = std::move(sink.Records),
        .Vertices = std::move(sink.Vertices),
        .Primitives = std::move(sink.Primitives),
        .TriangleIdCount = sink.TriangleIdCount,
        .LocalTriangleCount = sink.LocalTriangleCount,
    };
}

// Take the arenas whose size only the finished build knows, and trim the two it filled in place.
// Serial, because arena offsets follow call order.
void GpuBuffers::CommitMeshlets(MeshBuffers &buffers, MeshletBuild &build) {
    MeshletTriangleIds.Shrink(buffers.MeshletTriangles, build.TriangleIdCount);
    MeshletLocalTriangles.Shrink(buffers.MeshletLocalTriangles, build.LocalTriangleCount);

    buffers.MeshletVertices = MeshletVertexCorners.Allocate(build.Vertices);
    for (auto &record : build.Records) record.VertexOffset += buffers.MeshletVertices.Offset;
    buffers.Meshlets = Meshlets.Allocate(build.Records);
    for (auto &record : build.Primitives) record.MeshletOffset += buffers.Meshlets.Offset;
    buffers.Primitives = Primitives.Allocate(build.Primitives);
    for (auto &record : Meshlets.GetMutable(buffers.Meshlets)) record.Primitive += buffers.Primitives.Offset;
}
