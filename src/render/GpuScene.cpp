#include "render/GpuBuffers.h"
#include "render/MeshletBuild.h"

#include "FlatKeyMap.h"
#include "Parallel.h"
#include "gpu/CornerClassEncoding.h"
#include "gpu/MeshPrimitiveTopology.h"
#include "gpu/MeshletEditEdgeEncoding.h"
#include "gpu/MeshletGeometryEncoding.h"
#include "gpu/MeshletLimit.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"

#include "meshoptimizer.h"

#include <limits>
#include <numeric>

namespace {
constexpr size_t MeshletMaxVertices{size_t(MeshletLimit::MaxVertices)};
constexpr size_t MeshletMaxTriangles{size_t(MeshletLimit::MaxTriangles)};
// A primitive with more triangles than this is split into spatial chunks that clusterize on their
// own. The split comes from the triangle count alone, so a mesh's meshlets never depend on how many
// cores ran the build.
constexpr uint32_t ChunkTriangles{128u * 1024u};

std::array<uint32_t, 3> CanonicalTriangle(std::array<uint32_t, 3> triangle) {
    const std::array rotations{triangle, std::array{triangle[1], triangle[2], triangle[0]}, std::array{triangle[2], triangle[0], triangle[1]}};
    return *std::ranges::min_element(rotations);
}

uint32_t PackLocalTriangleOffset(uint32_t offset, MeshPrimitiveTopology topology) {
    assert((offset & ~uint32_t(MeshletGeometryEncoding::LocalTriangleOffsetMask)) == 0u);
    return offset | (uint32_t(topology) << uint32_t(MeshletGeometryEncoding::TopologyShift));
}

std::vector<uint32_t> BuildTriangleEditEdges(const Mesh &mesh, std::span<const uint32_t> face_first_triangles) {
    std::vector<uint32_t> result(mesh.TriangleIndexCount(), InvalidOffset);
    const auto write = [&](Mesh::HH halfedge, uint32_t slot) {
        const uint32_t edge = *mesh.GetEdge(halfedge);
        const auto canonical = mesh.GetHalfedge(Mesh::EH{edge}, 0u);
        const bool reversed = mesh.GetFromVertex(canonical) != mesh.GetFromVertex(halfedge);
        assert(mesh.GetFromVertex(canonical) == (reversed ? mesh.GetToVertex(halfedge) : mesh.GetFromVertex(halfedge)));
        assert(mesh.GetToVertex(canonical) == (reversed ? mesh.GetFromVertex(halfedge) : mesh.GetToVertex(halfedge)));
        assert(edge <= uint32_t(MeshletEditEdgeEncoding::EdgeMask));
        result[slot] = edge | (reversed ? uint32_t(MeshletEditEdgeEncoding::ReversedBit) : 0u);
    };
    for (const auto face : mesh.faces()) {
        const auto halfedges = mesh.fh_range(face);
        auto it = halfedges.begin();
        const auto end = halfedges.end();
        const auto first_halfedge = *it++;
        const auto second = *it++;
        const uint32_t first_triangle = face_first_triangles[*face];
        for (uint32_t triangle = 0u; it != end; ++it, ++triangle) {
            const uint32_t base = (first_triangle + triangle) * 3u;
            if (triangle == 0u) write(second, base);
            write(*it, base + 1u);
            if (auto next = it; ++next == end) write(first_halfedge, base + 2u);
        }
    }
    return result;
}

// A mixed-normal meshlet stores the never-culls cutoff, so the cone test needs no separate flag.
uint32_t PackCone(const meshopt_Bounds &bounds, bool cone_cull_safe) {
    return uint32_t(uint8_t(bounds.cone_axis_s8[0])) |
        uint32_t(uint8_t(bounds.cone_axis_s8[1])) << 8u |
        uint32_t(uint8_t(bounds.cone_axis_s8[2])) << 16u |
        uint32_t(uint8_t(cone_cull_safe ? bounds.cone_cutoff_s8 : int8_t{127})) << 24u;
}
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

// Draw data for one of a mesh's source triangle primitives, whose corner offsets start at its first triangle.
DrawData PrimitiveDrawData(const GpuBuffers &buffers, const MeshBuffers &mb, const MeshStore &meshes, uint32_t store_id, const PrimitiveTriangleRange &primitive) {
    const auto first_index = size_t(primitive.FirstTriangle) * 3;
    DrawData draw{
        .VertexSlot = mb.Vertices.Slot,
        .IndexSlotOffset = {mb.FaceIndices.Slot, mb.FaceIndices.Offset + uint32_t(first_index)},
        .ModelSlot = buffers.Instances.TransformBuffer.Slot,
        .ObjectIdSlot = meshes.GetFaceIdRange(store_id).Slot,
        .CornerClassOffset = meshes.GetCornerClassOffset(store_id),
        .CustomCornerMaskOffset = OffsetOrInvalid(meshes.GetCustomCornerMaskRange(store_id)),
        .CustomCornerNormalOffset = OffsetOrInvalid(meshes.GetCustomCornerNormalRange(store_id)),
        .CornerBase = uint32_t(first_index),
        .BaseSeamNormalOffset = OffsetOrInvalid(meshes.GetBaseSeamNormalRange(store_id)),
        .CornerTangentOffset = OffsetOrInvalid(meshes.GetCornerTangentRange(store_id)),
        .CornerColorOffset = OffsetOrInvalid(meshes.GetCornerColorRange(store_id)),
        .FaceIdOffset = meshes.GetFaceIdRange(store_id).Offset + primitive.FirstTriangle,
        .BaseFaceNormalOffset = meshes.GetFaceDataRange(store_id).Offset,
        .FaceFirstTriangleOffset = meshes.GetFaceDataRange(store_id).Offset,
        .VertexEdgeAdjacencyOffset = OffsetOrInvalid(meshes.GetVertexEdgeAdjacencyRange(store_id)),
        .VertexCountOrHeadImageSlot = mb.Vertices.Count,
        .ElementStateSlotOffset = meshes.GetFaceStateRange(store_id),
        .EditEdgeOffset = 0u,
        .InstanceStateSlot = buffers.Instances.StateBuffer.Slot,
        .VertexOffset = mb.Vertices.Offset,
        .MorphShadingAuthored = meshes.GetMorphShadingAuthored(store_id) ? 1u : 0u,
        .PrimitiveMaterialOffset = OffsetOrInvalid(meshes.GetPrimitiveMaterialRange(store_id)),
        .ElementPrimitiveOffset = OffsetOrInvalid(meshes.GetElementPrimitiveRange(store_id)),
    };
    if (draw.CornerClassOffset < uint32_t(CornerClassEncoding::UniformFaceOffset)) draw.CornerClassOffset += uint32_t(first_index);
    const auto advance_corner = [first_index](uint32_t &offset) {
        if (offset != InvalidOffset) offset += uint32_t(first_index);
    };
    advance_corner(draw.CornerTangentOffset);
    advance_corner(draw.CornerColorOffset);
    for (uint32_t set = 0; set < draw.CornerUvOffsets.size(); ++set) {
        draw.CornerUvOffsets[set] = OffsetOrInvalid(meshes.GetCornerUvRange(store_id, set));
        advance_corner(draw.CornerUvOffsets[set]);
    }
    return draw;
}
} // namespace

MeshletBuildInputs CaptureMeshletInputs(const GpuBuffers &buffers, const MeshBuffers &mb, const Mesh &mesh, const MeshStore &meshes) {
    const uint32_t store_id = mesh.GetStoreId();
    // A triangle mesh's draws index the store's corner array, so the clusterizer reads it there. An
    // n-gon mesh fans into a triangulated buffer of its own, written before this build.
    const auto corners = mesh.CornerVertices();
    const auto indices = corners.size() == mesh.TriangleIndexCount() ? corners : buffers.FaceIndexBuffer.Get(mb.FaceIndices);
    assert(indices.size() == mesh.TriangleIndexCount());

    const bool face_topology = mesh.FaceCount() > 0u;
    const bool line_topology = !face_topology && mesh.EdgeCount() != 0u;
    const auto primitive_ranges = meshes.GetPrimitiveTriangleRanges(store_id);
    MeshletBuildInputs inputs{
        .Indices = indices,
        .Vertices = mesh.GetVerticesSpan(),
        .ElementPrimitives = meshes.GetElementPrimitiveIndices(store_id),
        .TriangleEditEdges = face_topology ? BuildTriangleEditEdges(mesh, meshes.GetFaceFirstTriangles(store_id)) : std::vector<uint32_t>{},
        .PrimitiveTriangleRanges = {primitive_ranges.begin(), primitive_ranges.end()},
        .Weld = {
            .CornerClassOffset = meshes.GetCornerClassOffset(store_id),
            .CornerClasses = meshes.GetCornerClasses(store_id),
            .TriangleFaceIds = meshes.GetTriangleFaceIds(store_id),
            .CustomCornerMasks = meshes.GetCustomCornerMasks(store_id),
            .CornerUvs = {
                meshes.GetCornerUvs(store_id, 0),
                meshes.GetCornerUvs(store_id, 1),
                meshes.GetCornerUvs(store_id, 2),
                meshes.GetCornerUvs(store_id, 3),
            },
            .CornerTangents = meshes.GetCornerTangents(store_id),
            .CornerColors = meshes.GetCornerColors(store_id),
            .MorphShadingAuthored = meshes.GetMorphShadingAuthored(store_id),
        },
        .TriangleCount = mesh.TriangleIndexCount() / 3u,
        .ElementCount = face_topology ? 0u : (line_topology ? mesh.EdgeCount() : mesh.VertexCount()),
        .EdgeCount = mesh.EdgeCount(),
        .SourcePrimitiveCount = meshes.GetPrimitiveMaterialRange(store_id).Count,
        .FaceTopology = face_topology,
        .LineTopology = line_topology,
    };
    inputs.PrimitiveDraws.reserve(primitive_ranges.size());
    for (const auto &primitive : primitive_ranges) {
        inputs.PrimitiveDraws.push_back(PrimitiveDrawData(buffers, mb, meshes, store_id, primitive));
    }
    if (!face_topology) {
        if (line_topology) {
            inputs.EdgeIndices.resize(size_t(inputs.ElementCount) * 2u);
            mesh.WriteEdgeIndices(inputs.EdgeIndices);
        }
        inputs.ElementDraw = {
            .VertexSlot = mb.Vertices.Slot,
            .IndexSlotOffset = line_topology ? mb.EdgeIndices : mb.VertexIndices,
            .ModelSlot = buffers.Instances.TransformBuffer.Slot,
            .ObjectIdSlot = InvalidSlot,
            .CornerColorOffset = OffsetOrInvalid(meshes.GetCornerColorRange(store_id)),
            .VertexCountOrHeadImageSlot = mb.Vertices.Count,
            .InstanceStateSlot = buffers.Instances.StateBuffer.Slot,
            .VertexOffset = mb.Vertices.Offset,
            .PrimitiveMaterialOffset = OffsetOrInvalid(meshes.GetPrimitiveMaterialRange(store_id)),
            .ElementPrimitiveOffset = OffsetOrInvalid(meshes.GetElementPrimitiveRange(store_id)),
        };
    }
    return inputs;
}

// Touches only its own captured inputs and its own vectors, so this runs on any thread.
MeshletBuild BuildMeshlets(MeshletBuildInputs &in) {
    const auto vertices = in.Vertices;
    const auto face_ids = in.Weld.TriangleFaceIds;
    const auto element_primitives = in.ElementPrimitives;

    const bool face_topology = in.FaceTopology;
    const bool line_topology = in.LineTopology;
    // Forward shading keeps six deterministic unshared corners per element. Visibility is
    // position-only, so it safely shares four quad vertices and fits sixteen elements in the
    // triangle entry's 64-vertex output contract.
    constexpr uint32_t ElementsPerMeshlet{16u};
    const uint32_t element_count = in.ElementCount;
    const auto &edge_indices = in.EdgeIndices;
    std::vector<std::vector<uint32_t>> primitive_elements;
    if (!face_topology) {
        primitive_elements.resize(std::max(in.SourcePrimitiveCount, 1u));
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

    // One meshlet triangle carries one source triangle id and three local indices, so both lists are
    // sized exactly from the source ranges and written in place.
    MeshletBuild sink{
        .TriangleIds = std::vector<uint32_t>(face_topology ? in.TriangleCount : element_count),
        .LocalTriangles = std::vector<uint8_t>(face_topology ? size_t(in.TriangleCount) * 3 : 0),
        .EditEdges = std::move(in.TriangleEditEdges),
    };

    std::vector<uint8_t> flat_face_triangles;
    std::vector<uint32_t> chunk_triangles;
    std::vector<MeshletChunk> chunks;
    // One chunk's clusterization. It welds only its own corners and fills only the list range its
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
        const CornerWeldKey key{in.Weld, first_index};
        for (const auto triangle : chunk_triangle_ids) flat_face_triangles[triangle] = key.FlatFaceTriangle(triangle);

        std::vector<uint32_t> welded_indices(corner_count, 0u), representative_corners;
        std::vector<std::array<float, 3>> welded_positions;
        FlatKeyMap welded;
        welded.Reset(key.WordCount(), corner_count);
        const auto append_render_vertex = [&](uint32_t corner, uint32_t chunk_corner) {
            const uint32_t render_vertex = uint32_t(representative_corners.size());
            welded_indices[chunk_corner] = render_vertex;
            representative_corners.push_back(corner | (flat_face_triangles[corner / 3u] ? uint32_t(MeshletGeometryEncoding::FlatVertexBit) : 0u));
            const vec3 position = vertices[primitive_indices[corner]].Position;
            welded_positions.push_back({position.x, position.y, position.z});
            return render_vertex;
        };
        std::array<uint32_t, MaxWeldKeyWords> words{};
        for (uint32_t i = 0; i < chunk.TriangleCount; ++i) {
            for (uint32_t c = 0; c < 3u; ++c) {
                const uint32_t corner = chunk_triangle_ids[i] * 3u + c;
                const uint32_t chunk_corner = i * 3u + c;
                if (key.WeldsAlone(corner)) {
                    append_render_vertex(corner, chunk_corner);
                    continue;
                }
                key.Write(corner, primitive_indices[corner], flat_face_triangles[corner / 3u], words);
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
            source_triangles.Insert(source_key.data(), i);
        }

        const auto triangle_ids = std::span{sink.TriangleIds}.subspan(chunk.TriangleIdBase, chunk.TriangleCount);
        const auto local_triangles = std::span{sink.LocalTriangles}.subspan(size_t(chunk.TriangleIdBase) * 3, size_t(chunk.TriangleCount) * 3);
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
                const uint32_t chunk_triangle = *source;
                const uint32_t source_triangle = chunk_triangle_ids[chunk_triangle];
                const uint32_t source_index = primitive.FirstTriangle + source_triangle;
                *source = FlatKeyMap::Taken;
                cone_cull_safe &= flat_face_triangles[source_triangle] != 0u;
                uint32_t rotation = 0u;
                for (; rotation < 3u; ++rotation) {
                    bool matches = true;
                    for (uint32_t c = 0u; c < 3u; ++c) {
                        matches &= triangle[c] == welded_indices[chunk_triangle * 3u + (rotation + c) % 3u];
                    }
                    if (matches) break;
                }
                assert(rotation < 3u);
                const std::array source_edit_edges{
                    sink.EditEdges[source_index * 3u],
                    sink.EditEdges[source_index * 3u + 1u],
                    sink.EditEdges[source_index * 3u + 2u],
                };
                for (uint32_t c = 0; c < 3u; ++c) {
                    local_triangles[local_triangle_count++] = uint8_t(local_triangle[c] | (c == 0u && flat_face_triangles[source_triangle] ? uint8_t(MeshletGeometryEncoding::FlatTriangleBit) : 0u));
                    sink.EditEdges[source_index * 3u + c] = source_edit_edges[(rotation + c) % 3u];
                }
                triangle_ids[triangle_id_count++] = source_index;
            }
            const auto bounds = meshopt_computeMeshletBounds(
                local_vertices.data() + meshlet.vertex_offset,
                local_meshlet_triangles.data() + meshlet.triangle_offset,
                meshlet.triangle_count, welded_positions.front().data(), welded_positions.size(), sizeof(welded_positions.front())
            );
            chunk.Records.emplace_back(MeshletRecord{
                .TriangleOffset = chunk.TriangleIdBase + first_triangle_id,
                .TriangleCount = meshlet.triangle_count,
                .VertexOffset = first_vertex,
                .VertexCount = meshlet.vertex_count,
                .LocalTriangleOffset = PackLocalTriangleOffset(chunk.TriangleIdBase * 3u + first_local_triangle, MeshPrimitiveTopology::Triangle),
                .Primitive = primitive_record,
                .ConeAxisCutoff = PackCone(bounds, cone_cull_safe),
                .Center = {bounds.center[0], bounds.center[1], bounds.center[2]},
                .Radius = bounds.radius,
            });
        }
        // Every source triangle lands in exactly one meshlet, which is what fixes each chunk's range.
        assert(triangle_id_count == chunk.TriangleCount);
    };

    for (uint32_t primitive_record_index = 0u; primitive_record_index < in.PrimitiveTriangleRanges.size(); ++primitive_record_index) {
        const auto &primitive = in.PrimitiveTriangleRanges[primitive_record_index];
        const auto first_index = size_t(primitive.FirstTriangle) * 3;
        const auto index_count = size_t(primitive.TriangleCount) * 3;
        const auto primitive_indices = in.Indices.subspan(first_index, index_count);

        flat_face_triangles.assign(primitive.TriangleCount, 0u);
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
        const uint32_t primitive_record = uint32_t(sink.Primitives.size());
        ParallelFor(uint32_t(chunks.size()), [&](uint32_t i) { build_chunk(primitive, primitive_indices, primitive_record, chunks[i]); });

        // Chunks merge in split order, so the meshlets a mesh produces never depend on which chunk
        // finished first.
        const uint32_t first_meshlet = uint32_t(sink.Records.size());
        size_t record_total = 0, vertex_total = 0;
        for (const auto &chunk : chunks) {
            record_total += chunk.Records.size();
            vertex_total += chunk.Vertices.size();
        }
        sink.Records.reserve(sink.Records.size() + record_total);
        sink.Vertices.reserve(sink.Vertices.size() + vertex_total);
        for (auto &chunk : chunks) {
            const uint32_t vertex_base = uint32_t(sink.Vertices.size());
            for (auto &record : chunk.Records) record.VertexOffset += vertex_base;
            sink.Records.insert(sink.Records.end(), chunk.Records.begin(), chunk.Records.end());
            sink.Vertices.insert(sink.Vertices.end(), chunk.Vertices.begin(), chunk.Vertices.end());
            sink.TriangleIdCount += chunk.TriangleCount;
            sink.LocalTriangleCount += chunk.TriangleCount * 3u;
            chunk.Records = {};
            chunk.Vertices = {};
        }

        sink.Primitives.emplace_back(PrimitiveRecord{
            .Draw = in.PrimitiveDraws[primitive_record_index],
            .PrimitiveIndex = primitive.PrimitiveIndex,
            .FirstTriangle = primitive.FirstTriangle,
            .MeshletOffset = first_meshlet,
            .MeshletCount = uint32_t(sink.Records.size()) - first_meshlet,
            .Level0Count = uint32_t(sink.Records.size()) - first_meshlet,
        });
    }

    if (!face_topology) {
        for (uint32_t primitive_index = 0u; primitive_index < primitive_elements.size(); ++primitive_index) {
            const auto &elements = primitive_elements[primitive_index];
            if (elements.empty()) continue;
            const uint32_t first_meshlet = uint32_t(sink.Records.size());
            for (uint32_t base = 0u; base < elements.size(); base += ElementsPerMeshlet) {
                const uint32_t count = std::min(ElementsPerMeshlet, uint32_t(elements.size()) - base);
                const uint32_t first_element_id = sink.TriangleIdCount;
                const uint32_t first_vertex = uint32_t(sink.Vertices.size());
                vec3 lo(std::numeric_limits<float>::max());
                vec3 hi(std::numeric_limits<float>::lowest());
                for (uint32_t i = 0u; i < count; ++i) {
                    const uint32_t element = elements[base + i];
                    assert(sink.TriangleIdCount < sink.TriangleIds.size());
                    sink.TriangleIds[sink.TriangleIdCount++] = element;
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
                for (uint32_t i = first_vertex; i < uint32_t(sink.Vertices.size()); ++i) {
                    radius = std::max(radius, glm::distance(center, vec3(vertices[sink.Vertices[i]].Position)));
                }
                sink.Records.emplace_back(MeshletRecord{
                    .TriangleOffset = first_element_id,
                    .TriangleCount = count,
                    .VertexOffset = first_vertex,
                    .VertexCount = count * (line_topology ? 2u : 1u),
                    .LocalTriangleOffset = PackLocalTriangleOffset(sink.LocalTriangleCount, line_topology ? MeshPrimitiveTopology::Line : MeshPrimitiveTopology::Point),
                    .Primitive = uint32_t(sink.Primitives.size()),
                    .ConeAxisCutoff = uint32_t(uint8_t(127)) << 24u,
                    .Center = center,
                    .Radius = radius,
                });
            }

            sink.Primitives.emplace_back(PrimitiveRecord{
                .Draw = in.ElementDraw,
                .PrimitiveIndex = primitive_index,
                .FirstTriangle = 0u,
                .MeshletOffset = first_meshlet,
                .MeshletCount = uint32_t(sink.Records.size()) - first_meshlet,
                .Level0Count = uint32_t(sink.Records.size()) - first_meshlet,
            });
        }
    }

    if (face_topology) {
        // Choose one meshlet per topology element. Finest-LOD edit routing skips cone culling, and
        // conservative bounds rejection means a culled owner cannot contain a visible element.
        std::vector<uint8_t> vertex_owned(vertices.size());
        std::vector<uint8_t> edge_owned(in.EdgeCount);
        for (const auto &record : sink.Records) {
            const auto &primitive = in.PrimitiveTriangleRanges[record.Primitive];
            const uint32_t first_corner = primitive.FirstTriangle * 3u;
            for (uint32_t v = 0u; v < record.VertexCount; ++v) {
                auto &packed = sink.Vertices[record.VertexOffset + v];
                const uint32_t corner = packed & uint32_t(MeshletGeometryEncoding::CornerMask);
                const uint32_t vertex = in.Indices[first_corner + corner];
                if (vertex_owned[vertex] == 0u) {
                    vertex_owned[vertex] = 1u;
                    packed |= uint32_t(MeshletGeometryEncoding::EditVertexOwnerBit);
                }
            }

            for (uint32_t t = 0u; t < record.TriangleCount; ++t) {
                const uint32_t source_triangle = sink.TriangleIds[record.TriangleOffset + t];
                for (uint32_t c = 0u; c < 3u; ++c) {
                    auto &packed = sink.EditEdges[source_triangle * 3u + c];
                    if (packed == InvalidOffset) continue;
                    const uint32_t edge = packed & uint32_t(MeshletEditEdgeEncoding::EdgeMask);
                    if (edge_owned[edge] != 0u) packed = InvalidOffset;
                    else edge_owned[edge] = 1u;
                }
            }
        }
    }
    return sink;
}

// Reads the same captured inputs the level-0 build read, plus that build's own lists, so this runs
// on the thread that produced them.
ClusterLodBuild BuildMeshletClusterLod(const MeshletBuildInputs &in, const MeshletBuild &build) {
    // A face-less mesh clusters line or point elements, which carry no coarser level.
    if (!in.FaceTopology || build.Records.size() <= ClusterLodPartitionSize) return {};
    assert(build.Primitives.size() == in.PrimitiveTriangleRanges.size());

    std::vector<ClusterLodPrimitive> primitives(build.Primitives.size());
    for (uint32_t p = 0; p < primitives.size(); ++p) {
        primitives[p] = {
            .FirstTriangle = in.PrimitiveTriangleRanges[p].FirstTriangle,
            .TriangleCount = in.PrimitiveTriangleRanges[p].TriangleCount,
            .FirstCluster = build.Primitives[p].MeshletOffset,
            .ClusterCount = build.Primitives[p].MeshletCount,
        };
    }
    std::vector<ClusterLodSourceCluster> clusters(build.Records.size());
    for (uint32_t i = 0; i < clusters.size(); ++i) {
        const auto &record = build.Records[i];
        clusters[i] = {
            .FirstVertex = record.VertexOffset,
            .VertexCount = record.VertexCount,
            .FirstLocalTriangle = record.LocalTriangleOffset & uint32_t(MeshletGeometryEncoding::LocalTriangleOffsetMask),
            .TriangleCount = record.TriangleCount,
            .Center = record.Center,
            .Radius = record.Radius,
            // The never-culls cutoff marks a cluster whose geometric cone the material shader cannot trust.
            .ConeCullSafe = (record.ConeAxisCutoff >> 24u) != 127u,
        };
    }
    const ClusterLodMesh mesh{
        .CornerVertices = in.Indices,
        .Positions = &in.Vertices.front().Position.x,
        .PositionStride = sizeof(Vertex),
        // Empty normals derive geometric ones, which is what the simplifier weighs its collapses by.
        .CornerNormals = {},
        .Weld = in.Weld,
        .Primitives = primitives,
        .Clusters = clusters,
        .SourceVertexCorners = build.Vertices,
        .SourceLocalTriangles = std::span{build.LocalTriangles}.first(build.LocalTriangleCount),
    };
    return BuildClusterLod(mesh);
}

// Take every arena the finished build needs, and rebase the offsets it left relative to its own ranges.
// Serial, because arena offsets follow call order.
void CommitMeshlets(GpuBuffers &buffers, MeshBuffers &mb, MeshletBuild &build) {
    buffers.ReleaseMeshlets(mb);

    mb.MeshletTriangles = buffers.MeshletTriangleIds.Allocate(std::span{build.TriangleIds}.first(build.TriangleIdCount));
    mb.MeshletLocalTriangles = buffers.MeshletLocalTriangles.Allocate(std::span{build.LocalTriangles}.first(build.LocalTriangleCount));
    mb.MeshletEditEdges = buffers.MeshletEditEdgeIds.Allocate(build.EditEdges);
    mb.MeshletVertices = buffers.MeshletVertexCorners.Allocate(build.Vertices);
    for (auto &record : build.Records) {
        record.TriangleOffset += mb.MeshletTriangles.Offset;
        assert(((record.LocalTriangleOffset & uint32_t(MeshletGeometryEncoding::LocalTriangleOffsetMask)) + mb.MeshletLocalTriangles.Offset) <= uint32_t(MeshletGeometryEncoding::LocalTriangleOffsetMask));
        record.LocalTriangleOffset += mb.MeshletLocalTriangles.Offset;
        record.VertexOffset += mb.MeshletVertices.Offset;
    }
    mb.Meshlets = buffers.Meshlets.Allocate(build.Records);
    for (auto &record : build.Primitives) {
        record.MeshletOffset += mb.Meshlets.Offset;
        if (record.Draw.EditEdgeOffset != InvalidOffset) record.Draw.EditEdgeOffset += mb.MeshletEditEdges.Offset;
    }
    mb.Primitives = buffers.Primitives.Allocate(build.Primitives);
    for (auto &record : buffers.Meshlets.GetMutable(mb.Meshlets)) record.Primitive += mb.Primitives.Offset;
    // Every primitive presents one never-pruned span node, which is the whole traversal a mesh
    // without a DAG needs. A DAG commit replaces these with the real trees.
    auto placed_primitives = buffers.Primitives.GetMutable(mb.Primitives);
    std::vector<LodNode> nodes;
    nodes.reserve(placed_primitives.size());
    for (const auto &primitive : placed_primitives) {
        nodes.push_back(LodNode{
            .Error = std::numeric_limits<float>::infinity(),
            .FirstMeshlet = primitive.MeshletOffset,
            .MeshletCount = primitive.MeshletCount,
        });
    }
    mb.LodNodes = buffers.LodNodes.Allocate(nodes);
    for (uint32_t p = 0; p < placed_primitives.size(); ++p) {
        auto &primitive = placed_primitives[p];
        const uint32_t node = primitive.MeshletCount == 0u ? InvalidOffset : mb.LodNodes.Offset + p;
        primitive.LodRootNode = node;
        primitive.LodFinestNode = node;
    }
}

void CommitClusterLod(GpuBuffers &buffers, MeshBuffers &mb, const ClusterLodBuild &build) {
    if (build.Groups.empty()) return;
    assert(build.PrimitiveRanges.size() == mb.Primitives.Count);
    // The meshlet commit that produced this DAG's input dropped whatever DAG came before it.
    assert(mb.ClusterGroups.Count == 0);
    // The DAG's span trees replace the whole-run nodes the meshlet commit left.
    buffers.LodNodes.Release(mb.LodNodes);

    std::vector<ClusterGroup> groups(build.Groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        groups[i] = {.Center = build.Groups[i].Center, .Radius = build.Groups[i].Radius, .Error = build.Groups[i].Error};
    }
    mb.ClusterGroups = buffers.ClusterGroups.Allocate(groups);
    const auto placed_group = [group_offset = mb.ClusterGroups.Offset](uint32_t group) {
        return group == ClusterLodInvalid ? InvalidOffset : group + group_offset;
    };

    mb.LodNodes = buffers.LodNodes.Allocate(build.Nodes);

    mb.CoarseVertices = buffers.MeshletVertexCorners.Allocate(build.VertexCorners);
    mb.CoarseLocalTriangles = buffers.MeshletLocalTriangles.Allocate(build.LocalTriangles);

    // Each primitive keeps its original clusters in their existing order and gains its coarse ones
    // behind them, so a pinned instance still draws exactly the first Level0Count records.
    std::vector<MeshletRecord> records;
    records.reserve(mb.Meshlets.Count + build.Clusters.size());
    {
        const auto level0 = buffers.Meshlets.Get(mb.Meshlets);
        auto primitives = buffers.Primitives.GetMutable(mb.Primitives);
        for (uint32_t p = 0; p < primitives.size(); ++p) {
            auto &primitive = primitives[p];
            const auto &range = build.PrimitiveRanges[p];
            const uint32_t first = uint32_t(records.size());
            const uint32_t first_level0 = primitive.MeshletOffset - mb.Meshlets.Offset;
            for (uint32_t k = 0; k < primitive.Level0Count; ++k) {
                auto record = level0[first_level0 + k];
                record.GroupIndex = placed_group(build.Level0Groups[first_level0 + k]);
                records.push_back(record);
            }
            for (uint32_t c = 0; c < range.ClusterCount; ++c) {
                const auto &cluster = build.Clusters[range.FirstCluster + c];
                assert((cluster.LocalTriangleOffset + mb.CoarseLocalTriangles.Offset) <= uint32_t(MeshletGeometryEncoding::LocalTriangleOffsetMask));
                records.push_back(MeshletRecord{
                    // A coarse cluster names no source triangles.
                    .TriangleOffset = 0u,
                    .TriangleCount = cluster.TriangleCount,
                    .VertexOffset = cluster.VertexOffset + mb.CoarseVertices.Offset,
                    .VertexCount = cluster.VertexCount,
                    .LocalTriangleOffset = PackLocalTriangleOffset(cluster.LocalTriangleOffset + mb.CoarseLocalTriangles.Offset, MeshPrimitiveTopology::Triangle),
                    .Primitive = cluster.Primitive + mb.Primitives.Offset,
                    .GroupIndex = placed_group(cluster.GroupIndex),
                    .RefinedGroup = placed_group(cluster.RefinedGroup),
                    .ConeAxisCutoff = cluster.ConeAxisCutoff,
                    .Center = cluster.Center,
                    .Radius = cluster.Radius,
                });
            }
            primitive.MeshletOffset = first;
            primitive.MeshletCount = uint32_t(records.size()) - first;
        }
    }
    buffers.Meshlets.Release(mb.Meshlets);
    mb.Meshlets = buffers.Meshlets.Allocate(records);
    // Span nodes name records and children by mesh-local index, which the placed arenas rebase.
    for (auto &node : buffers.LodNodes.GetMutable(mb.LodNodes)) {
        node.FirstMeshlet += mb.Meshlets.Offset;
        if (node.ChildCount != 0u) node.ChildOffset += mb.LodNodes.Offset;
    }
    const auto placed_node = [node_offset = mb.LodNodes.Offset](uint32_t node) {
        return node == ClusterLodInvalid ? InvalidOffset : node + node_offset;
    };
    auto placed_primitives = buffers.Primitives.GetMutable(mb.Primitives);
    for (uint32_t p = 0; p < placed_primitives.size(); ++p) {
        auto &primitive = placed_primitives[p];
        primitive.MeshletOffset += mb.Meshlets.Offset;
        primitive.LodRootNode = placed_node(build.PrimitiveRanges[p].RootNode);
        primitive.LodFinestNode = placed_node(build.PrimitiveRanges[p].FinestNode);
    }
    buffers.MeshletLodDepth = std::max(buffers.MeshletLodDepth, build.NodeDepth);
}
