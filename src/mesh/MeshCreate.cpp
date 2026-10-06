
#include "mesh/MeshCreate.h"

#include "Parallel.h"
#include "Profile.h"
#include "mesh/CornerNormalOffset.h"
#include "mesh/MeshConnectivityGpu.h"
#include "mesh/VertexWeldGpu.h"
#include "metal/Dispatch.h"

#include "state/Scene.h"

namespace {
// An authored corner normal within 0.05 degrees of the derived one counts as derivable and is dropped.
// Export-pipeline rounding remains below this angle.
// Deliberate normal authoring remains well above it.
constexpr float AuthoredMatchDot{0.99999962f};

// Whether `normal` matches the unit-or-zero `reference` within the authored match gate.
// Returns nullopt when `normal` is degenerate.
// A zero reference matches nothing.
std::optional<bool> NormalsMatch(vec3 normal, vec3 reference) {
    const auto len = Length(normal);
    if (len < 1e-6f) return {};
    return Dot(normal / len, reference) >= AuthoredMatchDot;
}

// Contains source-derived data without arena or store ownership.
struct PreparedMesh {
    CornerLayers Layers;
    std::vector<vec3> AuthoredCornerNormals;
    // Target-major morph tangent deltas remain host-owned through welding.
    std::vector<vec3> MorphTangentDeltas;
};

// Orders faces by primitive and gathers corner channels while preserving CreateMesh inputs.
PreparedMesh PrepareMeshSources(MeshData &data, MeshVertexAttributes &attrs, MeshPrimitives &primitives) {
    const uint32_t face_count = data.FaceCount();

    // Sort only when needed; gathering corners below follows the reordered faces.
    auto &indices = primitives.ElementPrimitiveIndices;
    if (indices.size() == face_count && !std::ranges::is_sorted(indices)) {
        std::vector<uint32_t> perm(face_count);
        std::iota(perm.begin(), perm.end(), 0u);
        std::stable_sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) { return indices[a] < indices[b]; });
        // A triangle mesh keeps arithmetic offsets, so only its corners permute.
        const bool spelled_offsets = !data.FaceOffsets.empty();
        std::vector<uint32_t> sorted_offsets, sorted_corners, sorted_indices;
        if (spelled_offsets) {
            sorted_offsets.reserve(face_count + 1u);
            sorted_offsets.push_back(0u);
        }
        sorted_corners.reserve(data.FaceCorners.size());
        sorted_indices.reserve(face_count);
        for (const auto source : perm) {
            const auto face = data.Face(source);
            sorted_corners.insert(sorted_corners.end(), face.begin(), face.end());
            if (spelled_offsets) sorted_offsets.push_back(uint32_t(sorted_corners.size()));
            sorted_indices.push_back(indices[source]);
        }
        data.FaceOffsets = std::move(sorted_offsets);
        data.FaceCorners = std::move(sorted_corners);
        indices = std::move(sorted_indices);
    }

    // Tangents, colors and UVs attach to polygon corners before welding rewrites vertex indices.
    // The vertex buffer keeps defaults for these channels.
    PreparedMesh prepared;
    if (face_count > 0) {
        const auto gather_corners = [&]<typename T>(std::optional<std::vector<T>> &src, std::vector<T> &out, bool wires = true) {
            if (!src) return;
            out.reserve(wires ? data.HalfedgeCount() : data.FaceCorners.size());
            for (const auto v : data.FaceCorners) out.emplace_back((*src)[v]);
            if (wires)
                for (const auto &edge : data.Edges) {
                    out.emplace_back((*src)[edge[1]]);
                    out.emplace_back((*src)[edge[0]]);
                }
            src.reset();
        };
        gather_corners(attrs.Tangents, prepared.Layers.Tangents);
        gather_corners(attrs.Colors0, prepared.Layers.Colors);
        gather_corners(attrs.TexCoords0, prepared.Layers.Uvs[0]);
        gather_corners(attrs.TexCoords1, prepared.Layers.Uvs[1]);
        gather_corners(attrs.TexCoords2, prepared.Layers.Uvs[2]);
        gather_corners(attrs.TexCoords3, prepared.Layers.Uvs[3]);
        // Shading normals derive, so the authored stream only seeds the sharpness stores and the custom corner-normal layer.
        gather_corners(attrs.Normals, prepared.AuthoredCornerNormals, false);
    }
    return prepared;
}

// Seed the sharpness stores of a new face mesh: flat where the source shades flat, then where its authored normals say so.
void InitializeSharpness(MeshStore &meshes, const Mesh &mesh, const MeshData &data, const MeshPrimitives &primitives, bool flat_shaded, std::span<const vec3> authored) {
    const auto id = mesh.GetStoreId();
    if (mesh.FaceCount() == 0) return;
    const auto sharp_faces = meshes.EditFaceSharpness(id);
    if (flat_shaded) std::ranges::fill(sharp_faces, uint8_t{1});
    // Faces of primitives that ship no normals shade flat, like a fully normal-less mesh.
    else if (!primitives.AttributeFlags.empty()) {
        for (uint32_t fi = 0; fi < sharp_faces.size(); ++fi) {
            const auto pi = fi < primitives.ElementPrimitiveIndices.size() ? primitives.ElementPrimitiveIndices[fi] : 0u;
            if (pi < primitives.AttributeFlags.size() && !(primitives.AttributeFlags[pi] & MeshAttributeBit_Normal)) sharp_faces[fi] = 1;
        }
    }
    if (authored.empty()) return;

    // A face whose authored corner normals all match its geometric normal shades flat, recorded as face sharpness.
    // The weld rewrote the arena's positions and corners, so the face loops read them there.
    const auto &record = meshes.Get(id);
    const auto vertices = meshes.Arenas().Vertices.Get(record.Vertices);
    const auto corners = mesh.CornerVertices();
    uint32_t ci = 0;
    for (uint32_t fi = 0; fi < mesh.FaceCount(); ++fi) {
        const auto face = corners.subspan(data.FaceStart(fi), data.FaceSize(fi));
        const auto corner_count = uint32_t(face.size());
        const auto p0 = vertices[face[0] - meshes.Arenas().Vertices.First(record.Vertices)].Position;
        vec3 cross{};
        for (uint32_t k = 1u; k + 1u < face.size(); ++k)
            cross += Cross(vertices[face[k] - meshes.Arenas().Vertices.First(record.Vertices)].Position - p0, vertices[face[k + 1u] - meshes.Arenas().Vertices.First(record.Vertices)].Position - p0);
        const auto cross_len = Length(cross);
        bool flat = cross_len > 0.f;
        if (flat) {
            const auto face_normal = cross / cross_len;
            for (uint32_t k = 0; k < corner_count; ++k) {
                if (!NormalsMatch(authored[ci + k], face_normal).value_or(false)) {
                    flat = false;
                    break;
                }
            }
        }
        if (flat) sharp_faces[fi] = 1;
        ci += corner_count;
    }

    // Sharp-edge inference: an interior edge whose authored corner normals disagree across it at either endpoint splits shading there.
    // The split records as edge sharpness so seam sectors derive.
    if (mesh.EdgeCount() == 0) return;
    const auto sharp_edges = meshes.EditEdgeSharpness(id);
    const auto &c = mesh.GetConnectivity();
    const auto authored_at = [&](Mesh::FH fh, uint32_t k) {
        return authored[data.FaceStart(mesh.FaceOrdinal(fh)) + k];
    };
    const auto vertex_position = [&](Mesh::FH fh, Mesh::VH vh) -> std::optional<uint32_t> {
        uint32_t k = 0;
        for (const auto hh : mesh.fh_range(fh)) {
            if (mesh.GetToVertex(hh) == vh) return k;
            ++k;
        }
        return {};
    };
    const auto discontinuous = [&](Mesh::FH fa, Mesh::FH fb, Mesh::VH vh) {
        const auto ka = vertex_position(fa, vh), kb = vertex_position(fb, vh);
        if (!ka || !kb) return false;
        const auto nb = authored_at(fb, *kb);
        const auto lb = Length(nb);
        if (lb < 1e-6f) return false;
        return NormalsMatch(authored_at(fa, *ka), nb / lb) == false;
    };
    for (uint32_t ei = 0; ei < mesh.EdgeCount(); ++ei) {
        const auto hh = mesh.GetHalfedge(mesh.EdgeAt(ei), 0);
        const auto face = mesh.GetFace(hh);
        const auto opposite = c.Opposites[*hh];
        const auto opposite_face = opposite ? c.FaceOf(opposite) : Mesh::FH{};
        if (!face || !opposite_face) continue;
        if (discontinuous(face, opposite_face, mesh.GetFromVertex(hh)) || discontinuous(face, opposite_face, mesh.GetToVertex(hh))) {
            sharp_edges[ei] = 1;
        }
    }
}
} // namespace

std::vector<CreatedMesh> CreateMeshes(state::Scene &r, std::span<MeshSource> sources) {
    auto &meshes = r.Context.get<MeshStore>();
    // One reserve per arena for the whole batch, so no allocation below grows a buffer.
    for (const auto &source : sources) {
        meshes.PlanCreate(source.Data, source.Primitives, source.Deform.has_value(), source.Morph ? source.Morph->TargetCount : 0u, source.Attrs);
    }
    meshes.CommitReserves();

    std::vector<PreparedMesh> prepared(sources.size());
    {
        const profile::CpuScope scope{"PrepareMeshes"};
        ParallelFor(uint32_t(sources.size()), [&](uint32_t i) {
            prepared[i] = PrepareMeshSources(sources[i].Data, sources[i].Attrs, sources[i].Primitives);
        });
    }
    // Release host copies before in-place GPU welding.
    std::vector<uint32_t> ids(sources.size());
    {
        const profile::CpuScope scope{"CreateMeshSource"};
        for (uint32_t i = 0; i < sources.size(); ++i) {
            auto &source = sources[i];
            ids[i] = meshes.CreateMeshSource(source.Data);
            meshes.CreateDeformSource(ids[i], source.Deform, source.Morph);
            // Keep tangent deltas on the host so welding can return them compacted.
            if (source.Morph) prepared[i].MorphTangentDeltas = std::move(source.Morph->TangentDeltas);
            source.Data.Positions = std::vector<vec3>{};
            if (source.Data.FaceCount() > 0) source.Data.FaceCorners = std::vector<uint32_t>{};
            source.Data.Edges = std::vector<std::array<uint32_t, 2>>{};
            source.Deform.reset();
            source.Morph.reset();
        }
    }
    {
        // Complete all requested welds before connectivity reads the rewritten corners.
        std::vector<WeldTarget> weld_targets;
        weld_targets.reserve(sources.size());
        for (uint32_t i = 0; i < sources.size(); ++i) {
            if (sources[i].Weld) weld_targets.emplace_back(ids[i], &sources[i].Data, &prepared[i].MorphTangentDeltas, sources[i].KeepLooseVertices);
        }
        WeldMeshesNow(r, weld_targets);
    }
    BuildConnectivityNow(r, ids);

    std::vector<CreatedMesh> created;
    created.reserve(sources.size());
    for (uint32_t i = 0; i < sources.size(); ++i) {
        auto &source = sources[i];
        auto &authored = prepared[i].AuthoredCornerNormals;
        meshes.CreateMesh(ids[i], source.Data, source.Attrs, source.Primitives, prepared[i].Layers, !authored.empty());
        const Mesh mesh{meshes, ids[i]};
        InitializeSharpness(meshes, mesh, source.Data, source.Primitives, source.FlatShaded, authored);
        created.emplace_back(ids[i], std::move(prepared[i].MorphTangentDeltas), std::move(authored));
    }
    // The selection state's submit also runs the corner class writes.
    mtl::ComputeChain chain{meshes.BufferContext()};
    meshes.UpdateCornerClassification(r, chain, ids);
    meshes.EnsureSelectionState(r, chain, ids);
    return created;
}

CreatedMesh CreateMesh(state::Scene &r, MeshSource source) { return std::move(CreateMeshes(r, {&source, 1}).front()); }

void EncodeAuthoredCornerNormals(MeshStore &meshes, const Mesh &mesh, std::span<const vec3> authored) {
    const auto id = mesh.GetStoreId();
    const auto &record = meshes.Get(id);
    if (authored.empty() || record.TriangleCount == 0) return;
    // Source normals stay in polygon-corner order, independently of tessellation.
    const auto derived = meshes.GetCornerNormalView(id);
    const auto first = mesh.HalfedgeFirst();
    std::vector<CustomNormal> offsets(mesh.HalfEdgeCount());
    bool any = false;
    for (const auto face : mesh.faces())
        for (const auto corner : mesh.fh_range(face)) {
            const uint32_t h = *corner, i = h - first;
            const auto authored_normal = authored[i], normal = derived[h];
            if (NormalsMatch(authored_normal, normal).value_or(true)) continue;
            offsets[i].Offset = EncodeNormalOffset(authored_normal / Length(authored_normal), ComputeCornerFrame(normal, mesh, h));
            any = true;
        }
    if (any) meshes.SetCustomCornerNormals(id, offsets);
}

void UpdateMorphShadingAuthored(MeshStore &meshes, const Mesh &mesh, std::span<const CornerNormalSources> poses) {
    const auto id = mesh.GetStoreId();
    const auto &record = meshes.Get(id);
    const auto &arenas = meshes.Arenas();
    meshes.SetMorphShadingAuthored(id, false);
    if (!record.HasAuthoredNormals || record.TriangleCount == 0 || record.MorphTargetCount == 0) return;
    // A target authoring normal deltas states the morphed shading normals directly.
    bool has_authored_delta = false;
    arenas.Vertices.ForEach(record.Vertices, [&](uint32_t vertex, uint32_t) {
        for (uint32_t target = 0; target < record.MorphTargetCount && !has_authored_delta; ++target)
            has_authored_delta = arenas.Morph.Get(vertex, target).NormalDelta != vec3{0};
    });
    if (has_authored_delta) {
        meshes.SetMorphShadingAuthored(id, true);
        return;
    }
    if (poses.empty()) return;
    // Position-only targets pin the authored normals in place.
    // Authorship matters when any listed full-weight pose derives a corner normal away from the rest normal it would pin.
    const auto vertices = mesh.TriangleVertices();
    const CornerAttributeView<uint32_t> classes{arenas.CornerSectors.View(record.SectorBlockCount != 0u), meshes.GetTriangleCorners(id)};
    const TriangleFaceView face_ids{meshes.GetTriangleCorners(id), arenas.HalfedgeFaces.Buffer.GetSpan<uint32_t>()};
    const auto compose = [&](const CornerNormalSources &normals, uint32_t ci) {
        return ComposeCornerNormal(classes, arenas.NormalSectors.View(), arenas.FaceSharpness.Buffer.GetSpan<uint8_t>(), record.Classification, ci, vertices, face_ids, normals);
    };
    const CornerNormalSources rest{.VertexNormals = arenas.BaseVertexNormals.Buffer.GetSpan<vec3>(), .FaceNormals = arenas.BaseFaceNormals.Buffer.GetSpan<vec3>()};
    for (uint32_t ci = 0; ci < vertices.size(); ++ci) {
        const auto rest_normal = compose(rest, ci);
        if (rest_normal == vec3{0}) continue;
        for (const auto &pose : poses) {
            const auto posed = compose(pose, ci);
            if (posed != vec3{0} && Dot(rest_normal, posed) < AuthoredMatchDot) {
                meshes.SetMorphShadingAuthored(id, true);
                return;
            }
        }
    }
}
