#include "numeric/VectorMath.h"
#include "numeric/uvec2.h"
#include "numeric/vec2.h"

#include "mesh/MeshCreate.h"

#include "Parallel.h"
#include "Profile.h"
#include "mesh/CornerNormalOffset.h"
#include "mesh/MeshConnectivityGpu.h"
#include "mesh/VertexWeldGpu.h"

#include "state/Scene.h"

#include <bit>
#include <numeric>

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

    // Sort faces by primitive index so triangles are grouped by primitive in the index buffer.
    if (!primitives.ElementPrimitiveIndices.empty() && primitives.ElementPrimitiveIndices.size() == face_count &&
        !std::ranges::all_of(primitives.ElementPrimitiveIndices, [&](uint32_t pi) { return pi == primitives.ElementPrimitiveIndices[0]; })) {
        std::vector<uint32_t> perm(face_count);
        std::iota(perm.begin(), perm.end(), 0u);
        std::stable_sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) {
            return primitives.ElementPrimitiveIndices[a] < primitives.ElementPrimitiveIndices[b];
        });
        bool already_sorted = true;
        for (uint32_t i = 0; i < face_count; ++i) {
            if (perm[i] != i) {
                already_sorted = false;
                break;
            }
        }
        if (!already_sorted) {
            // A mesh of triangles keeps its offsets arithmetic, so only the corners permute.
            const bool spelled_offsets = !data.FaceOffsets.empty();
            std::vector<uint32_t> sorted_offsets, sorted_corners, sorted_fpi(face_count);
            if (spelled_offsets) {
                sorted_offsets.reserve(face_count + 1);
                sorted_offsets.emplace_back(0u);
            }
            sorted_corners.reserve(data.FaceCorners.size());
            for (uint32_t i = 0; i < face_count; ++i) {
                const auto face = data.Face(perm[i]);
                sorted_corners.insert(sorted_corners.end(), face.begin(), face.end());
                if (spelled_offsets) sorted_offsets.emplace_back(uint32_t(sorted_corners.size()));
                sorted_fpi[i] = primitives.ElementPrimitiveIndices[perm[i]];
            }
            data.FaceOffsets = std::move(sorted_offsets);
            data.FaceCorners = std::move(sorted_corners);
            primitives.ElementPrimitiveIndices = std::move(sorted_fpi);
        }
    }

    // Triangle-mesh tangent/color/UV channels are corner-domain: gathered into per-corner streams in fan order before welding rewrites the face indices.
    // The vertex buffer keeps defaults for these channels.
    PreparedMesh prepared;
    if (face_count > 0) {
        const uint32_t corner_total = (uint32_t(data.FaceCorners.size()) - 2u * face_count) * 3u;
        const auto gather_corners = [&]<typename T>(std::optional<std::vector<T>> &src, std::vector<T> &out) {
            if (!src) return;
            out.reserve(corner_total);
            for (uint32_t fi = 0; fi < face_count; ++fi) {
                const auto face = data.Face(fi);
                for (uint32_t k = 1; k + 1 < face.size(); ++k) {
                    out.emplace_back((*src)[face[0]]);
                    out.emplace_back((*src)[face[k]]);
                    out.emplace_back((*src)[face[k + 1]]);
                }
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
        gather_corners(attrs.Normals, prepared.AuthoredCornerNormals);
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
        const uint32_t corner_count = (face.size() - 2) * 3;
        const auto p0 = vertices[face[0]].Position;
        const auto cross = Cross(vertices[face[1]].Position - p0, vertices[face[2]].Position - p0);
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
    const auto first_triangles = meshes.Arenas().FaceFirstTriangles.Get(record.FaceData);
    const auto sharp_edges = meshes.EditEdgeSharpness(id);
    const auto &c = mesh.GetConnectivity();
    // The authored normal at face loop position `k`, read from any of its fan-corner slots.
    const auto authored_at = [&](Mesh::FH fh, uint32_t k) {
        const auto base = 3 * first_triangles[*fh];
        if (k == 0) return authored[base];
        const auto tri_count = mesh.GetValence(fh) - 2;
        return k - 1 < tri_count ? authored[base + 3 * (k - 1) + 1] : authored[base + 3 * (k - 2) + 2];
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
        const auto hh = mesh.GetHalfedge(Mesh::EH{ei}, 0);
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
            source.Deform.reset();
            source.Morph.reset();
        }
    }
    {
        // Complete all requested welds before connectivity reads the rewritten corners.
        std::vector<WeldTarget> weld_targets;
        weld_targets.reserve(sources.size());
        for (uint32_t i = 0; i < sources.size(); ++i) {
            if (sources[i].Weld) weld_targets.emplace_back(ids[i], &sources[i].Data, &prepared[i].MorphTangentDeltas);
        }
        WeldMeshesNow(r, weld_targets);
    }
    {
        // Allocate connectivity in source order, then build face meshes on the GPU and edge meshes on the host.
        const profile::CpuScope scope{"BuildConnectivity"};
        std::vector<uint32_t> face_mesh_ids;
        for (uint32_t i = 0; i < sources.size(); ++i) {
            const auto &data = sources[i].Data;
            const auto &record = meshes.Get(ids[i]);
            const uint32_t halfedges = data.FaceCount() > 0 ? record.FaceCorners.Count : data.HalfedgeCount();
            const bool face_starts = data.FaceCount() > 0 && halfedges != 3 * data.FaceCount();
            meshes.AllocateConnectivity(ids[i], record.Vertices.Count, halfedges, data.FaceCount(), face_starts, face_starts ? data.FaceOffsets : std::span<const uint32_t>{});
            if (data.FaceCount() > 0) face_mesh_ids.push_back(ids[i]);
        }
        BuildConnectivityNow(r, face_mesh_ids);
        for (uint32_t i = 0; i < sources.size(); ++i) {
            const auto &data = sources[i].Data;
            if (data.FaceCount() > 0) continue;
            meshes.FinishConnectivity(ids[i], BuildConnectivity(data.Edges, meshes.Get(ids[i]).Vertices.Count, meshes.GetConnectivityStorage(ids[i])));
        }
    }

    std::vector<CreatedMesh> created;
    created.reserve(sources.size());
    for (uint32_t i = 0; i < sources.size(); ++i) {
        auto &source = sources[i];
        auto &authored = prepared[i].AuthoredCornerNormals;
        meshes.CreateMesh(ids[i], source.Data, source.Attrs, source.Primitives, prepared[i].Layers, !authored.empty());
        const Mesh mesh{meshes, ids[i]};
        InitializeSharpness(meshes, mesh, source.Data, source.Primitives, source.FlatShaded, authored);
        meshes.UpdateCornerClassification(mesh);
        created.emplace_back(ids[i], std::move(prepared[i].MorphTangentDeltas), std::move(authored));
    }
    return created;
}

CreatedMesh CreateMesh(state::Scene &r, MeshSource source) { return std::move(CreateMeshes(r, {&source, 1}).front()); }

void EncodeAuthoredCornerNormals(MeshStore &meshes, const Mesh &mesh, std::span<const vec3> authored) {
    const auto id = mesh.GetStoreId();
    const auto &record = meshes.Get(id);
    if (authored.empty() || record.TriangleCount == 0) return;
    const auto indices = mesh.CreateTriangleIndices();
    // The custom layer is empty, so this is the raw derived normal per corner.
    const auto derived = meshes.GetCornerNormals(mesh, indices);
    const auto vertices = meshes.Arenas().Vertices.Get(record.Vertices);
    std::vector<uvec2> masks((derived.size() + 31) / 32, uvec2{0});
    std::vector<vec2> packed;
    for (size_t i = 0; i < derived.size() && i < authored.size(); ++i) {
        const auto authored_normal = authored[i];
        if (NormalsMatch(authored_normal, derived[i]).value_or(true)) continue;
        masks[i / 32].x |= 1u << (i % 32);
        packed.emplace_back(EncodeNormalOffset(authored_normal / Length(authored_normal), ComputeCornerFrame(derived[i], indices, vertices, i)));
    }
    if (packed.empty()) return;
    uint32_t rank = 0;
    for (auto &mask : masks) {
        mask.y = rank;
        rank += std::popcount(mask.x);
    }
    meshes.SetCustomCornerNormals(id, masks, packed);
}

void UpdateMorphShadingAuthored(MeshStore &meshes, const Mesh &mesh, std::span<const CornerNormalSources> poses) {
    const auto id = mesh.GetStoreId();
    const auto &record = meshes.Get(id);
    const auto &derived = meshes.GetDerived(id);
    const auto &arenas = meshes.Arenas();
    meshes.SetMorphShadingAuthored(id, false);
    if (!record.HasAuthoredNormals || record.TriangleCount == 0 || record.MorphTargetCount == 0) return;
    // A target authoring normal deltas states the morphed shading normals directly.
    if (std::ranges::any_of(arenas.MorphTargets.Get(record.MorphTargets), [](const auto &t) { return t.NormalDelta != vec3{0}; })) {
        meshes.SetMorphShadingAuthored(id, true);
        return;
    }
    if (poses.empty()) return;
    // Position-only targets pin the authored normals in place.
    // Authorship matters when any listed full-weight pose derives a corner normal away from the rest normal it would pin.
    const auto indices = mesh.CreateTriangleIndices();
    const auto classes = arenas.CornerClasses.Get(derived.CornerClasses);
    const auto face_ids = arenas.TriangleFaceIds.Get(record.TriangleFaceIds);
    const auto compose = [&](const CornerNormalSources &normals, uint32_t ci) {
        return ComposeCornerNormal(classes, derived.UniformCornerClass, ci, indices, face_ids, normals);
    };
    const CornerNormalSources rest{arenas.BaseVertexNormals.Get(record.Vertices), arenas.BaseSeamNormals.Get(derived.BaseSeamNormals), arenas.BaseFaceNormals.Get(record.FaceData)};
    for (uint32_t ci = 0; ci < indices.size(); ++ci) {
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
