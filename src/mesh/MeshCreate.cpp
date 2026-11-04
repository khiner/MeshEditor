
#include "mesh/MeshCreate.h"

#include "Parallel.h"
#include "Profile.h"
#include "SortUnique.h"
#include "mesh/MeshConnectivityGpu.h"
#include "mesh/MeshPipelines.h"
#include "mesh/Neighborhood.h"
#include "mesh/VertexWeldGpu.h"

#include "state/Registry.h"

namespace {
// An authored corner normal within 0.05 degrees of the derived one counts as derivable and is dropped.
constexpr float AuthoredMatchDot{0.99999962f};

// Whether `normal` matches the unit-or-zero `reference` within AuthoredMatchDot, or none for a degenerate `normal`.
std::optional<bool> NormalsMatch(vec3 normal, vec3 reference) {
    const auto len = Length(normal);
    if (len < 1e-6f) return {};
    return Dot(normal / len, reference) >= AuthoredMatchDot;
}

struct PreparedMesh {
    CornerLayers Layers;
    std::vector<CustomAttribute> CustomCorners; // A face mesh's custom attributes in the corner order of its layers
    std::vector<vec3> AuthoredCornerNormals;
    std::vector<vec3> MorphTangentDeltas;
};

PreparedMesh PrepareMeshSources(MeshData &data, MeshVertexAttributes &attrs, MeshPrimitives &primitives) {
    const uint32_t face_count = data.FaceCount();

    auto &indices = primitives.ElementPrimitiveIndices;
    if (indices.size() == face_count && !std::ranges::is_sorted(indices)) {
        std::vector<uint32_t> perm(face_count);
        std::ranges::iota(perm, 0u);
        std::ranges::stable_sort(perm, [&](uint32_t a, uint32_t b) { return indices[a] < indices[b]; });
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
    PreparedMesh prepared;
    if (face_count > 0) {
        const auto gather_corners = [&]<typename T>(std::optional<std::vector<T>> &src, std::vector<T> &out, bool wires = true) {
            if (!src) return;
            out.reserve(wires ? data.HalfedgeCount() : data.FaceCorners.size());
            for (const auto v : data.FaceCorners) out.emplace_back((*src)[v]);
            if (wires) {
                for (const auto &edge : data.Edges) {
                    out.emplace_back((*src)[edge[1]]);
                    out.emplace_back((*src)[edge[0]]);
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
        gather_corners(attrs.Normals, prepared.AuthoredCornerNormals, false);
        for (auto &custom : attrs.Custom) {
            const auto n = AttributeComponents(custom.Type);
            auto &corners = prepared.CustomCorners.emplace_back(custom.Name, custom.Type);
            corners.Words.reserve(size_t(data.HalfedgeCount()) * n);
            const auto gather = [&](uint32_t v) { corners.Words.insert(corners.Words.end(), custom.Words.begin() + size_t(v) * n, custom.Words.begin() + size_t(v + 1u) * n); };
            for (const auto v : data.FaceCorners) gather(v);
            for (const auto &edge : data.Edges) {
                gather(edge[1]);
                gather(edge[0]);
            }
        }
        attrs.Custom.clear();
    }
    return prepared;
}

void CreateCustomAttributes(MeshStore &meshes, uint32_t id, std::span<const CustomAttribute> attributes, bool per_corner) {
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const auto vertices = a.Vertices.Dense(record.Vertices), corners = a.FaceCorners.Dense(record.FaceCorners);
    for (const auto &custom : attributes) {
        const auto n = AttributeComponents(custom.Type);
        std::vector<uint32_t> vertex_words;
        bool uniform = true;
        if (per_corner) {
            vertex_words.assign(size_t(vertices.Count) * n, 0u);
            std::vector<uint8_t> seen(vertices.Count, 0u);
            const auto corner_vertices = a.FaceCorners.Get(corners);
            for (uint32_t c = 0u; c < corners.Count && uniform; ++c) {
                const auto v = corner_vertices[c];
                const auto value = std::span{custom.Words}.subspan(size_t(c) * n, n);
                const auto stored = std::span{vertex_words}.subspan(size_t(v) * n, n);
                if (!seen[v]) std::ranges::copy(value, stored.begin());
                else uniform = std::ranges::equal(value, stored);
                seen[v] = 1u;
            }
        }
        const auto &words = per_corner && uniform ? vertex_words : custom.Words;
        const auto handles = uniform ? vertices : corners;
        const auto domain = uniform ? AttributeDomain::Vertex : AttributeDomain::Corner;
        const auto attribute = meshes.AddAttribute(id, domain, custom.Type, {custom.Name.data(), strnlen(custom.Name.data(), custom.Name.size())}).value();
        WithAttributeScalar(custom.Type, [&]<AttributeScalar T>() {
            const auto values = std::make_unique<T[]>(handles.Count);
            for (uint32_t i = 0u; i < handles.Count; ++i) {
                if constexpr (std::same_as<T, bool>) values[i] = words[i] != 0u;
                else std::memcpy(&values[i], &words[size_t(i) * n], sizeof(T));
            }
            meshes.SetAttribute<T>(attribute, 0u, std::span<const T>{values.get(), handles.Count});
        });
    }
}

void InitializeSharpness(MeshStore &meshes, const MeshPipelines &pipelines, const Mesh &mesh, const MeshData &data, const MeshPrimitives &primitives, bool flat_shaded, std::span<const vec3> authored) {
    const auto id = mesh.GetStoreId();
    if (mesh.FaceCount() == 0) return;
    const auto sharp_faces = meshes.EditFaceSharpness(id);
    if (flat_shaded) {
        std::ranges::fill(sharp_faces, uint8_t{1});
    } else if (!primitives.AttributeFlags.empty()) {
        for (uint32_t fi = 0; fi < sharp_faces.size(); ++fi) {
            const auto pi = fi < primitives.ElementPrimitiveIndices.size() ? primitives.ElementPrimitiveIndices[fi] : 0u;
            if (pi < primitives.AttributeFlags.size() && !(primitives.AttributeFlags[pi] & MeshAttributeBit_Normal)) sharp_faces[fi] = 1;
        }
    }
    if (authored.empty()) return;

    // The weld rewrote the positions and corners, so the face loops read them from the GPU.
    const auto [faces, corners, positions] = ReadFaceLoops(meshes, pipelines, mesh);
    const auto authored_at = [&](uint32_t fi, uint32_t k) { return authored[data.FaceStart(fi) + k]; };
    for (uint32_t fi = 0; fi < faces.size(); ++fi) {
        const auto face = corners.Of(fi);
        const auto corner_count = uint32_t(face.size());
        const auto p0 = positions[face[0].Vertex];
        vec3 cross{};
        for (uint32_t k = 1u; k + 1u < face.size(); ++k) {
            cross += Cross(positions[face[k].Vertex] - p0, positions[face[k + 1u].Vertex] - p0);
        }
        const auto cross_len = Length(cross);
        bool flat = cross_len > 0.f;
        if (flat) {
            const auto face_normal = cross / cross_len;
            for (uint32_t k = 0; k < corner_count; ++k) {
                if (!NormalsMatch(authored_at(fi, k), face_normal).value_or(false)) {
                    flat = false;
                    break;
                }
            }
        }
        if (flat) sharp_faces[fi] = 1;
    }

    if (mesh.EdgeCount() == 0) return;
    const auto sharp_edges = meshes.EditEdgeSharpness(id);
    const auto vertex_position = [&](uint32_t fi, uint32_t vh) -> std::optional<uint32_t> {
        const auto face = corners.Of(fi);
        for (uint32_t k = 0; k < face.size(); ++k) {
            if (face[k].Vertex == vh) return k;
        }
        return {};
    };
    const auto discontinuous = [&](uint32_t fa, uint32_t fb, uint32_t vh) {
        const auto ka = vertex_position(fa, vh), kb = vertex_position(fb, vh);
        if (!ka || !kb) return false;
        const auto nb = authored_at(fb, *kb);
        const auto lb = Length(nb);
        if (lb < 1e-6f) return false;
        return NormalsMatch(authored_at(fa, *ka), nb / lb) == false;
    };
    // A new record's faces are dense, so a face handle is its index in the loops.
    for (uint32_t fi = 0; fi < faces.size(); ++fi) {
        const auto face = corners.Of(fi);
        for (uint32_t k = 0; k < face.size(); ++k) {
            const auto &corner = face[k];
            if (corner.Opposite == InvalidOffset || corner.OppositeFace == InvalidOffset || corner.Opposite < corner.Halfedge) continue;
            const auto opposite_face = corner.OppositeFace, from = face[(k + face.size() - 1u) % face.size()].Vertex;
            if (discontinuous(fi, opposite_face, from) || discontinuous(fi, opposite_face, corner.Vertex)) sharp_edges[corner.Edge] = 1;
        }
    }
}
} // namespace

std::vector<CreatedMesh> CreateMeshes(state::Registry &r, std::span<MeshSource> sources) {
    auto &meshes = r.Resources.get<MeshStore>();
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
    // Host copies release before the in-place GPU weld.
    std::vector<uint32_t> ids(sources.size());
    {
        const profile::CpuScope scope{"CreateMeshSource"};
        for (uint32_t i = 0; i < sources.size(); ++i) {
            auto &source = sources[i];
            ids[i] = meshes.CreateMeshSource(source.Data);
            meshes.CreateDeformSource(ids[i], source.Deform, source.Morph);
            if (source.Morph) prepared[i].MorphTangentDeltas = std::move(source.Morph->TangentDeltas);
            source.Data.Positions = std::vector<vec3>{};
            if (source.Data.FaceCount() > 0) source.Data.FaceCorners = std::vector<uint32_t>{};
            source.Data.Edges = std::vector<std::array<uint32_t, 2>>{};
            source.Deform.reset();
            source.Morph.reset();
        }
    }
    {
        // Every weld completes before connectivity reads the rewritten corners.
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
        const bool faces = source.Data.FaceCount() > 0u;
        CreateCustomAttributes(meshes, ids[i], faces ? prepared[i].CustomCorners : source.Attrs.Custom, faces);
        const Mesh mesh{meshes, ids[i]};
        InitializeSharpness(meshes, GetMeshPipelines(r), mesh, source.Data, source.Primitives, source.FlatShaded, authored);
        created.emplace_back(ids[i], std::move(prepared[i].MorphTangentDeltas), std::move(authored));
    }
    return created;
}

CreatedMesh CreateMesh(state::Registry &r, MeshSource source) { return std::move(CreateMeshes(r, {&source, 1}).front()); }

std::optional<NewMeshShading> EncodeNewMeshShading(state::Registry &r, mtl::ComputeChain &chain) {
    auto &meshes = r.Resources.get<MeshStore>();
    auto ids = meshes.TakeUnshaded();
    std::erase_if(ids, [&](uint32_t id) { return !meshes.TryGet(id); });
    if (ids.empty()) return {};
    SortUnique(ids);
    std::vector<MeshStore::CornerClassRequest> requests;
    requests.reserve(ids.size());
    for (const auto id : ids) requests.push_back({.StoreId = id});
    auto classes = meshes.EncodeCornerClassification(r, chain, requests);
    auto normals = EncodeAllNormalsMembership(r, chain, ids);
    return NewMeshShading{std::move(ids), std::move(classes), std::move(normals)};
}

void DeriveNewMeshShading(state::Registry &r, mtl::ComputeChain &chain, NewMeshShading &shading) {
    auto &meshes = r.Resources.get<MeshStore>();
    meshes.PlanCornerClassification(r, chain, shading.Classes);
    EncodeDeriveAllNormals(r, chain, shading.Normals);
    meshes.EnsureSelectionState(r, chain, shading.Ids);
}

void ShadeNewMeshes(state::Registry &r) {
    mtl::ComputeChain chain{r.Resources.get<MeshStore>().BufferContext()};
    auto shading = EncodeNewMeshShading(r, chain);
    if (!shading) return;
    chain.Submit();
    DeriveNewMeshShading(r, chain, *shading);
    chain.Submit();
}

uint32_t CreateShadedMesh(state::Registry &r, MeshSource source) {
    const auto id = CreateMesh(r, std::move(source)).StoreId;
    ShadeNewMeshes(r);
    return id;
}

void EncodeAuthoredCornerNormals(MeshStore &meshes, const MeshPipelines &pipelines, const Mesh &mesh, std::span<const vec3> authored) {
    const auto id = mesh.GetStoreId();
    const auto &record = meshes.Get(id);
    if (authored.empty() || record.TriangleCount == 0) return;
    // Authored normals stay in polygon corner order, independent of tessellation.
    const auto derived = meshes.GetCornerNormalView(id);
    const auto [faces, corners, positions] = ReadFaceLoops(meshes, pipelines, mesh);
    std::vector<CustomNormal> offsets(mesh.HalfEdgeCount());
    bool any = false;
    for (uint32_t fi = 0; fi < faces.size(); ++fi) {
        const auto face = corners.Of(fi);
        for (uint32_t k = 0; k < face.size(); ++k) {
            const uint32_t h = face[k].Halfedge;
            const auto authored_normal = authored[h], normal = derived[h];
            if (NormalsMatch(authored_normal, normal).value_or(true)) continue;
            const auto frame = ComputeCornerFrame(normal, positions[face[k].Vertex], positions[face[(k + 1u) % face.size()].Vertex], positions[face[(k + face.size() - 1u) % face.size()].Vertex]);
            offsets[h].Offset = EncodeNormalOffset(authored_normal / Length(authored_normal), frame);
            any = true;
        }
    }
    if (any) meshes.SetCustomCornerNormals(id, offsets);
}

void UpdateMorphShadingAuthored(MeshStore &meshes, const Mesh &mesh, std::span<const CornerNormalSources> poses) {
    const auto id = mesh.GetStoreId();
    const auto &record = meshes.Get(id);
    const auto &arenas = meshes.Arenas();
    meshes.SetMorphShadingAuthored(id, false);
    if (!record.HasAuthoredNormals || record.TriangleCount == 0 || record.MorphTargetCount == 0) return;
    bool has_authored_delta = false;
    arenas.Vertices.ForEach(record.Vertices, [&](ElementHandle handle, ElementAddress) {
        for (uint32_t target = 0; target < record.MorphTargetCount && !has_authored_delta; ++target) {
            has_authored_delta = arenas.Morph.Get(record.Morph, handle.Value, target, record.MorphTargetCount).NormalDelta != vec3{0};
        }
    });
    if (has_authored_delta) {
        meshes.SetMorphShadingAuthored(id, true);
        return;
    }
    if (poses.empty()) return;
    const auto vertices = mesh.TriangleVertices();
    const CornerAttributeView<uint32_t> classes{arenas.CornerSectors.View(record.CornerSectors), meshes.GetTriangleCorners(id)};
    const TriangleFaceView face_ids{meshes.GetTriangleCorners(id), arenas.HalfedgeFaces.Buffer.GetSpan<uint32_t>()};
    const auto compose = [&](const CornerNormalSources &normals, uint32_t ci) {
        return ComposeCornerNormal(meshes, id, classes, arenas.NormalSectors.View(record.NormalSectors), arenas.FaceSharpness.Buffer.GetSpan<uint8_t>(), record.Classification, ci, vertices, face_ids, normals);
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
