#pragma once

#include "AcousticMaterial.h"
#include "AudioTypes.h"
#include "ModalModes.h"
#include "SoundVertices.h"
#include "SurfaceContact.h"
#include "TransformMath.h"
#include "mesh/MeshPipelines.h"
#include "mesh/Neighborhood.h"
#include "render/Instance.h"
#include "render/MeshletSpatial.h"
#include "scene/HierarchyInstance.h"
#include "scene/SceneGraph.h"

inline const ModalSoundControls &ModalControls(const state::Registry &r) {
    static constexpr ModalSoundControls Defaults{};
    const auto view = r.view<const ModalSoundControls>();
    return view.empty() ? Defaults : r.get<const ModalSoundControls>(view.front());
}

inline bool IsModalSounding(const state::Registry &r, state::Entity e) {
    return r.valid(e) && r.all_of<ModalModes>(HierarchySource(r, e)) && r.all_of<SoundVertices, SoundVerticesModel>(e) && r.get<SoundVerticesModel>(e) == SoundVerticesModel::Modal;
}

// Mean surface curvature in 1/m where a contact touches a node, read from that node's own mesh at `world_point`.
// Empty when the node has no mesh.
inline std::optional<double> SurfaceCurvature(const state::Registry &r, state::Entity node, vec3 world_point) {
    if (!r.valid(node)) return std::nullopt;
    const auto *inst = r.try_get<const Instance>(node);
    const auto *owner = inst ? TryRecordOf(r, inst->Entity) : nullptr;
    if (!owner || owner->SpatialRoot == InvalidOffset) return std::nullopt;
    const auto &wt = *WorldTransformOf(r, node);
    const auto mesh = GetMesh(r, inst->Entity);
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto hit = ClosestMeshletPoint(meshes.Render(), *owner, mesh.VertexValues(), mesh.TriangleVertices(), InverseTransformPoint(wt, world_point));
    const auto &pipelines = GetMeshPipelines(r);
    const auto id = mesh.GetStoreId();
    const auto edges = VertexEdges(meshes, pipelines, id, hit.Vertices);
    std::vector<uint32_t> positioned{hit.Vertices.begin(), hit.Vertices.end()};
    for (const auto &edge : edges.Items) positioned.push_back(edge.Other);
    const auto positions = VertexPositions(meshes, pipelines, id, positioned);
    const auto normals = VertexNormals(meshes, pipelines, id, hit.Vertices);
    const auto sharpness = meshes.Arenas().EdgeSharpness.Buffer.GetSpan<uint8_t>();
    double local = 0;
    for (uint32_t i = 0, at = 3; i < 3; ++i) {
        const vec3 xi = positions[i], ni = Normalize(normals[i]);
        double sum = 0;
        int count = 0;
        bool sharp = false;
        for (const auto &edge : edges.Of(i)) {
            const auto address = meshes.HandleAddress(id, MeshStore::ElementDomain::Edge, ElementHandle{edge.Edge}).Value();
            sharp |= sharpness[address] != 0;
            if (edge.Sides >= 2u) {
                const vec3 d = positions[at] - xi;
                if (const double d2 = Dot(d, d); d2 >= 1e-20) {
                    sum += -2.0 * double(Dot(d, ni)) / d2;
                    ++count;
                }
            }
            ++at;
        }
        local += double(hit.Weights[i]) * (sharp || !count ? 0.0 : sum / count);
    }
    // Curvature is an inverse length, so dividing by the node's world scale converts it to 1/m.
    const float scale = MeanScale(wt.S);
    return scale > 0 ? local / scale : 0.0;
}

inline state::Entity NearestNodeWith(const state::Registry &r, state::Entity collider, state::Entity body, auto &&has) {
    for (auto e = collider; e != state::Null && r.valid(e) && e != body; e = ParentOrNull(r, e)) {
        if (has(e)) return e;
    }
    return body;
}

struct ContactNodes {
    state::Entity Model, Surface, Geometry;
};
inline ContactNodes ResolveContactNodes(const state::Registry &r, state::Entity collider, state::Entity body) {
    return {
        NearestNodeWith(r, collider, body, [&r](state::Entity e) { return r.all_of<ModalModes>(e); }),
        ContactSurfaceNode(r, collider, body),
        NearestNodeWith(r, collider, body, [&r](state::Entity e) {
            const auto *inst = r.try_get<const Instance>(e);
            const auto *owner = inst ? TryRecordOf(r, inst->Entity) : nullptr;
            return owner && owner->SpatialRoot != InvalidOffset;
        }),
    };
}

inline const AcousticMaterialProperties &MaterialOf(const state::Registry &r, state::Entity surface_node, state::Entity model_node) {
    const auto *mat = r.try_get<const AcousticMaterial>(surface_node);
    if (!mat) mat = r.try_get<const AcousticMaterial>(model_node);
    return mat ? mat->Properties : materials::acoustic::Steel.Properties;
}

// Reduces a non-uniform or mirrored world scale to a positive size ratio relative to the baked size.
inline float UniformScaleRatio(const state::Registry &r, state::Entity e, const ModalModes &modes) {
    const auto *world = WorldTransformOf(r, e);
    const float baked = MeanScale(modes.BakedScale);
    return world && baked > 0 ? std::clamp(MeanScale(world->S) / baked, 0.001f, 1000.f) : 1.f;
}

inline vec3 UnitOrZero(vec3 v) {
    const float len = Length(v);
    return len > 0 ? v / len : vec3{0};
}

inline uint32_t NearestSamplePoint(std::span<const vec3> positions, vec3 local_point) {
    const auto dist2 = [local_point](vec3 p) { const auto d = p - local_point; return Dot(d, d); };
    return uint32_t(std::ranges::distance(positions.begin(), std::ranges::min_element(positions, {}, dist2)));
}

// The largest |phi_n . j| over the modes for an impulse `j` at sample point `p`.
// Mode shapes are mass normalized, so the result accounts for body mass and strike position.
inline float PeakModalDrive(const ModalModes &modes, uint32_t p, vec3 j) {
    if (p >= modes.Shapes.size()) return 0;
    float peak = 0;
    for (const auto &shape : modes.Shapes[p]) peak = std::max(peak, std::abs(Dot(shape, j)));
    return peak;
}
