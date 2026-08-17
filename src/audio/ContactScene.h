#pragma once

#include "AcousticMaterial.h"
#include "AudioTypes.h"
#include "ModalModes.h"
#include "SoundVertices.h"
#include "SurfaceContact.h"
#include "TransformMath.h"
#include "mesh/Mesh.h"
#include "mesh/MeshBvh.h"
#include "render/GpuBufferOps.h"
#include "render/Instance.h"
#include "render/MeshBuffers.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"

#include <entt/entity/registry.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <optional>
#include <vector>

// What a contact looks up in the scene: which node carries the model, the surface and the geometry it reads, and where on the modal model a strike lands.
// Shared by the collision path and, where the surface-contact model is compiled in, by the sustained contacts it renders.

// The viewport's modal synthesis controls, or their defaults where no viewport carries any.
inline const ModalSoundControls &ModalControls(const entt::registry &r) {
    static constexpr ModalSoundControls Defaults{};
    const auto view = r.view<const ModalSoundControls>();
    return view.empty() ? Defaults : r.get<const ModalSoundControls>(view.front());
}

// A body that sounds by modal synthesis.
inline bool IsModalSounding(const entt::registry &r, entt::entity e) {
    return r.valid(e) && r.all_of<ModalModes, SoundVertices, SoundVerticesModel>(e) && r.get<SoundVerticesModel>(e) == SoundVerticesModel::Modal;
}

// A node's geometry is authored on the mesh it instances, so every lookup of one goes through its Instance.
// Its acoustic surface and material are the node's own, two nodes being able to instance one mesh and differ in both.
template<typename T> const T *AssetOf(const entt::registry &r, entt::entity node) {
    const auto *inst = r.try_get<const Instance>(node);
    return inst ? r.try_get<const T>(inst->Entity) : nullptr;
}

// Mean surface curvature (1/m) where a contact touches a node, read from that node's own mesh at `world_point`.
// Empty when the node has no mesh.
inline std::optional<double> SurfaceCurvature(const entt::registry &r, entt::entity node, vec3 world_point) {
    if (!r.valid(node)) return std::nullopt;
    const auto *inst = r.try_get<const Instance>(node);
    const auto *bvh = inst ? r.try_get<const MeshBvh>(inst->Entity) : nullptr;
    if (!bvh) return std::nullopt;
    const auto &wt = r.get<const WorldTransform>(node);
    const auto mesh = GetMesh(r, inst->Entity);
    const auto indices = GetFaceIndices(r, r.get<const MeshBuffers>(inst->Entity));
    const auto hit = bvh->ClosestPoint(mesh.GetVerticesSpan(), indices, InverseTransformPoint(wt, world_point));
    // Interpolate the triangle's per-vertex curvature at the contact's barycentric weights.
    double local = 0;
    for (uint32_t i = 0; i < 3; ++i) local += double(hit.Weights[i]) * bvh->MeanCurvature[hit.Vertices[i]];
    // Curvature is an inverse length, so the node's world scale converts the mesh's own units to meters.
    const float scale = MeanScale(wt.S);
    return scale > 0 ? local / scale : 0.0;
}

// The node a contact takes something from: the collider node it touched, or the nearest ancestor of it that has one.
// Falls back to the body when no collider along the way carries one.
inline entt::entity NearestNodeWith(const entt::registry &r, entt::entity collider, entt::entity body, auto &&has) {
    for (auto e = collider; e != null_entity && r.valid(e) && e != body; e = ParentOrNull(r, e)) {
        if (has(e)) return e;
    }
    return body;
}

// The nodes one side of a contact reads from.
struct ContactNodes {
    entt::entity Model, Surface, Geometry;
};
inline ContactNodes ResolveContactNodes(const entt::registry &r, entt::entity collider, entt::entity body) {
    return {
        // The model the contact excites, whose local space its position and directions are expressed in.
        // A body has one model however many colliders it has.
        NearestNodeWith(r, collider, body, [&r](entt::entity e) { return r.all_of<ModalModes>(e); }),
        ContactSurfaceNode(r, collider, body),
        // The mesh the contact reads its shape from, independent of which node carries the surface.
        NearestNodeWith(r, collider, body, [&r](entt::entity e) { return AssetOf<MeshBvh>(r, e) != nullptr; }),
    };
}

// A contact's elastic constants come from the material of the surface it touches, or from the one the node's modal model was derived from when that surface has none.
inline const AcousticMaterialProperties &MaterialOf(const entt::registry &r, entt::entity surface_node, entt::entity model_node) {
    const auto *mat = r.try_get<const AcousticMaterial>(surface_node);
    if (!mat) mat = r.try_get<const AcousticMaterial>(model_node);
    return mat ? mat->Properties : materials::acoustic::Steel.Properties;
}

// Reduce a (possibly non-uniform or mirrored) world scale to a positive size ratio relative to the baked size.
inline float UniformScaleRatio(const entt::registry &r, entt::entity e, const ModalModes &modes) {
    const auto *world = r.try_get<const WorldTransform>(e);
    const float baked = MeanScale(modes.BakedScale);
    return world && baked > 0 ? std::clamp(MeanScale(world->S) / baked, 0.001f, 1000.f) : 1.f;
}

// Unit direction of `v`, or zero when `v` has no length.
inline vec3 UnitOrZero(vec3 v) {
    const float len = glm::length(v);
    return len > 0 ? v / len : vec3{0};
}

// The sample point nearest `local_point`.
inline uint32_t NearestSamplePoint(const std::vector<vec3> &positions, vec3 local_point) {
    const auto dist2 = [local_point](vec3 p) { const auto d = p - local_point; return glm::dot(d, d); };
    return uint32_t(std::ranges::distance(positions.begin(), std::ranges::min_element(positions, {}, dist2)));
}

// The loudest mode an impulse of `j` at sample point `p` starts ringing, phi_n . j over the object's modes.
// Mode shapes are mass normalized, so this includes the body's mass and the strike's position where momentum alone does not.
inline float PeakModalDrive(const ModalModes &modes, uint32_t p, vec3 j) {
    if (p >= modes.Shapes.size()) return 0;
    float peak = 0;
    for (const auto &shape : modes.Shapes[p]) peak = std::max(peak, std::abs(glm::dot(shape, j)));
    return peak;
}
