#include "armature/ArmatureComponents.h"
#include "gizmo/GizmoInteraction.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "viewport/InteractionComponents.h"

#include "state/Scene.h"

namespace selection {
namespace {
template<typename F>
void ForEachEditInstance(const state::Scene &r, F &&f) {
    const auto active = FindActiveEntity(r);
    for (const auto [e, instance, ok, ri] : r.view<const Instance, const Selected, const ObjectKind, const RenderInstance>().each()) {
        if (ok.Value == ObjectType::Mesh) f(instance.Entity, e, active, r.all_of<ScaleLocked>(e));
    }
}

void ChoosePrimary(PrimaryEditInstanceMap &primaries, state::Entity mesh, state::Entity instance, state::Entity active) {
    auto &primary = primaries.try_emplace(mesh, instance).first->second;
    if (instance == active || (primary != active && instance < primary)) primary = instance;
}
} // namespace

PrimaryEditInstanceMap ComputePrimaryEditInstances(const state::Scene &r, bool include_scale_locked) {
    PrimaryEditInstanceMap primaries;
    ForEachEditInstance(r, [&](state::Entity mesh, state::Entity instance, state::Entity active, bool scale_locked) {
        if (include_scale_locked || !scale_locked) ChoosePrimary(primaries, mesh, instance, active);
    });
    return primaries;
}

PrimaryEditInstanceMaps ComputePrimaryEditInstanceMaps(const state::Scene &r) {
    PrimaryEditInstanceMaps result;
    ForEachEditInstance(r, [&](state::Entity mesh, state::Entity instance, state::Entity active, bool scale_locked) {
        ChoosePrimary(result.All, mesh, instance, active);
        if (!scale_locked) ChoosePrimary(result.Transformable, mesh, instance, active);
    });
    return result;
}

bool HasScaleLockedInstance(const state::Scene &r, state::Entity e) {
    for (const auto [_, instance] : r.view<const Instance, const ScaleLocked>().each()) {
        if (instance.Entity == e) return true;
    }
    return false;
}

std::unordered_set<state::Entity> GetSelectedMeshEntities(const state::Scene &r) {
    std::unordered_set<state::Entity> entities;
    for (const auto [e, instance] : r.view<const Instance, const Selected>().each()) {
        if (HasMesh(r, instance.Entity)) entities.emplace(instance.Entity);
    }
    return entities;
}

} // namespace selection

state::Entity FindArmatureObject(const state::Scene &r, state::Entity e) {
    if (e == state::Null) return state::Null;
    if (r.all_of<ArmatureObject>(e)) return e;
    if (const auto *sub = r.try_get<SubElementOf>(e); sub && r.all_of<ArmatureObject>(sub->Parent)) return sub->Parent;
    return state::Null;
}

state::Entity FindActiveBone(const state::Scene &r) {
    state::Entity result = state::Null;
    for (const auto e : r.view<BoneActive>()) {
        assert(result == state::Null && "Multiple BoneActive entities");
        result = e;
    }
    return result;
}

namespace TransformGizmo {
bool IsUsing(const state::Scene &r, state::Entity viewport) { return r.get<const GizmoInteraction>(viewport).IsUsing(); }
} // namespace TransformGizmo
