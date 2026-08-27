#include "viewport/ViewportOps.h"

#include "action/Bone.h"
#include "action/Emit.h"
#include "action/Object.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "gpu/ViewportTheme.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "scene/Entity.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "viewport/InteractionComponents.h"

#include <entt/entity/registry.hpp>

namespace TransformGizmo {
bool IsUsing(const entt::registry &r, entt::entity viewport) { return r.get<const GizmoInteraction>(viewport).IsUsing(); }
} // namespace TransformGizmo

bool SetInteractionMode(entt::registry &r, entt::entity viewport, InteractionMode mode) {
    if (r.get<const Interaction>(viewport).Mode == mode) return false;

    const auto active_entity = FindActiveEntity(r);
    const auto active_arm = active_entity != entt::null ? FindArmatureObject(r, active_entity) : entt::null;
    const bool active_is_armature = active_arm != entt::null;
    if (mode == InteractionMode::Edit && !AllSelectedAreMeshes(r) && !active_is_armature) return false;
    if (mode == InteractionMode::Pose && !active_is_armature) return false;

    r.clear<VertexForce>();
    auto &meshes = r.ctx().get<MeshStore>();
    if (r.get<const Interaction>(viewport).Mode == InteractionMode::Edit) {
        // Element states are display-only. The selection bits keep the selection, and entering Edit mode rederives them.
        for (const auto mesh_entity : r.view<const MeshElementSelection>()) {
            if (HasMesh(r, mesh_entity)) meshes.ClearElementStates(GetMesh(r, mesh_entity));
        }
        r.emplace_or_replace<ElementStatesDirty>(viewport);
    }

    if (mode == InteractionMode::Edit && !active_is_armature) {
        // Take bits only for selected meshes without them.
        // A mesh that has them keeps its remembered selection.
        if (const auto edit_element = r.get<const EditMode>(viewport).Value; edit_element != Element::None) {
            for (const auto mesh_entity : selection::GetSelectedMeshEntities(r)) {
                if (r.all_of<MeshElementSelection>(mesh_entity)) continue;
                const auto mesh = GetMesh(r, mesh_entity);
                const uint32_t count = selection::GetElementCount(mesh, edit_element);
                if (count == 0) continue;

                meshes.EnsureSelectionBits(mesh);
                selection::SelectAll(meshes.GetMutableSelectionBits(mesh.GetStoreId()), count);
                r.emplace<MeshElementSelection>(mesh_entity);
            }
        }
    }
    r.patch<Interaction>(viewport, [mode](auto &s) { s.Mode = mode; });
    r.patch<ViewportTheme>(viewport, [](auto &) {});
    return true;
}

void Delete(const entt::registry &r, entt::entity viewport) {
    if (IsBoneEditMode(r, viewport)) action::Emit(action::bone::DeleteSelected{});
    else action::Emit(action::object::Delete{});
}
void Duplicate(const entt::registry &r, entt::entity viewport) {
    if (IsBoneEditMode(r, viewport)) action::Emit(action::bone::DuplicateSelected{});
    else action::Emit(action::object::Duplicate{});
}
