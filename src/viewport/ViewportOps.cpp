#include "viewport/ViewportOps.h"

#include "action/Bone.h"
#include "action/Emit.h"
#include "action/Object.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "gpu/ViewportTheme.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "scene/Entity.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionQueries.h"
#include "viewport/InteractionComponents.h"

#include <entt/entity/registry.hpp>

namespace TransformGizmo {
bool IsUsing(const entt::registry &r, entt::entity viewport) { return r.get<const GizmoInteraction>(viewport).IsUsing(); }
} // namespace TransformGizmo

bool SetInteractionMode(entt::registry &r, entt::entity viewport, InteractionMode mode) {
    const auto current_mode = r.get<const Interaction>(viewport).Mode;
    if (current_mode == mode) return false;

    const auto active_entity = FindActiveEntity(r);
    const auto active_arm = active_entity != entt::null ? FindArmatureObject(r, active_entity) : entt::null;
    const bool active_is_armature = active_arm != entt::null;
    if (mode == InteractionMode::Edit && !AllSelectedAreMeshes(r) && !active_is_armature) return false;
    if (mode == InteractionMode::Pose && !active_is_armature) return false;

    r.clear<VertexForce>();
    auto &meshes = r.ctx().get<MeshStore>();
    std::vector<ElementRange> initialize_selection;
    const auto edit_ranges = [&](Element element) {
        std::vector<ElementRange> ranges;
        if (element == Element::None) return ranges;
        for (const auto mesh_entity : r.view<const MeshElementSelection, const MeshHandle>()) {
            const auto mesh = GetMesh(r, mesh_entity);
            meshes.EnsureSelectionBits(mesh);
            const auto count = selection::GetElementCount(mesh, element);
            if (count > 0) ranges.emplace_back(mesh_entity, meshes.GetSelectionBitOffset(mesh.GetStoreId(), element), count);
        }
        return ranges;
    };

    if (current_mode == InteractionMode::Excite) {
        if (const auto *baseline = r.try_get<const ExciteSelectionBaseline>(viewport)) {
            const auto ranges = edit_ranges(baseline->Mode);
            ApplyEditSelectionCommand(r, viewport, ranges, baseline->Mode, EditSelectionOperation::RestoreBaseline);
            for (const auto &range : ranges) {
                const auto &summary = meshes.GetSelectionSummary(r.get<const MeshHandle>(range.MeshEntity).StoreId);
                if (summary.ActiveHandle < range.Count) r.emplace_or_replace<MeshActiveElement>(range.MeshEntity, summary.ActiveHandle);
                else r.remove<MeshActiveElement>(range.MeshEntity);
            }
        }
        r.remove<ExciteSelectionBaseline>(viewport);
    } else if (mode == InteractionMode::Excite) {
        const auto edit_element = r.get<const EditMode>(viewport).Value;
        const auto ranges = edit_ranges(edit_element);
        ApplyEditSelectionCommand(r, viewport, ranges, edit_element, EditSelectionOperation::CaptureBaseline);
        r.emplace_or_replace<ExciteSelectionBaseline>(viewport, edit_element);
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
                r.emplace<MeshElementSelection>(mesh_entity);
                initialize_selection.emplace_back(
                    mesh_entity, meshes.GetSelectionBitOffset(mesh.GetStoreId(), edit_element), count
                );
            }
        }
    }
    r.patch<Interaction>(viewport, [mode](auto &s) { s.Mode = mode; });
    if (!initialize_selection.empty()) {
        ApplyEditSelectionCommand(
            r, viewport, initialize_selection, r.get<const EditMode>(viewport).Value,
            EditSelectionOperation::Fill
        );
    }
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
