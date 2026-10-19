#include "viewport/ViewportOps.h"
#include "state/Registry.h"

#include "action/Bone.h"
#include "action/Emit.h"
#include "action/Object.h"
#include "audio/SoundVertices.h"
#include "gpu/ViewportTheme.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "viewport/InteractionComponents.h"

bool SetInteractionMode(state::Registry &r, InteractionMode mode) {
    const auto current_mode = r.get<const Interaction>(state::Document).Mode;
    if (current_mode == mode) return false;

    const auto active_entity = FindActiveEntity(r);
    const auto active_arm = active_entity != state::Null ? FindArmatureObject(r, active_entity) : state::Null;
    const bool active_is_armature = active_arm != state::Null;
    const auto &flags = r.get<const SelectionFlags>(state::Document);
    if (mode == InteractionMode::Edit && !flags.AllMeshes && !active_is_armature) return false;
    if (mode == InteractionMode::Pose && !active_is_armature) return false;

    r.clear<VertexForce>();
    auto &meshes = r.Resources.get<MeshStore>();
    std::vector<ElementRange> initialize_selection;
    const auto edit_ranges = [&](Element element) {
        std::vector<ElementRange> ranges;
        if (element == Element::None) return ranges;
        for (const auto mesh_entity : r.view<const MeshElementSelection, const MeshHandle>()) {
            const auto mesh = GetDrawnMesh(r, mesh_entity);
            const auto count = mesh.ElementCount(element);
            if (count > 0) ranges.emplace_back(mesh_entity, count);
        }
        return ranges;
    };

    if (current_mode == InteractionMode::Excite) {
        if (const auto *baseline = r.try_get<const ExciteSelectionBaseline>(state::Document)) {
            const auto ranges = edit_ranges(baseline->Mode);
            ApplyEditSelectionCommand(r, ranges, baseline->Mode, EditSelectionOperation::RestoreBaseline);
            for (const auto &range : ranges) {
                const auto &summary = meshes.GetSelectionSummary(*DrawnStoreId(r, range.MeshEntity));
                if (summary.ActiveHandle < range.Count) r.emplace_or_replace<MeshActiveElement>(range.MeshEntity, summary.ActiveHandle);
                else r.remove<MeshActiveElement>(range.MeshEntity);
            }
        }
        r.remove<ExciteSelectionBaseline>(state::Document);
    } else if (mode == InteractionMode::Excite) {
        const auto edit_element = r.get<const EditMode>(state::Document).Value;
        const auto ranges = edit_ranges(edit_element);
        ApplyEditSelectionCommand(r, ranges, edit_element, EditSelectionOperation::CaptureBaseline);
        r.emplace_or_replace<ExciteSelectionBaseline>(state::Document, edit_element);
    }

    std::vector<state::Entity> converting;
    if (mode == InteractionMode::Edit && !active_is_armature) {
        // Take bits only for selected meshes without them.
        // A mesh that has them keeps its remembered selection, converted to the element mode it was stored in otherwise.
        if (const auto edit_element = r.get<const EditMode>(state::Document).Value; edit_element != Element::None) {
            std::vector<state::Entity> taking;
            for (const auto mesh_entity : flags.Meshes) {
                if (r.all_of<MeshElementSelection>(mesh_entity)) {
                    const auto id = *DrawnStoreId(r, mesh_entity);
                    if (meshes.Get(id).SelectionSummary.Count && meshes.GetSelectionSummary(id).Mode != edit_element) converting.push_back(mesh_entity);
                    continue;
                }
                if (GetDrawnMesh(r, mesh_entity).ElementCount(edit_element) == 0) continue;
                taking.push_back(mesh_entity);
            }
            for (const auto mesh_entity : taking) {
                const auto mesh = GetDrawnMesh(r, mesh_entity);
                r.emplace<MeshElementSelection>(mesh_entity);
                initialize_selection.emplace_back(
                    mesh_entity, mesh.ElementCount(edit_element)
                );
            }
        }
    }
    r.patch<Interaction>(state::Document, [mode](auto &s) { s.Mode = mode; });
    ConvertElementSelections(r, converting, r.get<const EditMode>(state::Document).Value);
    if (!initialize_selection.empty()) {
        ApplyEditSelectionCommand(r, initialize_selection, r.get<const EditMode>(state::Document).Value, EditSelectionOperation::Fill);
    }
    r.patch<ViewportTheme>(state::Document, [](auto &) {});
    return true;
}

void Delete(const state::Registry &r) {
    if (IsBoneEditMode(r)) action::Emit(action::bone::DeleteSelected{});
    else action::Emit(action::object::Delete{});
}
void Duplicate(const state::Registry &r) {
    if (IsBoneEditMode(r)) action::Emit(action::bone::DuplicateSelected{});
    else action::Emit(action::object::Duplicate{}, action::Phase::Stage);
}
