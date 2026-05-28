#include "viewport/ViewportOps.h"

#include "action/Action.h"
#include "action/Bone.h"
#include "action/Object.h"
#include "audio/SoundVertices.h"
#include "gpu/ViewportTheme.h"
#include "mesh/MeshStore.h"
#include "scene/Entity.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportInteractionState.h"

#include <entt/entity/registry.hpp>

void SetLookThrough(entt::registry &r, entt::entity viewport, entt::entity target) {
    const auto previous = LookThroughCameraEntity(r);
    if (previous == target) return;
    // Preserve the saved view across camera switches; only capture fresh on first entry.
    auto saved = previous != entt::null ? r.get<LookingThrough>(previous).SavedViewCamera : r.get<ViewCamera>(viewport);
    if (previous != entt::null) r.remove<LookingThrough>(previous);
    r.emplace<LookingThrough>(target, std::move(saved));
}

entt::entity LookThroughCameraEntity(const entt::registry &r) {
    auto view = r.view<LookingThrough>();
    return view.empty() ? entt::null : *view.begin();
}

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
        // Keep bitset ranges + bits so element selections survive toggling Edit mode off and back on.
        for (const auto [mesh_entity, br, mesh] : r.view<const MeshSelectionBitsetRange, const Mesh>().each()) {
            if (br.Count > 0) meshes.UpdateElementStates(mesh, Element::None, {}, {}, {}, {}, std::nullopt);
        }
        r.emplace_or_replace<ElementStatesDirty>(viewport);
    }
    if (mode == InteractionMode::Edit && !active_is_armature) {
        // Only assign ranges for selected meshes missing one; existing ranges preserve remembered selection.
        if (const auto edit_element = r.get<const EditMode>(viewport).Value; edit_element != Element::None) {
            uint32_t next_offset = 0;
            for (const auto [_, br] : r.view<const MeshSelectionBitsetRange>().each()) {
                next_offset = std::max(next_offset, (br.Offset + br.Count + 31) / 32 * 32);
            }
            auto *bits = r.get<SelectionBitsetRef>(viewport).Value.data();
            for (const auto mesh_entity : selection::GetSelectedMeshEntities(r)) {
                if (r.all_of<MeshSelectionBitsetRange>(mesh_entity)) continue;
                const auto &mesh = r.get<const Mesh>(mesh_entity);
                const uint32_t count = selection::GetElementCount(mesh, edit_element);
                if (count == 0) continue;

                selection::SelectAll(bits, next_offset, count);
                r.emplace<MeshSelectionBitsetRange>(mesh_entity, next_offset, count);
                next_offset = (next_offset + count + 31) / 32 * 32;
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
