#pragma once

#include "gpu/Element.h"
#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "selection/BoneSelection.h"

#include <entt/entity/fwd.hpp>

#include <vector>

// Selection and activation bits shared by object, bone, and edit overlays.
constexpr uint32_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};

// Marks a mesh whose elements are individually selectable: the store holds its selection bits,
// and it keeps them across edit-mode switches.
struct MeshElementSelection {};

// Derived on the GPU whenever a mesh selection changes.
struct MeshElementSelectionStats {
    uint32_t SelectedCount{}, SelectedVertexCount{};
    vec3 SelectedVertexPositionSum{};
    bool AnySharp{}, AnySmooth{};
};

// A mesh's elements (vertices/edges/faces) in the store's selection bits: its first bit, and the
// element count of the current edit mode.
struct ElementRange {
    entt::entity MeshEntity;
    uint32_t Offset, Count;
};

// Snapshot of selection state at the start of a shift+box-drag.
// Presence on viewport means an additive box-drag is active.
struct AdditiveBoxSelectBaseline {
    std::vector<entt::entity> SelectedEntities;
    std::vector<std::pair<entt::entity, BoneSelection>> BoneSelections;
    bool ElementSelectionCaptured{};
};

// Excite temporarily publishes its sparse vertex mask through the edit-selection storage.
// Preserve the authoritative edit domain so returning to Edit restores the remembered selection.
struct ExciteSelectionBaseline {
    Element Mode{Element::None};
};

struct EditSelectionDirty {}; // GPU selection changed; the viewport needs a reuse submit.

// ViewProj is the record-time view-projection, stamped into SceneViewUBO so replay resolves pixels against it.
struct PendingEditElementClick {
    uvec2 MousePx;
    bool Toggle;
    mat4 ViewProj;
};

// Object/bone box-select awaiting GPU resolution against current scene state.
struct PendingBoxSelect {
    std::pair<uvec2, uvec2> BoxPx;
    bool Additive;
    mat4 ViewProj;
};
struct PendingBoxSelectFinalize {};
struct BoxSelectStatsDirty {};

// Object/bone click pick awaiting GPU resolution. Cycle advances to the next overlapping hit.
struct PendingPick {
    uvec2 MousePx;
    bool Shift;
    bool Cycle;
    mat4 ViewProj;
};

// Selection ignores occlusion when true.
struct SelectionXRay {
    bool Value{false};
};

enum class SelectionGesture : uint8_t {
    Click,
    Box,
};

struct BoxSelectState {
    SelectionGesture Gesture{SelectionGesture::Box};
};
