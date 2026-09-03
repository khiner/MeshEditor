#pragma once

#include "gpu/Element.h"
#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "selection/BoneSelection.h"

#include <entt/entity/fwd.hpp>

#include <vector>

constexpr uint32_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};

struct MeshElementSelection {};

struct MeshElementSelectionStats {
    uint32_t SelectedCount{}, SelectedVertexCount{};
    vec3 SelectedVertexPositionSum{};
    bool AnySharp{}, AnySmooth{};
};

struct ElementRange {
    entt::entity MeshEntity;
    uint32_t Offset, Count;
};

struct AdditiveBoxSelectBaseline {
    std::vector<entt::entity> SelectedEntities;
    std::vector<std::pair<entt::entity, BoneSelection>> BoneSelections;
    bool ElementSelectionCaptured{};
};

// Excite uses edit-selection storage temporarily and restores this authoritative edit domain on return to Edit mode.
struct ExciteSelectionBaseline {
    Element Mode{Element::None};
};

struct EditSelectionDirty {};

// Preserve the record-time projection so replay resolves pixels in the recorded coordinate system.
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

// Object or bone click-pick awaiting GPU resolution.
// Cycle advances to the next overlapping hit.
struct PendingPick {
    uvec2 MousePx;
    bool Shift;
    bool Cycle;
    mat4 ViewProj;
};

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
