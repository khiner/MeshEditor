#pragma once

#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "selection/BoneSelection.h"

#include <entt/entity/fwd.hpp>

#include <span>
#include <vector>

// Per-element state bits in the element state buffers.
constexpr uint32_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};

// A contiguous span of a mesh's elements (vertices/edges/faces) in the SelectionBitset.
struct ElementRange {
    entt::entity MeshEntity;
    uint32_t Offset, Count;
};

// Snapshot of selection state at the start of a shift+box-drag.
// Presence on viewport means an additive box-drag is active.
struct AdditiveBoxSelectBaseline {
    std::vector<entt::entity> SelectedEntities;
    std::vector<std::pair<entt::entity, BoneSelection>> BoneSelections;
    std::vector<uint32_t> ElementBitset;
};

// Transient dirty flags on the viewport.
struct SelectionBitsDirty {}; // The bitset changed, the compute update is pending.
struct ElementStatesDirty {}; // The element state buffers changed, a submit is pending.

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

// Object/bone click pick awaiting GPU resolution. Cycle advances to the next overlapping hit.
struct PendingPick {
    uvec2 MousePx;
    bool Shift;
    bool Cycle;
    mat4 ViewProj;
};

// Non-owning span over the GPU-mapped SelectionBitset words.
struct SelectionBitsetRef {
    std::span<uint32_t> Value;
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

struct SelectedInstanceCount {
    uint32_t Value{0};
};
