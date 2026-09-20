#pragma once

#include "numeric/uvec2.h"

#include "gpu/Element.h"
#include "numeric/vec2.h"
#include "selection/BoneSelection.h"
#include "viewport/RenderView.h"

#include "state/Entity.h"

#include <vector>

constexpr uint32_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};

struct MeshElementSelection {};

struct ElementRange {
    state::Entity MeshEntity;
    uint32_t Offset, Count;
};

// The selection an additive box drag started from, recorded by the drag's first update and removed when its gesture ends.
struct AdditiveBoxSelectBaseline {
    std::vector<state::Entity> SelectedEntities;
    std::vector<std::pair<state::Entity, BoneSelection>> BoneSelections;
    bool ElementSelectionCaptured{};
};

// Excite uses edit-selection storage temporarily and restores this authoritative edit domain on return to Edit mode.
struct ExciteSelectionBaseline {
    Element Mode{Element::None};
};

// Preserve the rendered camera so replay resolves pixels with the same rasterization and culling inputs.
struct PendingEditElementClick {
    uvec2 MousePx;
    bool Toggle;
    RenderView View;
};

// Object/bone box-select awaiting GPU resolution against current scene state.
struct PendingBoxSelect {
    std::pair<uvec2, uvec2> BoxPx;
    bool Additive;
    RenderView View;
};

// Object or bone click-pick awaiting GPU resolution.
// Cycle advances to the next overlapping hit.
struct PendingPick {
    uvec2 MousePx;
    bool Shift;
    bool Cycle;
    RenderView View;
};

enum class SelectionGesture : uint8_t {
    Click,
    Box,
};

struct BoxSelectState {
    SelectionGesture Gesture{SelectionGesture::Box};
};
