#pragma once

#include "selection/BoneSelection.h"
#include "viewport/RenderView.h"

#include "selection/SelectionSchema.h"
#include "state/Entity.h"

#include <vector>

constexpr uint32_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};

struct MeshElementSelection {};

struct ElementRange {
    state::Entity MeshEntity;
    uint32_t Count;
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
// Each pending gesture resolves against the viewport it was emitted in, or the first viewport when that one is gone.
struct PendingEditElementClick {
    vec2 Mouse;
    bool Toggle;
    RenderView View;
    uint32_t ViewportId;
};

// Object/bone box-select awaiting GPU resolution against current scene state.
struct PendingBoxSelect {
    std::pair<vec2, vec2> Box;
    bool Additive;
    RenderView View;
    uint32_t ViewportId;
};

// Object or bone click-pick awaiting GPU resolution.
// Cycle advances to the next overlapping hit.
struct PendingPick {
    vec2 Mouse;
    bool Shift;
    bool Cycle;
    RenderView View;
    uint32_t ViewportId;
};

enum class SelectionGesture : uint8_t {
    Click,
    Box,
};

struct BoxSelectState {
    SelectionGesture Gesture{SelectionGesture::Box};
};
