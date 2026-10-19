#pragma once

#include "state/DirtySet.h"
#include "viewport/ViewportSchema.h"

// Interaction, transform gesture, and theme change sets, bound by RegisterSceneComponentHandlers.
struct ViewportChanges {
    state::DirtySet InteractionMode, EditMode;
    state::DirtySet TransformPending, TransformStart, TransformEnd;
    state::DirtySet Theme;
};
