#pragma once

#include "state/DirtySet.h"
#include "viewport/ViewportSchema.h"

// Viewport interaction, display, and view change sets, bound by RegisterSceneComponentHandlers.
struct ViewportChanges {
    state::DirtySet InteractionMode, EditMode;
    state::DirtySet TransformPending, TransformStart, TransformEnd;
    state::DirtySet Display, Theme, WorkspaceLights, StudioEnvironment;
    state::DirtySet PbrSpecialization; // The lighting the PBR pipelines specialize on changed.
    state::DirtySet SceneView, CameraLens;
};
