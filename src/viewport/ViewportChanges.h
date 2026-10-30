#pragma once

// Interaction, transform gesture, theme, and view change sets, bound by RegisterSceneComponentHandlers.
struct ViewportChanges {
    state::DirtySet InteractionMode, EditMode;
    state::DirtySet TransformPending, TransformEnd;
    state::DirtySet Theme;
    state::DirtySet View; // The viewports whose workspace values changed.
};
