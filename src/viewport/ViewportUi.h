#pragma once

#include "state/Entity.h"

struct FrameState;

// Routes mouse and keyboard input in the active viewport's panel.
void Interact(state::Registry &, FrameState &Frame);
// Draws the viewport panel's controls, which take input only when `active` marks the document's active viewport.
void InteractOverlay(state::Registry &, FrameState &Frame, state::Entity viewport, bool active);
// Draws the viewport panel's overlay above its image, with the gesture guides when `active` marks the document's active viewport.
void DrawOverlay(state::Registry &, FrameState &Frame, state::Entity viewport, bool active);
