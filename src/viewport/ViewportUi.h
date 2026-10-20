#pragma once

#include "state/Entity.h"

struct FrameState;
struct Viewport;

// Routes mouse and keyboard input in the active viewport's panel.
void Interact(state::Registry &, FrameState &Frame);
// Draws the viewport panel's controls, which take input only in the active viewport.
void InteractOverlay(state::Registry &, FrameState &Frame, const Viewport &);
// Draws the viewport panel's overlay above its image, with the gesture guides in the active viewport.
void DrawOverlay(state::Registry &, FrameState &Frame, const Viewport &);
