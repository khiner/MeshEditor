#pragma once

#include "state/Entity.h"

struct FrameState;
struct Viewport;

// Routes mouse and keyboard input in the active viewport's window.
void Interact(state::Registry &, FrameState &Frame);
// Draws the viewport window's controls, which take input only in the active viewport.
void InteractOverlay(state::Registry &, FrameState &Frame, const Viewport &);
// Draws the viewport window's overlay above its image, with the gesture guides in the active viewport.
void DrawOverlay(state::Registry &, FrameState &Frame, const Viewport &);
