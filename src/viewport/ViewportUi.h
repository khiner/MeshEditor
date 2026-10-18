#pragma once

#include "state/Entity.h"

struct FrameState;

void Interact(state::Registry &, state::Entity viewport, FrameState &Frame);
void InteractOverlay(state::Registry &, state::Entity viewport, FrameState &Frame);
void DrawOverlay(state::Registry &, state::Entity viewport, FrameState &Frame);
