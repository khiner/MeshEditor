#pragma once

#include "state/Entity.h"

struct FrameState;

void Interact(state::Scene &, state::Entity viewport, FrameState &Frame);
void InteractOverlay(state::Scene &, state::Entity viewport, FrameState &Frame);
void DrawOverlay(state::Scene &, state::Entity viewport, FrameState &Frame);
