#pragma once

#include "gpu/InteractionMode.h"

#include "state/Entity.h"


// Returns false if the requested interaction mode is unavailable.
bool SetInteractionMode(state::Scene &, state::Entity viewport, InteractionMode);

// Emit the mode-appropriate delete/duplicate of the current selection.
void Delete(const state::Scene &, state::Entity viewport);
void Duplicate(const state::Scene &, state::Entity viewport);
