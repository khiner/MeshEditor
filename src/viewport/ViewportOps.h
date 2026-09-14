#pragma once

#include "gpu/InteractionMode.h"

#include "state/Entity.h"

#include <string_view>

// Returns false if the requested interaction mode is unavailable.
bool SetInteractionMode(state::Scene &, state::Entity viewport, InteractionMode);

// Activates and lazily prefilters the studio HDRI at `index`.
// Falls back to index 0 if the name is not found.
void SetStudioEnvironment(state::Scene &, uint32_t index);
void SetStudioEnvironment(state::Scene &, std::string_view name);
void RebuildStudioEnvironments(state::Scene &);

// Emit the mode-appropriate delete/duplicate of the current selection.
void Delete(const state::Scene &, state::Entity viewport);
void Duplicate(const state::Scene &, state::Entity viewport);
