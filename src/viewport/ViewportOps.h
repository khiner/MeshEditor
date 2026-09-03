#pragma once

#include "gpu/InteractionMode.h"

#include <entt/entity/fwd.hpp>

#include <string_view>

// Returns false if the requested interaction mode is unavailable.
bool SetInteractionMode(entt::registry &, entt::entity viewport, InteractionMode);

// Activates and lazily prefilters the studio HDRI at `index`.
// Falls back to index 0 if the name is not found.
void SetStudioEnvironment(entt::registry &, uint32_t index);
void SetStudioEnvironment(entt::registry &, std::string_view name);
void RebuildStudioEnvironments(entt::registry &);

// Emit the mode-appropriate delete/duplicate of the current selection.
void Delete(const entt::registry &, entt::entity viewport);
void Duplicate(const entt::registry &, entt::entity viewport);
