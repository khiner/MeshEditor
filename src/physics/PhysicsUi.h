#pragma once

#include "state/Entity.h"

namespace physics_ui {
bool RenderResourceProperties(state::Registry &, state::Entity);
void RenderSimulation(state::Registry &); // The simulation settings, the body count, and the collision matrix of the filters.
void RenderEntityProperties(state::Registry &, state::Entity, bool explicit_target = false);
} // namespace physics_ui
