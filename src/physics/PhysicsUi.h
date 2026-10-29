#pragma once

#include "state/Entity.h"

namespace physics_ui {
bool RenderResourceProperties(state::Registry &, state::Entity);
void RenderTab(state::Registry &);
void RenderEntityProperties(state::Registry &, state::Entity, bool explicit_target = false);
} // namespace physics_ui
