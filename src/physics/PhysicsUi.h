#pragma once

#include "state/Entity.h"

namespace physics_ui {
void RenderTab(state::Registry &, state::Entity viewport);
void RenderEntityProperties(state::Registry &, state::Entity, state::Entity viewport);
} // namespace physics_ui
