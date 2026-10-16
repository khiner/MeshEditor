#pragma once

#include "Window.h"

namespace state {
struct Scene;
} // namespace state

namespace procedure {
void DrawNodesWindow(state::Scene &, state::Entity viewport, NodesWindow &, bool interactive);
void DrawNodeStyle(state::Scene &, state::Entity viewport);
} // namespace procedure
