#pragma once

#include "Window.h"

namespace state {
struct Registry;
} // namespace state

namespace procedure {
void DrawNodesWindow(state::Registry &, state::Entity viewport, NodesWindow &, bool interactive);
void DrawNodeStyle(state::Registry &, state::Entity viewport);
} // namespace procedure
