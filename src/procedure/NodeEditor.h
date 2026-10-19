#pragma once

#include "Window.h"

namespace state {
struct Registry;
} // namespace state

namespace procedure {
void DrawNodesWindow(state::Registry &, NodesWindow &, bool interactive);
void DrawNodeStyle(state::Registry &);
} // namespace procedure
