#pragma once

#include "Panel.h"

namespace state {
struct Registry;
} // namespace state

namespace procedure {
void DrawNodesPanel(state::Registry &, NodesPanel &, bool interactive);
void DrawNodeStyle(state::Registry &);
} // namespace procedure
