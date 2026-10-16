#pragma once

#include "Window.h"

namespace state {
struct Scene;
} // namespace state

namespace procedure {
// Draws the node editor over the procedure of the active mesh entity, and emits its edits as actions when interactive.
void DrawNodesWindow(state::Scene &, state::Entity viewport, NodesWindow &, bool interactive);
// Draws the node theme editor, emitting its edits as actions.
void DrawNodeStyle(state::Scene &, state::Entity viewport);
} // namespace procedure
