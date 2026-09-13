#pragma once

#include "Window.h"

namespace project {
struct Project;
void HandleHistoryShortcuts(Project &);
// Returns whether the user requested history clearing.
bool DrawHistoryWindow(Project &, HistoryWindow &, bool interactive);
} // namespace project
