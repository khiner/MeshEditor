#pragma once

#include "Panel.h"

namespace project {
struct Project;
void HandleHistoryShortcuts(Project &);
// Returns whether the user requested history clearing.
bool DrawHistoryPanel(Project &, HistoryPanel &, bool interactive);
} // namespace project
