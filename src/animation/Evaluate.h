#pragma once

#include "state/Entity.h"

namespace state {
struct Scene;
} // namespace state

namespace animation {
// Writes every channel of every active clip at `seconds` into its field.
// Pose channels write PosedLocal, seeded from the node's Transform.
// `persistent` false writes only pose channels, for a restore where history has written the rest.
void Evaluate(state::Scene &, state::Entity viewport, float seconds, bool persistent = true);
} // namespace animation
