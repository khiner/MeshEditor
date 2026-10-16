#pragma once

#include "state/Entity.h"

namespace state {
struct Scene;
} // namespace state

namespace animation {
// Evaluate active channels at seconds and publish changed fields.
// Pose channels write PosedLocal, initialized from Transform.
// With persistent false, write only pose channels because history restores the other fields.
void Evaluate(state::Scene &, state::Entity viewport, float seconds, bool persistent = true);
} // namespace animation
