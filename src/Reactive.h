#pragma once

#include "state/Scene.h"

using state::On;

template<typename Change>
auto &reactive(state::Scene &r) { return r.changes(state::Type<Change>()); }
