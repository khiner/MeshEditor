#pragma once
#include "gpu/Types.h"
#include "state/Entity.h"

namespace state {
struct Registry;
}

vec3 ExciteDirection(const state::Registry &, state::Entity, uint32_t);
