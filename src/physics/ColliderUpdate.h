#pragma once
#include "state/Entity.h"
#include <span>
// Rederives each entity's collider shape, fitting auto-fit dimensions to its mesh's vertex bounds.
void RederiveColliders(state::Scene &, std::span<const state::Entity>);
