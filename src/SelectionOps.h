#pragma once

#include "entt_fwd.h"

// Exclusive select: clears Selected/Active, then selects `e` (null clears everything).
void Select(entt::registry &, entt::entity);
void ToggleSelected(entt::registry &, entt::entity);
