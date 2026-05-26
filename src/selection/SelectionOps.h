#pragma once

#include "entt_fwd.h"

// Exclusive select: clears Selected/Active, then selects `e` (null clears everything).
void Select(entt::registry &, entt::entity);

// Exclusive bone select: clears BoneSelection/BoneActive, then selects `e` (null clears everything).
void SelectBone(entt::registry &, entt::entity);
