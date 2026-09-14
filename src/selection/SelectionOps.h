#pragma once

#include "state/Entity.h"

// Exclusive select: clears Selected/Active, then selects `e` (null clears everything).
void Select(state::Scene &, state::Entity);

// Exclusive bone select: clears BoneSelection/BoneActive, then selects `e` (null clears everything).
void SelectBone(state::Scene &, state::Entity);
