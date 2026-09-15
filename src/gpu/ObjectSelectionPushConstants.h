#pragma once

#include "gpu/Types.h"
#include "gpu/ObjectSelectQuery.h"

struct ObjectSelectionPushConstants {
    ObjectSelectQuery Query DEFAULT();
};
static_assert(sizeof(ObjectSelectionPushConstants) == 48, "ObjectSelectionPushConstants size");
