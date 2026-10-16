#pragma once

#include "gpu/ObjectSelectQuery.h"
#include "gpu/Types.h"

struct ObjectSelectionPushConstants {
    ObjectSelectQuery Query DEFAULT();
};
