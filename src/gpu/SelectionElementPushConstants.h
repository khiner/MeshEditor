#pragma once

#include "gpu/Types.h"
#include "gpu/ElementSelectQuery.h"

struct SelectionElementPushConstants {
    ElementSelectQuery Query DEFAULT();
};
static_assert(sizeof(SelectionElementPushConstants) == 40, "SelectionElementPushConstants size");
