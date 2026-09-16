#pragma once

#include "gpu/ElementSelectQuery.h"
#include "gpu/Types.h"

struct SelectionElementPushConstants {
    ElementSelectQuery Query DEFAULT();
};
static_assert(sizeof(SelectionElementPushConstants) == 40, "SelectionElementPushConstants size");
