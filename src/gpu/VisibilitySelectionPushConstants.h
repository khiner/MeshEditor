#pragma once

#include "gpu/Types.h"
#include "gpu/VisibilityShadingPushConstants.h"
#include "gpu/ObjectSelectQuery.h"

struct VisibilitySelectionPushConstants {
    VisibilityShadingPushConstants Visibility DEFAULT();
    ObjectSelectQuery Object DEFAULT();
    uvec2 Origin DEFAULT();
    uvec2 Extent DEFAULT();
};
static_assert(sizeof(VisibilitySelectionPushConstants) == 96, "VisibilitySelectionPushConstants size");
