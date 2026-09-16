#pragma once

#include "gpu/ObjectSelectQuery.h"
#include "gpu/Types.h"
#include "gpu/VisibilityShadingPushConstants.h"

struct VisibilitySelectionPushConstants {
    VisibilityShadingPushConstants Visibility DEFAULT();
    ObjectSelectQuery Object DEFAULT();
    uvec2 Origin DEFAULT();
    uvec2 Extent DEFAULT();
};
static_assert(sizeof(VisibilitySelectionPushConstants) == 96, "VisibilitySelectionPushConstants size");
