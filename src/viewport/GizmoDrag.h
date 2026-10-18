#pragma once

#include "gpu/Transform.h"
#include "viewport/ViewportSchema.h"

struct StartTransform {
    Transform T;
    Transform ParentDelta;
};

struct StartBoneLength {
    float Value;
};
