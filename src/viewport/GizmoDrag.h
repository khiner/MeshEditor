#pragma once

#include "gpu/Transform.h"

struct StartTransform {
    Transform T;
    Transform ParentDelta;
};

struct StartBoneLength {
    float Value;
};
