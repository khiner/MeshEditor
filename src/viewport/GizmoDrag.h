#pragma once

#include "gpu/Transform.h"

// Transform at start of gizmo manipulation. If present, actively manipulating.
struct StartTransform {
    Transform T;
    Transform ParentDelta; // parent_world * parent_inverse at drag start (identity if unparented)
};

// Bone display length captured at drag start (for head/tail partial transforms).
struct StartBoneLength {
    float Value;
};
