#pragma once

#include "gpu/Types.h"

// A vertex lies on a plane within a tolerance times the larger of one and its distance from the origin, else in front of it or behind it.
enum class PlaneSideKind : uint32_t {
    Positive,
    Negative,
    On,
};
// The tolerance the plane cut classifies vertices by.
GPU_CONSTANT float PlaneSideTolerance = 1e-4f;

#ifdef __METAL_VERSION__
inline PlaneSideKind PlaneSideOf(float distance, float3 p, float tolerance) {
    const float bound = tolerance * max(1.f, length(p));
    return distance > bound ? PlaneSideKind::Positive : distance < -bound ? PlaneSideKind::Negative :
                                                                            PlaneSideKind::On;
}
#endif
