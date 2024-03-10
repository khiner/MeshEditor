#include "Ray.h"

#include "numeric/mat4.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

// The origin is transformed using the full inverse transformation (including translation),
// while the direction is transformed using only the rotational part (ignoring translation) to maintain its directionality.
Ray Ray::WorldToLocal(const mat4 &model) const {
    const mat4 inv_model = glm::inverse(model);
    return {{inv_model * vec4{Origin, 1.f}}, glm::normalize(vec3{inv_model * vec4{Direction, 0.f}})};
}
