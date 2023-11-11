#include "Ray.h"

#include <glm/vec4.hpp>

using glm::vec3, glm::vec4, glm::mat4;

// The origin is transformed using the full inverse transformation (including translation),
// while the direction is transformed using only the rotational part (ignoring translation) to maintain its directionality.
Ray Ray::WorldToLocal(const mat4 &model) const {
    mat4 inv_model = glm::inverse(model);
    vec3 transformed_origin{inv_model * vec4{Origin, 1.f}};
    vec3 transformed_dir = glm::normalize(vec3{inv_model * vec4{Direction, 0.f}});
    return {transformed_origin, transformed_dir};
}
