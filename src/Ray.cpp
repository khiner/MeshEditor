#include "Ray.h"

#include "numeric/vec4.h"

Ray Ray::WorldToLocal(const mat4 &transp_inv_transform) const {
    const auto inv_transpose = glm::transpose(transp_inv_transform);
    return {{inv_transpose * vec4{Origin, 1.f}}, glm::normalize(vec3{inv_transpose * vec4{Direction, 0.f}})};
}
