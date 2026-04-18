#include "TransformMath.h"
#include "numeric/vec4.h"

#include <glm/gtx/matrix_decompose.hpp>

Transform ToTransform(const mat4 &m) {
    vec3 scale, translation, skew;
    vec4 perspective;
    quat rotation;
    if (!glm::decompose(m, scale, rotation, translation, skew, perspective)) return {};
    return {translation, glm::normalize(rotation), scale};
}
