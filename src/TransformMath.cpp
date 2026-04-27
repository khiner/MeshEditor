#include "TransformMath.h"

#include <cmath>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat3x3.hpp>

// Decompose a TRS matrix. Avoids glm::gtx::matrix_decompose: its determinant-vs-epsilon
// early-out (matrix_decompose.inl:54, "TODO: Fixme!") rejects matrices whose scale product
// is below float epsilon — e.g. quantized meshes with per-axis scale ~3.6e-6.
Transform ToTransform(const mat4 &m) {
    const vec3 translation{m[3]};
    vec3 col0{m[0]}, col1{m[1]}, col2{m[2]};
    vec3 scale{glm::length(col0), glm::length(col1), glm::length(col2)};
    // Negative det => mirrored; flip x by convention.
    if (glm::determinant(glm::mat3{col0, col1, col2}) < 0) {
        scale.x = -scale.x;
        col0 = -col0;
    }
    constexpr float Eps = 1e-30f;
    if (std::abs(scale.x) > Eps) col0 /= scale.x;
    if (std::abs(scale.y) > Eps) col1 /= scale.y;
    if (std::abs(scale.z) > Eps) col2 /= scale.z;
    const quat rotation = glm::quat_cast(glm::mat3{col0, col1, col2});
    return {translation, glm::normalize(rotation), scale};
}
