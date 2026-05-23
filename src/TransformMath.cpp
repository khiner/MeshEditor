#include "TransformMath.h"

// Decompose a TRS matrix. Avoids glm::gtx::matrix_decompose: its determinant-vs-epsilon
// early-out (matrix_decompose.inl:54, "TODO: Fixme!") rejects matrices whose scale product
// is below float epsilon — e.g. quantized meshes with per-axis scale ~3.6e-6.
Transform ToTransform(const mat4 &m) {
    const vec3 translation{m[3]};
    vec3 col0{m[0]}, col1{m[1]}, col2{m[2]};
    vec3 scale{glm::length(col0), glm::length(col1), glm::length(col2)};
    // Negative det => mirrored; absorb the sign into scale.x by convention so the resulting
    // rotation has det +1. Dividing col0 by the now-negative scale.x produces the correct
    // rotation column without an extra flip.
    if (glm::determinant(glm::mat3{col0, col1, col2}) < 0) scale.x = -scale.x;
    constexpr float Eps = 1e-30f;
    if (std::abs(scale.x) > Eps) col0 /= scale.x;
    if (std::abs(scale.y) > Eps) col1 /= scale.y;
    if (std::abs(scale.z) > Eps) col2 /= scale.z;
    const quat rotation = glm::quat_cast(glm::mat3{col0, col1, col2});
    return {translation, glm::normalize(rotation), scale};
}
