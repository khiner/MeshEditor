#pragma once

#include "gpu/Transform.h"
#include "numeric/mat4.h"

inline mat4 ToMatrix(const Transform &t) {
    return glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S);
}

Transform ToTransform(const mat4 &);

inline Transform ComposeLocalTransforms(const Transform &parent, const Transform &child) {
    return {parent.R * (parent.S * child.P) + parent.P, parent.R * child.R, parent.S * child.S};
}
