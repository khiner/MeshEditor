#include "scene/RotationUi.h"
#include "Variant.h"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>

#include <cmath>

quat ToRotation(const RotationUiVariant &v) {
    return std::visit(
        overloaded{
            [](const RotationQuat &q) { return glm::normalize(q.Value); },
            [](const RotationEuler &e) {
                const auto rads = glm::radians(e.Value);
                return glm::normalize(glm::quat_cast(glm::eulerAngleXYZ(rads.x, rads.y, rads.z)));
            },
            [](const RotationAxisAngle &a) {
                const auto axis = glm::normalize(vec3{a.Value});
                const auto angle = glm::radians(a.Value.w);
                return glm::normalize(quat{std::cos(angle / 2), axis * std::sin(angle / 2)});
            },
        },
        v
    );
}

RotationUiVariant ToUiVariant(quat rotation, std::size_t mode) {
    switch (mode) {
        case 1: {
            float x, y, z;
            glm::extractEulerAngleXYZ(glm::mat4_cast(rotation), x, y, z);
            return RotationEuler{glm::degrees(vec3{x, y, z})};
        }
        case 2: {
            const auto q = glm::normalize(rotation);
            return RotationAxisAngle{{glm::axis(q), glm::degrees(glm::angle(q))}};
        }
        default: return RotationQuat{rotation};
    }
}
