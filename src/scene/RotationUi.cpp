#include "scene/RotationUi.h"
#include "Variant.h"
#include "numeric/Angles.h"
#include "numeric/mat4.h"

#include <cmath>

quat ToRotation(const RotationUiVariant &v) {
    return std::visit(
        overloaded{
            [](const RotationQuat &q) { return numeric::Normalize(q.Value); },
            [](const RotationEuler &e) {
                const auto rads = numeric::Radians(e.Value);
                const auto rotation = numeric::AngleAxis(rads.z, {0, 0, 1}) * numeric::AngleAxis(rads.y, {0, 1, 0}) * numeric::AngleAxis(rads.x, {1, 0, 0});
                return numeric::Normalize(numeric::ToQuat(numeric::ToMat4(rotation)));
            },
            [](const RotationAxisAngle &a) {
                const auto axis = numeric::Normalize(vec3{a.Value});
                const auto angle = numeric::Radians(a.Value.w);
                return numeric::Normalize(quat{std::cos(angle / 2), axis * std::sin(angle / 2)});
            },
        },
        v
    );
}

RotationUiVariant ToUiVariant(quat rotation, size_t mode) {
    switch (mode) {
        case 1: {
            const auto euler = numeric::EulerAngles(numeric::ToQuat(numeric::ToMat4(rotation)));
            return RotationEuler{numeric::Degrees(euler)};
        }
        case 2: {
            const auto q = numeric::Normalize(rotation);
            return RotationAxisAngle{{numeric::Axis(q), numeric::Degrees(numeric::Angle(q))}};
        }
        default: return RotationQuat{rotation};
    }
}
