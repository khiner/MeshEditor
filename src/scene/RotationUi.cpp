#include "scene/RotationUi.h"

#include "Variant.h"
#include "numeric/Angles.h"
#include "numeric/MatrixMath.h"

#include <cmath>

using numeric::AngleAxis, numeric::Degrees, numeric::Radians;

quat ToRotation(const RotationUiVariant &v) {
    return std::visit(
        overloaded{
            [](const RotationQuat &q) { return Normalize(q.Value); },
            [](const RotationEuler &e) {
                const auto rads = Radians(e.Value);
                const auto rotation = AngleAxis(rads.z, {0, 0, 1}) * AngleAxis(rads.y, {0, 1, 0}) * AngleAxis(rads.x, {1, 0, 0});
                return Normalize(ToQuat(ToMat4(rotation)));
            },
            [](const RotationAxisAngle &a) {
                const auto axis = Normalize(vec3{a.Value});
                const auto angle = Radians(a.Value.w);
                return Normalize(quat{std::cos(angle / 2), axis * std::sin(angle / 2)});
            },
        },
        v
    );
}

RotationUiVariant ToUiVariant(quat rotation, size_t mode) {
    switch (mode) {
        case 1: {
            const auto euler = EulerAngles(ToQuat(ToMat4(rotation)));
            return RotationEuler{Degrees(euler)};
        }
        case 2: {
            const auto q = Normalize(rotation);
            return RotationAxisAngle{{Axis(q), Degrees(Angle(q))}};
        }
        default: return RotationQuat{rotation};
    }
}
