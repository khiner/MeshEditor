#include "Ray.h"

#include <glm/gtx/norm.hpp>
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

float Ray::SquaredDistanceToPoint(const glm::vec3 &point) const {
    const vec3 ray_to_point = point - Origin;
    const float t = glm::dot(ray_to_point, Direction);
    const vec3 closest_point_on_ray = Origin + t * Direction;
    return glm::distance2(point, closest_point_on_ray);
}
