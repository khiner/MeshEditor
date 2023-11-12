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

float Ray::SquaredDistanceToEdge(const glm::vec3 &v1, const glm::vec3 &v2) const {
    const vec3 edge = v2 - v1;
    const vec3 ray_to_v1 = v1 - Origin;

    // Closest point on the edge line to the ray's origin.
    const float t_edge = glm::dot(ray_to_v1, edge) / glm::dot(edge, edge);
    const vec3 closest_point_on_edge_line = v1 + glm::clamp(t_edge, 0.0f, 1.0f) * edge;

    // Project this point onto the ray.
    const float t_ray = glm::dot(closest_point_on_edge_line - Origin, Direction);
    const vec3 closest_point_on_ray = t_ray < 0 ? Origin : Origin + t_ray * Direction;
    return glm::length2(closest_point_on_ray - closest_point_on_edge_line);
}
