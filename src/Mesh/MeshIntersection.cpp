#include "MeshIntersection.h"

#include "BVH.h"
#include "numeric/ray.h"

using namespace he;

namespace MeshIntersection {

namespace {
constexpr float SquaredDistanceToLineSegment(const vec3 &v1, const vec3 &v2, const vec3 &p) noexcept {
    const vec3 edge = v2 - v1;
    const float t = glm::clamp(glm::dot(p - v1, edge) / glm::dot(edge, edge), 0.f, 1.f);
    const vec3 closest_p = v1 + t * edge;
    const vec3 diff = p - closest_p;
    return glm::dot(diff, diff);
}

// Moller-Trumbore ray-triangle intersection algorithm.
// Returns true if the ray (starting at `o` and traveling along `d`) intersects the given triangle.
// If ray intersects, returns the distance along the ray to the intersection point.
constexpr std::optional<float> IntersectTriangle(vec3 o, vec3 d, vec3 p1, vec3 p2, vec3 p3) noexcept {
    static constexpr float eps = 1e-7f; // Floating point error tolerance.

    const auto e1 = p2 - p1, e2 = p3 - p1;
    const auto h = glm::cross(d, e2);
    const float a = glm::dot(e1, h); // Barycentric coordinate
    if (a > -eps && a < eps) return {}; // Check if the ray is parallel to the triangle.

    // Check if the intersection point is inside the triangle (in barycentric coordinates).
    const auto s = o - p1;
    const float f = 1.f / a, u = f * glm::dot(s, h);
    if (u < 0.f || u > 1.f) return {};

    const auto q = glm::cross(s, e1);
    if (float v = f * glm::dot(d, q); v < 0.f || u + v > 1.f) return {};

    // Calculate the intersection point's distance along the ray and verify it's ahead of the ray's origin.
    if (float distance = f * glm::dot(e2, q); distance > eps) return {distance};
    return {};
}

constexpr std::optional<float> IntersectFace(const ray &ray, uint fi, const void *m) noexcept {
    const auto o = ray.o, d = ray.d;
    const auto &pm = *reinterpret_cast<const PolyMesh *>(m);
    auto fv_it = pm.cfv_iter(FH(fi));
    const auto v0 = *fv_it++;
    VH v1 = *fv_it++, v2;
    for (; fv_it; ++fv_it) {
        v2 = *fv_it;
        if (auto distance = IntersectTriangle(o, d, pm.GetPosition(v0), pm.GetPosition(v1), pm.GetPosition(v2))) return {distance};
        v1 = v2;
    }
    return {};
}
} // namespace

FH FindNearestIntersectingFace(const BVH &bvh, const PolyMesh &polymesh, const ray &ray, vec3 *nearest_intersect_point_out) {
    if (auto intersection = bvh.IntersectNearest(ray, IntersectFace, &polymesh)) {
        if (nearest_intersect_point_out) *nearest_intersect_point_out = ray(intersection->Distance);
        return {intersection->Index};
    }
    return {};
}

std::optional<Intersection> Intersect(const BVH &bvh, const PolyMesh &polymesh, const ray &ray) {
    return bvh.IntersectNearest(ray, IntersectFace, &polymesh);
}

VH FindNearestVertex(const BVH &bvh, const PolyMesh &polymesh, const ray &local_ray) {
    vec3 intersection_p;
    const auto face = FindNearestIntersectingFace(bvh, polymesh, local_ray, &intersection_p);
    if (!face) return {};

    VH closest_vertex{};
    float min_distance_sq = std::numeric_limits<float>::max();
    for (const auto vh : polymesh.fv_range(face)) {
        const auto diff = polymesh.GetPosition(vh) - intersection_p;
        if (float distance_sq = glm::dot(diff, diff); distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }

    return closest_vertex;
}

EH FindNearestEdge(const BVH &bvh, const PolyMesh &polymesh, const ray &local_ray) {
    vec3 intersection_p;
    const auto face = FindNearestIntersectingFace(bvh, polymesh, local_ray, &intersection_p);
    if (!face) return {};

    EH closest_edge;
    float min_distance_sq = std::numeric_limits<float>::max();
    for (auto heh : polymesh.fh_range(face)) {
        const auto eh = polymesh.GetEdge(heh);
        const auto heh0 = polymesh.GetHalfedge(eh, 0);
        const auto p1 = polymesh.GetPosition(polymesh.GetFromVertex(heh0));
        const auto p2 = polymesh.GetPosition(polymesh.GetToVertex(heh0));
        if (float distance_sq = SquaredDistanceToLineSegment(p1, p2, intersection_p);
            distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_edge = eh;
        }
    }

    return closest_edge;
}

} // namespace MeshIntersection
