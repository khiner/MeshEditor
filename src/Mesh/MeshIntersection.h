#pragma once

#include "Intersection.h"
#include "halfedge/PolyMesh.h"

#include <optional>

struct BVH;
struct ray;

namespace MeshIntersection {
std::optional<Intersection> Intersect(const BVH &bvh, const he::PolyMesh &polymesh, const ray &local_ray);

he::FH FindNearestIntersectingFace(const BVH &bvh, const he::PolyMesh &polymesh, const ray &local_ray, vec3 *nearest_intersect_point_out = nullptr);
// Returns a handle to the vertex nearest to the intersection point on the first intersecting face, or an invalid handle if no face intersects.
he::VH FindNearestVertex(const BVH &bvh, const he::PolyMesh &polymesh, const ray &local_ray);
// Returns a handle to the edge nearest to the intersection point on the first intersecting face, or an invalid handle if no face intersects.
he::EH FindNearestEdge(const BVH &bvh, const he::PolyMesh &polymesh, const ray &local_ray);
} // namespace MeshIntersection
