#pragma once
#include "gpu/Region.h"
#include "mesh/MeshData.h"
#include "numeric/VectorMath.h"
#include <optional>

namespace state {
struct Registry;
}
struct Mesh;

// A plane with the right-handed basis of its coordinates.
struct RegionPlane {
    vec3 Point, Normal, U, V;
    vec3 Lift(vec2 p) const { return Point + U * p.x + V * p.y; }
};
// The plane through the point with the normal, which must not be zero.
RegionPlane PlaneAt(vec3 point, vec3 normal);
std::optional<RegionPlane> FacePlane(state::Registry &, const Mesh &);

// An operand contributes its face loop edges, or with Silhouette the halfedges whose face faces the plane normal and whose opposite does not.
struct RegionOperand {
    uint32_t Record;
    mat3 Rotation{1.f};
    vec3 Translation{};
    bool Silhouette{};
};
// The loops bounding the fill of the operands' segments projected onto the plane, each in plane coordinates with the fill on its left.
// Returns none when a face operand's edge leaves the plane.
std::optional<std::vector<std::vector<vec2>>> RegionLoops(state::Registry &, const RegionPlane &, std::span<const RegionOperand>, RegionFill);
// The loops as faces on the plane, without points within the tie tolerance of the point before them and without loops left with fewer than three.
MeshData RegionData(const RegionPlane &, std::span<const std::vector<vec2>> loops);
