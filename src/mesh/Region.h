#pragma once
#include "gpu/Region.h"
#include "mesh/MeshData.h"
#include "numeric/VectorMath.h"
#include <functional>
#include <optional>

namespace mtl {
struct ComputeChain;
}
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
// A fill of one or two operands' segments projected onto a plane.
struct RegionTask {
    RegionPlane Plane;
    std::vector<RegionOperand> Operands;
    RegionFill Fill;
};
// Each task's loops bounding its fill, each in plane coordinates with the fill on its left, or none when a face operand's edge leaves the plane.
// Every task's passes of one stage record in one submit.
std::vector<std::optional<std::vector<std::vector<vec2>>>> RegionLoops(state::Registry &, std::span<const RegionTask>);
// Records the projection of the record's silhouette onto the plane with the other pending lanes' projections at the same step, every fill in one RegionLoops and every result in one creation once the chain submits.
// Calls `then` with the new record, or none when the record projects to nothing.
void RecordProjection(state::Registry &, mtl::ComputeChain &, uint32_t record, const RegionPlane &, std::function<void(std::optional<uint32_t>)> then);
// The loops as faces on the plane, without points within the tie tolerance of the point before them and without loops left with fewer than three.
MeshData RegionData(const RegionPlane &, std::span<const std::vector<vec2>> loops);
