#pragma once
#include "gpu/AttributeField.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace state {
struct Registry;
} // namespace state
namespace mtl {
struct ComputeChain;
} // namespace mtl
struct MeshStore;
struct MeshPipelines;

// Records the reduction of the source attribute's values onto each live element of the output attribute's domain, which holds the source's type.
void RecordDomainConversion(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t source, uint32_t output, DomainReduction);

// Gaussian curvature is the vertex angle defect and mean curvature the length-weighted signed edge dihedral angles, each over a third of the incident triangle area.
enum class CurvatureKind : uint8_t { Gaussian,
                                     Mean };
// Each pass below records into the attribute `output`, over the live elements of its record's domain.
// Per-vertex curvature of the rendered triangulation into a Float vertex attribute, with open edges contributing zero.
void RecordCurvature(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t output, CurvatureKind);
// The edges with fewer than two faces, the vertices on them, or the faces touching them, into a Bool attribute.
void RecordBoundary(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t output);
// The Bool seed of the output's domain grown by `steps` rings of adjacent elements, or shrunk by as many when `more` is false.
void RecordAdjacent(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t seed, uint32_t output, bool more, uint32_t steps);
// Records the components under the adjacency that hold an element of the Bool seed, in the output's domain, then calls `then`.
// Labels propagate in rounds until none changes, a count the mesh decides, so each round reads its change once the chain's next submit completes it, and the output records after the last round.
void RecordComponents(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t seed, uint32_t output, FieldAdjacency, std::function<void()> then);
// The vertices on the side of the plane through `point` with normal `normal`, or the edges or faces whose vertices all are, in a new Bool attribute named `name`, or none for a zero normal or a taken name.
std::optional<uint32_t> PlaneSideField(state::Registry &, uint32_t id, AttributeDomain, vec3 point, vec3 normal, PlaneSideKind, float tolerance, std::string_view name);
// Records the Plane Side field of a nonzero normal into the Bool attribute `output`, with the other pending lanes' fields at the same step.
void RecordPlaneSide(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t output, vec3 point, vec3 normal, PlaneSideKind, float tolerance);
// Per listed element, the lowest listed handle its component under the adjacency reaches through listed elements only.
// With parity, each label is that handle shifted left with the parity of the element's shortest path to it in the low bit.
std::vector<uint32_t> ComponentLabels(const MeshStore &, const MeshPipelines &, uint32_t id, Element domain, std::span<const uint32_t> handles, FieldAdjacency, bool parity = false);
// Records the passes of ComponentLabels in the record's lane of the chain and calls `then` with the labels once the chain completed the last round that changed none.
void RecordComponentLabels(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, Element domain, std::span<const uint32_t> handles, FieldAdjacency, bool parity, std::function<void(std::vector<uint32_t>)> then);
