#pragma once
#include "gpu/AttributeField.h"
#include "gpu/NamedAttributeRef.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace state {
struct Registry;
} // namespace state
struct MeshStore;
struct MeshPipelines;

// Each field writes a new attribute named `name`, or returns none when the name is taken.
// Gaussian curvature is the vertex angle defect and mean curvature the length-weighted signed edge dihedral angles, each over a third of the incident triangle area.
enum class CurvatureKind : uint8_t { Gaussian,
                                     Mean };
// Per-vertex curvature of the rendered triangulation as a Float attribute, with open edges contributing zero.
std::optional<uint32_t> CurvatureField(state::Registry &, uint32_t id, CurvatureKind, std::string_view name);
// The selection fields below write a Bool attribute in their domain.
// The edges with fewer than two faces, the vertices on them, or the faces touching them.
std::optional<uint32_t> BoundaryField(state::Registry &, uint32_t id, AttributeDomain, std::string_view name);
// The vertices on the side of the plane through `point` with normal `normal`, or the edges or faces whose vertices all are, or none for a zero normal.
std::optional<uint32_t> PlaneSideField(state::Registry &, uint32_t id, AttributeDomain, vec3 point, vec3 normal, PlaneSideKind, float tolerance, std::string_view name);
// The connected components holding the seed's elements in its domain.
std::optional<uint32_t> LinkedField(state::Registry &, uint32_t id, uint32_t seed, std::string_view name);
// Per listed element, the lowest listed handle its component under the adjacency reaches through listed elements only.
// With parity, each label is that handle shifted left with the parity of the element's path to it in the low bit, each step flipping it.
std::vector<uint32_t> ComponentLabels(const MeshStore &, const MeshPipelines &, uint32_t id, Element domain, std::span<const uint32_t> handles, FieldAdjacency, bool parity = false);
// The seed grown by `steps` rings of adjacent elements, or shrunk by as many when `more` is false.
std::optional<uint32_t> AdjacentField(state::Registry &, uint32_t id, uint32_t seed, bool more, uint32_t steps, std::string_view name);
// The edge loops through the seed edges, continuing at each vertex with four edges to the edge sharing no face with the one arrived on.
std::optional<uint32_t> EdgeLoopField(state::Registry &, uint32_t id, uint32_t seed, std::string_view name);
// The edge rings through the seed edges, continuing across each quad to the edge opposite the one arrived on.
std::optional<uint32_t> EdgeRingField(state::Registry &, uint32_t id, uint32_t seed, std::string_view name);
