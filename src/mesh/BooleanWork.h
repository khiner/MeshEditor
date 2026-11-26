#pragma once

#include "gpu/SlotOffset.h"

#include <span>
#include <vector>

struct MeshClosure;
struct MeshTopologyTask;
namespace mtl {
struct ComputeChain;
} // namespace mtl
namespace state {
struct Registry;
} // namespace state

// Boolean work in chain scratch, retained through topology counting and emission.
struct BooleanWork {
    SlotOffset Job; // The BooleanJob record
    uint32_t Vertices, Intersections, Triangles; // The count entries of each kind, in that order
    uint32_t OutputVertices, OutputTriangles;
};
struct BooleanSources {
    const MeshTopologyTask &Task;
    const MeshClosure &First, &Second;
};
// The scratch words of job tables a source pair adds to its batch, four per table entry: a record location, a first tile and at most one table count.
// They are its entries in its 18 pass tables, its two colliders' entries in two tables, and its three scans' records of three words and entries in three tables each.
constexpr uint32_t BooleanTableWords{4u * 18u + 2u * 2u * 4u + 3u * (3u + 3u * 4u)};
// Constructs each source pair's boolean work, with each stage of every pair in one submit.
std::vector<BooleanWork> RecordBooleanWork(state::Registry &, mtl::ComputeChain &, std::span<const BooleanSources>);
