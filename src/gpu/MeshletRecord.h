#pragma once

#include "gpu/Types.h"

// Stores immutable mesh topology shared by all instances.
// Its IDs are the render owner's triangle, edge or vertex handles by topology, and its vertex references the owner's corner handles, or vertex handles for lines and points.
// Procedural triangle soups use absolute index slots.
// Primitive, GroupIndex and RefinedGroup are handles of the owner's render sets.
// TriangleOffset, VertexOffset and LocalTriangleOffset are its runs' offsets in the owner's payload ranges, and InvalidOffset for a run it lacks.
// A coarse cluster has no source elements, and a line or point cluster no local triangles.
struct MeshletRecord {
    uint32_t TriangleOffset DEFAULT();
    uint32_t TriangleCount DEFAULT();
    uint32_t VertexOffset DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t LocalTriangleOffset DEFAULT();
    uint32_t Primitive DEFAULT();
    // A cluster renders when its group exceeds the error threshold and its refining group does not.
    // RefinedGroup is InvalidOffset for original geometry.
    uint32_t GroupIndex DEFAULT(InvalidOffset);
    uint32_t RefinedGroup DEFAULT(InvalidOffset);
    // Meshopt cone culling is exact only when every triangle uses its derived flat face normal.
    // Mixed-normal meshlets use cutoff 127 to disable cone culling.
    uint32_t ConeAxisCutoff DEFAULT();
    vec3 Center DEFAULT();
    float Radius DEFAULT();
    uint32_t Topology DEFAULT();
};
static_assert(sizeof(MeshletRecord) == 56, "MeshletRecord size");
