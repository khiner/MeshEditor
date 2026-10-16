#pragma once

#include "gpu/Types.h"

// Share topology among instances using render-owner handles.
// IDs reference triangles, edges, or vertices; vertex references name corners for triangles and vertices otherwise.
// Procedural triangle soups use absolute index slots.
// Primitive, GroupIndex, and RefinedGroup reference render sets.
// Payload offsets are relative to owner ranges; InvalidOffset denotes absence.
// Coarse clusters omit source elements; lines and points omit local triangles.
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
