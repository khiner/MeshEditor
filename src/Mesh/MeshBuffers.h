#pragma once

#include <vector>

#include "Vertex.h"

using uint = unsigned int;

// Faces mesh buffers: Vertices are duplicated for each face. Each vertex uses the face normal.
// Vertices mesh buffers: Vertices are not duplicated. Uses vertext normals.
// Edge mesh buffers: Vertices are duplicated. Each vertex uses the vertex normal.
struct MeshBuffers {
    MeshBuffers() = default;
    MeshBuffers(std::vector<Vertex3D> &&vertices, std::vector<uint> &&indices) : Vertices(vertices), Indices(indices) {}
    virtual ~MeshBuffers() = default;

    std::vector<Vertex3D> Vertices{};
    std::vector<uint> Indices{};
};
