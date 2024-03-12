#pragma once

#include <vector>

#include "Vertex.h"

using uint = unsigned int;

// Faces: Vertices are duplicated for each face. Each vertex uses the face normal.
// Vertices: Vertices are not duplicated. Uses vertext normals.
// Edge: Vertices are duplicated. Each vertex uses the vertex normal.
struct MeshBuffers {
    std::vector<Vertex3D> Vertices{};
    std::vector<uint> Indices{};
};
