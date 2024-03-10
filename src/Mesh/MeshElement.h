#pragma once

enum class MeshElement {
    None,
    Face, // Vertices are duplicated for each face. Each vertex uses the face normal.
    Vertex, // Vertices are not duplicated. Each vertex uses the vertex normal.
    Edge, // Vertices are duplicated. Each vertex uses the vertex normal.
};
