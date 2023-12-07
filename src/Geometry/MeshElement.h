#pragma once

enum class MeshElement {
    None,
    Faces, // Vertices are duplicated for each face. Each vertex uses the face normal.
    Vertices, // Vertices are not duplicated. Each vertex uses the vertex normal.
    Edges, // Vertices are duplicated. Each vertex uses the vertex normal.
};

enum class NormalMode {
    None,
    Faces, // Line for each face normal.
    Vertices, // Line for each vertex normal.
};
