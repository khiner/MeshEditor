#pragma once

enum class GeometryMode {
    None,
    Faces, // Vertices are duplicated for each face. Each vertex uses the face normal.
    Vertices, // Vertices are not duplicated. Each vertex uses the vertex normal.
    Edges, // Vertices are not duplicated. Each vertex uses the vertex normal.
};
