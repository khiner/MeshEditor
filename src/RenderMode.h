#pragma once

enum class GeometryMode {
    None,
    Faces, // Vertices are duplicated for each face. Each vertex uses the face normal.
    Vertices, // Vertices are not duplicated. Each vertex uses the vertex normal.
    Edges, // Vertices are not duplicated. Each vertex uses the vertex normal.
};

enum class RenderMode {
    None,
    Flat, // Uses faces geometry mode.
    Smooth, // Uses vertices geometry mode.
    Lines, // Uses edges geometry mode.
    Mesh, // Uses faces + edges geometry mode.
};
