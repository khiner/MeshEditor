#pragma once

enum class RenderMode {
    None,
    FacesAndEdges,
    Faces,
    Edges,
    Smooth, // Uses vertices geometry mode.
    Silhouette, // Uses the silhouette render pipeline.
};

enum class ColorMode {
    Mesh,
    Normals,
};
