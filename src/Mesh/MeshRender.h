#pragma once

#include "BBox.h"
#include "BVH.h"
#include "Vertex.h"
#include "halfedge/PolyMesh.h"
#include "numeric/vec4.h"

#include <unordered_set>
#include <vector>

struct RenderBuffers {
    std::vector<Vertex3D> Vertices;
    std::vector<uint> Indices;
};

namespace MeshRender {

// Rendering colors
inline vec4 VertexColor{1}, EdgeColor{0, 0, 0, 1};
inline vec4 SelectedColor{1, 0.478, 0, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Select
inline vec4 HighlightedColor{0, 0.647, 1, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Bevel
inline vec4 FaceNormalIndicatorColor{0.133, 0.867, 0.867, 1}; // Blender: Preferences->Themes->3D Viewport->Face Normal
inline vec4 VertexNormalIndicatorColor{0.137, 0.380, 0.867, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Normal
inline vec4 HighlightedFaceColor{0.790, 0.930, 1, 1}; // Custom
constexpr float NormalIndicatorLengthScale{0.25};

// Vertex: Triangulated face indices
// Face: Triangle fan for each face
// Edge: Edge line segment indices
std::vector<Vertex3D> CreateVertices(
    const he::PolyMesh &polymesh,
    he::Element render_element,
    const he::AnyHandle &selected = {},
    const std::unordered_set<he::AnyHandle, he::AnyHandleHash> &highlighted = {}
);
std::vector<uint> CreateIndices(const he::PolyMesh &polymesh, he::Element element);

std::vector<Vertex3D> CreateNormalVertices(const he::PolyMesh &polymesh, he::Element element);
std::vector<uint> CreateNormalIndices(const he::PolyMesh &polymesh, he::Element element);

std::vector<BBox> CreateFaceBoundingBoxes(const he::PolyMesh &polymesh);
RenderBuffers CreateBvhBuffers(const BVH &bvh, vec4 color);

BBox ComputeBoundingBox(const he::PolyMesh &polymesh);

} // namespace MeshRender
