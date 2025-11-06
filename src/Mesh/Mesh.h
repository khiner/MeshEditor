#pragma once

#include "BBox.h"
#include "Intersection.h"
#include "Vertex.h"
#include "halfedge/PolyMesh.h"

#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <filesystem>
#include <unordered_set>

namespace fs = std::filesystem;

struct BVH;
struct ray;

struct RenderBuffers {
    std::vector<Vertex3D> Vertices;
    std::vector<uint> Indices;
};

// A `Mesh` is a wrapper around a `he::PolyMesh`, privately available as `M`.
// Render modes:
// - Faces: Vertices are duplicated for each face. Each vertex uses the face normal.
// - Vertices: Vertices are not duplicated. Uses vertex normals.
// - Edge: Vertices are duplicated. Each vertex uses the vertex normal.
struct Mesh {
    BBox BoundingBox;

    using PolyMesh = he::PolyMesh;
    using VH = he::VH;
    using FH = he::FH;
    using EH = he::EH;
    using HH = he::HH;
    using AnyHandle = he::AnyHandle;
    using AnyHandleHash = he::AnyHandleHash;

    inline static vec4 VertexColor{1}, EdgeColor{0, 0, 0, 1};
    inline static vec4 SelectedColor{1, 0.478, 0, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Select
    inline static vec4 HighlightedColor{0, 0.647, 1, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Bevel
    inline static vec4 FaceNormalIndicatorColor{0.133, 0.867, 0.867, 1}; // Blender: Preferences->Themes->3D Viewport->Face Normal
    inline static vec4 VertexNormalIndicatorColor{0.137, 0.380, 0.867, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Normal
    inline static vec4 HighlightedFaceColor{0.790, 0.930, 1, 1}; // Custom
    static constexpr float NormalIndicatorLengthScale{0.25};

    Mesh(Mesh &&);
    Mesh(const Mesh &);
    Mesh(PolyMesh &&);
    ~Mesh();

    const Mesh &operator=(Mesh &&);

    const PolyMesh &GetPolyMesh() const { return M; }
    PolyMesh &GetPolyMesh() { return M; }

    std::vector<Vertex3D> CreateVertices(he::Element, const he::AnyHandle &select = {}) const;
    std::vector<uint> CreateIndices(he::Element) const;

    std::vector<Vertex3D> CreateNormalVertices(he::Element) const;
    std::vector<uint> CreateNormalIndices(he::Element) const;

    std::vector<BBox> CreateFaceBoundingBoxes() const;
    RenderBuffers CreateBvhBuffers(vec4 color) const;

    void HighlightVertex(VH vh) { HighlightedHandles.emplace(vh); }
    void ClearHighlights() { HighlightedHandles.clear(); }

    std::optional<Intersection> Intersect(const ray &local_ray) const;

    // Returns a handle to the vertex nearest to the intersection point on the first intersecting face, or an invalid handle if no face intersects.
    VH FindNearestVertex(const ray &local_ray) const;

    // Returns a handle to the edge nearest to the intersection point on the first intersecting face, or an invalid handle if no face intersects.
    EH FindNearestEdge(const ray &world_ray) const;
    FH FindNearestIntersectingFace(const ray &local_ray, vec3 *nearest_intersect_point_out = nullptr) const;

private:
    BBox ComputeBbox() const;

    PolyMesh M;
    std::unique_ptr<BVH> Bvh;
    // In addition to selected elements.
    std::unordered_set<AnyHandle, AnyHandleHash> HighlightedHandles;
};
