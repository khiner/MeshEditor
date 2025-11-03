#pragma once

#include "BBox.h"
#include "HalfEdge.h"
#include "Intersection.h"
#include "MeshElement.h"
#include "Vertex.h"

#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <filesystem>
#include <unordered_set>

namespace fs = std::filesystem;

struct BVH;
struct ray;

std::optional<he::PolyMesh> LoadPolyMesh(const fs::path &);

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

    // Adds half-edge handle comparison/conversion to `MeshElementIndex`.
    struct ElementIndex : MeshElementIndex {
        using MeshElementIndex::MeshElementIndex;
        ElementIndex(MeshElementIndex other) : MeshElementIndex(std::move(other)) {}
        ElementIndex(VH vh) : MeshElementIndex(MeshElement::Vertex, *vh) {}
        ElementIndex(EH eh) : MeshElementIndex(MeshElement::Edge, *eh) {}
        ElementIndex(FH fh) : MeshElementIndex(MeshElement::Face, *fh) {}

        bool operator==(ElementIndex other) const { return Element == other.Element && Index == other.Index; }

        bool operator==(VH vh) const { return Element == MeshElement::Vertex && Index == *vh; }
        bool operator==(EH eh) const { return Element == MeshElement::Edge && Index == *eh; }
        bool operator==(FH fh) const { return Element == MeshElement::Face && Index == *fh; }

        // Implicit conversion to half-edge handles.
        operator VH() const { return VH{Element == MeshElement::Vertex ? Index : -1}; }
        operator EH() const { return EH{Element == MeshElement::Edge ? Index : -1}; }
        operator FH() const { return FH{Element == MeshElement::Face ? Index : -1}; }
    };

    inline static constexpr vec4 DefaultFaceColor{0.7, 0.7, 0.7, 1};
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
    Mesh(std::vector<vec3> &&vertices, std::vector<std::vector<uint>> &&faces, vec4 color = DefaultFaceColor);
    ~Mesh();

    const Mesh &operator=(Mesh &&);

    bool operator==(const Mesh &other) const { return &M == &other.M; }

    uint GetVertexCount() const { return M.VertexCount(); }
    uint GetEdgeCount() const { return M.EdgeCount(); }
    uint GetFaceCount() const { return M.FaceCount(); }
    bool Empty() const { return GetVertexCount() == 0; }

    vec3 GetPosition(VH vh) const { return M.GetPosition(vh); }
    vec3 GetVertexNormal(VH vh) const { return M.GetNormal(vh); }
    const float *GetPositionData() const { return M.GetPositionData(); }

    vec3 GetFaceCenter(FH fh) const { return M.CalcFaceCentroid(fh); }
    vec3 GetFaceNormal(FH fh) const { return M.GetNormal(fh); }

    float CalcFaceArea(FH) const;

    std::vector<Vertex3D> CreateVertices(MeshElement, const ElementIndex &select = {}) const;
    std::vector<uint> CreateIndices(MeshElement) const;

    std::vector<Vertex3D> CreateNormalVertices(MeshElement) const;
    std::vector<uint> CreateNormalIndices(MeshElement) const;

    BBox ComputeBbox() const; // This is public, but use `BoundingBox` instead.

    std::vector<BBox> CreateFaceBoundingBoxes() const;
    RenderBuffers CreateBvhBuffers(vec4 color) const;

    void HighlightVertex(VH vh) { HighlightedElements.emplace(MeshElement::Vertex, *vh); }
    void ClearHighlights() { HighlightedElements.clear(); }

    void SetFaceColor(FH fh, vec4 color) { M.SetColor(fh, color); }
    void SetFaceColor(vec4 color) {
        for (const auto &fh : M.faces()) SetFaceColor(fh, color);
    }

    void SetFaces(const std::vector<vec3> &vertices, const std::vector<std::vector<uint>> &faces, vec4 color = DefaultFaceColor) {
        for (const auto &v : vertices) M.AddVertex(v);
        std::vector<std::vector<VH>> face_vhs_list;
        face_vhs_list.reserve(faces.size());
        for (const auto &face : faces) {
            std::vector<VH> face_vhs;
            face_vhs.reserve(face.size());
            for (auto vi : face) face_vhs.emplace_back(vi);
            face_vhs_list.emplace_back(std::move(face_vhs));
        }
        M.SetFaces(face_vhs_list);
        if (color != DefaultFaceColor) {
            for (const auto &fh : M.faces()) SetFaceColor(fh, color);
        }
    }

    bool VertexBelongsToFace(VH, FH) const;
    bool VertexBelongsToEdge(VH, EH) const;
    bool VertexBelongsToFaceEdge(VH, FH, EH) const;
    bool EdgeBelongsToFace(EH, FH) const;

    std::optional<Intersection> Intersect(const ray &local_ray) const;

    VH FindNearestVertex(vec3) const;
    // Returns a handle to the vertex nearest to the intersection point on the first intersecting face, or an invalid handle if no face intersects.
    VH FindNearestVertex(const ray &local_ray) const;

    // Returns a handle to the edge nearest to the intersection point on the first intersecting face, or an invalid handle if no face intersects.
    EH FindNearestEdge(const ray &world_ray) const;
    FH FindNearestIntersectingFace(const ray &local_ray, vec3 *nearest_intersect_point_out = nullptr) const;

    std::vector<uint> CreateTriangleIndices() const; // Triangulated face indices.
    std::vector<uint> CreateTriangulatedFaceIndices() const; // Triangle fan for each face.
    std::vector<uint> CreateEdgeIndices() const;

private:
    PolyMesh M;
    std::unique_ptr<BVH> Bvh;
    // In addition to selected elements.
    std::unordered_set<ElementIndex, MeshElementIndexHash> HighlightedElements;
};
