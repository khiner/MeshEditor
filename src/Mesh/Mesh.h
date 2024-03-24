#pragma once

#include <filesystem>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include "BBox.h"
#include "MeshElement.h"
#include "RenderMode.h"
#include "Vertex.h"

namespace fs = std::filesystem;

struct BVH;
struct Ray;

// Aliases for OpenMesh types to support `using namespace om;`.
namespace om {
using PolyMesh = OpenMesh::PolyMesh_ArrayKernelT<>;
using VH = OpenMesh::VertexHandle;
using FH = OpenMesh::FaceHandle;
using EH = OpenMesh::EdgeHandle;
using HH = OpenMesh::HalfedgeHandle;
using Point = OpenMesh::Vec3f;
}; // namespace om

inline vec3 ToGlm(const OpenMesh::Vec3f &v) { return {v[0], v[1], v[2]}; }
inline vec4 ToGlm(const OpenMesh::Vec3uc &c) {
    const auto cc = OpenMesh::color_cast<OpenMesh::Vec3f>(c);
    return {cc[0], cc[1], cc[2], 1};
}
inline om::Point ToOpenMesh(vec3 v) { return {v.x, v.y, v.z}; }
inline OpenMesh::Vec3uc ToOpenMesh(vec4 c) {
    const auto cc = OpenMesh::color_cast<OpenMesh::Vec3uc>(OpenMesh::Vec3f{c.r, c.g, c.b});
    return {cc[0], cc[1], cc[2]};
}

struct RenderBuffers {
    std::vector<Vertex3D> Vertices;
    std::vector<uint> Indices;
};

// A `Mesh` is a wrapper around an `OpenMesh::PolyMesh`, privately available as `M`.
// Render modes:
// - Faces: Vertices are duplicated for each face. Each vertex uses the face normal.
// - Vertices: Vertices are not duplicated. Uses vertex normals.
// - Edge: Vertices are duplicated. Each vertex uses the vertex normal.
struct Mesh {
    BBox BoundingBox;

    using PolyMesh = om::PolyMesh;
    using VH = om::VH;
    using FH = om::FH;
    using EH = om::EH;
    using HH = om::HH;
    using Point = om::Point;

    // Adds OpenMesh handle comparison/conversion to `MeshElementIndex`.
    struct ElementIndex : MeshElementIndex {
        using MeshElementIndex::MeshElementIndex;
        ElementIndex(const MeshElementIndex &other) : MeshElementIndex(other) {}

        bool operator==(VH vh) const { return Element == MeshElement::Vertex && Index == vh.idx(); }
        bool operator==(EH eh) const { return Element == MeshElement::Edge && Index == eh.idx(); }
        bool operator==(FH fh) const { return Element == MeshElement::Face && Index == fh.idx(); }

        // Implicit conversion to OpenMesh handles.
        operator VH() const { return Element == MeshElement::Vertex ? VH(Index) : VH(-1); }
        operator EH() const { return Element == MeshElement::Edge ? EH(Index) : EH(-1); }
        operator FH() const { return Element == MeshElement::Face ? FH(Index) : FH(-1); }
    };

    inline static const vec4 DefaultFaceColor = {0.7, 0.7, 0.7, 1};
    inline static vec4 EdgeColor{0, 0, 0, 1};
    inline static vec4 HighlightColor{0.929, 0.341, 0, 1}; // Blender's default `Preferences->Themes->3D Viewport->Object Selected`.
    inline static vec4 FaceNormalIndicatorColor{0.133, 0.867, 0.867, 1}; // Blender's default `Preferences->Themes->3D Viewport->Face Normal`.
    inline static vec4 VertexNormalIndicatorColor{0.137, 0.380, 0.867, 1}; // Blender's default `Preferences->Themes->3D Viewport->Vertex Normal`.
    inline static float NormalIndicatorLengthScale = 0.25;

    Mesh(Mesh &&);
    Mesh(const fs::path &);
    Mesh(std::vector<vec3> &&vertices, std::vector<std::vector<uint>> &&faces, vec4 color = DefaultFaceColor);
    ~Mesh();

    Mesh &operator=(const Mesh &other) {
        if (this != &other) M = other.M;
        return *this;
    }

    bool operator==(const Mesh &other) const { return &M == &other.M; }

    static bool Load(const fs::path &file_path, PolyMesh &out_mesh);

    const float *GetPositionData() const { return (const float *)M.points(); }

    vec3 GetPosition(VH vh) const { return ToGlm(M.point(vh)); }
    vec3 GetVertexNormal(VH vh) const { return ToGlm(M.normal(vh)); }

    vec3 GetFaceCenter(FH fh) const { return ToGlm(M.calc_face_centroid(fh)); }
    vec3 GetFaceNormal(FH fh) const { return ToGlm(M.normal(fh)); }

    float CalcFaceArea(FH) const;

    bool Empty() const { return M.n_vertices() == 0; }

    std::vector<Vertex3D> CreateVertices(MeshElement, ElementIndex highlight = {}) const;
    std::vector<uint> CreateIndices(MeshElement) const;

    std::vector<Vertex3D> CreateNormalVertices(MeshElement) const;
    std::vector<uint> CreateNormalIndices(MeshElement) const;

    BBox ComputeBbox() const; // This is public, but use `BoundingBox` instead.

    std::vector<BBox> CreateFaceBoundingBoxes() const;
    RenderBuffers CreateBvhBuffers(vec4 color) const;

    // Center of gravity
    // void Center() {
    //     auto points = OpenMesh::getPointsProperty(Mesh);
    //     auto cog = Mesh.vertices().avg(points);
    //     for (const auto &vh : Mesh.vertices()) {
    //         const auto &point = Mesh.point(vh);
    //         Mesh.set_point(vh, point - cog);
    //     }
    // }

    void SetFaceColor(FH fh, vec4 face_color) { M.set_color(fh, ToOpenMesh(face_color)); }
    void SetFaceColor(vec4 face_color) {
        for (const auto &fh : M.faces()) SetFaceColor(fh, face_color);
    }

    void AddFace(const std::vector<VH> &vertices, vec4 color = DefaultFaceColor) { SetFaceColor(M.add_face(vertices), color); }

    void SetFaces(const std::vector<vec3> &vertices, const std::vector<std::vector<uint>> &faces, vec4 color = DefaultFaceColor) {
        for (const auto &v : vertices) M.add_vertex(ToOpenMesh(v));
        for (const auto &face : faces) {
            std::vector<VH> face_vhs;
            face_vhs.reserve(face.size());
            for (const auto &vi : face) face_vhs.push_back(VH(vi));
            AddFace(std::move(face_vhs), color);
        }
    }

    bool VertexBelongsToFace(VH, FH) const;
    bool VertexBelongsToEdge(VH, EH) const;
    bool VertexBelongsToFaceEdge(VH, FH, EH) const;
    bool EdgeBelongsToFace(EH, FH) const;

    bool RayIntersectsFace(const Ray &, FH, float *distance_out = nullptr, vec3 *intersect_point_out = nullptr) const;

    bool RayIntersects(const Ray &local_ray) const; // Intersects any face.

    // Returns a handle to the vertex nearest to the intersection point on the first intersecting face, or an invalid handle if no face intersects.
    VH FindNearestVertex(const Ray &local_ray) const;
    // Returns a handle to the edge nearest to the intersection point on the first intersecting face, or an invalid handle if no face intersects.
    EH FindNearestEdge(const Ray &world_ray) const;
    // Returns a handle to the first face that intersects the world-space ray, or -1 if no face intersects.
    // If `nearest_intersect_point_out` is not null, sets it to the intersection point.
    FH FindNearestIntersectingFace(const Ray &local_ray, vec3 *nearest_intersect_point_out = nullptr) const;

private:
    PolyMesh M;
    std::unique_ptr<BVH> Bvh;

    std::vector<uint> CreateTriangleIndices() const; // Triangulated face indices.
    std::vector<uint> CreateTriangulatedFaceIndices() const; // Triangle fan for each face.
    std::vector<uint> CreateEdgeIndices() const;
};
