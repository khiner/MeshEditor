#pragma once

#include <filesystem>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include "MeshBuffers.h"
#include "MeshElement.h"
#include "NormalMode.h"
#include "RenderMode.h"

namespace fs = std::filesystem;

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

inline static vec3 ToGlm(const OpenMesh::Vec3f &v) { return {v[0], v[1], v[2]}; }
inline static vec4 ToGlm(const OpenMesh::Vec3uc &c) {
    const auto cc = OpenMesh::color_cast<OpenMesh::Vec3f>(c);
    return {cc[0], cc[1], cc[2], 1};
}
inline static om::Point ToOpenMesh(vec3 v) { return {v.x, v.y, v.z}; }
inline static OpenMesh::Vec3uc ToOpenMesh(vec4 c) {
    const auto cc = OpenMesh::color_cast<OpenMesh::Vec3uc>(OpenMesh::Vec3f{c.r, c.g, c.b});
    return {cc[0], cc[1], cc[2]};
}

// A `Mesh` is a wrapper around an `OpenMesh::PolyMesh`, privately available as `M`.
struct Mesh {
    using PolyMesh = om::PolyMesh;
    using VH = om::VH;
    using FH = om::FH;
    using EH = om::EH;
    using HH = om::HH;
    using Point = om::Point;

    inline static const vec4 DefaultFaceColor = {0.7, 0.7, 0.7, 1};
    inline static vec4 EdgeColor{0, 0, 0, 1};
    inline static vec4 HighlightColor{0.929, 0.341, 0, 1}; // Blender's default `Preferences->Themes->3D Viewport->Object Selected`.
    inline static vec4 FaceNormalIndicatorColor{0.133, 0.867, 0.867, 1}; // Blender's default `Preferences->Themes->3D Viewport->Face Normal`.
    inline static vec4 VertexNormalIndicatorColor{0.137, 0.380, 0.867, 1}; // Blender's default `Preferences->Themes->3D Viewport->Vertex Normal`.
    inline static float NormalIndicatorLengthScale = 0.25f;

    Mesh() {
        M.request_face_normals();
        M.request_vertex_normals();
        M.request_face_colors();
    }
    Mesh(Mesh &&mesh) : M(std::move(mesh.M)) {
        M.request_face_normals();
        M.request_vertex_normals();
        M.request_face_colors();
    }
    Mesh(const fs::path &file_path) {
        M.request_face_normals();
        M.request_vertex_normals();
        M.request_face_colors();
        Load(file_path, M);
    }

    ~Mesh() {
        M.release_vertex_normals();
        M.release_face_normals();
        M.release_face_colors();
    }

    static bool Load(const fs::path &file_path, PolyMesh &out_mesh);

    inline uint NumPositions() const { return M.n_vertices(); }
    inline uint NumFaces() const { return M.n_faces(); }

    inline const PolyMesh &GetMesh() const { return M; }

    inline const float *GetPositionData() const { return (const float *)M.points(); }

    inline vec3 GetPosition(VH vh) const { return ToGlm(M.point(vh)); }
    inline vec3 GetVertexNormal(VH vh) const { return ToGlm(M.normal(vh)); }
    inline vec3 GetFaceNormal(FH fh) const { return ToGlm(M.normal(fh)); }
    inline vec3 GetFaceCenter(FH fh) const { return ToGlm(M.calc_face_centroid(fh)); }

    float CalcFaceArea(FH) const;

    inline bool Empty() const { return M.n_vertices() == 0; }

    inline void UpdateNormals() { M.update_normals(); }

    inline MeshBuffers GenerateBuffers(MeshElement element, FH highlighted_face = FH{}, VH highlighted_vertex = VH{}, EH highlighted_edge = EH{}) const {
        return {GenerateVertices(element, highlighted_face, highlighted_vertex, highlighted_edge), GenerateIndices(element)};
    }
    inline MeshBuffers GenerateBuffers(NormalMode mode) const { return {GenerateVertices(mode), GenerateIndices(mode)}; }

    // [{min_x, min_y, min_z}, {max_x, max_y, max_z}]
    std::pair<vec3, vec3> ComputeBounds() const;

    // Centers the actual points to the center of gravity, not just a transform.
    // void Center() {
    //     auto points = OpenMesh::getPointsProperty(Mesh);
    //     auto cog = Mesh.vertices().avg(points);
    //     for (const auto &vh : Mesh.vertices()) {
    //         const auto &point = Mesh.point(vh);
    //         Mesh.set_point(vh, point - cog);
    //     }
    // }

    void SetMesh(const PolyMesh &mesh) {
        M = mesh;
        M.request_face_normals();
        M.request_vertex_normals();
    }

    void SetFaceColor(FH fh, vec4 face_color) { M.set_color(fh, ToOpenMesh(face_color)); }
    void SetFaceColor(vec4 face_color) {
        for (const auto &fh : M.faces()) SetFaceColor(fh, face_color);
    }

    void AddFace(const std::vector<VH> &vertices, vec4 color = DefaultFaceColor) {
        SetFaceColor(M.add_face(vertices), color);
    }

    bool DoesVertexBelongToFace(VH, FH) const;
    bool DoesVertexBelongToEdge(VH, EH) const;
    bool DoesVertexBelongToFaceEdge(VH, FH, EH) const;
    bool DoesEdgeBelongToFace(EH, FH) const;

    // Returns true if the ray intersects the given triangle.
    // If ray intersects, sets `distance_out` to the distance along the ray to the intersection point, and sets `intersect_point_out`, if not null.
    bool RayIntersectsTriangle(const Ray &, VH v1, VH v2, VH v3, float *distance_out, vec3 *intersect_point_out = nullptr) const;

    // Returns a handle to the first face that intersects the world-space ray, or -1 if no face intersects.
    // If `closest_intersect_point_out` is not null, sets it to the intersection point.
    VH FindNearestVertex(const Ray &local_ray) const;
    // If `closest_intersect_point_out` is not null, sets it to the intersection point.
    FH FindFirstIntersectingFace(const Ray &local_ray, vec3 *closest_intersect_point_out = nullptr) const;
    // Returns a handle to the edge nearest to the intersection point on the first intersecting face, or an invalid handle if no face intersects.
    EH FindNearestEdge(const Ray &world_ray) const;

protected:
    PolyMesh M;

    std::vector<Vertex3D> GenerateVertices(MeshElement, FH highlighted_face = FH{}, VH highlighted_vertex = VH{}, EH highlighted_edge = EH{}) const;
    std::vector<uint> GenerateIndices(MeshElement) const;
    std::vector<Vertex3D> GenerateVertices(NormalMode) const;
    std::vector<uint> GenerateIndices(NormalMode) const;

    std::vector<uint> GenerateTriangleIndices() const; // Triangulated face indices.
    std::vector<uint> GenerateTriangulatedFaceIndices() const; // Triangle fan for each face.
    std::vector<uint> GenerateEdgeIndices() const;
    std::vector<uint> GenerateFaceNormalIndicatorIndices() const;
    std::vector<uint> GenerateVertexNormalIndicatorIndices() const;
};
