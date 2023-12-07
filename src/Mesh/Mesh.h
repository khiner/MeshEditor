#pragma once

#include <filesystem>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "MeshBuffers.h"
#include "MeshElement.h"
#include "RenderMode.h"

namespace fs = std::filesystem;

struct Ray;

inline static glm::vec3 ToGlm(const OpenMesh::Vec3f &v) { return {v[0], v[1], v[2]}; }
inline static glm::vec4 ToGlm(const OpenMesh::Vec3uc &c) {
    const auto cc = OpenMesh::color_cast<OpenMesh::Vec3f>(c);
    return {cc[0], cc[1], cc[2], 1};
}
inline static OpenMesh::Vec3uc ToOpenMesh(const glm::vec4 &c) {
    const auto cc = OpenMesh::color_cast<OpenMesh::Vec3uc>(OpenMesh::Vec3f{c.r, c.g, c.b});
    return {cc[0], cc[1], cc[2]};
}

// Aliases for OpenMesh types to support `using namespace om;`.
namespace om {
using PolyMesh = OpenMesh::PolyMesh_ArrayKernelT<>;
using VH = OpenMesh::VertexHandle;
using FH = OpenMesh::FaceHandle;
using EH = OpenMesh::EdgeHandle;
using HH = OpenMesh::HalfedgeHandle;
using Point = OpenMesh::Vec3f;
}; // namespace om

// A `Mesh` is a wrapper around an `OpenMesh::PolyMesh`, privately available as `M`.
struct Mesh {
    using PolyMesh = om::PolyMesh;
    using VH = om::VH;
    using FH = om::FH;
    using EH = om::EH;
    using HH = om::HH;
    using Point = om::Point;

    inline static const glm::vec4 DefaultFaceColor = {0.7, 0.7, 0.7, 1};
    inline static glm::vec4 EdgeColor{0, 0, 0, 1};
    inline static glm::vec4 HighlightColor{0.929, 0.341, 0, 1}; // Blender's default `Preferences->Themes->3D Viewport->Object Selected`.
    inline static glm::vec4 FaceNormalIndicatorColor{0.133, 0.867, 0.867, 1}; // Blender's default `Preferences->Themes->3D Viewport->Face Normal`.
    inline static glm::vec4 VertexNormalIndicatorColor{0.137, 0.380, 0.867, 1}; // Blender's default `Preferences->Themes->3D Viewport->Vertex Normal`.
    inline static float NormalIndicatorLengthScale = 0.25f;

    Mesh() {
        M.request_face_normals();
        M.request_vertex_normals();
        M.request_face_colors();
    }
    Mesh(Mesh &&mesh) : M(std::move(mesh.M)) {}
    Mesh(const fs::path &file_path) {
        M.request_face_normals();
        M.request_vertex_normals();
        Load(file_path);
    }

    ~Mesh() {
        M.release_vertex_normals();
        M.release_face_normals();
    }

    bool Load(const fs::path &file_path);
    void Save(const fs::path &file_path) const;

    inline uint NumPositions() const { return M.n_vertices(); }
    inline uint NumFaces() const { return M.n_faces(); }

    inline const PolyMesh &GetMesh() const { return M; }

    inline const float *GetPositionData() const { return (const float *)M.points(); }

    inline glm::vec3 GetPosition(VH vh) const { return ToGlm(M.point(vh)); }
    inline glm::vec3 GetVertexNormal(VH vh) const { return ToGlm(M.normal(vh)); }
    inline glm::vec3 GetFaceNormal(FH fh) const { return ToGlm(M.normal(fh)); }
    inline glm::vec3 GetFaceCenter(FH fh) const { return ToGlm(M.calc_face_centroid(fh)); }

    inline bool Empty() const { return M.n_vertices() == 0; }

    inline void UpdateNormals() { M.update_normals(); }

    inline MeshBuffers GenerateBuffers(MeshElement element, FH highlighted_face = FH{}, VH highlighted_vertex = VH{}, EH highlighted_edge = EH{}) const {
        return {GenerateVertices(element, highlighted_face, highlighted_vertex, highlighted_edge), GenerateIndices(element)};
    }
    inline MeshBuffers GenerateBuffers(NormalMode mode) const {
        return {GenerateVertices(mode), GenerateIndices(mode)};
    }

    FH TriangulatedIndexToFace(uint triangle_index) const; // Convert index generated with `GenerateTriangulatedFaceIndices()` to a face handle.

    // [{min_x, min_y, min_z}, {max_x, max_y, max_z}]
    std::pair<glm::vec3, glm::vec3> ComputeBounds() const;

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

    void SetFaceColor(FH fh, const glm::vec4 &face_color) {
        M.set_color(fh, ToOpenMesh(face_color));
    }
    void SetFaceColor(const glm::vec4 &face_color) {
        for (const auto &fh : M.faces()) SetFaceColor(fh, face_color);
    }

    void AddFace(const std::vector<VH> &vertices, const glm::vec4 &color = DefaultFaceColor) {
        SetFaceColor(M.add_face(vertices), color);
    }

    bool DoesVertexBelongToFace(VH vertex, FH face) const;
    bool DoesVertexBelongToEdge(VH vertex, EH edge) const;
    bool DoesVertexBelongToFaceEdge(VH vertex, FH face, EH edge) const;
    bool DoesEdgeBelongToFace(EH edge, FH face) const;

    // Returns a handle to the first face that intersects the world-space ray, or -1 if no face intersects.
    // If `closest_intersect_point_out` is not null, sets it to the intersection point.
    // Provide triangulated face buffers, to avoid re-triangulating the mesh.
    VH FindNearestVertex(const MeshBuffers &tri_buffers, const Ray &ray_local) const;
    // If `closest_intersect_point_out` is not null, sets it to the intersection point.
    FH FindFirstIntersectingFace(const MeshBuffers &tri_buffers, const Ray &ray_local, glm::vec3 *closest_intersect_point_out = nullptr) const;
    // Returns a handle to the edge nearest to the intersection point on the first intersecting face, or an invalid handle if no face intersects.
    EH FindNearestEdge(const MeshBuffers &tri_buffers, const Ray &ray_world) const;

protected:
    PolyMesh M;

    std::vector<Vertex3D> GenerateVertices(MeshElement element, FH highlighted_face = FH{}, VH highlighted_vertex = VH{}, EH highlighted_edge = EH{}) const;
    std::vector<uint> GenerateIndices(MeshElement element) const;
    std::vector<Vertex3D> GenerateVertices(NormalMode mode) const;
    std::vector<uint> GenerateIndices(NormalMode mode) const;

    std::vector<uint> GenerateTriangleIndices() const; // Triangulated face indices.
    std::vector<uint> GenerateTriangulatedFaceIndices() const; // Triangle fan for each face.
    std::vector<uint> GenerateEdgeIndices() const;
    std::vector<uint> GenerateFaceNormalIndicatorIndices() const;
    std::vector<uint> GenerateVertexNormalIndicatorIndices() const;
};
