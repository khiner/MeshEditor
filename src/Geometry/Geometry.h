#pragma once

#include <filesystem>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "MeshElement.h"
#include "RenderMode.h"
#include "Vertex.h"

using uint = unsigned int;

namespace fs = std::filesystem;

inline static glm::vec3 ToGlm(const OpenMesh::Vec3f &v) { return {v[0], v[1], v[2]}; }
inline static glm::vec4 ToGlm(const OpenMesh::Vec3uc &c) {
    const auto cc = OpenMesh::color_cast<OpenMesh::Vec3f>(c);
    return {cc[0], cc[1], cc[2], 1};
}
inline static OpenMesh::Vec3uc ToOpenMesh(const glm::vec4 &c) {
    const auto cc = OpenMesh::color_cast<OpenMesh::Vec3uc>(OpenMesh::Vec3f{c.r, c.g, c.b});
    return {cc[0], cc[1], cc[2]};
}

// Aliases for OpenMesh types to avoid collision with our own `Mesh` type.
namespace om {
using Mesh = OpenMesh::PolyMesh_ArrayKernelT<>;
using VH = OpenMesh::VertexHandle;
using FH = OpenMesh::FaceHandle;
using EH = OpenMesh::EdgeHandle;
using HH = OpenMesh::HalfedgeHandle;
using Point = OpenMesh::Vec3f;
}; // namespace om

// Faces mesh buffers: Vertices are duplicated for each face. Each vertex uses the face normal.
// Vertices mesh buffers: Vertices are not duplicated. Uses vertext normals.
// Edge mesh buffers: Vertices are duplicated. Each vertex uses the vertex normal.
struct MeshBuffers {
    MeshBuffers() = default;
    MeshBuffers(std::vector<Vertex3D> &&vertices, std::vector<uint> &&indices) : Vertices(vertices), Indices(indices) {}
    virtual ~MeshBuffers() = default;

    std::vector<Vertex3D> Vertices{};
    std::vector<uint> Indices{};
};

// A `Geometry` is a wrapper around an `OpenMesh::PolyMesh`, privately available as `M`.
struct Geometry {
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

    Geometry() {
        M.request_face_normals();
        M.request_vertex_normals();
        M.request_face_colors();
    }
    Geometry(Geometry &&geometry) : M(std::move(geometry.M)) {}
    Geometry(const fs::path &file_path) {
        M.request_face_normals();
        M.request_vertex_normals();
        Load(file_path);
    }

    ~Geometry() {
        M.release_vertex_normals();
        M.release_face_normals();
    }

    bool Load(const fs::path &file_path);
    void Save(const fs::path &file_path) const;

    inline uint NumPositions() const { return M.n_vertices(); }
    inline uint NumFaces() const { return M.n_faces(); }

    inline const om::Mesh &GetMesh() const { return M; }

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

    void SetMesh(const om::Mesh &mesh) {
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

protected:
    om::Mesh M;

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
