#pragma once

#include <filesystem>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "GeometryMode.h"
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

using MeshType = OpenMesh::PolyMesh_ArrayKernelT<>;

struct Geometry {
    using VH = OpenMesh::VertexHandle;
    using FH = OpenMesh::FaceHandle;
    using EH = OpenMesh::EdgeHandle;
    using HH = OpenMesh::HalfedgeHandle;
    using Point = OpenMesh::Vec3f;

    inline static const glm::vec4 DefaultFaceColor = {0.7, 0.7, 0.7, 1};
    inline static glm::vec4 EdgeColor{0, 0, 0, 1};
    inline static glm::vec4 HighlightColor{0.929, 0.341, 0, 1}; // Blender's default `Preferences->Themes->3D Viewport->Object Selected`.
    inline static glm::vec4 FaceNormalIndicatorColor{0.133, 0.867, 0.867, 1}; // Blender's default `Preferences->Themes->3D Viewport->Face Normal`.
    inline static glm::vec4 VertexNormalIndicatorColor{0.137, 0.380, 0.867, 1}; // Blender's default `Preferences->Themes->3D Viewport->Vertex Normal`.
    inline static float NormalIndicatorLengthScale = 0.25f;

    Geometry() {
        Mesh.request_face_normals();
        Mesh.request_vertex_normals();
        Mesh.request_face_colors();
    }
    Geometry(Geometry &&geometry) : Mesh(std::move(geometry.Mesh)) {}
    Geometry(const fs::path &file_path) {
        Mesh.request_face_normals();
        Mesh.request_vertex_normals();
        Load(file_path);
    }

    ~Geometry() {
        Mesh.release_vertex_normals();
        Mesh.release_face_normals();
    }

    bool Load(const fs::path &file_path);
    void Save(const fs::path &file_path) const;

    inline uint NumPositions() const { return Mesh.n_vertices(); }
    inline uint NumFaces() const { return Mesh.n_faces(); }

    inline const MeshType &GetMesh() const { return Mesh; }

    inline const float *GetPositionData() const { return (const float *)Mesh.points(); }

    inline glm::vec3 GetPosition(VH vh) const { return ToGlm(Mesh.point(vh)); }
    inline glm::vec3 GetVertexNormal(VH vh) const { return ToGlm(Mesh.normal(vh)); }
    inline glm::vec3 GetFaceNormal(FH fh) const { return ToGlm(Mesh.normal(fh)); }
    inline glm::vec3 GetFaceCenter(FH fh) const { return ToGlm(Mesh.calc_face_centroid(fh)); }

    inline bool Empty() const { return Mesh.n_vertices() == 0; }

    std::vector<Vertex3D> GenerateVertices(GeometryMode mode, FH highlighted_face = FH{}, VH highlighted_vertex = VH{}, EH highlighted_edge = EH{});
    std::vector<uint> GenerateIndices(GeometryMode mode) const {
        switch (mode) {
            case GeometryMode::Faces: return GenerateTriangulatedFaceIndices();
            case GeometryMode::Edges: return GenerateEdgeIndices();
            case GeometryMode::Vertices: return GenerateTriangleIndices();
            default: return {};
        }
    }
    std::vector<Vertex3D> GenerateVertices(NormalIndicatorMode mode);
    std::vector<uint> GenerateIndices(NormalIndicatorMode mode) const {
        switch (mode) {
            case NormalIndicatorMode::Faces: return GenerateFaceNormalIndicatorIndices();
            case NormalIndicatorMode::Vertices: return GenerateVertexNormalIndicatorIndices();
            default: return {};
        }
    }

    std::vector<uint> GenerateTriangleIndices() const; // Face indices after calling `Mesh.triangulate()`.
    std::vector<uint> GenerateTriangulatedFaceIndices() const; // Triangle fan for each face.
    std::vector<uint> GenerateEdgeIndices() const;
    std::vector<uint> GenerateFaceNormalIndicatorIndices() const;
    std::vector<uint> GenerateVertexNormalIndicatorIndices() const;

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

    void SetMesh(const MeshType &mesh) {
        Mesh = mesh;
        Mesh.request_face_normals();
        Mesh.request_vertex_normals();
    }

    void SetFaceColor(FH fh, const glm::vec4 &face_color) {
        Mesh.set_color(fh, ToOpenMesh(face_color));
    }
    void SetFaceColor(const glm::vec4 &face_color) {
        for (const auto &fh : Mesh.faces()) SetFaceColor(fh, face_color);
    }

    void AddFace(const std::vector<VH> &vertices, const glm::vec4 &color = DefaultFaceColor) {
        SetFaceColor(Mesh.add_face(vertices), color);
    }

    bool DoesVertexBelongToFace(VH vertex, FH face) const;
    bool DoesVertexBelongToEdge(VH vertex, EH edge) const;
    bool DoesVertexBelongToFaceEdge(VH vertex, FH face, EH edge) const;
    bool DoesEdgeBelongToFace(EH edge, FH face) const;

protected:
    MeshType Mesh;
};
