#pragma once

#include <filesystem>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Vertex.h"

using uint = unsigned int;

enum class RenderMode {
    Flat,
    Smooth,
};

namespace fs = std::filesystem;

inline static glm::vec3 ToGlm(const OpenMesh::Vec3f &v) { return {v[0], v[1], v[2]}; }

struct Geometry {
    using MeshType = OpenMesh::PolyMesh_ArrayKernelT<>;
    using VH = OpenMesh::VertexHandle;
    using FH = OpenMesh::FaceHandle;
    using EH = OpenMesh::EdgeHandle;
    using HH = OpenMesh::HalfedgeHandle;
    using Point = OpenMesh::Vec3f;

    Geometry() {
        Mesh.request_face_normals();
        Mesh.request_vertex_normals();
    }
    Geometry(Geometry &&geometry) : Mesh(std::move(geometry.Mesh)) {
        UpdateBuffersFromMesh();
    }
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

    inline uint NumVertices() const { return Vertices.size(); }
    inline uint NumPositions() const { return Mesh.n_vertices(); }
    inline uint NumFaces() const { return Mesh.n_faces(); }
    inline uint NumIndices() const { return Indices.size(); }

    inline const MeshType &GetMesh() const { return Mesh; }

    inline const Vertex3D *GetVertexData() const { return Vertices.data(); }
    inline const float *GetPositionData() const { return (const float *)Mesh.points(); }
    inline const uint *GetIndexData() const { return Indices.data(); }

    inline const std::vector<Vertex3D> &GetVertices() const { return Vertices; }
    inline glm::vec3 GetPosition(uint index) const { return ToGlm(Mesh.point(VH(index))); }
    inline glm::vec3 GetVertexNormal(uint index) const { return ToGlm(Mesh.normal(VH(index))); }
    inline glm::vec3 GetFaceNormal(uint index) const { return ToGlm(Mesh.normal(FH(index))); }
    inline glm::vec3 GetFaceCenter(uint index) const { return ToGlm(Mesh.calc_face_centroid(FH(index))); }

    uint FindVertextNearestTo(const glm::vec3 point) const;
    inline bool Empty() const { return Vertices.empty(); }

    std::vector<uint> GenerateTriangleIndices() const;
    std::vector<uint> GenerateTriangulatedFaceIndices() const;

    // [{min_x, min_y, min_z}, {max_x, max_y, max_z}]
    std::pair<glm::vec3, glm::vec3> ComputeBounds() const;

    // Centers the actual points to the center of gravity, not just a transform.
    void Center() {
        auto points = OpenMesh::getPointsProperty(Mesh);
        auto cog = Mesh.vertices().avg(points);
        for (const auto &vh : Mesh.vertices()) {
            const auto &point = Mesh.point(vh);
            Mesh.set_point(vh, point - cog);
        }
        UpdateVertices(); // todo Normals/indices are not affected.
    }

    void SetMesh(const MeshType &mesh) {
        Clear();
        Mesh = mesh;
        Mesh.request_face_normals();
        Mesh.request_vertex_normals();
        UpdateBuffersFromMesh();
    }

    void Clear() {
        Mesh.clear();
        Vertices.clear();
        Dirty = true;
    }

protected:
    MeshType Mesh;
    RenderMode ActiveRenderMode{RenderMode::Flat};
    mutable bool Dirty{true};

    void UpdateBuffersFromMesh() {
        UpdateVertices();
        UpdateIndices();
    }

    void UpdateVertices() {
        Vertices.clear();

        Mesh.update_normals();
        glm::vec4 color = {1, 1, 1, 1}; // todo
        if (ActiveRenderMode == RenderMode::Flat) {
            for (const auto &fh : Mesh.faces()) {
                const auto &n = Mesh.normal(fh); // Duplicate the normal for each vertex.
                for (const auto &vh : Mesh.fv_range(fh)) {
                    Vertices.emplace_back(ToGlm(Mesh.point(vh)), ToGlm(n), color);
                }
            }
        } else {
            for (const auto &vh : Mesh.vertices()) {
                Vertices.emplace_back(ToGlm(Mesh.point(vh)), ToGlm(Mesh.normal(vh)), color);
            }
        }

        Dirty = true;
    }

    void UpdateIndices() {
        Indices =
            ActiveRenderMode == RenderMode::Flat ? GenerateTriangulatedFaceIndices() :
                                                   GenerateTriangleIndices();
        Dirty = true;
    }

    std::vector<Vertex3D> Vertices;
    std::vector<uint> Indices;
};