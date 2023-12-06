#pragma once

#include "Geometry.h"
#include "VulkanBuffer.h"

struct VulkanContext;
struct Ray;

// todo edge mode buffers can share vertex buffers with vertices mode.
struct GeometryBuffers {
    std::vector<Vertex3D> Vertices{};
    std::vector<uint> Indices{};
    VulkanBuffer VertexBuffer{vk::BufferUsageFlagBits::eVertexBuffer};
    VulkanBuffer IndexBuffer{vk::BufferUsageFlagBits::eIndexBuffer};
};

struct GeometryInstance {
    GeometryInstance(const VulkanContext &, Geometry &&);

    void CreateOrUpdateBuffers(GeometryMode);
    void CreateOrUpdateBuffers(); // Update all buffers.

    const GeometryBuffers &GetBuffers(GeometryMode mode) const { return BuffersForMode.at(mode); }
    const GeometryBuffers *GetFaceNormalIndicatorBuffers() const { return FaceNormalIndicatorBuffers.get(); }
    const GeometryBuffers *GetVertexNormalIndicatorBuffers() const { return VertexNormalIndicatorBuffers.get(); }

    // Returns a handle to the first face that intersects the world-space ray, or -1 if no face intersects.
    // If `closest_intersect_point_out` is not null, sets it to the intersection point.
    Geometry::FH FindFirstIntersectingFace(const Ray &ray_world, glm::vec3 *closest_intersect_point_out = nullptr) const;
    Geometry::FH FindFirstIntersectingFaceLocal(const Ray &ray_local, glm::vec3 *closest_intersect_point_out = nullptr) const; // Local space equivalent.

    // Returns a handle to the vertex nearest to the intersecction point on the first intersecting face, or an invalid handle if no face intersects.
    Geometry::VH FindNearestVertex(const Ray &ray_world) const;
    Geometry::VH FindNearestVertexLocal(const Ray &ray_local) const; // Local space equivalent.

    // Returns a handle to the edge nearest to the intersection point on the first intersecting face, or an invalid handle if no face intersects.
    Geometry::EH FindNearestEdge(const Ray &ray_world) const;
    Geometry::EH FindNearestEdgeLocal(const Ray &ray_local) const; // Local space equivalent.

    void ShowNormalIndicators(NormalIndicatorMode mode, bool show);

    bool HighlightFace(Geometry::FH face) {
        if (face == HighlightedFace) return false;

        HighlightedFace = face;
        HighlightedVertex = Geometry::VH{};
        HighlightedEdge = Geometry::EH{};
        CreateOrUpdateBuffers();
        return true;
    }
    bool HighlightVertex(Geometry::VH vertex) {
        if (vertex == HighlightedVertex) return false;

        HighlightedVertex = vertex;
        HighlightedFace = Geometry::FH{};
        HighlightedEdge = Geometry::EH{};
        CreateOrUpdateBuffers();
        return true;
    }
    bool HighlightEdge(Geometry::EH edge) {
        if (edge == HighlightedEdge) return false;

        HighlightedEdge = edge;
        HighlightedFace = Geometry::FH{};
        HighlightedVertex = Geometry::VH{};
        CreateOrUpdateBuffers();
        return true;
    }

    std::string GetHighlightLabel() const;

    const VulkanContext &VC;
    glm::mat4 Model{1};

private:
    Geometry G;
    std::unordered_map<GeometryMode, GeometryBuffers> BuffersForMode;
    std::unique_ptr<GeometryBuffers> FaceNormalIndicatorBuffers, VertexNormalIndicatorBuffers;

    Geometry::FH HighlightedFace{};
    Geometry::VH HighlightedVertex{};
    Geometry::EH HighlightedEdge{};
};
