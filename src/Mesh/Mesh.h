#pragma once

#include "Geometry/Geometry.h"
#include "VulkanBuffer.h"

struct VulkanContext;
struct Ray;

struct VkMeshBuffers {
    VkMeshBuffers(const VulkanContext &vc) : VC(vc) {}
    VkMeshBuffers(const VulkanContext &vc, std::vector<Vertex3D> &&vertices, std::vector<uint> &&indices)
        : VC(vc), Buffers(std::move(vertices), std::move(indices)) {
        CreateOrUpdateBuffers();
    }
    VkMeshBuffers(const VulkanContext &vc, MeshBuffers &&buffers)
        : VC(vc), Buffers(std::move(buffers)) {
        CreateOrUpdateBuffers();
    }
    ~VkMeshBuffers() = default;

    const std::vector<Vertex3D> &GetVertices() const { return Buffers.Vertices; }
    const std::vector<uint> &GetIndices() const { return Buffers.Indices; }

    inline void Set(MeshBuffers &&buffers) {
        Buffers = std::move(buffers);
        CreateOrUpdateBuffers();
    }

    void CreateOrUpdateBuffers();
    void Bind(vk::CommandBuffer cb) const;

    const VulkanContext &VC;
    MeshBuffers Buffers;
    VulkanBuffer VertexBuffer{vk::BufferUsageFlagBits::eVertexBuffer};
    VulkanBuffer IndexBuffer{vk::BufferUsageFlagBits::eIndexBuffer};
};

// A `Mesh` is a wrapper around a `Geometry`, a `Model` matrix, Vulkan buffers, and other application state.
struct Mesh {
    Mesh(const VulkanContext &, Geometry &&);

    void CreateOrUpdateBuffers(MeshElement);
    void CreateOrUpdateBuffers(); // Update all buffers.

    const VkMeshBuffers &GetBuffers(MeshElement element) const;
    const VkMeshBuffers *GetFaceNormalIndicatorBuffers() const;
    const VkMeshBuffers *GetVertexNormalIndicatorBuffers() const;

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

    void ShowNormalIndicators(NormalMode mode, bool show);

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
    std::unordered_map<MeshElement, std::unique_ptr<VkMeshBuffers>> ElementBuffers;
    std::unique_ptr<VkMeshBuffers> FaceNormalIndicatorBuffers, VertexNormalIndicatorBuffers;

    Geometry::FH HighlightedFace{};
    Geometry::VH HighlightedVertex{};
    Geometry::EH HighlightedEdge{};
};
