#pragma once

#include "numeric/mat4.h"

#include "mesh/Mesh.h"
#include "mesh/VkMeshBuffers.h"

// An `Object` is a wrapper around a `Mesh`, a `Model` transform, Vulkan buffers, and other application state.
struct Object {
    Object(const VulkanContext &, Mesh &&);

    void CreateOrUpdateBuffers(MeshElement);
    void CreateOrUpdateBuffers(); // Update all buffers.

    const VkMeshBuffers &GetBuffers(MeshElement element) const;
    const VkMeshBuffers *GetFaceNormalIndicatorBuffers() const;
    const VkMeshBuffers *GetVertexNormalIndicatorBuffers() const;

    Mesh::FH FindFirstIntersectingFace(const Ray &world_ray, vec3 *closest_intersect_point_out = nullptr) const;
    Mesh::VH FindNearestVertex(const Ray &world_ray) const;
    Mesh::EH FindNearestEdge(const Ray &world_ray) const;

    void ShowNormalIndicators(NormalMode, bool show);

    bool HighlightFace(Mesh::FH face) {
        if (face == HighlightedFace) return false;

        HighlightedFace = face;
        HighlightedVertex = Mesh::VH{};
        HighlightedEdge = Mesh::EH{};
        CreateOrUpdateBuffers();
        return true;
    }
    bool HighlightVertex(Mesh::VH vertex) {
        if (vertex == HighlightedVertex) return false;

        HighlightedFace = Mesh::FH{};
        HighlightedVertex = vertex;
        HighlightedEdge = Mesh::EH{};
        CreateOrUpdateBuffers();
        return true;
    }
    bool HighlightEdge(Mesh::EH edge) {
        if (edge == HighlightedEdge) return false;

        HighlightedFace = Mesh::FH{};
        HighlightedVertex = Mesh::VH{};
        HighlightedEdge = edge;
        CreateOrUpdateBuffers();
        return true;
    }

    std::string GetHighlightLabel() const;

    const VulkanContext &VC;
    mat4 Model{1};

private:
    Mesh M;
    std::unordered_map<MeshElement, std::unique_ptr<VkMeshBuffers>> ElementBuffers;
    std::unique_ptr<VkMeshBuffers> FaceNormalIndicatorBuffers, VertexNormalIndicatorBuffers;

    Mesh::FH HighlightedFace{};
    Mesh::VH HighlightedVertex{};
    Mesh::EH HighlightedEdge{};
};
