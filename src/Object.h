#pragma once

#include "numeric/mat4.h"

#include "mesh/Mesh.h"
#include "mesh/VkMeshBuffers.h"

// An `Object` is a wrapper around a `Mesh`, a `Model` transform, Vulkan buffers, and other application state.
struct Object {
    Object(const VulkanContext &, Mesh &&);

    const VulkanContext &VC;
    mat4 Model{1};

    void CreateOrUpdateBuffers(MeshElement);
    void CreateOrUpdateBuffers(); // Update all buffers.

    const VkMeshBuffers &GetBuffers(MeshElement element) const;
    const VkMeshBuffers *GetFaceNormalIndicatorBuffers() const;
    const VkMeshBuffers *GetVertexNormalIndicatorBuffers() const;

    Mesh::FH FindFirstIntersectingFace(const Ray &world_ray, vec3 *closest_intersect_point_out = nullptr) const;
    Mesh::VH FindNearestVertex(const Ray &world_ray) const;
    Mesh::EH FindNearestEdge(const Ray &world_ray) const;

    void ShowNormalIndicators(NormalMode, bool show);

    bool HighlightFace(Mesh::FH fh) {
        if (fh == HighlightedFace) return false;

        HighlightedFace = fh;
        HighlightedVertex = Mesh::VH{};
        HighlightedEdge = Mesh::EH{};
        CreateOrUpdateBuffers();
        return true;
    }
    bool HighlightVertex(Mesh::VH vh) {
        if (vh == HighlightedVertex) return false;

        HighlightedFace = Mesh::FH{};
        HighlightedVertex = vh;
        HighlightedEdge = Mesh::EH{};
        CreateOrUpdateBuffers();
        return true;
    }
    bool HighlightEdge(Mesh::EH eh) {
        if (eh == HighlightedEdge) return false;

        HighlightedFace = Mesh::FH{};
        HighlightedVertex = Mesh::VH{};
        HighlightedEdge = eh;
        CreateOrUpdateBuffers();
        return true;
    }

    std::string GetHighlightLabel() const;

private:
    Mesh M;
    std::unordered_map<MeshElement, std::unique_ptr<VkMeshBuffers>> ElementBuffers;
    std::unique_ptr<VkMeshBuffers> FaceNormalIndicatorBuffers, VertexNormalIndicatorBuffers;

    Mesh::FH HighlightedFace{};
    Mesh::VH HighlightedVertex{};
    Mesh::EH HighlightedEdge{};
};
