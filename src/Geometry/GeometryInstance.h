#pragma once

#include "VulkanBuffer.h"

#include "Geometry.h"

struct VulkanContext;
struct Ray;

struct GeometryInstance {
    // todo line mode buffers can share vertex buffers with smooth mode.
    struct Buffers {
        // Redundantly store the vertices and indices in the CPU for easy access.
        std::vector<Vertex3D> Vertices;
        std::vector<uint> Indices;
        VulkanBuffer VertexBuffer{vk::BufferUsageFlagBits::eVertexBuffer};
        VulkanBuffer IndexBuffer{vk::BufferUsageFlagBits::eIndexBuffer};
    };

    GeometryInstance(const VulkanContext &, Geometry &&);

    const Buffers &GetBuffers(GeometryMode mode) const { return BuffersForMode.at(mode); }

    // Returns a handle to the first face that intersects the ray, or -1 if no face intersects.
    // The ray is assumed to be in world space.
    Geometry::FH FindFirstIntersectingFace(const Ray &) const;

    void SetEdgeColor(const glm::vec4 &);

    const VulkanContext &VC;
    glm::mat4 Model{1};

private:
    Geometry G;
    std::unordered_map<GeometryMode, Buffers> BuffersForMode;

    void CreateOrUpdateBuffers(GeometryMode);
};
