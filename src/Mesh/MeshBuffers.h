#pragma once

#include "vulkan/VulkanBuffer.h"

// Faces: Vertices are duplicated for each face. Each vertex uses the face normal.
// Vertices: Vertices are not duplicated. Uses vertex normals.
// Edge: Vertices are duplicated. Each vertex uses the vertex normal.
struct MeshBuffers {
    VulkanBuffer Vertices, Indices;

    MeshBuffers(vk::DeviceSize vertex_bytes, vk::DeviceSize index_bytes) {
        Vertices = VulkanBuffer{vk::BufferUsageFlagBits::eVertexBuffer, vertex_bytes};
        Indices = VulkanBuffer{vk::BufferUsageFlagBits::eIndexBuffer, index_bytes};
    }
};
