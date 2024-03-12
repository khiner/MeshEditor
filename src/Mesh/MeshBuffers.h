#pragma once

#include "vulkan/VulkanBuffer.h"

// Faces: Vertices are duplicated for each face. Each vertex uses the face normal.
// Vertices: Vertices are not duplicated. Uses vertex normals.
// Edge: Vertices are duplicated. Each vertex uses the vertex normal.
struct MeshBuffers {
    VulkanBuffer VertexBuffer{vk::BufferUsageFlagBits::eVertexBuffer};
    VulkanBuffer IndexBuffer{vk::BufferUsageFlagBits::eIndexBuffer};
};
