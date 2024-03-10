#include "VkMeshBuffers.h"

#include "vulkan/VulkanContext.h"

void VkMeshBuffers::CreateOrUpdateBuffers() {
    VertexBuffer.Size = sizeof(Vertex3D) * GetVertices().size();
    IndexBuffer.Size = sizeof(uint) * GetIndices().size();
    VC.CreateOrUpdateBuffer(VertexBuffer, GetVertices().data());
    VC.CreateOrUpdateBuffer(IndexBuffer, GetIndices().data());
}

void VkMeshBuffers::Bind(vk::CommandBuffer cb) const {
    static const vk::DeviceSize vertex_buffer_offsets[] = {0};
    cb.bindVertexBuffers(0, *VertexBuffer.Buffer, vertex_buffer_offsets);
    cb.bindIndexBuffer(*IndexBuffer.Buffer, 0, vk::IndexType::eUint32);
}

void VkMeshBuffers::Draw(vk::CommandBuffer cb, uint instance_count) const {
    cb.drawIndexed(IndexBuffer.Size / sizeof(uint), instance_count, 0, 0, 0);
}
