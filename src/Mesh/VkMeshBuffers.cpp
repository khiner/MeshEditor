#include "VkMeshBuffers.h"

#include "MeshBuffers.h"
#include "vulkan/VulkanContext.h"

VkMeshBuffers::VkMeshBuffers(const VulkanContext &vc) : VC(vc) {}
VkMeshBuffers::VkMeshBuffers(const VulkanContext &vc, MeshBuffers &&buffers) : VC(vc) { Set(std::move(buffers)); }
VkMeshBuffers::~VkMeshBuffers() = default;

void VkMeshBuffers::Set(MeshBuffers &&buffers) {
    VertexBuffer.Size = sizeof(Vertex3D) * buffers.Vertices.size();
    IndexBuffer.Size = sizeof(uint) * buffers.Indices.size();
    VC.CreateOrUpdateBuffer(VertexBuffer, buffers.Vertices.data());
    VC.CreateOrUpdateBuffer(IndexBuffer, buffers.Indices.data());
}

void VkMeshBuffers::Bind(vk::CommandBuffer cb) const {
    static const vk::DeviceSize vertex_buffer_offsets[] = {0};
    cb.bindVertexBuffers(0, *VertexBuffer.Buffer, vertex_buffer_offsets);
    cb.bindIndexBuffer(*IndexBuffer.Buffer, 0, vk::IndexType::eUint32);
}

void VkMeshBuffers::Draw(vk::CommandBuffer cb, uint instance_count) const {
    cb.drawIndexed(IndexBuffer.Size / sizeof(uint), instance_count, 0, 0, 0);
}
