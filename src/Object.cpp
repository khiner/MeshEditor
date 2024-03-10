#include "Object.h"

#include "Ray.h"
#include "vulkan/VulkanContext.h"

#include <format>

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
    cb.drawIndexed(IndexBuffer.Size / sizeof(uint), 1, 0, 0, 0);
}

Object::Object(const VulkanContext &vc, Mesh &&mesh)
    : VC(vc), M(std::move(mesh)) {
    CreateOrUpdateBuffers();
}

void Object::CreateOrUpdateBuffers() {
    static const std::vector AllElements{MeshElement::Faces, MeshElement::Vertices, MeshElement::Edges};
    for (const auto element : AllElements) CreateOrUpdateBuffers(element);
}

void Object::CreateOrUpdateBuffers(MeshElement element) {
    if (!ElementBuffers.contains(element)) ElementBuffers[element] = std::make_unique<VkMeshBuffers>(VC);
    M.UpdateNormals(); // todo only update when necessary.
    ElementBuffers[element]->Set(M.GenerateBuffers(element, HighlightedFace, HighlightedVertex, HighlightedEdge));
}

const VkMeshBuffers &Object::GetBuffers(MeshElement mode) const { return *ElementBuffers.at(mode); }
const VkMeshBuffers *Object::GetFaceNormalIndicatorBuffers() const { return FaceNormalIndicatorBuffers.get(); }
const VkMeshBuffers *Object::GetVertexNormalIndicatorBuffers() const { return VertexNormalIndicatorBuffers.get(); }

void Object::ShowNormalIndicators(NormalMode mode, bool show) {
    auto &buffers = mode == NormalMode::Faces ? FaceNormalIndicatorBuffers : VertexNormalIndicatorBuffers;
    buffers.reset();
    if (!show) return;

    // Don't need to update normals to render lines.
    buffers = std::make_unique<VkMeshBuffers>(VC, M.GenerateBuffers(mode));
}

Mesh::FH Object::FindFirstIntersectingFace(const Ray &ray_world, vec3 *closest_intersect_point_out) const {
    return M.FindFirstIntersectingFace(ray_world.WorldToLocal(Model), closest_intersect_point_out);
}

Mesh::VH Object::FindNearestVertex(const Ray &ray_world) const {
    return M.FindNearestVertex(ray_world.WorldToLocal(Model));
}

Mesh::EH Object::FindNearestEdge(const Ray &ray_world) const { return M.FindNearestEdge(ray_world.WorldToLocal(Model)); }

std::string Object::GetHighlightLabel() const {
    if (HighlightedFace.is_valid()) return std::format("Hovered face {}", HighlightedFace.idx());
    if (HighlightedVertex.is_valid()) return std::format("Hovered vertex {}", HighlightedVertex.idx());
    if (HighlightedEdge.is_valid()) return std::format("Hovered edge {}", HighlightedEdge.idx());
    return "Hovered: None";
}
