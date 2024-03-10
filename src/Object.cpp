#include "Object.h"

#include <format>

#include "Ray.h"
#include "Registry.h"
#include "vulkan/VulkanContext.h"

Object::Object(const VulkanContext &vc, const Registry &r, Mesh &&mesh, uint instance)
    : VC(vc), R(r), M(std::move(mesh)), Instance(instance) {
    CreateOrUpdateBuffers();
}

void Object::CreateOrUpdateBuffers() {
    static const std::vector AllElements{MeshElement::Face, MeshElement::Vertex, MeshElement::Edge};
    for (const auto element : AllElements) CreateOrUpdateBuffers(element);
}

const mat4 &Object::GetModel() const { return R.Models[Instance]; }

void Object::CreateOrUpdateBuffers(MeshElement element) {
    if (!ElementBuffers.contains(element)) ElementBuffers[element] = std::make_unique<VkMeshBuffers>(VC);
    M.UpdateNormals(); // todo only update when necessary.
    ElementBuffers[element]->Set(M.GenerateBuffers(element, HighlightedFace, HighlightedVertex, HighlightedEdge));
}

const VkMeshBuffers &Object::GetBuffers(MeshElement element) const { return *ElementBuffers.at(element); }
const VkMeshBuffers *Object::GetFaceNormalIndicatorBuffers() const { return FaceNormalIndicatorBuffers.get(); }
const VkMeshBuffers *Object::GetVertexNormalIndicatorBuffers() const { return VertexNormalIndicatorBuffers.get(); }

void Object::ShowNormalIndicators(NormalMode mode, bool show) {
    auto &buffers = mode == NormalMode::Face ? FaceNormalIndicatorBuffers : VertexNormalIndicatorBuffers;
    buffers.reset();
    if (!show) return;

    // Don't need to update normals to render lines.
    buffers = std::make_unique<VkMeshBuffers>(VC, M.GenerateBuffers(mode));
}

Mesh::FH Object::FindFirstIntersectingFace(const Ray &world_ray, vec3 *closest_intersect_point_out) const {
    return M.FindFirstIntersectingFace(world_ray.WorldToLocal(GetModel()), closest_intersect_point_out);
}

Mesh::VH Object::FindNearestVertex(const Ray &world_ray) const { return M.FindNearestVertex(world_ray.WorldToLocal(GetModel())); }
Mesh::EH Object::FindNearestEdge(const Ray &world_ray) const { return M.FindNearestEdge(world_ray.WorldToLocal(GetModel())); }

std::string Object::GetHighlightLabel() const {
    if (HighlightedFace.is_valid()) return std::format("Hovered face {}", HighlightedFace.idx());
    if (HighlightedVertex.is_valid()) return std::format("Hovered vertex {}", HighlightedVertex.idx());
    if (HighlightedEdge.is_valid()) return std::format("Hovered edge {}", HighlightedEdge.idx());
    return "Hovered: None";
}
