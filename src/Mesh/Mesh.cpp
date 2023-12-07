#include "Mesh/Mesh.h"

#include "Ray.h"
#include "VulkanContext.h"

#include <format>
#include <glm/gtx/norm.hpp>
#include <glm/mat3x3.hpp>

using glm::vec3, glm::vec4, glm::mat3, glm::mat4;

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

Mesh::Mesh(const VulkanContext &vc, Geometry &&geometry)
    : VC(vc), G(std::move(geometry)) {
    CreateOrUpdateBuffers();
}

void Mesh::CreateOrUpdateBuffers() {
    static const std::vector AllElements{MeshElement::Faces, MeshElement::Vertices, MeshElement::Edges};
    for (const auto element : AllElements) CreateOrUpdateBuffers(element);
}

void Mesh::CreateOrUpdateBuffers(MeshElement element) {
    if (!ElementBuffers.contains(element)) ElementBuffers[element] = std::make_unique<VkMeshBuffers>(VC);
    G.UpdateNormals(); // todo only update when necessary.
    ElementBuffers[element]->Set(G.GenerateBuffers(element, HighlightedFace, HighlightedVertex, HighlightedEdge));
}

const VkMeshBuffers &Mesh::GetBuffers(MeshElement mode) const { return *ElementBuffers.at(mode); }
const VkMeshBuffers *Mesh::GetFaceNormalIndicatorBuffers() const { return FaceNormalIndicatorBuffers.get(); }
const VkMeshBuffers *Mesh::GetVertexNormalIndicatorBuffers() const { return VertexNormalIndicatorBuffers.get(); }

void Mesh::ShowNormalIndicators(NormalMode mode, bool show) {
    auto &buffers = mode == NormalMode::Faces ? FaceNormalIndicatorBuffers : VertexNormalIndicatorBuffers;
    buffers.reset();
    if (!show) return;

    G.UpdateNormals(); // todo do we need this for lines?
    buffers = std::make_unique<VkMeshBuffers>(VC, G.GenerateBuffers(mode));
}

// Moller-Trumbore ray-triangle intersection algorithm.
// Returns true if the ray intersects the triangle.
// If ray intersects, sets `distance_out` to the distance along the ray to the intersection point, and sets `intersect_point_out`, if not null.
static bool RayIntersectsTriangle(const Ray &ray, const mat3 &triangle, float *distance_out, vec3 *intersect_point_out = nullptr) {
    static const float Epsilon = 1e-7f; // Floating point error tolerance.

    const vec3 edge1 = triangle[1] - triangle[0], edge2 = triangle[2] - triangle[0];
    const vec3 h = glm::cross(ray.Direction, edge2);
    const float a = glm::dot(edge1, h); // Barycentric coordinate a.

    // Check if the ray is parallel to the triangle.
    if (a > -Epsilon && a < Epsilon) return false;

    // Check if the intersection point is inside the triangle (in barycentric coordinates).
    const float f = 1.0 / a;
    const vec3 s = ray.Origin - triangle[0];
    const float u = f * glm::dot(s, h);
    if (u < 0.0 || u > 1.0) return false;

    const vec3 q = glm::cross(s, edge1);
    const float v = f * glm::dot(ray.Direction, q);
    if (v < 0.0 || u + v > 1.0) return false;

    // Calculate the intersection point's distance along the ray and verify it's positive (ahead of the ray's origin).
    const float distance = f * glm::dot(edge2, q);
    if (distance > Epsilon) {
        if (distance_out) *distance_out = distance;
        if (intersect_point_out) *intersect_point_out = ray.Origin + ray.Direction * distance;
        return true;
    }
    return false;
}

Geometry::FH Mesh::FindFirstIntersectingFaceLocal(const Ray &ray_local, vec3 *closest_intersect_point_out) const {
    const auto &tri_buffers = GetBuffers(MeshElement::Faces); // Triangulated face buffers
    const std::vector<uint> &tri_indices = tri_buffers.GetIndices();
    const std::vector<Vertex3D> &tri_verts = tri_buffers.GetVertices();

    // Avoid allocations in the loop.
    int closest_tri_i = -1;
    mat3 triangle;
    float distance;
    float closest_distance = std::numeric_limits<float>::max();
    vec3 intersect_point;
    vec3 closest_intersection_point; // Only tracked for output.
    for (size_t tri_i = 0; tri_i < tri_indices.size() / 3; tri_i++) {
        triangle[0] = tri_verts[tri_indices[tri_i * 3 + 0]].Position;
        triangle[1] = tri_verts[tri_indices[tri_i * 3 + 1]].Position;
        triangle[2] = tri_verts[tri_indices[tri_i * 3 + 2]].Position;
        if (RayIntersectsTriangle(ray_local, triangle, &distance, &intersect_point) && distance < closest_distance) {
            closest_distance = distance;
            closest_intersection_point = intersect_point;
            closest_tri_i = tri_i;
        }
    }

    if (closest_tri_i != -1) {
        if (closest_intersect_point_out) *closest_intersect_point_out = closest_intersection_point;
        return G.TriangulatedIndexToFace(closest_tri_i);
    }
    return Geometry::FH{};
}
Geometry::FH Mesh::FindFirstIntersectingFace(const Ray &ray_world, vec3 *closest_intersect_point_out) const {
    return FindFirstIntersectingFaceLocal(ray_world.WorldToLocal(Model), closest_intersect_point_out);
}

Geometry::VH Mesh::FindNearestVertexLocal(const Ray &ray_local) const {
    vec3 intersection_point;
    const auto face = FindFirstIntersectingFaceLocal(ray_local, &intersection_point);
    if (!face.is_valid()) return Geometry::VH{};

    Geometry::VH closest_vertex;
    float closest_distance_sq = std::numeric_limits<float>::max();
    const auto &mesh = G.GetMesh();
    for (const auto &vh : mesh.fv_range(face)) {
        const float distance_sq = glm::distance2(G.GetPosition(vh), intersection_point);
        if (distance_sq < closest_distance_sq) {
            closest_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }

    return closest_vertex;
}
Geometry::VH Mesh::FindNearestVertex(const Ray &ray_world) const {
    return FindNearestVertexLocal(ray_world.WorldToLocal(Model));
}

static float SquaredDistanceToLineSegment(const vec3 &v1, const vec3 &v2, const vec3 &point) {
    const vec3 edge = v2 - v1;
    const float t = glm::clamp(glm::dot(point - v1, edge) / glm::dot(edge, edge), 0.f, 1.f);
    const vec3 closest_point = v1 + t * edge;
    return glm::distance2(point, closest_point);
}

Geometry::EH Mesh::FindNearestEdgeLocal(const Ray &ray_local) const {
    vec3 intersection_point;
    const auto face = FindFirstIntersectingFaceLocal(ray_local, &intersection_point);
    if (!face.is_valid()) return Geometry::EH{};

    Geometry::EH closest_edge;
    float closest_distance_sq = std::numeric_limits<float>::max();
    const auto &mesh = G.GetMesh();
    for (const auto &heh : mesh.fh_range(face)) {
        const auto &edge_handle = mesh.edge_handle(heh);
        const auto &v1 = G.GetPosition(mesh.from_vertex_handle(mesh.halfedge_handle(edge_handle, 0)));
        const auto &v2 = G.GetPosition(mesh.to_vertex_handle(mesh.halfedge_handle(edge_handle, 0)));
        const float distance_sq = SquaredDistanceToLineSegment(v1, v2, intersection_point);
        if (distance_sq < closest_distance_sq) {
            closest_distance_sq = distance_sq;
            closest_edge = edge_handle;
        }
    }

    return closest_edge;
}

Geometry::EH Mesh::FindNearestEdge(const Ray &ray_world) const {
    return FindNearestEdgeLocal(ray_world.WorldToLocal(Model));
}

std::string Mesh::GetHighlightLabel() const {
    if (HighlightedFace.is_valid()) return std::format("Hovered face {}", HighlightedFace.idx());
    if (HighlightedVertex.is_valid()) return std::format("Hovered vertex {}", HighlightedVertex.idx());
    if (HighlightedEdge.is_valid()) return std::format("Hovered edge {}", HighlightedEdge.idx());
    return "Hovered: None";
}
