#include "GeometryInstance.h"

#include "Ray.h"
#include "VulkanContext.h"

#include <format>
#include <glm/mat3x3.hpp>

using glm::vec3, glm::vec4, glm::mat3, glm::mat4;

GeometryInstance::GeometryInstance(const VulkanContext &vc, Geometry &&geometry)
    : VC(vc), G(std::move(geometry)) {
    CreateOrUpdateBuffers();
}

void GeometryInstance::CreateOrUpdateBuffers() {
    static const std::vector AllModes{GeometryMode::Faces, GeometryMode::Vertices, GeometryMode::Edges};
    for (const auto mode : AllModes) CreateOrUpdateBuffers(mode);
}

void GeometryInstance::CreateOrUpdateBuffers(GeometryMode mode) {
    auto &buffers = BuffersForMode[mode];
    buffers.Vertices = G.GenerateVertices(mode, HighlightedFace, HighlightedVertex, HighlightedEdge);
    buffers.VertexBuffer.Size = sizeof(Vertex3D) * buffers.Vertices.size();
    VC.CreateOrUpdateBuffer(buffers.VertexBuffer, buffers.Vertices.data());

    buffers.Indices = G.GenerateIndices(mode);
    buffers.IndexBuffer.Size = sizeof(uint) * buffers.Indices.size();
    VC.CreateOrUpdateBuffer(buffers.IndexBuffer, buffers.Indices.data());
}

// Moller-Trumbore ray-triangle intersection algorithm.
// Returns true if the ray intersects the triangle, and sets `distance` to the distance along the ray to the intersection point.
static bool RayIntersectsTriangle(const Ray &ray, const glm::mat3 &triangle, float &distance) {
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
    distance = f * glm::dot(edge2, q);
    return distance > Epsilon;
}

Geometry::FH GeometryInstance::FindFirstIntersectingFaceLocal(const Ray &ray_local) const {
    const auto &tri_buffers = GetBuffers(GeometryMode::Faces); // Triangulated face buffers
    const std::vector<uint> &tri_indices = tri_buffers.Indices;
    const std::vector<Vertex3D> &tri_verts = tri_buffers.Vertices;

    float closest_distance = std::numeric_limits<float>::max();
    int closest_tri_i = -1;
    mat3 triangle; // Use a single triangle to avoid allocations in the loop.
    float distance;
    for (size_t tri_i = 0; tri_i < tri_buffers.Indices.size() / 3; tri_i++) {
        triangle[0] = tri_verts[tri_indices[tri_i * 3 + 0]].Position;
        triangle[1] = tri_verts[tri_indices[tri_i * 3 + 1]].Position;
        triangle[2] = tri_verts[tri_indices[tri_i * 3 + 2]].Position;
        if (RayIntersectsTriangle(ray_local, triangle, distance) && distance < closest_distance) {
            closest_distance = distance;
            closest_tri_i = tri_i;
        }
    }

    return closest_tri_i == -1 ? Geometry::FH{} : G.TriangulatedIndexToFace(closest_tri_i);
}
Geometry::FH GeometryInstance::FindFirstIntersectingFace(const Ray &ray_world) const {
    return FindFirstIntersectingFaceLocal(ray_world.WorldToLocal(Model));
}

Geometry::VH GeometryInstance::FindNearestVertexLocal(const Ray &ray_local) const {
    const auto face = FindFirstIntersectingFaceLocal(ray_local);
    if (!face.is_valid()) return Geometry::VH{};

    float closest_distance_sq = std::numeric_limits<float>::max();
    Geometry::VH closest_vertex;
    const auto &mesh = G.GetMesh();
    for (const auto &vh : mesh.fv_range(face)) {
        const float distance_sq = ray_local.SquaredDistanceToPoint(ToGlm(mesh.point(vh)));
        if (distance_sq < closest_distance_sq) {
            closest_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }

    return closest_vertex;
}
Geometry::VH GeometryInstance::FindNearestVertex(const Ray &ray_world) const {
    return FindNearestVertexLocal(ray_world.WorldToLocal(Model));
}

Geometry::EH GeometryInstance::FindNearestEdgeLocal(const Ray &ray_local) const {
    float closest_distance_sq = std::numeric_limits<float>::max();
    Geometry::EH closest_edge;
    const auto &mesh = G.GetMesh();
    for (const auto &eh : mesh.edges()) {
        const auto heh = mesh.halfedge_handle(eh, 0);
        const auto &v1 = ToGlm(mesh.point(mesh.from_vertex_handle(heh)));
        const auto &v2 = ToGlm(mesh.point(mesh.to_vertex_handle(heh)));
        const float distance_sq = ray_local.SquaredDistanceToEdge(v1, v2);
        if (distance_sq < closest_distance_sq) {
            closest_distance_sq = distance_sq;
            closest_edge = eh;
        }
    }

    return closest_edge;
}

Geometry::EH GeometryInstance::FindNearestEdge(const Ray &ray_world) const {
    return FindNearestEdgeLocal(ray_world.WorldToLocal(Model));
}

std::string GeometryInstance::GetHighlightLabel() const {
    if (HighlightedFace.is_valid()) return std::format("Hovered face {}", HighlightedFace.idx());
    if (HighlightedVertex.is_valid()) return std::format("Hovered vertex {}", HighlightedVertex.idx());
    if (HighlightedEdge.is_valid()) return std::format("Hovered edge {}", HighlightedEdge.idx());
    return "Hovered: None";
}