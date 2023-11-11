#include "GeometryInstance.h"

#include "Ray.h"

#include "VulkanContext.h"

using glm::vec3, glm::vec4;

GeometryInstance::GeometryInstance(const VulkanContext &vc, Geometry &&geometry)
    : VC(vc), G(std::move(geometry)) {
    static const std::vector AllModes{GeometryMode::Faces, GeometryMode::Vertices, GeometryMode::Edges};
    for (const auto mode : AllModes) CreateOrUpdateBuffers(mode);
}

void GeometryInstance::CreateOrUpdateBuffers(GeometryMode mode) {
    auto &buffers = BuffersForMode[mode];
    buffers.Vertices = G.GenerateVertices(mode);
    buffers.VertexBuffer.Size = sizeof(Vertex3D) * buffers.Vertices.size();
    VC.CreateOrUpdateBuffer(buffers.VertexBuffer, buffers.Vertices.data());

    buffers.Indices = G.GenerateIndices(mode);
    buffers.IndexBuffer.Size = sizeof(uint) * buffers.Indices.size();
    VC.CreateOrUpdateBuffer(buffers.IndexBuffer, buffers.Indices.data());
}

void GeometryInstance::SetEdgeColor(const vec4 &color) {
    G.SetEdgeColor(color);
    CreateOrUpdateBuffers(GeometryMode::Edges);
}

// Moller-Trumbore ray-triangle intersection algorithm.
// Returns true if the ray intersects the triangle, and sets `distance` to the distance along the ray to the intersection point.
static bool RayIntersectsTriangle(const Ray &ray, const glm::vec3 &tri_a, const glm::vec3 &tri_b, const glm::vec3 &tri_c, float &distance) {
    static const float Epsilon = 1e-7f; // Floating point error tolerance.

    const vec3 edge1 = tri_b - tri_a, edge2 = tri_c - tri_a;
    const vec3 h = glm::cross(ray.Direction, edge2);
    const float a = glm::dot(edge1, h); // Barycentric coordinate a.

    // Check if the ray is parallel to the triangle.
    if (a > -Epsilon && a < Epsilon) return false;

    // Check if the intersection point is inside the triangle (in barycentric coordinates).
    const float f = 1.0 / a;
    const vec3 s = ray.Origin - tri_a;
    const float u = f * glm::dot(s, h);
    if (u < 0.0 || u > 1.0) return false;

    const vec3 q = glm::cross(s, edge1);
    const float v = f * glm::dot(ray.Direction, q);
    if (v < 0.0 || u + v > 1.0) return false;

    // Calculate the intersection point's distance along the ray and verify it's positive (ahead of the ray's origin).
    distance = f * glm::dot(edge2, q);
    return distance > Epsilon;
}

Geometry::FH GeometryInstance::FindFirstIntersectingFace(const Ray &ray) const {
    const auto &tri_buffers = GetBuffers(GeometryMode::Faces); // Triangulated face buffers
    const std::vector<uint> &tri_indices = tri_buffers.Indices;
    const std::vector<Vertex3D> &tri_verts = tri_buffers.Vertices;

    float closest_distance = std::numeric_limits<float>::max();
    int closest_tri_i = -1;
    for (size_t tri_i = 0; tri_i < tri_buffers.Indices.size() / 3; tri_i++) {
        const vec3 tri_a{Model * vec4{tri_verts[tri_indices[tri_i * 3 + 0]].Position, 1}};
        const vec3 tri_b{Model * vec4{tri_verts[tri_indices[tri_i * 3 + 1]].Position, 1}};
        const vec3 tri_c{Model * vec4{tri_verts[tri_indices[tri_i * 3 + 2]].Position, 1}};

        float distance;
        if (RayIntersectsTriangle(ray, tri_a, tri_b, tri_c, distance) && distance < closest_distance) {
            closest_distance = distance;
            closest_tri_i = tri_i;
        }
    }

    return closest_tri_i == -1 ? Geometry::FH{} : G.TriangulatedIndexToFace(closest_tri_i);
}
