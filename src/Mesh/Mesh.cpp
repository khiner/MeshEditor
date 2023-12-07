#include "Mesh.h"

#include "Ray.h"
#include "World.h"

#include <algorithm>
#include <ranges>

#include <glm/gtx/norm.hpp>

using namespace om;
using glm::vec3, glm::vec4, glm::mat3, glm::mat4;

using std::ranges::any_of;

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

static float SquaredDistanceToLineSegment(const vec3 &v1, const vec3 &v2, const vec3 &point) {
    const vec3 edge = v2 - v1;
    const float t = glm::clamp(glm::dot(point - v1, edge) / glm::dot(edge, edge), 0.f, 1.f);
    const vec3 closest_point = v1 + t * edge;
    return glm::distance2(point, closest_point);
}

bool Mesh::Load(const fs::path &file_path) {
    OpenMesh::IO::Options read_options; // No options used yet, but keeping this here for future use.
    if (!OpenMesh::IO::read_mesh(M, file_path.string(), read_options)) {
        std::cerr << "Error loading mesh: " << file_path << std::endl;
        return false;
    }
    return true;
}

bool Mesh::DoesVertexBelongToFace(VH vertex, FH face) const {
    return face.is_valid() && any_of(M.fv_range(face), [&](const auto &vh) { return vh == vertex; });
}

bool Mesh::DoesVertexBelongToEdge(VH vertex, EH edge) const {
    return edge.is_valid() && any_of(M.voh_range(vertex), [&](const auto &heh) { return M.edge_handle(heh) == edge; });
}

bool Mesh::DoesVertexBelongToFaceEdge(VH vertex, FH face, EH edge) const {
    return face.is_valid() && edge.is_valid() &&
        any_of(M.voh_range(vertex), [&](const auto &heh) {
               return M.edge_handle(heh) == edge && (M.face_handle(heh) == face || M.face_handle(M.opposite_halfedge_handle(heh)) == face);
           });
}

bool Mesh::DoesEdgeBelongToFace(EH edge, FH face) const {
    return face.is_valid() && any_of(M.fh_range(face), [&](const auto &heh) { return M.edge_handle(heh) == edge; });
}

VH Mesh::FindNearestVertex(const MeshBuffers &tri_buffers, const Ray &ray_local) const {
    vec3 intersection_point;
    const auto face = FindFirstIntersectingFace(tri_buffers, ray_local, &intersection_point);
    if (!face.is_valid()) return VH{};

    VH closest_vertex;
    float closest_distance_sq = std::numeric_limits<float>::max();
    for (const auto &vh : M.fv_range(face)) {
        const float distance_sq = glm::distance2(GetPosition(vh), intersection_point);
        if (distance_sq < closest_distance_sq) {
            closest_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }

    return closest_vertex;
}

FH Mesh::FindFirstIntersectingFace(const MeshBuffers &tri_buffers, const Ray &ray_local, vec3 *closest_intersect_point_out) const {
    const std::vector<Vertex3D> &tri_verts = tri_buffers.Vertices;
    const std::vector<uint> &tri_indices = tri_buffers.Indices;

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
        return TriangulatedIndexToFace(closest_tri_i);
    }
    return Mesh::FH{};
}

EH Mesh::FindNearestEdge(const MeshBuffers &tri_buffers, const Ray &ray_local) const {
    vec3 intersection_point;
    const auto face = FindFirstIntersectingFace(tri_buffers, ray_local, &intersection_point);
    if (!face.is_valid()) return Mesh::EH{};

    Mesh::EH closest_edge;
    float closest_distance_sq = std::numeric_limits<float>::max();
    for (const auto &heh : M.fh_range(face)) {
        const auto &edge_handle = M.edge_handle(heh);
        const auto &v1 = GetPosition(M.from_vertex_handle(M.halfedge_handle(edge_handle, 0)));
        const auto &v2 = GetPosition(M.to_vertex_handle(M.halfedge_handle(edge_handle, 0)));
        const float distance_sq = SquaredDistanceToLineSegment(v1, v2, intersection_point);
        if (distance_sq < closest_distance_sq) {
            closest_distance_sq = distance_sq;
            closest_edge = edge_handle;
        }
    }

    return closest_edge;
}

std::vector<uint> Mesh::GenerateIndices(MeshElement element) const {
    switch (element) {
        case MeshElement::Faces: return GenerateTriangulatedFaceIndices();
        case MeshElement::Edges: return GenerateEdgeIndices();
        case MeshElement::Vertices: return GenerateTriangleIndices();
        default: return {};
    }
}
std::vector<uint> Mesh::GenerateIndices(NormalMode mode) const {
    switch (mode) {
        case NormalMode::Faces: return GenerateFaceNormalIndicatorIndices();
        case NormalMode::Vertices: return GenerateVertexNormalIndicatorIndices();
        default: return {};
    }
}

std::vector<Vertex3D> Mesh::GenerateVertices(MeshElement element, FH highlighted_face, VH highlighted_vertex, EH highlighted_edge) const {
    std::vector<Vertex3D> vertices;
    if (element == MeshElement::Faces) {
        vertices.reserve(M.n_faces() * 3); // At least 3 vertices per face.
        for (const auto &fh : M.faces()) {
            const auto &fn = M.normal(fh);
            const auto &fc = M.color(fh);
            for (const auto &vh : M.fv_range(fh)) {
                const vec4 color = vh == highlighted_vertex || fh == highlighted_face || DoesVertexBelongToFaceEdge(vh, fh, highlighted_edge) ? HighlightColor : ToGlm(fc);
                vertices.emplace_back(GetPosition(vh), ToGlm(fn), color);
            }
        }
    } else if (element == MeshElement::Vertices) {
        vertices.reserve(M.n_vertices());
        for (const auto &vh : M.vertices()) {
            const vec4 color = vh == highlighted_vertex || DoesVertexBelongToFace(vh, highlighted_face) || DoesVertexBelongToEdge(vh, highlighted_edge) ? HighlightColor : vec4{1};
            vertices.emplace_back(GetPosition(vh), GetVertexNormal(vh), color);
        }
    } else if (element == MeshElement::Edges) {
        vertices.reserve(M.n_edges() * 2);
        for (const auto &eh : M.edges()) {
            const auto &heh = M.halfedge_handle(eh, 0);
            const auto &vh0 = M.from_vertex_handle(heh);
            const auto &vh1 = M.to_vertex_handle(heh);
            const vec4 color = eh == highlighted_edge || vh0 == highlighted_vertex || vh1 == highlighted_vertex || DoesEdgeBelongToFace(eh, highlighted_face) ? HighlightColor : EdgeColor;
            vertices.emplace_back(GetPosition(vh0), GetVertexNormal(vh0), color);
            vertices.emplace_back(GetPosition(vh1), GetVertexNormal(vh1), color);
        }
    }

    return vertices;
}

static float CalcFaceArea(const PolyMesh &mesh, FH fh) {
    std::vector<OpenMesh::Vec3f> vertices;
    vertices.reserve(mesh.valence(fh));
    for (const auto &vh : mesh.fv_range(fh)) vertices.emplace_back(mesh.point(vh));

    float area{0};
    for (size_t i = 1; i < vertices.size() - 1; ++i) {
        const auto &v0 = vertices[0], &v1 = vertices[i], &v2 = vertices[i + 1];
        const auto cross_product = (v1 - v0) % (v2 - v0);
        area += cross_product.norm() * 0.5;
    }

    return area;
}

std::vector<Vertex3D> Mesh::GenerateVertices(NormalMode mode) const {
    std::vector<Vertex3D> vertices;
    if (mode == NormalMode::Faces) {
        // Line for each face normal, with length scaled by the face area.
        vertices.reserve(M.n_faces() * 2);
        for (const auto &fh : M.faces()) {
            const auto &fn = M.normal(fh);
            const vec3 point = ToGlm(M.calc_face_centroid(fh));
            vertices.emplace_back(point, ToGlm(fn), FaceNormalIndicatorColor);
            vertices.emplace_back(point + NormalIndicatorLengthScale * CalcFaceArea(M, fh) * ToGlm(fn), ToGlm(fn), FaceNormalIndicatorColor);
        }
    } else if (mode == NormalMode::Vertices) {
        // Line for each vertex normal, with length scaled by the average edge length.
        vertices.reserve(M.n_vertices() * 2);
        for (const auto &vh : M.vertices()) {
            const auto &vn = M.normal(vh);
            const auto &voh_range = M.voh_range(vh);
            const float total_edge_length = std::reduce(voh_range.begin(), voh_range.end(), 0.f, [&](float total, const auto &heh) {
                return total + M.calc_edge_length(heh);
            });
            const float avg_edge_length = total_edge_length / M.valence(vh);
            const vec3 point = ToGlm(M.point(vh));
            vertices.emplace_back(point, ToGlm(vn), VertexNormalIndicatorColor);
            vertices.emplace_back(point + NormalIndicatorLengthScale * avg_edge_length * ToGlm(vn), ToGlm(vn), VertexNormalIndicatorColor);
        }
    }

    return vertices;
}

// [{min_x, min_y, min_z}, {max_x, max_y, max_z}]
std::pair<vec3, vec3> Mesh::ComputeBounds() const {
    static const float min_float = std::numeric_limits<float>::lowest();
    static const float max_float = std::numeric_limits<float>::max();

    vec3 min(max_float), max(min_float);
    for (const auto &vh : M.vertices()) {
        const auto &p = M.point(vh);
        min.x = std::min(min.x, p[0]);
        min.y = std::min(min.y, p[1]);
        min.z = std::min(min.z, p[2]);
        max.x = std::max(max.x, p[0]);
        max.y = std::max(max.y, p[1]);
        max.z = std::max(max.z, p[2]);
    }

    return {min, max};
}

std::vector<uint> Mesh::GenerateTriangleIndices() const {
    auto triangulated_mesh = M; // `triangulate` is in-place, so we need to make a copy.
    triangulated_mesh.triangulate();
    std::vector<uint> indices;
    for (const auto &fh : triangulated_mesh.faces()) {
        auto v_it = triangulated_mesh.cfv_iter(fh);
        indices.insert(indices.end(), {uint(v_it->idx()), uint((++v_it)->idx()), uint((++v_it)->idx())});
    }
    return indices;
}

std::vector<uint> Mesh::GenerateTriangulatedFaceIndices() const {
    std::vector<uint> indices;
    uint index = 0;
    for (const auto &fh : M.faces()) {
        const auto valence = M.valence(fh);
        for (uint i = 0; i < valence - 2; ++i) {
            indices.insert(indices.end(), {index, index + i + 1, index + i + 2});
        }
        index += valence;
    }
    return indices;
}

Mesh::FH Mesh::TriangulatedIndexToFace(uint triangle_index) const {
    for (const auto &fh : M.faces()) {
        const auto valence = M.valence(fh);
        if (triangle_index < valence - 2) return fh;
        triangle_index -= valence - 2;
    }
    throw std::runtime_error("Invalid triangle index: " + std::to_string(triangle_index));
}

std::vector<uint> Mesh::GenerateEdgeIndices() const {
    std::vector<uint> indices;
    indices.reserve(M.n_edges() * 2);
    for (uint ei = 0; ei < M.n_edges(); ++ei) {
        indices.push_back(ei * 2);
        indices.push_back(ei * 2 + 1);
    }
    return indices;
}

std::vector<uint> Mesh::GenerateFaceNormalIndicatorIndices() const {
    std::vector<uint> indices;
    indices.reserve(M.n_faces() * 2);
    for (uint fi = 0; fi < M.n_faces(); ++fi) {
        indices.push_back(fi * 2);
        indices.push_back(fi * 2 + 1);
    }
    return indices;
}

std::vector<uint> Mesh::GenerateVertexNormalIndicatorIndices() const {
    std::vector<uint> indices;
    indices.reserve(M.n_vertices() * 2);
    for (uint vi = 0; vi < M.n_vertices(); ++vi) {
        indices.push_back(vi * 2);
        indices.push_back(vi * 2 + 1);
    }
    return indices;
}
