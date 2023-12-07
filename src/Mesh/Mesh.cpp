#include "Mesh.h"

#include "Ray.h"
#include "World.h"

#include <algorithm>
#include <ranges>

#include <glm/gtx/norm.hpp>

using namespace om;
using glm::vec3, glm::vec4, glm::mat3, glm::mat4;

using std::ranges::any_of;

bool Mesh::Load(const fs::path &file_path, PolyMesh &out_mesh) {
    OpenMesh::IO::Options read_options; // No options used yet, but keeping this here for future use.
    if (!OpenMesh::IO::read_mesh(out_mesh, file_path.string(), read_options)) {
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

static float SquaredDistanceToLineSegment(const vec3 &v1, const vec3 &v2, const vec3 &point) {
    const vec3 edge = v2 - v1;
    const float t = glm::clamp(glm::dot(point - v1, edge) / glm::dot(edge, edge), 0.f, 1.f);
    const vec3 closest_point = v1 + t * edge;
    return glm::distance2(point, closest_point);
}

// Moller-Trumbore ray-triangle intersection algorithm.
bool Mesh::RayIntersectsTriangle(const Ray &ray, VH v1, VH v2, VH v3, float *distance_out, vec3 *intersect_point_out) const {
    static const float Epsilon = 1e-7f; // Floating point error tolerance.

    const Point &ray_origin = ToOpenMesh(ray.Origin);
    const Point &ray_dir = ToOpenMesh(ray.Direction);
    const Point &p1 = M.point(v1), &p2 = M.point(v2), &p3 = M.point(v3);
    const Point edge1 = p2 - p1, edge2 = p3 - p1;
    const Point h = ray_dir % edge2;
    const float a = edge1.dot(h); // Barycentric coordinate a.

    // Check if the ray is parallel to the triangle.
    if (a > -Epsilon && a < Epsilon) return false;

    // Check if the intersection point is inside the triangle (in barycentric coordinates).
    const float f = 1.0 / a;
    const Point s = ray_origin - p1;
    const float u = f * s.dot(h);
    if (u < 0.0 || u > 1.0) return false;

    const Point q = s % edge1;
    const float v = f * ray_dir.dot(q);
    if (v < 0.0 || u + v > 1.0) return false;

    // Calculate the intersection point's distance along the ray and verify it's positive (ahead of the ray's origin).
    const float distance = f * edge2.dot(q);
    if (distance > Epsilon) {
        if (distance_out) *distance_out = distance;
        if (intersect_point_out) *intersect_point_out = ray.Origin + ray.Direction * distance;
        return true;
    }
    return false;
}

VH Mesh::FindNearestVertex(const Ray &ray_local) const {
    vec3 intersection_point;
    const auto face = FindFirstIntersectingFace(ray_local, &intersection_point);
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

FH Mesh::FindFirstIntersectingFace(const Ray &ray_local, vec3 *closest_intersect_point_out) const {
    // Avoid allocations in the loop.
    FH closest_face{};
    float distance;
    float closest_distance = std::numeric_limits<float>::max();
    vec3 intersect_point;
    vec3 closest_intersection_point; // Only tracked for output.
    for (const auto &fh : M.faces()) {
        auto fv_it = M.cfv_iter(fh);
        const VH v0 = *fv_it;
        ++fv_it;
        for (; fv_it != M.cfv_end(fh); ++fv_it) {
            const VH v1 = *fv_it, v2 = *(++fv_it);
            --fv_it;
            if (RayIntersectsTriangle(ray_local, v0, v1, v2, &distance, &intersect_point) && distance < closest_distance) {
                closest_distance = distance;
                closest_intersection_point = intersect_point;
                closest_face = fh;
            }
        }
    }

    if (closest_face.is_valid() && closest_intersect_point_out) *closest_intersect_point_out = closest_intersection_point;
    return closest_face;
}

EH Mesh::FindNearestEdge(const Ray &ray_local) const {
    vec3 intersection_point;
    const auto face = FindFirstIntersectingFace(ray_local, &intersection_point);
    if (!face.is_valid()) return Mesh::EH{};

    Mesh::EH closest_edge;
    float closest_distance_sq = std::numeric_limits<float>::max();
    for (const auto &heh : M.fh_range(face)) {
        const auto &edge_handle = M.edge_handle(heh);
        const auto &p1 = GetPosition(M.from_vertex_handle(M.halfedge_handle(edge_handle, 0)));
        const auto &p2 = GetPosition(M.to_vertex_handle(M.halfedge_handle(edge_handle, 0)));
        const float distance_sq = SquaredDistanceToLineSegment(p1, p2, intersection_point);
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

float Mesh::CalcFaceArea(FH fh) const {
    float area{0};
    auto fv_it = M.cfv_iter(fh);
    const auto p0 = M.point(*fv_it);
    ++fv_it;
    for (; fv_it != M.cfv_end(fh); ++fv_it) {
        const Point p1 = M.point(*fv_it), p2 = M.point(*(++fv_it));
        --fv_it;
        const auto cross_product = (p1 - p0) % (p2 - p0);
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
            vertices.emplace_back(point + NormalIndicatorLengthScale * CalcFaceArea(fh) * ToGlm(fn), ToGlm(fn), FaceNormalIndicatorColor);
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
    std::vector<uint> indices;
    for (const auto &fh : M.faces()) {
        auto fv_it = M.cfv_iter(fh);
        const VH v0 = *fv_it;
        ++fv_it;
        for (; fv_it != M.cfv_end(fh); ++fv_it) {
            const VH v1 = *fv_it, v2 = *(++fv_it);
            --fv_it;
            indices.insert(indices.end(), {uint(v0.idx()), uint(v1.idx()), uint(v2.idx())});
        }
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
