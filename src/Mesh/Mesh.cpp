#include "Mesh.h"

#include <algorithm>
#include <ranges>

#include "BVH.h"
#include "Ray.h"
#include "World.h"

using namespace om;

using std::ranges::any_of;

Mesh::Mesh(Mesh &&other)
    : BoundingBox(other.BoundingBox), M(std::move(other.M)), Bvh(std::move(other.Bvh)) {
    other.Bvh.reset();
}
Mesh::Mesh(const fs::path &file_path) {
    M.request_vertex_normals();
    M.request_face_normals();
    M.request_face_colors();
    Load(file_path, M);
    SetFaceColor(DefaultFaceColor);
    UpdateNormals();
    BoundingBox = ComputeBbox();
    Bvh = std::make_unique<BVH>(CreateFaceBoundingBoxes());
}
Mesh::Mesh(std::vector<vec3> &&vertices, std::vector<std::vector<uint>> &&faces, vec4 color) {
    M.request_vertex_normals();
    M.request_face_normals();
    M.request_face_colors();
    SetFaces(std::move(vertices), std::move(faces), color);
    UpdateNormals();
    BoundingBox = ComputeBbox();
    Bvh = std::make_unique<BVH>(CreateFaceBoundingBoxes());
}
Mesh::~Mesh() {
    M.release_vertex_normals();
    M.release_face_normals();
    M.release_face_colors();
}

bool Mesh::Load(const fs::path &file_path, PolyMesh &out_mesh) {
    OpenMesh::IO::Options read_options; // No options used yet, but keeping this here for future use.
    if (!OpenMesh::IO::read_mesh(out_mesh, file_path.string(), read_options)) {
        std::cerr << "Error loading mesh: " << file_path << std::endl;
        return false;
    }
    return true;
}

bool Mesh::VertexBelongsToFace(VH vh, FH fh) const {
    return vh.is_valid() && fh.is_valid() && any_of(M.fv_range(fh), [&](const auto &vh_o) { return vh_o == vh; });
}

bool Mesh::VertexBelongsToEdge(VH vh, EH eh) const {
    return vh.is_valid() && eh.is_valid() && any_of(M.voh_range(vh), [&](const auto &heh) { return M.edge_handle(heh) == eh; });
}

bool Mesh::VertexBelongsToFaceEdge(VH vh, FH fh, EH eh) const {
    return fh.is_valid() && eh.is_valid() &&
        any_of(M.voh_range(vh), [&](const auto &heh) {
               return M.edge_handle(heh) == eh && (M.face_handle(heh) == fh || M.face_handle(M.opposite_halfedge_handle(heh)) == fh);
           });
}

bool Mesh::EdgeBelongsToFace(EH eh, FH fh) const {
    return eh.is_valid() && fh.is_valid() && any_of(M.fh_range(fh), [&](const auto &heh) { return M.edge_handle(heh) == eh; });
}

static float SquaredDistanceToLineSegment(const vec3 &v1, const vec3 &v2, const vec3 &p) {
    const vec3 edge = v2 - v1;
    const float t = glm::clamp(glm::dot(p - v1, edge) / glm::dot(edge, edge), 0.f, 1.f);
    const vec3 closest_p = v1 + t * edge;
    const vec3 diff = p - closest_p;
    return glm::dot(diff, diff);
}

float Mesh::CalcFaceArea(FH fh) const {
    float area{0};
    auto fv_it = M.cfv_iter(fh);
    const Point p0 = M.point(*fv_it++);
    Point p1 = M.point(*fv_it++), p2;
    for (; fv_it.is_valid(); ++fv_it) {
        p2 = M.point(*fv_it);
        const auto cross_product = (p1 - p0) % (p2 - p0);
        area += cross_product.norm() * 0.5;
        p1 = p2;
    }
    return area;
}

std::vector<BBox> Mesh::CreateFaceBoundingBoxes() const {
    std::vector<BBox> boxes;
    boxes.reserve(M.n_faces());
    for (const auto &fh : M.faces()) {
        BBox box;
        for (const auto &vh : M.fv_range(fh)) {
            const auto &point = M.point(vh);
            box.Min = glm::min(box.Min, ToGlm(point));
            box.Max = glm::max(box.Max, ToGlm(point));
        }
        boxes.push_back(box);
    }
    return boxes;
}

RenderBuffers Mesh::CreateBvhBuffers(vec4 color) const {
    if (!Bvh) return {};

    std::vector<BBox> boxes = Bvh->CreateInternalBoxes();
    std::vector<Vertex3D> vertices;
    vertices.reserve(boxes.size() * 8);
    std::vector<uint> indices;
    indices.reserve(boxes.size() * BBox::EdgeIndices.size());
    for (uint i = 0; i < boxes.size(); ++i) {
        const auto &box = boxes[i];
        for (auto &corner : box.Corners()) vertices.emplace_back(corner, vec3{}, color);

        const uint index_offset = i * 8;
        for (const auto &index : BBox::EdgeIndices) indices.push_back(index_offset + index);
    }
    return {std::move(vertices), std::move(indices)};
}

// Moller-Trumbore ray-triangle intersection algorithm.
// Returns true if the ray intersects the given triangle.
// If ray intersects, sets `distance_out` to the distance along the ray to the intersection point, and sets `intersect_point_out`, if not null.
bool RayIntersectsTriangle(const Mesh::PolyMesh &m, const Ray &ray, VH v1, VH v2, VH v3, float *distance_out, vec3 *intersect_point_out) {
    static const float eps = 1e-7f; // Floating point error tolerance.

    const Point ray_origin = ToOpenMesh(ray.Origin), ray_dir = ToOpenMesh(ray.Direction);
    const Point &p1 = m.point(v1), &p2 = m.point(v2), &p3 = m.point(v3);
    const Point edge1 = p2 - p1, edge2 = p3 - p1;
    const Point h = ray_dir % edge2;
    const float a = edge1.dot(h); // Barycentric coordinate
    if (a > -eps && a < eps) return false; // Check if the ray is parallel to the triangle.

    // Check if the intersection point is inside the triangle (in barycentric coordinates).
    const Point s = ray_origin - p1;
    const float f = 1.0 / a, u = f * s.dot(h);
    if (u < 0.0 || u > 1.0) return false;

    const Point q = s % edge1;
    const float v = f * ray_dir.dot(q);
    if (v < 0.0 || u + v > 1.0) return false;

    // Calculate the intersection point's distance along the ray and verify it's ahead of the ray's origin.
    const float distance = f * edge2.dot(q);
    if (distance > eps) {
        if (distance_out) *distance_out = distance;
        if (intersect_point_out) *intersect_point_out = ray(distance);
        return true;
    }
    return false;
}

bool Mesh::RayIntersectsFace(const Ray &ray, FH fh, float *distance_out, vec3 *intersect_point_out) const {
    auto fv_it = M.cfv_iter(fh);
    const VH v0 = *fv_it++;
    VH v1 = *fv_it++, v2;
    for (; fv_it.is_valid(); ++fv_it) {
        v2 = *fv_it;
        if (RayIntersectsTriangle(M, ray, v0, v1, v2, distance_out, intersect_point_out)) return true;
        v1 = v2;
    }
    return false;
}

bool Mesh::RayIntersects(const Ray &local_ray) const {
    auto callback = [this, &local_ray](uint fi) { return RayIntersectsFace(local_ray, FH{int(fi)}); };
    return Bvh->Intersect(local_ray, callback).has_value();
}

FH Mesh::FindFirstIntersectingFace(const Ray &local_ray, vec3 *closest_intersect_point_out) const {
    // Avoid allocations in the loop.
    float distance = 0, closest_distance = std::numeric_limits<float>::max();
    vec3 intersect_point;
    FH closest_face{};
    for (const auto &fh : M.faces()) {
        if (RayIntersectsFace(local_ray, fh, &distance, &intersect_point) && distance < closest_distance) {
            closest_distance = distance;
            closest_face = fh;
            if (closest_intersect_point_out) *closest_intersect_point_out = intersect_point;
        }
    }

    return closest_face;
}

VH Mesh::FindNearestVertex(const Ray &local_ray) const {
    vec3 intersection_point;
    const auto face = FindFirstIntersectingFace(local_ray, &intersection_point);
    if (!face.is_valid()) return VH{};

    VH closest_vertex;
    float closest_distance_sq = std::numeric_limits<float>::max();
    for (const auto &vh : M.fv_range(face)) {
        const vec3 diff = GetPosition(vh) - intersection_point;
        const float distance_sq = glm::dot(diff, diff);
        if (distance_sq < closest_distance_sq) {
            closest_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }

    return closest_vertex;
}

EH Mesh::FindNearestEdge(const Ray &local_ray) const {
    vec3 intersection_point;
    const auto face = FindFirstIntersectingFace(local_ray, &intersection_point);
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

std::vector<uint> Mesh::CreateIndices(MeshElement element) const {
    switch (element) {
        case MeshElement::Vertex: return CreateTriangleIndices();
        case MeshElement::Edge: return CreateEdgeIndices();
        case MeshElement::Face: return CreateTriangulatedFaceIndices();
        case MeshElement::None: return {};
    }
}
std::vector<uint> Mesh::CreateNormalIndices(MeshElement mode) const {
    if (mode == MeshElement::None || mode == MeshElement::Edge) return {};

    const uint n = mode == MeshElement::Face ? M.n_faces() : M.n_vertices();
    std::vector<uint> indices;
    indices.reserve(n * 2);
    for (uint i = 0; i < n; ++i) {
        indices.push_back(i * 2);
        indices.push_back(i * 2 + 1);
    }
    return indices;
}

std::vector<Vertex3D> Mesh::CreateVertices(MeshElement element, ElementIndex highlighted) const {
    std::vector<Vertex3D> vertices;
    if (element == MeshElement::Vertex) {
        vertices.reserve(M.n_vertices());
        for (const auto &vh : M.vertices()) {
            const vec4 color = vh == highlighted || VertexBelongsToFace(vh, highlighted) || VertexBelongsToEdge(vh, highlighted) ? HighlightColor : vec4{1};
            vertices.emplace_back(GetPosition(vh), GetVertexNormal(vh), color);
        }
    } else if (element == MeshElement::Edge) {
        vertices.reserve(M.n_edges() * 2);
        for (const auto &eh : M.edges()) {
            const auto heh = M.halfedge_handle(eh, 0);
            const auto vh0 = M.from_vertex_handle(heh), vh1 = M.to_vertex_handle(heh);
            const vec4 color = eh == highlighted || vh0 == highlighted || vh1 == highlighted || EdgeBelongsToFace(eh, highlighted) ? HighlightColor : EdgeColor;
            vertices.emplace_back(GetPosition(vh0), GetVertexNormal(vh0), color);
            vertices.emplace_back(GetPosition(vh1), GetVertexNormal(vh1), color);
        }
    } else if (element == MeshElement::Face) {
        vertices.reserve(M.n_faces() * 3); // Lower bound assuming all faces are triangles.
        for (const auto &fh : M.faces()) {
            const auto &fn = M.normal(fh);
            const auto &fc = M.color(fh);
            for (const auto &vh : M.fv_range(fh)) {
                const vec4 color = vh == highlighted || fh == highlighted || VertexBelongsToFaceEdge(vh, fh, highlighted) ? HighlightColor : ToGlm(fc);
                vertices.emplace_back(GetPosition(vh), ToGlm(fn), color);
            }
        }
    }

    return vertices;
}

std::vector<Vertex3D> Mesh::CreateNormalVertices(MeshElement mode) const {
    std::vector<Vertex3D> vertices;
    if (mode == MeshElement::Vertex) {
        // Line for each vertex normal, with length scaled by the average edge length.
        vertices.reserve(M.n_vertices() * 2);
        for (const auto &vh : M.vertices()) {
            const auto &vn = GetVertexNormal(vh);
            const auto &voh_range = M.voh_range(vh);
            const float total_edge_length = std::reduce(voh_range.begin(), voh_range.end(), 0.f, [&](float total, const auto &heh) {
                return total + M.calc_edge_length(heh);
            });
            const float avg_edge_length = total_edge_length / M.valence(vh);
            const vec3 p = GetPosition(vh);
            vertices.emplace_back(p, vn, VertexNormalIndicatorColor);
            vertices.emplace_back(p + NormalIndicatorLengthScale * avg_edge_length * vn, vn, VertexNormalIndicatorColor);
        }
    } else if (mode == MeshElement::Face) {
        // Line for each face normal, with length scaled by the face area.
        vertices.reserve(M.n_faces() * 2);
        for (const auto &fh : M.faces()) {
            const vec3 fn = GetFaceNormal(fh);
            const vec3 p = ToGlm(M.calc_face_centroid(fh));
            vertices.emplace_back(p, fn, FaceNormalIndicatorColor);
            vertices.emplace_back(p + NormalIndicatorLengthScale * std::sqrt(CalcFaceArea(fh)) * fn, fn, FaceNormalIndicatorColor);
        }
    }
    return vertices;
}

BBox Mesh::ComputeBbox() const {
    BBox bbox;
    for (const auto &vh : M.vertices()) {
        const auto &v = ToGlm(M.point(vh));
        bbox.Min = glm::min(bbox.Min, v);
        bbox.Max = glm::max(bbox.Max, v);
    }
    return bbox;
}

std::vector<uint> Mesh::CreateTriangleIndices() const {
    std::vector<uint> indices;
    for (const auto &fh : M.faces()) {
        auto fv_it = M.cfv_iter(fh);
        const VH v0 = *fv_it++;
        VH v1 = *fv_it++, v2;
        for (; fv_it.is_valid(); ++fv_it) {
            v2 = *fv_it;
            indices.insert(indices.end(), {uint(v0.idx()), uint(v1.idx()), uint(v2.idx())});
            v1 = v2;
        }
    }
    return indices;
}

std::vector<uint> Mesh::CreateTriangulatedFaceIndices() const {
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

std::vector<uint> Mesh::CreateEdgeIndices() const {
    std::vector<uint> indices;
    indices.reserve(M.n_edges() * 2);
    for (uint ei = 0; ei < M.n_edges(); ++ei) {
        indices.push_back(2 * ei);
        indices.push_back(2 * ei + 1);
    }
    return indices;
}
