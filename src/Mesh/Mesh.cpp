#include "Mesh.h"

#include <algorithm>
#include <ranges>
#include <unordered_set>

#include "BVH.h"
#include "Ray.h"

using namespace om;

using std::ranges::any_of;

Mesh::Mesh(const fs::path &file_path) {
    Load(file_path, M);
    // if (IsTriangleSoup()) M = DeduplicateVertices();
    // Deduplicate even if not strictly triangle soup. Assumes this is a surface mesh.
    M = DeduplicateVertices();

    M.request_vertex_normals();
    M.request_face_normals();
    M.request_face_colors();
    SetFaceColor(DefaultFaceColor);
    M.update_normals();
    BoundingBox = ComputeBbox();
    Bvh = std::make_unique<BVH>(CreateFaceBoundingBoxes());
}
Mesh::Mesh(std::vector<vec3> &&vertices, std::vector<std::vector<uint>> &&faces, vec4 color) {
    M.request_vertex_normals();
    M.request_face_normals();
    M.request_face_colors();
    SetFaces(std::move(vertices), std::move(faces), color);
    M.update_normals();
    BoundingBox = ComputeBbox();
    Bvh = std::make_unique<BVH>(CreateFaceBoundingBoxes());
}
Mesh::Mesh(Mesh &&other)
    : BoundingBox(other.BoundingBox), M(std::move(other.M)), Bvh(std::move(other.Bvh)), HighlightedElements(std::move(other.HighlightedElements)) {
    other.Bvh.reset();
}
Mesh::~Mesh() {
    M.release_vertex_normals();
    M.release_face_normals();
    M.release_face_colors();
}

const Mesh &Mesh::operator=(Mesh &&other) {
    if (this != &other) {
        BoundingBox = std::move(other.BoundingBox);
        M = std::move(other.M);
        Bvh = std::move(other.Bvh);
        HighlightedElements = std::move(other.HighlightedElements);
    }
    return *this;
}

bool Mesh::Load(const fs::path &file_path, PolyMesh &out_mesh) {
    OpenMesh::IO::Options read_options; // No options used yet, but keeping this here for future use.
    if (!OpenMesh::IO::read_mesh(out_mesh, file_path.string(), read_options)) {
        std::cerr << "Error loading mesh: " << file_path << std::endl;
        return false;
    }
    return true;
}

struct VertexHash {
    size_t operator()(const Point &p) const {
        return std::hash<float>{}(p[0]) ^ std::hash<float>{}(p[1]) ^ std::hash<float>{}(p[2]);
    }
};

Mesh::PolyMesh Mesh::DeduplicateVertices() {
    PolyMesh deduped;
    std::unordered_map<Point, VH, VertexHash> unique_vertices;

    // Add unique vertices.
    for (auto v_it = M.vertices_begin(); v_it != M.vertices_end(); ++v_it) {
        const auto p = M.point(*v_it);
        if (auto [it, inserted] = unique_vertices.try_emplace(p, VH()); inserted) {
            it->second = deduped.add_vertex(p);
        }
    }
    // Add faces.
    for (const auto &fh : M.faces()) {
        std::vector<VH> new_face;
        new_face.reserve(M.valence(fh));
        for (const auto &vh : M.fv_range(fh)) new_face.emplace_back(unique_vertices.at(M.point(vh)));
        deduped.add_face(new_face);
    }
    return deduped;
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

FH Mesh::FindNearestIntersectingFace(const Ray &local_ray, vec3 *nearest_intersect_point_out) const {
    float distance = 0, min_distance = std::numeric_limits<float>::max();
    vec3 intersect_point;
    FH nearest_face{};
    Bvh->Intersect(local_ray, [&](uint fi) {
        if (RayIntersectsFace(local_ray, FH{int(fi)}, &distance, &intersect_point) && distance < min_distance) {
            min_distance = distance;
            nearest_face = FH{int(fi)};
            if (nearest_intersect_point_out) *nearest_intersect_point_out = intersect_point;
        }
        return false; // We want the nearest face, not just any intersecting face.
    });
    return nearest_face;
}

std::optional<float> Mesh::Intersect(const Ray &local_ray) const {
    float distance = 0, min_distance = std::numeric_limits<float>::max();
    Bvh->Intersect(local_ray, [&](uint fi) {
        if (RayIntersectsFace(local_ray, FH{int(fi)}, &distance) && distance < min_distance) {
            min_distance = distance;
        }
        return false; // We want the nearest intersection, not just any intersection.
    });
    return min_distance < std::numeric_limits<float>::max() ? std::make_optional(min_distance) : std::nullopt;
}
bool Mesh::RayIntersects(const Ray &local_ray) const {
    auto callback = [this, &local_ray](uint fi) { return RayIntersectsFace(local_ray, FH{int(fi)}); };
    return Bvh->Intersect(local_ray, callback).has_value();
}

VH Mesh::FindNearestVertex(vec3 world_point) const {
    VH closest_vertex;
    float min_distance_sq = std::numeric_limits<float>::max();
    for (const auto &vh : M.vertices()) {
        const vec3 diff = GetPosition(vh) - world_point;
        const float distance_sq = glm::dot(diff, diff);
        if (distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }
    return closest_vertex;
}

VH Mesh::FindNearestVertex(const Ray &local_ray) const {
    vec3 intersection_point;
    const auto face = FindNearestIntersectingFace(local_ray, &intersection_point);
    if (!face.is_valid()) return VH{};

    VH closest_vertex;
    float min_distance_sq = std::numeric_limits<float>::max();
    for (const auto &vh : M.fv_range(face)) {
        const vec3 diff = GetPosition(vh) - intersection_point;
        const float distance_sq = glm::dot(diff, diff);
        if (distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }

    return closest_vertex;
}

EH Mesh::FindNearestEdge(const Ray &local_ray) const {
    vec3 intersection_point;
    const auto face = FindNearestIntersectingFace(local_ray, &intersection_point);
    if (!face.is_valid()) return Mesh::EH{};

    Mesh::EH closest_edge;
    float min_distance_sq = std::numeric_limits<float>::max();
    for (const auto &heh : M.fh_range(face)) {
        const auto &edge_handle = M.edge_handle(heh);
        const auto &p1 = GetPosition(M.from_vertex_handle(M.halfedge_handle(edge_handle, 0)));
        const auto &p2 = GetPosition(M.to_vertex_handle(M.halfedge_handle(edge_handle, 0)));
        const float distance_sq = SquaredDistanceToLineSegment(p1, p2, intersection_point);
        if (distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
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

// Used as an intermediate for creating render vertices (`Vertex3D`).
struct VerticesHandle {
    Mesh::ElementIndex Parent; // A vertex can belong to itself, an edge, or a face.
    std::vector<Mesh::VH> VHs;
};

std::vector<Vertex3D> Mesh::CreateVertices(MeshElement render_element, const ElementIndex &highlight) const {
    std::vector<VerticesHandle> handles;
    if (render_element == MeshElement::Vertex) {
        handles.reserve(M.n_vertices());
        for (const auto &vh : M.vertices()) handles.emplace_back(vh, std::vector<VH>{vh});
    } else if (render_element == MeshElement::Edge) {
        handles.reserve(M.n_edges() * 2);
        for (const auto &eh : M.edges()) {
            const auto heh = M.halfedge_handle(eh, 0);
            handles.emplace_back(eh, std::vector<VH>{M.from_vertex_handle(heh), M.to_vertex_handle(heh)});
        }
    } else if (render_element == MeshElement::Face) {
        handles.reserve(M.n_faces() * 3); // Lower bound assuming all faces are triangles.
        for (const auto &fh : M.faces()) {
            for (const auto &vh : M.fv_range(fh)) handles.emplace_back(fh, std::vector<VH>{vh});
        }
    }

    static std::unordered_set<ElementIndex, MeshElementIndexHash> AllHighlights;
    AllHighlights.clear();
    AllHighlights.insert(HighlightedElements.begin(), HighlightedElements.end());
    AllHighlights.emplace(highlight);

    std::vector<Vertex3D> vertices;
    for (const auto &handle : handles) {
        const auto &parent = handle.Parent;
        const auto normal = ToGlm(render_element == MeshElement::Vertex || render_element == MeshElement::Edge ? M.normal(handle.VHs[0]) : M.normal(FH(handle.Parent)));
        for (const auto vh : handle.VHs) {
            const bool is_highlighted =
                // todo different colors for persistent/selection highlights (`HighlightedElements` vs `highlight`)
                //   - actually, best approach may be to add a boolean `highlight` to `Vertex3D` and let the shader handle the color.
                //   - this would enable very fast highlight color ubo updates, which we'll need for fading alpha representing strike response intensity.
                AllHighlights.contains(vh) || AllHighlights.contains(parent) ||
                // Note: If we want to support `HighlightedElements` having `MeshElement::Edge` or `MeshElement::Face` elements (not just the selection `highlight`),
                // we'd need to update the methods to accept sets of `ElementIndex` instead of just one.
                (render_element == MeshElement::Vertex && (VertexBelongsToFace(parent, highlight) || VertexBelongsToEdge(parent, highlight))) ||
                (render_element == MeshElement::Edge && EdgeBelongsToFace(parent, highlight)) ||
                (render_element == MeshElement::Face && VertexBelongsToFaceEdge(vh, parent, highlight));
            const vec4 color = is_highlighted         ? HighlightColor :
                render_element == MeshElement::Vertex ? VertexColor :
                render_element == MeshElement::Edge   ? EdgeColor :
                                                        ToGlm(M.color(FH(parent)));
            vertices.emplace_back(GetPosition(vh), normal, color);
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
        const auto v = ToGlm(M.point(vh));
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
