#include "Mesh.h"

#include "BVH.h"
#include "numeric/ray.h"

#include <algorithm>
#include <ranges>

using namespace om;

using std::ranges::any_of;

Mesh::Mesh(PolyMesh &&m) : M(std::move(m)) {
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
Mesh::Mesh(const Mesh &other)
    : BoundingBox(other.BoundingBox), M(other.M), HighlightedElements(other.HighlightedElements) {
    Bvh = std::make_unique<BVH>(CreateFaceBoundingBoxes());
}
Mesh::~Mesh() {
    M.release_vertex_normals();
    M.release_face_normals();
    M.release_face_colors();
}

// todo this was here for entt, but something tells me it doesn't need this anymore
const Mesh &Mesh::operator=(Mesh &&other) {
    if (this != &other) {
        BoundingBox = std::move(other.BoundingBox);
        M = std::move(other.M);
        Bvh = std::move(other.Bvh);
        HighlightedElements = std::move(other.HighlightedElements);
    }
    return *this;
}

namespace {
constexpr float SquaredDistanceToLineSegment(const Point &v1, const Point &v2, const Point &p) noexcept {
    const auto edge = v2 - v1;
    const float t = glm::clamp((p - v1).dot(edge) / edge.dot(edge), 0.f, 1.f);
    const auto closest_p = v1 + t * edge;
    const auto diff = p - closest_p;
    return diff.dot(diff);
}

// Moller-Trumbore ray-triangle intersection algorithm.
// Returns true if the ray (starting at `o` and traveling along `d`) intersects the given triangle.
// If ray intersects, returns the distance along the ray to the intersection point.
constexpr std::optional<float> IntersectTriangle(Point o, Point d, Point p1, Point p2, Point p3) noexcept {
    static constexpr float eps = 1e-7f; // Floating point error tolerance.

    const Point e1 = p2 - p1, e2 = p3 - p1;
    const Point h = d % e2;
    const float a = e1.dot(h); // Barycentric coordinate
    if (a > -eps && a < eps) return {}; // Check if the ray is parallel to the triangle.

    // Check if the intersection point is inside the triangle (in barycentric coordinates).
    const Point s = o - p1;
    const float f = 1.f / a, u = f * s.dot(h);
    if (u < 0.f || u > 1.f) return {};

    const Point q = s % e1;
    if (float v = f * d.dot(q); v < 0.f || u + v > 1.f) return {};

    // Calculate the intersection point's distance along the ray and verify it's ahead of the ray's origin.
    if (float distance = f * e2.dot(q); distance > eps) return {distance};
    return {};
}

constexpr std::optional<float> IntersectFace(const ray &ray, uint fi, const void *m) noexcept {
    const Point o = ToOpenMesh(ray.o), d = ToOpenMesh(ray.d);
    const auto &pm = *reinterpret_cast<const PolyMesh *>(m);
    auto fv_it = pm.cfv_iter(FH(fi));
    const VH v0 = *fv_it++;
    VH v1 = *fv_it++, v2;
    for (; fv_it.is_valid(); ++fv_it) {
        v2 = *fv_it;
        if (auto distance = IntersectTriangle(o, d, pm.point(v0), pm.point(v1), pm.point(v2))) return {distance};
        v1 = v2;
    }
    return {};
}

struct VertexHash {
    constexpr size_t operator()(const Point &p) const noexcept {
        return std::hash<float>{}(p[0]) ^ std::hash<float>{}(p[1]) ^ std::hash<float>{}(p[2]);
    }
};

// Used as an intermediate for creating render vertices (`Vertex3D`).
struct VerticesHandle {
    Mesh::ElementIndex Parent; // A vertex can belong to itself, an edge, or a face.
    std::vector<Mesh::VH> VHs;
};

Mesh::PolyMesh DeduplicateVertices(const Mesh::PolyMesh &m) {
    PolyMesh deduped;

    std::unordered_map<Point, VH, VertexHash> unique_vertices;
    for (auto v_it = m.vertices_begin(); v_it != m.vertices_end(); ++v_it) {
        const auto p = m.point(*v_it);
        if (auto [it, inserted] = unique_vertices.try_emplace(p, VH()); inserted) {
            it->second = deduped.add_vertex(p);
        }
    }
    // Add faces.
    for (const auto &fh : m.faces()) {
        std::vector<VH> new_face;
        new_face.reserve(m.valence(fh));
        for (const auto &vh : m.fv_range(fh)) new_face.emplace_back(unique_vertices.at(m.point(vh)));
        deduped.add_face(new_face);
    }
    return deduped;
}
} // namespace

std::optional<Mesh::PolyMesh> LoadPolyMesh(const fs::path &file_path) {
    static Mesh::PolyMesh mesh;
    OpenMesh::IO::Options read_options; // No options used yet.
    if (!OpenMesh::IO::read_mesh(mesh, file_path.string(), read_options)) {
        return {};
    }
    // if (IsTriangleSoup()) M = DeduplicateVertices();
    // Deduplicate even if not strictly triangle soup. Assumes this is a surface mesh.
    return DeduplicateVertices(mesh);
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
    for (auto fh : M.faces()) {
        BBox box;
        for (auto vh : M.fv_range(fh)) {
            const auto p = GetPosition(vh);
            box.Min = glm::min(box.Min, p);
            box.Max = glm::max(box.Max, p);
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

FH Mesh::FindNearestIntersectingFace(const ray &ray, Point *nearest_intersect_point_out) const {
    if (auto intersection = Bvh->IntersectNearest(ray, IntersectFace, &M)) {
        if (nearest_intersect_point_out) *nearest_intersect_point_out = ToOpenMesh(ray(intersection->Distance));
        return FH(intersection->Index);
    }
    return FH{};
}

std::optional<Intersection> Mesh::Intersect(const ray &ray) const {
    return Bvh->IntersectNearest(ray, IntersectFace, &M);
}

VH Mesh::FindNearestVertex(Point p) const {
    VH closest_vertex;
    float min_distance_sq = std::numeric_limits<float>::max();
    for (auto vh : M.vertices()) {
        auto diff = M.point(vh) - p;
        if (const float distance_sq = diff.dot(diff); distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }
    return closest_vertex;
}

VH Mesh::FindNearestVertex(const ray &local_ray) const {
    Point intersection_point;
    const auto face = FindNearestIntersectingFace(local_ray, &intersection_point);
    if (!face.is_valid()) return VH{};

    VH closest_vertex{};
    float min_distance_sq = std::numeric_limits<float>::max();
    for (auto vh : M.fv_range(face)) {
        const auto diff = M.point(vh) - intersection_point;
        if (float distance_sq = diff.dot(diff); distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }

    return closest_vertex;
}

EH Mesh::FindNearestEdge(const ray &local_ray) const {
    Point intersection_point;
    const auto face = FindNearestIntersectingFace(local_ray, &intersection_point);
    if (!face.is_valid()) return Mesh::EH{};

    Mesh::EH closest_edge;
    float min_distance_sq = std::numeric_limits<float>::max();
    for (auto heh : M.fh_range(face)) {
        const auto eh = M.edge_handle(heh);
        const auto p1 = M.point(M.from_vertex_handle(M.halfedge_handle(eh, 0)));
        const auto p2 = M.point(M.to_vertex_handle(M.halfedge_handle(eh, 0)));
        if (float distance_sq = SquaredDistanceToLineSegment(p1, p2, intersection_point);
            distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_edge = eh;
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

std::vector<Vertex3D> Mesh::CreateVertices(MeshElement render_element, const ElementIndex &selected) const {
    std::vector<VerticesHandle> handles;
    if (render_element == MeshElement::Vertex) {
        handles.reserve(M.n_vertices());
        for (auto vh : M.vertices()) handles.emplace_back(vh, std::vector<VH>{vh});
    } else if (render_element == MeshElement::Edge) {
        handles.reserve(M.n_edges() * 2);
        for (auto eh : M.edges()) {
            auto heh = M.halfedge_handle(eh, 0);
            handles.emplace_back(eh, std::vector<VH>{M.from_vertex_handle(heh), M.to_vertex_handle(heh)});
        }
    } else if (render_element == MeshElement::Face) {
        handles.reserve(M.n_faces() * 3); // Lower bound assuming all faces are triangles.
        for (auto fh : M.faces()) {
            for (auto vh : M.fv_range(fh)) handles.emplace_back(fh, std::vector<VH>{vh});
        }
    }

    std::vector<Vertex3D> vertices;
    for (const auto &handle : handles) {
        const auto parent = handle.Parent;
        const auto normal = ToGlm(render_element == MeshElement::Vertex || render_element == MeshElement::Edge ? M.normal(handle.VHs[0]) : M.normal(FH(handle.Parent)));
        for (const auto vh : handle.VHs) {
            const bool is_selected =
                (selected == vh || selected == parent) ||
                // Note: If we want to support `HighlightedElements` having `MeshElement::Edge` or `MeshElement::Face` elements (not just the selection `highlight`),
                // we need to update the methods to accept sets of `ElementIndex` instead of just one.
                (render_element == MeshElement::Vertex && (vh.idx() == selected.idx() || VertexBelongsToFace(parent, selected) || VertexBelongsToEdge(parent, selected))) ||
                (render_element == MeshElement::Edge && EdgeBelongsToFace(parent, selected)) ||
                (render_element == MeshElement::Face && VertexBelongsToFaceEdge(vh, parent, selected));
            const bool is_highlighted =
                HighlightedElements.contains(vh) || HighlightedElements.contains(parent);
            const vec4 color = is_selected            ? SelectedColor :
                is_highlighted                        ? HighlightedColor :
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
        for (auto vh : M.vertices()) {
            const auto vn = GetVertexNormal(vh);
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
        for (auto fh : M.faces()) {
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
    for (auto vh : M.vertices()) {
        const auto v = ToGlm(M.point(vh));
        bbox.Min = glm::min(bbox.Min, v);
        bbox.Max = glm::max(bbox.Max, v);
    }
    return bbox;
}

std::vector<uint> Mesh::CreateTriangleIndices() const {
    std::vector<uint> indices;
    for (auto fh : M.faces()) {
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
    for (auto fh : M.faces()) {
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
