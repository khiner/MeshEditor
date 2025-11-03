#include "Mesh.h"

#include "BVH.h"
#include "numeric/ray.h"

#include <algorithm>
#include <ranges>

using namespace he;

using std::ranges::any_of;

Mesh::Mesh(PolyMesh &&m) : M(std::move(m)) {
    SetFaceColor(DefaultFaceColor);
    BoundingBox = ComputeBbox();
    Bvh = std::make_unique<BVH>(CreateFaceBoundingBoxes());
}
Mesh::Mesh(std::vector<vec3> &&vertices, std::vector<std::vector<uint>> &&faces, vec4 color)
    : M(std::move(vertices), std::move(faces)) {
    if (color != DefaultFaceColor) SetFaceColor(color);
    BoundingBox = ComputeBbox();
    Bvh = std::make_unique<BVH>(CreateFaceBoundingBoxes());
}
Mesh::Mesh(Mesh &&other)
    : BoundingBox(other.BoundingBox), M(std::move(other.M)), Bvh(std::move(other.Bvh)), HighlightedHandles(std::move(other.HighlightedHandles)) {
    other.Bvh.reset();
}
Mesh::Mesh(const Mesh &other)
    : BoundingBox(other.BoundingBox), M(other.M), HighlightedHandles(other.HighlightedHandles) {
    Bvh = std::make_unique<BVH>(CreateFaceBoundingBoxes());
}
Mesh::~Mesh() {}

// todo this was here for entt, but something tells me it doesn't need this anymore
const Mesh &Mesh::operator=(Mesh &&other) {
    if (this != &other) {
        BoundingBox = std::move(other.BoundingBox);
        M = std::move(other.M);
        Bvh = std::move(other.Bvh);
        HighlightedHandles = std::move(other.HighlightedHandles);
    }
    return *this;
}

namespace {
constexpr float SquaredDistanceToLineSegment(const vec3 &v1, const vec3 &v2, const vec3 &p) noexcept {
    const vec3 edge = v2 - v1;
    const float t = glm::clamp(glm::dot(p - v1, edge) / glm::dot(edge, edge), 0.f, 1.f);
    const vec3 closest_p = v1 + t * edge;
    const vec3 diff = p - closest_p;
    return glm::dot(diff, diff);
}

// Moller-Trumbore ray-triangle intersection algorithm.
// Returns true if the ray (starting at `o` and traveling along `d`) intersects the given triangle.
// If ray intersects, returns the distance along the ray to the intersection point.
constexpr std::optional<float> IntersectTriangle(vec3 o, vec3 d, vec3 p1, vec3 p2, vec3 p3) noexcept {
    static constexpr float eps = 1e-7f; // Floating point error tolerance.

    const vec3 e1 = p2 - p1, e2 = p3 - p1;
    const vec3 h = glm::cross(d, e2);
    const float a = glm::dot(e1, h); // Barycentric coordinate
    if (a > -eps && a < eps) return {}; // Check if the ray is parallel to the triangle.

    // Check if the intersection point is inside the triangle (in barycentric coordinates).
    const vec3 s = o - p1;
    const float f = 1.f / a, u = f * glm::dot(s, h);
    if (u < 0.f || u > 1.f) return {};

    const vec3 q = glm::cross(s, e1);
    if (float v = f * glm::dot(d, q); v < 0.f || u + v > 1.f) return {};

    // Calculate the intersection point's distance along the ray and verify it's ahead of the ray's origin.
    if (float distance = f * glm::dot(e2, q); distance > eps) return {distance};
    return {};
}

constexpr std::optional<float> IntersectFace(const ray &ray, uint fi, const void *m) noexcept {
    const vec3 o = ray.o, d = ray.d;
    const auto &pm = *reinterpret_cast<const PolyMesh *>(m);
    auto fv_it = pm.cfv_iter(FH(fi));
    const VH v0 = *fv_it++;
    VH v1 = *fv_it++, v2;
    for (; fv_it; ++fv_it) {
        v2 = *fv_it;
        if (auto distance = IntersectTriangle(o, d, pm.GetPosition(v0), pm.GetPosition(v1), pm.GetPosition(v2))) return {distance};
        v1 = v2;
    }
    return {};
}

// Used as an intermediate for creating render vertices
struct VerticesHandle {
    he::AnyHandle Parent; // A vertex can belong to itself, an edge, or a face.
    std::vector<Mesh::VH> VHs;
};

PolyMesh DeduplicateVertices(const PolyMesh &m) {
    struct VertexHash {
        constexpr size_t operator()(const vec3 &p) const noexcept {
            return std::hash<float>{}(p[0]) ^ std::hash<float>{}(p[1]) ^ std::hash<float>{}(p[2]);
        }
    };

    std::vector<vec3> vertices;
    std::unordered_map<vec3, uint, VertexHash> unique_vertices;
    for (const auto vh : m.vertices()) {
        const auto p = m.GetPosition(vh);
        if (auto [it, inserted] = unique_vertices.try_emplace(p, vertices.size()); inserted) {
            vertices.emplace_back(p);
        }
    }

    std::vector<std::vector<uint>> faces;
    faces.reserve(m.FaceCount());
    for (const auto &fh : m.faces()) {
        std::vector<uint> new_face;
        new_face.reserve(m.GetValence(fh));
        for (const auto &vh : m.fv_range(fh)) new_face.emplace_back(unique_vertices.at(m.GetPosition(vh)));
        faces.emplace_back(std::move(new_face));
    }

    return PolyMesh(std::move(vertices), std::move(faces));
}
} // namespace

std::optional<PolyMesh> LoadPolyMesh(const fs::path &path) {
    auto mesh = he::ReadMesh(path);
    if (!mesh) return {};
    // Deduplicate even if not strictly triangle soup. Assumes this is a surface mesh.
    return DeduplicateVertices(*mesh);
}

bool Mesh::VertexBelongsToFace(VH vh, FH fh) const {
    return vh && fh && any_of(M.fv_range(fh), [&](const auto &vh_o) { return vh_o == vh; });
}

bool Mesh::VertexBelongsToEdge(VH vh, EH eh) const {
    return vh && eh && any_of(M.voh_range(vh), [&](const auto &heh) { return M.GetEdge(heh) == eh; });
}

bool Mesh::VertexBelongsToFaceEdge(VH vh, FH fh, EH eh) const {
    return fh && eh && any_of(M.voh_range(vh), [&](const auto &heh) {
               return M.GetEdge(heh) == eh && (M.GetFace(heh) == fh || M.GetFace(M.GetOppositeHalfedge(heh)) == fh);
           });
}

bool Mesh::EdgeBelongsToFace(EH eh, FH fh) const {
    return eh && fh && any_of(M.fh_range(fh), [&](const auto &heh) { return M.GetEdge(heh) == eh; });
}

float Mesh::CalcFaceArea(FH fh) const {
    float area{0};
    auto fv_it = M.cfv_iter(fh);
    const auto p0 = M.GetPosition(*fv_it++);
    for (vec3 p1 = M.GetPosition(*fv_it++), p2; fv_it; ++fv_it) {
        p2 = M.GetPosition(*fv_it);
        const auto cross_product = glm::cross(p1 - p0, p2 - p0);
        area += glm::length(cross_product) * 0.5f;
        p1 = p2;
    }
    return area;
}

std::vector<BBox> Mesh::CreateFaceBoundingBoxes() const {
    std::vector<BBox> boxes;
    boxes.reserve(M.FaceCount());
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

FH Mesh::FindNearestIntersectingFace(const ray &ray, vec3 *nearest_intersect_point_out) const {
    if (auto intersection = Bvh->IntersectNearest(ray, IntersectFace, &M)) {
        if (nearest_intersect_point_out) *nearest_intersect_point_out = ray(intersection->Distance);
        return FH(intersection->Index);
    }
    return FH{};
}

std::optional<Intersection> Mesh::Intersect(const ray &ray) const {
    return Bvh->IntersectNearest(ray, IntersectFace, &M);
}

VH Mesh::FindNearestVertex(vec3 p) const {
    VH closest_vertex;
    float min_distance_sq = std::numeric_limits<float>::max();
    for (const auto vh : M.vertices()) {
        const vec3 diff = M.GetPosition(vh) - p;
        if (const float distance_sq = glm::dot(diff, diff); distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }
    return closest_vertex;
}

VH Mesh::FindNearestVertex(const ray &local_ray) const {
    vec3 intersection_p;
    const auto face = FindNearestIntersectingFace(local_ray, &intersection_p);
    if (!face) return {};

    VH closest_vertex{};
    float min_distance_sq = std::numeric_limits<float>::max();
    for (const auto vh : M.fv_range(face)) {
        const vec3 diff = M.GetPosition(vh) - intersection_p;
        if (float distance_sq = glm::dot(diff, diff); distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }

    return closest_vertex;
}

EH Mesh::FindNearestEdge(const ray &local_ray) const {
    vec3 intersection_p;
    const auto face = FindNearestIntersectingFace(local_ray, &intersection_p);
    if (!face) return {};

    Mesh::EH closest_edge;
    float min_distance_sq = std::numeric_limits<float>::max();
    for (auto heh : M.fh_range(face)) {
        const auto eh = M.GetEdge(heh);
        const auto heh0 = M.GetHalfedge(eh, 0);
        const auto p1 = M.GetPosition(M.GetFromVertex(heh0));
        const auto p2 = M.GetPosition(M.GetToVertex(heh0));
        if (float distance_sq = SquaredDistanceToLineSegment(p1, p2, intersection_p);
            distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_edge = eh;
        }
    }

    return closest_edge;
}

std::vector<uint> Mesh::CreateIndices(he::Element element) const {
    switch (element) {
        case he::Element::Vertex: return CreateTriangleIndices();
        case he::Element::Edge: return CreateEdgeIndices();
        case he::Element::Face: return CreateTriangulatedFaceIndices();
        case he::Element::None: return {};
    }
}
std::vector<uint> Mesh::CreateNormalIndices(he::Element element) const {
    if (element == he::Element::None || element == he::Element::Edge) return {};

    const auto n = element == he::Element::Face ? M.FaceCount() : M.VertexCount();
    std::vector<uint> indices;
    indices.reserve(n * 2);
    for (uint i = 0; i < n; ++i) {
        indices.push_back(i * 2);
        indices.push_back(i * 2 + 1);
    }
    return indices;
}

std::vector<Vertex3D> Mesh::CreateVertices(he::Element render_element, const he::AnyHandle &selected) const {
    std::vector<VerticesHandle> handles;
    if (render_element == he::Element::Vertex) {
        handles.reserve(M.VertexCount());
        for (const auto vh : M.vertices()) handles.emplace_back(vh, std::vector<VH>{vh});
    } else if (render_element == he::Element::Edge) {
        handles.reserve(M.EdgeCount() * 2);
        for (const auto eh : M.edges()) {
            const auto heh = M.GetHalfedge(eh, 0);
            handles.emplace_back(eh, std::vector<VH>{M.GetFromVertex(heh), M.GetToVertex(heh)});
        }
    } else if (render_element == he::Element::Face) {
        handles.reserve(M.FaceCount() * 3); // Lower bound assuming all faces are triangles.
        for (const auto fh : M.faces()) {
            for (const auto vh : M.fv_range(fh)) handles.emplace_back(fh, std::vector<VH>{vh});
        }
    }

    std::vector<Vertex3D> vertices;
    for (const auto &handle : handles) {
        const auto parent = handle.Parent;
        const auto normal = render_element == he::Element::Vertex || render_element == he::Element::Edge ? M.GetNormal(handle.VHs[0]) : M.GetNormal(FH(handle.Parent));
        for (const auto vh : handle.VHs) {
            const bool is_selected =
                selected == vh || selected == parent ||
                // Note: If we want to support `HighlightedHandles` having handle types (not just the selection `highlight`),
                // we need to update the methods to accept sets of `AnyHandle` instead of just one.
                (render_element == he::Element::Vertex && (VertexBelongsToFace(parent, selected) || VertexBelongsToEdge(parent, selected))) ||
                (render_element == he::Element::Edge && EdgeBelongsToFace(parent, selected)) ||
                (render_element == he::Element::Face && VertexBelongsToFaceEdge(vh, parent, selected));
            const bool is_highlighted =
                HighlightedHandles.contains(vh) || HighlightedHandles.contains(parent);
            const auto color = is_selected         ? SelectedColor :
                is_highlighted                     ? HighlightedColor :
                render_element == he::Element::Vertex ? VertexColor :
                render_element == he::Element::Edge   ? EdgeColor :
                                                     M.GetColor(FH(parent));
            vertices.emplace_back(GetPosition(vh), normal, color);
        }
    }

    return vertices;
}

std::vector<Vertex3D> Mesh::CreateNormalVertices(he::Element element) const {
    std::vector<Vertex3D> vertices;
    if (element == he::Element::Vertex) {
        // Line for each vertex normal, with length scaled by the average edge length.
        vertices.reserve(M.VertexCount() * 2);
        for (const auto vh : M.vertices()) {
            const auto vn = GetVertexNormal(vh);
            const auto &voh_range = M.voh_range(vh);
            const float total_edge_length = std::reduce(voh_range.begin(), voh_range.end(), 0.f, [&](float total, const auto &heh) {
                return total + M.CalcEdgeLength(heh);
            });
            const float avg_edge_length = total_edge_length / M.GetValence(vh);
            const auto p = GetPosition(vh);
            vertices.emplace_back(p, vn, VertexNormalIndicatorColor);
            vertices.emplace_back(p + NormalIndicatorLengthScale * avg_edge_length * vn, vn, VertexNormalIndicatorColor);
        }
    } else if (element == he::Element::Face) {
        // Line for each face normal, with length scaled by the face area.
        vertices.reserve(M.FaceCount() * 2);
        for (const auto fh : M.faces()) {
            const auto fn = GetFaceNormal(fh);
            const auto p = M.CalcFaceCentroid(fh);
            vertices.emplace_back(p, fn, FaceNormalIndicatorColor);
            vertices.emplace_back(p + NormalIndicatorLengthScale * std::sqrt(CalcFaceArea(fh)) * fn, fn, FaceNormalIndicatorColor);
        }
    }
    return vertices;
}

BBox Mesh::ComputeBbox() const {
    BBox bbox;
    for (const auto vh : M.vertices()) {
        const auto p = M.GetPosition(vh);
        bbox.Min = glm::min(bbox.Min, p);
        bbox.Max = glm::max(bbox.Max, p);
    }
    return bbox;
}

std::vector<uint> Mesh::CreateTriangleIndices() const {
    std::vector<uint> indices;
    for (const auto fh : M.faces()) {
        auto fv_it = M.cfv_iter(fh);
        const auto v0 = *fv_it++;
        VH v1 = *fv_it++, v2;
        for (; fv_it; ++fv_it) {
            v2 = *fv_it;
            indices.insert(indices.end(), {uint(*v0), uint(*v1), uint(*v2)});
            v1 = v2;
        }
    }
    return indices;
}

std::vector<uint> Mesh::CreateTriangulatedFaceIndices() const {
    std::vector<uint> indices;
    uint index = 0;
    for (const auto fh : M.faces()) {
        const auto valence = M.GetValence(fh);
        for (uint i = 0; i < valence - 2; ++i) {
            indices.insert(indices.end(), {index, index + i + 1, index + i + 2});
        }
        index += valence;
    }
    return indices;
}

std::vector<uint> Mesh::CreateEdgeIndices() const {
    std::vector<uint> indices;
    indices.reserve(M.EdgeCount() * 2);
    for (uint ei = 0; ei < M.EdgeCount(); ++ei) {
        indices.push_back(2 * ei);
        indices.push_back(2 * ei + 1);
    }
    return indices;
}
