#include "Geometry.h"

#include "World.h"

#include <algorithm>
#include <ranges>

using glm::vec3, glm::vec4, glm::mat4;

using std::ranges::any_of;

bool Geometry::Load(const fs::path &file_path) {
    OpenMesh::IO::Options read_options; // No options used yet, but keeping this here for future use.
    if (!OpenMesh::IO::read_mesh(Mesh, file_path.string(), read_options)) {
        std::cerr << "Error loading mesh: " << file_path << std::endl;
        return false;
    }
    return true;
}

bool Geometry::DoesVertexBelongToFace(VH vertex, FH face) const {
    return face.is_valid() && any_of(Mesh.fv_range(face), [&](const auto &vh) { return vh == vertex; });
}

bool Geometry::DoesVertexBelongToEdge(VH vertex, EH edge) const {
    return edge.is_valid() && any_of(Mesh.voh_range(vertex), [&](const auto &heh) { return Mesh.edge_handle(heh) == edge; });
}

bool Geometry::DoesVertexBelongToFaceEdge(VH vertex, FH face, EH edge) const {
    return face.is_valid() && edge.is_valid() &&
        any_of(Mesh.voh_range(vertex), [&](const auto &heh) {
               return Mesh.edge_handle(heh) == edge && (Mesh.face_handle(heh) == face || Mesh.face_handle(Mesh.opposite_halfedge_handle(heh)) == face);
           });
}

bool Geometry::DoesEdgeBelongToFace(EH edge, FH face) const {
    return face.is_valid() && any_of(Mesh.fh_range(face), [&](const auto &heh) { return Mesh.edge_handle(heh) == edge; });
}

std::vector<Vertex3D> Geometry::GenerateVertices(GeometryMode mode, FH highlighted_face, VH highlighted_vertex, EH highlighted_edge) {
    Mesh.update_normals(); // todo only update when necessary.

    std::vector<Vertex3D> vertices;
    if (mode == GeometryMode::Faces) {
        vertices.reserve(Mesh.n_faces() * 3); // At least 3 vertices per face.
        for (const auto &fh : Mesh.faces()) {
            const auto &fn = Mesh.normal(fh);
            const auto &fc = Mesh.color(fh);
            for (const auto &vh : Mesh.fv_range(fh)) {
                const vec4 color = vh == highlighted_vertex || fh == highlighted_face || DoesVertexBelongToFaceEdge(vh, fh, highlighted_edge) ? HighlightColor : ToGlm(fc);
                vertices.emplace_back(GetPosition(vh), ToGlm(fn), color);
            }
        }
    } else if (mode == GeometryMode::Vertices) {
        vertices.reserve(Mesh.n_vertices());
        for (const auto &vh : Mesh.vertices()) {
            const vec4 color = vh == highlighted_vertex || DoesVertexBelongToFace(vh, highlighted_face) || DoesVertexBelongToEdge(vh, highlighted_edge) ? HighlightColor : vec4{1};
            vertices.emplace_back(GetPosition(vh), GetVertexNormal(vh), color);
        }
    } else if (mode == GeometryMode::Edges) {
        vertices.reserve(Mesh.n_edges() * 2);
        for (const auto &eh : Mesh.edges()) {
            const auto &heh = Mesh.halfedge_handle(eh, 0);
            const auto &vh0 = Mesh.from_vertex_handle(heh);
            const auto &vh1 = Mesh.to_vertex_handle(heh);
            const vec4 color = eh == highlighted_edge || vh0 == highlighted_vertex || vh1 == highlighted_vertex || DoesEdgeBelongToFace(eh, highlighted_face) ? HighlightColor : EdgeColor;
            vertices.emplace_back(GetPosition(vh0), GetVertexNormal(vh0), color);
            vertices.emplace_back(GetPosition(vh1), GetVertexNormal(vh1), color);
        }
    }

    return vertices;
}

static float CalcFaceArea(const MeshType &mesh, MeshType::FaceHandle fh) {
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

std::vector<Vertex3D> Geometry::GenerateVertices(NormalIndicatorMode mode) {
    Mesh.update_normals(); // todo only update when necessary.

    std::vector<Vertex3D> vertices;
    if (mode == NormalIndicatorMode::Faces) {
        // Line for each face normal, with length scaled by the face area.
        vertices.reserve(Mesh.n_faces() * 2);
        for (const auto &fh : Mesh.faces()) {
            const auto &fn = Mesh.normal(fh);
            const vec3 point = ToGlm(Mesh.calc_face_centroid(fh));
            vertices.emplace_back(point, ToGlm(fn), FaceNormalIndicatorColor);
            vertices.emplace_back(point + NormalIndicatorLengthScale * CalcFaceArea(Mesh, fh) * ToGlm(fn), ToGlm(fn), FaceNormalIndicatorColor);
        }
    } else if (mode == NormalIndicatorMode::Vertices) {
        // Line for each vertex normal, with length scaled by the average edge length.
        vertices.reserve(Mesh.n_vertices() * 2);
        for (const auto &vh : Mesh.vertices()) {
            const auto &vn = Mesh.normal(vh);
            const auto &voh_range = Mesh.voh_range(vh);
            const float total_edge_length = std::reduce(voh_range.begin(), voh_range.end(), 0.f, [&](float total, const auto &heh) {
                return total + Mesh.calc_edge_length(heh);
            });
            const float avg_edge_length = total_edge_length / Mesh.valence(vh);
            const vec3 point = ToGlm(Mesh.point(vh));
            vertices.emplace_back(point, ToGlm(vn), VertexNormalIndicatorColor);
            vertices.emplace_back(point + NormalIndicatorLengthScale * avg_edge_length * ToGlm(vn), ToGlm(vn), VertexNormalIndicatorColor);
        }
    }

    return vertices;
}

// [{min_x, min_y, min_z}, {max_x, max_y, max_z}]
std::pair<vec3, vec3> Geometry::ComputeBounds() const {
    static const float min_float = std::numeric_limits<float>::lowest();
    static const float max_float = std::numeric_limits<float>::max();

    vec3 min(max_float), max(min_float);
    for (const auto &vh : Mesh.vertices()) {
        const auto &p = Mesh.point(vh);
        min.x = std::min(min.x, p[0]);
        min.y = std::min(min.y, p[1]);
        min.z = std::min(min.z, p[2]);
        max.x = std::max(max.x, p[0]);
        max.y = std::max(max.y, p[1]);
        max.z = std::max(max.z, p[2]);
    }

    return {min, max};
}

std::vector<uint> Geometry::GenerateTriangleIndices() const {
    auto triangulated_mesh = Mesh; // `triangulate` is in-place, so we need to make a copy.
    triangulated_mesh.triangulate();
    std::vector<uint> indices;
    for (const auto &fh : triangulated_mesh.faces()) {
        auto v_it = triangulated_mesh.cfv_iter(fh);
        indices.insert(indices.end(), {uint(v_it->idx()), uint((++v_it)->idx()), uint((++v_it)->idx())});
    }
    return indices;
}

std::vector<uint> Geometry::GenerateTriangulatedFaceIndices() const {
    std::vector<uint> indices;
    uint index = 0;
    for (const auto &fh : Mesh.faces()) {
        const auto valence = Mesh.valence(fh);
        for (uint i = 0; i < valence - 2; ++i) {
            indices.insert(indices.end(), {index, index + i + 1, index + i + 2});
        }
        index += valence;
    }
    return indices;
}

Geometry::FH Geometry::TriangulatedIndexToFace(uint triangle_index) const {
    for (const auto &fh : Mesh.faces()) {
        const auto valence = Mesh.valence(fh);
        if (triangle_index < valence - 2) return fh;
        triangle_index -= valence - 2;
    }
    throw std::runtime_error("Invalid triangle index: " + std::to_string(triangle_index));
}

std::vector<uint> Geometry::GenerateEdgeIndices() const {
    std::vector<uint> indices;
    indices.reserve(Mesh.n_edges() * 2);
    for (uint ei = 0; ei < Mesh.n_edges(); ++ei) {
        indices.push_back(ei * 2);
        indices.push_back(ei * 2 + 1);
    }
    return indices;
}

std::vector<uint> Geometry::GenerateFaceNormalIndicatorIndices() const {
    std::vector<uint> indices;
    indices.reserve(Mesh.n_faces() * 2);
    for (uint fi = 0; fi < Mesh.n_faces(); ++fi) {
        indices.push_back(fi * 2);
        indices.push_back(fi * 2 + 1);
    }
    return indices;
}

std::vector<uint> Geometry::GenerateVertexNormalIndicatorIndices() const {
    std::vector<uint> indices;
    indices.reserve(Mesh.n_vertices() * 2);
    for (uint vi = 0; vi < Mesh.n_vertices(); ++vi) {
        indices.push_back(vi * 2);
        indices.push_back(vi * 2 + 1);
    }
    return indices;
}
