#include "Geometry.h"

#include <ranges>

using glm::vec3, glm::vec4;

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
    static const vec4 HighlightColor{1, 0, 0, 1};

    std::vector<Vertex3D> vertices;
    Mesh.update_normals();

    if (mode == GeometryMode::Faces) {
        vertices.reserve(Mesh.n_faces() * 3); // At least 3 vertices per face.
        for (const auto &fh : Mesh.faces()) {
            const auto &fn = Mesh.normal(fh);
            const auto &fc = Mesh.color(fh);
            for (const auto &vh : Mesh.fv_range(fh)) {
                const vec4 color = vh == highlighted_vertex || fh == highlighted_face || DoesVertexBelongToFaceEdge(vh, fh, highlighted_edge) ? HighlightColor : vec4{fc[0], fc[1], fc[2], 1};
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

std::vector<uint> Geometry::GenerateLineIndices() const {
    std::vector<uint> indices;
    indices.reserve(Mesh.n_edges() * 2);
    for (uint edge_i = 0; edge_i < Mesh.n_edges(); ++edge_i) {
        indices.push_back(edge_i * 2);
        indices.push_back(edge_i * 2 + 1);
    }
    return indices;
}
