#include "HalfEdge.h"

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include <algorithm>
#include <fstream>

namespace he {

static uint64_t make_edge_key(int from, int to) {
    return (static_cast<uint64_t>(from) << 32) | static_cast<uint64_t>(to);
}

VH PolyMesh::AddVertex(const vec3 &p) {
    // Vertices.emplace_back(p, vec3{0}, HH{});
    Positions.emplace_back(p);
    Normals.emplace_back(vec3{0});
    OutgoingHalfedges.emplace_back(HH{});
    return VH(Positions.size() - 1);
}

FH PolyMesh::AddFace(const std::vector<VH> &vertices) {
    if (vertices.size() < 3) return {}; // Invalid face

    const auto fi = Faces.size();
    const auto start_he_i = Halfedges.size();
    // Create the face
    Faces.emplace_back(HH(start_he_i), vec3{0}, vec4{1, 1, 1, 1});

    // Create halfedges for this face
    for (size_t i = 0; i < vertices.size(); ++i) {
        const auto to_v = vertices[i];
        const auto from_v = vertices[i == 0 ? vertices.size() - 1 : i - 1];
        const HH hh(start_he_i + i);
        Halfedges.emplace_back(Halfedge{
            .Vertex = to_v,
            .Next = HH(start_he_i + (i + 1) % vertices.size()),
            .Opposite = {},
            .Edge = {},
            .Face = FH(fi),
        });

        if (!OutgoingHalfedges[from_v.idx()].is_valid()) OutgoingHalfedges[from_v.idx()] = hh;
        HalfedgeMap[make_edge_key(from_v.idx(), to_v.idx())] = hh;
    }

    // Find opposites and create edges
    for (size_t i = 0; i < vertices.size(); ++i) {
        const HH hh(start_he_i + i);
        const auto to_v = vertices[i];
        const auto from_v = vertices[i == 0 ? vertices.size() - 1 : i - 1];
        // Look for opposite halfedge in map
        if (const auto it = HalfedgeMap.find(make_edge_key(to_v.idx(), from_v.idx()));
            it != HalfedgeMap.end()) {
            // Found existing opposite halfedge
            const auto opposite_hh = it->second;
            Halfedges[hh.idx()].Opposite = opposite_hh;
            Halfedges[opposite_hh.idx()].Opposite = hh;
            // They should share the same edge
            Halfedges[hh.idx()].Edge = Halfedges[opposite_hh.idx()].Edge;
        } else {
            // Create new edge
            Edges.emplace_back(hh);
            Halfedges[hh.idx()].Edge = EH(Edges.size() - 1);
        }
    }

    return FH(fi);
}

HH PolyMesh::GetHalfedge(EH eh, uint i) const {
    if (!eh.is_valid() || eh.idx() >= static_cast<int>(Edges.size())) return {};
    const auto h0 = Edges[eh.idx()].Halfedge;
    return i == 0 ? h0 : (i == 1 && h0.is_valid() ? Halfedges[h0.idx()].Opposite : HH{});
}

HH PolyMesh::GetOppositeHalfedge(HH hh) const {
    if (!hh.is_valid() || hh.idx() >= static_cast<int>(Halfedges.size())) return {};
    return Halfedges[hh.idx()].Opposite;
}

VH PolyMesh::GetFromVertex(HH hh) const {
    if (!hh.is_valid() || hh.idx() >= static_cast<int>(Halfedges.size())) return {};

    // From-vertex is the to-vertex of the opposite halfedge
    const auto opp = Halfedges[hh.idx()].Opposite;
    return opp.is_valid() ? Halfedges[opp.idx()].Vertex : VH{};
}

uint PolyMesh::GetValence(VH vh) const {
    if (!vh.is_valid() || vh.idx() >= static_cast<int>(Positions.size())) return 0;

    const auto start = OutgoingHalfedges[vh.idx()];
    if (!start.is_valid()) return 0;

    uint count{0};
    auto current = start;
    do {
        count++;
        // Move to next outgoing halfedge
        const auto next_he = Halfedges[current.idx()].Next;
        if (!next_he.is_valid()) break;

        const auto opp = Halfedges[next_he.idx()].Opposite;
        if (!opp.is_valid()) break;

        current = opp;
    } while (current != start && current.is_valid());

    return count;
}

uint PolyMesh::GetValence(FH fh) const {
    if (!fh.is_valid() || fh.idx() >= static_cast<int>(Faces.size())) return 0;

    uint count{0};
    const auto start = Faces[fh.idx()].Halfedge;
    auto current = start;
    do {
        count++;
        current = Halfedges[current.idx()].Next;
    } while (current != start && current.is_valid());

    return count;
}

vec3 PolyMesh::CalcFaceCentroid(FH fh) const {
    if (!fh.is_valid() || fh.idx() >= static_cast<int>(Faces.size())) return vec3{0};

    vec3 centroid{0};
    uint count{0};
    const auto start = Faces[fh.idx()].Halfedge;
    auto current = start;
    do {
        const auto vh = Halfedges[current.idx()].Vertex;
        centroid += Positions[vh.idx()];
        count++;
        current = Halfedges[current.idx()].Next;
    } while (current != start && current.is_valid());

    if (count > 0) centroid /= static_cast<float>(count);
    return centroid;
}

float PolyMesh::CalcEdgeLength(HH hh) const {
    if (!hh.is_valid() || hh.idx() >= static_cast<int>(Halfedges.size())) return 0;

    const auto from_v = GetFromVertex(hh);
    const auto to_v = Halfedges[hh.idx()].Vertex;
    if (!from_v.is_valid() || !to_v.is_valid()) return 0;
    return glm::length(Positions[to_v.idx()] - Positions[from_v.idx()]);
}

PolyMesh::VertexOutgoingHalfedgeRange PolyMesh::voh_range(VH vh) const {
    return {this, vh.is_valid() && vh.idx() < static_cast<int>(OutgoingHalfedges.size()) ? OutgoingHalfedges[vh.idx()] : HH{}};
}

void PolyMesh::UpdateNormals() {
    ComputeFaceNormals();
    ComputeVertexNormals();
}

void PolyMesh::ComputeFaceNormals() {
    for (uint fi = 0; fi < FaceCount(); ++fi) {
        const auto start = Faces[fi].Halfedge;
        // Get first three vertices for normal calculation
        const auto h0 = start, h1 = Halfedges[h0.idx()].Next, h2 = Halfedges[h1.idx()].Next;
        const auto p0 = Positions[Halfedges[h0.idx()].Vertex.idx()];
        const auto p1 = Positions[Halfedges[h1.idx()].Vertex.idx()];
        const auto p2 = Positions[Halfedges[h2.idx()].Vertex.idx()];
        Faces[fi].Normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    }
}

void PolyMesh::ComputeVertexNormals() {
    // Reset all vertex normals
    for (auto &n : Normals) n = vec3{0};

    // Accumulate face normals to vertices
    for (uint fi = 0; fi < FaceCount(); ++fi) {
        const auto &face_normal = Faces[fi].Normal;
        const auto start = Faces[fi].Halfedge;
        auto current = start;
        do {
            const auto vh = Halfedges[current.idx()].Vertex;
            Normals[vh.idx()] += face_normal;
            current = Halfedges[current.idx()].Next;
        } while (current != start && current.is_valid());
    }

    // Normalize
    for (auto &n : Normals) n = glm::normalize(n);
}

PolyMesh::FaceVertexIterator &PolyMesh::FaceVertexIterator::operator++() {
    First = false;
    CurrentHalfedge = Mesh->Halfedges[CurrentHalfedge.idx()].Next;
    return *this;
}
PolyMesh::VertexOutgoingHalfedgeIterator &PolyMesh::VertexOutgoingHalfedgeIterator::operator++() {
    First = false;
    // Move to next outgoing halfedge: opposite of next
    const auto next_he = Mesh->Halfedges[CurrentHalfedge.idx()].Next;
    CurrentHalfedge = next_he.is_valid() ? Mesh->Halfedges[next_he.idx()].Opposite : HH{};
    return *this;
}
PolyMesh::FaceHalfedgeIterator &PolyMesh::FaceHalfedgeIterator::operator++() {
    First = false;
    CurrentHalfedge = Mesh->Halfedges[CurrentHalfedge.idx()].Next;
    return *this;
}

namespace {
std::optional<he::PolyMesh> read_obj(const std::filesystem::path &path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.string().c_str())) {
        return {};
    }

    he::PolyMesh mesh;
    // Add all vertices
    std::vector<he::VH> verts;
    verts.reserve(attrib.vertices.size() / 3);
    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        verts.emplace_back(mesh.AddVertex({attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]}));
    }
    // Add all faces from all shapes
    for (const auto &shape : shapes) {
        size_t vi = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const auto fv = shape.mesh.num_face_vertices[f];
            std::vector<he::VH> face_verts;
            face_verts.reserve(fv);
            for (size_t vi_f = 0; vi_f < fv; ++vi_f) {
                face_verts.emplace_back(verts[shape.mesh.indices[vi + vi_f].vertex_index]);
            }
            mesh.AddFace(face_verts);
            vi += fv;
        }
    }

    return mesh;
}

std::optional<he::PolyMesh> read_ply(const std::filesystem::path &path) {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) return {};

        tinyply::PlyFile ply_file;
        ply_file.parse_header(file);

        std::shared_ptr<tinyply::PlyData> vertices, faces;
        try {
            vertices = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
        } catch (...) {
            return {};
        }
        try {
            faces = ply_file.request_properties_from_element("face", {"vertex_indices"}, 0);
        } catch (...) {
            // Try alternative face property name
            try {
                faces = ply_file.request_properties_from_element("face", {"vertex_index"}, 0);
            } catch (...) {
                return {};
            }
        }

        ply_file.read(file);

        he::PolyMesh mesh;
        // Add vertices
        std::vector<he::VH> verts;
        verts.reserve(vertices->count);

        auto add_vertices = [&](const auto *data) {
            for (size_t i = 0; i < vertices->count; ++i) {
                verts.emplace_back(mesh.AddVertex({data[i * 3], data[i * 3 + 1], data[i * 3 + 2]}));
            }
        };
        if (vertices->t == tinyply::Type::FLOAT32) add_vertices(reinterpret_cast<const float *>(vertices->buffer.get()));
        else if (vertices->t == tinyply::Type::FLOAT64) add_vertices(reinterpret_cast<const double *>(vertices->buffer.get()));
        else return {};

        // Add faces (tinyply stores list properties as: count, index0, index1, ...)
        const auto *face_data = reinterpret_cast<const uint8_t *>(faces->buffer.get());
        const auto idx_size = (faces->t == tinyply::Type::UINT32 || faces->t == tinyply::Type::INT32) ? 4 :
            (faces->t == tinyply::Type::UINT16 || faces->t == tinyply::Type::INT16)                   ? 2 :
                                                                                                        1;

        size_t offset = 0;
        for (size_t f = 0; f < faces->count; ++f) {
            const auto face_size = face_data[offset++];
            std::vector<he::VH> face_verts;
            face_verts.reserve(face_size);
            for (uint8_t v = 0; v < face_size; ++v) {
                const uint32_t vi = idx_size == 4 ? *reinterpret_cast<const uint32_t *>(&face_data[offset]) :
                    idx_size == 2                 ? *reinterpret_cast<const uint16_t *>(&face_data[offset]) :
                                                    face_data[offset];
                offset += idx_size;
                face_verts.emplace_back(verts[vi]);
            }
            mesh.AddFace(face_verts);
        }

        return mesh;
    } catch (...) {
        return {};
    }
}
} // namespace

std::optional<PolyMesh> ReadMesh(const std::filesystem::path &path) {
    const auto ext = path.extension().string();
    if (ext == ".ply" || ext == ".PLY") return read_ply(path);
    return read_obj(path); // Try obj as default
}

} // namespace he
