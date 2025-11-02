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

VH PolyMesh::AddVertex(const vec3 &p) {
    Positions.emplace_back(p);
    Normals.emplace_back(vec3{0});
    OutgoingHalfedges.emplace_back(HH{});
    return VH(Positions.size() - 1);
}

FH PolyMesh::AddFace(const std::vector<VH> &vertices) {
    if (vertices.size() < 3) return {}; // Invalid face

    static auto MakeEdgeKey = [](int from, int to) { return (static_cast<uint64_t>(from) << 32) | static_cast<uint64_t>(to); };

    const auto fi = Faces.size();
    const auto start_he_i = Halfedges.size();
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

        if (!OutgoingHalfedges[*from_v].IsValid()) OutgoingHalfedges[*from_v] = hh;
        HalfedgeMap[MakeEdgeKey(*from_v, *to_v)] = hh;
    }

    // Find opposites and create edges
    for (size_t i = 0; i < vertices.size(); ++i) {
        const HH hh(start_he_i + i);
        const auto to_v = vertices[i];
        const auto from_v = vertices[i == 0 ? vertices.size() - 1 : i - 1];
        // Look for opposite halfedge in map
        if (const auto it = HalfedgeMap.find(MakeEdgeKey(*to_v, *from_v));
            it != HalfedgeMap.end()) {
            // Found existing opposite halfedge
            const auto opposite_hh = it->second;
            Halfedges[*hh].Opposite = opposite_hh;
            Halfedges[*opposite_hh].Opposite = hh;
            // They should share the same edge
            Halfedges[*hh].Edge = Halfedges[*opposite_hh].Edge;
        } else {
            // Create new edge
            Edges.emplace_back(hh);
            Halfedges[*hh].Edge = EH(Edges.size() - 1);
        }
    }

    return FH(fi);
}

HH PolyMesh::GetHalfedge(EH eh, uint i) const {
    if (!eh.IsValid() || *eh >= static_cast<int>(Edges.size())) return {};
    const auto h0 = Edges[*eh].Halfedge;
    return i == 0 ? h0 : (i == 1 && h0.IsValid() ? Halfedges[*h0].Opposite : HH{});
}

HH PolyMesh::GetOppositeHalfedge(HH hh) const {
    if (!hh.IsValid() || *hh >= static_cast<int>(Halfedges.size())) return {};
    return Halfedges[*hh].Opposite;
}

VH PolyMesh::GetFromVertex(HH hh) const {
    if (!hh.IsValid() || *hh >= static_cast<int>(Halfedges.size())) return {};

    // From-vertex is the to-vertex of the opposite halfedge
    const auto opp = Halfedges[*hh].Opposite;
    return opp.IsValid() ? Halfedges[*opp].Vertex : VH{};
}

uint PolyMesh::GetValence(VH vh) const {
    if (!vh.IsValid() || *vh >= static_cast<int>(Positions.size())) return 0;

    const auto start = OutgoingHalfedges[*vh];
    if (!start.IsValid()) return 0;

    uint count{0};
    auto current = start;
    do {
        count++;
        // Move to next outgoing halfedge
        const auto next_he = Halfedges[*current].Next;
        if (!next_he.IsValid()) break;

        const auto opp = Halfedges[*next_he].Opposite;
        if (!opp.IsValid()) break;

        current = opp;
    } while (current != start && current.IsValid());

    return count;
}

uint PolyMesh::GetValence(FH fh) const {
    if (!fh.IsValid() || *fh >= static_cast<int>(Faces.size())) return 0;

    uint count{0};
    const auto start = Faces[*fh].Halfedge;
    auto current = start;
    do {
        count++;
        current = Halfedges[*current].Next;
    } while (current != start && current.IsValid());

    return count;
}

vec3 PolyMesh::CalcFaceCentroid(FH fh) const {
    if (!fh.IsValid() || *fh >= static_cast<int>(Faces.size())) return vec3{0};

    vec3 centroid{0};
    uint count{0};
    const auto start = Faces[*fh].Halfedge;
    auto current = start;
    do {
        const auto vh = Halfedges[*current].Vertex;
        centroid += Positions[*vh];
        count++;
        current = Halfedges[*current].Next;
    } while (current != start && current.IsValid());

    if (count > 0) centroid /= static_cast<float>(count);
    return centroid;
}

float PolyMesh::CalcEdgeLength(HH hh) const {
    if (!hh.IsValid() || *hh >= static_cast<int>(Halfedges.size())) return 0;

    const auto from_v = GetFromVertex(hh);
    const auto to_v = Halfedges[*hh].Vertex;
    if (!from_v.IsValid() || !to_v.IsValid()) return 0;
    return glm::length(Positions[*to_v] - Positions[*from_v]);
}

void PolyMesh::UpdateNormals() {
    ComputeFaceNormals();
    ComputeVertexNormals();
}

void PolyMesh::ComputeFaceNormals() {
    for (uint fi = 0; fi < FaceCount(); ++fi) {
        const auto start = Faces[fi].Halfedge;
        const auto h0 = start, h1 = Halfedges[*h0].Next, h2 = Halfedges[*h1].Next;
        const auto p0 = Positions[*Halfedges[*h0].Vertex];
        const auto p1 = Positions[*Halfedges[*h1].Vertex];
        const auto p2 = Positions[*Halfedges[*h2].Vertex];
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
            const auto vh = Halfedges[*current].Vertex;
            Normals[*vh] += face_normal;
            current = Halfedges[*current].Next;
        } while (current != start && current.IsValid());
    }

    // Normalize
    for (auto &n : Normals) n = glm::normalize(n);
}

PolyMesh::FaceVertexIterator &PolyMesh::FaceVertexIterator::operator++() {
    First = false;
    CurrentHalfedge = Mesh->Halfedges[*CurrentHalfedge].Next;
    return *this;
}
PolyMesh::VertexOutgoingHalfedgeIterator &PolyMesh::VertexOutgoingHalfedgeIterator::operator++() {
    First = false;
    // Move to next outgoing halfedge: opposite of next
    const auto next_he = Mesh->Halfedges[*CurrentHalfedge].Next;
    CurrentHalfedge = next_he.IsValid() ? Mesh->Halfedges[*next_he].Opposite : HH{};
    return *this;
}
PolyMesh::FaceHalfedgeIterator &PolyMesh::FaceHalfedgeIterator::operator++() {
    First = false;
    CurrentHalfedge = Mesh->Halfedges[*CurrentHalfedge].Next;
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
        auto AddVertices = [&](const auto *data) {
            for (size_t i = 0; i < vertices->count; ++i) {
                verts.emplace_back(mesh.AddVertex({data[i * 3], data[i * 3 + 1], data[i * 3 + 2]}));
            }
        };
        if (vertices->t == tinyply::Type::FLOAT32) AddVertices(reinterpret_cast<const float *>(vertices->buffer.get()));
        else if (vertices->t == tinyply::Type::FLOAT64) AddVertices(reinterpret_cast<const double *>(vertices->buffer.get()));
        else return {};

        // Add faces (tinyply stores list properties as: count, index0, index1, ...)
        const auto *face_data = reinterpret_cast<const uint8_t *>(faces->buffer.get());
        const auto idx_size = faces->t == tinyply::Type::UINT32 || faces->t == tinyply::Type::INT32 ? 4 :
            faces->t == tinyply::Type::UINT16 || faces->t == tinyply::Type::INT16                   ? 2 :
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
