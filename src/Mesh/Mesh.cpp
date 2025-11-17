#include "Mesh.h"

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <ranges>

using std::ranges::any_of, std::ranges::find_if, std::ranges::distance;

namespace {
constexpr uint64_t MakeEdgeKey(uint from, uint to) {
    return (static_cast<uint64_t>(from) << 32) | static_cast<uint64_t>(to);
}
} // namespace

Mesh::Mesh(std::vector<vec3> &&vertices, std::vector<std::vector<uint>> &&faces) {
    // Reserve and initialize vertex data
    Positions = std::move(vertices);
    Normals.resize(Positions.size());
    OutgoingHalfedges.resize(Positions.size());

    std::unordered_map<uint64_t, HH> halfedge_map;
    for (const auto &face : faces) {
        assert(face.size() >= 3);

        const auto fi = Faces.size();
        const auto start_he_i = Halfedges.size();
        Faces.emplace_back(HH(start_he_i), vec3{0});

        // Create halfedges, find opposites, and create edges
        for (size_t i = 0; i < face.size(); ++i) {
            const auto to_v = face[i];
            const auto from_v = face[i == 0 ? face.size() - 1 : i - 1];
            Halfedges.emplace_back(Halfedge{
                .Vertex = VH(to_v),
                .Next = HH(start_he_i + (i + 1) % face.size()),
                .Opposite = HH{},
                .Face = FH(fi),
            });

            const HH hh(start_he_i + i);
            if (!OutgoingHalfedges[from_v]) OutgoingHalfedges[from_v] = hh;
            halfedge_map.emplace(MakeEdgeKey(from_v, to_v), hh);

            // Look for opposite halfedge (from previously added faces)
            if (const auto it = halfedge_map.find(MakeEdgeKey(to_v, from_v));
                it != halfedge_map.end()) {
                // Found existing opposite halfedge
                const auto opposite_hh = it->second;
                Halfedges[*hh].Opposite = opposite_hh;
                Halfedges[*opposite_hh].Opposite = hh;
                // They should share the same edge
                HalfedgeToEdge.emplace_back(HalfedgeToEdge[*opposite_hh]);
            } else {
                // Create new edge
                Edges.emplace_back(hh);
                HalfedgeToEdge.emplace_back(Edges.size() - 1);
            }
        }
    }
    ComputeFaceNormals();
    ComputeVertexNormals();
    SetColor(DefaultMeshColor);
}

he::VH Mesh::GetFromVertex(HH hh) const {
    assert(*hh < Halfedges.size());
    const auto opp = Halfedges[*hh].Opposite;
    if (opp) return Halfedges[*opp].Vertex;

    // For boundary halfedges, find the previous halfedge in the face loop
    const auto range = FaceHalfedgeRange{this, Halfedges[*hh].Next};
    auto it = find_if(range, [&](HH h) { return Halfedges[*h].Next == hh; });
    return it != range.end() ? Halfedges[**it].Vertex : VH{};
}

uint Mesh::GetValence(VH vh) const { return distance(voh_range(vh)); }
uint Mesh::GetValence(FH fh) const { return distance(fh_range(fh)); }

vec3 Mesh::CalcFaceCentroid(FH fh) const {
    assert(*fh < Faces.size());
    vec3 centroid{0};
    uint count{0};
    for (auto vh : fv_range(fh)) {
        centroid += Positions[*vh];
        count++;
    }
    return count > 0 ? centroid / static_cast<float>(count) : centroid;
}

float Mesh::CalcEdgeLength(HH hh) const {
    assert(*hh < Halfedges.size());
    const auto from_v = GetFromVertex(hh);
    const auto to_v = Halfedges[*hh].Vertex;
    if (!from_v || !to_v) return 0;
    return glm::length(Positions[*to_v] - Positions[*from_v]);
}

void Mesh::ComputeFaceNormals() {
    for (uint fi = 0; fi < FaceCount(); ++fi) {
        auto it = cfv_iter(FH(fi));
        const auto p0 = Positions[**it];
        const auto p1 = Positions[**(++it)];
        const auto p2 = Positions[**(++it)];
        Faces[fi].Normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    }
}

void Mesh::ComputeVertexNormals() {
    // Reset all vertex normals
    for (auto &n : Normals) n = vec3{0};

    // Accumulate face normals to vertices
    for (uint fi = 0; fi < FaceCount(); ++fi) {
        const auto &face_normal = Faces[fi].Normal;
        for (const auto vh : fv_range(FH(fi))) {
            Normals[*vh] += face_normal;
        }
    }

    // Normalize
    for (auto &n : Normals) n = glm::normalize(n);
}

float Mesh::CalcFaceArea(FH fh) const {
    assert(*fh < FaceCount());
    float area{0};
    auto fv_it = cfv_iter(fh);
    const auto p0 = Positions[**fv_it++];
    for (vec3 p1 = Positions[**fv_it++], p2; fv_it; ++fv_it) {
        p2 = Positions[**fv_it];
        const auto cross_product = glm::cross(p1 - p0, p2 - p0);
        area += glm::length(cross_product) * 0.5f;
        p1 = p2;
    }
    return area;
}

he::VH Mesh::FindNearestVertex(vec3 p) const {
    VH closest_vertex;
    float min_distance_sq = std::numeric_limits<float>::max();
    for (const auto vh : vertices()) {
        const vec3 diff = Positions[*vh] - p;
        if (const float distance_sq = glm::dot(diff, diff); distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }
    return closest_vertex;
}

bool Mesh::VertexBelongsToFace(VH vh, FH fh) const {
    return vh && fh && any_of(fv_range(fh), [vh](const auto &fv) { return fv == vh; });
}

bool Mesh::VertexBelongsToEdge(VH vh, EH eh) const {
    return vh && eh && any_of(voh_range(vh), [&](const auto &heh) { return GetEdge(heh) == eh; });
}

bool Mesh::VertexBelongsToFaceEdge(VH vh, FH fh, EH eh) const {
    return fh && eh && any_of(voh_range(vh), [&](const auto &heh) {
               return GetEdge(heh) == eh && (GetFace(heh) == fh || GetFace(GetOppositeHalfedge(heh)) == fh);
           });
}

bool Mesh::EdgeBelongsToFace(EH eh, FH fh) const {
    return eh && fh && any_of(fh_range(fh), [&](const auto &heh) { return GetEdge(heh) == eh; });
}

std::vector<uint> Mesh::CreateTriangleIndices() const {
    std::vector<uint> indices;
    for (const auto fh : faces()) {
        auto fv_it = cfv_iter(fh);
        const auto v0 = *fv_it++;
        VH v1 = *fv_it++, v2;
        for (; fv_it; ++fv_it) {
            v2 = *fv_it;
            indices.insert(indices.end(), {*v0, *v1, *v2});
            v1 = v2;
        }
    }
    return indices;
}

std::vector<uint> Mesh::CreateTriangulatedFaceIndices() const {
    std::vector<uint> indices;
    uint index = 0;
    for (const auto fh : faces()) {
        const auto valence = GetValence(fh);
        for (uint i = 0; i < valence - 2; ++i) {
            indices.insert(indices.end(), {index, index + i + 1, index + i + 2});
        }
        index += valence;
    }
    return indices;
}

std::vector<uint> Mesh::CreateEdgeIndices() const {
    std::vector<uint> indices;
    indices.reserve(EdgeCount() * 2);
    for (uint ei = 0; ei < EdgeCount(); ++ei) {
        indices.emplace_back(2 * ei);
        indices.emplace_back(2 * ei + 1);
    }
    return indices;
}

Mesh Mesh::WithDeduplicatedVertices() const {
    struct VertexHash {
        constexpr size_t operator()(const vec3 &p) const noexcept {
            return std::hash<float>{}(p[0]) ^ std::hash<float>{}(p[1]) ^ std::hash<float>{}(p[2]);
        }
    };

    std::vector<vec3> vertices;
    vertices.reserve(VertexCount());
    std::unordered_map<vec3, uint, VertexHash> index_by_vertex;
    for (const auto vh : this->vertices()) {
        const auto p = GetPosition(vh);
        if (auto [it, inserted] = index_by_vertex.try_emplace(p, vertices.size()); inserted) {
            vertices.emplace_back(p);
        }
    }

    std::vector<std::vector<uint>> faces;
    faces.reserve(FaceCount());
    for (const auto &fh : this->faces()) {
        std::vector<uint> new_face;
        new_face.reserve(GetValence(fh));
        for (const auto &vh : fv_range(fh)) new_face.emplace_back(index_by_vertex.at(GetPosition(vh)));
        faces.emplace_back(std::move(new_face));
    }

    return {std::move(vertices), std::move(faces)};
}

namespace {
std::optional<Mesh> ReadObj(const std::filesystem::path &path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.string().c_str())) {
        return {};
    }

    // Build vertices
    std::vector<vec3> vertices;
    vertices.reserve(attrib.vertices.size() / 3);
    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        vertices.emplace_back(attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]);
    }

    // Build faces from all shapes
    std::vector<std::vector<uint>> faces;
    for (const auto &shape : shapes) {
        size_t vi = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const auto fv = shape.mesh.num_face_vertices[f];
            std::vector<uint> face_verts;
            face_verts.reserve(fv);
            for (size_t vi_f = 0; vi_f < fv; ++vi_f) {
                face_verts.emplace_back(shape.mesh.indices[vi + vi_f].vertex_index);
            }
            faces.emplace_back(std::move(face_verts));
            vi += fv;
        }
    }

    return Mesh{std::move(vertices), std::move(faces)};
}

std::optional<Mesh> ReadPly(const std::filesystem::path &path) {
    try {
        std::ifstream file{path, std::ios::binary};
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

        // Build vertices
        std::vector<vec3> vertex_list;
        vertex_list.reserve(vertices->count);
        auto AddVertices = [&](const auto *data) {
            for (size_t i = 0; i < vertices->count; ++i) {
                vertex_list.emplace_back(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
            }
        };
        if (vertices->t == tinyply::Type::FLOAT32) AddVertices(reinterpret_cast<const float *>(vertices->buffer.get()));
        else if (vertices->t == tinyply::Type::FLOAT64) AddVertices(reinterpret_cast<const double *>(vertices->buffer.get()));
        else return {};

        // Build faces (tinyply stores list properties as: count, index0, index1, ...)
        const auto *face_data = reinterpret_cast<const uint8_t *>(faces->buffer.get());
        const auto idx_size = faces->t == tinyply::Type::UINT32 || faces->t == tinyply::Type::INT32 ? 4 :
            faces->t == tinyply::Type::UINT16 || faces->t == tinyply::Type::INT16                   ? 2 :
                                                                                                      1;

        size_t offset = 0;
        std::vector<std::vector<uint>> face_list;
        face_list.reserve(faces->count);
        for (size_t f = 0; f < faces->count; ++f) {
            const auto face_size = face_data[offset++];
            std::vector<uint> face_verts;
            face_verts.reserve(face_size);
            for (uint8_t v = 0; v < face_size; ++v) {
                const uint vi = idx_size == 4 ? *reinterpret_cast<const uint *>(&face_data[offset]) :
                    idx_size == 2             ? *reinterpret_cast<const uint16_t *>(&face_data[offset]) :
                                                face_data[offset];
                offset += idx_size;
                face_verts.emplace_back(vi);
            }
            face_list.emplace_back(std::move(face_verts));
        }

        return Mesh{std::move(vertex_list), std::move(face_list)};
    } catch (...) {
        return {};
    }
}
} // namespace

std::optional<Mesh> Mesh::Load(const std::filesystem::path &path) {
    const auto ext = path.extension().string();
    // Try obj as default, even if it's not .obj
    auto mesh = ext == ".ply" || ext == ".PLY" ? ReadPly(path) : ReadObj(path);
    if (!mesh) return {};
    // Deduplicate even if not strictly triangle soup. Assumes this is a surface mesh.
    return mesh->WithDeduplicatedVertices();
}
