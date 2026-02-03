#include "MeshStore.h"

#include "MeshData.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

namespace {
constexpr uint8_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};

MeshData ReadObj(const std::filesystem::path &path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.string().c_str())) {
        throw std::runtime_error{"Failed to load OBJ: " + err};
    }

    MeshData data;
    data.Positions.reserve(attrib.vertices.size() / 3);
    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        data.Positions.emplace_back(attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]);
    }

    for (const auto &shape : shapes) {
        size_t vi = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const auto fv = shape.mesh.num_face_vertices[f];
            std::vector<uint> face_verts;
            face_verts.reserve(fv);
            for (size_t vi_f = 0; vi_f < fv; ++vi_f) {
                face_verts.emplace_back(shape.mesh.indices[vi + vi_f].vertex_index);
            }
            data.Faces.emplace_back(std::move(face_verts));
            vi += fv;
        }
    }

    return data;
}

MeshData ReadPly(const std::filesystem::path &path) {
    std::ifstream file{path, std::ios::binary};
    if (!file) throw std::runtime_error{"Failed to open: " + path.string()};

    tinyply::PlyFile ply_file;
    ply_file.parse_header(file);

    auto vertices = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
    std::shared_ptr<tinyply::PlyData> faces;
    try {
        faces = ply_file.request_properties_from_element("face", {"vertex_indices"}, 0);
    } catch (...) {
        faces = ply_file.request_properties_from_element("face", {"vertex_index"}, 0);
    }
    ply_file.read(file);

    MeshData data;
    data.Positions.reserve(vertices->count);
    auto AddVertices = [&](const auto *raw) {
        for (size_t i = 0; i < vertices->count; ++i) {
            data.Positions.emplace_back(raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]);
        }
    };
    if (vertices->t == tinyply::Type::FLOAT32) AddVertices(reinterpret_cast<const float *>(vertices->buffer.get()));
    else if (vertices->t == tinyply::Type::FLOAT64) AddVertices(reinterpret_cast<const double *>(vertices->buffer.get()));
    else throw std::runtime_error{"Unsupported vertex type"};

    std::span face_buf{faces->buffer.get(), faces->buffer.size_bytes()};
    size_t idx_size;
    switch (faces->t) {
        case tinyply::Type::UINT32:
        case tinyply::Type::INT32: idx_size = 4; break;
        case tinyply::Type::UINT16:
        case tinyply::Type::INT16: idx_size = 2; break;
        case tinyply::Type::UINT8:
        case tinyply::Type::INT8: idx_size = 1; break;
        default: throw std::runtime_error{"Unsupported index type"};
    }

    size_t offset = 0;
    data.Faces.reserve(faces->count);
    for (size_t f = 0; f < faces->count; ++f) {
        const auto face_size = face_buf[offset++];
        std::vector<uint> face_verts;
        face_verts.reserve(face_size);
        for (uint8_t v = 0; v < face_size; ++v) {
            uint vi = 0;
            std::memcpy(&vi, &face_buf[offset], idx_size);
            offset += idx_size;
            face_verts.emplace_back(vi);
        }
        data.Faces.emplace_back(std::move(face_verts));
    }

    return data;
}

MeshData DeduplicateVertices(const MeshData &mesh) {
    struct VertexHash {
        constexpr size_t operator()(const vec3 &p) const noexcept {
            return std::hash<float>{}(p.x) ^ std::hash<float>{}(p.y) ^ std::hash<float>{}(p.z);
        }
    };

    MeshData deduped;
    deduped.Positions.reserve(mesh.Positions.size());
    std::unordered_map<vec3, uint, VertexHash> index_by_vertex;
    for (const auto &p : mesh.Positions) {
        if (auto [it, inserted] = index_by_vertex.try_emplace(p, deduped.Positions.size()); inserted) {
            deduped.Positions.emplace_back(p);
        }
    }

    deduped.Faces.reserve(mesh.Faces.size());
    for (const auto &face : mesh.Faces) {
        std::vector<uint> new_face;
        new_face.reserve(face.size());
        for (const auto idx : face) new_face.emplace_back(index_by_vertex.at(mesh.Positions[idx]));
        deduped.Faces.emplace_back(std::move(new_face));
    }

    return deduped;
}

std::vector<uint32_t> CreateFaceElementIds(const std::vector<std::vector<uint32_t>> &faces) {
    std::vector<uint32_t> ids;
    ids.reserve(faces.size() * 3);
    for (uint32_t fi = 0; fi < faces.size(); ++fi) {
        if (const uint32_t valence = faces[fi].size(); valence >= 3) {
            for (uint32_t i = 0; i < valence - 2; ++i) ids.insert(ids.end(), 3, fi + 1);
        }
    }
    return ids;
}
} // namespace

MeshStore::MeshStore(mvk::BufferContext &ctx)
    : VerticesBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer},
      VertexStateBuffer{ctx, 1, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
      FaceStateBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
      EdgeStateBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
      FaceIdBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer},
      FaceNormalBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::FaceNormalBuffer} {}

void MeshStore::UpdateNormals(const Mesh &mesh) {
    const auto id = mesh.GetStoreId();
    {
        auto face_normals = GetFaceNormals(id);
        for (uint fi = 0; fi < mesh.FaceCount(); ++fi) {
            auto it = mesh.cfv_iter(Mesh::FH{fi});
            const auto p0 = mesh.GetPosition(*it), p1 = mesh.GetPosition(*++it), p2 = mesh.GetPosition(*++it);
            face_normals[fi] = glm::normalize(glm::cross(p1 - p0, p2 - p0));
        }
    }
    {
        auto vertices = GetVertices(id);
        for (auto &v : vertices) v.Normal = vec3{0};
        for (uint fi = 0; fi < mesh.FaceCount(); ++fi) {
            const auto &face_normal = mesh.GetNormal(Mesh::FH{fi});
            for (const auto vh : mesh.fv_range(Mesh::FH{fi})) vertices[*vh].Normal += face_normal;
        }
        for (auto &v : vertices) v.Normal = glm::normalize(v.Normal);
    }
}
Mesh MeshStore::CreateMesh(MeshData &&data) {
    const auto id = AcquireId();
    auto &entry = Entries[id];
    entry.Alive = true;
    entry.Vertices = VerticesBuffer.Allocate(data.Positions.size());
    EnsureVertexStateCapacity(entry.Vertices);
    auto vertex_span = VerticesBuffer.GetMutable(entry.Vertices);
    for (size_t i = 0; i < data.Positions.size(); ++i) vertex_span[i].Position = data.Positions[i];
    entry.FaceIds = FaceIdBuffer.Allocate(CreateFaceElementIds(data.Faces));
    entry.FaceNormals = FaceNormalBuffer.Allocate(data.Faces.size());
    entry.FaceStates = FaceStateBuffer.Allocate(data.Faces.size());
    Mesh mesh{*this, id, std::move(data.Faces)};
    entry.EdgeStates = EdgeStateBuffer.Allocate(mesh.EdgeCount() * 2);
    UpdateNormals(mesh);
    std::ranges::fill(GetVertexStates(entry.Vertices), 0);
    std::ranges::fill(FaceStateBuffer.GetMutable(entry.FaceStates), 0);
    std::ranges::fill(EdgeStateBuffer.GetMutable(entry.EdgeStates), 0);
    return mesh;
}

Mesh MeshStore::CloneMesh(const Mesh &mesh) {
    const auto id = AcquireId();
    auto &entry = Entries[id];
    entry.Alive = true;
    {
        const auto src_vertices = GetVertices(mesh.GetStoreId());
        entry.Vertices = VerticesBuffer.Allocate(src_vertices.size());
        EnsureVertexStateCapacity(entry.Vertices);
        auto dst_vertices = VerticesBuffer.GetMutable(entry.Vertices);
        std::copy(src_vertices.begin(), src_vertices.end(), dst_vertices.begin());
    }
    {
        const auto src_face_ids = FaceIdBuffer.Get(Entries.at(mesh.GetStoreId()).FaceIds);
        entry.FaceIds = FaceIdBuffer.Allocate(src_face_ids);
        entry.FaceNormals = FaceNormalBuffer.Allocate(mesh.FaceCount());
        entry.FaceStates = FaceStateBuffer.Allocate(mesh.FaceCount());
        entry.EdgeStates = EdgeStateBuffer.Allocate(mesh.EdgeCount() * 2);
        const auto src_face_normals = GetFaceNormals(mesh.GetStoreId());
        auto dst_face_normals = FaceNormalBuffer.GetMutable(entry.FaceNormals);
        std::copy(src_face_normals.begin(), src_face_normals.end(), dst_face_normals.begin());
    }
    std::ranges::fill(GetVertexStates(entry.Vertices), 0);
    std::ranges::fill(FaceStateBuffer.GetMutable(entry.FaceStates), 0);
    std::ranges::fill(EdgeStateBuffer.GetMutable(entry.EdgeStates), 0);
    return {*this, id, mesh};
}

std::expected<Mesh, std::string> MeshStore::LoadMesh(const std::filesystem::path &path) {
    try {
        const auto ext = path.extension();
        auto data = ext == ".ply" || ext == ".PLY" ? ReadPly(path) : ReadObj(path);
        return CreateMesh(DeduplicateVertices(data));
    } catch (const std::exception &e) {
        return std::unexpected{e.what()};
    }
}

void MeshStore::SetPositions(const Mesh &mesh, std::span<const vec3> positions) {
    auto vertex_span = VerticesBuffer.GetMutable(Entries.at(mesh.GetStoreId()).Vertices);
    for (size_t i = 0; i < positions.size(); ++i) vertex_span[i].Position = positions[i];
    UpdateNormals(mesh);
}
void MeshStore::SetPosition(const Mesh &mesh, uint32_t index, vec3 position) {
    VerticesBuffer.GetMutable(Entries.at(mesh.GetStoreId()).Vertices)[index].Position = position;
    // Caller is responsible for updating normals
}

void MeshStore::Release(uint32_t id) {
    if (id >= Entries.size() || !Entries[id].Alive) return;
    auto &entry = Entries[id];
    VerticesBuffer.Release(entry.Vertices);
    FaceIdBuffer.Release(entry.FaceIds);
    FaceNormalBuffer.Release(entry.FaceNormals);
    FaceStateBuffer.Release(entry.FaceStates);
    EdgeStateBuffer.Release(entry.EdgeStates);
    entry = {};
    FreeIds.emplace_back(id);
}

void MeshStore::EnsureVertexStateCapacity(BufferRange range) {
    const auto required_size = static_cast<vk::DeviceSize>(range.Offset + range.Count) * sizeof(uint8_t);
    VertexStateBuffer.Reserve(required_size);
    VertexStateBuffer.UsedSize = std::max(VertexStateBuffer.UsedSize, required_size);
}

std::span<uint8_t> MeshStore::GetVertexStates(BufferRange range) {
    const auto byte_offset = static_cast<vk::DeviceSize>(range.Offset) * sizeof(uint8_t);
    const auto byte_size = static_cast<vk::DeviceSize>(range.Count) * sizeof(uint8_t);
    const auto bytes = VertexStateBuffer.GetMutableRange(byte_offset, byte_size);
    if (bytes.empty()) return {};
    return {reinterpret_cast<uint8_t *>(bytes.data()), range.Count};
}

using namespace he;

void MeshStore::UpdateElementStates(
    const Mesh &mesh,
    Element element,
    const std::unordered_set<VH> &selected_vertices,
    const std::unordered_set<EH> &selected_edges,
    const std::unordered_set<EH> &active_edges,
    const std::unordered_set<FH> &selected_faces,
    std::optional<uint32_t> active_handle
) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    auto face_states = FaceStateBuffer.GetMutable(entry.FaceStates);
    auto edge_states = EdgeStateBuffer.GetMutable(entry.EdgeStates);
    auto vertex_states = GetVertexStates(entry.Vertices);

    std::ranges::fill(face_states, 0);
    std::ranges::fill(edge_states, 0);
    std::ranges::fill(vertex_states, 0);

    if (element == Element::Face) {
        for (const auto fh : selected_faces) {
            face_states[*fh] |= ElementStateSelected;
            if (active_handle == *fh) {
                face_states[*active_handle] |= ElementStateActive;
            }
        }
    }

    if (element == Element::Edge || element == Element::Face) {
        for (uint32_t ei = 0; ei < mesh.EdgeCount(); ++ei) {
            uint8_t state = 0;
            if (selected_edges.contains(EH{ei})) {
                state |= ElementStateSelected;
                if ((element == Element::Edge && active_handle == ei) || active_edges.contains(EH{ei})) {
                    state |= ElementStateActive;
                }
            }
            edge_states[2 * ei] = edge_states[2 * ei + 1] = state;
        }
    } else if (element == Element::Vertex) {
        for (uint32_t ei = 0; ei < mesh.EdgeCount(); ++ei) {
            const auto heh = mesh.GetHalfedge(EH{ei}, 0);
            edge_states[2 * ei] = selected_vertices.contains(mesh.GetFromVertex(heh)) ? ElementStateSelected : 0;
            edge_states[2 * ei + 1] = selected_vertices.contains(mesh.GetToVertex(heh)) ? ElementStateSelected : 0;
        }
    }

    if (!selected_vertices.empty()) {
        for (const auto vh : selected_vertices) {
            vertex_states[*vh] |= ElementStateSelected;
            if (element == Element::Vertex && active_handle == *vh) {
                vertex_states[*active_handle] |= ElementStateActive;
            }
        }
    }
}

uint32_t MeshStore::AcquireId() {
    if (!FreeIds.empty()) {
        const auto id = FreeIds.back();
        FreeIds.pop_back();
        return id;
    }
    Entries.emplace_back();
    return Entries.size() - 1;
}
