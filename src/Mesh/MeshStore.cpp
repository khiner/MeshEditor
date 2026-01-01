#include "MeshStore.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

namespace {
std::optional<MeshData> ReadObj(const std::filesystem::path &path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.string().c_str())) {
        return {};
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

std::optional<MeshData> ReadPly(const std::filesystem::path &path) {
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
            try {
                faces = ply_file.request_properties_from_element("face", {"vertex_index"}, 0);
            } catch (...) {
                return {};
            }
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
        else return {};

        const auto *face_data = reinterpret_cast<const uint8_t *>(faces->buffer.get());
        const auto idx_size = faces->t == tinyply::Type::UINT32 || faces->t == tinyply::Type::INT32 ? 4 :
            faces->t == tinyply::Type::UINT16 || faces->t == tinyply::Type::INT16                   ? 2 :
                                                                                                      1;

        size_t offset = 0;
        data.Faces.reserve(faces->count);
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
            data.Faces.emplace_back(std::move(face_verts));
        }

        return data;
    } catch (...) {
        return {};
    }
}

MeshData DeduplicateVertices(const MeshData &mesh) {
    struct VertexHash {
        constexpr size_t operator()(const vec3 &p) const noexcept {
            return std::hash<float>{}(p[0]) ^ std::hash<float>{}(p[1]) ^ std::hash<float>{}(p[2]);
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
        const auto valence = static_cast<uint32_t>(faces[fi].size());
        if (valence < 3) continue;
        for (uint32_t i = 0; i < valence - 2; ++i) {
            ids.insert(ids.end(), 3, fi + 1);
        }
    }
    return ids;
}

} // namespace

void MeshStore::Init(mvk::BufferContext &ctx) {
    VerticesBuffer = std::make_unique<BufferArena<Vertex3D>>(ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer);
    FaceIdBuffer = std::make_unique<BufferArena<uint32_t>>(ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer);
    FaceNormalBuffer = std::make_unique<BufferArena<vec3>>(ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::FaceNormalBuffer);
}

Mesh MeshStore::CreateMesh(MeshData &&data) {
    const uint32_t id = AcquireId();
    auto &entry = Entries[id];
    entry.Alive = true;
    entry.Vertices = VerticesBuffer->Allocate(static_cast<uint32_t>(data.Positions.size()));
    auto vertex_span = VerticesBuffer->GetMutable(entry.Vertices);
    for (size_t i = 0; i < data.Positions.size(); ++i) {
        vertex_span[i].Position = data.Positions[i];
        vertex_span[i].Normal = vec3{0};
    }
    const auto face_ids = CreateFaceElementIds(data.Faces);
    entry.FaceIds = FaceIdBuffer->Allocate(face_ids);
    entry.FaceNormals = FaceNormalBuffer->Allocate(static_cast<uint32_t>(data.Faces.size()));
    Mesh mesh{*this, id, std::move(data.Faces)};
    {
        auto face_normals = GetFaceNormalsMutable(mesh.GetStoreId());
        for (uint fi = 0; fi < mesh.FaceCount(); ++fi) {
            auto it = mesh.cfv_iter(Mesh::FH(fi));
            const auto p0 = mesh.GetPosition(*it);
            const auto p1 = mesh.GetPosition(*++it);
            const auto p2 = mesh.GetPosition(*++it);
            face_normals[fi] = glm::normalize(glm::cross(p1 - p0, p2 - p0));
        }
        FlushFaceNormals(mesh.GetStoreId());
    }
    {
        auto vertices = GetVerticesMutable(mesh.GetStoreId());
        for (auto &v : vertices) v.Normal = vec3{0};

        for (uint fi = 0; fi < mesh.FaceCount(); ++fi) {
            const auto &face_normal = mesh.GetNormal(Mesh::FH(fi));
            for (const auto vh : mesh.fv_range(Mesh::FH(fi))) {
                vertices[*vh].Normal += face_normal;
            }
        }

        for (auto &v : vertices) v.Normal = glm::normalize(v.Normal);
        FlushVertices(mesh.GetStoreId());
    }
    return mesh;
}

Mesh MeshStore::CloneMesh(const Mesh &mesh) {
    const uint32_t id = AcquireId();
    auto &entry = Entries[id];
    entry.Alive = true;

    const auto src_vertices = GetVertices(mesh.GetStoreId());
    entry.Vertices = VerticesBuffer->Allocate(static_cast<uint32_t>(src_vertices.size()));
    auto dst_vertices = VerticesBuffer->GetMutable(entry.Vertices);
    std::copy(src_vertices.begin(), src_vertices.end(), dst_vertices.begin());

    const auto src_face_ids = FaceIdBuffer->Get(Entries.at(mesh.GetStoreId()).FaceIds);
    entry.FaceIds = FaceIdBuffer->Allocate(src_face_ids);
    entry.FaceNormals = FaceNormalBuffer->Allocate(static_cast<uint32_t>(mesh.FaceCount()));
    const auto src_face_normals = GetFaceNormals(mesh.GetStoreId());
    auto dst_face_normals = FaceNormalBuffer->GetMutable(entry.FaceNormals);
    std::copy(src_face_normals.begin(), src_face_normals.end(), dst_face_normals.begin());

    Mesh clone{*this, id, mesh};
    return clone;
}

std::optional<Mesh> MeshStore::LoadMesh(const std::filesystem::path &path) {
    const auto ext = path.extension().string();
    if (auto mesh = ext == ".ply" || ext == ".PLY" ? ReadPly(path) : ReadObj(path)) {
        return CreateMesh(DeduplicateVertices(*mesh));
    }
    return {};
}

std::span<const Vertex3D> MeshStore::GetVertices(uint32_t id) const { return VerticesBuffer->Get(Entries.at(id).Vertices); }
std::span<const vec3> MeshStore::GetFaceNormals(uint32_t id) const { return FaceNormalBuffer->Get(Entries.at(id).FaceNormals); }
MeshStore::Range MeshStore::GetVerticesRange(uint32_t id) const { return Entries.at(id).Vertices; }
uint32_t MeshStore::GetVerticesSlot() const { return VerticesBuffer->Buffer.Slot; }
MeshStore::Range MeshStore::GetFaceIdRange(uint32_t id) const { return Entries.at(id).FaceIds; }
MeshStore::Range MeshStore::GetFaceNormalRange(uint32_t id) const { return Entries.at(id).FaceNormals; }
uint32_t MeshStore::GetFaceIdSlot() const { return FaceIdBuffer->Buffer.Slot; }
uint32_t MeshStore::GetFaceNormalSlot() const { return FaceNormalBuffer->Buffer.Slot; }
std::span<Vertex3D> MeshStore::GetVerticesMutable(uint32_t id) { return VerticesBuffer->GetMutable(Entries.at(id).Vertices); }
std::span<vec3> MeshStore::GetFaceNormalsMutable(uint32_t id) { return FaceNormalBuffer->GetMutable(Entries.at(id).FaceNormals); }

void MeshStore::UpdateVertices(uint32_t id, std::span<const Vertex3D> vertices) { VerticesBuffer->Update(Entries.at(id).Vertices, vertices); }
void MeshStore::FlushVertices(uint32_t id) { VerticesBuffer->Flush(Entries.at(id).Vertices); }
void MeshStore::FlushFaceNormals(uint32_t id) { FaceNormalBuffer->Flush(Entries.at(id).FaceNormals); }

void MeshStore::Release(uint32_t id) {
    if (id >= Entries.size() || !Entries[id].Alive) return;
    auto &entry = Entries[id];
    VerticesBuffer->Release(entry.Vertices);
    FaceIdBuffer->Release(entry.FaceIds);
    FaceNormalBuffer->Release(entry.FaceNormals);
    entry = {};
    FreeIds.emplace_back(id);
}

uint32_t MeshStore::AcquireId() {
    if (!FreeIds.empty()) {
        const uint32_t id = FreeIds.back();
        FreeIds.pop_back();
        return id;
    }
    Entries.emplace_back();
    return static_cast<uint32_t>(Entries.size() - 1);
}
