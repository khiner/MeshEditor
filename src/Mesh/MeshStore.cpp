#include "MeshStore.h"

#include <cassert>
#include <fstream>
#include <ranges>
#include <string>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

using std::views::transform, std::ranges::to;

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
} // namespace

void MeshStore::Init(mvk::BufferContext &ctx) {
    VerticesBuffer = std::make_unique<Megabuffer<Vertex3D>>(ctx, mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer);
}

Mesh MeshStore::CreateMesh(MeshData &&data) {
    assert(VerticesBuffer && "MeshStore not initialized with buffer context.");
    const uint32_t id = AcquireId();
    auto &entry = Entries[id];
    entry.Alive = true;
    auto vertices = data.Positions | transform([](const vec3 &position) { return Vertex3D{position, {}}; }) | to<std::vector<Vertex3D>>();
    entry.Vertices = VerticesBuffer->Allocate(vertices);
    return Mesh{*this, id, std::move(data.Faces)};
}

std::optional<Mesh> MeshStore::LoadMesh(const std::filesystem::path &path) {
    const auto ext = path.extension().string();
    if (auto mesh = ext == ".ply" || ext == ".PLY" ? ReadPly(path) : ReadObj(path)) {
        return CreateMesh(DeduplicateVertices(*mesh));
    }
    return {};
}

std::span<const Vertex3D> MeshStore::GetVertices(uint32_t id) const {
    assert(VerticesBuffer && "MeshStore not initialized with buffer context.");
    return VerticesBuffer->Get(Entries.at(id).Vertices);
}

std::span<Vertex3D> MeshStore::GetVerticesMutable(uint32_t id) {
    assert(VerticesBuffer && "MeshStore not initialized with buffer context.");
    return VerticesBuffer->GetMutable(Entries.at(id).Vertices);
}

void MeshStore::UpdateVertices(uint32_t id, std::span<const Vertex3D> vertices) {
    assert(VerticesBuffer && "MeshStore not initialized with buffer context.");
    VerticesBuffer->Update(Entries.at(id).Vertices, vertices);
}

void MeshStore::Release(uint32_t id) {
    if (id >= Entries.size() || !Entries[id].Alive) return;
    assert(VerticesBuffer && "MeshStore not initialized with buffer context.");
    auto &entry = Entries[id];
    VerticesBuffer->Release(entry.Vertices);
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
