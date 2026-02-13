#include "MeshStore.h"

#include "MeshData.h"
#include "MorphTargetData.h"

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

MeshData DeduplicateVertices(MeshData &&mesh) {
    struct VertexHash {
        constexpr size_t operator()(const vec3 &p) const noexcept {
            return std::hash<float>{}(p.x) ^ std::hash<float>{}(p.y) ^ std::hash<float>{}(p.z);
        }
    };

    MeshData deduped;
    deduped.Positions.reserve(mesh.Positions.size());
    std::unordered_map<vec3, uint, VertexHash> index_by_vertex;
    for (const auto &p : mesh.Positions) {
        if (const auto [it, inserted] = index_by_vertex.try_emplace(p, deduped.Positions.size()); inserted) {
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
    uint32_t triangle_count{0};
    for (const auto &face : faces) triangle_count += face.size() - 2;

    std::vector<uint32_t> ids;
    ids.reserve(triangle_count);
    // Assign the same 1-indexed face ID to each of the face's triangles.
    for (uint32_t fi = 0; fi < faces.size(); ++fi) ids.insert(ids.end(), faces[fi].size() - 2, fi + 1);
    return ids;
}
} // namespace

MeshStore::MeshStore(mvk::BufferContext &ctx)
    : VerticesBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer},
      FaceNormalBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::FaceNormalBuffer},
      VertexStateBuffer{ctx, 1, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
      FaceStateBuffer{ctx, 1, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
      EdgeStateBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
      TriangleFaceIdBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer},
      BoneDeformBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::BoneDeformBuffer},
      MorphTargetBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MorphTargetBuffer} {}

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

Mesh MeshStore::CreateMesh(MeshData &&data, std::optional<ArmatureDeformData> deform_data, std::optional<MorphTargetData> morph_data) {
    if (deform_data && (deform_data->Joints.size() != data.Positions.size() || deform_data->Weights.size() != data.Positions.size())) {
        throw std::runtime_error{"ArmatureDeformData channel counts must match the position count."};
    }
    const auto vertex_count = static_cast<uint32_t>(data.Positions.size());
    const auto vertices = AllocateVertices(vertex_count);
    auto vertex_span = VerticesBuffer.GetMutable(vertices);
    for (uint32_t i = 0; i < vertex_count; ++i) {
        vertex_span[i] = {.Position = data.Positions[i]};
    }
    Range bone_deform{};
    if (deform_data) {
        bone_deform = BoneDeformBuffer.Allocate(vertex_count);
        auto bd_span = BoneDeformBuffer.GetMutable(bone_deform);
        for (uint32_t i = 0; i < vertex_count; ++i) {
            const auto &joints = deform_data->Joints[i];
            bd_span[i] = BoneDeformVertex{
                .Joints = {joints[0], joints[1], joints[2], joints[3]},
                .Weights = deform_data->Weights[i],
            };
        }
    }
    Range morph_targets{};
    uint32_t morph_target_count{0};
    std::vector<float> default_morph_weights;
    if (morph_data && morph_data->TargetCount > 0 && vertex_count > 0) {
        morph_target_count = morph_data->TargetCount;
        const auto total = morph_target_count * vertex_count;
        morph_targets = MorphTargetBuffer.Allocate(total);
        auto mt_span = MorphTargetBuffer.GetMutable(morph_targets);
        for (uint32_t i = 0; i < total; ++i) {
            mt_span[i] = MorphTargetVertex{.PositionDelta = morph_data->PositionDeltas[i]};
        }
        default_morph_weights = std::move(morph_data->DefaultWeights);
        default_morph_weights.resize(morph_target_count, 0.f);
    }
    const auto face_normals = AllocateFaces(data.Faces.size());
    const auto id = AcquireId({
        .Vertices = vertices,
        .FaceNormals = face_normals,
        .EdgeStates = {},
        .TriangleFaceIds = TriangleFaceIdBuffer.Allocate(CreateFaceElementIds(data.Faces)),
        .BoneDeform = bone_deform,
        .MorphTargets = morph_targets,
        .MorphTargetCount = morph_target_count,
        .DefaultMorphWeights = std::move(default_morph_weights),
        .Alive = true,
    });
    auto &entry = Entries[id];
    Mesh mesh{*this, id, std::move(data.Faces)};
    entry.EdgeStates = EdgeStateBuffer.Allocate(mesh.EdgeCount() * 2);
    UpdateNormals(mesh);
    std::ranges::fill(GetVertexStates(vertices), 0);
    std::ranges::fill(GetFaceStates(face_normals), 0);
    std::ranges::fill(EdgeStateBuffer.GetMutable(entry.EdgeStates), 0);
    return mesh;
}

Mesh MeshStore::CloneMesh(const Mesh &mesh) {
    const auto src_id = mesh.GetStoreId();
    const auto src_vertices = GetVertices(src_id);
    const auto vertices = AllocateVertices(src_vertices.size());
    std::ranges::copy(src_vertices, VerticesBuffer.GetMutable(vertices).begin());

    const auto face_normals = AllocateFaces(mesh.FaceCount());
    std::ranges::copy(GetFaceNormals(src_id), FaceNormalBuffer.GetMutable(face_normals).begin());

    const auto edge_states = EdgeStateBuffer.Allocate(mesh.EdgeCount() * 2);
    const auto &src_entry = Entries.at(src_id);
    const auto id = AcquireId({
        .Vertices = vertices,
        .FaceNormals = face_normals,
        .EdgeStates = edge_states,
        .TriangleFaceIds = TriangleFaceIdBuffer.Allocate(TriangleFaceIdBuffer.Get(src_entry.TriangleFaceIds)),
        .BoneDeform = src_entry.BoneDeform.Count > 0 ? BoneDeformBuffer.Allocate(BoneDeformBuffer.Get(src_entry.BoneDeform)) : Range{},
        .MorphTargets = src_entry.MorphTargets.Count > 0 ? MorphTargetBuffer.Allocate(MorphTargetBuffer.Get(src_entry.MorphTargets)) : Range{},
        .MorphTargetCount = src_entry.MorphTargetCount,
        .DefaultMorphWeights = src_entry.DefaultMorphWeights,
        .Alive = true,
    });

    std::ranges::fill(GetVertexStates(vertices), 0);
    std::ranges::fill(GetFaceStates(face_normals), 0);
    std::ranges::fill(EdgeStateBuffer.GetMutable(edge_states), 0);
    return {*this, id, mesh};
}

std::expected<Mesh, std::string> MeshStore::LoadMesh(const std::filesystem::path &path) {
    auto ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    MeshData data;
    try {
        if (ext == ".ply") data = ReadPly(path);
        else if (ext == ".obj") data = ReadObj(path);
        else return std::unexpected{"Unsupported file format: " + ext};
    } catch (const std::exception &e) {
        return std::unexpected{e.what()};
    }
    return CreateMesh(DeduplicateVertices(std::move(data)));
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
    TriangleFaceIdBuffer.Release(entry.TriangleFaceIds);
    FaceNormalBuffer.Release(entry.FaceNormals);
    EdgeStateBuffer.Release(entry.EdgeStates);
    if (entry.BoneDeform.Count > 0) BoneDeformBuffer.Release(entry.BoneDeform);
    if (entry.MorphTargets.Count > 0) MorphTargetBuffer.Release(entry.MorphTargets);
    entry = {};
    FreeIds.emplace_back(id);
}

Range MeshStore::AllocateVertices(uint32_t count) {
    const auto range = VerticesBuffer.Allocate(count);
    const auto required_size = static_cast<vk::DeviceSize>(range.Offset + range.Count) * sizeof(uint8_t);
    VertexStateBuffer.Reserve(required_size);
    VertexStateBuffer.UsedSize = std::max(VertexStateBuffer.UsedSize, required_size);
    return range;
}

Range MeshStore::AllocateFaces(uint32_t count) {
    const auto range = FaceNormalBuffer.Allocate(count);
    const auto required_size = static_cast<vk::DeviceSize>(range.Offset + range.Count) * sizeof(uint8_t);
    FaceStateBuffer.Reserve(required_size);
    FaceStateBuffer.UsedSize = std::max(FaceStateBuffer.UsedSize, required_size);
    return range;
}

std::span<uint8_t> MeshStore::GetFaceStates(Range range) {
    const auto bytes = FaceStateBuffer.GetMutableRange(range.Offset * sizeof(uint8_t), range.Count * sizeof(uint8_t));
    return {reinterpret_cast<uint8_t *>(bytes.data()), range.Count};
}

std::span<uint8_t> MeshStore::GetVertexStates(Range range) {
    const auto bytes = VertexStateBuffer.GetMutableRange(range.Offset * sizeof(uint8_t), range.Count * sizeof(uint8_t));
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
    auto face_states = GetFaceStates(entry.FaceNormals);
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

uint32_t MeshStore::AcquireId(Entry &&entry) {
    if (!FreeIds.empty()) {
        const auto id = FreeIds.back();
        FreeIds.pop_back();
        Entries[id] = std::move(entry);
        return id;
    }
    Entries.emplace_back(std::move(entry));
    return Entries.size() - 1;
}
