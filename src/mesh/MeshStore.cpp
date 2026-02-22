#include "MeshStore.h"

#include "MeshData.h"
#include "MorphTargetData.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

namespace {
constexpr uint8_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};

struct MeshDataWithMaterials {
    MeshData Mesh;
    std::vector<MaterialData> Materials;
};

MaterialData ToPbrMaterial(const tinyobj::material_t &material, uint32_t index) {
    // OBJ/MTL uses Phong terms; convert to glTF-style metallic-roughness with a common heuristic.
    const vec3 kd{material.diffuse[0], material.diffuse[1], material.diffuse[2]};
    const vec3 ks{material.specular[0], material.specular[1], material.specular[2]};

    const auto shininess = std::max(material.shininess, 0.f);
    const auto roughness = std::clamp(std::sqrt(2.f / (shininess + 2.f)), 0.04f, 1.f);
    const auto specular_strength = std::max(ks.x, std::max(ks.y, ks.z));
    const auto metallic = std::clamp((specular_strength - 0.04f) / (1.f - 0.04f), 0.f, 1.f);
    const auto base_color = glm::mix(kd, ks, metallic);
    const auto alpha = std::clamp(material.dissolve, 0.f, 1.f);
    auto name = material.name.empty() ? "Material" + std::to_string(index) : material.name;
    return {.BaseColorFactor = {base_color, alpha}, .MetallicFactor = metallic, .RoughnessFactor = roughness, .Name = std::move(name)};
}

MaterialData DefaultMaterial(std::string name = "Default") {
    return {.BaseColorFactor = {1.f, 1.f, 1.f, 1.f}, .MetallicFactor = 0.f, .RoughnessFactor = 1.f, .Name = std::move(name)};
}

MeshDataWithMaterials ReadObj(const std::filesystem::path &path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.string().c_str())) {
        throw std::runtime_error{"Failed to load OBJ: " + err};
    }

    MeshDataWithMaterials result{};
    auto &data = result.Mesh;
    data.Positions.reserve(attrib.vertices.size() / 3);
    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        data.Positions.emplace_back(attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]);
    }

    std::unordered_map<int, uint32_t> material_to_local_index;
    std::unordered_map<int, uint32_t> material_to_primitive;
    std::vector<uint32_t> face_primitive_indices;
    std::vector<uint32_t> primitive_material_indices;
    const auto local_material_index = [&](int material_id) -> uint32_t {
        if (const auto it = material_to_local_index.find(material_id); it != material_to_local_index.end()) return it->second;
        const auto local_index = static_cast<uint32_t>(result.Materials.size());
        if (material_id >= 0 && material_id < static_cast<int>(materials.size())) {
            result.Materials.emplace_back(ToPbrMaterial(materials[size_t(material_id)], uint32_t(material_id)));
        } else {
            result.Materials.emplace_back(DefaultMaterial());
        }
        material_to_local_index.emplace(material_id, local_index);
        return local_index;
    };
    const auto primitive_index = [&](int material_id) -> uint32_t {
        if (const auto it = material_to_primitive.find(material_id); it != material_to_primitive.end()) return it->second;
        const auto index = static_cast<uint32_t>(primitive_material_indices.size());
        primitive_material_indices.emplace_back(local_material_index(material_id));
        material_to_primitive.emplace(material_id, index);
        return index;
    };

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
            const int material_id = f < shape.mesh.material_ids.size() ? shape.mesh.material_ids[f] : -1;
            face_primitive_indices.emplace_back(primitive_index(material_id));
            vi += fv;
        }
    }

    if (!face_primitive_indices.empty()) {
        data.FacePrimitiveIndices = std::move(face_primitive_indices);
        data.PrimitiveMaterialIndices = std::move(primitive_material_indices);
    }
    return result;
}

template<typename T>
float NormalizeColor(T value) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::clamp(static_cast<float>(value), 0.f, 1.f);
    } else if constexpr (std::is_signed_v<T>) {
        const auto den = static_cast<float>(std::numeric_limits<T>::max());
        return std::clamp(static_cast<float>(value) / den, 0.f, 1.f);
    } else {
        const auto den = static_cast<float>(std::numeric_limits<T>::max());
        return std::clamp(static_cast<float>(value) / den, 0.f, 1.f);
    }
}

template<typename T>
vec3 AverageColor(tinyply::PlyData &colors) {
    const auto *values = reinterpret_cast<const T *>(colors.buffer.get());
    vec3 sum{0.f};
    for (size_t i = 0; i < colors.count; ++i) {
        sum.x += NormalizeColor(values[i * 3]);
        sum.y += NormalizeColor(values[i * 3 + 1]);
        sum.z += NormalizeColor(values[i * 3 + 2]);
    }
    return colors.count > 0 ? sum / static_cast<float>(colors.count) : vec3{1.f};
}

vec3 ComputeAverageVertexColor(tinyply::PlyData &colors) {
    switch (colors.t) {
        case tinyply::Type::UINT8: return AverageColor<uint8_t>(colors);
        case tinyply::Type::UINT16: return AverageColor<uint16_t>(colors);
        case tinyply::Type::UINT32: return AverageColor<uint32_t>(colors);
        case tinyply::Type::INT8: return AverageColor<int8_t>(colors);
        case tinyply::Type::INT16: return AverageColor<int16_t>(colors);
        case tinyply::Type::INT32: return AverageColor<int32_t>(colors);
        case tinyply::Type::FLOAT32: return AverageColor<float>(colors);
        case tinyply::Type::FLOAT64: return AverageColor<double>(colors);
        default: return {1.f, 1.f, 1.f};
    }
}

MeshDataWithMaterials ReadPly(const std::filesystem::path &path) {
    std::ifstream file{path, std::ios::binary};
    if (!file) throw std::runtime_error{"Failed to open: " + path.string()};

    tinyply::PlyFile ply_file;
    ply_file.parse_header(file);

    auto vertices = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
    std::shared_ptr<tinyply::PlyData> colors;
    try {
        colors = ply_file.request_properties_from_element("vertex", {"red", "green", "blue"});
    } catch (...) {
        try {
            colors = ply_file.request_properties_from_element("vertex", {"r", "g", "b"});
        } catch (...) {}
    }
    std::shared_ptr<tinyply::PlyData> faces;
    try {
        faces = ply_file.request_properties_from_element("face", {"vertex_indices"}, 0);
    } catch (...) {
        faces = ply_file.request_properties_from_element("face", {"vertex_index"}, 0);
    }
    ply_file.read(file);

    MeshDataWithMaterials result{};
    auto &data = result.Mesh;
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

    if (colors && colors->count == vertices->count && !data.Faces.empty()) {
        // Bake vertex colors down to one albedo value for now (no per-vertex color channel in the render path yet).
        const auto avg = ComputeAverageVertexColor(*colors);
        result.Materials.emplace_back(
            MaterialData{
                .BaseColorFactor = {avg.x, avg.y, avg.z, 1.f},
                .MetallicFactor = 0.f,
                .RoughnessFactor = 1.f,
                .Name = "VertexColor",
            }
        );
        data.FacePrimitiveIndices = std::vector<uint32_t>(data.Faces.size(), 0u);
        data.PrimitiveMaterialIndices = std::vector<uint32_t>{0u};
    }

    return result;
}

MeshData DeduplicateVertices(MeshData &&mesh) {
    struct VertexHash {
        constexpr size_t operator()(const vec3 &p) const noexcept {
            return std::hash<float>{}(p.x) ^ std::hash<float>{}(p.y) ^ std::hash<float>{}(p.z);
        }
    };

    MeshData deduped;
    deduped.Positions.reserve(mesh.Positions.size());
    if (mesh.Normals && mesh.Normals->size() == mesh.Positions.size()) {
        deduped.Normals = std::vector<vec3>{};
        deduped.Normals->reserve(mesh.Normals->size());
    }
    if (mesh.Tangents && mesh.Tangents->size() == mesh.Positions.size()) {
        deduped.Tangents = std::vector<vec4>{};
        deduped.Tangents->reserve(mesh.Tangents->size());
    }
    if (mesh.Colors0 && mesh.Colors0->size() == mesh.Positions.size()) {
        deduped.Colors0 = std::vector<vec4>{};
        deduped.Colors0->reserve(mesh.Colors0->size());
    }
    if (mesh.TexCoords0 && mesh.TexCoords0->size() == mesh.Positions.size()) {
        deduped.TexCoords0 = std::vector<vec2>{};
        deduped.TexCoords0->reserve(mesh.TexCoords0->size());
    }
    if (mesh.TexCoords1 && mesh.TexCoords1->size() == mesh.Positions.size()) {
        deduped.TexCoords1 = std::vector<vec2>{};
        deduped.TexCoords1->reserve(mesh.TexCoords1->size());
    }
    if (mesh.TexCoords2 && mesh.TexCoords2->size() == mesh.Positions.size()) {
        deduped.TexCoords2 = std::vector<vec2>{};
        deduped.TexCoords2->reserve(mesh.TexCoords2->size());
    }
    if (mesh.TexCoords3 && mesh.TexCoords3->size() == mesh.Positions.size()) {
        deduped.TexCoords3 = std::vector<vec2>{};
        deduped.TexCoords3->reserve(mesh.TexCoords3->size());
    }
    std::unordered_map<vec3, uint, VertexHash> index_by_vertex;
    std::vector<uint32_t> remap(mesh.Positions.size(), 0u);
    for (uint32_t i = 0; i < mesh.Positions.size(); ++i) {
        const auto &p = mesh.Positions[i];
        const auto [it, inserted] = index_by_vertex.try_emplace(p, deduped.Positions.size());
        remap[i] = it->second;
        if (inserted) {
            deduped.Positions.emplace_back(p);
            if (deduped.Normals) deduped.Normals->emplace_back((*mesh.Normals)[i]);
            if (deduped.Tangents) deduped.Tangents->emplace_back((*mesh.Tangents)[i]);
            if (deduped.Colors0) deduped.Colors0->emplace_back((*mesh.Colors0)[i]);
            if (deduped.TexCoords0) deduped.TexCoords0->emplace_back((*mesh.TexCoords0)[i]);
            if (deduped.TexCoords1) deduped.TexCoords1->emplace_back((*mesh.TexCoords1)[i]);
            if (deduped.TexCoords2) deduped.TexCoords2->emplace_back((*mesh.TexCoords2)[i]);
            if (deduped.TexCoords3) deduped.TexCoords3->emplace_back((*mesh.TexCoords3)[i]);
        }
    }

    deduped.Faces.reserve(mesh.Faces.size());
    for (const auto &face : mesh.Faces) {
        std::vector<uint> new_face;
        new_face.reserve(face.size());
        for (const auto idx : face) new_face.emplace_back(remap[idx]);
        deduped.Faces.emplace_back(std::move(new_face));
    }
    deduped.Edges.reserve(mesh.Edges.size());
    for (const auto &edge : mesh.Edges) deduped.Edges.emplace_back(std::array<uint32_t, 2>{remap[edge[0]], remap[edge[1]]});
    deduped.FacePrimitiveIndices = std::move(mesh.FacePrimitiveIndices);
    deduped.PrimitiveMaterialIndices = std::move(mesh.PrimitiveMaterialIndices);

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

std::vector<uint32_t> CreateFaceFirstTriIndices(const std::vector<std::vector<uint32_t>> &faces) {
    std::vector<uint32_t> first_tris(faces.size());
    uint32_t tri_offset = 0;
    for (uint32_t fi = 0; fi < faces.size(); ++fi) {
        first_tris[fi] = tri_offset;
        tri_offset += faces[fi].size() - 2;
    }
    return first_tris;
}
} // namespace

MeshStore::MeshStore(mvk::BufferContext &ctx)
    : FaceFirstTriangleBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer},
      FacePrimitiveBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::FacePrimitiveBuffer},
      PrimitiveMaterialBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::PrimitiveMaterialBuffer},
      BoneDeformBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::BoneDeformBuffer},
      MorphTargetBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MorphTargetBuffer},
      VerticesBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer},
      VertexStateBuffer{ctx, 1, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
      FaceStateBuffer{ctx, 1, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
      EdgeStateBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
      TriangleFaceIdBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer} {}

void MeshStore::UpdateNormals(const Mesh &mesh, bool skip_nonzero) {
    const auto face_cross = [&](Mesh::FH fh) {
        auto it = mesh.cfv_iter(fh);
        const auto p0 = mesh.GetPosition(*it), p1 = mesh.GetPosition(*++it), p2 = mesh.GetPosition(*++it);
        return glm::cross(p1 - p0, p2 - p0);
    };

    auto vertices = GetVertices(mesh.GetStoreId());
    if (skip_nonzero) {
        std::vector<bool> needs_normal(vertices.size());
        for (uint32_t i = 0; i < vertices.size(); ++i) needs_normal[i] = vertices[i].Normal == vec3{0};
        for (const auto fh : mesh.faces()) {
            const auto cross = face_cross(fh);
            for (const auto vh : mesh.fv_range(fh)) {
                if (needs_normal[*vh]) vertices[*vh].Normal += cross;
            }
        }
        for (uint32_t i = 0; i < vertices.size(); ++i) {
            if (needs_normal[i]) vertices[i].Normal = glm::normalize(vertices[i].Normal);
        }
    } else {
        for (auto &v : vertices) v.Normal = vec3{0};
        for (const auto fh : mesh.faces()) {
            const auto cross = face_cross(fh);
            for (const auto vh : mesh.fv_range(fh)) vertices[*vh].Normal += cross;
        }
        for (auto &v : vertices) v.Normal = glm::normalize(v.Normal);
    }
}

Mesh MeshStore::CreateMesh(MeshData &&data, std::optional<ArmatureDeformData> deform_data, std::optional<MorphTargetData> morph_data) {
    if (deform_data && (deform_data->Joints.size() != data.Positions.size() || deform_data->Weights.size() != data.Positions.size())) {
        throw std::runtime_error{"ArmatureDeformData channel counts must match the position count."};
    }
    const auto vertex_count = static_cast<uint32_t>(data.Positions.size());
    if (data.TexCoords0 && data.TexCoords0->size() != vertex_count) throw std::runtime_error{"MeshData.TexCoords0 must match vertex count."};
    if (data.TexCoords1 && data.TexCoords1->size() != vertex_count) throw std::runtime_error{"MeshData.TexCoords1 must match vertex count."};
    if (data.TexCoords2 && data.TexCoords2->size() != vertex_count) throw std::runtime_error{"MeshData.TexCoords2 must match vertex count."};
    if (data.TexCoords3 && data.TexCoords3->size() != vertex_count) throw std::runtime_error{"MeshData.TexCoords3 must match vertex count."};
    if (data.Tangents && data.Tangents->size() != vertex_count) throw std::runtime_error{"MeshData.Tangents must match vertex count."};
    if (data.Colors0 && data.Colors0->size() != vertex_count) throw std::runtime_error{"MeshData.Colors0 must match vertex count."};
    if (data.FacePrimitiveIndices && data.FacePrimitiveIndices->size() != data.Faces.size()) throw std::runtime_error{"MeshData.FacePrimitiveIndices must match face count."};
    const auto vertices = AllocateVertices(vertex_count);
    auto vertex_span = VerticesBuffer.GetMutable(vertices);
    for (uint32_t i = 0; i < vertex_count; ++i) {
        vertex_span[i] = {
            .Position = data.Positions[i],
            .Tangent = data.Tangents ? (*data.Tangents)[i] : vec4{0.f, 0.f, 0.f, 1.f},
            .Color = data.Colors0 ? (*data.Colors0)[i] : vec4{1.f},
            .TexCoord0 = data.TexCoords0 ? (*data.TexCoords0)[i] : vec2{0},
            .TexCoord1 = data.TexCoords1 ? (*data.TexCoords1)[i] : vec2{0},
            .TexCoord2 = data.TexCoords2 ? (*data.TexCoords2)[i] : vec2{0},
            .TexCoord3 = data.TexCoords3 ? (*data.TexCoords3)[i] : vec2{0},
        };
    }
    Range bone_deform{};
    if (deform_data) {
        bone_deform = BoneDeformBuffer.Allocate(vertex_count);
        auto bd_span = BoneDeformBuffer.GetMutable(bone_deform);
        for (uint32_t i = 0; i < vertex_count; ++i) {
            bd_span[i] = BoneDeformVertex{
                .Joints = deform_data->Joints[i],
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
        const bool has_normal_deltas = !morph_data->NormalDeltas.empty();
        for (uint32_t i = 0; i < total; ++i) {
            mt_span[i] = MorphTargetVertex{
                .PositionDelta = morph_data->PositionDeltas[i],
                .NormalDelta = has_normal_deltas ? morph_data->NormalDeltas[i] : vec3{0},
            };
        }
        default_morph_weights = std::move(morph_data->DefaultWeights);
        default_morph_weights.resize(morph_target_count, 0.f);
    }

    Range faces{}, triangle_face_ids{}, face_primitives{}, primitive_materials{};
    if (!data.Faces.empty()) {
        const auto first_tris = CreateFaceFirstTriIndices(data.Faces);
        faces = AllocateFaces(data.Faces.size());
        std::ranges::copy(first_tris, FaceFirstTriangleBuffer.GetMutable(faces).begin());
        triangle_face_ids = TriangleFaceIdBuffer.Allocate(CreateFaceElementIds(data.Faces));

        std::vector<uint32_t> face_primitive_indices(data.Faces.size(), 0u);
        if (data.FacePrimitiveIndices) face_primitive_indices = *data.FacePrimitiveIndices;
        face_primitives = FacePrimitiveBuffer.Allocate(face_primitive_indices);

        const auto primitive_count = data.FacePrimitiveIndices ?
            (face_primitive_indices.empty() ? 0u : *std::ranges::max_element(face_primitive_indices) + 1u) :
            1u;
        if (data.PrimitiveMaterialIndices && data.PrimitiveMaterialIndices->size() != primitive_count) {
            throw std::runtime_error{"MeshData.PrimitiveMaterialIndices must match primitive count."};
        }
        std::vector<uint32_t> primitive_material_indices(primitive_count, 0u);
        if (data.PrimitiveMaterialIndices) primitive_material_indices = *data.PrimitiveMaterialIndices;
        primitive_materials = PrimitiveMaterialBuffer.Allocate(primitive_material_indices);
    }

    const auto id = AcquireId({
        .Vertices = vertices,
        .FaceData = faces,
        .EdgeStates = {},
        .TriangleFaceIds = triangle_face_ids,
        .FacePrimitives = face_primitives,
        .PrimitiveMaterials = primitive_materials,
        .BoneDeform = bone_deform,
        .MorphTargets = morph_targets,
        .MorphTargetCount = morph_target_count,
        .DefaultMorphWeights = std::move(default_morph_weights),
        .Alive = true,
    });

    auto mesh = [&]() -> Mesh {
        if (!data.Faces.empty()) return {*this, id, std::move(data.Faces)};
        if (!data.Edges.empty()) return {*this, id, std::move(data.Edges), vertex_count};
        return {*this, id, vertex_count};
    }();

    auto &entry = Entries[id];

    if (!data.Faces.empty()) {
        if (data.Normals) {
            auto verts = GetVertices(id);
            for (uint32_t i = 0; i < vertex_count; ++i) verts[i].Normal = (*data.Normals)[i];
            UpdateNormals(mesh, /*skip_nonzero=*/true);
        } else {
            UpdateNormals(mesh);
        }
    }

    entry.EdgeStates = EdgeStateBuffer.Allocate(mesh.EdgeCount() * 2);
    std::ranges::fill(GetVertexStates(vertices), 0);
    if (faces.Count > 0) std::ranges::fill(GetFaceStates(faces), 0);
    if (entry.EdgeStates.Count > 0) std::ranges::fill(EdgeStateBuffer.GetMutable(entry.EdgeStates), 0);
    return mesh;
}

Mesh MeshStore::CloneMesh(const Mesh &mesh) {
    const auto src_id = mesh.GetStoreId();
    const auto src_vertices = GetVertices(src_id);
    const auto vertices = AllocateVertices(src_vertices.size());
    std::ranges::copy(src_vertices, VerticesBuffer.GetMutable(vertices).begin());

    const auto faces = AllocateFaces(mesh.FaceCount());
    std::ranges::copy(FaceFirstTriangleBuffer.Get(Entries.at(src_id).FaceData), FaceFirstTriangleBuffer.GetMutable(faces).begin());

    const auto edge_states = EdgeStateBuffer.Allocate(mesh.EdgeCount() * 2);
    const auto &src_entry = Entries.at(src_id);
    const auto id = AcquireId({
        .Vertices = vertices,
        .FaceData = faces,
        .EdgeStates = edge_states,
        .TriangleFaceIds = TriangleFaceIdBuffer.Allocate(TriangleFaceIdBuffer.Get(src_entry.TriangleFaceIds)),
        .FacePrimitives = src_entry.FacePrimitives.Count > 0 ? FacePrimitiveBuffer.Allocate(FacePrimitiveBuffer.Get(src_entry.FacePrimitives)) : Range{},
        .PrimitiveMaterials = src_entry.PrimitiveMaterials.Count > 0 ? PrimitiveMaterialBuffer.Allocate(PrimitiveMaterialBuffer.Get(src_entry.PrimitiveMaterials)) : Range{},
        .BoneDeform = src_entry.BoneDeform.Count > 0 ? BoneDeformBuffer.Allocate(BoneDeformBuffer.Get(src_entry.BoneDeform)) : Range{},
        .MorphTargets = src_entry.MorphTargets.Count > 0 ? MorphTargetBuffer.Allocate(MorphTargetBuffer.Get(src_entry.MorphTargets)) : Range{},
        .MorphTargetCount = src_entry.MorphTargetCount,
        .DefaultMorphWeights = src_entry.DefaultMorphWeights,
        .Alive = true,
    });

    std::ranges::fill(GetVertexStates(vertices), 0);
    std::ranges::fill(GetFaceStates(faces), 0);
    std::ranges::fill(EdgeStateBuffer.GetMutable(edge_states), 0);
    return {*this, id, mesh};
}

std::expected<MeshWithMaterials, std::string> MeshStore::LoadMesh(const std::filesystem::path &path) {
    auto ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    MeshDataWithMaterials source_data;
    try {
        if (ext == ".ply") source_data = ReadPly(path);
        else if (ext == ".obj") source_data = ReadObj(path);
        else return std::unexpected{"Unsupported file format: " + ext};
    } catch (const std::exception &e) {
        return std::unexpected{e.what()};
    }
    return MeshWithMaterials{
        .Mesh = CreateMesh(DeduplicateVertices(std::move(source_data.Mesh))),
        .Materials = std::move(source_data.Materials),
    };
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
    FaceFirstTriangleBuffer.Release(entry.FaceData);
    FacePrimitiveBuffer.Release(entry.FacePrimitives);
    PrimitiveMaterialBuffer.Release(entry.PrimitiveMaterials);
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
    const auto range = FaceFirstTriangleBuffer.Allocate(count);
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
    auto face_states = GetFaceStates(entry.FaceData);
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
