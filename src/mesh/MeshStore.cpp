#include "MeshStore.h"
#include "MeshAttributes.h"
#include "MeshData.h"

#include <limits>
#include <numeric>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

namespace {
constexpr uint8_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};

struct MeshDataWithMaterials {
    MeshData Mesh;
    MeshVertexAttributes Attrs;
    MeshPrimitives Primitives;
    std::vector<ObjPlyMaterial> Materials;
};

std::optional<std::filesystem::path> ResolveTexturePath(const std::filesystem::path &base_dir, const std::string &texture_name) {
    if (texture_name.empty()) return std::nullopt;
    auto texture_path = std::filesystem::path{texture_name};
    if (texture_path.is_relative()) texture_path = base_dir / texture_path;
    if (texture_path.is_relative()) texture_path = std::filesystem::absolute(texture_path);
    return texture_path.lexically_normal();
}

ObjPlyMaterial ToObjPlyMaterial(const tinyobj::material_t &material, uint32_t index, const std::filesystem::path &base_dir) {
    // OBJ/MTL uses Phong terms; convert to metallic-roughness with a common heuristic.
    const vec3 kd{material.diffuse[0], material.diffuse[1], material.diffuse[2]};
    const vec3 ks{material.specular[0], material.specular[1], material.specular[2]};

    const auto shininess = std::max(material.shininess, 0.f);
    const auto roughness = std::clamp(std::sqrt(2.f / (shininess + 2.f)), 0.04f, 1.f);
    const auto specular_strength = std::max(ks.x, std::max(ks.y, ks.z));
    const auto metallic = std::clamp((specular_strength - 0.04f) / (1.f - 0.04f), 0.f, 1.f);
    const auto base_color = glm::mix(kd, ks, metallic);
    const auto alpha = std::clamp(material.dissolve, 0.f, 1.f);
    auto name = material.name.empty() ? "Material" + std::to_string(index) : material.name;
    return {
        .BaseColorFactor = {base_color, alpha},
        .MetallicFactor = metallic,
        .RoughnessFactor = roughness,
        .Name = std::move(name),
        .BaseColorTexturePath = ResolveTexturePath(base_dir, material.diffuse_texname),
        .NormalTexturePath = ResolveTexturePath(base_dir, material.normal_texname.empty() ? material.bump_texname : material.normal_texname),
        .HasAlphaTexture = !material.alpha_texname.empty(),
    };
}

ObjPlyMaterial DefaultMaterial(std::string name = "Default") {
    return {.BaseColorFactor = {1.f, 1.f, 1.f, 1.f}, .MetallicFactor = 0.f, .RoughnessFactor = 1.f, .Name = std::move(name)};
}

MeshDataWithMaterials ReadObj(const std::filesystem::path &path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    const auto mtl_base_dir = path.parent_path().string();
    if (!tinyobj::LoadObj(
            &attrib,
            &shapes,
            &materials,
            &warn,
            &err,
            path.string().c_str(),
            mtl_base_dir.empty() ? nullptr : mtl_base_dir.c_str()
        )) {
        throw std::runtime_error{"Failed to load OBJ: " + err};
    }

    MeshDataWithMaterials result{};
    auto &data = result.Mesh;
    auto &attrs = result.Attrs;
    const bool has_normals = !attrib.normals.empty();
    const bool has_texcoords = !attrib.texcoords.empty();
    if (has_normals) attrs.Normals = std::vector<vec3>{};
    if (has_texcoords) attrs.TexCoords0 = std::vector<vec2>{};

    struct ObjVertexKey {
        int PositionIndex;
        int NormalIndex;
        int TexCoordIndex;
        bool operator==(const ObjVertexKey &other) const = default;
    };
    struct ObjVertexKeyHash {
        size_t operator()(const ObjVertexKey &key) const noexcept {
            size_t hash = std::hash<int>{}(key.PositionIndex);
            hash ^= (std::hash<int>{}(key.NormalIndex) << 1u);
            hash ^= (std::hash<int>{}(key.TexCoordIndex) << 2u);
            return hash;
        }
    };

    std::unordered_map<ObjVertexKey, uint32_t, ObjVertexKeyHash> vertex_cache;
    const auto FindOrAddVertex = [&](const tinyobj::index_t &index) -> uint32_t {
        const ObjVertexKey key{.PositionIndex = index.vertex_index, .NormalIndex = index.normal_index, .TexCoordIndex = index.texcoord_index};
        if (const auto it = vertex_cache.find(key); it != vertex_cache.end()) return it->second;

        if (index.vertex_index < 0 || size_t(index.vertex_index) * 3u + 2u >= attrib.vertices.size()) {
            throw std::runtime_error{std::format("OBJ '{}' references invalid vertex index {}.", path.string(), index.vertex_index)};
        }

        const auto vertex_index = static_cast<uint32_t>(data.Positions.size());
        data.Positions.emplace_back(
            attrib.vertices[size_t(index.vertex_index) * 3u],
            attrib.vertices[size_t(index.vertex_index) * 3u + 1u],
            attrib.vertices[size_t(index.vertex_index) * 3u + 2u]
        );

        if (attrs.Normals) {
            if (index.normal_index >= 0 && size_t(index.normal_index) * 3u + 2u < attrib.normals.size()) {
                attrs.Normals->emplace_back(
                    attrib.normals[size_t(index.normal_index) * 3u],
                    attrib.normals[size_t(index.normal_index) * 3u + 1u],
                    attrib.normals[size_t(index.normal_index) * 3u + 2u]
                );
            } else {
                attrs.Normals->emplace_back(vec3{0.f});
            }
        }

        if (attrs.TexCoords0) {
            if (index.texcoord_index >= 0 && size_t(index.texcoord_index) * 2u + 1u < attrib.texcoords.size()) {
                attrs.TexCoords0->emplace_back(
                    attrib.texcoords[size_t(index.texcoord_index) * 2u],
                    attrib.texcoords[size_t(index.texcoord_index) * 2u + 1u]
                );
            } else {
                attrs.TexCoords0->emplace_back(vec2{0.f});
            }
        }

        vertex_cache.emplace(key, vertex_index);
        return vertex_index;
    };

    std::unordered_map<int, uint32_t> material_to_local_index;
    std::unordered_map<int, uint32_t> material_to_primitive;
    std::vector<uint32_t> face_primitive_indices;
    std::vector<uint32_t> primitive_material_indices;
    const auto local_material_index = [&](int material_id) -> uint32_t {
        if (const auto it = material_to_local_index.find(material_id); it != material_to_local_index.end()) return it->second;
        const auto local_index = static_cast<uint32_t>(result.Materials.size());
        if (material_id >= 0 && material_id < static_cast<int>(materials.size())) {
            result.Materials.emplace_back(ToObjPlyMaterial(materials[size_t(material_id)], uint32_t(material_id), path.parent_path()));
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
                face_verts.emplace_back(FindOrAddVertex(shape.mesh.indices[vi + vi_f]));
            }
            data.Faces.emplace_back(std::move(face_verts));
            const int material_id = f < shape.mesh.material_ids.size() ? shape.mesh.material_ids[f] : -1;
            face_primitive_indices.emplace_back(primitive_index(material_id));
            vi += fv;
        }
    }

    if (!face_primitive_indices.empty()) {
        result.Primitives.FacePrimitiveIndices = std::move(face_primitive_indices);
        result.Primitives.MaterialIndices = std::move(primitive_material_indices);
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
        result.Materials.emplace_back(ObjPlyMaterial{.BaseColorFactor = {avg.x, avg.y, avg.z, 1.f}, .MetallicFactor = 0.f, .RoughnessFactor = 1.f, .Name = "VertexColor"});
        result.Primitives.FacePrimitiveIndices = std::vector<uint32_t>(data.Faces.size(), 0u);
        result.Primitives.MaterialIndices = std::vector<uint32_t>{0u};
    }

    return result;
}

std::pair<MeshData, MeshVertexAttributes> DeduplicateVertices(MeshData &&mesh, MeshVertexAttributes &&attrs) {
    // todo we shouldn't deduplicate vertices when textures are present.
    // position-only merging split causes UV seams / hard normals get collapsed,
    // which breaks texture mapping and shading.
    // However, this breaks RealImpact/excitation behavior.
    // const bool has_vertex_channels =
    //     attrs.Normals.has_value() ||
    //     attrs.Tangents.has_value() ||
    //     attrs.Colors0.has_value() ||
    //     attrs.TexCoords0.has_value() ||
    //     attrs.TexCoords1.has_value() ||
    //     attrs.TexCoords2.has_value() ||
    //     attrs.TexCoords3.has_value();
    // if (has_vertex_channels) return {std::move(mesh), std::move(attrs)};

    struct VertexHash {
        constexpr size_t operator()(const vec3 &p) const noexcept {
            return std::hash<float>{}(p.x) ^ std::hash<float>{}(p.y) ^ std::hash<float>{}(p.z);
        }
    };

    MeshData deduped;
    MeshVertexAttributes deduped_attrs;
    deduped.Positions.reserve(mesh.Positions.size());
    auto init_attr = [&](auto &dst, const auto &src) {
        if (src && src->size() == mesh.Positions.size()) {
            dst.emplace();
            dst->reserve(src->size());
        }
    };
    init_attr(deduped_attrs.Normals, attrs.Normals);
    init_attr(deduped_attrs.Tangents, attrs.Tangents);
    init_attr(deduped_attrs.Colors0, attrs.Colors0);
    init_attr(deduped_attrs.TexCoords0, attrs.TexCoords0);
    init_attr(deduped_attrs.TexCoords1, attrs.TexCoords1);
    init_attr(deduped_attrs.TexCoords2, attrs.TexCoords2);
    init_attr(deduped_attrs.TexCoords3, attrs.TexCoords3);
    std::unordered_map<vec3, uint, VertexHash> index_by_vertex;
    std::vector<uint32_t> remap(mesh.Positions.size(), 0u);
    for (uint32_t i = 0; i < mesh.Positions.size(); ++i) {
        const auto &p = mesh.Positions[i];
        const auto [it, inserted] = index_by_vertex.try_emplace(p, deduped.Positions.size());
        remap[i] = it->second;
        if (inserted) {
            deduped.Positions.emplace_back(p);
            if (deduped_attrs.Normals) deduped_attrs.Normals->emplace_back((*attrs.Normals)[i]);
            if (deduped_attrs.Tangents) deduped_attrs.Tangents->emplace_back((*attrs.Tangents)[i]);
            if (deduped_attrs.Colors0) deduped_attrs.Colors0->emplace_back((*attrs.Colors0)[i]);
            if (deduped_attrs.TexCoords0) deduped_attrs.TexCoords0->emplace_back((*attrs.TexCoords0)[i]);
            if (deduped_attrs.TexCoords1) deduped_attrs.TexCoords1->emplace_back((*attrs.TexCoords1)[i]);
            if (deduped_attrs.TexCoords2) deduped_attrs.TexCoords2->emplace_back((*attrs.TexCoords2)[i]);
            if (deduped_attrs.TexCoords3) deduped_attrs.TexCoords3->emplace_back((*attrs.TexCoords3)[i]);
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

    return {std::move(deduped), std::move(deduped_attrs)};
}

} // namespace

MeshStore::MeshStore(mvk::BufferContext &ctx)
    : FaceFirstTriangleBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer},
      FacePrimitiveBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::FacePrimitiveBuffer},
      PrimitiveMaterialBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::PrimitiveMaterialBuffer},
      BoneDeformBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::BoneDeformBuffer},
      MorphTargetBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MorphTargetBuffer},
      VerticesBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer},
      VertexStateBuffer{ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
      FaceStateBuffer{ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
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

std::pair<uint32_t, Range> MeshStore::AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &attrs) {
    const auto vertex_count = static_cast<uint32_t>(positions.size());
    const auto vertices = AllocateVertices(vertex_count);
    auto vertex_span = VerticesBuffer.GetMutable(vertices);
    for (uint32_t i = 0; i < vertex_count; ++i) {
        vertex_span[i] = {
            .Position = positions[i],
            .Tangent = attrs.Tangents ? (*attrs.Tangents)[i] : vec4{0.f, 0.f, 0.f, 1.f},
            .Color = attrs.Colors0 ? (*attrs.Colors0)[i] : vec4{1.f},
            .TexCoord0 = attrs.TexCoords0 ? (*attrs.TexCoords0)[i] : vec2{0},
            .TexCoord1 = attrs.TexCoords1 ? (*attrs.TexCoords1)[i] : vec2{0},
            .TexCoord2 = attrs.TexCoords2 ? (*attrs.TexCoords2)[i] : vec2{0},
            .TexCoord3 = attrs.TexCoords3 ? (*attrs.TexCoords3)[i] : vec2{0},
        };
    }
    if (attrs.Normals) {
        for (uint32_t i = 0; i < vertex_count; ++i) vertex_span[i].Normal = (*attrs.Normals)[i];
    }
    const auto id = AcquireId({.Vertices = vertices, .FaceData = {}, .Alive = true});
    return {id, vertices};
}

void MeshStore::PlanCreate(const MeshData &data, const MeshPrimitives &primitives, bool has_deform, uint32_t morph_target_count) {
    const auto vertices = static_cast<uint32_t>(data.Positions.size());
    const auto faces = static_cast<uint32_t>(data.Faces.size());
    uint32_t triangles = 0;
    for (const auto &face : data.Faces) triangles += face.size() - 2;
    Pending.Vertices += vertices;
    Pending.Faces += faces;
    Pending.Triangles += triangles;
    Pending.EdgeStates += triangles + 2 * faces; // manifold estimate: halfedges ≈ triangles + 2*faces
    Pending.Primitives += primitives.MaterialIndices.size();
    if (has_deform) Pending.BoneDeformVertices += vertices;
    if (morph_target_count > 0) Pending.MorphTargetEntries += morph_target_count * vertices;
}

void MeshStore::PlanClone(const Mesh &mesh) {
    const auto &e = Entries.at(mesh.GetStoreId());
    Pending.Vertices += e.Vertices.Count;
    Pending.Faces += e.FaceData.Count;
    Pending.Triangles += e.TriangleFaceIds.Count;
    Pending.EdgeStates += e.EdgeStates.Count;
    Pending.Primitives += e.PrimitiveMaterials.Count;
    Pending.BoneDeformVertices += e.BoneDeform.Count;
    Pending.MorphTargetEntries += e.MorphTargets.Count;
}

void MeshStore::CommitReserves() {
    VerticesBuffer.ReserveAdditional(Pending.Vertices);
    FaceFirstTriangleBuffer.ReserveAdditional(Pending.Faces);
    FacePrimitiveBuffer.ReserveAdditional(Pending.Faces);
    TriangleFaceIdBuffer.ReserveAdditional(Pending.Triangles);
    EdgeStateBuffer.ReserveAdditional(Pending.EdgeStates);
    PrimitiveMaterialBuffer.ReserveAdditional(Pending.Primitives);
    BoneDeformBuffer.ReserveAdditional(Pending.BoneDeformVertices);
    MorphTargetBuffer.ReserveAdditional(Pending.MorphTargetEntries);
    // Mirror buffers (uint8_t state per element, no arena — shared ranges with data arenas).
    if (Pending.Vertices > 0) VertexStateBuffer.Reserve(VertexStateBuffer.UsedSize + Pending.Vertices);
    if (Pending.Faces > 0) FaceStateBuffer.Reserve(FaceStateBuffer.UsedSize + Pending.Faces);
    Pending = {};
}

Mesh MeshStore::CreateMesh(MeshData &&data, MeshVertexAttributes &&attrs, MeshPrimitives &&primitives, std::optional<ArmatureDeformData> deform, std::optional<MorphTargetData> morph) {
    const auto vertex_count = static_cast<uint32_t>(data.Positions.size());
    auto [id, vertices] = AllocateVertexBuffer(data.Positions, attrs);
    auto &entry = Entries[id];

    if (deform) {
        entry.BoneDeform = BoneDeformBuffer.Allocate(vertex_count);
        auto bd_span = BoneDeformBuffer.GetMutable(entry.BoneDeform);
        for (uint32_t i = 0; i < vertex_count; ++i) {
            bd_span[i] = {.Joints = deform->Joints[i], .Weights = deform->Weights[i]};
        }
    }
    if (morph && morph->TargetCount > 0 && vertex_count > 0) {
        entry.MorphTargetCount = morph->TargetCount;
        const auto total = entry.MorphTargetCount * vertex_count;
        entry.MorphTargets = MorphTargetBuffer.Allocate(total);
        auto mt_span = MorphTargetBuffer.GetMutable(entry.MorphTargets);
        const bool has_normal_deltas = !morph->NormalDeltas.empty();
        for (uint32_t i = 0; i < total; ++i) {
            mt_span[i] = MorphTargetVertex{
                .PositionDelta = morph->PositionDeltas[i],
                .NormalDelta = has_normal_deltas ? morph->NormalDeltas[i] : vec3{0},
            };
        }
        entry.DefaultMorphWeights = std::move(morph->DefaultWeights);
        entry.DefaultMorphWeights.resize(entry.MorphTargetCount, 0.f);
    }

    if (!data.Faces.empty()) {
        const auto face_count = static_cast<uint32_t>(data.Faces.size());

        // Sort faces by primitive index so triangles are grouped by primitive in the index buffer.
        if (!primitives.FacePrimitiveIndices.empty() && primitives.FacePrimitiveIndices.size() == face_count &&
            !std::ranges::all_of(primitives.FacePrimitiveIndices, [&](uint32_t pi) { return pi == primitives.FacePrimitiveIndices[0]; })) {
            std::vector<uint32_t> perm(face_count);
            std::iota(perm.begin(), perm.end(), 0u);
            std::stable_sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) {
                return primitives.FacePrimitiveIndices[a] < primitives.FacePrimitiveIndices[b];
            });
            bool already_sorted = true;
            for (uint32_t i = 0; i < face_count; ++i) {
                if (perm[i] != i) {
                    already_sorted = false;
                    break;
                }
            }
            if (!already_sorted) {
                std::vector<std::vector<uint32_t>> sorted_faces(face_count);
                std::vector<uint32_t> sorted_fpi(face_count);
                for (uint32_t i = 0; i < face_count; ++i) {
                    sorted_faces[i] = std::move(data.Faces[perm[i]]);
                    sorted_fpi[i] = primitives.FacePrimitiveIndices[perm[i]];
                }
                data.Faces = std::move(sorted_faces);
                primitives.FacePrimitiveIndices = std::move(sorted_fpi);
            }
        }

        // Write face-first-triangle offsets directly into GPU buffer.
        entry.FaceData = AllocateFaces(face_count);
        auto first_tri_span = FaceFirstTriangleBuffer.GetMutable(entry.FaceData);
        uint32_t tri_offset = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            first_tri_span[fi] = tri_offset;
            tri_offset += data.Faces[fi].size() - 2;
        }

        entry.TriangleCount = tri_offset;

        // Write triangle-to-face IDs directly into GPU buffer.
        entry.TriangleFaceIds = TriangleFaceIdBuffer.Allocate(tri_offset);
        auto tri_face_span = TriangleFaceIdBuffer.GetMutable(entry.TriangleFaceIds);
        uint32_t ti = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            const auto n_tris = data.Faces[fi].size() - 2;
            for (size_t t = 0; t < n_tris; ++t) tri_face_span[ti++] = fi + 1;
        }

        // Write face-to-primitive mapping directly into GPU buffer.
        entry.FacePrimitives = FacePrimitiveBuffer.Allocate(face_count);
        auto fp_span = FacePrimitiveBuffer.GetMutable(entry.FacePrimitives);
        if (!primitives.FacePrimitiveIndices.empty()) {
            std::ranges::copy(primitives.FacePrimitiveIndices, fp_span.begin());
        } else {
            std::ranges::fill(fp_span, 0u);
        }

        const auto primitive_count = !primitives.MaterialIndices.empty() ?
            (primitives.FacePrimitiveIndices.empty() ? 1u : *std::ranges::max_element(primitives.FacePrimitiveIndices) + 1u) :
            1u;
        entry.PrimitiveMaterials = PrimitiveMaterialBuffer.Allocate(primitive_count);
        auto pm_span = PrimitiveMaterialBuffer.GetMutable(entry.PrimitiveMaterials);
        if (!primitives.MaterialIndices.empty()) {
            std::ranges::copy(primitives.MaterialIndices, pm_span.begin());
        } else {
            std::ranges::fill(pm_span, 0u);
        }

        // Compute per-primitive triangle ranges from the (now sorted) face data.
        {
            const auto fp = FacePrimitiveBuffer.Get(entry.FacePrimitives);
            const auto fft = FaceFirstTriangleBuffer.Get(entry.FaceData);
            auto &ranges = entry.PrimitiveTriangleRanges;
            if (face_count > 0) {
                uint32_t current_prim = fp[0];
                uint32_t range_first_tri = fft[0];
                for (uint32_t fi = 1; fi < face_count; ++fi) {
                    if (fp[fi] != current_prim) {
                        ranges.push_back({current_prim, range_first_tri, fft[fi] - range_first_tri});
                        current_prim = fp[fi];
                        range_first_tri = fft[fi];
                    }
                }
                ranges.push_back({current_prim, range_first_tri, entry.TriangleCount - range_first_tri});
            }
        }
    }

    auto mesh = [&]() -> Mesh {
        if (!data.Faces.empty()) return {*this, id, std::move(data.Faces)};
        if (!data.Edges.empty()) return {*this, id, std::move(data.Edges), vertex_count};
        return {*this, id, vertex_count};
    }();

    if (!data.Faces.empty() && !attrs.Normals) {
        UpdateNormals(mesh);
    }

    entry.EdgeStates = EdgeStateBuffer.Allocate(mesh.EdgeCount() * 2);
    ClearElementStates(vertices, entry.FaceData, entry.EdgeStates);
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
        .TriangleCount = src_entry.TriangleCount,
        .DefaultMorphWeights = src_entry.DefaultMorphWeights,
        .PrimitiveTriangleRanges = src_entry.PrimitiveTriangleRanges,
        .Alive = true,
    });

    ClearElementStates(vertices, faces, edge_states);
    return {*this, id, mesh};
}

std::expected<MeshWithMaterials, std::string> MeshStore::LoadMesh(const std::filesystem::path &path) {
    auto ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    MeshDataWithMaterials source;
    try {
        if (ext == ".ply") source = ReadPly(path);
        else if (ext == ".obj") source = ReadObj(path);
        else return std::unexpected{"Unsupported file format: " + ext};
    } catch (const std::exception &e) {
        return std::unexpected{e.what()};
    }
    auto [mesh, attrs] = DeduplicateVertices(std::move(source.Mesh), std::move(source.Attrs));
    return MeshWithMaterials{
        .Mesh = CreateMesh(std::move(mesh), std::move(attrs), std::move(source.Primitives)),
        .Materials = std::move(source.Materials),
    };
}

void MeshStore::SetPositions(const Mesh &mesh, std::span<const vec3> positions) {
    auto vertex_span = VerticesBuffer.GetMutable(Entries.at(mesh.GetStoreId()).Vertices);
    for (size_t i = 0; i < positions.size(); ++i) vertex_span[i].Position = positions[i];
    UpdateNormals(mesh);
}
void MeshStore::SetPositions(uint32_t store_id, std::span<const vec3> positions) {
    auto vertex_span = VerticesBuffer.GetMutable(Entries.at(store_id).Vertices);
    for (size_t i = 0; i < positions.size(); ++i) vertex_span[i].Position = positions[i];
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

// Mirror buffer helpers — uint8_t state per element, sizeof == 1.
static void SyncMirror(mvk::Buffer &mirror, Range range) {
    auto end = vk::DeviceSize(range.Offset + range.Count);
    mirror.Reserve(end);
    mirror.UsedSize = std::max(mirror.UsedSize, end);
}
static std::span<uint8_t> GetStates(mvk::Buffer &buf, Range range) {
    return {reinterpret_cast<uint8_t *>(buf.GetMutableRange(range.Offset, range.Count).data()), range.Count};
}

Range MeshStore::AllocateVertices(uint32_t count) {
    const auto range = VerticesBuffer.Allocate(count);
    SyncMirror(VertexStateBuffer, range);
    return range;
}

Range MeshStore::AllocateFaces(uint32_t count) {
    const auto range = FaceFirstTriangleBuffer.Allocate(count);
    SyncMirror(FaceStateBuffer, range);
    return range;
}

std::span<uint8_t> MeshStore::GetFaceStates(Range range) { return GetStates(FaceStateBuffer, range); }
std::span<uint8_t> MeshStore::GetVertexStates(Range range) { return GetStates(VertexStateBuffer, range); }

std::span<const uint8_t> MeshStore::GetVertexStates(Range range) const {
    return {reinterpret_cast<const uint8_t *>(VertexStateBuffer.GetMappedData().subspan(range.Offset, range.Count).data()), range.Count};
}

std::span<const uint8_t> MeshStore::GetVertexStates(uint32_t id) const {
    return GetVertexStates(Entries.at(id).Vertices);
}

void MeshStore::ClearElementStates(Range vertices, Range faces, Range edges) {
    std::ranges::fill(GetVertexStates(vertices), 0);
    if (faces.Count > 0) std::ranges::fill(GetFaceStates(faces), 0);
    if (edges.Count > 0) std::ranges::fill(EdgeStateBuffer.GetMutable(edges), 0);
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

void MeshStore::UpdateEdgeStatesFromFaces(const Mesh &mesh, std::span<const uint32_t> selected_faces, std::optional<uint32_t> active_face) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    auto edge_states = EdgeStateBuffer.GetMutable(entry.EdgeStates);
    std::ranges::fill(edge_states, uint8_t{0});

    for (const auto fi : selected_faces) {
        if (fi >= mesh.FaceCount()) continue;
        for (const auto heh : mesh.fh_range(FH{fi})) {
            const auto ei = *mesh.GetEdge(heh);
            edge_states[2 * ei] |= ElementStateSelected;
            edge_states[2 * ei + 1] |= ElementStateSelected;
        }
    }

    if (active_face && *active_face < mesh.FaceCount()) {
        for (const auto heh : mesh.fh_range(FH{*active_face})) {
            const auto ei = *mesh.GetEdge(heh);
            edge_states[2 * ei] |= ElementStateActive;
            edge_states[2 * ei + 1] |= ElementStateActive;
        }
    }
}

void MeshStore::UpdateEdgeStatesFromVertices(const Mesh &mesh) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    auto edge_states = EdgeStateBuffer.GetMutable(entry.EdgeStates);
    const auto vertex_states = GetVertexStates(entry.Vertices);
    for (uint32_t ei = 0; ei < mesh.EdgeCount(); ++ei) {
        const auto heh = mesh.GetHalfedge(EH{ei}, 0);
        edge_states[2 * ei] = vertex_states[*mesh.GetFromVertex(heh)] & ElementStateSelected;
        edge_states[2 * ei + 1] = vertex_states[*mesh.GetToVertex(heh)] & ElementStateSelected;
    }
}

void MeshStore::UpdateFaceStatesFromVertices(const Mesh &mesh) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    auto face_states = GetFaceStates(entry.FaceData);
    if (face_states.empty()) return;
    const auto vertex_states = GetVertexStates(entry.Vertices);
    for (uint32_t fi = 0; fi < mesh.FaceCount(); ++fi) {
        uint8_t state = ElementStateSelected;
        for (const auto vh : mesh.fv_range(FH{fi})) {
            if (!(vertex_states[*vh] & ElementStateSelected)) {
                state = 0;
                break;
            }
        }
        face_states[fi] = state;
    }
}

void MeshStore::UpdateFaceStatesFromEdges(const Mesh &mesh) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    auto face_states = GetFaceStates(entry.FaceData);
    if (face_states.empty()) return;
    const auto edge_states = EdgeStateBuffer.GetMutable(entry.EdgeStates);
    for (uint32_t fi = 0; fi < mesh.FaceCount(); ++fi) {
        uint8_t state = ElementStateSelected;
        for (const auto heh : mesh.fh_range(FH{fi})) {
            if (!(edge_states[2 * *mesh.GetEdge(heh)] & ElementStateSelected)) {
                state = 0;
                break;
            }
        }
        face_states[fi] = state;
    }
}

void MeshStore::UpdateVertexStatesFromElements(const Mesh &mesh, std::span<const uint32_t> handles, Element element, std::optional<uint32_t> active_handle) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    auto vertex_states = GetVertexStates(entry.Vertices);
    std::ranges::fill(vertex_states, uint8_t{0});

    if (element == Element::Face) {
        for (const auto fi : handles) {
            for (const auto vh : mesh.fv_range(FH{fi})) vertex_states[*vh] |= ElementStateSelected;
        }
        if (active_handle && *active_handle < mesh.FaceCount()) {
            for (const auto vh : mesh.fv_range(FH{*active_handle})) vertex_states[*vh] |= ElementStateActive;
        }
    } else if (element == Element::Edge) {
        for (const auto ei : handles) {
            const auto hh = mesh.GetHalfedge(EH{ei}, 0);
            vertex_states[*mesh.GetFromVertex(hh)] |= ElementStateSelected;
            vertex_states[*mesh.GetToVertex(hh)] |= ElementStateSelected;
        }
        if (active_handle && *active_handle < mesh.EdgeCount()) {
            const auto hh = mesh.GetHalfedge(EH{*active_handle}, 0);
            vertex_states[*mesh.GetFromVertex(hh)] |= ElementStateActive;
            vertex_states[*mesh.GetToVertex(hh)] |= ElementStateActive;
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
