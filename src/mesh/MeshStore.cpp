#include "MeshStore.h"

#include "MeshAttributes.h"
#include "vulkan/BufferArena.h"

#include <glm/geometric.hpp>
#include <zpp_bits.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include <numeric>

namespace {
constexpr uint8_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1}, ElementStateExcited{1u << 2};

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
    const auto specular_strength = std::max({ks.x, ks.y, ks.z});
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
    // Append the strided source element at `index` (when valid), else `fallback`.
    const auto append_attr = [](auto &dst, const auto &src, int index, auto fallback) {
        constexpr auto N = decltype(fallback)::length();
        if (index >= 0 && size_t(index) * N + (N - 1) < src.size()) {
            for (glm::length_t c = 0; c < N; ++c) fallback[c] = src[size_t(index) * N + c];
        }
        dst->emplace_back(fallback);
    };
    const auto FindOrAddVertex = [&](const tinyobj::index_t &index) -> uint32_t {
        const ObjVertexKey key{.PositionIndex = index.vertex_index, .NormalIndex = index.normal_index, .TexCoordIndex = index.texcoord_index};
        if (const auto it = vertex_cache.find(key); it != vertex_cache.end()) return it->second;

        if (index.vertex_index < 0 || size_t(index.vertex_index) * 3u + 2u >= attrib.vertices.size()) {
            throw std::runtime_error{std::format("OBJ '{}' references invalid vertex index {}.", path.string(), index.vertex_index)};
        }

        const uint32_t vertex_index = data.Positions.size();
        data.Positions.emplace_back(
            attrib.vertices[size_t(index.vertex_index) * 3u],
            attrib.vertices[size_t(index.vertex_index) * 3u + 1u],
            attrib.vertices[size_t(index.vertex_index) * 3u + 2u]
        );

        if (attrs.Normals) append_attr(attrs.Normals, attrib.normals, index.normal_index, vec3{0.f});
        if (attrs.TexCoords0) append_attr(attrs.TexCoords0, attrib.texcoords, index.texcoord_index, vec2{0.f});

        vertex_cache.emplace(key, vertex_index);
        return vertex_index;
    };

    std::unordered_map<int, uint32_t> material_to_local_index;
    std::unordered_map<int, uint32_t> material_to_primitive;
    std::vector<uint32_t> face_primitive_indices;
    std::vector<uint32_t> primitive_material_indices;
    const auto local_material_index = [&](int material_id) -> uint32_t {
        if (const auto it = material_to_local_index.find(material_id); it != material_to_local_index.end()) return it->second;
        const uint32_t local_index = result.Materials.size();
        if (material_id >= 0 && material_id < int(materials.size())) {
            result.Materials.emplace_back(ToObjPlyMaterial(materials[material_id], material_id, path.parent_path()));
        } else {
            result.Materials.emplace_back(DefaultMaterial());
        }
        material_to_local_index.emplace(material_id, local_index);
        return local_index;
    };
    const auto primitive_index = [&](int material_id) -> uint32_t {
        if (const auto it = material_to_primitive.find(material_id); it != material_to_primitive.end()) return it->second;
        const uint32_t index = primitive_material_indices.size();
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
        return std::clamp(float(value), 0.f, 1.f);
    } else if constexpr (std::is_signed_v<T>) {
        const auto den = float(std::numeric_limits<T>::max());
        return std::clamp(value / den, 0.f, 1.f);
    } else {
        const auto den = float(std::numeric_limits<T>::max());
        return std::clamp(value / den, 0.f, 1.f);
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
    return colors.count > 0 ? sum / float(colors.count) : vec3{1.f};
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

    const std::span face_buf{faces->buffer.get(), faces->buffer.size_bytes()};
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
            uint32_t vi = 0;
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

// Merge vertices identical in every vertex-domain channel, compared bitwise: position, skin joints/weights, and morph deltas.
// Corner-domain channels (normals/tangents/colors/UVs) split per corner, so they never block a merge.
// New indices assign in first-seen order, keeping the result deterministic.
// Rewrites positions/faces/edges/deform/morph in place.
void WeldVertices(MeshData &data, std::optional<ArmatureDeformData> &deform, std::optional<MorphTargetData> &morph) {
    const uint32_t count = data.Positions.size();
    if (count == 0) return;
    const uint32_t target_count = morph ? morph->TargetCount : 0u;
    const bool has_deform = deform.has_value();
    const bool has_norm_deltas = morph && !morph->NormalDeltas.empty();
    const bool has_tan_deltas = morph && !morph->TangentDeltas.empty();
    const uint32_t delta_channels = target_count * (1u + (has_norm_deltas ? 1u : 0u) + (has_tan_deltas ? 1u : 0u));
    const uint32_t stride = sizeof(vec3) + (has_deform ? sizeof(uvec4) + sizeof(vec4) : 0u) + delta_channels * sizeof(vec3);

    std::vector<std::byte> keys(size_t(count) * stride);
    for (uint32_t i = 0; i < count; ++i) {
        auto *dst = keys.data() + size_t(i) * stride;
        const auto append = [&dst](const auto &v) {
            std::memcpy(dst, &v, sizeof(v));
            dst += sizeof(v);
        };
        append(data.Positions[i]);
        if (has_deform) {
            append(deform->Joints[i]);
            append(deform->Weights[i]);
        }
        for (uint32_t t = 0; t < target_count; ++t) {
            append(morph->PositionDeltas[size_t(t) * count + i]);
            if (has_norm_deltas) append(morph->NormalDeltas[size_t(t) * count + i]);
            if (has_tan_deltas) append(morph->TangentDeltas[size_t(t) * count + i]);
        }
    }

    std::vector<uint32_t> remap(count);
    std::vector<uint32_t> reps; // First source index of each welded vertex
    reps.reserve(count);
    std::unordered_map<std::string_view, uint32_t> index_by_key;
    index_by_key.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const std::string_view key{reinterpret_cast<const char *>(keys.data() + size_t(i) * stride), stride};
        const auto [it, inserted] = index_by_key.try_emplace(key, uint32_t(reps.size()));
        if (inserted) reps.emplace_back(i);
        remap[i] = it->second;
    }
    const uint32_t welded = reps.size();
    if (welded == count) return;

    const auto compact = [&](auto &channel) {
        std::remove_reference_t<decltype(channel)> out(welded);
        for (uint32_t n = 0; n < welded; ++n) out[n] = channel[reps[n]];
        channel = std::move(out);
    };
    compact(data.Positions);
    if (has_deform) {
        compact(deform->Joints);
        compact(deform->Weights);
    }
    if (morph) {
        const auto compact_deltas = [&](std::vector<vec3> &channel) {
            if (channel.empty()) return;
            std::vector<vec3> out(size_t(target_count) * welded);
            for (uint32_t t = 0; t < target_count; ++t) {
                for (uint32_t n = 0; n < welded; ++n) out[size_t(t) * welded + n] = channel[size_t(t) * count + reps[n]];
            }
            channel = std::move(out);
        };
        compact_deltas(morph->PositionDeltas);
        compact_deltas(morph->NormalDeltas);
        compact_deltas(morph->TangentDeltas);
    }
    for (auto &face : data.Faces) {
        for (auto &v : face) v = remap[v];
    }
    for (auto &edge : data.Edges) edge = {remap[edge[0]], remap[edge[1]]};
}

} // namespace

struct MeshStore::Buffers {
    explicit Buffers(mvk::BufferContext &ctx)
        : FaceFirstTriangleBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer},
          FacePrimitiveBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::FacePrimitiveBuffer},
          PrimitiveMaterialBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::PrimitiveMaterialBuffer},
          BoneDeformBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::BoneDeformBuffer},
          MorphTargetBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::MorphTargetBuffer},
          VerticesBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer},
          OverlayVerticesBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer},
          VertexStateBuffer{ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          FaceStateBuffer{ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          FaceSharpnessBuffer{ctx, 0, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          EdgeStateBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          TriangleFaceIdBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer},
          CornerNormalBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::CornerNormalBuffer},
          EdgeSharpnessBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          CustomCornerNormalBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          CornerTangentBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::CornerTangentBuffer},
          CornerColorBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::CornerColorBuffer},
          CornerUvBuffer{ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::CornerUvBuffer} {}

    BufferArena<uint32_t> FaceFirstTriangleBuffer; // Per-face index of first triangle in the index buffer
    BufferArena<uint32_t> FacePrimitiveBuffer; // Per-face source primitive index
    BufferArena<uint32_t> PrimitiveMaterialBuffer; // Primitive index -> material index
    BufferArena<BoneDeformVertex> BoneDeformBuffer;
    BufferArena<MorphTargetVertex> MorphTargetBuffer;
    BufferArena<Vertex> VerticesBuffer;
    // Overlay geometry, kept out of ForEachArena so Serialize skips it. No state mirror (overlays aren't edited).
    BufferArena<Vertex> OverlayVerticesBuffer;
    mvk::Buffer VertexStateBuffer; // Mirrors VerticesBuffer
    mvk::Buffer FaceStateBuffer; // Mirrors FaceFirstTriangleBuffer
    mvk::Buffer FaceSharpnessBuffer; // Mirrors FaceFirstTriangleBuffer. 1 = flat-shaded face (canonical sharpness store)
    BufferArena<uint8_t> EdgeStateBuffer;
    BufferArena<uint32_t> TriangleFaceIdBuffer; // 1-indexed map from face triangles (in mesh face order) to source face ID
    BufferArena<vec3> CornerNormalBuffer; // Per-corner shading normals, in triangulated face-fan order
    BufferArena<uint8_t> EdgeSharpnessBuffer; // One byte per edge, 1 = sharp (canonical sharpness store)
    BufferArena<vec3> CustomCornerNormalBuffer; // Authored corner normals, vec3(0) = use derived
    BufferArena<vec4> CornerTangentBuffer; // Corner-domain attribute layers, fan order like CornerNormalBuffer
    BufferArena<vec4> CornerColorBuffer;
    BufferArena<vec2> CornerUvBuffer; // Up to four ranges per mesh, one per UV set

    // Visit every BufferArena in a fixed order, so Serialize/Deserialize stay in lockstep.
    void ForEachArena(auto &&f) {
        f(VerticesBuffer);
        f(FaceFirstTriangleBuffer);
        f(FacePrimitiveBuffer);
        f(PrimitiveMaterialBuffer);
        f(BoneDeformBuffer);
        f(MorphTargetBuffer);
        f(EdgeStateBuffer);
        f(TriangleFaceIdBuffer);
        f(CornerNormalBuffer);
        f(EdgeSharpnessBuffer);
        f(CustomCornerNormalBuffer);
        f(CornerTangentBuffer);
        f(CornerColorBuffer);
        f(CornerUvBuffer);
    }
    static constexpr size_t ArenaCount = 14;
};

namespace {
// Save/restore a plain mirror buffer (no allocator) by its used byte region.
std::vector<std::byte> SaveBuffer(const mvk::Buffer &b) {
    const auto mapped = b.GetMappedData();
    const auto used = std::min(size_t(b.UsedSize), mapped.size());
    return {mapped.begin(), mapped.begin() + used};
}
void RestoreBuffer(mvk::Buffer &b, std::span<const std::byte> bytes) {
    b.Reserve(bytes.size());
    if (!bytes.empty()) b.Update(bytes, 0);
    b.UsedSize = bytes.size();
}
} // namespace

std::vector<std::byte> MeshStore::Serialize() const {
    std::vector<ArenaState> arenas;
    arenas.reserve(Buffers::ArenaCount);
    B->ForEachArena([&](const auto &a) { arenas.push_back(a.Save()); });
    const auto vertex_state = SaveBuffer(B->VertexStateBuffer);
    const auto face_state = SaveBuffer(B->FaceStateBuffer);
    const auto face_sharpness = SaveBuffer(B->FaceSharpnessBuffer);

    // Serialize from non-const copies: zpp mis-encodes a const aggregate this large, and these match the types Deserialize reads back into.
    auto entries = Entries;
    auto free_ids = FreeIds;
    std::vector<std::byte> out;
    zpp::bits::out archive{out};
    if (zpp::bits::failure(archive(arenas, vertex_state, face_state, face_sharpness, entries, free_ids))) return {};
    out.resize(archive.position());
    return out;
}

void MeshStore::Deserialize(std::span<const std::byte> bytes) {
    std::vector<ArenaState> arenas;
    std::vector<std::byte> vertex_state, face_state, face_sharpness;
    std::vector<Entry> entries;
    std::vector<uint32_t> free_ids;
    zpp::bits::in archive{bytes};
    if (zpp::bits::failure(archive(arenas, vertex_state, face_state, face_sharpness, entries, free_ids))) return;
    if (arenas.size() != Buffers::ArenaCount) return;

    size_t i = 0;
    B->ForEachArena([&](auto &a) { a.Restore(std::move(arenas[i++])); });
    RestoreBuffer(B->VertexStateBuffer, vertex_state);
    RestoreBuffer(B->FaceStateBuffer, face_state);
    RestoreBuffer(B->FaceSharpnessBuffer, face_sharpness);
    Entries = std::move(entries);
    FreeIds = std::move(free_ids);
}

MeshStore::MeshStore(mvk::BufferContext &ctx) : B{std::make_unique<Buffers>(ctx)} {}
MeshStore::~MeshStore() = default;
MeshStore::MeshStore(MeshStore &&) noexcept = default;
MeshStore &MeshStore::operator=(MeshStore &&) noexcept = default;

std::span<const Vertex> MeshStore::GetVertices(uint32_t id) const { return B->VerticesBuffer.Get(Entries.at(id).Vertices); }
std::span<Vertex> MeshStore::GetVertices(uint32_t id) { return B->VerticesBuffer.GetMutable(Entries.at(id).Vertices); }
SlottedRange MeshStore::GetVerticesRange(uint32_t id) const { return B->VerticesBuffer.Slotted(Entries.at(id).Vertices); }
SlottedRange MeshStore::GetBoneDeformRange(uint32_t id) const { return B->BoneDeformBuffer.Slotted(Entries.at(id).BoneDeform); }
SlottedRange MeshStore::GetMorphTargetRange(uint32_t id) const { return B->MorphTargetBuffer.Slotted(Entries.at(id).MorphTargets); }

std::span<const BoneDeformVertex> MeshStore::GetBoneDeform(uint32_t id) const { return B->BoneDeformBuffer.Get(Entries.at(id).BoneDeform); }
std::span<const MorphTargetVertex> MeshStore::GetMorphTargets(uint32_t id) const { return B->MorphTargetBuffer.Get(Entries.at(id).MorphTargets); }

uint32_t MeshStore::GetVertexStateSlot() const { return B->VertexStateBuffer.Slot; }
uint32_t MeshStore::GetCornerNormalSlot() const { return B->CornerNormalBuffer.Buffer.Slot; }
uint32_t MeshStore::GetCornerTangentSlot() const { return B->CornerTangentBuffer.Buffer.Slot; }
uint32_t MeshStore::GetCornerColorSlot() const { return B->CornerColorBuffer.Buffer.Slot; }
uint32_t MeshStore::GetCornerUvSlot() const { return B->CornerUvBuffer.Buffer.Slot; }
uint32_t MeshStore::GetEdgeSharpnessSlot() const { return B->EdgeSharpnessBuffer.Buffer.Slot; }
uint32_t MeshStore::GetFacePrimitiveSlot() const { return B->FacePrimitiveBuffer.Buffer.Slot; }
uint32_t MeshStore::GetPrimitiveMaterialSlot() const { return B->PrimitiveMaterialBuffer.Buffer.Slot; }
uint32_t MeshStore::GetBoneDeformSlot() const { return B->BoneDeformBuffer.Buffer.Slot; }
uint32_t MeshStore::GetMorphTargetSlot() const { return B->MorphTargetBuffer.Buffer.Slot; }

SlottedRange MeshStore::GetFaceStateRange(uint32_t id) const { return {Entries.at(id).FaceData, B->FaceStateBuffer.Slot}; }
SlottedRange MeshStore::GetEdgeStateRange(uint32_t id) const { return B->EdgeStateBuffer.Slotted(Entries.at(id).EdgeStates); }
Range MeshStore::GetEdgeSharpnessRange(uint32_t id) const { return Entries.at(id).EdgeSharpness; }
Range MeshStore::GetCornerNormalRange(uint32_t id) const { return Entries.at(id).CornerNormals; }
Range MeshStore::GetCornerTangentRange(uint32_t id) const { return Entries.at(id).CornerTangents; }
Range MeshStore::GetCornerColorRange(uint32_t id) const { return Entries.at(id).CornerColors; }
Range MeshStore::GetCornerUvRange(uint32_t id, uint32_t set) const { return Entries.at(id).CornerUvs.at(set); }
std::span<const vec4> MeshStore::GetCornerTangents(uint32_t id) const { return B->CornerTangentBuffer.Get(Entries.at(id).CornerTangents); }
std::span<const vec4> MeshStore::GetCornerColors(uint32_t id) const { return B->CornerColorBuffer.Get(Entries.at(id).CornerColors); }
std::span<const vec2> MeshStore::GetCornerUvs(uint32_t id, uint32_t set) const { return B->CornerUvBuffer.Get(Entries.at(id).CornerUvs.at(set)); }
SlottedRange MeshStore::GetFaceIdRange(uint32_t id) const { return B->TriangleFaceIdBuffer.Slotted(Entries.at(id).TriangleFaceIds); }
SlottedRange MeshStore::GetFacePrimitiveRange(uint32_t id) const { return B->FacePrimitiveBuffer.Slotted(Entries.at(id).FacePrimitives); }
SlottedRange MeshStore::GetPrimitiveMaterialRange(uint32_t id) const { return B->PrimitiveMaterialBuffer.Slotted(Entries.at(id).PrimitiveMaterials); }

std::span<const uint32_t> MeshStore::GetTriangleFaceIds(uint32_t id) const { return B->TriangleFaceIdBuffer.Get(Entries.at(id).TriangleFaceIds); }
std::span<const uint32_t> MeshStore::GetFaceFirstTriangles(uint32_t id) const { return B->FaceFirstTriangleBuffer.Get(Entries.at(id).FaceData); }
std::span<const uint32_t> MeshStore::GetFacePrimitiveIndices(uint32_t id) const { return B->FacePrimitiveBuffer.Get(Entries.at(id).FacePrimitives); }
std::span<uint32_t> MeshStore::GetFacePrimitiveIndices(uint32_t id) { return B->FacePrimitiveBuffer.GetMutable(Entries.at(id).FacePrimitives); }
std::span<const uint32_t> MeshStore::GetPrimitiveMaterialIndices(uint32_t id) const { return B->PrimitiveMaterialBuffer.Get(Entries.at(id).PrimitiveMaterials); }
std::span<uint32_t> MeshStore::GetPrimitiveMaterialIndices(uint32_t id) { return B->PrimitiveMaterialBuffer.GetMutable(Entries.at(id).PrimitiveMaterials); }

namespace {
// Area-weighted smooth vertex normals. `position` maps a vertex index to its position and
// `normal_at` to its output normal.
void AccumulateSmoothVertexNormals(const Mesh &mesh, auto &&position, auto &&normal_at) {
    for (const auto vh : mesh.vertices()) normal_at(*vh) = vec3{0};
    for (const auto fh : mesh.faces()) {
        auto it = mesh.cfv_iter(fh);
        const auto p0 = position(**it);
        const auto p1 = position(**++it);
        const auto p2 = position(**++it);
        const auto cross = glm::cross(p1 - p0, p2 - p0);
        for (const auto vh : mesh.fv_range(fh)) normal_at(*vh) += cross;
    }
    for (const auto vh : mesh.vertices()) normal_at(*vh) = glm::normalize(normal_at(*vh));
}
} // namespace

void MeshStore::UpdateVertexNormals(const Mesh &mesh) {
    auto vertices = GetVertices(mesh.GetStoreId());
    AccumulateSmoothVertexNormals(
        mesh,
        [&](uint32_t vi) { return vertices[vi].Position; },
        [&](uint32_t vi) -> vec3 & { return vertices[vi].Normal; }
    );
}

namespace {
// Sector-cut corner derivation. `position` and `smooth_normal` map a vertex index to its
// position and full-fan smooth normal. Corners of sharp faces take the face normal, smooth
// corners at vertices touching a sharp edge or sharp face average their fan sector (cut at
// the discontinuities), and all other smooth corners take `smooth_normal`.
void DeriveCornerNormals(
    const Mesh &mesh,
    std::span<const uint8_t> sharp_faces, std::span<const uint8_t> sharp_edges,
    auto &&position, auto &&smooth_normal, std::span<vec3> corners
) {
    const auto &c = mesh.GetConnectivity();
    const auto face_sharp = [&](Mesh::FH fh) { return *fh < sharp_faces.size() && sharp_faces[*fh] != 0; };
    const auto edge_sharp = [&](Mesh::HH hh) { const auto eh = mesh.GetEdge(hh); return *eh < sharp_edges.size() && sharp_edges[*eh] != 0; };

    // Vertices touching a discontinuity need sector fans; all others take the vertex normal.
    // Face crosses are only consumed at discontinuities, and each is shared by every fan it appears in.
    static thread_local std::vector<uint8_t> touched;
    static thread_local std::vector<vec3> face_crosses;
    touched.clear();
    const auto is_set = [](uint8_t s) { return s != 0; };
    if (std::ranges::any_of(sharp_edges, is_set) || std::ranges::any_of(sharp_faces, is_set)) {
        touched.assign(mesh.VertexCount(), 0);
        for (uint32_t ei = 0; ei < sharp_edges.size(); ++ei) {
            if (!sharp_edges[ei]) continue;
            const auto hh = mesh.GetHalfedge(Mesh::EH{ei}, 0);
            if (const auto to = mesh.GetToVertex(hh)) touched[*to] = 1;
            if (const auto from = mesh.GetFromVertex(hh)) touched[*from] = 1;
        }
        for (uint32_t fi = 0; fi < sharp_faces.size(); ++fi) {
            if (!sharp_faces[fi]) continue;
            for (const auto vh : mesh.fv_range(Mesh::FH{fi})) touched[*vh] = 1;
        }
        face_crosses.resize(mesh.FaceCount());
        for (const auto fh : mesh.faces()) {
            auto it = mesh.cfv_iter(fh);
            const auto p0 = position(**it);
            const auto p1 = position(**++it);
            const auto p2 = position(**++it);
            face_crosses[*fh] = glm::cross(p1 - p0, p2 - p0);
        }
    }
    const auto face_cross = [&](Mesh::FH fh) { return face_crosses[*fh]; };

    // Average the smooth-face sector around `h_in`'s corner vertex, walking both ways from
    // `fh` until a sharp edge, sharp face, or boundary cuts the fan.
    const auto sector_normal = [&](Mesh::FH fh, Mesh::HH h_in) {
        constexpr uint32_t MaxFan{64};
        auto acc = face_cross(fh);
        bool full_loop = false;
        auto h = h_in;
        for (uint32_t i = 0; i < MaxFan; ++i) {
            const auto out = c.Halfedges[*h].Next; // Leaves the corner vertex within the current face.
            if (edge_sharp(out)) break;
            const auto opp = c.Halfedges[*out].Opposite;
            if (!opp) break;
            const auto nf = c.Halfedges[*opp].Face;
            if (!nf) break;
            if (nf == fh) {
                full_loop = true;
                break;
            }
            if (face_sharp(nf)) break;
            acc += face_cross(nf);
            h = opp;
        }
        if (!full_loop) {
            h = h_in;
            for (uint32_t i = 0; i < MaxFan; ++i) {
                if (edge_sharp(h)) break;
                const auto opp = c.Halfedges[*h].Opposite;
                if (!opp) break;
                const auto nf = c.Halfedges[*opp].Face;
                if (!nf || nf == fh || face_sharp(nf)) break;
                acc += face_cross(nf);
                // Continue from the halfedge entering the corner vertex within `nf`.
                auto prev = opp;
                for (auto walk = c.Halfedges[*opp].Next; walk != opp; walk = c.Halfedges[*walk].Next) prev = walk;
                h = prev;
            }
        }
        return glm::normalize(acc);
    };

    uint32_t ci = 0;
    std::vector<vec3> face_corners; // Per-face corner normals in vertex order, emitted in fan order.
    for (const auto fh : mesh.faces()) {
        if (face_sharp(fh)) {
            const auto n = glm::normalize(face_cross(fh));
            const auto tri_count = mesh.GetValence(fh) - 2;
            for (uint32_t t = 0; t < tri_count * 3; ++t) corners[ci++] = n;
            continue;
        }
        face_corners.clear();
        for (const auto hh : mesh.fh_range(fh)) {
            const auto vh = mesh.GetToVertex(hh);
            face_corners.emplace_back(!touched.empty() && touched[*vh] ? sector_normal(fh, hh) : smooth_normal(*vh));
        }
        for (uint32_t i = 1; i + 1 < face_corners.size(); ++i) {
            corners[ci++] = face_corners[0];
            corners[ci++] = face_corners[i];
            corners[ci++] = face_corners[i + 1];
        }
    }
}
} // namespace

void MeshStore::UpdateCornerNormals(const Mesh &mesh) {
    const auto id = mesh.GetStoreId();
    const auto corners = B->CornerNormalBuffer.GetMutable(Entries.at(id).CornerNormals);
    if (corners.empty()) return;
    const auto vertices = GetVertices(id);
    DeriveCornerNormals(
        mesh, GetFaceSharpness(id), GetEdgeSharpness(id),
        [&](uint32_t vi) { return vertices[vi].Position; },
        [&](uint32_t vi) { return vertices[vi].Normal; },
        corners
    );
    ComposeCustomCornerNormals(id, corners);
}

// Authored corner normals override the derived values wherever they are non-zero.
void MeshStore::ComposeCustomCornerNormals(uint32_t id, std::span<vec3> corners) const {
    const auto custom = B->CustomCornerNormalBuffer.Get(Entries.at(id).CustomCornerNormals);
    for (size_t i = 0; i < corners.size() && i < custom.size(); ++i) {
        if (custom[i] != vec3{0}) corners[i] = custom[i];
    }
}

void MeshStore::UpdateCornerNormals(const Mesh &mesh, std::span<const vec3> positions) {
    const auto id = mesh.GetStoreId();
    const auto corners = B->CornerNormalBuffer.GetMutable(Entries.at(id).CornerNormals);
    if (corners.empty()) return;
    // Smooth vertex normals recomputed from the drag-transformed positions, matching a commit.
    static thread_local std::vector<vec3> vertex_normals;
    vertex_normals.resize(positions.size());
    AccumulateSmoothVertexNormals(
        mesh,
        [&](uint32_t vi) { return positions[vi]; },
        [&](uint32_t vi) -> vec3 & { return vertex_normals[vi]; }
    );
    DeriveCornerNormals(
        mesh, GetFaceSharpness(id), GetEdgeSharpness(id),
        [&](uint32_t vi) { return positions[vi]; },
        [&](uint32_t vi) { return vertex_normals[vi]; },
        corners
    );
    ComposeCustomCornerNormals(id, corners);
}

void MeshStore::SetEdgeSharpnessByAngle(const Mesh &mesh, float angle) {
    const auto id = mesh.GetStoreId();
    auto sharp = GetEdgeSharpness(id);
    if (sharp.empty()) return;
    const auto vertices = GetVertices(id);
    static thread_local std::vector<vec3> face_normals;
    face_normals.resize(mesh.FaceCount());
    for (const auto fh : mesh.faces()) {
        auto it = mesh.cfv_iter(fh);
        const auto p0 = vertices[**it].Position;
        const auto p1 = vertices[**++it].Position;
        const auto p2 = vertices[**++it].Position;
        face_normals[*fh] = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    }
    const auto &c = mesh.GetConnectivity();
    const float cos_angle = std::cos(angle);
    for (uint32_t ei = 0; ei < mesh.EdgeCount(); ++ei) {
        const auto hh = mesh.GetHalfedge(Mesh::EH{ei}, 0);
        const auto face = mesh.GetFace(hh);
        const auto opposite = c.Halfedges[*hh].Opposite;
        const auto opposite_face = opposite ? c.Halfedges[*opposite].Face : Mesh::FH{};
        sharp[ei] = face && opposite_face && glm::dot(face_normals[*face], face_normals[*opposite_face]) < cos_angle ? 1 : 0;
    }
}

SharpnessSummary MeshStore::GetFaceSharpnessSummary(uint32_t id) const {
    const auto s = GetFaceSharpness(id);
    const auto sharp = [](uint8_t b) { return b != 0; };
    return {std::ranges::any_of(s, sharp), !s.empty() && std::ranges::all_of(s, sharp)};
}

namespace {
void WriteVertices(std::span<Vertex> dst, std::span<const vec3> positions, const MeshVertexAttributes &attrs) {
    for (uint32_t i = 0; i < positions.size(); ++i) {
        dst[i] = {
            .Position = positions[i],
            .Color = attrs.Colors0 ? (*attrs.Colors0)[i] : vec4{1.f},
        };
    }
    if (attrs.Normals) {
        for (uint32_t i = 0; i < positions.size(); ++i) dst[i].Normal = (*attrs.Normals)[i];
    }
}
} // namespace

std::pair<uint32_t, Range> MeshStore::AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &attrs) {
    const auto vertices = AllocateVertices(positions.size());
    WriteVertices(B->VerticesBuffer.GetMutable(vertices), positions, attrs);
    return {AcquireId({.Vertices = vertices, .FaceData = {}, .Alive = true}), vertices};
}

std::pair<uint32_t, Range> MeshStore::AllocateOverlayVertexBuffer(std::span<const vec3> positions) {
    const auto vertices = B->OverlayVerticesBuffer.Allocate(positions.size());
    WriteVertices(B->OverlayVerticesBuffer.GetMutable(vertices), positions, {});
    if (!OverlayFreeIds.empty()) {
        const auto id = OverlayFreeIds.back();
        OverlayFreeIds.pop_back();
        OverlayEntries[id] = vertices;
        return {id, vertices};
    }
    OverlayEntries.push_back(vertices);
    return {uint32_t(OverlayEntries.size() - 1), vertices};
}

SlottedRange MeshStore::GetOverlayVerticesRange(uint32_t id) const { return B->OverlayVerticesBuffer.Slotted(OverlayEntries.at(id)); }

void MeshStore::ReleaseOverlay(uint32_t id) {
    if (id >= OverlayEntries.size()) return;
    B->OverlayVerticesBuffer.Release(OverlayEntries[id]);
    OverlayEntries[id] = {};
    OverlayFreeIds.push_back(id);
}

void MeshStore::PlanCreate(const MeshData &data, const MeshPrimitives &primitives, bool has_deform, uint32_t morph_target_count, const MeshVertexAttributes &attrs) {
    const uint32_t vertices = data.Positions.size();
    const uint32_t faces = data.Faces.size();
    uint32_t triangles = 0;
    for (const auto &face : data.Faces) triangles += face.size() - 2;
    Pending.Vertices += vertices;
    Pending.Faces += faces;
    Pending.Triangles += triangles;
    Pending.Edges += (triangles + 2 * faces + 1) / 2; // manifold estimate: edges ≈ halfedges / 2
    Pending.EdgeStates += triangles + 2 * faces; // manifold estimate: halfedges ≈ triangles + 2*faces
    Pending.Primitives += primitives.MaterialIndices.size();
    if (has_deform) Pending.BoneDeformVertices += vertices;
    if (morph_target_count > 0) Pending.MorphTargetEntries += morph_target_count * vertices;
    if (triangles > 0) {
        const uint32_t corners = triangles * 3;
        if (attrs.Tangents) Pending.CornerTangents += corners;
        if (attrs.Colors0) Pending.CornerColors += corners;
        for (const auto *uvs : {&attrs.TexCoords0, &attrs.TexCoords1, &attrs.TexCoords2, &attrs.TexCoords3}) {
            if (*uvs) Pending.CornerUvs += corners;
        }
    }
}

void MeshStore::PlanClone(const Mesh &mesh) {
    const auto &e = Entries.at(mesh.GetStoreId());
    Pending.Vertices += e.Vertices.Count;
    Pending.Faces += e.FaceData.Count;
    Pending.Triangles += e.TriangleFaceIds.Count;
    Pending.Edges += e.EdgeSharpness.Count;
    Pending.EdgeStates += e.EdgeStates.Count;
    Pending.Primitives += e.PrimitiveMaterials.Count;
    Pending.BoneDeformVertices += e.BoneDeform.Count;
    Pending.MorphTargetEntries += e.MorphTargets.Count;
    Pending.CornerTangents += e.CornerTangents.Count;
    Pending.CornerColors += e.CornerColors.Count;
    for (const auto &uvs : e.CornerUvs) Pending.CornerUvs += uvs.Count;
}

void MeshStore::CommitReserves() {
    B->VerticesBuffer.ReserveAdditional(Pending.Vertices);
    B->FaceFirstTriangleBuffer.ReserveAdditional(Pending.Faces);
    B->FacePrimitiveBuffer.ReserveAdditional(Pending.Faces);
    B->TriangleFaceIdBuffer.ReserveAdditional(Pending.Triangles);
    B->CornerNormalBuffer.ReserveAdditional(Pending.Triangles * 3);
    B->EdgeStateBuffer.ReserveAdditional(Pending.EdgeStates);
    B->EdgeSharpnessBuffer.ReserveAdditional(Pending.Edges);
    B->PrimitiveMaterialBuffer.ReserveAdditional(Pending.Primitives);
    B->BoneDeformBuffer.ReserveAdditional(Pending.BoneDeformVertices);
    B->MorphTargetBuffer.ReserveAdditional(Pending.MorphTargetEntries);
    B->CornerTangentBuffer.ReserveAdditional(Pending.CornerTangents);
    B->CornerColorBuffer.ReserveAdditional(Pending.CornerColors);
    B->CornerUvBuffer.ReserveAdditional(Pending.CornerUvs);
    // Mirror buffers (uint8_t state per element, no arena — shared ranges with data arenas).
    if (Pending.Vertices > 0) B->VertexStateBuffer.Reserve(B->VertexStateBuffer.UsedSize + Pending.Vertices);
    if (Pending.Faces > 0) {
        B->FaceStateBuffer.Reserve(B->FaceStateBuffer.UsedSize + Pending.Faces);
        B->FaceSharpnessBuffer.Reserve(B->FaceSharpnessBuffer.UsedSize + Pending.Faces);
    }
    Pending = {};
}

CreatedMesh MeshStore::CreateMesh(MeshData &&data, MeshVertexAttributes &&attrs, MeshPrimitives &&primitives, bool flat_shaded, std::optional<ArmatureDeformData> deform, std::optional<MorphTargetData> morph, bool weld) {
    const uint32_t face_count = data.Faces.size();

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

    // Triangle-mesh tangent/color/UV channels are corner-domain: gathered into per-corner streams in fan order before welding rewrites the face indices.
    // The vertex buffer keeps defaults for these channels.
    std::vector<vec4> corner_tangents, corner_colors;
    std::array<std::vector<vec2>, 4> corner_uvs;
    std::vector<vec3> authored_corner_normals;
    if (face_count > 0) {
        uint32_t corner_total = 0;
        for (const auto &face : data.Faces) corner_total += (face.size() - 2) * 3;
        const auto gather_corners = [&]<typename T>(std::optional<std::vector<T>> &src, std::vector<T> &out) {
            if (!src) return;
            out.reserve(corner_total);
            for (const auto &face : data.Faces) {
                for (uint32_t k = 1; k + 1 < face.size(); ++k) {
                    out.emplace_back((*src)[face[0]]);
                    out.emplace_back((*src)[face[k]]);
                    out.emplace_back((*src)[face[k + 1]]);
                }
            }
            src.reset();
        };
        gather_corners(attrs.Tangents, corner_tangents);
        gather_corners(attrs.Colors0, corner_colors);
        gather_corners(attrs.TexCoords0, corner_uvs[0]);
        gather_corners(attrs.TexCoords1, corner_uvs[1]);
        gather_corners(attrs.TexCoords2, corner_uvs[2]);
        gather_corners(attrs.TexCoords3, corner_uvs[3]);
        if (weld) {
            // Authored normals recover from this stream after welding: faceted faces fill the face-sharpness store and the remainder lands in the custom corner-normal layer.
            // The vertex normal becomes purely derived.
            gather_corners(attrs.Normals, authored_corner_normals);
            WeldVertices(data, deform, morph);
        }
    }

    const uint32_t vertex_count = data.Positions.size();
    auto [id, vertices] = AllocateVertexBuffer(data.Positions, attrs);
    auto &entry = Entries[id];

    if (deform) {
        entry.BoneDeform = B->BoneDeformBuffer.Allocate(vertex_count);
        auto bd_span = B->BoneDeformBuffer.GetMutable(entry.BoneDeform);
        for (uint32_t i = 0; i < vertex_count; ++i) {
            bd_span[i] = {.Joints = deform->Joints[i], .Weights = deform->Weights[i]};
        }
    }
    if (morph && morph->TargetCount > 0 && vertex_count > 0) {
        entry.MorphTargetCount = morph->TargetCount;
        const auto total = entry.MorphTargetCount * vertex_count;
        entry.MorphTargets = B->MorphTargetBuffer.Allocate(total);
        auto mt_span = B->MorphTargetBuffer.GetMutable(entry.MorphTargets);
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
        // Write face-first-triangle offsets directly into GPU buffer.
        entry.FaceData = AllocateFaces(face_count);
        auto first_tri_span = B->FaceFirstTriangleBuffer.GetMutable(entry.FaceData);
        uint32_t tri_offset = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            first_tri_span[fi] = tri_offset;
            tri_offset += data.Faces[fi].size() - 2;
        }

        entry.TriangleCount = tri_offset;
        entry.CornerNormals = B->CornerNormalBuffer.Allocate(tri_offset * 3);

        if (!corner_tangents.empty()) entry.CornerTangents = B->CornerTangentBuffer.Allocate(std::span<const vec4>{corner_tangents});
        if (!corner_colors.empty()) entry.CornerColors = B->CornerColorBuffer.Allocate(std::span<const vec4>{corner_colors});
        for (uint32_t set = 0; set < corner_uvs.size(); ++set) {
            if (!corner_uvs[set].empty()) entry.CornerUvs[set] = B->CornerUvBuffer.Allocate(std::span<const vec2>{corner_uvs[set]});
        }

        // Write triangle-to-face IDs directly into GPU buffer.
        entry.TriangleFaceIds = B->TriangleFaceIdBuffer.Allocate(tri_offset);
        auto tri_face_span = B->TriangleFaceIdBuffer.GetMutable(entry.TriangleFaceIds);
        uint32_t ti = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            const auto n_tris = data.Faces[fi].size() - 2;
            for (size_t t = 0; t < n_tris; ++t) tri_face_span[ti++] = fi + 1;
        }

        // Write face-to-primitive mapping directly into GPU buffer.
        entry.FacePrimitives = B->FacePrimitiveBuffer.Allocate(face_count);
        auto fp_span = B->FacePrimitiveBuffer.GetMutable(entry.FacePrimitives);
        if (!primitives.FacePrimitiveIndices.empty()) {
            std::ranges::copy(primitives.FacePrimitiveIndices, fp_span.begin());
        } else {
            std::ranges::fill(fp_span, 0u);
        }

        const auto primitive_count = !primitives.MaterialIndices.empty() ?
            (primitives.FacePrimitiveIndices.empty() ? 1u : *std::ranges::max_element(primitives.FacePrimitiveIndices) + 1u) :
            1u;
        entry.PrimitiveMaterials = B->PrimitiveMaterialBuffer.Allocate(primitive_count);
        auto pm_span = B->PrimitiveMaterialBuffer.GetMutable(entry.PrimitiveMaterials);
        if (!primitives.MaterialIndices.empty()) {
            std::ranges::copy(primitives.MaterialIndices, pm_span.begin());
        } else {
            std::ranges::fill(pm_span, 0u);
        }

        // Compute per-primitive triangle ranges from the (now sorted) face data.
        {
            const auto fp = B->FacePrimitiveBuffer.Get(entry.FacePrimitives);
            const auto fft = B->FaceFirstTriangleBuffer.Get(entry.FaceData);
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

    MeshConnectivity conn = [&] {
        if (!data.Faces.empty()) return BuildConnectivity(data.Faces, vertex_count);
        if (!data.Edges.empty()) return BuildConnectivity(data.Edges, vertex_count);
        MeshConnectivity c;
        c.VertexCount = vertex_count;
        return c;
    }();
    const Mesh mesh{*this, id, conn};

    entry.EdgeStates = B->EdgeStateBuffer.Allocate(mesh.EdgeCount() * 2);
    entry.EdgeSharpness = B->EdgeSharpnessBuffer.Allocate(mesh.EdgeCount());
    ClearElementStates(vertices, entry.FaceData, entry.EdgeStates);
    if (entry.FaceData.Count > 0) std::ranges::fill(GetFaceSharpness(id), uint8_t(flat_shaded ? 1 : 0));
    if (entry.EdgeSharpness.Count > 0) std::ranges::fill(GetEdgeSharpness(id), uint8_t{0});

    // An authored corner normal within this dot of the derived one counts as derivable and is dropped.
    constexpr float AuthoredMatchDot{0.9999999f};
    // A face whose authored corner normals all match its geometric normal shades flat, recorded as face sharpness.
    if (!authored_corner_normals.empty() && entry.FaceData.Count > 0) {
        auto sharp = GetFaceSharpness(id);
        uint32_t ci = 0;
        for (uint32_t fi = 0; fi < data.Faces.size(); ++fi) {
            const auto &face = data.Faces[fi];
            const uint32_t corner_count = (face.size() - 2) * 3;
            const auto cross = glm::cross(data.Positions[face[1]] - data.Positions[face[0]], data.Positions[face[2]] - data.Positions[face[0]]);
            const auto cross_len = glm::length(cross);
            bool flat = cross_len > 0.f;
            if (flat) {
                const auto face_normal = cross / cross_len;
                for (uint32_t k = 0; k < corner_count; ++k) {
                    const auto authored = authored_corner_normals[ci + k];
                    const auto authored_len = glm::length(authored);
                    if (authored_len < 1e-6f || glm::dot(authored / authored_len, face_normal) < AuthoredMatchDot) {
                        flat = false;
                        break;
                    }
                }
            }
            if (flat) sharp[fi] = 1;
            ci += corner_count;
        }
    }

    if (!data.Faces.empty() && !attrs.Normals) UpdateVertexNormals(mesh);
    UpdateCornerNormals(mesh);

    // Authored corner normals the sharpness stores can't reproduce land in the custom layer verbatim.
    if (!authored_corner_normals.empty() && entry.CornerNormals.Count > 0) {
        const auto derived = B->CornerNormalBuffer.Get(entry.CornerNormals);
        std::vector<vec3> custom(derived.size(), vec3{0});
        bool any_custom = false;
        for (size_t i = 0; i < derived.size() && i < authored_corner_normals.size(); ++i) {
            const auto authored = authored_corner_normals[i];
            const auto authored_len = glm::length(authored);
            if (authored_len < 1e-6f) continue;
            // The authored normal stays unless it provably matches the derived one, so degenerate (zero or NaN) derived normals keep it.
            if (!(glm::dot(authored / authored_len, derived[i]) >= AuthoredMatchDot)) {
                custom[i] = authored;
                any_custom = true;
            }
        }
        if (any_custom) {
            entry.CustomCornerNormals = B->CustomCornerNormalBuffer.Allocate(std::span<const vec3>{custom});
            ComposeCustomCornerNormals(id, B->CornerNormalBuffer.GetMutable(entry.CornerNormals));
        }
    }

    return {id, std::move(conn), morph ? std::move(morph->TangentDeltas) : std::vector<vec3>{}};
}

CreatedMesh MeshStore::CloneMesh(const Mesh &mesh) {
    const auto src_id = mesh.GetStoreId();
    const auto src_vertices = GetVertices(src_id);
    const auto vertices = AllocateVertices(src_vertices.size());
    std::ranges::copy(src_vertices, B->VerticesBuffer.GetMutable(vertices).begin());

    const auto faces = AllocateFaces(mesh.FaceCount());
    std::ranges::copy(B->FaceFirstTriangleBuffer.Get(Entries.at(src_id).FaceData), B->FaceFirstTriangleBuffer.GetMutable(faces).begin());

    const auto edge_states = B->EdgeStateBuffer.Allocate(mesh.EdgeCount() * 2);
    const auto &src_entry = Entries.at(src_id);
    const auto id = AcquireId({
        .Vertices = vertices,
        .FaceData = faces,
        .CornerNormals = B->CornerNormalBuffer.Clone(src_entry.CornerNormals),
        .CustomCornerNormals = B->CustomCornerNormalBuffer.Clone(src_entry.CustomCornerNormals),
        .CornerTangents = B->CornerTangentBuffer.Clone(src_entry.CornerTangents),
        .CornerColors = B->CornerColorBuffer.Clone(src_entry.CornerColors),
        .CornerUvs = {B->CornerUvBuffer.Clone(src_entry.CornerUvs[0]), B->CornerUvBuffer.Clone(src_entry.CornerUvs[1]), B->CornerUvBuffer.Clone(src_entry.CornerUvs[2]), B->CornerUvBuffer.Clone(src_entry.CornerUvs[3])},
        .EdgeSharpness = B->EdgeSharpnessBuffer.Clone(src_entry.EdgeSharpness),
        .EdgeStates = edge_states,
        .TriangleFaceIds = B->TriangleFaceIdBuffer.Clone(src_entry.TriangleFaceIds),
        .FacePrimitives = B->FacePrimitiveBuffer.Clone(src_entry.FacePrimitives),
        .PrimitiveMaterials = B->PrimitiveMaterialBuffer.Clone(src_entry.PrimitiveMaterials),
        .BoneDeform = B->BoneDeformBuffer.Clone(src_entry.BoneDeform),
        .MorphTargets = B->MorphTargetBuffer.Clone(src_entry.MorphTargets),
        .MorphTargetCount = src_entry.MorphTargetCount,
        .TriangleCount = src_entry.TriangleCount,
        .DefaultMorphWeights = src_entry.DefaultMorphWeights,
        .PrimitiveTriangleRanges = src_entry.PrimitiveTriangleRanges,
        .Alive = true,
    });

    MeshConnectivity conn = mesh.GetConnectivity();
    ClearElementStates(vertices, faces, edge_states);
    if (faces.Count > 0) std::ranges::copy(GetFaceSharpness(src_id), GetFaceSharpness(id).begin());
    return {id, std::move(conn)};
}

std::expected<MeshWithMaterials, std::string> MeshStore::LoadMesh(const std::filesystem::path &path, bool weld) {
    auto ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
    MeshDataWithMaterials source;
    try {
        if (ext == ".ply") source = ReadPly(path);
        else if (ext == ".obj") source = ReadObj(path);
        else return std::unexpected{"Unsupported file format: " + ext};
    } catch (const std::exception &e) {
        return std::unexpected{e.what()};
    }
    return MeshWithMaterials{
        .Mesh = CreateMesh(std::move(source.Mesh), std::move(source.Attrs), std::move(source.Primitives), false, {}, {}, weld),
        .Materials = std::move(source.Materials),
    };
}

void MeshStore::SetPositions(const Mesh &mesh, std::span<const vec3> positions) {
    auto vertex_span = B->VerticesBuffer.GetMutable(Entries.at(mesh.GetStoreId()).Vertices);
    for (size_t i = 0; i < positions.size(); ++i) vertex_span[i].Position = positions[i];
    UpdateVertexNormals(mesh);
}
void MeshStore::SetPosition(const Mesh &mesh, uint32_t index, vec3 position) {
    B->VerticesBuffer.GetMutable(Entries.at(mesh.GetStoreId()).Vertices)[index].Position = position;
    // Caller is responsible for updating normals
}

void MeshStore::Release(uint32_t id) {
    if (id >= Entries.size() || !Entries[id].Alive) return;
    auto &entry = Entries[id];
    B->VerticesBuffer.Release(entry.Vertices);
    B->CornerNormalBuffer.Release(entry.CornerNormals);
    B->CustomCornerNormalBuffer.Release(entry.CustomCornerNormals);
    B->CornerTangentBuffer.Release(entry.CornerTangents);
    B->CornerColorBuffer.Release(entry.CornerColors);
    for (const auto &uvs : entry.CornerUvs) B->CornerUvBuffer.Release(uvs);
    B->EdgeSharpnessBuffer.Release(entry.EdgeSharpness);
    B->TriangleFaceIdBuffer.Release(entry.TriangleFaceIds);
    B->FaceFirstTriangleBuffer.Release(entry.FaceData);
    B->FacePrimitiveBuffer.Release(entry.FacePrimitives);
    B->PrimitiveMaterialBuffer.Release(entry.PrimitiveMaterials);
    B->EdgeStateBuffer.Release(entry.EdgeStates);
    B->BoneDeformBuffer.Release(entry.BoneDeform);
    B->MorphTargetBuffer.Release(entry.MorphTargets);
    entry = {};
    FreeIds.emplace_back(id);
}

void MeshStore::Clear() {
    B->ForEachArena([](auto &a) { a.Reset(); });
    B->OverlayVerticesBuffer.Reset();
    B->VertexStateBuffer.UsedSize = 0;
    B->FaceStateBuffer.UsedSize = 0;
    B->FaceSharpnessBuffer.UsedSize = 0;
    Entries.clear();
    FreeIds.clear();
    OverlayEntries.clear();
    OverlayFreeIds.clear();
    Pending = {};
}

// Mirror buffer helpers — uint8_t state per element, sizeof == 1.
static void SyncMirror(mvk::Buffer &mirror, Range range) {
    auto end = vk::DeviceSize(range.Offset) + range.Count;
    mirror.Reserve(end);
    mirror.UsedSize = std::max(mirror.UsedSize, end);
}
static std::span<uint8_t> GetStates(mvk::Buffer &buf, Range range) {
    return {reinterpret_cast<uint8_t *>(buf.GetMutableRange(range.Offset, range.Count).data()), range.Count};
}

Range MeshStore::AllocateVertices(uint32_t count) {
    const auto range = B->VerticesBuffer.Allocate(count);
    SyncMirror(B->VertexStateBuffer, range);
    return range;
}

Range MeshStore::AllocateFaces(uint32_t count) {
    const auto range = B->FaceFirstTriangleBuffer.Allocate(count);
    SyncMirror(B->FaceStateBuffer, range);
    SyncMirror(B->FaceSharpnessBuffer, range);
    return range;
}

std::span<const uint8_t> MeshStore::GetFaceSharpness(uint32_t id) const {
    const auto range = Entries.at(id).FaceData;
    return {reinterpret_cast<const uint8_t *>(B->FaceSharpnessBuffer.GetMappedData().subspan(range.Offset, range.Count).data()), range.Count};
}
std::span<uint8_t> MeshStore::GetFaceSharpness(uint32_t id) { return GetStates(B->FaceSharpnessBuffer, Entries.at(id).FaceData); }
std::span<const uint8_t> MeshStore::GetEdgeSharpness(uint32_t id) const { return B->EdgeSharpnessBuffer.Get(Entries.at(id).EdgeSharpness); }
std::span<uint8_t> MeshStore::GetEdgeSharpness(uint32_t id) { return B->EdgeSharpnessBuffer.GetMutable(Entries.at(id).EdgeSharpness); }
std::span<const vec3> MeshStore::GetCornerNormals(uint32_t id) const { return B->CornerNormalBuffer.Get(Entries.at(id).CornerNormals); }

std::span<uint8_t> MeshStore::GetFaceStates(Range range) { return GetStates(B->FaceStateBuffer, range); }
std::span<uint8_t> MeshStore::GetVertexStates(Range range) { return GetStates(B->VertexStateBuffer, range); }

std::span<const uint8_t> MeshStore::GetVertexStates(Range range) const {
    return {reinterpret_cast<const uint8_t *>(B->VertexStateBuffer.GetMappedData().subspan(range.Offset, range.Count).data()), range.Count};
}

std::span<const uint8_t> MeshStore::GetVertexStates(uint32_t id) const {
    return GetVertexStates(Entries.at(id).Vertices);
}

void MeshStore::ClearElementStates(Range vertices, Range faces, Range edges) {
    std::ranges::fill(GetVertexStates(vertices), 0);
    if (faces.Count > 0) std::ranges::fill(GetFaceStates(faces), 0);
    if (edges.Count > 0) std::ranges::fill(B->EdgeStateBuffer.GetMutable(edges), 0);
}

using namespace he;

void MeshStore::UpdateSoundVertexStates(const Mesh &mesh, std::span<const uint32_t> vertices, std::optional<uint32_t> active_vertex, std::optional<uint32_t> excited_vertex) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    ClearElementStates(entry.Vertices, entry.FaceData, entry.EdgeStates);
    auto vertex_states = GetVertexStates(entry.Vertices);
    for (const auto v : vertices) {
        if (v >= vertex_states.size()) continue;
        vertex_states[v] |= ElementStateSelected;
        if (active_vertex == v) vertex_states[v] |= ElementStateActive;
    }
    if (excited_vertex && *excited_vertex < vertex_states.size()) vertex_states[*excited_vertex] |= ElementStateExcited;
    UpdateEdgeStatesFromVertices(mesh);
}

void MeshStore::UpdateEdgeStatesFromFaces(const Mesh &mesh, std::optional<uint32_t> active_face) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    auto edge_states = B->EdgeStateBuffer.GetMutable(entry.EdgeStates);
    std::ranges::fill(edge_states, uint8_t{0});

    const auto face_states = GetFaceStates(entry.FaceData);
    const uint32_t face_count = std::min(uint32_t(face_states.size()), mesh.FaceCount());
    for (uint32_t fi = 0; fi < face_count; ++fi) {
        if (!(face_states[fi] & ElementStateSelected)) continue;
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
    auto edge_states = B->EdgeStateBuffer.GetMutable(entry.EdgeStates);
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
    const auto edge_states = B->EdgeStateBuffer.GetMutable(entry.EdgeStates);
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

void MeshStore::UpdateVertexStatesFromFaces(const Mesh &mesh, std::optional<uint32_t> active_face) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    auto vertex_states = GetVertexStates(entry.Vertices);
    std::ranges::fill(vertex_states, uint8_t{0});

    const auto face_states = GetFaceStates(entry.FaceData);
    const uint32_t face_count = std::min(uint32_t(face_states.size()), mesh.FaceCount());
    for (uint32_t fi = 0; fi < face_count; ++fi) {
        if (!(face_states[fi] & ElementStateSelected)) continue;
        for (const auto vh : mesh.fv_range(FH{fi})) vertex_states[*vh] |= ElementStateSelected;
    }
    if (active_face && *active_face < mesh.FaceCount()) {
        for (const auto vh : mesh.fv_range(FH{*active_face})) vertex_states[*vh] |= ElementStateActive;
    }
}

void MeshStore::UpdateVertexStatesFromEdges(const Mesh &mesh, std::optional<uint32_t> active_edge) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    auto vertex_states = GetVertexStates(entry.Vertices);
    std::ranges::fill(vertex_states, uint8_t{0});

    const auto edge_states = B->EdgeStateBuffer.Get(entry.EdgeStates);
    const uint32_t edge_count = std::min(uint32_t(edge_states.size() / 2), mesh.EdgeCount());
    for (uint32_t ei = 0; ei < edge_count; ++ei) {
        if (!(edge_states[2 * ei] & ElementStateSelected)) continue;
        const auto hh = mesh.GetHalfedge(EH{ei}, 0);
        vertex_states[*mesh.GetFromVertex(hh)] |= ElementStateSelected;
        vertex_states[*mesh.GetToVertex(hh)] |= ElementStateSelected;
    }
    if (active_edge && *active_edge < mesh.EdgeCount()) {
        const auto hh = mesh.GetHalfedge(EH{*active_edge}, 0);
        vertex_states[*mesh.GetFromVertex(hh)] |= ElementStateActive;
        vertex_states[*mesh.GetToVertex(hh)] |= ElementStateActive;
    }
}

uint32_t MeshStore::AcquireId(Entry &&entry) {
    if (!FreeIds.empty()) {
        const auto reused = FreeIds.back();
        FreeIds.pop_back();
        Entries[reused] = std::move(entry);
        return reused;
    }
    Entries.emplace_back(std::move(entry));
    return uint32_t(Entries.size() - 1);
}
