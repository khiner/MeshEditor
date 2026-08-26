#include "MeshStore.h"

#include "MeshAttributes.h"
#include "Profile.h"
#include "action/SerializeGlm.h" // glm hooks for the entry's authored-normal stash
#include "gpu/CornerClass.h"
#include "gpu/CornerClassEncoding.h"
#include "gpu/FanItemEncoding.h"
#include "metal/BufferArena.h"

#include <glm/geometric.hpp>
#include <zpp_bits.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include <bit>
#include <format>
#include <numeric>

namespace {
constexpr uint8_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};
constexpr uint32_t ClassTagShift{uint32_t(CornerClassEncoding::TagShift)}, ClassIndexMask{uint32_t(CornerClassEncoding::IndexMask)};
constexpr uint32_t UniformFaceOffset{uint32_t(CornerClassEncoding::UniformFaceOffset)};
constexpr uint32_t FanLoopShift{uint32_t(FanItemEncoding::LoopShift)};

constexpr uint32_t BitWords(uint32_t bits) { return (bits + 31u) / 32u; }

std::optional<std::filesystem::path> ResolveTexturePath(const std::filesystem::path &base_dir, const std::string &texture_name) {
    if (texture_name.empty()) return std::nullopt;
    auto texture_path = std::filesystem::path{texture_name};
    if (texture_path.is_relative()) texture_path = base_dir / texture_path;
    if (texture_path.is_relative()) texture_path = std::filesystem::absolute(texture_path);
    return texture_path.lexically_normal();
}

ObjPlyMaterial ToObjPlyMaterial(const tinyobj::material_t &material, uint32_t index, const std::filesystem::path &base_dir) {
    // OBJ/MTL uses Phong terms, converted here to metallic-roughness with a common heuristic.
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

    std::vector<uint32_t> face_verts;
    for (const auto &shape : shapes) {
        size_t vi = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const auto fv = shape.mesh.num_face_vertices[f];
            face_verts.clear();
            for (size_t vi_f = 0; vi_f < fv; ++vi_f) {
                face_verts.emplace_back(FindOrAddVertex(shape.mesh.indices[vi + vi_f]));
            }
            data.AddFace(face_verts);
            const int material_id = f < shape.mesh.material_ids.size() ? shape.mesh.material_ids[f] : -1;
            face_primitive_indices.emplace_back(primitive_index(material_id));
            vi += fv;
        }
    }

    if (!face_primitive_indices.empty()) {
        result.Primitives.ElementPrimitiveIndices = std::move(face_primitive_indices);
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
    data.ReserveFaces(uint32_t(faces->count), 3u);
    std::vector<uint32_t> face_verts;
    for (size_t f = 0; f < faces->count; ++f) {
        const auto face_size = face_buf[offset++];
        face_verts.clear();
        for (uint8_t v = 0; v < face_size; ++v) {
            uint32_t vi = 0;
            std::memcpy(&vi, &face_buf[offset], idx_size);
            offset += idx_size;
            face_verts.emplace_back(vi);
        }
        data.AddFace(face_verts);
    }

    if (colors && colors->count == vertices->count && data.FaceCount() > 0) {
        // Bake vertex colors down to one albedo value, since the render path carries no per-vertex color channel.
        const auto avg = ComputeAverageVertexColor(*colors);
        result.Materials.emplace_back(ObjPlyMaterial{.BaseColorFactor = {avg.x, avg.y, avg.z, 1.f}, .MetallicFactor = 0.f, .RoughnessFactor = 1.f, .Name = "VertexColor"});
        result.Primitives.ElementPrimitiveIndices = std::vector<uint32_t>(data.FaceCount(), 0u);
        result.Primitives.MaterialIndices = std::vector<uint32_t>{0u};
    }

    return result;
}

} // namespace

struct MeshStore::Buffers {
    explicit Buffers(mtl::BufferContext &ctx)
        : FaceFirstTriangleBuffer{ctx, SlotType::ObjectIdBuffer},
          ElementPrimitiveBuffer{ctx, SlotType::ElementPrimitiveBuffer},
          PrimitiveMaterialBuffer{ctx, SlotType::PrimitiveMaterialBuffer},
          BoneDeformBuffer{ctx, SlotType::BoneDeformBuffer},
          MorphTargetBuffer{ctx, SlotType::MorphTargetBuffer},
          VerticesBuffer{ctx, SlotType::VertexBuffer},
          VertexStateBuffer{ctx, 0, SlotType::Buffer},
          FaceStateBuffer{ctx, 0, SlotType::Buffer},
          FaceSharpnessBuffer{ctx, 0, SlotType::Buffer},
          EdgeStateBuffer{ctx, SlotType::Buffer},
          FaceCornerBuffer{ctx, SlotType::IndexBuffer},
          ConnectivityBuffer{ctx, SlotType::Buffer},
          TriangleFaceIdBuffer{ctx, SlotType::ObjectIdBuffer},
          CornerClassBuffer{ctx, SlotType::Buffer},
          BaseSeamNormalBuffer{ctx, SlotType::Buffer},
          BaseVertexNormalBuffer{ctx, 0, SlotType::Buffer},
          BaseFaceNormalBuffer{ctx, 0, SlotType::Buffer},
          PointNormalBuffer{ctx, SlotType::Buffer},
          EdgeSharpnessBuffer{ctx, SlotType::Buffer},
          TetPositionBuffer{ctx, SlotType::Buffer},
          TetEdgeIndexBuffer{ctx, SlotType::Buffer},
          SoundVertexBuffer{ctx, SlotType::Buffer},
          CustomCornerMaskBuffer{ctx, SlotType::Buffer},
          CustomCornerNormalBuffer{ctx, SlotType::Buffer},
          CornerTangentBuffer{ctx, SlotType::CornerTangentBuffer},
          CornerColorBuffer{ctx, SlotType::CornerColorBuffer},
          CornerUvBuffer{ctx, SlotType::CornerUvBuffer},
          AdjacencyBuffer{ctx, SlotType::Buffer} {}

    BufferArena<uint32_t> FaceFirstTriangleBuffer; // Per-face index of first triangle in the index buffer
    BufferArena<uint32_t> ElementPrimitiveBuffer; // Source primitive index per drawn element (per face, or per vertex for point/line meshes)
    BufferArena<uint32_t> PrimitiveMaterialBuffer; // Primitive index -> material index
    BufferArena<BoneDeformVertex> BoneDeformBuffer;
    BufferArena<MorphTargetVertex> MorphTargetBuffer;
    BufferArena<Vertex> VerticesBuffer;
    mtl::Buffer VertexStateBuffer; // Mirrors VerticesBuffer
    mtl::Buffer FaceStateBuffer; // Mirrors FaceFirstTriangleBuffer
    mtl::Buffer FaceSharpnessBuffer; // Mirrors FaceFirstTriangleBuffer. 1 = flat-shaded face (canonical sharpness store)
    BufferArena<uint8_t> EdgeStateBuffer;
    BufferArena<uint32_t> FaceCornerBuffer;
    // Each mesh's half-edge connectivity, laid out as the entry's sub-ranges describe.
    BufferArena<uint32_t> ConnectivityBuffer;
    BufferArena<uint32_t> TriangleFaceIdBuffer; // 1-indexed map from face triangles (in mesh face order) to source face ID
    BufferArena<uint32_t> CornerClassBuffer; // Per-corner CornerClass values, from the sharpness stores
    BufferArena<vec3> BaseSeamNormalBuffer; // Composed sector normal per seam corner
    // Mirrors VerticesBuffer, one vec3 per vertex slot: derived smooth normals for triangle meshes, authored normals for face-less meshes
    mtl::Buffer BaseVertexNormalBuffer;
    mtl::Buffer BaseFaceNormalBuffer; // Mirrors FaceFirstTriangleBuffer, one derived face normal per face slot
    BufferArena<vec3> PointNormalBuffer; // Authored normals of face-less meshes, in vertex order
    BufferArena<uint8_t> EdgeSharpnessBuffer; // One byte per edge, 1 = sharp (canonical sharpness store)
    // Canonical tetrahedral wireframe geometry, one range per mesh that carries a modal solve.
    BufferArena<vec3> TetPositionBuffer;
    BufferArena<uint32_t> TetEdgeIndexBuffer; // Two indices per tet edge
    // Excitable vertex handles per sounding mesh. Derived from the sound model, so Serialize skips it.
    BufferArena<uint32_t> SoundVertexBuffer;
    BufferArena<uvec2> CustomCornerMaskBuffer; // Custom corner-normal presence: a (bitset word, exclusive rank) pair per 32 corners
    BufferArena<vec2> CustomCornerNormalBuffer; // Authored corner-normal (polar, azimuth) offsets from the derived normal, packed to the masked corners
    BufferArena<vec4> CornerTangentBuffer; // Corner-domain attribute layers, one value per corner in fan order
    BufferArena<vec4> CornerColorBuffer;
    BufferArena<vec2> CornerUvBuffer; // Up to four ranges per mesh, one per UV set
    BufferArena<uint32_t> AdjacencyBuffer; // CSR incidence tables, each (bucket count + 1) offsets then items: vertex-to-triangle, vertex-to-edge, and seam-corner-to-triangle ranges

    // Visit every serialized BufferArena in a fixed order, so Serialize/Deserialize stay in lockstep.
    // The derived arenas rebuild from connectivity and the sharpness stores after Deserialize.
    void ForEachSerializedArena(auto &&f) {
        f(VerticesBuffer);
        f(FaceFirstTriangleBuffer);
        f(ElementPrimitiveBuffer);
        f(PrimitiveMaterialBuffer);
        f(BoneDeformBuffer);
        f(MorphTargetBuffer);
        f(EdgeStateBuffer);
        f(TriangleFaceIdBuffer);
        f(FaceCornerBuffer);
        f(ConnectivityBuffer);
        f(EdgeSharpnessBuffer);
        f(CustomCornerMaskBuffer);
        f(CustomCornerNormalBuffer);
        f(CornerTangentBuffer);
        f(CornerColorBuffer);
        f(CornerUvBuffer);
        f(PointNormalBuffer);
        f(TetPositionBuffer);
        f(TetEdgeIndexBuffer);
    }
    static constexpr size_t SerializedArenaCount = 20;

    void ForEachDerivedArena(auto &&f) {
        f(AdjacencyBuffer);
        f(CornerClassBuffer);
        f(BaseSeamNormalBuffer);
    }
};

namespace {
// Save/restore a plain mirror buffer (no allocator) by its used byte region.
std::vector<std::byte> SaveBuffer(const mtl::Buffer &b) {
    const auto mapped = b.Contents();
    const auto used = std::min(size_t(b.UsedSize), mapped.size());
    return {mapped.begin(), mapped.begin() + used};
}
void RestoreBuffer(mtl::Buffer &b, std::span<const std::byte> bytes) {
    b.Reserve(bytes.size());
    if (!bytes.empty()) b.Update(bytes, 0);
    b.UsedSize = bytes.size();
}

// Size a mirror buffer to cover the mirrored arena's element range.
template<typename T> void SyncMirror(mtl::Buffer &mirror, Range range) {
    const auto end = uint64_t(range.Offset + range.Count) * sizeof(T);
    mirror.Reserve(end);
    mirror.UsedSize = std::max(mirror.UsedSize, end);
}
} // namespace

void MeshStore::FillBaseVertexNormalMirror(Range vertices, Range point_normals) {
    const auto normals = B->BaseVertexNormalBuffer.GetMutableSpan<vec3>(vertices);
    if (point_normals.Count > 0) std::ranges::copy(B->PointNormalBuffer.Get(point_normals), normals.begin());
    else std::ranges::fill(normals, vec3{0});
}

std::vector<std::byte> MeshStore::Serialize() const {
    std::vector<ArenaState> arenas;
    arenas.reserve(Buffers::SerializedArenaCount);
    B->ForEachSerializedArena([&](const auto &a) { arenas.push_back(a.Save()); });
    const auto vertex_state = SaveBuffer(B->VertexStateBuffer);
    const auto face_state = SaveBuffer(B->FaceStateBuffer);
    const auto face_sharpness = SaveBuffer(B->FaceSharpnessBuffer);

    // Serialize from non-const copies: zpp mis-encodes a const aggregate this large, and these match the types Deserialize reads back into.
    auto entries = Entries;
    auto free_ids = FreeIds;
    // The derived state rebuilds after restore with its own arena layout.
    // Clear its ranges and the authored-normal stash from the serialized entries.
    for (auto &e : entries) {
        e.CornerClasses = e.VertexFanAdjacency = e.VertexEdgeAdjacency = e.SeamFans = e.BaseSeamNormals = {};
        e.MorphShadingAuthored = false;
        e.SeamCornerCount = 0;
        e.UniformCornerClass = CornerClass::Vertex;
        e.AuthoredCornerNormals = {};
    }
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
    if (arenas.size() != Buffers::SerializedArenaCount) return;

    size_t i = 0;
    B->ForEachSerializedArena([&](auto &a) { a.Restore(std::move(arenas[i++])); });
    // RebuildDerived refills arenas per mesh once connectivity is restored.
    B->ForEachDerivedArena([](auto &a) { a.Reset(); });
    RestoreBuffer(B->VertexStateBuffer, vertex_state);
    RestoreBuffer(B->FaceStateBuffer, face_state);
    RestoreBuffer(B->FaceSharpnessBuffer, face_sharpness);
    Entries = std::move(entries);
    FreeIds = std::move(free_ids);

    // Triangle meshes rederive their region, and face-less meshes take their authored point normals back.
    B->BaseVertexNormalBuffer.UsedSize = 0;
    B->BaseFaceNormalBuffer.UsedSize = 0;
    for (const auto &e : Entries) {
        if (!e.Alive) continue;
        SyncMirror<vec3>(B->BaseVertexNormalBuffer, e.Vertices);
        SyncMirror<vec3>(B->BaseFaceNormalBuffer, e.FaceData);
        FillBaseVertexNormalMirror(e.Vertices, e.PointNormals);
    }
}

// Derived arena offsets follow rebuild order, so sort by store id for a deterministic layout.
void MeshStore::RebuildDerived(std::span<Mesh> meshes) {
    std::ranges::sort(meshes, {}, &Mesh::GetStoreId);
    for (const auto &mesh : meshes) {
        BuildVertexAdjacency(mesh);
        UpdateCornerClassification(mesh);
    }
}

uint32_t MeshStore::GetCornerClassOffset(uint32_t id) const {
    const auto &entry = Entries.at(id);
    if (entry.CornerClasses.Count > 0) return entry.CornerClasses.Offset;
    return entry.UniformCornerClass == CornerClass::Face ? UniformFaceOffset : InvalidOffset;
}

std::span<const uint32_t> MeshStore::GetCornerClasses(uint32_t id) const {
    return B->CornerClassBuffer.Get(Entries.at(id).CornerClasses);
}

std::span<const uvec2> MeshStore::GetCustomCornerMasks(uint32_t id) const {
    return B->CustomCornerMaskBuffer.Get(Entries.at(id).CustomCornerMasks);
}

MeshStore::MeshStore(mtl::BufferContext &ctx) : B{std::make_unique<Buffers>(ctx)} {}
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
uint32_t MeshStore::GetCornerTangentSlot() const { return B->CornerTangentBuffer.Buffer.Slot; }
uint32_t MeshStore::GetCornerColorSlot() const { return B->CornerColorBuffer.Buffer.Slot; }
uint32_t MeshStore::GetCornerUvSlot() const { return B->CornerUvBuffer.Buffer.Slot; }
uint32_t MeshStore::GetEdgeSharpnessSlot() const { return B->EdgeSharpnessBuffer.Buffer.Slot; }
uint32_t MeshStore::GetTetPositionSlot() const { return B->TetPositionBuffer.Buffer.Slot; }
uint32_t MeshStore::GetTetEdgeIndexSlot() const { return B->TetEdgeIndexBuffer.Buffer.Slot; }

TetBuffers MeshStore::AllocateTets(std::span<const vec3> positions, std::span<const uint32_t> edge_indices) {
    const TetBuffers tets{B->TetPositionBuffer.Allocate(uint32_t(positions.size())), B->TetEdgeIndexBuffer.Allocate(uint32_t(edge_indices.size()))};
    std::ranges::copy(positions, B->TetPositionBuffer.GetMutable(tets.Positions).begin());
    std::ranges::copy(edge_indices, B->TetEdgeIndexBuffer.GetMutable(tets.EdgeIndices).begin());
    return tets;
}

Range MeshStore::AllocateSoundVertices(std::span<const uint32_t> vertices) {
    const auto range = B->SoundVertexBuffer.Allocate(uint32_t(vertices.size()));
    std::ranges::copy(vertices, B->SoundVertexBuffer.GetMutable(range).begin());
    return range;
}

void MeshStore::ReleaseSoundVertices(Range range) { B->SoundVertexBuffer.Release(range); }
std::span<const uint32_t> MeshStore::GetSoundVertices(Range range) const { return B->SoundVertexBuffer.Get(range); }
uint32_t MeshStore::GetSoundVertexSlot() const { return B->SoundVertexBuffer.Buffer.Slot; }

void MeshStore::ReleaseTets(TetBuffers tets) {
    B->TetPositionBuffer.Release(tets.Positions);
    B->TetEdgeIndexBuffer.Release(tets.EdgeIndices);
}
uint32_t MeshStore::GetElementPrimitiveSlot() const { return B->ElementPrimitiveBuffer.Buffer.Slot; }
uint32_t MeshStore::GetPrimitiveMaterialSlot() const { return B->PrimitiveMaterialBuffer.Buffer.Slot; }
uint32_t MeshStore::GetBoneDeformSlot() const { return B->BoneDeformBuffer.Buffer.Slot; }
uint32_t MeshStore::GetMorphTargetSlot() const { return B->MorphTargetBuffer.Buffer.Slot; }
uint32_t MeshStore::GetAdjacencySlot() const { return B->AdjacencyBuffer.Buffer.Slot; }
uint32_t MeshStore::GetCornerClassSlot() const { return B->CornerClassBuffer.Buffer.Slot; }
uint32_t MeshStore::GetCustomCornerMaskSlot() const { return B->CustomCornerMaskBuffer.Buffer.Slot; }
uint32_t MeshStore::GetCustomCornerNormalSlot() const { return B->CustomCornerNormalBuffer.Buffer.Slot; }
uint32_t MeshStore::GetBaseSeamNormalSlot() const { return B->BaseSeamNormalBuffer.Buffer.Slot; }
uint32_t MeshStore::GetBaseVertexNormalSlot() const { return B->BaseVertexNormalBuffer.Slot; }
uint32_t MeshStore::GetBaseFaceNormalSlot() const { return B->BaseFaceNormalBuffer.Slot; }
uint32_t MeshStore::GetFaceFirstTriangleSlot() const { return B->FaceFirstTriangleBuffer.Buffer.Slot; }

std::span<const vec3> MeshStore::GetBaseVertexNormals(uint32_t id) const { return B->BaseVertexNormalBuffer.GetSpan<vec3>(Entries.at(id).Vertices); }
std::span<vec3> MeshStore::GetBaseVertexNormals(uint32_t id) { return B->BaseVertexNormalBuffer.GetMutableSpan<vec3>(Entries.at(id).Vertices); }
std::span<const vec3> MeshStore::GetBaseFaceNormals(uint32_t id) const { return B->BaseFaceNormalBuffer.GetSpan<vec3>(Entries.at(id).FaceData); }
std::span<vec3> MeshStore::GetBaseFaceNormals(uint32_t id) { return B->BaseFaceNormalBuffer.GetMutableSpan<vec3>(Entries.at(id).FaceData); }
std::span<const vec3> MeshStore::GetBaseSeamNormals(uint32_t id) const { return B->BaseSeamNormalBuffer.Get(Entries.at(id).BaseSeamNormals); }
std::span<vec3> MeshStore::GetBaseSeamNormals(uint32_t id) { return B->BaseSeamNormalBuffer.GetMutable(Entries.at(id).BaseSeamNormals); }
std::span<const vec3> MeshStore::GetPointNormals(uint32_t id) const { return B->PointNormalBuffer.Get(Entries.at(id).PointNormals); }

SlottedRange MeshStore::GetFaceStateRange(uint32_t id) const { return {Entries.at(id).FaceData, B->FaceStateBuffer.Slot}; }
SlottedRange MeshStore::GetEdgeStateRange(uint32_t id) const { return B->EdgeStateBuffer.Slotted(Entries.at(id).EdgeStates); }
Range MeshStore::GetEdgeSharpnessRange(uint32_t id) const { return Entries.at(id).EdgeSharpness; }
Range MeshStore::GetCornerTangentRange(uint32_t id) const { return Entries.at(id).CornerTangents; }
Range MeshStore::GetCornerColorRange(uint32_t id) const { return Entries.at(id).CornerColors; }
Range MeshStore::GetCornerUvRange(uint32_t id, uint32_t set) const { return Entries.at(id).CornerUvs.at(set); }
std::span<const vec4> MeshStore::GetCornerTangents(uint32_t id) const { return B->CornerTangentBuffer.Get(Entries.at(id).CornerTangents); }
std::span<const vec4> MeshStore::GetCornerColors(uint32_t id) const { return B->CornerColorBuffer.Get(Entries.at(id).CornerColors); }
std::span<const vec2> MeshStore::GetCornerUvs(uint32_t id, uint32_t set) const { return B->CornerUvBuffer.Get(Entries.at(id).CornerUvs.at(set)); }
SlottedRange MeshStore::GetFaceIdRange(uint32_t id) const { return B->TriangleFaceIdBuffer.Slotted(Entries.at(id).TriangleFaceIds); }
SlottedRange MeshStore::GetElementPrimitiveRange(uint32_t id) const { return B->ElementPrimitiveBuffer.Slotted(Entries.at(id).ElementPrimitives); }
SlottedRange MeshStore::GetPrimitiveMaterialRange(uint32_t id) const { return B->PrimitiveMaterialBuffer.Slotted(Entries.at(id).PrimitiveMaterials); }

std::span<const uint32_t> MeshStore::GetTriangleFaceIds(uint32_t id) const { return B->TriangleFaceIdBuffer.Get(Entries.at(id).TriangleFaceIds); }
void MeshStore::EnsureEdgeStates(const Mesh &mesh) {
    auto &entry = Entries.at(mesh.GetStoreId());
    if (entry.EdgeStates.Count > 0 || mesh.EdgeCount() == 0) return;
    entry.EdgeStates = B->EdgeStateBuffer.Allocate(mesh.EdgeCount() * 2);
    std::ranges::fill(B->EdgeStateBuffer.GetMutable(entry.EdgeStates), 0);
}

std::span<const uint32_t> MeshStore::GetFaceCorners(uint32_t id) const { return B->FaceCornerBuffer.Get(Entries.at(id).FaceCorners); }
SlottedRange MeshStore::GetFaceCornerRange(uint32_t id) const { return B->FaceCornerBuffer.Slotted(Entries.at(id).FaceCorners); }
std::span<const uint32_t> MeshStore::GetFaceFirstTriangles(uint32_t id) const { return B->FaceFirstTriangleBuffer.Get(Entries.at(id).FaceData); }
std::span<const uint32_t> MeshStore::GetElementPrimitiveIndices(uint32_t id) const { return B->ElementPrimitiveBuffer.Get(Entries.at(id).ElementPrimitives); }
std::span<uint32_t> MeshStore::GetElementPrimitiveIndices(uint32_t id) { return B->ElementPrimitiveBuffer.GetMutable(Entries.at(id).ElementPrimitives); }
std::span<const uint32_t> MeshStore::GetPrimitiveMaterialIndices(uint32_t id) const { return B->PrimitiveMaterialBuffer.Get(Entries.at(id).PrimitiveMaterials); }
std::span<uint32_t> MeshStore::GetPrimitiveMaterialIndices(uint32_t id) { return B->PrimitiveMaterialBuffer.GetMutable(Entries.at(id).PrimitiveMaterials); }

namespace {
VertexAdjacency SliceAdjacency(std::span<const uint32_t> words, uint32_t bucket_count) {
    if (words.empty()) return {};
    return {words.first(bucket_count + 1), words.subspan(bucket_count + 1)};
}

// An authored corner normal within 0.05 degrees of the derived one counts as derivable and is dropped.
// Export-pipeline rounding sits below this angle and deliberate normal authoring sits well above it.
constexpr float AuthoredMatchDot{0.99999962f};

// Whether `normal` matches the unit-or-zero `reference` within the authored match gate.
// Nullopt when `normal` is degenerate. A zero reference matches nothing.
std::optional<bool> NormalsMatch(vec3 normal, vec3 reference) {
    const auto len = glm::length(normal);
    if (len < 1e-6f) return {};
    return glm::dot(normal / len, reference) >= AuthoredMatchDot;
}

// Orthonormal frame anchoring a corner's authored-normal offset.
// Axes: the derived normal, the corner's first non-degenerate outgoing triangle edge projected off it, and their cross.
// Degenerate inputs take fixed fallback axes, deterministic from the same inputs, so encode and decode rebuild the same frame.
// The vertex shader rebuilds the frame from current local positions, so offsets ride the deformation.
struct CornerNormalFrame {
    vec3 Normal, Ref, Ortho;
};
CornerNormalFrame ComputeCornerFrame(vec3 normal, std::span<const uint32_t> indices, std::span<const Vertex> vertices, uint32_t ci) {
    const auto n = glm::length(normal) > 0.f ? normal : vec3{0, 0, 1};
    const auto tri = ci / 3 * 3;
    const auto k = ci - tri;
    const auto p0 = vertices[indices[tri + k]].Position;
    const auto ref = [&]() -> vec3 {
        for (uint32_t other = 1; other < 3; ++other) {
            const auto edge = vertices[indices[tri + (k + other) % 3]].Position - p0;
            const auto rejected = edge - n * glm::dot(edge, n);
            const auto len = glm::length(rejected);
            // An edge nearly parallel to the normal rejects to cancellation noise, so its perpendicular part must be a meaningful fraction of its length to anchor the frame.
            if (len > 1e-3f * glm::length(edge)) return rejected / len;
        }
        const auto axis = std::abs(n.x) < 0.5f ? vec3{1, 0, 0} : vec3{0, 1, 0};
        return glm::normalize(glm::cross(n, axis));
    }();
    return {n, ref, glm::cross(n, ref)};
}

// A custom normal as (polar, azimuth) angles in the corner frame.
vec2 EncodeNormalOffset(vec3 custom, const CornerNormalFrame &frame) {
    const auto polar = std::acos(std::clamp(glm::dot(custom, frame.Normal), -1.f, 1.f));
    const auto azimuth = std::atan2(glm::dot(custom, frame.Ortho), glm::dot(custom, frame.Ref));
    return {polar, azimuth};
}

vec3 DecodeNormalOffset(vec2 offset, const CornerNormalFrame &frame) {
    return std::cos(offset.x) * frame.Normal + std::sin(offset.x) * (std::cos(offset.y) * frame.Ref + std::sin(offset.y) * frame.Ortho);
}

// The corner's class normal from the given per-class sources: its vertex's smooth normal, its face's normal, or its seam sector's normal.
vec3 ComposeCornerNormal(std::span<const uint32_t> classes, CornerClass uniform_class, uint32_t ci, std::span<const uint32_t> indices, std::span<const uint32_t> face_ids, const CornerNormalSources &sources) {
    const auto value = classes.empty() ? uint32_t(uniform_class) << ClassTagShift : classes[ci];
    switch (CornerClass(value >> ClassTagShift)) {
        case CornerClass::Face: return sources.FaceNormals[face_ids[ci / 3] - 1];
        case CornerClass::Seam: return sources.SeamNormals[value & ClassIndexMask];
        default: return sources.VertexNormals[indices[ci]];
    }
}
} // namespace

void MeshStore::UpdateCornerClassification(const Mesh &mesh) {
    const profile::CpuScope scope{"CornerClassification"};
    const auto id = mesh.GetStoreId();
    auto &entry = Entries.at(id);
    if (entry.TriangleCount == 0) return;
    const auto sharp_faces = GetFaceSharpness(id);
    const auto sharp_edges = GetEdgeSharpness(id);
    const auto is_set = [](uint8_t s) { return s != 0; };
    const auto [any_face_sharp, all_faces_sharp] = GetFaceSharpnessSummary(id);
    const bool any_sharp = any_face_sharp || std::ranges::any_of(sharp_edges, is_set);
    // A mesh classifying every corner the same stores just the uniform tag.
    // No sharpness means every corner Vertex, and all-sharp faces mean every corner Face.
    if (!any_sharp || all_faces_sharp) {
        entry.UniformCornerClass = all_faces_sharp ? CornerClass::Face : CornerClass::Vertex;
        B->CornerClassBuffer.Release(entry.CornerClasses);
        B->AdjacencyBuffer.Release(entry.SeamFans);
        B->BaseSeamNormalBuffer.Release(entry.BaseSeamNormals);
        entry.CornerClasses = entry.SeamFans = entry.BaseSeamNormals = {};
        entry.SeamCornerCount = 0;
        return;
    }
    if (entry.CornerClasses.Count == 0) entry.CornerClasses = B->CornerClassBuffer.Allocate(entry.TriangleCount * 3);
    const auto classes = B->CornerClassBuffer.GetMutable(entry.CornerClasses);
    const auto &c = mesh.GetConnectivity();
    const auto face_sharp = [&](Mesh::FH fh) { return *fh < sharp_faces.size() && sharp_faces[*fh] != 0; };
    const auto edge_sharp = [&](Mesh::HH hh) { const auto eh = mesh.GetEdge(hh); return *eh < sharp_edges.size() && sharp_edges[*eh] != 0; };

    // Vertices touching a discontinuity get seam sectors, and all others take the vertex normal.
    static thread_local std::vector<uint8_t> touched;
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

    static thread_local std::vector<uint32_t> seam_offsets, seam_items;
    const auto add_incident = [&](Mesh::FH fh, uint32_t k) { seam_items.push_back(*fh | (k << FanLoopShift)); };
    const auto loop_position = [&](Mesh::FH fh, Mesh::HH h_v) {
        uint32_t k = 0;
        for (auto h = c.FaceHalfedge(*fh); h != h_v; h = c.Next(h)) ++k;
        return k;
    };
    seam_offsets.clear();
    seam_items.clear();
    seam_offsets.push_back(0);

    // Collect the seam sector around `h_in`'s corner vertex, walking both ways from `fh` until a sharp edge, sharp face, or boundary cuts the fan.
    const auto collect_sector = [&](Mesh::FH fh, Mesh::HH h_in, uint32_t k_in) {
        constexpr uint32_t MaxFan{64};
        add_incident(fh, k_in);
        bool full_loop = false;
        auto h = h_in;
        for (uint32_t i = 0; i < MaxFan; ++i) {
            const auto out = c.Next(h); // Leaves the corner vertex within the current face.
            if (edge_sharp(out)) break;
            const auto opp = c.Opposites[*out];
            if (!opp) break;
            const auto nf = c.FaceOf(opp);
            if (!nf) break;
            if (nf == fh) {
                full_loop = true;
                break;
            }
            if (face_sharp(nf)) break;
            add_incident(nf, loop_position(nf, opp));
            h = opp;
        }
        if (!full_loop) {
            h = h_in;
            for (uint32_t i = 0; i < MaxFan; ++i) {
                if (edge_sharp(h)) break;
                const auto opp = c.Opposites[*h];
                if (!opp) break;
                const auto nf = c.FaceOf(opp);
                if (!nf || nf == fh || face_sharp(nf)) break;
                // Continue from the halfedge entering the corner vertex within `nf`.
                auto prev = opp;
                for (auto walk = c.Next(opp); walk != opp; walk = c.Next(walk)) prev = walk;
                add_incident(nf, loop_position(nf, prev));
                h = prev;
            }
        }
        seam_offsets.push_back(uint32_t(seam_items.size()));
    };

    uint32_t ci = 0;
    static thread_local std::vector<uint32_t> face_classes; // Per-face corner classes in vertex order, emitted in fan order.
    for (const auto fh : mesh.faces()) {
        const auto tri_count = mesh.GetValence(fh) - 2;
        if (face_sharp(fh)) {
            for (uint32_t t = 0; t < tri_count * 3; ++t) classes[ci++] = uint32_t(CornerClass::Face) << ClassTagShift;
            continue;
        }
        face_classes.clear();
        uint32_t k = 0;
        for (const auto hh : mesh.fh_range(fh)) {
            const auto vh = mesh.GetToVertex(hh);
            if (touched[*vh]) {
                const auto s = uint32_t(seam_offsets.size() - 1);
                collect_sector(fh, hh, k);
                face_classes.push_back((uint32_t(CornerClass::Seam) << ClassTagShift) | s);
            } else {
                face_classes.push_back(uint32_t(CornerClass::Vertex) << ClassTagShift);
            }
            ++k;
        }
        for (uint32_t i = 1; i + 1 < face_classes.size(); ++i) {
            classes[ci++] = face_classes[0];
            classes[ci++] = face_classes[i];
            classes[ci++] = face_classes[i + 1];
        }
    }

    entry.SeamCornerCount = uint32_t(seam_offsets.size() - 1);
    const auto seam_words = entry.SeamCornerCount > 0 ? uint32_t(seam_offsets.size() + seam_items.size()) : 0u;
    if (entry.SeamFans.Count != seam_words) {
        B->AdjacencyBuffer.Release(entry.SeamFans);
        entry.SeamFans = B->AdjacencyBuffer.Allocate(seam_words);
    }
    if (seam_words > 0) {
        const auto out = B->AdjacencyBuffer.GetMutable(entry.SeamFans);
        std::ranges::copy(seam_offsets, out.begin());
        std::ranges::copy(seam_items, out.begin() + seam_offsets.size());
    }
    if (entry.BaseSeamNormals.Count != entry.SeamCornerCount) {
        B->BaseSeamNormalBuffer.Release(entry.BaseSeamNormals);
        entry.BaseSeamNormals = B->BaseSeamNormalBuffer.Allocate(entry.SeamCornerCount);
    }
}

// Vertex corners take the vertex's smooth normal, Face corners their face's normal, and Seam corners their composed sector normal.
std::span<const vec3> MeshStore::GetCornerNormals(const Mesh &mesh) const {
    return GetCornerNormals(mesh, mesh.CreateTriangleIndices());
}

std::span<const vec3> MeshStore::GetCornerNormals(const Mesh &mesh, std::span<const uint32_t> indices) const {
    const auto id = mesh.GetStoreId();
    const auto &entry = Entries.at(id);
    static thread_local std::vector<vec3> corners;
    corners.resize(size_t{entry.TriangleCount} * 3);
    if (corners.empty()) return corners;
    const CornerNormalSources sources{GetBaseVertexNormals(id), GetBaseSeamNormals(id), GetBaseFaceNormals(id)};
    const auto classes = B->CornerClassBuffer.Get(entry.CornerClasses);
    const auto face_ids = B->TriangleFaceIdBuffer.Get(entry.TriangleFaceIds);
    for (uint32_t ci = 0; ci < corners.size(); ++ci) {
        corners[ci] = ComposeCornerNormal(classes, entry.UniformCornerClass, ci, indices, face_ids, sources);
    }
    const auto masks = B->CustomCornerMaskBuffer.Get(entry.CustomCornerMasks);
    if (masks.empty()) return corners;
    const auto packed = B->CustomCornerNormalBuffer.Get(entry.CustomCornerNormals);
    const auto vertices = B->VerticesBuffer.Get(entry.Vertices);
    size_t next = 0;
    for (size_t w = 0; w < masks.size(); ++w) {
        for (auto word = masks[w].x; word != 0; word &= word - 1) {
            const auto i = w * 32 + std::countr_zero(word);
            corners[i] = DecodeNormalOffset(packed[next++], ComputeCornerFrame(corners[i], indices, vertices, i));
        }
    }
    return corners;
}

void MeshStore::EncodeAuthoredCornerNormals(const Mesh &mesh) {
    auto &entry = Entries.at(mesh.GetStoreId());
    const auto authored = std::exchange(entry.AuthoredCornerNormals, {});
    if (authored.empty() || entry.TriangleCount == 0) return;
    const auto indices = mesh.CreateTriangleIndices();
    // The custom layer is empty at this point, so this is the raw derived normal per corner.
    const auto derived = GetCornerNormals(mesh, indices);
    const auto vertices = B->VerticesBuffer.Get(entry.Vertices);
    std::vector<uvec2> masks((derived.size() + 31) / 32, uvec2{0});
    std::vector<vec2> packed;
    for (size_t i = 0; i < derived.size() && i < authored.size(); ++i) {
        const auto authored_normal = authored[i];
        if (NormalsMatch(authored_normal, derived[i]).value_or(true)) continue;
        masks[i / 32].x |= 1u << (i % 32);
        packed.emplace_back(EncodeNormalOffset(authored_normal / glm::length(authored_normal), ComputeCornerFrame(derived[i], indices, vertices, i)));
    }
    if (packed.empty()) return;
    uint32_t rank = 0;
    for (auto &mask : masks) {
        mask.y = rank;
        rank += std::popcount(mask.x);
    }
    entry.CustomCornerMasks = B->CustomCornerMaskBuffer.Allocate(std::span<const uvec2>{masks});
    entry.CustomCornerNormals = B->CustomCornerNormalBuffer.Allocate(std::span<const vec2>{packed});
}

bool MeshStore::MorphTargetsAuthorNormalDeltas(uint32_t id) const {
    return std::ranges::any_of(GetMorphTargets(id), [](const auto &t) { return t.NormalDelta != vec3{0}; });
}

void MeshStore::UpdateMorphShadingAuthored(const Mesh &mesh, std::span<const CornerNormalSources> poses) {
    const auto id = mesh.GetStoreId();
    auto &entry = Entries.at(id);
    entry.MorphShadingAuthored = false;
    if (!entry.HasAuthoredNormals || entry.TriangleCount == 0 || entry.MorphTargetCount == 0) return;
    // A target authoring normal deltas states the morphed shading normals directly.
    if (MorphTargetsAuthorNormalDeltas(id)) {
        entry.MorphShadingAuthored = true;
        return;
    }
    if (poses.empty()) return;
    // Position-only targets pin the authored normals in place.
    // Authorship matters when any listed full-weight pose derives a corner normal away from the rest normal it would pin.
    const auto indices = mesh.CreateTriangleIndices();
    const auto classes = B->CornerClassBuffer.Get(entry.CornerClasses);
    const auto face_ids = B->TriangleFaceIdBuffer.Get(entry.TriangleFaceIds);
    const auto compose = [&](const CornerNormalSources &normals, uint32_t ci) {
        return ComposeCornerNormal(classes, entry.UniformCornerClass, ci, indices, face_ids, normals);
    };
    const CornerNormalSources rest{GetBaseVertexNormals(id), GetBaseSeamNormals(id), GetBaseFaceNormals(id)};
    for (uint32_t ci = 0; ci < indices.size(); ++ci) {
        const auto rest_normal = compose(rest, ci);
        if (rest_normal == vec3{0}) continue;
        for (const auto &pose : poses) {
            const auto posed = compose(pose, ci);
            if (posed != vec3{0} && glm::dot(rest_normal, posed) < AuthoredMatchDot) {
                entry.MorphShadingAuthored = true;
                return;
            }
        }
    }
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
        const auto opposite = c.Opposites[*hh];
        const auto opposite_face = opposite ? c.FaceOf(opposite) : Mesh::FH{};
        sharp[ei] = face && opposite_face && glm::dot(face_normals[*face], face_normals[*opposite_face]) < cos_angle ? 1 : 0;
    }
}

SharpnessSummary MeshStore::GetFaceSharpnessSummary(uint32_t id) const {
    const auto s = GetFaceSharpness(id);
    const auto sharp = [](uint8_t b) { return b != 0; };
    return {std::ranges::any_of(s, sharp), !s.empty() && std::ranges::all_of(s, sharp)};
}

namespace {
void WriteVertices(std::span<Vertex> dst, std::span<const vec3> positions) {
    for (uint32_t i = 0; i < positions.size(); ++i) dst[i] = {.Position = positions[i]};
}
} // namespace

void MeshStore::AllocateConnectivity(uint32_t id, uint32_t vertex_count, uint32_t halfedge_count, uint32_t face_count, bool face_starts) {
    auto &entry = Entries.at(id);
    entry.ConnectivityVertices = vertex_count;
    entry.ConnectivityHalfedges = halfedge_count;
    entry.ConnectivityFaces = face_count;
    entry.ConnectivityFaceStarts = face_starts;
    const auto words = BitWords(halfedge_count);
    // Outgoing, opposites, the two bit tables, the samples at their bound, then an n-gon mesh's face starts.
    entry.Connectivity = B->ConnectivityBuffer.Allocate(vertex_count + halfedge_count + 3 * words + (face_starts ? face_count : 0u));
}

ConnectivityStorage MeshStore::SliceConnectivity(const Entry &entry) const {
    const auto vertices = entry.ConnectivityVertices, halfedges = entry.ConnectivityHalfedges;
    const auto words = BitWords(halfedges);
    const auto run = B->ConnectivityBuffer.GetMutable(entry.Connectivity);
    const auto handles = [](std::span<uint32_t> w) { return std::span{reinterpret_cast<he::HH *>(w.data()), w.size()}; };
    return {
        .OutgoingHalfedges = handles(run.subspan(0, vertices)),
        .Opposites = handles(run.subspan(vertices, halfedges)),
        .EdgeFirstBits = run.subspan(vertices + halfedges, words),
        .EdgeFirstRanks = run.subspan(vertices + halfedges + words, words),
        .EdgeSamples = run.subspan(vertices + halfedges + 2 * words, words),
        .Faces = entry.ConnectivityFaceStarts ?
            std::span{reinterpret_cast<MeshConnectivity::Face *>(run.data() + vertices + halfedges + 3 * words), entry.ConnectivityFaces} :
            std::span<MeshConnectivity::Face>{},
    };
}

ConnectivityStorage MeshStore::GetConnectivityStorage(uint32_t id) { return SliceConnectivity(Entries.at(id)); }

void MeshStore::PlaceConnectivity(uint32_t id, const BuiltConnectivity &built) {
    auto &entry = Entries.at(id);
    entry.ConnectivityEdgeCount = built.EdgeCount;
    // Only a non-manifold mesh keeps an edge list, which the bit ranks cannot answer for.
    if (built.Edges.empty()) return;
    entry.ConnectivityEdges = B->ConnectivityBuffer.Allocate(uint32_t(built.Edges.size()));
    entry.ConnectivityHalfedgeToEdge = B->ConnectivityBuffer.Allocate(uint32_t(built.HalfedgeToEdge.size()));
    const auto edges = B->ConnectivityBuffer.GetMutable(entry.ConnectivityEdges);
    const auto halfedge_to_edge = B->ConnectivityBuffer.GetMutable(entry.ConnectivityHalfedgeToEdge);
    for (uint32_t i = 0; i < built.Edges.size(); ++i) edges[i] = *built.Edges[i];
    for (uint32_t i = 0; i < built.HalfedgeToEdge.size(); ++i) halfedge_to_edge[i] = *built.HalfedgeToEdge[i];
}

SlottedRange MeshStore::GetConnectivityRange(uint32_t id) const { return B->ConnectivityBuffer.Slotted(Entries.at(id).Connectivity); }

void MeshStore::SetConnectivityEdgeCount(uint32_t id, uint32_t edge_count) { Entries.at(id).ConnectivityEdgeCount = edge_count; }

MeshConnectivity MeshStore::GetConnectivity(uint32_t id) const {
    const auto &entry = Entries.at(id);
    MeshConnectivity c;
    c.VertexCount = entry.ConnectivityVertices;
    c.EdgeCount = entry.ConnectivityEdgeCount;
    c.FaceCount = entry.ConnectivityFaces;
    if (entry.Connectivity.Count == 0) return c;
    const auto run = SliceConnectivity(entry);
    c.OutgoingHalfedges = run.OutgoingHalfedges;
    c.Opposites = run.Opposites;
    c.Faces = run.Faces;
    // A non-manifold mesh reads its edges from the list instead of the bit ranks.
    if (entry.ConnectivityEdges.Count > 0) {
        const auto edges = B->ConnectivityBuffer.Get(entry.ConnectivityEdges);
        c.Edges = {reinterpret_cast<const he::HH *>(edges.data()), edges.size()};
        const auto halfedge_to_edge = B->ConnectivityBuffer.Get(entry.ConnectivityHalfedgeToEdge);
        c.HalfedgeToEdge = {reinterpret_cast<const he::EH *>(halfedge_to_edge.data()), halfedge_to_edge.size()};
        return c;
    }
    c.EdgeFirstBits = run.EdgeFirstBits;
    c.EdgeFirstRanks = run.EdgeFirstRanks;
    c.EdgeSamples = run.EdgeSamples.first(BitWords(entry.ConnectivityEdgeCount));
    return c;
}

uint32_t MeshStore::CreateMeshSource(const MeshData &data) {
    const auto vertices = AllocateVertices(data.Positions.size());
    WriteVertices(B->VerticesBuffer.GetMutable(vertices), data.Positions);
    const auto id = AcquireId({.Vertices = vertices, .Alive = true});
    // A face mesh's corners are its face loops. An edge mesh has no weld and takes its corners in CreateMesh.
    if (data.FaceCount() > 0) Entries[id].FaceCorners = B->FaceCornerBuffer.Allocate(std::span<const uint32_t>{data.FaceCorners});
    return id;
}

void MeshStore::CreateDeformSource(uint32_t id, const std::optional<ArmatureDeformData> &deform, const std::optional<MorphTargetData> &morph) {
    auto &entry = Entries[id];
    const uint32_t vertex_count = entry.Vertices.Count;
    if (vertex_count == 0) return;
    if (deform) {
        entry.BoneDeform = B->BoneDeformBuffer.Allocate(vertex_count);
        auto bone_deform = B->BoneDeformBuffer.GetMutable(entry.BoneDeform);
        for (uint32_t i = 0; i < vertex_count; ++i) {
            bone_deform[i] = {.Joints = deform->Joints[i], .Weights = deform->Weights[i]};
        }
    }
    if (morph && morph->TargetCount > 0) {
        entry.MorphTargetCount = morph->TargetCount;
        const uint32_t total = entry.MorphTargetCount * vertex_count;
        entry.MorphTargets = B->MorphTargetBuffer.Allocate(total);
        auto morph_targets = B->MorphTargetBuffer.GetMutable(entry.MorphTargets);
        const bool has_normal_deltas = !morph->NormalDeltas.empty();
        for (uint32_t i = 0; i < total; ++i) {
            morph_targets[i] = {
                .PositionDelta = morph->PositionDeltas[i],
                .NormalDelta = has_normal_deltas ? morph->NormalDeltas[i] : vec3{0},
            };
        }
        entry.DefaultMorphWeights = morph->DefaultWeights;
        entry.DefaultMorphWeights.resize(entry.MorphTargetCount, 0.f);
    }
}

void MeshStore::ShrinkMeshSource(uint32_t id, uint32_t welded_vertices) {
    auto &entry = Entries.at(id);
    B->VerticesBuffer.Shrink(entry.Vertices, welded_vertices);
    B->BoneDeformBuffer.Shrink(entry.BoneDeform, welded_vertices);
    B->MorphTargetBuffer.Shrink(entry.MorphTargets, entry.MorphTargetCount * welded_vertices);
}

uint32_t MeshStore::AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &attrs) {
    const auto vertices = AllocateVertices(positions.size());
    WriteVertices(B->VerticesBuffer.GetMutable(vertices), positions);
    // Face-less meshes keep authored normals as primary point normals, mirrored for shader reads.
    const auto point_normals = attrs.Normals ? B->PointNormalBuffer.Allocate(std::span<const vec3>{*attrs.Normals}) : Range{};
    FillBaseVertexNormalMirror(vertices, point_normals);
    return AcquireId({.Vertices = vertices, .FaceData = {}, .PointNormals = point_normals, .Alive = true});
}

void MeshStore::PlanCreate(const MeshData &data, const MeshPrimitives &primitives, bool has_deform, uint32_t morph_target_count, const MeshVertexAttributes &attrs) {
    const uint32_t vertices = data.Positions.size();
    const uint32_t faces = data.FaceCount();
    const uint32_t triangles = uint32_t(data.FaceCorners.size()) - 2u * faces;
    const uint32_t edges = (triangles + 2 * faces + 1) / 2; // manifold estimate: edges ≈ halfedges / 2
    Pending.Vertices += vertices;
    Pending.Faces += faces;
    Pending.Triangles += triangles;
    Pending.FaceCorners += uint32_t(data.FaceCorners.size()) + uint32_t(data.Edges.size()) * 2u;
    Pending.Edges += edges;
    Pending.AdjacencyWords += 2 * (vertices + 1) + triangles * 3 + edges * 2;
    // Outgoing, opposites, the two bit tables and the samples, plus an n-gon mesh's face starts.
    const uint32_t halfedges = faces > 0 ? uint32_t(data.FaceCorners.size()) : uint32_t(data.Edges.size()) * 2u;
    Pending.ConnectivityWords += vertices + halfedges + 3 * BitWords(halfedges) + (halfedges == 3 * faces ? 0u : faces);
    Pending.Primitives += primitives.MaterialIndices.size();
    if (has_deform) Pending.BoneDeformVertices += vertices;
    if (morph_target_count > 0) Pending.MorphTargetEntries += morph_target_count * vertices;
    // Point and line meshes index their primitive per vertex, triangle meshes per face.
    Pending.ElementPrimitiveIndices += faces > 0 ? faces : uint32_t(primitives.ElementPrimitiveIndices.size());
    if (triangles > 0) {
        const uint32_t corners = triangles * 3;
        if (attrs.Tangents) Pending.CornerTangents += corners;
        if (attrs.Colors0) Pending.CornerColors += corners;
        for (const auto *uvs : {&attrs.TexCoords0, &attrs.TexCoords1, &attrs.TexCoords2, &attrs.TexCoords3}) {
            if (*uvs) Pending.CornerUvs += corners;
        }
    } else if (attrs.Colors0) {
        Pending.CornerColors += vertices; // Point and line meshes hold one color per vertex.
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
    Pending.ElementPrimitiveIndices += e.ElementPrimitives.Count;
    Pending.BoneDeformVertices += e.BoneDeform.Count;
    Pending.MorphTargetEntries += e.MorphTargets.Count;
    Pending.CornerTangents += e.CornerTangents.Count;
    Pending.CornerColors += e.CornerColors.Count;
    for (const auto &uvs : e.CornerUvs) Pending.CornerUvs += uvs.Count;
    Pending.AdjacencyWords += e.VertexFanAdjacency.Count + e.VertexEdgeAdjacency.Count + e.SeamFans.Count;
    Pending.ConnectivityWords += e.Connectivity.Count + e.ConnectivityEdges.Count + e.ConnectivityHalfedgeToEdge.Count;
}

void MeshStore::CommitReserves() {
    B->VerticesBuffer.ReserveAdditional(Pending.Vertices);
    B->FaceFirstTriangleBuffer.ReserveAdditional(Pending.Faces);
    B->ElementPrimitiveBuffer.ReserveAdditional(Pending.ElementPrimitiveIndices);
    B->TriangleFaceIdBuffer.ReserveAdditional(Pending.Triangles);
    B->CornerClassBuffer.ReserveAdditional(Pending.Triangles * 3);
    B->EdgeStateBuffer.ReserveAdditional(Pending.EdgeStates);
    B->FaceCornerBuffer.ReserveAdditional(Pending.FaceCorners);
    B->EdgeSharpnessBuffer.ReserveAdditional(Pending.Edges);
    B->PrimitiveMaterialBuffer.ReserveAdditional(Pending.Primitives);
    B->BoneDeformBuffer.ReserveAdditional(Pending.BoneDeformVertices);
    B->MorphTargetBuffer.ReserveAdditional(Pending.MorphTargetEntries);
    B->CornerTangentBuffer.ReserveAdditional(Pending.CornerTangents);
    B->CornerColorBuffer.ReserveAdditional(Pending.CornerColors);
    B->CornerUvBuffer.ReserveAdditional(Pending.CornerUvs);
    B->AdjacencyBuffer.ReserveAdditional(Pending.AdjacencyWords);
    B->ConnectivityBuffer.ReserveAdditional(Pending.ConnectivityWords);
    // Mirror buffers hold one byte per element with no arena, sharing ranges with the data arenas.
    if (Pending.Vertices > 0) B->VertexStateBuffer.Reserve(B->VertexStateBuffer.UsedSize + Pending.Vertices);
    if (Pending.Faces > 0) {
        B->FaceStateBuffer.Reserve(B->FaceStateBuffer.UsedSize + Pending.Faces);
        B->FaceSharpnessBuffer.Reserve(B->FaceSharpnessBuffer.UsedSize + Pending.Faces);
    }
    Pending = {};
}

namespace {
// Fill `out` with CSR incidence over `bucket_count` buckets: (bucket_count + 1) item-start offsets, then the items.
// `emit(add)` calls add(bucket, item) in a fixed order and runs twice (count, then scatter).
void WriteCsr(std::span<uint32_t> out, uint32_t bucket_count, auto &&emit) {
    const auto offsets = out.first(bucket_count + 1);
    const auto items = out.subspan(bucket_count + 1);
    std::ranges::fill(offsets, 0u);
    emit([&](uint32_t bucket, uint32_t) { ++offsets[bucket + 1]; });
    for (uint32_t b = 0; b < bucket_count; ++b) offsets[b + 1] += offsets[b];
    std::vector<uint32_t> cursors(offsets.begin(), offsets.end() - 1);
    emit([&](uint32_t bucket, uint32_t item) { items[cursors[bucket]++] = item; });
}
} // namespace

bool BuildsFanAdjacencyOnGpu(const Mesh &mesh) {
    const auto &c = mesh.GetConnectivity();
    return c.FaceCount > 0 && c.Faces.empty();
}

bool BuildsEdgeAdjacencyOnGpu(const Mesh &mesh) {
    return BuildsFanAdjacencyOnGpu(mesh) && mesh.GetConnectivity().HalfedgeToEdge.empty();
}

namespace {
// Call `add(vertex, item)` once per vertex-fan incidence, in the order the table records it.
// A face's halfedges are the contiguous run starting at its first, so a halfedge's index gives its
// loop position too.
void EmitFanIncidence(const Mesh &mesh, auto &&add) {
    const auto &c = mesh.GetConnectivity();
    const auto corners = mesh.CornerVertices();
    for (uint32_t face = 0; face < c.FaceCount; ++face) {
        const auto first = *c.FaceHalfedge(face), last = c.FaceEnd(face);
        for (auto h = first; h < last; ++h) add(corners[h], face | ((h - first) << FanLoopShift));
    }
}

// Call `add(vertex, item)` once per vertex-edge incidence, in the order the table records it.
void EmitEdgeIncidence(const Mesh &mesh, auto &&add) {
    const auto &c = mesh.GetConnectivity();
    const auto corners = mesh.CornerVertices();
    for (uint32_t edge = 0; edge < c.EdgeCount; ++edge) {
        const auto h = c.EdgeHalfedge(edge);
        const auto opposite = c.Opposites[*h];
        add(corners[*(opposite ? opposite : c.Previous(h))], edge);
        add(corners[*h], edge);
    }
}
} // namespace

void MeshStore::BuildVertexAdjacency(const Mesh &mesh) {
    const profile::CpuScope scope{"VertexAdjacency"};
    const auto id = mesh.GetStoreId();
    auto &entry = Entries.at(id);
    const uint32_t vertex_count = mesh.VertexCount();
    if (entry.TriangleCount > 0) {
        // Every halfedge of a face-topology mesh belongs to a face, so the face loops cover all of them.
        entry.VertexFanAdjacency = B->AdjacencyBuffer.Allocate(vertex_count + 1 + mesh.HalfEdgeCount());
        if (!BuildsFanAdjacencyOnGpu(mesh)) {
            WriteCsr(B->AdjacencyBuffer.GetMutable(entry.VertexFanAdjacency), vertex_count, [&](auto &&add) { EmitFanIncidence(mesh, add); });
        }
    }
    if (mesh.EdgeCount() > 0) {
        entry.VertexEdgeAdjacency = B->AdjacencyBuffer.Allocate(vertex_count + 1 + mesh.EdgeCount() * 2);
        if (!BuildsEdgeAdjacencyOnGpu(mesh)) {
            WriteCsr(B->AdjacencyBuffer.GetMutable(entry.VertexEdgeAdjacency), vertex_count, [&](auto &&add) { EmitEdgeIncidence(mesh, add); });
        }
    }
}

std::string MeshStore::CheckVertexAdjacency(const Mesh &mesh) const {
    const auto vertex_count = mesh.VertexCount();
    const auto check = [&](std::string_view name, Range range, auto &&emit) {
        if (range.Count == 0) return std::string{};
        std::vector<uint32_t> reference(range.Count);
        WriteCsr(reference, vertex_count, emit);
        const auto stored = B->AdjacencyBuffer.Get(range);
        for (uint32_t i = 0; i < range.Count; ++i) {
            if (reference[i] == stored[i]) continue;
            const bool offset = i <= vertex_count;
            return std::format(
                "mesh {} {} {} {} differs: {} against {}",
                mesh.GetStoreId(), name, offset ? "offset" : "item", offset ? i : i - vertex_count - 1, stored[i], reference[i]
            );
        }
        return std::string{};
    };
    const auto &entry = Entries.at(mesh.GetStoreId());
    if (auto fan = check("fan", entry.VertexFanAdjacency, [&](auto &&add) { EmitFanIncidence(mesh, add); }); !fan.empty()) return fan;
    return check("edge", entry.VertexEdgeAdjacency, [&](auto &&add) { EmitEdgeIncidence(mesh, add); });
}

PreparedMesh PrepareMeshSources(MeshData &data, MeshVertexAttributes &attrs, MeshPrimitives &primitives) {
    const uint32_t face_count = data.FaceCount();

    // Sort faces by primitive index so triangles are grouped by primitive in the index buffer.
    if (!primitives.ElementPrimitiveIndices.empty() && primitives.ElementPrimitiveIndices.size() == face_count &&
        !std::ranges::all_of(primitives.ElementPrimitiveIndices, [&](uint32_t pi) { return pi == primitives.ElementPrimitiveIndices[0]; })) {
        std::vector<uint32_t> perm(face_count);
        std::iota(perm.begin(), perm.end(), 0u);
        std::stable_sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) {
            return primitives.ElementPrimitiveIndices[a] < primitives.ElementPrimitiveIndices[b];
        });
        bool already_sorted = true;
        for (uint32_t i = 0; i < face_count; ++i) {
            if (perm[i] != i) {
                already_sorted = false;
                break;
            }
        }
        if (!already_sorted) {
            // A mesh of triangles keeps its offsets arithmetic, so only the corners permute.
            const bool spelled_offsets = !data.FaceOffsets.empty();
            std::vector<uint32_t> sorted_offsets, sorted_corners, sorted_fpi(face_count);
            if (spelled_offsets) {
                sorted_offsets.reserve(face_count + 1);
                sorted_offsets.emplace_back(0u);
            }
            sorted_corners.reserve(data.FaceCorners.size());
            for (uint32_t i = 0; i < face_count; ++i) {
                const auto face = data.Face(perm[i]);
                sorted_corners.insert(sorted_corners.end(), face.begin(), face.end());
                if (spelled_offsets) sorted_offsets.emplace_back(uint32_t(sorted_corners.size()));
                sorted_fpi[i] = primitives.ElementPrimitiveIndices[perm[i]];
            }
            data.FaceOffsets = std::move(sorted_offsets);
            data.FaceCorners = std::move(sorted_corners);
            primitives.ElementPrimitiveIndices = std::move(sorted_fpi);
        }
    }

    // Triangle-mesh tangent/color/UV channels are corner-domain: gathered into per-corner streams in fan order before welding rewrites the face indices.
    // The vertex buffer keeps defaults for these channels.
    std::vector<vec4> corner_tangents, corner_colors;
    std::array<std::vector<vec2>, 4> corner_uvs;
    std::vector<vec3> authored_corner_normals;
    if (face_count > 0) {
        const uint32_t corner_total = (uint32_t(data.FaceCorners.size()) - 2u * face_count) * 3u;
        const auto gather_corners = [&]<typename T>(std::optional<std::vector<T>> &src, std::vector<T> &out) {
            if (!src) return;
            out.reserve(corner_total);
            for (uint32_t fi = 0; fi < face_count; ++fi) {
                const auto face = data.Face(fi);
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
        // Authored normals recover from this stream: faceted faces fill the face-sharpness store and the remainder lands in the custom corner-normal layer.
        // Shading normals are derived.
        gather_corners(attrs.Normals, authored_corner_normals);
    }

    return {
        .CornerTangents = std::move(corner_tangents),
        .CornerColors = std::move(corner_colors),
        .CornerUvs = std::move(corner_uvs),
        .AuthoredCornerNormals = std::move(authored_corner_normals),
        .MorphTangentDeltas = {}, // Taken from the source once the arenas hold the rest of the vertex domain.
    };
}

BuiltConnectivity BuildPreparedConnectivity(const MeshStore &store, uint32_t id, const MeshData &data, const ConnectivityStorage &storage) {
    const uint32_t vertex_count = store.GetVerticesRange(id).Count;
    if (data.FaceCount() > 0) return BuildConnectivity(data.FaceOffsets, store.GetFaceCorners(id), vertex_count, storage);
    if (!data.Edges.empty()) return BuildConnectivity(data.Edges, vertex_count, storage);
    return {};
}

CreatedMesh MeshStore::CreateMesh(uint32_t id, MeshData &&data, MeshVertexAttributes &&attrs, MeshPrimitives &&primitives, PreparedMesh &&prepared, bool flat_shaded) {
    const profile::CpuScope scope{"CreateMesh"};
    const uint32_t face_count = data.FaceCount();
    auto &corner_tangents = prepared.CornerTangents;
    auto &corner_colors = prepared.CornerColors;
    auto &corner_uvs = prepared.CornerUvs;
    auto &authored_corner_normals = prepared.AuthoredCornerNormals;

    // CreateMeshSource and CreateDeformSource already took the positions, corners, skin and morph
    // channels into the arenas, and a weld has trimmed every one of those ranges to what it left.
    auto &entry = Entries[id];
    const auto vertices = entry.Vertices;
    // A face-less mesh keeps its authored normals as primary point normals, mirrored for shader reads.
    entry.PointNormals = attrs.Normals ? B->PointNormalBuffer.Allocate(std::span<const vec3>{*attrs.Normals}) : Range{};
    FillBaseVertexNormalMirror(vertices, entry.PointNormals);

    const auto write_primitive_tables = [&](uint32_t element_count, uint32_t primitive_count) {
        entry.ElementPrimitives = B->ElementPrimitiveBuffer.Allocate(element_count);
        auto fp_span = B->ElementPrimitiveBuffer.GetMutable(entry.ElementPrimitives);
        if (!primitives.ElementPrimitiveIndices.empty()) std::ranges::copy(primitives.ElementPrimitiveIndices, fp_span.begin());
        else std::ranges::fill(fp_span, 0u);

        entry.PrimitiveMaterials = B->PrimitiveMaterialBuffer.Allocate(primitive_count);
        auto pm_span = B->PrimitiveMaterialBuffer.GetMutable(entry.PrimitiveMaterials);
        if (!primitives.MaterialIndices.empty()) std::ranges::copy(primitives.MaterialIndices, pm_span.begin());
        else std::ranges::fill(pm_span, 0u);
    };

    if (face_count > 0) {
        entry.FaceData = AllocateFaces(face_count);
        auto first_tri_span = B->FaceFirstTriangleBuffer.GetMutable(entry.FaceData);
        uint32_t tri_offset = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            first_tri_span[fi] = tri_offset;
            tri_offset += data.FaceSize(fi) - 2u;
        }

        entry.TriangleCount = tri_offset;

        if (!corner_tangents.empty()) entry.CornerTangents = B->CornerTangentBuffer.Allocate(std::span<const vec4>{corner_tangents});
        if (!corner_colors.empty()) entry.CornerColors = B->CornerColorBuffer.Allocate(std::span<const vec4>{corner_colors});
        for (uint32_t set = 0; set < corner_uvs.size(); ++set) {
            if (!corner_uvs[set].empty()) entry.CornerUvs[set] = B->CornerUvBuffer.Allocate(std::span<const vec2>{corner_uvs[set]});
        }

        entry.TriangleFaceIds = B->TriangleFaceIdBuffer.Allocate(tri_offset);
        auto tri_face_span = B->TriangleFaceIdBuffer.GetMutable(entry.TriangleFaceIds);
        uint32_t ti = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            const auto n_tris = data.FaceSize(fi) - 2u;
            for (size_t t = 0; t < n_tris; ++t) tri_face_span[ti++] = fi + 1;
        }

        const auto primitive_count = !primitives.MaterialIndices.empty() ?
            (primitives.ElementPrimitiveIndices.empty() ? 1u : *std::ranges::max_element(primitives.ElementPrimitiveIndices) + 1u) :
            1u;
        write_primitive_tables(face_count, primitive_count);

        // Faces come sorted by primitive, so each primitive's triangles are one contiguous range.
        {
            const auto fp = B->ElementPrimitiveBuffer.Get(entry.ElementPrimitives);
            const auto fft = B->FaceFirstTriangleBuffer.Get(entry.FaceData);
            auto &ranges = entry.PrimitiveTriangleRanges;
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
    } else if (!primitives.ElementPrimitiveIndices.empty()) {
        // Point and line meshes carry one color and one primitive index per vertex.
        if (attrs.Colors0) entry.CornerColors = B->CornerColorBuffer.Allocate(std::span<const vec4>{*attrs.Colors0});
        // Primitive indices are source-wide, so the material table spans every primitive of the source mesh.
        const auto primitive_count = primitives.MaterialIndices.empty() ? 1u : uint32_t(primitives.MaterialIndices.size());
        write_primitive_tables(primitives.ElementPrimitiveIndices.size(), primitive_count);
    }

    // One corner vertex index per halfedge, canonical for the connectivity and for a triangle mesh's
    // draws. A face mesh's corners are its face loops, and an edge mesh's are each edge's endpoints.
    if (face_count == 0 && !data.Edges.empty()) {
        entry.FaceCorners = B->FaceCornerBuffer.Allocate(uint32_t(data.Edges.size()) * 2u);
        auto corners = B->FaceCornerBuffer.GetMutable(entry.FaceCorners);
        for (uint32_t e = 0; e < data.Edges.size(); ++e) {
            corners[e * 2u] = data.Edges[e][1];
            corners[e * 2u + 1u] = data.Edges[e][0];
        }
    }

    const Mesh mesh{*this, id};

    // A mesh with no faces always draws its edges, so it takes its edge states now. A face mesh
    // waits for a wireframe or edit overlay to ask.
    if (face_count == 0) entry.EdgeStates = B->EdgeStateBuffer.Allocate(mesh.EdgeCount() * 2);
    entry.EdgeSharpness = B->EdgeSharpnessBuffer.Allocate(mesh.EdgeCount());
    ClearElementStates(vertices, entry.FaceData, entry.EdgeStates);
    if (entry.FaceData.Count > 0) {
        auto sharpness = GetFaceSharpness(id);
        std::ranges::fill(sharpness, uint8_t(flat_shaded ? 1 : 0));
        // Faces of primitives that ship no normals shade flat, like a fully normal-less mesh.
        if (!flat_shaded && !primitives.AttributeFlags.empty()) {
            for (uint32_t fi = 0; fi < sharpness.size(); ++fi) {
                const auto pi = fi < primitives.ElementPrimitiveIndices.size() ? primitives.ElementPrimitiveIndices[fi] : 0u;
                if (pi < primitives.AttributeFlags.size() && !(primitives.AttributeFlags[pi] & MeshAttributeBit_Normal)) sharpness[fi] = 1;
            }
        }
    }
    if (entry.EdgeSharpness.Count > 0) std::ranges::fill(GetEdgeSharpness(id), uint8_t{0});
    BuildVertexAdjacency(mesh);

    // A face whose authored corner normals all match its geometric normal shades flat, recorded as face sharpness.
    if (!authored_corner_normals.empty() && entry.FaceData.Count > 0) {
        auto sharp = GetFaceSharpness(id);
        // The weld rewrote the arena's positions and corners, so the face loops read them there.
        const auto source_vertices = GetVertices(id);
        const auto corners = GetFaceCorners(id);
        uint32_t ci = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            const auto face = corners.subspan(data.FaceStart(fi), data.FaceSize(fi));
            const uint32_t corner_count = (face.size() - 2) * 3;
            const auto p0 = source_vertices[face[0]].Position;
            const auto cross = glm::cross(source_vertices[face[1]].Position - p0, source_vertices[face[2]].Position - p0);
            const auto cross_len = glm::length(cross);
            bool flat = cross_len > 0.f;
            if (flat) {
                const auto face_normal = cross / cross_len;
                for (uint32_t k = 0; k < corner_count; ++k) {
                    if (!NormalsMatch(authored_corner_normals[ci + k], face_normal).value_or(false)) {
                        flat = false;
                        break;
                    }
                }
            }
            if (flat) sharp[fi] = 1;
            ci += corner_count;
        }
    }

    // Sharp-edge inference: an interior edge whose authored corner normals disagree across it at either endpoint splits shading there.
    // The split records as edge sharpness so seam sectors derive.
    if (!authored_corner_normals.empty() && entry.FaceData.Count > 0 && entry.EdgeSharpness.Count > 0) {
        const auto first_triangles = GetFaceFirstTriangles(id);
        auto sharp_edges = GetEdgeSharpness(id);
        const auto &c = mesh.GetConnectivity();
        // The authored normal at face loop position `k`, read from any of its fan-corner slots.
        const auto authored_at = [&](Mesh::FH fh, uint32_t k) {
            const auto base = 3 * first_triangles[*fh];
            if (k == 0) return authored_corner_normals[base];
            const auto tri_count = mesh.GetValence(fh) - 2;
            return k - 1 < tri_count ? authored_corner_normals[base + 3 * (k - 1) + 1] : authored_corner_normals[base + 3 * (k - 2) + 2];
        };
        const auto vertex_position = [&](Mesh::FH fh, Mesh::VH vh) -> std::optional<uint32_t> {
            uint32_t k = 0;
            for (const auto hh : mesh.fh_range(fh)) {
                if (mesh.GetToVertex(hh) == vh) return k;
                ++k;
            }
            return {};
        };
        const auto discontinuous = [&](Mesh::FH fa, Mesh::FH fb, Mesh::VH vh) {
            const auto ka = vertex_position(fa, vh), kb = vertex_position(fb, vh);
            if (!ka || !kb) return false;
            const auto nb = authored_at(fb, *kb);
            const auto lb = glm::length(nb);
            if (lb < 1e-6f) return false;
            return NormalsMatch(authored_at(fa, *ka), nb / lb) == false;
        };
        for (uint32_t ei = 0; ei < mesh.EdgeCount(); ++ei) {
            const auto hh = mesh.GetHalfedge(Mesh::EH{ei}, 0);
            const auto face = mesh.GetFace(hh);
            const auto opposite = c.Opposites[*hh];
            const auto opposite_face = opposite ? c.FaceOf(opposite) : Mesh::FH{};
            if (!face || !opposite_face) continue;
            if (discontinuous(face, opposite_face, mesh.GetFromVertex(hh)) || discontinuous(face, opposite_face, mesh.GetToVertex(hh))) {
                sharp_edges[ei] = 1;
            }
        }
    }

    UpdateCornerClassification(mesh);

    // Authored corner normals the sharpness stores can't reproduce land in the custom layer, encoded as offsets from the derived normal so they follow every deformation of the surface.
    // The encode needs the derived base normals, so the stream stashes here until EncodeAuthoredCornerNormals consumes it.
    if (!authored_corner_normals.empty() && entry.TriangleCount > 0) {
        entry.HasAuthoredNormals = true;
        entry.AuthoredCornerNormals = std::move(authored_corner_normals);
    }

    return {id, std::move(prepared.MorphTangentDeltas)};
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
        .CornerClasses = B->CornerClassBuffer.Clone(src_entry.CornerClasses),
        .CustomCornerMasks = B->CustomCornerMaskBuffer.Clone(src_entry.CustomCornerMasks),
        .CustomCornerNormals = B->CustomCornerNormalBuffer.Clone(src_entry.CustomCornerNormals),
        .CornerTangents = B->CornerTangentBuffer.Clone(src_entry.CornerTangents),
        .CornerColors = B->CornerColorBuffer.Clone(src_entry.CornerColors),
        .CornerUvs = {B->CornerUvBuffer.Clone(src_entry.CornerUvs[0]), B->CornerUvBuffer.Clone(src_entry.CornerUvs[1]), B->CornerUvBuffer.Clone(src_entry.CornerUvs[2]), B->CornerUvBuffer.Clone(src_entry.CornerUvs[3])},
        .EdgeSharpness = B->EdgeSharpnessBuffer.Clone(src_entry.EdgeSharpness),
        .EdgeStates = edge_states,
        .TriangleFaceIds = B->TriangleFaceIdBuffer.Clone(src_entry.TriangleFaceIds),
        .ElementPrimitives = B->ElementPrimitiveBuffer.Clone(src_entry.ElementPrimitives),
        .PrimitiveMaterials = B->PrimitiveMaterialBuffer.Clone(src_entry.PrimitiveMaterials),
        .VertexFanAdjacency = B->AdjacencyBuffer.Clone(src_entry.VertexFanAdjacency),
        .VertexEdgeAdjacency = B->AdjacencyBuffer.Clone(src_entry.VertexEdgeAdjacency),
        .Connectivity = B->ConnectivityBuffer.Clone(src_entry.Connectivity),
        .ConnectivityEdges = B->ConnectivityBuffer.Clone(src_entry.ConnectivityEdges),
        .ConnectivityHalfedgeToEdge = B->ConnectivityBuffer.Clone(src_entry.ConnectivityHalfedgeToEdge),
        .ConnectivityVertices = src_entry.ConnectivityVertices,
        .ConnectivityHalfedges = src_entry.ConnectivityHalfedges,
        .ConnectivityEdgeCount = src_entry.ConnectivityEdgeCount,
        .ConnectivityFaces = src_entry.ConnectivityFaces,
        .ConnectivityFaceStarts = src_entry.ConnectivityFaceStarts,
        .SeamFans = B->AdjacencyBuffer.Clone(src_entry.SeamFans),
        .BaseSeamNormals = B->BaseSeamNormalBuffer.Clone(src_entry.BaseSeamNormals),
        .PointNormals = B->PointNormalBuffer.Clone(src_entry.PointNormals),
        .BoneDeform = B->BoneDeformBuffer.Clone(src_entry.BoneDeform),
        .MorphTargets = B->MorphTargetBuffer.Clone(src_entry.MorphTargets),
        .SeamCornerCount = src_entry.SeamCornerCount,
        .MorphTargetCount = src_entry.MorphTargetCount,
        .TriangleCount = src_entry.TriangleCount,
        .UniformCornerClass = src_entry.UniformCornerClass,
        .HasAuthoredNormals = src_entry.HasAuthoredNormals,
        .MorphShadingAuthored = src_entry.MorphShadingAuthored,
        .DefaultMorphWeights = src_entry.DefaultMorphWeights,
        .PrimitiveTriangleRanges = src_entry.PrimitiveTriangleRanges,
        .Alive = true,
    });

    ClearElementStates(vertices, faces, edge_states);
    if (faces.Count > 0) std::ranges::copy(GetFaceSharpness(src_id), GetFaceSharpness(id).begin());
    std::ranges::copy(GetBaseVertexNormals(src_id), GetBaseVertexNormals(id).begin());
    if (faces.Count > 0) std::ranges::copy(GetBaseFaceNormals(src_id), GetBaseFaceNormals(id).begin());
    return {id};
}

std::expected<MeshDataWithMaterials, std::string> ReadMeshFile(const std::filesystem::path &path) {
    auto ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
    try {
        if (ext == ".ply") return ReadPly(path);
        if (ext == ".obj") return ReadObj(path);
    } catch (const std::exception &e) {
        return std::unexpected{e.what()};
    }
    return std::unexpected{"Unsupported file format: " + ext};
}

void MeshStore::Release(uint32_t id) {
    if (id >= Entries.size() || !Entries[id].Alive) return;
    auto &entry = Entries[id];
    B->VerticesBuffer.Release(entry.Vertices);
    B->CornerClassBuffer.Release(entry.CornerClasses);
    B->BaseSeamNormalBuffer.Release(entry.BaseSeamNormals);
    B->CustomCornerMaskBuffer.Release(entry.CustomCornerMasks);
    B->CustomCornerNormalBuffer.Release(entry.CustomCornerNormals);
    B->CornerTangentBuffer.Release(entry.CornerTangents);
    B->CornerColorBuffer.Release(entry.CornerColors);
    for (const auto &uvs : entry.CornerUvs) B->CornerUvBuffer.Release(uvs);
    B->EdgeSharpnessBuffer.Release(entry.EdgeSharpness);
    B->TriangleFaceIdBuffer.Release(entry.TriangleFaceIds);
    B->FaceFirstTriangleBuffer.Release(entry.FaceData);
    B->ElementPrimitiveBuffer.Release(entry.ElementPrimitives);
    B->PrimitiveMaterialBuffer.Release(entry.PrimitiveMaterials);
    B->EdgeStateBuffer.Release(entry.EdgeStates);
    B->FaceCornerBuffer.Release(entry.FaceCorners);
    B->ConnectivityBuffer.Release(entry.Connectivity);
    B->ConnectivityBuffer.Release(entry.ConnectivityEdges);
    B->ConnectivityBuffer.Release(entry.ConnectivityHalfedgeToEdge);
    B->AdjacencyBuffer.Release(entry.VertexFanAdjacency);
    B->AdjacencyBuffer.Release(entry.VertexEdgeAdjacency);
    B->AdjacencyBuffer.Release(entry.SeamFans);
    B->PointNormalBuffer.Release(entry.PointNormals);
    B->BoneDeformBuffer.Release(entry.BoneDeform);
    B->MorphTargetBuffer.Release(entry.MorphTargets);
    entry = {};
    FreeIds.emplace_back(id);
}

void MeshStore::Clear() {
    B->ForEachSerializedArena([](auto &a) { a.Reset(); });
    B->ForEachDerivedArena([](auto &a) { a.Reset(); });
    B->VertexStateBuffer.UsedSize = 0;
    B->FaceStateBuffer.UsedSize = 0;
    B->FaceSharpnessBuffer.UsedSize = 0;
    B->BaseVertexNormalBuffer.UsedSize = 0;
    B->BaseFaceNormalBuffer.UsedSize = 0;
    Entries.clear();
    FreeIds.clear();
    Pending = {};
}

Range MeshStore::AllocateVertices(uint32_t count) {
    const auto range = B->VerticesBuffer.Allocate(count);
    SyncMirror<uint8_t>(B->VertexStateBuffer, range);
    SyncMirror<vec3>(B->BaseVertexNormalBuffer, range);
    return range;
}

Range MeshStore::AllocateFaces(uint32_t count) {
    const auto range = B->FaceFirstTriangleBuffer.Allocate(count);
    SyncMirror<uint8_t>(B->FaceStateBuffer, range);
    SyncMirror<uint8_t>(B->FaceSharpnessBuffer, range);
    SyncMirror<vec3>(B->BaseFaceNormalBuffer, range);
    return range;
}

std::span<const uint8_t> MeshStore::GetFaceSharpness(uint32_t id) const { return B->FaceSharpnessBuffer.GetSpan<uint8_t>(Entries.at(id).FaceData); }
std::span<uint8_t> MeshStore::GetFaceSharpness(uint32_t id) { return B->FaceSharpnessBuffer.GetMutableSpan<uint8_t>(Entries.at(id).FaceData); }
std::span<const uint8_t> MeshStore::GetEdgeSharpness(uint32_t id) const { return B->EdgeSharpnessBuffer.Get(Entries.at(id).EdgeSharpness); }
std::span<uint8_t> MeshStore::GetEdgeSharpness(uint32_t id) { return B->EdgeSharpnessBuffer.GetMutable(Entries.at(id).EdgeSharpness); }

VertexAdjacency MeshStore::GetVertexEdgeAdjacency(uint32_t id) const {
    const auto &e = Entries.at(id);
    return SliceAdjacency(B->AdjacencyBuffer.Get(e.VertexEdgeAdjacency), e.Vertices.Count);
}

std::span<uint8_t> MeshStore::GetFaceStates(Range range) { return B->FaceStateBuffer.GetMutableSpan<uint8_t>(range); }
std::span<uint8_t> MeshStore::GetVertexStates(Range range) { return B->VertexStateBuffer.GetMutableSpan<uint8_t>(range); }
std::span<const uint8_t> MeshStore::GetVertexStates(Range range) const { return B->VertexStateBuffer.GetSpan<uint8_t>(range); }
std::span<const uint8_t> MeshStore::GetVertexStates(uint32_t id) const { return GetVertexStates(Entries.at(id).Vertices); }

void MeshStore::ClearElementStates(Range vertices, Range faces, Range edges) {
    std::ranges::fill(GetVertexStates(vertices), 0);
    if (faces.Count > 0) std::ranges::fill(GetFaceStates(faces), 0);
    if (edges.Count > 0) std::ranges::fill(B->EdgeStateBuffer.GetMutable(edges), 0);
}

void MeshStore::ClearElementStates(const Mesh &mesh) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    ClearElementStates(entry.Vertices, entry.FaceData, entry.EdgeStates);
}

using namespace he;


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

void MeshStore::UpdateSoundVertexStates(const Mesh &mesh, std::span<const uint32_t> vertices) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    ClearElementStates(entry.Vertices, entry.FaceData, entry.EdgeStates);
    auto vertex_states = GetVertexStates(entry.Vertices);
    for (const auto v : vertices) {
        if (v < vertex_states.size()) vertex_states[v] = ElementStateSelected;
    }
    UpdateEdgeStatesFromVertices(mesh);
}

void MeshStore::UpdateEdgeStatesFromVertices(const Mesh &mesh) {
    const auto &entry = Entries.at(mesh.GetStoreId());
    auto edge_states = B->EdgeStateBuffer.GetMutable(entry.EdgeStates);
    if (edge_states.empty()) return;
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
    if (edge_states.empty()) return;
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
