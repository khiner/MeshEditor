#pragma once

#include "ArmatureDeformData.h"
#include "Mesh.h"
#include "MorphTargetData.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/MorphTargetVertex.h"
#include "vulkan/BufferArena.h"

#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>

struct ObjPlyMaterial {
    vec4 BaseColorFactor;
    float MetallicFactor, RoughnessFactor;
    std::string Name;

    // OBJ fields
    std::optional<std::filesystem::path> BaseColorTexturePath{}, NormalTexturePath{};
    bool HasAlphaTexture{false};
};

struct MeshWithMaterials {
    Mesh Mesh;
    std::vector<ObjPlyMaterial> Materials;
};

struct MeshData;
struct MeshVertexAttributes;
struct MeshPrimitives;

// Owns mesh vertex data (canonical CPU/GPU storage) used by all systems, including rendering.
struct MeshStore {
    explicit MeshStore(mvk::BufferContext &);

    Mesh CreateMesh(MeshData &&, MeshVertexAttributes &&, MeshPrimitives &&, std::optional<ArmatureDeformData> = {}, std::optional<MorphTargetData> = {});
    Mesh CloneMesh(const Mesh &);
    std::expected<MeshWithMaterials, std::string> LoadMesh(const std::filesystem::path &);

    // Allocate vertex-only store entry (no topology, no face/edge/primitive/material buffers).
    // Returns {storeId, vertexRange}. Release via Release(storeId).
    std::pair<uint32_t, Range> AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &attrs);

    void SetPositions(const Mesh &, std::span<const vec3>);
    void SetPositions(uint32_t store_id, std::span<const vec3>); // Position-only, no normals. For topology-free entries.
    void SetPosition(const Mesh &, uint32_t index, vec3 position); // Single vertex, no normal update

    std::span<const Vertex> GetVertices(uint32_t id) const { return VerticesBuffer.Get(Entries.at(id).Vertices); }
    std::span<Vertex> GetVertices(uint32_t id) { return VerticesBuffer.GetMutable(Entries.at(id).Vertices); }
    SlottedRange GetVerticesRange(uint32_t id) const { return {Entries.at(id).Vertices, VerticesBuffer.Buffer.Slot}; }

    SlottedRange GetBoneDeformRange(uint32_t id) const {
        const auto &entry = Entries.at(id);
        if (entry.BoneDeform.Count == 0) return {};
        return {entry.BoneDeform, BoneDeformBuffer.Buffer.Slot};
    }

    SlottedRange GetMorphTargetRange(uint32_t id) const {
        const auto &entry = Entries.at(id);
        if (entry.MorphTargets.Count == 0) return {};
        return {entry.MorphTargets, MorphTargetBuffer.Buffer.Slot};
    }
    uint32_t GetMorphTargetCount(uint32_t id) const { return Entries.at(id).MorphTargetCount; }
    std::span<const float> GetDefaultMorphWeights(uint32_t id) const { return Entries.at(id).DefaultMorphWeights; }

    uint32_t GetVertexStateSlot() const { return VertexStateBuffer.Slot; }
    std::span<const uint8_t> GetVertexStates(uint32_t id) const;
    SlottedRange GetFaceStateRange(uint32_t id) const { return {Entries.at(id).FaceData, FaceStateBuffer.Slot}; }
    SlottedRange GetEdgeStateRange(uint32_t id) const { return {Entries.at(id).EdgeStates, EdgeStateBuffer.Buffer.Slot}; }
    SlottedRange GetFaceFirstTriRange(uint32_t id) const { return {Entries.at(id).FaceData, FaceFirstTriangleBuffer.Buffer.Slot}; }

    SlottedRange GetFaceIdRange(uint32_t id) const { return {Entries.at(id).TriangleFaceIds, TriangleFaceIdBuffer.Buffer.Slot}; }
    SlottedRange GetFacePrimitiveRange(uint32_t id) const { return {Entries.at(id).FacePrimitives, FacePrimitiveBuffer.Buffer.Slot}; }
    SlottedRange GetPrimitiveMaterialRange(uint32_t id) const { return {Entries.at(id).PrimitiveMaterials, PrimitiveMaterialBuffer.Buffer.Slot}; }

    std::span<const uint32_t> GetTriangleFaceIds(uint32_t id) const { return TriangleFaceIdBuffer.Get(Entries.at(id).TriangleFaceIds); }
    std::span<const uint32_t> GetFacePrimitiveIndices(uint32_t id) const { return FacePrimitiveBuffer.Get(Entries.at(id).FacePrimitives); }
    std::span<uint32_t> GetFacePrimitiveIndices(uint32_t id) { return FacePrimitiveBuffer.GetMutable(Entries.at(id).FacePrimitives); }
    std::span<const uint32_t> GetPrimitiveMaterialIndices(uint32_t id) const { return PrimitiveMaterialBuffer.Get(Entries.at(id).PrimitiveMaterials); }
    std::span<uint32_t> GetPrimitiveMaterialIndices(uint32_t id) { return PrimitiveMaterialBuffer.GetMutable(Entries.at(id).PrimitiveMaterials); }

    void UpdateElementStates(
        const Mesh &, Element,
        const std::unordered_set<he::VH> &selected_vertices,
        const std::unordered_set<he::EH> &selected_edges,
        const std::unordered_set<he::EH> &active_edges,
        const std::unordered_set<he::FH> &selected_faces,
        std::optional<uint32_t> active_handle
    );
    void UpdateEdgeStatesFromFaces(const Mesh &, std::span<const uint32_t> selected_faces, std::optional<uint32_t> active_face);
    void UpdateEdgeStatesFromVertices(const Mesh &);
    void UpdateFaceStatesFromVertices(const Mesh &);
    void UpdateFaceStatesFromEdges(const Mesh &);
    // Writes vertex state buffer from non-vertex element handles (Face/Edge), for the GPU transform preview shader.
    void UpdateVertexStatesFromElements(const Mesh &, std::span<const uint32_t> handles, Element, std::optional<uint32_t> active_handle = {});
    void UpdateNormals(const Mesh &, bool skip_nonzero = false);

    void Release(uint32_t id);

    BufferArena<uint32_t> FaceFirstTriangleBuffer; // Per-face index of first triangle in the index buffer
    BufferArena<uint32_t> FacePrimitiveBuffer; // Per-face source primitive index
    BufferArena<uint32_t> PrimitiveMaterialBuffer; // Primitive index -> material index
    BufferArena<BoneDeformVertex> BoneDeformBuffer;
    BufferArena<MorphTargetVertex> MorphTargetBuffer;

private:
    BufferArena<Vertex> VerticesBuffer;

    mvk::Buffer VertexStateBuffer; // Mirrors VerticesBuffer
    mvk::Buffer FaceStateBuffer; // Mirrors FaceFirstTriangleBuffer
    BufferArena<uint8_t> EdgeStateBuffer;

    BufferArena<uint32_t> TriangleFaceIdBuffer; // 1-indexed map from face triangles (in mesh face order) to source face ID

    struct Entry {
        Range Vertices;
        Range FaceData; // Per-face range shared by FaceFirstTriangleBuffer and FaceStateBuffer
        Range EdgeStates{}, TriangleFaceIds{}, FacePrimitives{}, PrimitiveMaterials{};
        Range BoneDeform{}, MorphTargets{};
        uint32_t MorphTargetCount{0};
        std::vector<float> DefaultMorphWeights{};
        bool Alive{false};
    };

    std::vector<Entry> Entries{};
    std::vector<uint32_t> FreeIds{};

    uint32_t AcquireId(Entry &&);
    Range AllocateVertices(uint32_t count);
    Range AllocateFaces(uint32_t count);
    std::span<uint8_t> GetFaceStates(Range);
    std::span<uint8_t> GetVertexStates(Range);
    std::span<const uint8_t> GetVertexStates(Range) const;
};
