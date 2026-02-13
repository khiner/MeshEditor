#pragma once

#include "../vulkan/BufferArena.h"
#include "ArmatureDeformData.h"
#include "Mesh.h"
#include "MorphTargetData.h"
#include "gpu/BoneDeformVertex.h"
#include "gpu/MorphTargetVertex.h"

#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>

struct MeshData;

// Owns mesh vertex data (canonical CPU/GPU storage) used by all systems, including rendering.
struct MeshStore {
    explicit MeshStore(mvk::BufferContext &);

    Mesh CreateMesh(MeshData &&, std::optional<ArmatureDeformData> = {}, std::optional<MorphTargetData> = {});
    Mesh CloneMesh(const Mesh &);
    std::expected<Mesh, std::string> LoadMesh(const std::filesystem::path &);

    void SetPositions(const Mesh &, std::span<const vec3>);
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
    SlottedRange GetFaceStateRange(uint32_t id) const { return {Entries.at(id).FaceNormals, FaceStateBuffer.Slot}; }
    SlottedRange GetEdgeStateRange(uint32_t id) const { return {Entries.at(id).EdgeStates, EdgeStateBuffer.Buffer.Slot}; }

    std::span<const vec3> GetFaceNormals(uint32_t id) const { return FaceNormalBuffer.Get(Entries.at(id).FaceNormals); }
    std::span<vec3> GetFaceNormals(uint32_t id) { return FaceNormalBuffer.GetMutable(Entries.at(id).FaceNormals); }
    SlottedRange GetFaceNormalRange(uint32_t id) const { return {Entries.at(id).FaceNormals, FaceNormalBuffer.Buffer.Slot}; }

    SlottedRange GetFaceIdRange(uint32_t id) const { return {Entries.at(id).TriangleFaceIds, TriangleFaceIdBuffer.Buffer.Slot}; }

    void UpdateElementStates(
        const Mesh &mesh,
        he::Element element,
        const std::unordered_set<he::VH> &selected_vertices,
        const std::unordered_set<he::EH> &selected_edges,
        const std::unordered_set<he::EH> &active_edges,
        const std::unordered_set<he::FH> &selected_faces,
        std::optional<uint32_t> active_handle
    );
    void UpdateNormals(const Mesh &);

    void Release(uint32_t id);

private:
    BufferArena<Vertex> VerticesBuffer;
    BufferArena<vec3> FaceNormalBuffer;

    mvk::Buffer VertexStateBuffer; // Mirrors VerticesBuffer
    mvk::Buffer FaceStateBuffer; // Mirrors FaceNormalBuffer
    BufferArena<uint8_t> EdgeStateBuffer;

    BufferArena<uint32_t> TriangleFaceIdBuffer; // 1-indexed map from face triangles (in mesh face order) to source face ID
    BufferArena<BoneDeformVertex> BoneDeformBuffer;
    BufferArena<MorphTargetVertex> MorphTargetBuffer;

    struct Entry {
        Range Vertices, FaceNormals;
        Range EdgeStates;
        Range TriangleFaceIds;
        Range BoneDeform;
        Range MorphTargets;
        uint32_t MorphTargetCount{0};
        std::vector<float> DefaultMorphWeights;
        bool Alive{false};
    };

    std::vector<Entry> Entries{};
    std::vector<uint32_t> FreeIds{};

    uint32_t AcquireId(Entry &&);
    Range AllocateVertices(uint32_t count);
    Range AllocateFaces(uint32_t count);
    std::span<uint8_t> GetFaceStates(Range);
    std::span<uint8_t> GetVertexStates(Range);
};
