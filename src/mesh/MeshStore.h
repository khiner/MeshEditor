#pragma once

#include "../vulkan/BufferArena.h"
#include "Mesh.h"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <unordered_set>

struct MeshData;

// Owns mesh vertex data (canonical CPU/GPU storage) used by all systems, including rendering.
struct MeshStore {
    explicit MeshStore(mvk::BufferContext &);

    struct Entry {
        BufferRange Vertices, FaceIds, FaceNormals;
        BufferRange FaceStates, EdgeStates;
        bool Alive{false};
    };

    Mesh CreateMesh(MeshData &&);
    Mesh CloneMesh(const Mesh &);
    std::expected<Mesh, std::string> LoadMesh(const std::filesystem::path &);

    void SetPositions(const Mesh &, std::span<const vec3>);
    void SetPosition(const Mesh &, uint32_t index, vec3 position); // Single vertex, no normal update

    std::span<const Vertex> GetVertices(uint32_t id) const { return VerticesBuffer.Get(Entries.at(id).Vertices); }
    std::span<Vertex> GetVertices(uint32_t id) { return VerticesBuffer.GetMutable(Entries.at(id).Vertices); }
    SlottedBufferRange GetVerticesBuffer(uint32_t id) const { return {Entries.at(id).Vertices, VerticesBuffer.Buffer.Slot}; }

    SlottedBufferRange GetVertexStateBuffer(uint32_t id) const { return {Entries.at(id).Vertices, VertexStateBuffer.Slot}; }
    uint32_t GetVertexStateSlot() const { return VertexStateBuffer.Slot; }
    SlottedBufferRange GetFaceStateBuffer(uint32_t id) const { return {Entries.at(id).FaceStates, FaceStateBuffer.Buffer.Slot}; }
    SlottedBufferRange GetEdgeStateBuffer(uint32_t id) const { return {Entries.at(id).EdgeStates, EdgeStateBuffer.Buffer.Slot}; }

    std::span<const vec3> GetFaceNormals(uint32_t id) const { return FaceNormalBuffer.Get(Entries.at(id).FaceNormals); }
    std::span<vec3> GetFaceNormals(uint32_t id) { return FaceNormalBuffer.GetMutable(Entries.at(id).FaceNormals); }
    SlottedBufferRange GetFaceNormalBuffer(uint32_t id) const { return {Entries.at(id).FaceNormals, FaceNormalBuffer.Buffer.Slot}; }

    SlottedBufferRange GetFaceIdBuffer(uint32_t id) const { return {Entries.at(id).FaceIds, FaceIdBuffer.Buffer.Slot}; }

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
    uint32_t AcquireId();

    std::vector<Entry> Entries;
    std::vector<uint32_t> FreeIds;
    BufferArena<Vertex> VerticesBuffer;
    mvk::Buffer VertexStateBuffer;
    BufferArena<uint8_t> FaceStateBuffer;
    BufferArena<uint8_t> EdgeStateBuffer;
    BufferArena<uint32_t> FaceIdBuffer;
    BufferArena<vec3> FaceNormalBuffer;

    void EnsureVertexStateCapacity(BufferRange);
    std::span<uint8_t> GetVertexStates(BufferRange);
};
