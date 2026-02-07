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

    Mesh CreateMesh(MeshData &&);
    Mesh CloneMesh(const Mesh &);
    std::expected<Mesh, std::string> LoadMesh(const std::filesystem::path &);

    void SetPositions(const Mesh &, std::span<const vec3>);
    void SetPosition(const Mesh &, uint32_t index, vec3 position); // Single vertex, no normal update

    std::span<const Vertex> GetVertices(uint32_t id) const { return VerticesBuffer.Get(Entries.at(id).Vertices); }
    std::span<Vertex> GetVertices(uint32_t id) { return VerticesBuffer.GetMutable(Entries.at(id).Vertices); }
    SlottedBufferRange GetVerticesRange(uint32_t id) const { return {Entries.at(id).Vertices, VerticesBuffer.Buffer.Slot}; }

    uint32_t GetVertexStateSlot() const { return VertexStateBuffer.Slot; }
    SlottedBufferRange GetFaceStateRange(uint32_t id) const { return {Entries.at(id).FaceNormals, FaceStateBuffer.Slot}; }
    SlottedBufferRange GetEdgeStateRange(uint32_t id) const { return {Entries.at(id).EdgeStates, EdgeStateBuffer.Buffer.Slot}; }

    std::span<const vec3> GetFaceNormals(uint32_t id) const { return FaceNormalBuffer.Get(Entries.at(id).FaceNormals); }
    std::span<vec3> GetFaceNormals(uint32_t id) { return FaceNormalBuffer.GetMutable(Entries.at(id).FaceNormals); }
    SlottedBufferRange GetFaceNormalRange(uint32_t id) const { return {Entries.at(id).FaceNormals, FaceNormalBuffer.Buffer.Slot}; }

    SlottedBufferRange GetFaceIdRange(uint32_t id) const { return {Entries.at(id).TriangleFaceIds, TriangleFaceIdBuffer.Buffer.Slot}; }

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

    struct Entry {
        BufferRange Vertices, FaceNormals;
        BufferRange EdgeStates;
        BufferRange TriangleFaceIds;
        bool Alive{false};
    };

    std::vector<Entry> Entries{};
    std::vector<uint32_t> FreeIds{};

    uint32_t AcquireId(Entry &&);
    BufferRange AllocateVertices(uint32_t count);
    BufferRange AllocateFaces(uint32_t count);
    std::span<uint8_t> GetFaceStates(BufferRange);
    std::span<uint8_t> GetVertexStates(BufferRange);
};
