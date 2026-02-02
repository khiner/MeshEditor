#pragma once

#include "../vulkan/BufferArena.h"
#include "Mesh.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

struct MeshData;

// Owns mesh vertex data (canonical CPU/GPU storage) used by all systems, including rendering.
struct MeshStore {
    explicit MeshStore(mvk::BufferContext &);

    struct Entry {
        BufferRange Vertices, FaceIds, FaceNormals;
        bool Alive{false};
    };

    Mesh CreateMesh(MeshData &&);
    Mesh CloneMesh(const Mesh &);
    std::optional<Mesh> LoadMesh(const std::filesystem::path &);

    void SetPositions(const Mesh &, std::span<const vec3>);
    void SetPosition(const Mesh &, uint32_t index, vec3 position); // Single vertex, no normal update

    std::span<const Vertex> GetVertices(uint32_t id) const;
    std::span<Vertex> GetVertices(uint32_t id);
    BufferRange GetVerticesRange(uint32_t id) const;
    uint32_t GetVerticesSlot() const;
    std::span<const uint8_t> GetVertexStates(uint32_t id) const;
    std::span<uint8_t> GetVertexStates(uint32_t id);
    BufferRange GetVertexStateRange(uint32_t id) const;
    uint32_t GetVertexStateSlot() const;

    std::span<const vec3> GetFaceNormals(uint32_t id) const;
    std::span<vec3> GetFaceNormals(uint32_t id);
    BufferRange GetFaceNormalRange(uint32_t id) const;
    uint32_t GetFaceNormalSlot() const;

    BufferRange GetFaceIdRange(uint32_t id) const;
    uint32_t GetFaceIdSlot() const;

    void UpdateNormals(const Mesh &);

    void Release(uint32_t id);

private:
    static constexpr uint32_t InvalidStoreId{~0u};

    uint32_t AcquireId();

    std::vector<Entry> Entries;
    std::vector<uint32_t> FreeIds;
    std::unique_ptr<BufferArena<Vertex>> VerticesBuffer;
    mvk::Buffer VertexStateBuffer;
    std::unique_ptr<BufferArena<uint32_t>> FaceIdBuffer;
    std::unique_ptr<BufferArena<vec3>> FaceNormalBuffer;

    void EnsureVertexStateCapacity(BufferRange);
    std::span<const uint8_t> GetVertexStates(BufferRange) const;
    std::span<uint8_t> GetVertexStates(BufferRange);
};
