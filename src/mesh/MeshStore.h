#pragma once

#include "../vulkan/BufferArena.h"
#include "Mesh.h"

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

    std::span<const Vertex3D> GetVertices(uint32_t id) const;
    std::span<Vertex3D> GetVertices(uint32_t id);
    BufferRange GetVerticesRange(uint32_t id) const;
    uint32_t GetVerticesSlot() const;

    std::span<const vec3> GetFaceNormals(uint32_t id) const;
    std::span<vec3> GetFaceNormals(uint32_t id);
    BufferRange GetFaceNormalRange(uint32_t id) const;
    uint32_t GetFaceNormalSlot() const;

    BufferRange GetFaceIdRange(uint32_t id) const;
    uint32_t GetFaceIdSlot() const;

    void Release(uint32_t id);

private:
    static constexpr uint32_t InvalidStoreId{~0u};

    uint32_t AcquireId();

    void UpdateNormals(const Mesh &);

    std::vector<Entry> Entries;
    std::vector<uint32_t> FreeIds;
    std::unique_ptr<BufferArena<Vertex3D>> VerticesBuffer;
    std::unique_ptr<BufferArena<uint32_t>> FaceIdBuffer;
    std::unique_ptr<BufferArena<vec3>> FaceNormalBuffer;
};
