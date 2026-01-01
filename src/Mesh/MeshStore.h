#pragma once

#include "Mesh.h"
#include "MeshData.h"

#include "../Vulkan/Megabuffer.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>

struct MeshStore {
    MeshStore() = default;
    explicit MeshStore(mvk::BufferContext &ctx) { Init(ctx); }

    void Init(mvk::BufferContext &ctx);

    using Range = BufferRange;
    struct Entry {
        Range Vertices;
        Range FaceIds;
        Range FaceNormals;
        bool Alive{false};
    };

    Mesh CreateMesh(MeshData &&);
    std::optional<Mesh> LoadMesh(const std::filesystem::path &);

    std::span<const Vertex3D> GetVertices(uint32_t id) const;
    std::span<Vertex3D> GetVerticesMutable(uint32_t id);
    void UpdateVertices(uint32_t id, std::span<const Vertex3D> vertices);
    void FlushVertices(uint32_t id);
    std::span<const vec3> GetFaceNormals(uint32_t id) const;
    std::span<vec3> GetFaceNormalsMutable(uint32_t id);
    void FlushFaceNormals(uint32_t id);
    Range GetVerticesRange(uint32_t id) const;
    uint32_t GetVerticesSlot() const;
    Range GetFaceIdRange(uint32_t id) const;
    Range GetFaceNormalRange(uint32_t id) const;
    uint32_t GetFaceIdSlot() const;
    uint32_t GetFaceNormalSlot() const;
    void Release(uint32_t id);

private:
    static constexpr uint32_t InvalidStoreId{~0u};

    uint32_t AcquireId();
    std::vector<Entry> Entries;
    std::vector<uint32_t> FreeIds;
    std::unique_ptr<Megabuffer<Vertex3D>> VerticesBuffer;
    std::unique_ptr<Megabuffer<uint32_t>> FaceIdBuffer;
    std::unique_ptr<Megabuffer<vec3>> FaceNormalBuffer;
};
