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
        bool Alive{false};
    };

    Mesh CreateMesh(MeshData &&);
    std::optional<Mesh> LoadMesh(const std::filesystem::path &);

    std::span<const Vertex3D> GetVertices(uint32_t id) const;
    std::span<Vertex3D> GetVerticesMutable(uint32_t id);
    void UpdateVertices(uint32_t id, std::span<const Vertex3D> vertices);
    void Release(uint32_t id);

private:
    static constexpr uint32_t InvalidStoreId{~0u};

    uint32_t AcquireId();
    std::vector<Entry> Entries;
    std::vector<uint32_t> FreeIds;
    std::unique_ptr<Megabuffer<Vertex3D>> VerticesBuffer;
};
