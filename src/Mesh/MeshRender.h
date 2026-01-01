#pragma once

#include "../Vulkan/Megabuffer.h"
#include "BBox.h"
#include "MeshData.h"
#include "mesh/Handle.h"
#include "numeric/vec4.h"

#include <entt_fwd.h>

#include <unordered_map>
#include <vector>

struct WorldMatrix;
struct Mesh;

struct Visible {};

struct RenderInstance {
    uint32_t BufferIndex{0}; // Slot in GPU model instance buffer
    uint32_t ObjectId{0};
};

// Stored on mesh entities.
// Holds the `WorldMatrix` of all instances of the mesh.
struct ModelsBuffer {
    mvk::Buffer Buffer;
    mvk::Buffer ObjectIds; // Per-instance ObjectIds for selection/silhouette rendering.
};

// Component for entities that render a mesh via instancing.
// References an entity with Mesh+MeshBuffers+ModelsBuffer components.
struct MeshInstance {
    entt::entity MeshEntity;
};

struct RenderBuffers {
    RenderBuffers(uint32_t vertex_range_id, mvk::Buffer &&indices)
        : VertexRangeId(vertex_range_id), Indices(std::move(indices)) {}
    RenderBuffers(uint32_t vertex_slot, uint32_t vertex_offset, uint32_t vertex_count, mvk::Buffer &&indices)
        : VertexSlot(vertex_slot), VertexOffset(vertex_offset), VertexCount(vertex_count), Indices(std::move(indices)) {}
    RenderBuffers(RenderBuffers &&) = default;
    RenderBuffers &operator=(RenderBuffers &&) = default;
    RenderBuffers(const RenderBuffers &) = delete;
    RenderBuffers &operator=(const RenderBuffers &) = delete;

    uint32_t VertexRangeId{InvalidSlot};
    uint32_t VertexSlot{InvalidSlot};
    uint32_t VertexOffset{0};
    uint32_t VertexCount{0};
    mvk::Buffer Indices;
};

struct BoundingBoxesBuffers {
    RenderBuffers Buffers;
};

struct MeshBuffers {
    MeshBuffers(RenderBuffers &&faces, RenderBuffers &&edges, RenderBuffers &&vertices)
        : Faces{std::move(faces)}, Edges{std::move(edges)}, Vertices{std::move(vertices)} {}
    MeshBuffers(const MeshBuffers &) = delete;
    MeshBuffers &operator=(const MeshBuffers &) = delete;

    RenderBuffers Faces, Edges, Vertices;
    std::unordered_map<he::Element, RenderBuffers> NormalIndicators;
};

struct ElementStateBuffer {
    mvk::Buffer Buffer;
};
struct MeshElementStateBuffers {
    ElementStateBuffer Faces, Edges, Vertices;
};
// Returns `std::nullopt` if the entity is not Visible (and thus does not have a RenderInstance).
std::optional<uint32_t> GetModelBufferIndex(const entt::registry &, entt::entity);
void UpdateModelBuffer(entt::registry &, entt::entity, const WorldMatrix &);

namespace MeshRender {
// Rendering colors
inline vec4 VertexColor{1}, EdgeColor{0, 0, 0, 1};

constexpr vec4 ActiveColor{1, 1, 1, 1};
constexpr vec4 SelectedColor{1, 0.478, 0, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Select
constexpr vec4 FaceNormalIndicatorColor{0.133, 0.867, 0.867, 1}; // Blender: Preferences->Themes->3D Viewport->Face Normal
constexpr vec4 VertexNormalIndicatorColor{0.137, 0.380, 0.867, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Normal
constexpr vec4 UnselectedVertexEditColor{0, 0, 0, 1}; // Unselected vertex in edit mode
constexpr float NormalIndicatorLengthScale{0.25};
constexpr uint32_t ElementStateSelected{1u << 0};
constexpr uint32_t ElementStateActive{1u << 1};

std::vector<uint> CreateFaceIndices(const Mesh &);
std::vector<uint> CreateEdgeIndices(const Mesh &);
std::vector<uint> CreateVertexIndices(const Mesh &);
std::vector<Vertex3D> CreateNormalVertices(const Mesh &, he::Element);
std::vector<uint> CreateNormalIndices(const Mesh &, he::Element);

BBox ComputeBoundingBox(const Mesh &);

} // namespace MeshRender

struct VertexMegabuffer {
    explicit VertexMegabuffer(mvk::BufferContext &ctx)
        : Storage(ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer),
          Buffer(Storage.Buffer) {}

    uint32_t Allocate(std::span<const Vertex3D> vertices) {
        uint32_t id = AcquireId();
        Ranges[id] = Storage.Allocate(vertices);
        return id;
    }

    void Update(uint32_t id, std::span<const Vertex3D> vertices) {
        if (id == InvalidSlot) return;
        Storage.Update(Ranges.at(id), vertices);
    }

    void Release(uint32_t id) {
        if (id == InvalidSlot) return;
        Storage.Release(Ranges.at(id));
        Ranges[id] = {};
        FreeIds.emplace_back(id);
    }

    BufferRange Get(uint32_t id) const { return Ranges.at(id); }

    Megabuffer<Vertex3D> Storage;
    mvk::Buffer &Buffer;

private:
    std::vector<BufferRange> Ranges;
    std::vector<uint32_t> FreeIds;

    uint32_t AcquireId() {
        if (!FreeIds.empty()) {
            const uint32_t id = FreeIds.back();
            FreeIds.pop_back();
            return id;
        }
        Ranges.emplace_back();
        return static_cast<uint32_t>(Ranges.size() - 1);
    }
};
