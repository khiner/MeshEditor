#pragma once

#include "../vulkan/BufferArena.h"
#include "BBox.h"
#include "MeshData.h"
#include "Vertex3D.h"
#include "mesh/Handle.h"
#include "numeric/vec4.h"

#include <entt_fwd.h>

#include <unordered_map>
#include <vector>

struct WorldMatrix;
struct Mesh;

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

enum class IndexKind {
    Face,
    Edge,
    Vertex
};

struct SlottedBufferRange {
    BufferRange Range;
    uint32_t Slot;
};

struct RenderBuffers {
    RenderBuffers(BufferRange vertices, SlottedBufferRange indices, IndexKind index_type)
        : Vertices(vertices), Indices(indices), IndexType(index_type) {}
    RenderBuffers(RenderBuffers &&) = default;
    RenderBuffers &operator=(RenderBuffers &&) = default;
    RenderBuffers(const RenderBuffers &) = delete;
    RenderBuffers &operator=(const RenderBuffers &) = delete;

    BufferRange Vertices;
    SlottedBufferRange Indices;
    IndexKind IndexType;
};

struct BoundingBoxesBuffers {
    RenderBuffers Buffers;
};

struct MeshBuffers {
    MeshBuffers(SlottedBufferRange vertices, SlottedBufferRange face_indices, SlottedBufferRange edge_indices, SlottedBufferRange vertex_indices)
        : Vertices{vertices}, FaceIndices{face_indices}, EdgeIndices{edge_indices}, VertexIndices{vertex_indices} {}
    MeshBuffers(const MeshBuffers &) = delete;
    MeshBuffers &operator=(const MeshBuffers &) = delete;

    SlottedBufferRange Vertices;
    SlottedBufferRange FaceIndices, EdgeIndices, VertexIndices;
    std::unordered_map<he::Element, RenderBuffers> NormalIndicators;
};

struct MeshElementStateBuffers {
    mvk::Buffer Faces, Edges, Vertices;
};

// Returns `std::nullopt` if the entity does not have a RenderInstance (i.e., is not visible).
std::optional<uint32_t> GetModelBufferIndex(const entt::registry &, entt::entity);
void UpdateModelBuffer(entt::registry &, entt::entity, const WorldMatrix &);

namespace MeshRender {
constexpr uint32_t ElementStateSelected{1u << 0}, ElementStateActive{1u << 1};

std::vector<uint> CreateVertexIndices(const Mesh &);
std::vector<Vertex3D> CreateNormalVertices(const Mesh &, he::Element);
std::vector<uint> CreateNormalIndices(const Mesh &, he::Element);

BBox ComputeBoundingBox(const Mesh &);

} // namespace MeshRender
