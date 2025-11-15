#pragma once

#include "../Vulkan/UniqueBuffers.h"
#include "BBox.h"
#include "Mesh.h"
#include "mesh/Handle.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <entt_fwd.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

struct BVH;
struct WorldMatrix;

// Submitted to shaders
struct Vertex3D {
    vec3 Position;
    vec3 Normal;
    vec4 Color;
};

struct MeshRenderBuffers {
    std::vector<Vertex3D> Vertices;
    std::vector<uint> Indices;
};

struct Visible {};

struct RenderInstance {
    uint32_t BufferIndex{0}; // Slot in GPU model instance buffer
};

// Stored on mesh entities.
// Holds the `WorldMatrix` of all instances of the mesh.
struct ModelsBuffer {
    mvk::UniqueBuffers Buffer;
};

// Component for entities that render a mesh via instancing.
// References an entity with Mesh+MeshBuffers+ModelsBuffer components.
struct MeshInstance {
    entt::entity MeshEntity{null_entity};
};

struct RenderBuffers {
    RenderBuffers(mvk::UniqueBuffers &&vertices, mvk::UniqueBuffers &&indices)
        : Vertices(std::move(vertices)), Indices(std::move(indices)) {}
    RenderBuffers(RenderBuffers &&) = default;
    RenderBuffers(const RenderBuffers &) = delete;
    RenderBuffers &operator=(const RenderBuffers &) = delete;

    mvk::UniqueBuffers Vertices, Indices;
};

struct BoundingBoxesBuffers {
    RenderBuffers Buffers;
};
struct BvhBoxesBuffers {
    RenderBuffers Buffers;
};
using RenderBuffersByElement = std::unordered_map<he::Element, RenderBuffers>;
struct MeshBuffers {
    MeshBuffers(RenderBuffersByElement &&mesh, RenderBuffersByElement &&normal_indicators)
        : Mesh{std::move(mesh)}, NormalIndicators{std::move(normal_indicators)} {}
    MeshBuffers(const MeshBuffers &) = delete;
    MeshBuffers &operator=(const MeshBuffers &) = delete;

    RenderBuffersByElement Mesh, NormalIndicators;
};

// Returns `std::nullopt` if the entity is not Visible (and thus does not have a RenderInstance).
std::optional<uint32_t> GetModelBufferIndex(const entt::registry &, entt::entity);
void UpdateModelBuffer(entt::registry &, entt::entity, const WorldMatrix &);

namespace MeshRender {
// Rendering colors
inline vec4 VertexColor{1}, EdgeColor{0, 0, 0, 1};
inline vec4 SelectedColor{1, 0.478, 0, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Select
inline vec4 HighlightedColor{0, 0.647, 1, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Bevel
inline vec4 FaceNormalIndicatorColor{0.133, 0.867, 0.867, 1}; // Blender: Preferences->Themes->3D Viewport->Face Normal
inline vec4 VertexNormalIndicatorColor{0.137, 0.380, 0.867, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Normal
inline vec4 HighlightedFaceColor{0.790, 0.930, 1, 1}; // Custom
constexpr float NormalIndicatorLengthScale{0.25};

// Vertex: Triangulated face indices
// Face: Triangle fan for each face
// Edge: Edge line segment indices
std::vector<Vertex3D> CreateVertices(
    const Mesh &mesh,
    he::Element render_element,
    const he::AnyHandle &selected = {},
    const std::unordered_set<he::AnyHandle, he::AnyHandleHash> &highlighted = {}
);
std::vector<uint> CreateIndices(const Mesh &mesh, he::Element element);

std::vector<Vertex3D> CreateNormalVertices(const Mesh &mesh, he::Element element);
std::vector<uint> CreateNormalIndices(const Mesh &mesh, he::Element element);

std::vector<BBox> CreateFaceBoundingBoxes(const Mesh &mesh);
MeshRenderBuffers CreateBvhBuffers(const BVH &bvh, vec4 color);

BBox ComputeBoundingBox(const Mesh &mesh);

} // namespace MeshRender
