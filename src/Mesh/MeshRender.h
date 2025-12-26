#pragma once

#include "../Slots.h"
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
    uint32_t ObjectId{0};
};

// Stored on mesh entities.
// Holds the `WorldMatrix` of all instances of the mesh.
struct ModelsBuffer {
    mvk::UniqueBuffers Buffer;
    mvk::UniqueBuffers ObjectIds; // Per-instance ObjectIds for selection/silhouette rendering.
    uint32_t Slot{InvalidSlot};
    uint32_t ObjectIdSlot{InvalidSlot};
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
    uint32_t VertexSlot{InvalidSlot};
    uint32_t IndexSlot{InvalidSlot};
};

struct BoundingBoxesBuffers {
    RenderBuffers Buffers;
};
struct BvhBoxesBuffers {
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

// Returns `std::nullopt` if the entity is not Visible (and thus does not have a RenderInstance).
std::optional<uint32_t> GetModelBufferIndex(const entt::registry &, entt::entity);
void UpdateModelBuffer(entt::registry &, entt::entity, const WorldMatrix &);

namespace MeshRender {
// Rendering colors
inline vec4 VertexColor{1}, EdgeColor{0, 0, 0, 1};

constexpr vec4 ActiveColor{1, 1, 1, 1};
constexpr vec4 SelectedColor{1, 0.478, 0, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Select
constexpr vec4 HighlightedColor{0, 0.647, 1, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Bevel
constexpr vec4 FaceNormalIndicatorColor{0.133, 0.867, 0.867, 1}; // Blender: Preferences->Themes->3D Viewport->Face Normal
constexpr vec4 VertexNormalIndicatorColor{0.137, 0.380, 0.867, 1}; // Blender: Preferences->Themes->3D Viewport->Vertex Normal
constexpr vec4 HighlightedFaceColor{0.790, 0.930, 1, 1}; // Custom
constexpr vec4 UnselectedVertexEditColor{0, 0, 0, 1}; // Unselected vertex in edit mode
constexpr float NormalIndicatorLengthScale{0.25};

// Create vertices for faces (with optional smooth/flat shading) or edges
std::vector<Vertex3D> CreateFaceVertices(
    const Mesh &mesh,
    bool smooth_shading,
    const std::unordered_set<he::VH> &highlighted_vertices = {},
    const std::unordered_set<he::FH> &selected_faces = {},
    const std::unordered_set<he::FH> &active_faces = {}
);
std::vector<Vertex3D> CreateEdgeVertices(
    const Mesh &mesh,
    he::Element edit_mode,
    const std::unordered_set<he::VH> &selected_vertices = {},
    const std::unordered_set<he::EH> &selected_edges = {},
    const std::unordered_set<he::VH> &active_vertices = {},
    const std::unordered_set<he::EH> &active_edges = {}
);
std::vector<Vertex3D> CreateVertexPoints(
    const Mesh &mesh,
    he::Element edit_mode,
    const std::unordered_set<he::VH> &selected_vertices = {},
    const std::unordered_set<he::VH> &active_vertices = {}
);
std::vector<uint> CreateFaceIndices(const Mesh &mesh);
std::vector<uint> CreateEdgeIndices(const Mesh &mesh);
std::vector<uint> CreateVertexIndices(const Mesh &mesh);

std::vector<Vertex3D> CreateNormalVertices(const Mesh &mesh, he::Element element);
std::vector<uint> CreateNormalIndices(const Mesh &mesh, he::Element element);

std::vector<BBox> CreateFaceBoundingBoxes(const Mesh &mesh);
MeshRenderBuffers CreateBvhBuffers(const BVH &bvh, vec4 color);

BBox ComputeBoundingBox(const Mesh &mesh);

} // namespace MeshRender
