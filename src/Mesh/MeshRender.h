#pragma once

#include "../Vulkan/Buffer.h"
#include "BBox.h"
#include "Mesh.h"
#include "mesh/Handle.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <entt_fwd.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

struct WorldMatrix;

// Submitted to shaders
struct Vertex3D {
    vec3 Position;
    vec3 Normal;
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
    mvk::Buffer Buffer;
    mvk::Buffer ObjectIds; // Per-instance ObjectIds for selection/silhouette rendering.
};

// Component for entities that render a mesh via instancing.
// References an entity with Mesh+MeshBuffers+ModelsBuffer components.
struct MeshInstance {
    entt::entity MeshEntity{null_entity};
};

struct RenderBuffers {
    RenderBuffers(mvk::Buffer &&vertices, mvk::Buffer &&indices)
        : Vertices(std::move(vertices)), Indices(std::move(indices)) {}
    RenderBuffers(RenderBuffers &&) = default;
    RenderBuffers(const RenderBuffers &) = delete;
    RenderBuffers &operator=(const RenderBuffers &) = delete;

    mvk::Buffer Vertices, Indices;
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

struct MeshFaceIdBuffer {
    mvk::Buffer Faces;
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

// Create vertices for faces (with optional smooth/flat shading) or edges
std::vector<Vertex3D> CreateFaceVertices(
    const Mesh &mesh,
    bool smooth_shading
);
std::vector<Vertex3D> CreateEdgeVertices(
    const Mesh &mesh
);
std::vector<Vertex3D> CreateVertexPoints(
    const Mesh &mesh
);
std::vector<uint> CreateFaceIndices(const Mesh &mesh);
std::vector<uint> CreateEdgeIndices(const Mesh &mesh);
std::vector<uint> CreateVertexIndices(const Mesh &mesh);
std::vector<uint32_t> CreateFaceElementIds(const Mesh &mesh);

std::vector<Vertex3D> CreateNormalVertices(const Mesh &mesh, he::Element element);
std::vector<uint> CreateNormalIndices(const Mesh &mesh, he::Element element);

std::vector<BBox> CreateFaceBoundingBoxes(const Mesh &mesh);

BBox ComputeBoundingBox(const Mesh &mesh);

} // namespace MeshRender
