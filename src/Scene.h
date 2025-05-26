#pragma once

#include "Camera.h"
#include "ModelGizmo.h"
#include "Shader.h"
#include "Vulkan/BufferManager.h"
#include "Vulkan/Image.h"
#include "mesh/MeshElement.h"
#include "mesh/Primitive.h"
#include "mesh/Vertex.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <entt/entity/fwd.hpp>

#include <memory>
#include <set>
#include <unordered_map>

using uint = uint32_t;

namespace mvk {
struct RenderBuffers {
    Buffer Vertices, Indices;
};
} // namespace mvk

struct Path {
    fs::path Value;
};

struct Mesh;
struct Excitable;
struct RenderBuffers; // Mesh render buffers

struct Position {
    vec3 Value;
};
struct Rotation {
    quat Value;
};

struct Model {
    Model(mat4 transform)
        : Transform{std::move(transform)}, InvTransform{glm::transpose(glm::inverse(Transform))} {}

    mat4 Transform;
    // `InvTransform` is the _transpose_ of the inverse of `Transform`.
    // Since this rarely changes, we precompute it and send it to the shader.
    mat4 InvTransform;
};

struct ViewProj {
    mat4 View{1}, Projection{1};
};

struct ViewProjNearFar {
    mat4 View{1}, Projection{1};
    float Near, Far;
};

struct Lights {
    // RGB: Color of the light emitting from the view position.
    // A: Ambient intensity. (Using a single vec4 for 16-byte alignment.)
    vec4 ViewColorAndAmbient;
    // RGB: Color of the directional light.
    // A: Intensity of the directional light.
    vec4 DirectionalColorAndIntensity;
    vec3 Direction;
};

struct World {
    const vec3 Origin{0, 0, 0}, Up{0, 1, 0};
};

enum class ShaderPipelineType {
    Fill,
    Line,
    Grid,
    SilhouetteDepthObject,
    SilhouetteEdgeDepthObject,
    SilhouetteEdgeColor,
    DebugNormals,
};
using SPT = ShaderPipelineType;

struct ShaderBindingDescriptor {
    SPT PipelineType;
    std::string_view BindingName;
    const vk::DescriptorBufferInfo *BufferInfo{nullptr};
    const vk::DescriptorImageInfo *ImageInfo{nullptr};
};
struct PipelineRenderer {
    vk::UniqueRenderPass RenderPass;
    std::unordered_map<SPT, ShaderPipeline> ShaderPipelines;

    void CompileShaders();

    std::vector<vk::WriteDescriptorSet> GetDescriptors(std::vector<ShaderBindingDescriptor> &&) const;

    // If `model_index` is set, only the model at that index is rendered.
    // Otherwise, all models are rendered.
    void Render(vk::CommandBuffer, SPT, const mvk::RenderBuffers &, const mvk::Buffer &models, std::optional<uint> model_index = std::nullopt) const;
};

enum class SelectionMode {
    Object, // Select objects
    Edit, // Select individual mesh elements (vertices, edges, faces)
    // Available when any `Excitable` objects are present.
    // Mouse-down on an excitable object adds an `ExcitedVertex` component to the object entity, and mouse-up removes it.
    Excite,
};

enum class RenderMode {
    None,
    FacesAndEdges,
    Faces,
    Edges,
    Vertices,
};

enum class ColorMode {
    Mesh,
    Normals,
};

inline std::string to_string(SelectionMode mode) {
    switch (mode) {
        case SelectionMode::Object: return "Object";
        case SelectionMode::Edit: return "Edit";
        case SelectionMode::Excite: return "Excite";
    }
}

struct MeshCreateInfo {
    std::string Name{};
    vec3 Position{0};
    quat Rotation{1, 0, 0, 0};
    vec3 Scale{1};
    bool Select{true}, Visible{true};
};

static constexpr Camera CreateDefaultCamera() { return {{0, 0, 2}, {0, 0, 0}, 60, 0.01, 100}; }

inline static vk::SampleCountFlagBits GetMaxUsableSampleCount(vk::PhysicalDevice pd) {
    const auto props = pd.getProperties();
    const auto counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64) return vk::SampleCountFlagBits::e64;
    if (counts & vk::SampleCountFlagBits::e32) return vk::SampleCountFlagBits::e32;
    if (counts & vk::SampleCountFlagBits::e16) return vk::SampleCountFlagBits::e16;
    if (counts & vk::SampleCountFlagBits::e8) return vk::SampleCountFlagBits::e8;
    if (counts & vk::SampleCountFlagBits::e4) return vk::SampleCountFlagBits::e4;
    if (counts & vk::SampleCountFlagBits::e2) return vk::SampleCountFlagBits::e2;

    return vk::SampleCountFlagBits::e1;
}

struct ScenePipelines;

struct SceneVulkanResources {
    vk::Instance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::Device Device;
    uint32_t QueueFamily;
    vk::Queue Queue;
    vk::DescriptorPool DescriptorPool;
};

struct Scene {
    Scene(SceneVulkanResources, entt::registry &);
    ~Scene();

    World World{};

    entt::entity AddMesh(Mesh &&, MeshCreateInfo = {});
    entt::entity AddMesh(const fs::path &, MeshCreateInfo = {});

    entt::entity AddPrimitive(Primitive, MeshCreateInfo = {});
    entt::entity AddInstance(entt::entity, MeshCreateInfo = {});

    void ReplaceMesh(entt::entity, Mesh &&);
    void ClearMeshes();

    void DestroyInstance(entt::entity);
    void DestroyEntity(entt::entity);

    void SetModel(entt::entity, vec3 position, quat rotation, vec3 scale);

    void SetVisible(entt::entity, bool);

    entt::entity GetParentEntity(entt::entity) const;
    void SetActive(entt::entity);
    void ToggleSelected(entt::entity);

    vk::Extent2D GetExtent() const { return Extent; }
    vk::ImageView GetResolveImageView() const;

    // Handle mouse/keyboard interactions.
    void Interact();
    ray GetMouseWorldRay() const; // World space ray from the mouse into the scene.

    // Renders to a texture sampler and image view that can be accessed with `GetTextureSampler()` and `GetResolveImageView()`.
    // The extent of the resolve image can be found with `GetExtent()` after the call,
    // and it will be equal to the dimensions of `GetContentRegionAvail()` at the beginning of the call.
    // Returns true if the scene was updated, which can happen when the window size or background color changes.
    bool Render();
    void RenderGizmo();
    void RenderControls();
    mvk::ImageResource RenderBitmapToImage(std::span<const std::byte> data, uint width, uint height) const;

    std::optional<uint> GetModelBufferIndex(entt::entity);
    void UpdateRenderBuffers(entt::entity);
    void RecordRenderCommandBuffer();
    void InvalidateCommandBuffer();

    void OnCreateSelected(entt::registry &, entt::entity);
    void OnDestroySelected(entt::registry &, entt::entity);

    void OnCreateExcitable(entt::registry &, entt::entity);
    void OnUpdateExcitable(entt::registry &, entt::entity);
    void OnDestroyExcitable(entt::registry &, entt::entity);

    void OnCreateExcitedVertex(entt::registry &, entt::entity);
    void OnDestroyExcitedVertex(entt::registry &, entt::entity);

private:
    SceneVulkanResources Vk;
    entt::registry &R;
    vk::UniqueCommandPool CommandPool;
    vk::UniqueCommandBuffer TransferCommandBuffer, RenderCommandBuffer;
    std::array<vk::CommandBuffer, 2> CommandBuffers;
    vk::UniqueFence RenderFence, TransferFence;
    mvk::BufferManager BufferManager;

    Camera Camera{CreateDefaultCamera()};
    Lights Lights{{1, 1, 1, 0.1}, {1, 1, 1, 0.15}, {-1, -1, -1}};

    vec4 EdgeColor{1, 1, 1, 1}; // Used for line mode.
    vec4 MeshEdgeColor{0, 0, 0, 1}; // Used for faces mode.

    RenderMode RenderMode{RenderMode::FacesAndEdges};
    ColorMode ColorMode{ColorMode::Mesh};

    std::set<SelectionMode> SelectionModes{SelectionMode::Object, SelectionMode::Edit};
    SelectionMode SelectionMode{SelectionMode::Object};
    MeshElementIndex EditingElement{MeshElement::Face, -1};

    vk::Extent2D Extent;
    vk::ClearColorValue BackgroundColor{0.22, 0.22, 0.22, 1.f};

    mvk::Buffer TransformBuffer, ViewProjNearFarBuffer, LightsBuffer, SilhouetteDisplayBuffer;

    std::unique_ptr<ScenePipelines> Pipelines;

    struct ModelGizmoState {
        ModelGizmo::Op Op{ModelGizmo::Op::Translate};
        bool Show{false};
        bool Snap; // Snap translate and scale gizmos.
        vec3 SnapValue{0.5f};
    };
    ModelGizmoState MGizmo;

    bool ShowGrid{true};

    // Selected entity render settings
    bool ShowBoundingBoxes{false}, ShowBvhBoxes{false};
    static inline const std::vector NormalElements{MeshElement::Vertex, MeshElement::Face};
    std::unordered_set<MeshElement> ShownNormalElements{};
    vec4 ActiveSilhouetteColor{1, 0.627, 0.157, 1}; // Blender's default `Preferences->Themes->3D Viewport->Active Object`.
    vec4 SelectedSilhouetteColor{0.929, 0.341, 0, 1}; // Blender's default `Preferences->Themes->3D Viewport->Object Selected`.

    bool CommandBufferDirty{false};

    void SetSelectionMode(::SelectionMode);
    void SetEditingElement(MeshElementIndex);

    const Mesh &GetActiveMesh() const;

    void RenderEntitiesTable(std::string name, const std::vector<entt::entity> &);

    // VK buffer update methods
    void UpdateTransformBuffers();
    void UpdateModelBuffer(entt::entity);
    void UpdateEdgeColors();
    void UpdateHighlightedVertices(entt::entity, const Excitable &);
    void UpdateEntitySelectionOverlays(entt::entity);
    void RemoveEntitySelectionOverlays(entt::entity);

    mvk::RenderBuffers CreateRenderBuffers(std::vector<Vertex3D> &&vertices, std::vector<uint> &&indices) {
        return {
            BufferManager.Create(as_bytes(vertices), vk::BufferUsageFlagBits::eVertexBuffer),
            BufferManager.Create(as_bytes(indices), vk::BufferUsageFlagBits::eIndexBuffer)
        };
    }

    mvk::RenderBuffers CreateRenderBuffers(RenderBuffers &&);

    template<size_t N>
    mvk::RenderBuffers CreateRenderBuffers(std::vector<Vertex3D> &&vertices, const std::array<uint, N> &indices) {
        return {
            BufferManager.Create(as_bytes(vertices), vk::BufferUsageFlagBits::eVertexBuffer),
            BufferManager.Create(as_bytes(indices), vk::BufferUsageFlagBits::eIndexBuffer)
        };
    }

    void WaitFor(vk::Fence) const;
};
