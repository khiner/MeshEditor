#pragma once

#include "Camera.h"
#include "ImGuizmo.h"
#include "RenderMode.h"
#include "Shader.h"
#include "World.h"
#include "mesh/MeshElement.h"
#include "mesh/Primitive.h"
#include "numeric/mat4.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <entt/entity/fwd.hpp>
#include <glm/gtx/quaternion.hpp>

#include <set>

struct Visible {}; // Tag to mark entities as visible in the scene
struct Frozen {}; // Tag to disable entity transform changes

struct Path {
    fs::path Value;
};

struct Mesh;
struct MeshVkData;
struct Excitable;
struct VulkanContext;
struct VkRenderBuffers;
struct VulkanBuffer;
struct ImageResource;

struct Position {
    vec3 Value;
};
struct Rotation {
    glm::quat Value;
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

struct SilhouetteDisplay {
    vec4 Color;
};

enum class ShaderPipelineType {
    Fill,
    Line,
    Grid,
    Silhouette,
    EdgeDetection,
    Texture,
    DebugNormals,
};
using SPT = ShaderPipelineType;

struct RenderPipeline {
    struct ShaderBindingDescriptor {
        SPT PipelineType;
        std::string BindingName;
        std::optional<vk::DescriptorBufferInfo> BufferInfo{};
        std::optional<vk::DescriptorImageInfo> ImageInfo{};
    };

    RenderPipeline(const VulkanContext &);
    virtual ~RenderPipeline();

    const VulkanContext &VC;

    // Updates images and framebuffer based on the new extent.
    // These resources are reused by future renders that don't change the extent.
    virtual void SetExtent(vk::Extent2D) = 0;

    const ShaderPipeline *GetShaderPipeline(SPT spt) const { return ShaderPipelines.at(spt).get(); }
    void CompileShaders();

    void UpdateDescriptors(std::vector<ShaderBindingDescriptor> &&) const;

    // If `model_index` is set, only the model at that index is rendered.
    // Otherwise, all models are rendered.
    void Render(vk::CommandBuffer, SPT, const VulkanBuffer &vertices, const VulkanBuffer &indices, const VulkanBuffer &models, std::optional<uint> model_index = std::nullopt) const;
    void Render(vk::CommandBuffer, SPT, const VkRenderBuffers &, const VulkanBuffer &models, std::optional<uint> model_index = std::nullopt) const;

protected:
    vk::UniqueFramebuffer Framebuffer;
    vk::UniqueRenderPass RenderPass;
    std::unordered_map<SPT, std::unique_ptr<ShaderPipeline>> ShaderPipelines;
};

struct MainPipeline : RenderPipeline {
    MainPipeline(const VulkanContext &);

    vk::SampleCountFlagBits MsaaSamples;
    vk::Extent2D Extent;
    // Perform depth testing, render into a multisampled offscreen image, and resolve into a single-sampled image.
    std::unique_ptr<ImageResource> DepthImage, OffscreenImage, ResolveImage;

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer, const vk::ClearColorValue &background_color) const;
};

struct SilhouettePipeline : RenderPipeline {
    SilhouettePipeline(const VulkanContext &);

    vk::Extent2D Extent;
    std::unique_ptr<ImageResource> OffscreenImage; // Single-sampled image without a depth buffer.

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer) const;
};

struct EdgeDetectionPipeline : RenderPipeline {
    EdgeDetectionPipeline(const VulkanContext &);

    vk::Extent2D Extent;
    std::unique_ptr<ImageResource> OffscreenImage; // Single-sampled image without a depth buffer.

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer) const;
};

enum class SelectionMode {
    Object, // Select objects
    Edit, // Select individual mesh elements (vertices, edges, faces)
    // Available when any `Excitable` objects are present.
    // Mouse-down on an excitable object adds an `ExcitedVertex` component to the object entity, and mouse-up removes it.
    Excite,
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
    glm::quat Rotation{1, 0, 0, 0};
    vec3 Scale{1};
    bool Select{true}, Visible{true};
};

static constexpr Camera CreateDefaultCamera(const World &world) { return {world.Up, {0, 0, 2}, world.Origin, 60, 0.01, 100}; }

struct Scene {
    Scene(const VulkanContext &, entt::registry &);
    ~Scene();

    const VulkanContext &VC;
    entt::registry &R;
    World World{};

    std::optional<uint> GetModelBufferIndex(entt::entity);
    entt::entity GetSelectedEntity() const { return SelectedEntity; }

    entt::entity AddMesh(Mesh &&, MeshCreateInfo info = {});
    entt::entity AddMesh(const fs::path &, MeshCreateInfo info = {});

    entt::entity AddPrimitive(Primitive, MeshCreateInfo info = {});
    entt::entity AddInstance(entt::entity, MeshCreateInfo info = {});

    void ReplaceMesh(entt::entity, Mesh &&);
    void ClearMeshes();

    void DestroyInstance(entt::entity);
    void DestroyEntity(entt::entity);

    void SetModel(entt::entity, vec3 position, glm::quat rotation, vec3 scale);

    void SetVisible(entt::entity, bool);

    void SelectEntity(entt::entity entity) {
        SelectedEntity = entity;
        InvalidateCommandBuffer();
    }

    const vk::Extent2D &GetExtent() const { return Extent; }
    vk::SampleCountFlagBits GetMsaaSamples() const { return MainPipeline.MsaaSamples; }
    vk::ImageView GetResolveImageView() const;

    // Handle mouse/keyboard interactions.
    void Interact();
    // Renders to a texture sampler and image view that can be accessed with `GetTextureSampler()` and `GetResolveImageView()`.
    // The extent of the resolve image can be found with `GetExtent()` after the call,
    // and it will be equal to the dimensions of `GetContentRegionAvail()` at the beginning of the call.
    // Returns true if the scene was updated, which can happen when the window size or background color changes.
    bool Render();
    void RenderGizmo();
    void RenderControls();

    // These do _not_ re-submit the command buffer. Callers must do so manually if needed.
    void CompileShaders();

    void UpdateRenderBuffers(entt::entity, MeshElementIndex highlight_element = {});
    void RecordCommandBuffer();
    void SubmitCommandBuffer(vk::Fence fence = nullptr) const;
    void InvalidateCommandBuffer();

    void OnCreateExcitable(entt::registry &, entt::entity);
    void OnUpdateExcitable(entt::registry &, entt::entity);
    void OnDestroyExcitable(entt::registry &, entt::entity);

    void OnCreateExcitedVertex(entt::registry &, entt::entity);
    void OnDestroyExcitedVertex(entt::registry &, entt::entity);

private:
    Camera Camera{CreateDefaultCamera(World)};
    Lights Lights{{1, 1, 1, 0.1}, {1, 1, 1, 0.15}, {-1, -1, -1}};

    vec4 EdgeColor{1, 1, 1, 1}; // Used for line mode.
    vec4 MeshEdgeColor{0, 0, 0, 1}; // Used for faces mode.

    RenderMode RenderMode{RenderMode::FacesAndEdges};
    ColorMode ColorMode{ColorMode::Mesh};

    std::set<SelectionMode> SelectionModes{SelectionMode::Object, SelectionMode::Edit};
    SelectionMode SelectionMode{SelectionMode::Object};
    MeshElementIndex SelectedElement{MeshElement::Face, -1};

    entt::entity SelectedEntity;
    std::unique_ptr<MeshVkData> MeshVkData;

    vk::Extent2D Extent;
    vk::ClearColorValue BackgroundColor{0.22, 0.22, 0.22, 1.f};
    vk::UniqueSampler SilhouetteFillImageSampler, SilhouetteEdgeImageSampler;

    std::unique_ptr<VulkanBuffer> TransformBuffer, ViewProjNearFarBuffer, LightsBuffer, SilhouetteDisplayBuffer;

    MainPipeline MainPipeline;
    SilhouettePipeline SilhouettePipeline;
    EdgeDetectionPipeline EdgeDetectionPipeline;
    std::vector<std::unique_ptr<RenderPipeline>> RenderPipelines;

    ImGuizmo::Operation ActiveGizmoOp{ImGuizmo::Operation::Translate};
    bool ShowModelGizmo{false};

    bool ShowGrid{true}, ShowBoundingBoxes{false}, ShowBvhBoxes{false};
    SilhouetteDisplay SilhouetteDisplay{{1, 0.627, 0.157, 1.}}; // Blender's default `Preferences->Themes->3D Viewport->Active Object`.

    bool CommandBufferDirty{false};

    void SetSelectionMode(::SelectionMode);
    void SetSelectedElement(MeshElementIndex);

    const Mesh &GetSelectedMesh() const;

    void RenderEntitiesTable(std::string name, const std::vector<entt::entity> &);

    // VK buffer update methods.
    void UpdateTransformBuffers();
    void UpdateModelBuffer(entt::entity);
    void UpdateEdgeColors();
    void UpdateHighlightedVertices(entt::entity, const Excitable &);

    void SetExtent(vk::Extent2D);
};
