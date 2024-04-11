#pragma once

#include <entt/entity/registry.hpp>

#include "numeric/mat4.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include "Camera.h"
#include "RenderMode.h"
#include "Shader.h"
#include "World.h"
#include "mesh/MeshElement.h"
#include "mesh/Primitive.h"

struct Visible {}; // A tag component to mark entities that are visible.

struct Mesh;
struct VulkanContext;
struct VkRenderBuffers;
struct MeshVkData;
struct VulkanBuffer;

struct ImageResource {
    // The `image` in the view info is overwritten.
    void Create(const VulkanContext &, vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Dereference forwards to the image.
    const vk::Image &operator*() const { return *Image; }

    vk::UniqueImage Image;
    vk::UniqueImageView View;
    vk::UniqueDeviceMemory Memory;
};

struct Model {
    Model(mat4 &&transform)
        : Transform{std::move(transform)}, InvTransform{glm::transpose(glm::inverse(Transform))} {}

    mat4 Transform{1};
    // `InvTransform` is the _transpose_ of the inverse of `Transform`.
    // Since this rarely changes, we precompute it and send it to the shader.
    mat4 InvTransform{1};
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

struct Gizmo;

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
    ImageResource DepthImage, OffscreenImage, ResolveImage;

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer, const vk::ClearColorValue &background_color) const;
};

struct SilhouettePipeline : RenderPipeline {
    SilhouettePipeline(const VulkanContext &);

    vk::Extent2D Extent;
    ImageResource OffscreenImage; // Single-sampled image without a depth buffer.

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer) const;
};

struct EdgeDetectionPipeline : RenderPipeline {
    EdgeDetectionPipeline(const VulkanContext &);

    vk::Extent2D Extent;
    ImageResource OffscreenImage; // Single-sampled image without a depth buffer.

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer) const;
};

enum class SelectionMode {
    Object, // Select objects
    Edit, // Select individual mesh elements (vertices, edges, faces)
};

struct Scene {
    Scene(const VulkanContext &, entt::registry &);
    ~Scene();

    const VulkanContext &VC;
    entt::registry &R;
    World World{};

    std::optional<uint> GetModelBufferIndex(entt::entity);
    entt::entity GetSelectedEntity() const { return SelectedEntity; }
    entt::entity GetParentEntity(entt::entity) const;
    Mesh &GetMesh(entt::entity) const;

    entt::entity AddMesh(Mesh &&, const mat4 &transform = {1}, bool submit = true, bool select = true, bool visible = true);
    entt::entity AddMesh(const fs::path &, const mat4 &transform = {1}, bool submit = true, bool select = true, bool visible = true);
    entt::entity AddPrimitive(Primitive, const mat4 &transform = {1}, bool submit = true, bool select = true, bool visible = true);

    void ReplaceMesh(entt::entity, Mesh &&);
    void ClearMeshes();

    mat4 GetModel(entt::entity) const;
    void SetModel(entt::entity, mat4 &&, bool submit = true);

    void SetVisible(entt::entity, bool);

    entt::entity AddInstance(entt::entity, mat4 &&transform = {1}, bool visible = true);
    void DestroyInstance(entt::entity);
    void DestroyEntity(entt::entity);

    const vk::Extent2D &GetExtent() const { return Extent; }
    vk::SampleCountFlagBits GetMsaaSamples() const { return MainPipeline.MsaaSamples; }
    vk::ImageView GetResolveImageView() const { return *MainPipeline.ResolveImage.View; }

    // Renders to a texture sampler and image view that can be accessed with `GetTextureSampler()` and `GetResolveImageView()`.
    // The extent of the resolve image can be found with `GetExtent()` after the call,
    // and it will be equal to the dimensions of `GetContentRegionAvail()` at the beginning of the call.
    // Returns true if the scene was updated, which can happen when the window size or background color changes.
    bool Render();
    void RenderGizmo();
    void RenderConfig();

    // These do _not_ re-submit the command buffer. Callers must do so manually if needed.
    void CompileShaders();

    Camera CreateDefaultCamera() const { return {World.Up, {0, 0, 2}, World.Origin, 60, 0.01, 100}; }

    void UpdateRenderBuffers(entt::entity, const MeshElementIndex &highlight_element = {});
    void RecordCommandBuffer();
    void SubmitCommandBuffer(vk::Fence fence = nullptr) const;
    void RecordAndSubmitCommandBuffer(vk::Fence fence = nullptr);

private:
    Camera Camera{CreateDefaultCamera()};
    Lights Lights{{1, 1, 1, 0.1}, {1, 1, 1, 0.15}, {-1, -1, -1}};

    vec4 EdgeColor{1, 1, 1, 1}; // Used for line mode.
    vec4 MeshEdgeColor{0, 0, 0, 1}; // Used for faces mode.

    RenderMode RenderMode{RenderMode::FacesAndEdges};
    ColorMode ColorMode{ColorMode::Mesh};

    SelectionMode SelectionMode{SelectionMode::Object};
    MeshElement SelectionElement{MeshElement::Face};
    MeshElementIndex SelectedElement{};

    entt::entity SelectedEntity{entt::null};
    std::unique_ptr<MeshVkData> MeshVkData;

    vk::Extent2D Extent;
    vk::ClearColorValue BackgroundColor;
    vk::UniqueSampler SilhouetteFillImageSampler, SilhouetteEdgeImageSampler;

    std::unique_ptr<VulkanBuffer> TransformBuffer, ViewProjNearFarBuffer, LightsBuffer, SilhouetteDisplayBuffer;

    MainPipeline MainPipeline;
    SilhouettePipeline SilhouettePipeline;
    EdgeDetectionPipeline EdgeDetectionPipeline;
    std::vector<std::unique_ptr<RenderPipeline>> RenderPipelines;

    std::unique_ptr<Gizmo> Gizmo;

    bool ShowGrid{true}, ShowBoundingBoxes{false}, ShowBvhBoxes{false};
    SilhouetteDisplay SilhouetteDisplay{{1, 0.627, 0.157, 1.}}; // Blender's default `Preferences->Themes->3D Viewport->Active Object`.
    vec4 BgColor{0.22, 0.22, 0.22, 1.};

    void SelectEntity(entt::entity entity, bool submit = false) {
        SelectedEntity = entity;
        if (submit) RecordAndSubmitCommandBuffer();
    }

    void SetSelectionMode(::SelectionMode mode) {
        SelectionMode = mode;
        RecordAndSubmitCommandBuffer();
    }

    Mesh &GetSelectedMesh() const;

    // VK buffer update methods.
    void UpdateTransformBuffers();
    void UpdateModelBuffer(entt::entity);
    void UpdateEdgeColors();

    void SetExtent(vk::Extent2D);
};
