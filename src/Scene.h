#pragma once

#include "Camera.h"
#include "ModelGizmo.h"
#include "RenderMode.h"
#include "Shader.h"
#include "World.h"
#include "mesh/MeshElement.h"
#include "mesh/Primitive.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <entt/entity/fwd.hpp>

#include <set>
#include <unordered_map>

struct Selected {}; // Entity is selected (multiple can be selected)
// Active selected entity
// Invariants:
//   * <=1 entity is active at a time.
//   * If an entity is Active, it is also Selected.
//   * Most recently Selected entity is Active.
struct Active {};
struct Visible {}; // Visible in the scene
struct Frozen {}; // Disable entity transform changes

struct Path {
    fs::path Value;
};

struct Mesh;
struct MeshVkData;
struct Excitable;
struct VulkanContext;
struct VkRenderBuffers;
struct VulkanBuffer;

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

struct ShaderBindingDescriptor {
    SPT PipelineType;
    std::string BindingName;
    std::optional<vk::DescriptorBufferInfo> BufferInfo{};
    std::optional<vk::DescriptorImageInfo> ImageInfo{};
};
struct PipelineRenderer {
    vk::UniqueRenderPass RenderPass;
    std::unordered_map<SPT, ShaderPipeline> ShaderPipelines;

    void CompileShaders();

    std::vector<vk::WriteDescriptorSet> GetDescriptors(std::vector<ShaderBindingDescriptor> &&) const;

    // If `model_index` is set, only the model at that index is rendered.
    // Otherwise, all models are rendered.
    void Render(vk::CommandBuffer, SPT, const VulkanBuffer &vertices, const VulkanBuffer &indices, const VulkanBuffer &models, std::optional<uint> model_index = std::nullopt) const;
    void Render(vk::CommandBuffer, SPT, const VkRenderBuffers &, const VulkanBuffer &models, std::optional<uint> model_index = std::nullopt) const;
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
    quat Rotation{1, 0, 0, 0};
    vec3 Scale{1};
    bool Select{true}, Visible{true};
};

static constexpr Camera CreateDefaultCamera() { return {{0, 0, 2}, {0, 0, 0}, 60, 0.01, 100}; }

struct MainPipelineResources;
struct SilhouettePipelineResources;
struct EdgeDetectionPipelineResources;

struct Scene {
    Scene(const VulkanContext &, entt::registry &);
    ~Scene();

    const VulkanContext &VC;
    World World{};

    std::optional<uint> GetModelBufferIndex(entt::entity);

    entt::entity AddMesh(Mesh &&, MeshCreateInfo info = {});
    entt::entity AddMesh(const fs::path &, MeshCreateInfo info = {});

    entt::entity AddPrimitive(Primitive, MeshCreateInfo info = {});
    entt::entity AddInstance(entt::entity, MeshCreateInfo info = {});

    void ReplaceMesh(entt::entity, Mesh &&);
    void ClearMeshes();

    void DestroyInstance(entt::entity);
    void DestroyEntity(entt::entity);

    void SetModel(entt::entity, vec3 position, quat rotation, vec3 scale);

    void SetVisible(entt::entity, bool);

    entt::entity GetParentEntity(entt::entity) const;
    void SetActive(entt::entity);
    entt::entity GetActiveEntity() const;
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

    // These do _not_ re-submit the command buffer. Callers must do so manually if needed.
    void CompileShaders();

    void UpdateRenderBuffers(entt::entity);
    void RecordCommandBuffer();
    void SubmitCommandBuffer(vk::Fence fence = nullptr) const;
    void InvalidateCommandBuffer();

    void OnCreateExcitable(entt::registry &, entt::entity);
    void OnUpdateExcitable(entt::registry &, entt::entity);
    void OnDestroyExcitable(entt::registry &, entt::entity);

    void OnCreateExcitedVertex(entt::registry &, entt::entity);
    void OnDestroyExcitedVertex(entt::registry &, entt::entity);

private:
    entt::registry &R;
    vk::UniqueCommandBuffer CommandBuffer;

    Camera Camera{CreateDefaultCamera()};
    Lights Lights{{1, 1, 1, 0.1}, {1, 1, 1, 0.15}, {-1, -1, -1}};

    vec4 EdgeColor{1, 1, 1, 1}; // Used for line mode.
    vec4 MeshEdgeColor{0, 0, 0, 1}; // Used for faces mode.

    RenderMode RenderMode{RenderMode::FacesAndEdges};
    ColorMode ColorMode{ColorMode::Mesh};

    std::set<SelectionMode> SelectionModes{SelectionMode::Object, SelectionMode::Edit};
    SelectionMode SelectionMode{SelectionMode::Object};
    MeshElementIndex EditingElement{MeshElement::Face, -1};

    std::unique_ptr<MeshVkData> MeshVkData;

    vk::Extent2D Extent;
    vk::ClearColorValue BackgroundColor{0.22, 0.22, 0.22, 1.f};
    vk::UniqueSampler SilhouetteFillImageSampler, SilhouetteEdgeImageSampler;

    std::unique_ptr<VulkanBuffer> TransformBuffer, ViewProjNearFarBuffer, LightsBuffer, SilhouetteDisplayBuffer;

    PipelineRenderer MainRenderer, SilhouetteRenderer, EdgeDetectionRenderer;
    std::unique_ptr<MainPipelineResources> MainResources;
    std::unique_ptr<SilhouettePipelineResources> SilhouetteResources;
    std::unique_ptr<EdgeDetectionPipelineResources> EdgeDetectionResources;

    struct ModelGizmoState {
        ModelGizmo::Op Op{ModelGizmo::Op::Translate};
        bool Show{false};
        bool Snap; // Snap translate and scale gizmos.
        vec3 SnapValue{0.5f};
    };
    ModelGizmoState MGizmo;

    bool ShowGrid{true}, ShowBoundingBoxes{false}, ShowBvhBoxes{false};
    vec4 ActiveSilhouetteColor{1, 0.627, 0.157, 1}; // Blender's default `Preferences->Themes->3D Viewport->Active Object`.
    vec4 SelectedSilhouetteColor{0.929, 0.341, 0, 1}; // Blender's default `Preferences->Themes->3D Viewport->Object Selected`.

    bool CommandBufferDirty{false};

    void SetSelectionMode(::SelectionMode);
    void SetEditingElement(MeshElementIndex);

    const Mesh &GetActiveMesh() const;

    void RenderEntitiesTable(std::string name, const std::vector<entt::entity> &);

    // VK buffer update methods.
    void UpdateTransformBuffers();
    void UpdateModelBuffer(entt::entity);
    void UpdateEdgeColors();
    void UpdateHighlightedVertices(entt::entity, const Excitable &);

    void SetExtent(vk::Extent2D);
};
