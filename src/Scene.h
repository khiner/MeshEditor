#pragma once

#include "Camera.h"
#include "TransformGizmo.h"
#include "Vulkan/Image.h"
#include "mesh/Handle.h"
#include "numeric/vec2.h"
#include "numeric/vec4.h"

#include "entt_fwd.h"

#include <filesystem>
#include <memory>
#include <set>
#include <unordered_set>

using uint = uint32_t;
namespace fs = std::filesystem;

struct Path {
    fs::path Value;
};

struct World {
    const vec3 Origin{0, 0, 0}, Up{0, 1, 0};
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

enum class InteractionMode {
    Object, // Select objects
    Edit, // Select individual mesh elements (vertices, edges, faces)
    // Available when any `Excitable` objects are present.
    // Mouse-down on an excitable object adds an `ExcitedVertex` component to the object entity, and mouse-up removes it.
    Excite,
};

enum class ViewportShadingMode {
    Wireframe,
    Solid,
};

enum class ColorMode {
    Mesh,
    Normals,
};

inline std::string to_string(InteractionMode mode) {
    switch (mode) {
        case InteractionMode::Object: return "Object";
        case InteractionMode::Edit: return "Edit";
        case InteractionMode::Excite: return "Excite";
    }
}

struct MeshCreateInfo {
    std::string Name{};
    Transform Transform{};

    enum class SelectBehavior {
        Exclusive,
        Additive,
        None,
    };
    SelectBehavior Select{SelectBehavior::Exclusive};
    bool Visible{true};
};

static constexpr Camera CreateDefaultCamera() { return {{0, 0, 2}, {0, 0, 0}, glm::radians(60.f), 0.01, 100}; }

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

struct SvgResource;
struct ScenePipelines;
struct SceneUniqueBuffers;

struct SceneVulkanResources {
    vk::Instance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::Device Device;
    uint32_t QueueFamily;
    vk::Queue Queue;
    vk::DescriptorPool DescriptorPool;
};

struct Mesh;
struct Excitable;

struct Scene {
    Scene(SceneVulkanResources, entt::registry &);
    ~Scene();

    void LoadIcons(vk::Device);

    World World{};

    entt::entity AddMesh(Mesh &&, MeshCreateInfo = {});
    entt::entity AddMesh(const fs::path &, MeshCreateInfo = {});

    entt::entity Duplicate(entt::entity, std::optional<MeshCreateInfo> = {});
    entt::entity DuplicateLinked(entt::entity, std::optional<MeshCreateInfo> = {});

    void ReplaceMesh(entt::entity, Mesh &&);
    void ClearMeshes();

    void Destroy(entt::entity);

    void SetVisible(entt::entity, bool);

    entt::entity GetMeshEntity(entt::entity) const;
    entt::entity GetActiveMeshEntity() const;
    void Select(entt::entity);
    void ToggleSelected(entt::entity);

    // Actions on selected entities
    void Duplicate();
    void DuplicateLinked();
    void Delete();

    vk::Extent2D GetExtent() const { return Extent; }
    vk::ImageView GetViewportImageView() const;

    // Handle mouse/keyboard interactions.
    void Interact();
    ray GetMouseWorldRay() const; // World space ray from the mouse into the scene.

    // Renders to an image view that can be accessed with `GetViewportImageView()`.
    // The extent of the resolve image can be found with `GetExtent()` after the call,
    // and it will be equal to the dimensions of `GetContentRegionAvail()` at the beginning of the call.
    // Returns true if the scene was updated, which can happen when the window size or background color changes.
    bool RenderViewport(); // The main scene viewport (not to be confused with ImGui viewports)
    // The "overlay" is everything drawn ontop of the viewport with ImGui, independent of the main scene vulkan pipeline:
    // - Orientation/Transform gizmos
    // - Active object center-dot
    void RenderOverlay();
    void RenderControls();

    mvk::ImageResource RenderBitmapToImage(std::span<const std::byte> data, uint width, uint height) const;

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

    std::string DebugBufferHeapUsage() const;

private:
    SceneVulkanResources Vk;
    entt::registry &R;
    vk::UniqueCommandPool CommandPool;
    vk::UniqueCommandBuffer RenderCommandBuffer;
    vk::UniqueFence RenderFence, TransferFence;

    Camera Camera{CreateDefaultCamera()};
    Lights Lights{{1, 1, 1, 0.1}, {1, 1, 1, 0.15}, {-1, -1, -1}};

    vec4 EdgeColor{1, 1, 1, 1}; // Used for line mode.
    vec4 MeshEdgeColor{0, 0, 0, 1}; // Used for faces mode.

    ViewportShadingMode ViewportShading{ViewportShadingMode::Solid};
    bool SmoothShading{false}; // true = vertex normals (smooth), false = face normals (flat)
    ColorMode ColorMode{ColorMode::Mesh};

    std::set<InteractionMode> InteractionModes{InteractionMode::Object, InteractionMode::Edit};
    InteractionMode InteractionMode{InteractionMode::Object};
    he::AnyHandle EditingHandle{he::Element::Face};
    vec2 AccumulatedWrapMouseDelta{0, 0};

    vk::Extent2D Extent;
    vk::ClearColorValue BackgroundColor{0.25, 0.25, 0.25, 1.f};
    struct Colors {
        vec4 Active{1, 0.627, 0.157, 1}; // Blender's default `Preferences->Themes->3D Viewport->Active Object`.
        vec4 Selected{0.929, 0.341, 0, 1}; // Blender's default `Preferences->Themes->3D Viewport->Object Selected`.
    };

    Colors Colors;
    uint SilhouetteEdgeWidth{2};

    std::unique_ptr<ScenePipelines> Pipelines;
    std::unique_ptr<SceneUniqueBuffers> UniqueBuffers;

    enum class SelectionMode { Click,
                               Box };
    SelectionMode SelectionMode{SelectionMode::Click};
    std::optional<vec2> BoxSelectStart, BoxSelectEnd;

    struct TransformGizmoState {
        TransformGizmo::Config Config;
        TransformGizmo::Mode Mode;
    };
    TransformGizmoState MGizmo;
    std::optional<TransformGizmo::TransformType> StartScreenTransform;
    bool TransformModePillsHovered{false};

    struct TransformIcons {
        std::unique_ptr<SvgResource> Select, SelectBox, Move, Rotate, Scale, Universal;
    };
    TransformIcons Icons;

    bool ShowGrid{true};
    // Selected entity render settings
    bool ShowBoundingBoxes{false}, ShowBvhBoxes{false};
    static inline const std::vector<he::Element> NormalElements{he::Element::Vertex, he::Element::Face};
    std::unordered_set<he::Element> ShownNormalElements{};

    bool CommandBufferDirty{false};

    void SetInteractionMode(::InteractionMode);
    void SetEditingHandle(he::AnyHandle);

    void RenderEntityControls(entt::entity);
    void RenderEntitiesTable(std::string name, entt::entity parent);

    // VK buffer update methods
    void UpdateTransformBuffers();
    void UpdateEdgeColors();
    void UpdateHighlightedVertices(entt::entity, const Excitable &);
    void UpdateEntitySelectionOverlays(entt::entity);

    void WaitFor(vk::Fence) const;
};
