#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "Camera.h"
#include "Geometry/GeometryMode.h"
#include "RenderMode.h"
#include "Vertex.h"
#include "VulkanContext.h"

#include <filesystem>

namespace fs = std::filesystem;

enum class SelectionMode {
    None,
    Face,
    Vertex,
    Edge,
};

struct Transform {
    glm::mat4 Model;
    glm::mat4 View;
    glm::mat4 Projection;
};

struct ViewProjection {
    glm::mat4 View;
    glm::mat4 Projection;
};

struct ViewProjNearFar {
    glm::mat4 View;
    glm::mat4 Projection;
    float Near;
    float Far;
};

struct Light {
    glm::vec4 ColorAndAmbient; // RGB = color, A = ambient intensity. This is done for 16-byte alignment.
    glm::vec3 Direction;
};

struct Gizmo;

struct ImageResource {
    // The `image` in the view info is overwritten.
    void Create(const VulkanContext &, vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Dereference forwards to the image.
    const vk::Image &operator*() const { return *Image; }

    vk::UniqueImage Image;
    vk::UniqueImageView View;
    vk::UniqueDeviceMemory Memory;
};

using ShaderType = vk::ShaderStageFlagBits;
using ShaderPaths = std::unordered_map<ShaderType, fs::path>;
using ShaderModules = std::unordered_map<ShaderType, vk::UniqueShaderModule>;

struct Shaders {
    Shaders(ShaderPaths &&paths) : Paths(std::move(paths)) {}
    std::vector<vk::PipelineShaderStageCreateInfo> CompileAll(const vk::UniqueDevice &); // Populates `Modules`.
    std::vector<uint> Compile(ShaderType) const;

    inline static const std::vector AllTypes{ShaderType::eVertex, ShaderType::eFragment};
    ShaderPaths Paths; // Paths are relative to the `Shaders` directory.
    ShaderModules Modules;
};

struct ShaderPipeline {
    ShaderPipeline(
        const VulkanContext &, Shaders &&,
        vk::PolygonMode polygon_mode = vk::PolygonMode::eFill,
        vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList,
        bool test_depth = true, bool write_depth = true,
        vk::SampleCountFlagBits msaa_samples = vk::SampleCountFlagBits::e1
    );
    virtual ~ShaderPipeline() = default;

    void Compile(const vk::UniqueRenderPass &);

    const VulkanContext &VC;

    Shaders Shaders;
    vk::PolygonMode PolygonMode;
    vk::PrimitiveTopology Topology;
    bool TestDepth, WriteDepth;
    vk::SampleCountFlagBits MsaaSamples;

    vk::UniqueDescriptorSetLayout DescriptorSetLayout;
    vk::UniqueDescriptorSet DescriptorSet;
    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline Pipeline;
};

struct GeometryInstance;

struct Scene {
    Scene(const VulkanContext &);
    ~Scene();

    const vk::Extent2D &GetExtent() const { return Extent; }
    vk::SampleCountFlagBits GetMsaaSamples() const { return MsaaSamples; }
    vk::Sampler GetTextureSampler() const { return TextureSampler.get(); }
    vk::ImageView GetResolveImageView() const { return ResolveImage.View.get(); }

    // Renders to a texture sampler and image view that can be accessed with `GetTextureSampler()` and `GetResolveImageView()`.
    // The extent of the resolve image can be found with `GetExtent()` after the call,
    // and it will be equal to the dimensions of `GetContentRegionAvail()` at the beginning of the call.
    // Returns true if the scene was updated, which can happen when the window size or background color changes.
    bool Render();
    void RenderGizmo();
    void RenderControls();

    void CompileShaders(); // Doesn't submit command buffer.

    void UpdateTransform(); // Updates buffers that depend on model/view/transform. (Does not submit command buffer.)
    void UpdateGeometryEdgeColors();

    const VulkanContext &VC;

private:
    // Recreates transform, render images (see below) and framebuffer based on the new extent.
    // These are then reused by future renders that don't change the extent.
    void SetExtent(vk::Extent2D);
    void RecordCommandBuffer();
    void SubmitCommandBuffer(vk::Fence fence = nullptr) const;

    Camera Camera{{0, 0, 2}, Origin, 60, 0.1, 100};
    Light Light{{1, 1, 1, 0.6}, {0, 0, -1}}; // White light coming from the Z direction.

    glm::vec4 EdgeColor{1, 1, 1, 1}; // Used for line mode.
    glm::vec4 MeshEdgeColor{0, 0, 0, 1}; // Used for mesh mode.

    RenderMode Mode{RenderMode::FacesAndEdges};
    SelectionMode SelectionMode{SelectionMode::None};

    vk::SampleCountFlagBits MsaaSamples;

    vk::UniqueFramebuffer Framebuffer;
    vk::UniqueRenderPass RenderPass;

    VulkanBuffer ViewProjectionBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ViewProjection)};
    VulkanBuffer ViewProjNearFarBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ViewProjNearFar)};
    VulkanBuffer TransformBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Transform)};
    VulkanBuffer LightBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Light)};

    vk::Extent2D Extent;
    vk::ClearColorValue BackgroundColor;

    // We use three images in the render pass:
    // 1) Perform depth testing.
    // 2) Render into a multisampled offscreen image.
    // 3) Resolve into a single-sampled resolve image.
    // All images are referenced by the framebuffer and thus must be kept in memory.
    ImageResource DepthImage, OffscreenImage, ResolveImage;
    vk::UniqueSampler TextureSampler;

    enum class ShaderPipelineType {
        Fill,
        Line,
        Grid,
    };
    std::unordered_map<ShaderPipelineType, std::unique_ptr<ShaderPipeline>> ShaderPipelines;

    std::unique_ptr<Gizmo> Gizmo;
    std::vector<std::unique_ptr<GeometryInstance>> GeometryInstances;

    bool ShowGrid{true};
};
