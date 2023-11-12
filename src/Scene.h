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

struct Scene;

struct ShaderPipeline {
    // Paths are relative to the `Shaders` directory.
    ShaderPipeline(
        const Scene &,
        const fs::path &vert_shader_path, const fs::path &frag_shader_path,
        vk::PolygonMode polygon_mode = vk::PolygonMode::eFill,
        vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList
    );
    virtual ~ShaderPipeline() = default;

    void CompileShaders();

    const Scene &S;

    fs::path VertexShaderPath, FragmentShaderPath;
    vk::PolygonMode PolygonMode;
    vk::PrimitiveTopology Topology;
    vk::UniqueDescriptorSetLayout DescriptorSetLayout;
    vk::UniqueDescriptorSet DescriptorSet;
    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline Pipeline;
};

struct FillShaderPipeline : public ShaderPipeline {
    FillShaderPipeline(const Scene &, const fs::path &vert_shader_path, const fs::path &frag_shader_path);
};

struct LineShaderPipeline : public ShaderPipeline {
    LineShaderPipeline(const Scene &, const fs::path &vert_shader_path, const fs::path &frag_shader_path);
};

struct GridShaderPipeline : public ShaderPipeline {
    GridShaderPipeline(const Scene &, const fs::path &vert_shader_path, const fs::path &frag_shader_path);
};

struct GeometryInstance;

struct Scene {
    Scene(const VulkanContext &);
    ~Scene();

    VkSampler GetTextureSampler() const { return TextureSampler.get(); }
    VkImageView GetResolveImageView() const { return ResolveImage.View.get(); }
    const vk::Extent2D &GetExtent() const { return Extent; }

    // Renders to a texture sampler and image view that can be accessed with `GetTextureSampler()` and `GetResolveImageView()`.
    // The extent of the resolve image can be found with `GetExtent()` after the call,
    // and it will be equal to the dimensions of `GetContentRegionAvail()` at the beginning of the call.
    // Returns true if the scene was updated, which can happen when the window size or background color changes.
    bool Render();
    void RenderGizmo();
    void RenderControls();

    void RecompileShaders();

    void UpdateTransform(); // Updates buffers that depend on model/view/transform. (Does not submit command buffer.)

    void UpdateGeometryEdgeColors();

    const VulkanContext &VC;

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

private:
    // Recreates transform, render images (see below) and framebuffer based on the new extent.
    // These are then reused by future renders that don't change the extent.
    void SetExtent(vk::Extent2D);
    void RecordCommandBuffer();
    void SubmitCommandBuffer(vk::Fence fence = nullptr) const;

    inline ShaderPipeline *GetShaderPipeline() const {
        switch (Mode) {
            case RenderMode::Faces:
            case RenderMode::Smooth:
                return FillShaderPipeline.get();
            case RenderMode::Edges:
                return LineShaderPipeline.get();
            default:
                throw std::runtime_error("Invalid render mode.");
        }
    }

    // Rendering of grid lines derives from a plane at z = 0.
    // See `GridLines.vert` and `GridLines.frag` for details.
    // Based on https://asliceofrendering.com/scene%20helper/2020/01/05/InfiniteGrid.
    void CreateGrid();

    vk::Extent2D Extent;
    vk::ClearColorValue BackgroundColor;

    // We use three images in the render pass:
    // 1) Perform depth testing.
    // 2) Render into a multisampled offscreen image.
    // 3) Resolve into a single-sampled resolve image.
    // All images are referenced by the framebuffer and thus must be kept in memory.
    ImageResource DepthImage, OffscreenImage, ResolveImage;

    std::unique_ptr<FillShaderPipeline> FillShaderPipeline;
    std::unique_ptr<LineShaderPipeline> LineShaderPipeline;
    std::unique_ptr<GridShaderPipeline> GridShaderPipeline;

    vk::UniqueSampler TextureSampler;

    std::unique_ptr<Gizmo> Gizmo;

    std::vector<std::unique_ptr<GeometryInstance>> GeometryInstances;
    std::unique_ptr<GeometryInstance> Grid;
};
