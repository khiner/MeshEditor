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

struct Scene;

struct Transform {
    glm::mat4 Model;
    glm::mat4 View;
    glm::mat4 Projection;
};

struct Light {
    glm::vec4 ColorAndAmbient; // RGB = color, A = ambient intensity. This is done for 16-byte alignment.
    glm::vec3 Direction;
};

struct Gizmo;

struct Buffer {
    vk::BufferUsageFlags Usage{};
    vk::DeviceSize Size{0};

    // GPU buffer.
    vk::UniqueBuffer Buffer{};
    vk::UniqueDeviceMemory Memory{};

    // Host staging buffer, used to transfer data to the GPU.
    vk::UniqueBuffer StagingBuffer{};
    vk::UniqueDeviceMemory StagingMemory{};
};

struct ImageResource {
    // The `image` in the view info is overwritten.
    void Create(const VulkanContext &, vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Dereference forwards to the image.
    const vk::Image &operator*() const { return *Image; }

    vk::UniqueImage Image;
    vk::UniqueImageView View;
    vk::UniqueDeviceMemory Memory;
};

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

struct Geometry;

struct GeometryInstance {
    // todo line mode buffers can share vertex buffers with smooth mode.
    struct Buffers {
        Buffer VertexBuffer{vk::BufferUsageFlagBits::eVertexBuffer};
        Buffer IndexBuffer{vk::BufferUsageFlagBits::eIndexBuffer};
    };

    GeometryInstance(const Scene &, Geometry &&);

    const Buffers &GetBuffers(GeometryMode mode) const { return BuffersForMode.at(mode); }

    void SetEdgeColor(const glm::vec4 &);

    const Scene &S;

private:
    std::unique_ptr<Geometry> G;
    std::unordered_map<GeometryMode, Buffers> BuffersForMode;

    void CreateOrUpdateBuffers(GeometryMode);
};

struct Scene {
    Scene(const VulkanContext &);
    ~Scene();

    VkSampler GetTextureSampler() const { return TextureSampler.get(); }
    VkImageView GetResolveImageView() const { return ResolveImage.View.get(); }

    // Returns true if the scene was updated.
    bool Render(uint width, uint height, const vk::ClearColorValue &bg_color);
    void RenderGizmo();
    void RenderControls();

    void RecompileShaders();

    Transform GetTransform() const;
    void UpdateTransform();
    void UpdateLight();

    void CreateOrUpdateBuffer(Buffer &buffer, const void *data, bool force_recreate = false) const;

    void UpdateGeometryEdgeColors() {
        const auto &edge_color = Mode == RenderMode::FacesAndEdges ? MeshEdgeColor : EdgeColor;
        for (auto &geometry : Geometries) {
            geometry.SetEdgeColor(edge_color);
            geometry.GetBuffers(GeometryMode::Edges);
        }
    }

    const VulkanContext &VC;

    Camera Camera{{0, 0, 2}, Origin, 45, 0.1, 100};
    Light Light{{1, 1, 1, 0.6}, {0, 0, -1}}; // White light coming from the Z direction.

    RenderMode Mode{RenderMode::FacesAndEdges};

    glm::vec4 EdgeColor{1, 1, 1, 1}; // Used for line mode.
    glm::vec4 MeshEdgeColor{0, 0, 0, 1}; // Used for mesh mode.

    const uint FramebufferCount{1};
    vk::SampleCountFlagBits MsaaSamples;

    vk::UniqueFramebuffer Framebuffer;
    vk::UniqueRenderPass RenderPass;

    Buffer TransformBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Transform)};
    Buffer LightBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Light)};

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

    vk::Extent2D Extent;
    vk::ClearColorValue BackgroundColor;
    vk::UniqueCommandPool CommandPool;
    std::vector<vk::UniqueCommandBuffer> CommandBuffers;
    std::vector<vk::UniqueCommandBuffer> TransferCommandBuffers;
    vk::UniqueFence RenderFence;

    // We use three images in the render pass:
    // 1) Perform depth testing.
    // 2) Render into a multisampled offscreen image.
    // 3) Resolve into a single-sampled resolve image.
    // All images are referenced by the framebuffer and thus must be kept in memory.
    ImageResource DepthImage, OffscreenImage, ResolveImage;

    std::unique_ptr<FillShaderPipeline> FillShaderPipeline;
    std::unique_ptr<LineShaderPipeline> LineShaderPipeline;

    vk::UniqueSampler TextureSampler;

    std::unique_ptr<Gizmo> Gizmo;
    glm::mat4 ModelTransform{1};

    std::vector<GeometryInstance> Geometries;
};
