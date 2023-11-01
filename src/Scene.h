#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "Camera.h"
#include "Vertex.h"
#include "VulkanContext.h"

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

struct ShaderPipeline {
    ShaderPipeline(const Scene &);
    ~ShaderPipeline() = default;

    void CompileShaders();

    void CreateOrUpdateBuffer(Buffer &buffer, const void *data);

    const Scene &S;

    vk::UniqueDescriptorSetLayout DescriptorSetLayout;
    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline Pipeline;

    vk::UniqueDescriptorSet DescriptorSet;

    Buffer VertexBuffer{vk::BufferUsageFlagBits::eVertexBuffer};
    Buffer IndexBuffer{vk::BufferUsageFlagBits::eIndexBuffer};
    Buffer TransformBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Transform)};
    Buffer LightBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Light)};

    std::vector<vk::UniqueCommandBuffer> TransferCommandBuffers;
};

struct Scene {
    Scene(const VulkanContext &);
    ~Scene();

    VkSampler GetTextureSampler() const { return TextureSampler.get(); }
    VkImageView GetResolveImageView() const { return ResolveImageView.get(); }

    // Returns true if the scene was updated.
    bool Render(uint width, uint height, const vk::ClearColorValue &bg_color);
    void RenderGizmo();
    void RenderControls();

    void CompileShaders();

    Transform GetTransform() const;
    void UpdateTransform();
    void UpdateLight();

    const VulkanContext &VC;

    Camera Camera{{0, 0, 2}, Origin, 45, 0.1, 100};
    Light Light{{1, 1, 1, 0.6}, {0, 0, -1}}; // White light coming from the Z direction.

    const uint FramebufferCount{1};
    vk::SampleCountFlagBits MsaaSamples;
    vk::Extent2D Extent;

    vk::UniqueFramebuffer Framebuffer;
    vk::UniqueRenderPass RenderPass;
    vk::UniqueCommandPool CommandPool;
    std::vector<vk::UniqueCommandBuffer> CommandBuffers;

private:
    // Recreates transform, render images (see below) and framebuffer based on the new extent.
    // These are then reused by future renders that don't change the extent.
    void SetExtent(vk::Extent2D);

    // We use three images in the render pass:
    // 1) Perform depth testing.
    // 2) Render into a multisampled offscreen image.
    // 3) Resolve into a single-sampled resolve image.
    vk::UniqueImage DepthImage;
    vk::UniqueImageView DepthImageView;
    vk::UniqueDeviceMemory DepthImageMemory;

    vk::UniqueImage OffscreenImage;
    vk::UniqueImageView OffscreenImageView;
    vk::UniqueDeviceMemory OffscreenImageMemory;

    vk::UniqueImage ResolveImage;
    vk::UniqueImageView ResolveImageView;
    vk::UniqueDeviceMemory ResolveImageMemory;

    ShaderPipeline ShaderPipeline;
    vk::UniqueSampler TextureSampler;

    std::unique_ptr<Gizmo> Gizmo;
    glm::mat4 ModelTransform{1};

    bool Dirty{true};
};
