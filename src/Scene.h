#pragma once

#include "Vertex.h"
#include "VulkanContext.h"

struct Scene;

struct ShaderPipeline {
    ShaderPipeline(const Scene &);
    ~ShaderPipeline() = default;

    void CompileShaders();
    void CreateVertexBuffers(const std::vector<Vertex2D> &vertices);

    const Scene &S;

    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline Pipeline;

    vk::UniqueBuffer VertexBuffer;
    vk::UniqueDeviceMemory VertexBufferMemory;
    std::vector<vk::UniqueCommandBuffer> VertexTransferCommandBuffers;
};

struct Scene {
    Scene(const VulkanContext &);
    ~Scene() = default; // Using unique handles, so no need to manually destroy anything.

    // Returns true if the scene was updated.
    bool Render(uint width, uint height, const vk::ClearColorValue &bg_color);

    void CompileShaders();

    const VulkanContext &VC;

    const uint FrameBufferCount{1};
    vk::SampleCountFlagBits MsaaSamples;
    vk::Extent2D Extent;

    vk::UniqueRenderPass RenderPass;
    vk::UniqueCommandPool CommandPool;
    std::vector<vk::UniqueCommandBuffer> CommandBuffers;

    // The scene is rendered to an offscreen image and then resolved to this image using MSAA.
    vk::UniqueImage ResolveImage;
    vk::UniqueImageView ResolveImageView;
    vk::UniqueDeviceMemory ResolveImageMemory;
    vk::UniqueSampler TextureSampler;

    ShaderPipeline ShaderPipeline;

    bool HasNewShaders{true};
};
