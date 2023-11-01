#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "Camera.h"
#include "Vertex.h"
#include "VulkanContext.h"

struct Scene;

static const glm::vec4 White{1, 1, 1, 1};

struct Transform {
    glm::mat4 Model;
    glm::mat4 View;
    glm::mat4 Projection;
};

struct Light {
    glm::vec3 Direction;
    glm::vec3 Color;
};

struct Gizmo;

struct ShaderPipeline {
    ShaderPipeline(const Scene &);
    ~ShaderPipeline() = default;

    void CompileShaders();

    void CreateBuffer(const void *data, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::UniqueBuffer &buffer_out, vk::UniqueDeviceMemory &buffer_memory_out, vk::UniqueBuffer &staging_buffer_out, vk::UniqueDeviceMemory &staging_memory_out);
    void UpdateBuffer(const void *data, vk::DeviceSize size, vk::UniqueBuffer &buffer_out, vk::UniqueBuffer &staging_buffer_out, vk::UniqueDeviceMemory &staging_buffer_memory_out);

    void CreateVertexBuffers(const std::vector<Vertex3D> &);
    void CreateIndexBuffers(const std::vector<uint16_t> &);
    void CreateTransformBuffers(const Transform &);
    void CreateLightBuffers(const Light &);

    const Scene &S;

    vk::UniqueDescriptorSetLayout DescriptorSetLayout;
    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline Pipeline;

    vk::UniqueDescriptorSet DescriptorSet;

    vk::UniqueBuffer VertexBuffer;
    vk::UniqueDeviceMemory VertexBufferMemory;
    vk::UniqueBuffer VertexStagingBuffer;
    vk::UniqueDeviceMemory VertexStagingBufferMemory;

    vk::UniqueBuffer IndexBuffer;
    vk::UniqueDeviceMemory IndexBufferMemory;
    vk::UniqueBuffer IndexStagingBuffer;
    vk::UniqueDeviceMemory IndexStagingBufferMemory;

    vk::UniqueBuffer TransformBuffer;
    vk::UniqueDeviceMemory TransformBufferMemory;
    vk::UniqueBuffer TransformStagingBuffer;
    vk::UniqueDeviceMemory TransformStagingBufferMemory;

    vk::UniqueBuffer LightBuffer;
    vk::UniqueBuffer LightStagingBuffer;
    vk::UniqueDeviceMemory LightBufferMemory;
    vk::UniqueDeviceMemory LightStagingBufferMemory;

    std::vector<vk::UniqueCommandBuffer> TransferCommandBuffers;
};

struct Scene {
    Scene(const VulkanContext &);
    ~Scene();

    // Returns true if the scene was updated.
    bool Render(uint width, uint height, const vk::ClearColorValue &bg_color);
    void RenderGizmo();
    void RenderControls();

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

    Camera Camera{{0, 0, 1}, Origin, 45, 0.1, 100};
    Light Light{{0, 0, -1}, White}; // White light coming from the Z direction.

    ShaderPipeline ShaderPipeline;

    std::unique_ptr<Gizmo> Gizmo;
    bool Dirty{true};
};
