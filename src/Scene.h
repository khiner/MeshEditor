#pragma once

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "Vertex.h"
#include "VulkanContext.h"

struct Scene;

static const glm::mat4 I{1};
static const glm::vec3 Origin{0, 0, 0};
static const glm::vec3 Up{0, 1, 0};

static const glm::vec4 White{1, 1, 1, 1};

struct Transform {
    glm::mat4 Model;
    glm::mat4 View;
    glm::mat4 Projection;
};

struct Camera {
    inline glm::mat4 GetViewMatrix() const { return glm::lookAt(Position, Target, Up); }
    inline glm::mat4 GetProjectionMatrix(float aspect_ratio) const { return glm::perspective(glm::radians(FieldOfView), aspect_ratio, NearClip, FarClip); }

    glm::vec3 Position;
    glm::vec3 Target;
    float FieldOfView;
    float NearClip, FarClip;
};

struct Light {
    glm::vec3 Direction;
    glm::vec3 Color;
};

struct ShaderPipeline {
    ShaderPipeline(const Scene &);
    ~ShaderPipeline() = default;

    void CompileShaders();

    void CreateBuffer(const void *data, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::UniqueBuffer &buffer_out, vk::UniqueDeviceMemory &buffer_memory_out);

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

    vk::UniqueBuffer IndexBuffer;
    vk::UniqueDeviceMemory IndexBufferMemory;

    vk::UniqueBuffer TransformBuffer;
    vk::UniqueDeviceMemory TransformBufferMemory;

    vk::UniqueBuffer LightBuffer;
    vk::UniqueDeviceMemory LightBufferMemory;

    std::vector<vk::UniqueCommandBuffer> TransferCommandBuffers;
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

    Camera Camera{{0, 0, 1}, Origin, 45, 0.1, 100};
    Light Light{{0, 0, -1}, White}; // White light coming from the Z direction.

    ShaderPipeline ShaderPipeline;
    bool HasNewShaders{true};
};
