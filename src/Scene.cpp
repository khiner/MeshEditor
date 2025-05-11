#include "Scene.h"
#include "Widgets.h" // imgui

#include "Excitable.h"
#include "OrientationGizmo.h"
#include "Registry.h"
#include "Scale.h"
#include "mesh/Arrow.h"
#include "mesh/Primitives.h"
#include "numeric/mat3.h"

#include <entt/entity/registry.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <format>
#include <ranges>

using std::ranges::find, std::ranges::find_if, std::ranges::to;
using std::views::transform;

namespace {
struct SceneNode {
    entt::entity Parent = entt::null;
    std::vector<entt::entity> Children;
    // `ModelIndices` maps entities to their index in the models buffer. Includes parent.
    // It's only present in parent nodes.
    // This enables contiguous storage of models in the buffer, with erases and appends but no inserts.
    std::unordered_map<entt::entity, uint> ModelIndices;
};

entt::entity GetParentEntity(const entt::registry &r, entt::entity entity) {
    if (entity == entt::null) return entt::null;

    if (const auto *node = r.try_get<SceneNode>(entity)) {
        return node->Parent == entt::null ? entity : GetParentEntity(r, node->Parent);
    }
    return entity;
}

template<typename Component>
entt::entity FindEntity(const entt::registry &registry) {
    auto all_active = registry.view<Component>();
    assert(all_active.size() <= 1);
    return all_active.empty() ? entt::null : *all_active.begin();
}
} // namespace

entt::entity Scene::GetParentEntity(entt::entity entity) const { return ::GetParentEntity(R, entity); }
const Mesh &Scene::GetActiveMesh() const { return R.get<Mesh>(GetParentEntity(FindEntity<Active>(R))); }
entt::entity Scene::GetActiveEntity() const { return FindEntity<Active>(R); }

void Scene::SetActive(entt::entity entity) {
    if (FindEntity<Active>(R) == entity) return;

    R.clear<Active, Selected>();
    if (entity != entt::null) {
        R.emplace_or_replace<Active>(entity);
        R.emplace_or_replace<Selected>(entity);
    }
    InvalidateCommandBuffer();
}
void Scene::ToggleSelected(entt::entity entity) {
    if (entity == entt::null) return;

    if (R.all_of<Selected>(entity)) {
        R.remove<Selected>(entity);
    } else {
        R.emplace_or_replace<Selected>(entity);
    }
    InvalidateCommandBuffer();
}

using MeshBuffers = std::unordered_map<MeshElement, mvk::RenderBuffers>;
struct MeshVkData {
    std::unordered_map<entt::entity, MeshBuffers> Main, NormalIndicators;
    std::unordered_map<entt::entity, mvk::Buffer> Models;
    std::unordered_map<entt::entity, mvk::RenderBuffers> Boxes, BvhBoxes;
};

std::vector<Vertex3D> CreateBoxVertices(const BBox &box, const vec4 &color) {
    return box.Corners() |
        // Normals don't matter for wireframes.
        transform([&color](const auto &corner) { return Vertex3D(corner, vec3{}, color); }) |
        to<std::vector>();
}

const std::vector AllNormalElements{MeshElement::Vertex, MeshElement::Face};

const vk::ClearColorValue Transparent{0, 0, 0, 0};

namespace Format {
const auto Vec3 = vk::Format::eR32G32B32Sfloat;
const auto Vec4 = vk::Format::eR32G32B32A32Sfloat;
} // namespace Format

namespace {
void UpdateModel(entt::registry &r, entt::entity entity, vec3 position, quat rotation, vec3 scale) {
    r.emplace_or_replace<Position>(entity, position);
    r.emplace_or_replace<Rotation>(entity, rotation);

    // Frozen entities can't have their scale changed.
    if (!r.all_of<Frozen>(entity)) r.emplace_or_replace<Scale>(entity, scale);
    else scale = r.all_of<Scale>(entity) ? r.get<Scale>(entity).Value : vec3{1};

    r.emplace_or_replace<Model>(entity, glm::translate({1}, position) * glm::mat4_cast(glm::normalize(rotation)) * glm::scale({1}, scale));
}

vk::SampleCountFlagBits GetMaxUsableSampleCount(const vk::PhysicalDevice pd) {
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

vk::PipelineVertexInputStateCreateInfo CreateVertexInputState() {
    static const std::vector<vk::VertexInputBindingDescription> bindings{
        {0, sizeof(Vertex3D), vk::VertexInputRate::eVertex},
        {1, 2 * sizeof(mat4), vk::VertexInputRate::eInstance},
    };
    static const std::vector<vk::VertexInputAttributeDescription> attrs{
        {0, 0, Format::Vec3, offsetof(Vertex3D, Position)},
        {1, 0, Format::Vec3, offsetof(Vertex3D, Normal)},
        {2, 0, Format::Vec4, offsetof(Vertex3D, Color)},
        // Model mat4, one vec4 per row
        {3, 1, Format::Vec4, 0},
        {4, 1, Format::Vec4, sizeof(vec4)},
        {5, 1, Format::Vec4, 2 * sizeof(vec4)},
        {6, 1, Format::Vec4, 3 * sizeof(vec4)},
        // Inverse model mat4, one vec4 per row
        {7, 1, Format::Vec4, 4 * sizeof(vec4)},
        {8, 1, Format::Vec4, 5 * sizeof(vec4)},
        {9, 1, Format::Vec4, 6 * sizeof(vec4)},
        {10, 1, Format::Vec4, 7 * sizeof(vec4)},
    };
    return {{}, bindings, attrs};
}
} // namespace

void PipelineRenderer::CompileShaders() {
    for (auto &shader_pipeline : std::views::values(ShaderPipelines)) shader_pipeline.Compile(*RenderPass);
}

void PipelineRenderer::Render(vk::CommandBuffer cb, SPT spt, const mvk::Buffer &vertices, const mvk::Buffer &indices, const mvk::Buffer &models, std::optional<uint> model_index) const {
    const auto &shader_pipeline = ShaderPipelines.at(spt);
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *shader_pipeline.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shader_pipeline.PipelineLayout, 0, *shader_pipeline.DescriptorSet, {});

    // Bind buffers
    static const vk::DeviceSize vertex_buffer_offsets[] = {0}, models_buffer_offsets[] = {0};
    cb.bindVertexBuffers(0, {vertices.DeviceBuffer}, vertex_buffer_offsets);
    cb.bindIndexBuffer(indices.DeviceBuffer, 0, vk::IndexType::eUint32);
    cb.bindVertexBuffers(1, {models.DeviceBuffer}, models_buffer_offsets);

    // Draw
    const uint index_count = indices.Size / sizeof(uint);
    const uint first_instance = model_index.value_or(0);
    const uint instance_count = model_index.has_value() ? 1 : models.Size / sizeof(Model);
    cb.drawIndexed(index_count, instance_count, 0, 0, first_instance);
}

void PipelineRenderer::Render(vk::CommandBuffer cb, SPT spt, const mvk::RenderBuffers &render_buffers, const mvk::Buffer &models, std::optional<uint> model_index) const {
    Render(cb, spt, render_buffers.Vertices, render_buffers.Indices, models, model_index);
}

namespace {
PipelineRenderer MainPipelineRenderer(vk::Device d, vk::PhysicalDevice pd, vk::DescriptorPool descriptor_pool) {
    const auto MsaaSamples = GetMaxUsableSampleCount(pd);
    const std::vector<vk::AttachmentDescription> attachments{
        // Depth attachment.
        {{}, mvk::ImageFormat::Depth, MsaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        // Multisampled offscreen image.
        {{}, mvk::ImageFormat::Color, MsaaSamples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        // Single-sampled resolve.
        {{}, mvk::ImageFormat::Color, vk::SampleCountFlagBits::e1, {}, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
    const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::AttachmentReference resolve_attachment_ref{2, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, &resolve_attachment_ref, &depth_attachment_ref};

    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(
        SPT::Fill,
        ShaderPipeline{
            d, descriptor_pool,
            Shaders{
                {{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "Lighting.frag"}}
            },
            CreateVertexInputState(),
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            CreateColorBlendAttachment(true), CreateDepthStencil(), MsaaSamples
        }
    );
    pipelines.emplace(
        SPT::Line,
        ShaderPipeline{
            d, descriptor_pool,
            Shaders{
                {{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "VertexColor.frag"}}
            },
            CreateVertexInputState(),
            vk::PolygonMode::eLine, vk::PrimitiveTopology::eLineList,
            CreateColorBlendAttachment(true), CreateDepthStencil(), MsaaSamples
        }
    );
    pipelines.emplace(
        SPT::Grid,
        ShaderPipeline{
            d, descriptor_pool,
            Shaders{
                {{ShaderType::eVertex, "GridLines.vert"}, {ShaderType::eFragment, "GridLines.frag"}}
            },
            vk::PipelineVertexInputStateCreateInfo{},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            CreateColorBlendAttachment(true), CreateDepthStencil(true, false), MsaaSamples
        }
    );
    // We render all the silhouette edge texture's pixels regardless of the tested depth value,
    // but also explicitly override the depth buffer to make edge pixels "stick" to the mesh they are derived from.
    // We should be able to just set depth testing to false and depth writing to true, but it seems that some GPUs or drivers
    // optimize out depth writes when depth testing is disabled, so instead we configure a depth test that always passes.
    pipelines.emplace(
        SPT::Texture,
        ShaderPipeline{
            d, descriptor_pool,
            Shaders{
                {{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SilhouetteEdgeTexture.frag"}}
            },
            vk::PipelineVertexInputStateCreateInfo{},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            CreateColorBlendAttachment(true), CreateDepthStencil(true, true, vk::CompareOp::eAlways), MsaaSamples
        }
    );
    pipelines.emplace(
        SPT::DebugNormals,
        ShaderPipeline{
            d, descriptor_pool,
            Shaders{
                {{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "Normals.frag"}}
            },
            CreateVertexInputState(),
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            CreateColorBlendAttachment(true), CreateDepthStencil(), MsaaSamples
        }
    );
    return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
}

PipelineRenderer SilhouettePipelineRenderer(vk::Device d, vk::DescriptorPool descriptor_pool) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Single-sampled offscreen image.
        {{}, mvk::ImageFormat::Float, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference color_attachment_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, nullptr};
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(
        SPT::Silhouette,
        ShaderPipeline{
            d, descriptor_pool,
            Shaders{
                {{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "Depth.frag"}}
            },
            CreateVertexInputState(),
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
            CreateColorBlendAttachment(false), std::nullopt, vk::SampleCountFlagBits::e1
        }
    );
    return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
}

PipelineRenderer EdgeDetectionPipelineRenderer(vk::Device d, vk::DescriptorPool descriptor_pool) {
    const std::vector<vk::AttachmentDescription> attachments{
        // Single-sampled offscreen image.
        {{}, mvk::ImageFormat::Float, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
    };
    const vk::AttachmentReference color_attachment_ref{0, vk::ImageLayout::eColorAttachmentOptimal};
    const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, nullptr};
    std::unordered_map<SPT, ShaderPipeline> pipelines;
    pipelines.emplace(
        SPT::EdgeDetection,
        ShaderPipeline{
            d, descriptor_pool,
            Shaders{
                {{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "MeshEdges.frag"}}
            },
            vk::PipelineVertexInputStateCreateInfo{},
            vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
            CreateColorBlendAttachment(false), std::nullopt, vk::SampleCountFlagBits::e1
        }
    );
    return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
}
} // namespace

namespace {
// Adapted from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2 for 64-bits.
uint64_t NextPowerOfTwo(uint64_t x) {
    if (x == 0) return 1;

    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}
} // namespace

mvk::ImageResource Scene::RenderBitmapToImage(const void *data, uint32_t width, uint32_t height) const {
    auto image = mvk::CreateImage(
        Vk.Device, Vk.PhysicalDevice,
        {{},
         vk::ImageType::e2D,
         mvk::ImageFormat::Color,
         {width, height, 1},
         1,
         1,
         vk::SampleCountFlagBits::e1,
         vk::ImageTiling::eOptimal,
         vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
         vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, mvk::ImageFormat::Color, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );
    // Write the bitmap into a staging buffer.
    const auto buffer_size = width * height * 4; // 4 bytes per pixel
    auto staging_buffer = BufferAllocator->Allocate(buffer_size, mvk::MemoryUsage::CpuOnly);
    BufferAllocator->WriteRegion(staging_buffer, data, 0, buffer_size);

    // Record commands to copy from staging buffer to Vulkan image.
    const auto &cb = *TransferCommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Transition the image layout to be ready for data transfer.
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {},
        vk::ImageMemoryBarrier{
            {}, // srcAccessMask
            vk::AccessFlagBits::eTransferWrite, // dstAccessMask
            vk::ImageLayout::eUndefined, // oldLayout
            vk::ImageLayout::eTransferDstOptimal, // newLayout
            VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
            VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
            *image.Image, // image
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} // subresourceRange
        }
    );

    // Copy buffer to image.
    cb.copyBufferToImage(
        staging_buffer, *image.Image, vk::ImageLayout::eTransferDstOptimal,
        vk::BufferImageCopy{
            0, // bufferOffset
            0, // bufferRowLength (tightly packed)
            0, // bufferImageHeight
            {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, // imageSubresource
            {0, 0, 0}, // imageOffset
            {width, height, 1} // imageExtent
        }
    );

    // Transition the image layout to be ready for shader sampling.
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {},
        vk::ImageMemoryBarrier{
            vk::AccessFlagBits::eTransferWrite, // srcAccessMask
            vk::AccessFlagBits::eShaderRead, // dstAccessMask
            vk::ImageLayout::eTransferDstOptimal, // oldLayout
            vk::ImageLayout::eShaderReadOnlyOptimal, // newLayout
            VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
            VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
            *image.Image, // image
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} // subresourceRange
        }
    );

    cb.end();
    SubmitTransfer();

    return image;
}

std::vector<vk::WriteDescriptorSet> PipelineRenderer::GetDescriptors(std::vector<ShaderBindingDescriptor> &&descriptors) const {
    std::vector<vk::WriteDescriptorSet> write_descriptor_sets;
    for (const auto &descriptor : descriptors) {
        const auto &sp = ShaderPipelines.at(descriptor.PipelineType);
        const auto *buffer_info = descriptor.BufferInfo.has_value() ? &(*descriptor.BufferInfo) : nullptr;
        const auto *image_info = descriptor.ImageInfo.has_value() ? &(*descriptor.ImageInfo) : nullptr;
        if (auto ds = sp.CreateWriteDescriptorSet(descriptor.BindingName, buffer_info, image_info)) {
            write_descriptor_sets.push_back(*ds);
        }
    }
    return write_descriptor_sets;
}

struct MainPipelineResources {
    MainPipelineResources(vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass, vk::Extent2D extent, vk::SampleCountFlagBits msaa_samples)
        : DepthImage{mvk::CreateImage(
              d, pd,
              {{},
               vk::ImageType::e2D,
               mvk::ImageFormat::Depth,
               vk::Extent3D{extent, 1},
               1,
               1,
               msaa_samples,
               vk::ImageTiling::eOptimal,
               vk::ImageUsageFlagBits::eDepthStencilAttachment,
               vk::SharingMode::eExclusive},
              {{}, {}, vk::ImageViewType::e2D, mvk::ImageFormat::Depth, {}, {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}}
          )},
          OffscreenImage{mvk::CreateImage(
              d, pd,
              {{},
               vk::ImageType::e2D,
               mvk::ImageFormat::Color,
               vk::Extent3D{extent, 1},
               1,
               1,
               msaa_samples,
               vk::ImageTiling::eOptimal,
               vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
               vk::SharingMode::eExclusive},
              {{}, {}, vk::ImageViewType::e2D, mvk::ImageFormat::Color, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
          )},
          ResolveImage{mvk::CreateImage(
              d, pd,
              {
                  {},
                  vk::ImageType::e2D,
                  mvk::ImageFormat::Color,
                  vk::Extent3D{extent, 1},
                  1,
                  1,
                  vk::SampleCountFlagBits::e1,
                  vk::ImageTiling::eOptimal,
                  vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
                  vk::SharingMode::eExclusive,
              },
              {{}, {}, vk::ImageViewType::e2D, mvk::ImageFormat::Color, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
          )} {
        const std::array image_views{*DepthImage.View, *OffscreenImage.View, *ResolveImage.View};
        Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
    }

    // Perform depth testing, render into a multisampled offscreen image, and resolve into a single-sampled image.
    mvk::ImageResource DepthImage, OffscreenImage, ResolveImage;
    vk::UniqueFramebuffer Framebuffer;
};
struct SilhouettePipelineResources {
    SilhouettePipelineResources(vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass, vk::Extent2D extent)
        : OffscreenImage{mvk::CreateImage(
              d, pd,
              {{},
               vk::ImageType::e2D,
               mvk::ImageFormat::Float,
               vk::Extent3D{extent, 1},
               1,
               1,
               vk::SampleCountFlagBits::e1,
               vk::ImageTiling::eOptimal,
               vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
               vk::SharingMode::eExclusive},
              {{}, {}, vk::ImageViewType::e2D, mvk::ImageFormat::Float, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
          )} {
        const std::array image_views{*OffscreenImage.View};
        Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
    }

    mvk::ImageResource OffscreenImage; // Single-sampled image without a depth buffer.
    vk::UniqueFramebuffer Framebuffer;
};
struct EdgeDetectionPipelineResources {
    EdgeDetectionPipelineResources(vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass, vk::Extent2D extent)
        : OffscreenImage{mvk::CreateImage(
              d, pd,
              {{},
               vk::ImageType::e2D,
               mvk::ImageFormat::Float,
               vk::Extent3D{extent, 1},
               1,
               1,
               vk::SampleCountFlagBits::e1,
               vk::ImageTiling::eOptimal,
               vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
               vk::SharingMode::eExclusive},
              {{}, {}, vk::ImageViewType::e2D, mvk::ImageFormat::Float, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
          )} {
        const std::array image_views{*OffscreenImage.View};
        Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
    }

    mvk::ImageResource OffscreenImage; // Single-sampled image without a depth buffer.
    vk::UniqueFramebuffer Framebuffer;
};

Scene::Scene(SceneVulkanResources vc, entt::registry &r)
    : Vk(vc),
      BufferAllocator(std::make_unique<mvk::BufferAllocator>(Vk.PhysicalDevice, Vk.Device, Vk.Instance)),
      R(r),
      CommandPool(Vk.Device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, Vk.QueueFamily})),
      CommandBuffer(std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1u}).front())),
      TransferCommandBuffer(std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())),
      RenderFence(Vk.Device.createFenceUnique({})),
      MeshVkData(std::make_unique<::MeshVkData>()), MainRenderer(MainPipelineRenderer(Vk.Device, Vk.PhysicalDevice, Vk.DescriptorPool)),
      SilhouetteRenderer(SilhouettePipelineRenderer(Vk.Device, Vk.DescriptorPool)), EdgeDetectionRenderer(EdgeDetectionPipelineRenderer(Vk.Device, Vk.DescriptorPool)) {
    // EnTT listeners
    R.on_construct<Excitable>().connect<&Scene::OnCreateExcitable>(*this);
    R.on_update<Excitable>().connect<&Scene::OnUpdateExcitable>(*this);
    R.on_destroy<Excitable>().connect<&Scene::OnDestroyExcitable>(*this);

    R.on_construct<ExcitedVertex>().connect<&Scene::OnCreateExcitedVertex>(*this);
    R.on_destroy<ExcitedVertex>().connect<&Scene::OnDestroyExcitedVertex>(*this);

    UpdateEdgeColors();

    TransformBuffer = std::make_unique<mvk::Buffer>(BufferAllocator->AllocateMvk(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ViewProj)));
    ViewProjNearFarBuffer = std::make_unique<mvk::Buffer>(BufferAllocator->AllocateMvk(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ViewProjNearFar)));
    UpdateTransformBuffers();

    LightsBuffer = std::make_unique<mvk::Buffer>(BufferAllocator->AllocateMvk(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Lights)));
    UpdateBuffer(*LightsBuffer, &Lights, 0, sizeof(Lights));
    SilhouetteDisplayBuffer = std::make_unique<mvk::Buffer>(BufferAllocator->AllocateMvk(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ActiveSilhouetteColor)));
    UpdateBuffer(*SilhouetteDisplayBuffer, &ActiveSilhouetteColor, 0, sizeof(ActiveSilhouetteColor));
    vk::DescriptorBufferInfo transform_buffer{TransformBuffer->DeviceBuffer, 0, VK_WHOLE_SIZE};

    Vk.Device.updateDescriptorSets(
        MainRenderer.GetDescriptors({
            {SPT::Fill, "ViewProjectionUBO", transform_buffer},
            {SPT::Fill, "LightsUBO", vk::DescriptorBufferInfo{LightsBuffer->DeviceBuffer, 0, VK_WHOLE_SIZE}},
            {SPT::Line, "ViewProjectionUBO", transform_buffer},
            {SPT::Grid, "ViewProjNearFarUBO", vk::DescriptorBufferInfo{ViewProjNearFarBuffer->DeviceBuffer, 0, VK_WHOLE_SIZE}},
            {SPT::Texture, "SilhouetteDisplayUBO", vk::DescriptorBufferInfo{SilhouetteDisplayBuffer->DeviceBuffer, 0, VK_WHOLE_SIZE}},
            {SPT::DebugNormals, "ViewProjectionUBO", transform_buffer},
        }),
        {}
    );
    Vk.Device.updateDescriptorSets(
        SilhouetteRenderer.GetDescriptors({
            {SPT::Silhouette, "ViewProjectionUBO", transform_buffer},
        }),
        {}
    );

    CompileShaders();

    AddPrimitive(Primitive::Cube, {.Select = true, .Visible = true});
}

Scene::~Scene() {}; // Using unique handles, so no need to manually destroy anything.

void Scene::UpdateHighlightedVertices(entt::entity entity, const Excitable &excitable) {
    if (auto *mesh = R.try_get<Mesh>(entity)) {
        mesh->ClearHighlights();
        if (SelectionMode == SelectionMode::Excite) {
            for (const auto vertex : excitable.ExcitableVertices) {
                mesh->HighlightVertex(Mesh::VH(vertex));
            }
        }
        UpdateRenderBuffers(entity);
    }
}

void Scene::OnCreateExcitable(entt::registry &r, entt::entity entity) {
    SelectionModes.insert(SelectionMode::Excite);
    SetSelectionMode(SelectionMode::Excite);
    UpdateHighlightedVertices(entity, r.get<Excitable>(entity));
}
void Scene::OnUpdateExcitable(entt::registry &r, entt::entity entity) {
    UpdateHighlightedVertices(entity, r.get<Excitable>(entity));
}
void Scene::OnDestroyExcitable(entt::registry &r, entt::entity entity) {
    // The last excitable entity is being destroyed.
    if (r.storage<Excitable>().size() == 1) {
        if (SelectionMode == SelectionMode::Excite) SetSelectionMode(*SelectionModes.begin());
        SelectionModes.erase(SelectionMode::Excite);
    }

    static constexpr Excitable EmptyExcitable{};
    UpdateHighlightedVertices(entity, EmptyExcitable);
}

void Scene::OnCreateExcitedVertex(entt::registry &r, entt::entity entity) {
    auto &excited_vertex = r.get<ExcitedVertex>(entity);
    // Orient the camera towards the excited vertex.
    const auto vh = Mesh::VH(excited_vertex.Vertex);
    const auto &mesh = r.get<Mesh>(entity);
    const auto &transform = r.get<Model>(entity).Transform;
    const vec3 vertex_pos{transform * vec4{mesh.GetPosition(vh), 1}};
    Camera.SetTargetDirection(glm::normalize(vertex_pos - Camera.Target));

    // Create vertex indicator arrow pointing at the excited vertex.
    const vec3 normal{transform * vec4{mesh.GetVertexNormal(vh), 0}};
    const float scale_factor = 0.1f * mesh.BoundingBox.DiagonalLength();
    auto vertex_indicator_mesh = Arrow();
    vertex_indicator_mesh.SetFaceColor({1, 0, 0, 1});
    excited_vertex.IndicatorEntity = AddMesh(
        std::move(vertex_indicator_mesh),
        {.Name = "Excite vertex indicator",
         .Position = vertex_pos + 0.05f * scale_factor * normal,
         .Rotation = glm::rotation(World.Up, normal),
         .Scale = vec3{scale_factor},
         .Select = false}
    );
}
void Scene::OnDestroyExcitedVertex(entt::registry &r, entt::entity entity) {
    const auto &excited_vertex = r.get<ExcitedVertex>(entity);
    DestroyEntity(excited_vertex.IndicatorEntity);
}

vk::ImageView Scene::GetResolveImageView() const {
    return *MainResources->ResolveImage.View;
}

void Scene::SetVisible(entt::entity entity, bool visible) {
    const bool already_visible = R.all_of<Visible>(entity);
    if ((visible && already_visible) || (!visible && !already_visible)) return;

    const auto parent = GetParentEntity(entity);
    auto &parent_node = R.get<SceneNode>(parent);
    auto &model_indices = parent_node.ModelIndices;
    if (visible) {
        // Insert model index as the max value + 1.
        const uint new_model_index = entity == parent || model_indices.empty() ?
            0 :
            std::ranges::max_element(model_indices, [](auto &a, auto &b) { return a.second < b.second; })->second + 1;
        for (auto &[_, model_index] : model_indices) {
            if (model_index >= new_model_index) ++model_index;
        }
        model_indices.emplace(entity, new_model_index);
        const auto &model = R.get<Model>(entity);
        InsertBufferRegion(MeshVkData->Models.at(parent), &model, new_model_index * sizeof(Model), sizeof(Model));
        R.emplace<Visible>(entity);
    } else {
        R.remove<Visible>(entity);
        const uint old_model_index = *GetModelBufferIndex(entity);
        EraseBufferRegion(MeshVkData->Models.at(parent), old_model_index * sizeof(Model), sizeof(Model));
        model_indices.erase(entity);
        for (auto &[_, model_index] : model_indices) {
            if (model_index > old_model_index) --model_index;
        }
    }
}

mvk::RenderBuffers Scene::CreateRenderBuffers(RenderBuffers &&buffers) {
    return {
        CreateBuffer(vk::BufferUsageFlagBits::eVertexBuffer, buffers.Vertices.data(), sizeof(Vertex3D) * buffers.Vertices.size()),
        CreateBuffer(vk::BufferUsageFlagBits::eIndexBuffer, buffers.Indices.data(), sizeof(uint) * buffers.Indices.size())
    };
}

entt::entity Scene::AddMesh(Mesh &&mesh, MeshCreateInfo info) {
    const auto entity = R.create();

    auto node = R.emplace<SceneNode>(entity); // No parent or children.
    UpdateModel(R, entity, info.Position, info.Rotation, info.Scale);
    R.emplace<Name>(entity, CreateName(R, info.Name));

    MeshVkData->Models.emplace(entity, BufferAllocator->AllocateMvk(vk::BufferUsageFlagBits::eVertexBuffer, sizeof(Model)));
    SetVisible(entity, true); // Always set visibility to true first, since this sets up the model buffer/indices.
    if (!info.Visible) SetVisible(entity, false);

    MeshBuffers mesh_buffers{};
    for (auto element : AllElements) { // todo only create buffers for viewed elements.
        mesh_buffers.emplace(element, CreateRenderBuffers(mesh.CreateVertices(element), mesh.CreateIndices(element)));
    }
    MeshVkData->Main.emplace(entity, std::move(mesh_buffers));
    MeshVkData->NormalIndicators.emplace(entity, MeshBuffers{});

    if (ShowBoundingBoxes) {
        MeshVkData->Boxes.emplace(entity, CreateRenderBuffers(CreateBoxVertices(mesh.BoundingBox, EdgeColor), BBox::EdgeIndices));
    }

    R.emplace<Mesh>(entity, std::move(mesh));

    if (info.Select) SetActive(entity);
    InvalidateCommandBuffer();
    return entity;
}

entt::entity Scene::AddMesh(const fs::path &path, MeshCreateInfo info) {
    auto polymesh = LoadPolyMesh(path);
    if (!polymesh) throw std::runtime_error(std::format("Failed to load mesh: {}", path.string()));
    const auto entity = AddMesh({std::move(*polymesh)}, std::move(info));
    R.emplace<Path>(entity, path);
    return entity;
}

entt::entity Scene::AddPrimitive(Primitive primitive, MeshCreateInfo info) {
    if (info.Name.empty()) info.Name = to_string(primitive);
    auto entity = AddMesh(CreateDefaultPrimitive(primitive), std::move(info));
    R.emplace<Primitive>(entity, primitive);
    return entity;
}

void Scene::ClearMeshes() {
    std::vector<entt::entity> entities;
    for (auto entity : R.view<Mesh>()) entities.emplace_back(entity);
    for (auto entity : entities) DestroyEntity(entity);
    InvalidateCommandBuffer();
}

void Scene::ReplaceMesh(entt::entity entity, Mesh &&mesh) {
    for (auto &[element, buffers] : MeshVkData->Main.at(entity)) {
        UpdateBuffer(buffers.Vertices, mesh.CreateVertices(element));
        UpdateBuffer(buffers.Indices, mesh.CreateIndices(element));
    }
    for (auto &[element, buffers] : MeshVkData->NormalIndicators.at(entity)) {
        UpdateBuffer(buffers.Vertices, mesh.CreateNormalVertices(element));
        UpdateBuffer(buffers.Indices, mesh.CreateNormalIndices(element));
    }
    if (auto buffers = MeshVkData->Boxes.find(entity); buffers != MeshVkData->Boxes.end()) {
        UpdateBuffer(buffers->second.Vertices, CreateBoxVertices(mesh.BoundingBox, EdgeColor));
        // Box indices are always the same.
    }
    R.replace<Mesh>(entity, std::move(mesh));
}

entt::entity Scene::AddInstance(entt::entity parent, MeshCreateInfo info) {
    const auto entity = R.create();
    // For now, we assume one-level deep hierarchy, so we don't allocate a models buffer for the instance.
    R.emplace<SceneNode>(entity, parent);
    auto &parent_node = R.get<SceneNode>(parent);
    parent_node.Children.emplace_back(entity);
    UpdateModel(R, entity, info.Position, info.Rotation, info.Scale);
    R.emplace<Name>(entity, info.Name.empty() ? std::format("{} instance {}", GetName(R, parent), parent_node.Children.size()) : CreateName(R, info.Name));
    auto &model_buffer = MeshVkData->Models.at(parent);
    EnsureBufferHasAllocated(model_buffer, model_buffer.Size + sizeof(Model));
    SetVisible(entity, info.Visible);
    if (info.Select) SetActive(entity);
    InvalidateCommandBuffer();

    return entity;
}
void Scene::DestroyInstance(entt::entity instance) {
    SetVisible(instance, false);
    std::erase(R.get<SceneNode>(R.get<SceneNode>(instance).Parent).Children, instance);
    R.destroy(instance);
    InvalidateCommandBuffer();
}

void Scene::DestroyEntity(entt::entity entity) {
    if (const auto parent_entity = GetParentEntity(entity); parent_entity != entity) return DestroyInstance(entity);

    Vk.Device.waitIdle(); // xxx device blocking should be more targeted
    MeshVkData->Main.erase(entity);
    MeshVkData->NormalIndicators.erase(entity);
    MeshVkData->Models.erase(entity);
    MeshVkData->Boxes.erase(entity);

    const auto &node = R.get<SceneNode>(entity);
    for (const auto child : node.Children) R.destroy(child);
    R.destroy(entity);
    InvalidateCommandBuffer();
}

void Scene::SetSelectionMode(::SelectionMode mode) {
    if (SelectionMode == mode) return;

    SelectionMode = mode;
    for (const auto &[entity, mesh] : R.view<Mesh>().each()) {
        const bool highlight_faces = SelectionMode == SelectionMode::Excite && R.try_get<Excitable>(entity);
        mesh.SetFaceColor(highlight_faces ? Mesh::HighlightedFaceColor : Mesh::DefaultFaceColor);
        UpdateRenderBuffers(entity);
    }
    const auto entity = FindEntity<Active>(R);
    UpdateHighlightedVertices(entity, R.get<Excitable>(entity));
}
void Scene::SetEditingElement(MeshElementIndex element) {
    if (R.storage<Active>().empty()) return;

    EditingElement = element;
    UpdateRenderBuffers(GetParentEntity(FindEntity<Active>(R)));
}

void Scene::SetModel(entt::entity entity, vec3 position, quat rotation, vec3 scale) {
    UpdateModel(R, entity, position, rotation, scale);
    UpdateModelBuffer(entity);
    InvalidateCommandBuffer();
}

void Scene::WaitForRender() const {
    if (auto wait_result = Vk.Device.waitForFences(*RenderFence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    Vk.Device.resetFences(*RenderFence);
}

// TODO Use separate fence/semaphores for buffer updates and rendering?
void Scene::SubmitTransfer() const {
    vk::SubmitInfo submit;
    submit.setCommandBuffers(*TransferCommandBuffer);
    Vk.Queue.submit(submit, *RenderFence);
    WaitForRender();
}

// Get the model VK buffer index.
// Returns `std::nullopt` if the entity is not visible (and thus does not have a rendered model).
std::optional<uint> Scene::GetModelBufferIndex(entt::entity entity) {
    if (entity == entt::null) return std::nullopt;

    const auto parent_entity = GetParentEntity(entity);
    if (parent_entity == entt::null) return std::nullopt;

    if (const auto *parent_node = R.try_get<SceneNode>(parent_entity)) {
        const auto &model_indices = parent_node->ModelIndices;
        if (const auto it = model_indices.find(entity); it != model_indices.end()) return it->second;
    }
    return std::nullopt;
}

void Scene::UpdateRenderBuffers(entt::entity entity) {
    if (const auto *mesh = R.try_get<Mesh>(entity)) {
        auto &mesh_buffers = MeshVkData->Main.at(entity);
        const bool is_active = GetParentEntity(FindEntity<Active>(R)) == entity;
        const Mesh::ElementIndex selected_element{
            is_active && SelectionMode == SelectionMode::Edit       ? EditingElement :
                is_active && SelectionMode == SelectionMode::Excite ? MeshElementIndex{MeshElement::Vertex, int(R.get<Excitable>(entity).SelectedVertex())} :
                                                                      MeshElementIndex{}
        };
        for (auto element : AllElements) { // todo only update buffers for viewed elements.
            UpdateBuffer(mesh_buffers.at(element).Vertices, mesh->CreateVertices(element, selected_element));
        }
        InvalidateCommandBuffer();
    };
}

mvk::Buffer Scene::CreateBuffer(vk::BufferUsageFlags flags, const void *data, vk::DeviceSize size) const {
    auto buffer = BufferAllocator->AllocateMvk(flags, size);
    buffer.Size = size;
    BufferAllocator->WriteRegion(buffer.HostBuffer, data, 0, size);
    // Copy data from the staging buffer to the device buffer.
    const auto &cb = *TransferCommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cb.copyBuffer(buffer.HostBuffer, buffer.DeviceBuffer, vk::BufferCopy{0, 0, size});
    cb.end();
    SubmitTransfer();
    return buffer;
}
bool Scene::EnsureBufferHasAllocated(mvk::Buffer &buffer, vk::DeviceSize required_size) const {
    if (required_size == 0) return false;

    if (required_size > BufferAllocator->GetAllocatedSize(buffer)) {
        // Create a new buffer with enough space.
        // Copy the old buffer into the new buffer (host and device), and replace the old buffer.
        auto new_buffer = BufferAllocator->AllocateMvk(buffer.Usage, NextPowerOfTwo(required_size));
        if (buffer.Size > 0) {
            BufferAllocator->WriteRegion(new_buffer.HostBuffer, BufferAllocator->GetData(buffer.HostBuffer), 0, buffer.Size);
            new_buffer.Size = buffer.Size;
            // Device->device copy
            const auto &cb = *TransferCommandBuffer;
            cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cb.copyBuffer(buffer.DeviceBuffer, new_buffer.DeviceBuffer, vk::BufferCopy{0, 0, buffer.Size});
            cb.end();
            SubmitTransfer();
        }

        buffer = std::move(new_buffer);
        return true;
    }
    return false;
}
void Scene::UpdateBuffer(mvk::Buffer &buffer, const void *data, vk::DeviceSize offset, vk::DeviceSize size) const {
    if (size == 0) size = buffer.Size;
    assert(size > 0);

    const auto required_size = offset + size;
    EnsureBufferHasAllocated(buffer, required_size);
    buffer.Size = std::max(buffer.Size, required_size);
    BufferAllocator->WriteRegion(buffer.HostBuffer, data, offset, size);

    // Staging->device copy
    const auto &cb = *TransferCommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cb.copyBuffer(buffer.HostBuffer, buffer.DeviceBuffer, vk::BufferCopy{offset, offset, size});
    cb.end();
    SubmitTransfer();
}
void Scene::InsertBufferRegion(mvk::Buffer &buffer, const void *data, vk::DeviceSize offset, vk::DeviceSize size) const {
    if (size == 0 || buffer.Size + size > BufferAllocator->GetAllocatedSize(buffer)) return;

    if (offset < buffer.Size) {
        BufferAllocator->MoveRegion(buffer.HostBuffer, offset, offset + size, buffer.Size - offset);
    }
    BufferAllocator->WriteRegion(buffer.HostBuffer, data, offset, size);
    buffer.Size += size;

    // Staging->device copy
    const auto &cb = *TransferCommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    assert(buffer.Size > offset);
    cb.copyBuffer(buffer.HostBuffer, buffer.DeviceBuffer, vk::BufferCopy{offset, offset, buffer.Size - offset});
    cb.end();
    SubmitTransfer();
}
void Scene::EraseBufferRegion(mvk::Buffer &buffer, vk::DeviceSize offset, vk::DeviceSize size) const {
    if (size == 0 || offset + size > buffer.Size) return;

    if (const auto move_size = buffer.Size - (offset + size); move_size > 0) {
        BufferAllocator->MoveRegion(buffer.HostBuffer, offset + size, offset, move_size);

        const auto &cb = *TransferCommandBuffer;
        cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cb.copyBuffer(buffer.HostBuffer, buffer.DeviceBuffer, vk::BufferCopy{offset, offset, move_size});
        cb.end();
        SubmitTransfer();
    }
    buffer.Size -= size;
}

void Scene::SetExtent(vk::Extent2D extent) {
    Extent = extent;
    UpdateTransformBuffers(); // Depends on the aspect ratio.

    MainResources = std::make_unique<MainPipelineResources>(Vk.Device, Vk.PhysicalDevice, *MainRenderer.RenderPass, extent, GetMaxUsableSampleCount(Vk.PhysicalDevice));
    SilhouetteResources = std::make_unique<SilhouettePipelineResources>(Vk.Device, Vk.PhysicalDevice, *SilhouetteRenderer.RenderPass, extent);
    SilhouetteFillImageSampler = Vk.Device.createSamplerUnique({
        {},
        vk::Filter::eNearest,
        vk::Filter::eNearest,
        vk::SamplerMipmapMode::eNearest,
        // Prevent edge detection from wrapping around to the other side of the image.
        // Instead, use the pixel value at the nearest edge.
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
    });

    Vk.Device.updateDescriptorSets(
        EdgeDetectionRenderer.GetDescriptors({
            {SPT::EdgeDetection, "Tex", std::nullopt, vk::DescriptorImageInfo{*SilhouetteFillImageSampler, *SilhouetteResources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal}},
        }),
        {}
    );
    EdgeDetectionResources = std::make_unique<EdgeDetectionPipelineResources>(Vk.Device, Vk.PhysicalDevice, *EdgeDetectionRenderer.RenderPass, extent);
    SilhouetteEdgeImageSampler = Vk.Device.createSamplerUnique({{}, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest});
    Vk.Device.updateDescriptorSets(
        MainRenderer.GetDescriptors({
            {SPT::Texture, "SilhouetteEdgeTexture", std::nullopt, vk::DescriptorImageInfo{*SilhouetteEdgeImageSampler, *EdgeDetectionResources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal}},
        }),
        {}
    );
}

std::vector<std::pair<SPT, MeshElement>> GetPipelineElements(RenderMode render_mode, ColorMode color_mode) {
    const SPT fill_pipeline = color_mode == ColorMode::Mesh ? SPT::Fill : SPT::DebugNormals;
    switch (render_mode) {
        case RenderMode::Vertices: return {{fill_pipeline, MeshElement::Vertex}};
        case RenderMode::Edges: return {{SPT::Line, MeshElement::Edge}};
        case RenderMode::Faces: return {{fill_pipeline, MeshElement::Face}};
        case RenderMode::FacesAndEdges: return {{fill_pipeline, MeshElement::Face}, {SPT::Line, MeshElement::Edge}};
        case RenderMode::None: return {};
    }
}

void Scene::RecordCommandBuffer() {
    const auto &cb = *CommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(Extent.width), float(Extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, Extent});

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, {}, {}, // No dependency flags, memory barriers, or buffer memory barriers
        std::vector<vk::ImageMemoryBarrier>{{
            {},
            {},
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            *MainResources->ResolveImage.Image,
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }}
    );

    const auto active_entity = FindEntity<Active>(R);
    const auto active_mesh_entity = GetParentEntity(active_entity);
    const auto active_model_buffer_index = GetModelBufferIndex(active_entity);
    const bool render_silhouette = active_model_buffer_index && SelectionMode == SelectionMode::Object;
    if (render_silhouette) {
        // Render the silhouette edges for the active mesh instance.
        {
            static const std::vector<vk::ClearValue> clear_values{{Transparent}};
            const vk::Rect2D rect{{0, 0}, {SilhouetteResources->OffscreenImage.Extent.width, SilhouetteResources->OffscreenImage.Extent.height}};
            cb.beginRenderPass({*SilhouetteRenderer.RenderPass, *SilhouetteResources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);
        }

        SilhouetteRenderer.Render(
            cb,
            SPT::Silhouette,
            MeshVkData->Main.at(active_mesh_entity).at(MeshElement::Vertex),
            MeshVkData->Models.at(active_mesh_entity),
            *active_model_buffer_index
        );
        cb.endRenderPass();
        {
            static const std::vector<vk::ClearValue> clear_values{{Transparent}};
            const vk::Rect2D rect{{0, 0}, {EdgeDetectionResources->OffscreenImage.Extent.width, EdgeDetectionResources->OffscreenImage.Extent.height}};
            cb.beginRenderPass({*EdgeDetectionRenderer.RenderPass, *EdgeDetectionResources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);
        }

        EdgeDetectionRenderer.ShaderPipelines.at(SPT::EdgeDetection).RenderQuad(cb);
        cb.endRenderPass();
    }

    // Render meshes.
    // todo:
    //   - reorganize mesh VK buffers to reduce the number of draw calls and pipeline switches.
    //   - update `MeshVkData->Models` and `GetModelBufferIndex` to keep `Models` contiguous with only visible, or
    //   - keep all models in the `MeshVkData` but then update `drawIndexed` to use a different strategy:
    //     -  https://www.reddit.com/r/vulkan/comments/b7u2hu/way_to_draw_multiple_meshes_with_different/
    //        vkCmdDrawIndexedIndirectCount & put the offsets in a UBO indexed with gl_DrawId.
    const auto &meshes = R.view<const Mesh>();
    {
        const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {BackgroundColor}};
        const vk::Rect2D rect{{0, 0}, {MainResources->OffscreenImage.Extent.width, MainResources->OffscreenImage.Extent.height}};
        cb.beginRenderPass({*MainRenderer.RenderPass, *MainResources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);
    }
    meshes.each([this, &cb](auto entity, auto &) {
        const auto &buffers = MeshVkData->Main.at(entity);
        const auto &models = MeshVkData->Models.at(entity);
        for (const auto [pipeline, element] : GetPipelineElements(RenderMode, ColorMode)) {
            MainRenderer.Render(cb, pipeline, buffers.at(element), models);
        }
    });

    // Render silhouette edge texture.
    if (render_silhouette) MainRenderer.ShaderPipelines.at(SPT::Texture).RenderQuad(cb);

    // Render normal indicators.
    meshes.each([this, &cb](auto entity, auto &) {
        const auto &buffers = MeshVkData->NormalIndicators.at(entity);
        const auto &models = MeshVkData->Models.at(entity);
        for (const auto &[element, normal_indicators] : buffers) {
            MainRenderer.Render(cb, SPT::Line, normal_indicators, models);
        }
    });

    if (ShowBoundingBoxes) {
        meshes.each([this, &cb](auto entity, auto &) {
            if (auto buffers = MeshVkData->Boxes.find(entity); buffers != MeshVkData->Boxes.end()) {
                MainRenderer.Render(cb, SPT::Line, buffers->second, MeshVkData->Models.at(entity));
            }
        });
    }
    if (ShowBvhBoxes) {
        meshes.each([this, &cb](auto entity, auto &) {
            if (auto buffers = MeshVkData->BvhBoxes.find(entity); buffers != MeshVkData->BvhBoxes.end()) {
                MainRenderer.Render(cb, SPT::Line, buffers->second, MeshVkData->Models.at(entity));
            }
        });
    }

    if (ShowGrid) MainRenderer.ShaderPipelines.at(SPT::Grid).RenderQuad(cb);

    cb.endRenderPass();
    cb.end();
}

void Scene::SubmitCommandBuffer(vk::Fence fence) const {
    vk::SubmitInfo submit;
    submit.setCommandBuffers(*CommandBuffer);
    Vk.Queue.submit(submit, fence);
}

void Scene::InvalidateCommandBuffer() {
    CommandBufferDirty = true;
}

void Scene::CompileShaders() {
    MainRenderer.CompileShaders();
    SilhouetteRenderer.CompileShaders();
    EdgeDetectionRenderer.CompileShaders();
}

void Scene::UpdateEdgeColors() {
    Mesh::EdgeColor = RenderMode == RenderMode::FacesAndEdges ? MeshEdgeColor : EdgeColor;
    for (auto entity : R.view<Mesh>()) UpdateRenderBuffers(entity);
}

void Scene::UpdateTransformBuffers() {
    const float aspect_ratio = Extent.width == 0 || Extent.height == 0 ? 1.f : float(Extent.width) / float(Extent.height);
    const ViewProj view_proj{Camera.GetView(), Camera.GetProjection(aspect_ratio)};
    UpdateBuffer(*TransformBuffer, &view_proj, 0, sizeof(ViewProj));

    const ViewProjNearFar vpnf{view_proj.View, view_proj.Projection, Camera.NearClip, Camera.FarClip};
    UpdateBuffer(*ViewProjNearFarBuffer, &vpnf, 0, sizeof(ViewProjNearFar));
    InvalidateCommandBuffer();
}

void Scene::UpdateModelBuffer(entt::entity entity) {
    if (const auto buffer_index = GetModelBufferIndex(entity)) {
        const auto &model = R.get<Model>(entity);
        UpdateBuffer(MeshVkData->Models.at(GetParentEntity(entity)), &model, *buffer_index * sizeof(Model), sizeof(Model));
    }
}

using namespace ImGui;

namespace {
vec2 ToGlm(ImVec2 v) { return {v.x, v.y}; }
vk::Extent2D ToVkExtent(vec2 e) { return {uint(e.x), uint(e.y)}; }

void Capitalize(std::string &str) {
    if (!str.empty() && str[0] >= 'a' && str[0] <= 'z') str[0] += 'A' - 'a';
}

// We already cache the transpose of the inverse transform for all models,
// so save some compute by using that.
ray WorldToLocal(const ray &r, const mat4 &model_i_t) {
    const auto model_i = glm::transpose(model_i_t);
    return {{model_i * vec4{r.o, 1}}, glm::normalize(vec3{model_i * vec4{r.d, 0}})};
}

std::multimap<float, entt::entity> IntersectedEntitiesByDistance(const entt::registry &r, const ray &world_ray) {
    std::multimap<float, entt::entity> entities_by_distance;
    for (const auto &[entity, model] : r.view<const Model>().each()) {
        if (!r.all_of<Visible>(entity)) continue;

        const auto &mesh = r.get<const Mesh>(GetParentEntity(r, entity));
        if (auto intersection = mesh.Intersect(WorldToLocal(world_ray, model.InvTransform))) {
            entities_by_distance.emplace(intersection->Distance, entity);
        }
    }
    return entities_by_distance;
}

entt::entity CycleIntersectedEntity(const entt::registry &r, entt::entity active_entity, const ray &mouse_world_ray) {
    if (const auto entities_by_distance = IntersectedEntitiesByDistance(r, mouse_world_ray); !entities_by_distance.empty()) {
        // Cycle through hovered entities.
        auto it = find_if(entities_by_distance, [active_entity](const auto &entry) { return entry.second == active_entity; });
        if (it != entities_by_distance.end()) ++it;
        if (it == entities_by_distance.end()) it = entities_by_distance.begin();
        return it->second;
    }
    return entt::null;
}

// Nearest intersection across all meshes.
struct EntityIntersection {
    entt::entity Entity;
    Intersection Intersection;
    vec3 Position;
};

std::optional<EntityIntersection> IntersectNearest(const entt::registry &r, const ray &world_ray) {
    float nearest_distance = std::numeric_limits<float>::max();
    std::optional<EntityIntersection> nearest;
    for (const auto &[entity, model] : r.view<const Model>().each()) {
        if (!r.all_of<Visible>(entity)) continue;

        const auto &mesh = r.get<const Mesh>(GetParentEntity(r, entity));
        const auto local_ray = WorldToLocal(world_ray, model.InvTransform);
        if (auto intersection = mesh.Intersect(local_ray);
            intersection && intersection->Distance < nearest_distance) {
            nearest_distance = intersection->Distance;
            nearest = {entity, *intersection, local_ray(nearest_distance)};
        }
    }
    return nearest;
}
} // namespace

// Returns a world space ray from the mouse into the scene.
ray Scene::GetMouseWorldRay() const {
    // Mouse pos in content region
    const vec2 mouse_pos = ToGlm((GetMousePos() - GetCursorScreenPos()) / GetContentRegionAvail());
    // Normalized Device Coordinates in [-1,1]^2
    const vec2 mouse_pos_ndc{2 * mouse_pos.x - 1, 1 - 2 * mouse_pos.y};
    return Camera.ClipPosToWorldRay(mouse_pos_ndc, Extent.width / Extent.height);
}

void Scene::Interact() {
    if (Extent.width == 0 || Extent.height == 0) return;

    const auto active_entity = FindEntity<Active>(R);
    // Handle keyboard input.
    if (IsWindowFocused()) {
        if (IsKeyPressed(ImGuiKey_Tab)) {
            // Cycle to the next selection mode, wrapping around to the first.
            auto it = find(SelectionModes, SelectionMode);
            SetSelectionMode(++it != SelectionModes.end() ? *it : *SelectionModes.begin());
        }
        if (SelectionMode == SelectionMode::Edit) {
            if (IsKeyPressed(ImGuiKey_1)) SetEditingElement({MeshElement::Vertex, -1});
            else if (IsKeyPressed(ImGuiKey_2)) SetEditingElement({MeshElement::Edge, -1});
            else if (IsKeyPressed(ImGuiKey_3)) SetEditingElement({MeshElement::Face, -1});
        }
        if (active_entity != entt::null && (IsKeyPressed(ImGuiKey_Delete) || IsKeyPressed(ImGuiKey_Backspace))) {
            DestroyEntity(active_entity);
        }
    }

    // Handle mouse input.
    if (!IsMouseDown(ImGuiMouseButton_Left) && R.all_of<ExcitedVertex>(active_entity)) {
        R.erase<ExcitedVertex>(active_entity);
    }
    if (!IsWindowHovered()) return;

    // Mouse wheel for camera rotation, Cmd+wheel to zoom.
    const auto &io = GetIO();
    if (const vec2 wheel{io.MouseWheelH, io.MouseWheel}; wheel != vec2{0, 0}) {
        if (io.KeyCtrl || io.KeySuper) {
            Camera.SetTargetDistance(std::max(Camera.Distance * (1 - wheel.y / 16.f), 0.01f));
        } else {
            Camera.AddYawPitch(wheel * 0.1f);
        }
    }
    if (!IsMouseClicked(ImGuiMouseButton_Left) || ModelGizmo::CurrentOp() != ModelGizmo::Op::NoOp || OrientationGizmo::IsActive()) return;

    // Handle mouse selection.
    const auto mouse_world_ray = GetMouseWorldRay();
    if (SelectionMode == SelectionMode::Edit) {
        if (EditingElement.Element != MeshElement::None && active_entity != entt::null && R.all_of<Visible>(active_entity)) {
            const auto &model = R.get<Model>(active_entity);
            const auto mouse_ray = WorldToLocal(mouse_world_ray, model.InvTransform);
            const auto &mesh = GetActiveMesh();
            {
                const auto nearest_vertex = mesh.FindNearestVertex(mouse_ray);
                if (EditingElement.Element == MeshElement::Vertex) SetEditingElement(Mesh::ElementIndex{nearest_vertex});
                else if (EditingElement.Element == MeshElement::Edge) SetEditingElement(Mesh::ElementIndex{mesh.FindNearestEdge(mouse_ray)});
                else if (EditingElement.Element == MeshElement::Face) SetEditingElement(Mesh::ElementIndex{mesh.FindNearestIntersectingFace(mouse_ray)});
            }
        }
    } else if (SelectionMode == SelectionMode::Object) {
        const auto intersected = CycleIntersectedEntity(R, active_entity, mouse_world_ray);
        if (intersected != entt::null && IsKeyDown(ImGuiMod_Shift)) {
            if (R.all_of<Active>(intersected)) {
                R.remove<Active, Selected>(intersected);
            } else {
                R.emplace_or_replace<Selected>(intersected);
                R.clear<Active>();
                R.emplace<Active>(intersected);
            }
            InvalidateCommandBuffer();
        } else {
            SetActive(intersected);
        }
    } else if (SelectionMode == SelectionMode::Excite) {
        // Excite the nearest entity if it's excitable.
        if (const auto nearest = IntersectNearest(R, mouse_world_ray)) {
            if (const auto *excitable = R.try_get<Excitable>(nearest->Entity)) {
                // Find the nearest excitable vertex.
                std::optional<uint> nearest_excite_vertex;
                float min_dist_sq = std::numeric_limits<float>::max();
                const auto &mesh = R.get<Mesh>(GetParentEntity(nearest->Entity));
                const auto p = nearest->Position;
                for (uint excite_vertex : excitable->ExcitableVertices) {
                    const auto diff = p - mesh.GetPosition(Mesh::VH(excite_vertex));
                    if (float dist_sq = glm::dot(diff, diff); dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        nearest_excite_vertex = excite_vertex;
                    }
                }
                if (nearest_excite_vertex) {
                    R.emplace<ExcitedVertex>(nearest->Entity, *nearest_excite_vertex, 1.f);
                }
            }
        }
    }
}

bool Scene::Render() {
    const vec2 content_region = ToGlm(GetContentRegionAvail());
    const bool extent_changed = Extent.width != content_region.x || Extent.height != content_region.y;
    if (!extent_changed && !CommandBufferDirty) return false;

    if (extent_changed) SetExtent(ToVkExtent(content_region));
    RecordCommandBuffer();
    SubmitCommandBuffer(*RenderFence);
    CommandBufferDirty = false;

    // The contract is that the caller may use the resolve image and sampler immediately after `Scene::Render` returns.
    // Returning `true` indicates that the resolve image/sampler have been recreated.
    WaitForRender();
    return extent_changed;
}

void Scene::RenderGizmo() {
    const auto content_region = ToGlm(GetContentRegionAvail());
    const float line_height = GetTextLineHeightWithSpacing();
    const auto window_pos = ToGlm(GetWindowPos());
    auto view = Camera.GetView();
    if (MGizmo.Show && !R.storage<Active>().empty()) {
        const auto active_entity = FindEntity<Active>(R);
        const auto proj = Camera.GetProjection(float(Extent.width) / float(Extent.height));
        if (auto model = R.get<Model>(active_entity).Transform;
            ModelGizmo::Draw(ModelGizmo::Local, MGizmo.Op, window_pos + line_height, content_region, model, view, proj, MGizmo.Snap ? std::optional{MGizmo.SnapValue} : std::nullopt)) {
            // Decompose affine model matrix into pos, scale, and orientation.
            const vec3 position = model[3];
            const vec3 scale{glm::length(model[0]), glm::length(model[1]), glm::length(model[2])};
            const auto orientation = glm::quat_cast(mat3{vec3{model[0]} / scale.x, vec3{model[1]} / scale.y, vec3{model[2]} / scale.z});
            SetModel(active_entity, position, orientation, scale);
        }
    }
    static constexpr float OGizmoSize{90};
    const float padding = 2 * line_height;
    const auto pos = window_pos + vec2{GetWindowContentRegionMax().x, GetWindowContentRegionMin().y} - vec2{OGizmoSize, 0} + vec2{-padding, padding};
    OrientationGizmo::Draw(pos, OGizmoSize, Camera);
    if (Camera.Tick()) UpdateTransformBuffers();
}

namespace {
std::optional<Mesh> PrimitiveEditor(Primitive primitive, bool is_create = true) {
    const char *create_label = is_create ? "Add" : "Update";
    if (primitive == Primitive::Rect) {
        static vec2 size{1, 1};
        InputFloat2("Size", &size.x);
        if (Button(create_label)) return Rect(size / 2.f);
    } else if (primitive == Primitive::Circle) {
        static float r = 0.5;
        InputFloat("Radius", &r);
        if (Button(create_label)) return Circle(r);
    } else if (primitive == Primitive::Cube) {
        static vec3 size{1.0, 1.0, 1.0};
        InputFloat3("Size", &size.x);
        if (Button(create_label)) return Cuboid(size / 2.f);
    } else if (primitive == Primitive::IcoSphere) {
        static float r = 0.5;
        static int subdivisions = 3;
        InputFloat("Radius", &r);
        InputInt("Subdivisions", &subdivisions);
        if (Button(create_label)) return IcoSphere(r, uint(subdivisions));
    } else if (primitive == Primitive::UVSphere) {
        static float r = 0.5;
        InputFloat("Radius", &r);
        if (Button(create_label)) return UVSphere(r);
    } else if (primitive == Primitive::Torus) {
        static vec2 radii{0.5, 0.2};
        static glm::ivec2 n_segments = {32, 16};
        InputFloat2("Major/minor radius", &radii.x);
        InputInt2("Major/minor segments", &n_segments.x);
        if (Button(create_label)) return Torus(radii.x, radii.y, uint(n_segments.x), uint(n_segments.y));
    } else if (primitive == Primitive::Cylinder) {
        static float r = 1, h = 1;
        InputFloat("Radius", &r);
        InputFloat("Height", &h);
        if (Button(create_label)) return Cylinder(r, h);
    } else if (primitive == Primitive::Cone) {
        static float r = 1, h = 1;
        InputFloat("Radius", &r);
        InputFloat("Height", &h);
        if (Button(create_label)) return Cone(r, h);
    }

    return std::nullopt;
}

void RenderMat4(const mat4 &m) {
    for (uint i = 0; i < 4; ++i) {
        Text("%.2f, %.2f, %.2f, %.2f", m[0][i], m[1][i], m[2][i], m[3][i]);
    }
}
} // namespace

void Scene::RenderControls() {
    if (BeginTabBar("Scene controls")) {
        const auto active_entity = FindEntity<Active>(R);
        const auto active_mesh_entity = GetParentEntity(active_entity);
        if (BeginTabItem("Object")) {
            {
                PushID("SelectionMode");
                AlignTextToFramePadding();
                TextUnformatted("Selection mode:");
                int selection_mode = int(SelectionMode);
                bool selection_mode_changed = false;
                for (const auto mode : SelectionModes) {
                    SameLine();
                    selection_mode_changed |= RadioButton(to_string(mode).c_str(), &selection_mode, int(mode));
                }
                if (selection_mode_changed) SetSelectionMode(::SelectionMode(selection_mode));
                if (SelectionMode == SelectionMode::Edit) {
                    AlignTextToFramePadding();
                    TextUnformatted("Edit mode:");
                    int element_selection_mode = int(EditingElement.Element);
                    for (const auto element : AllElements) {
                        auto name = to_string(element);
                        Capitalize(name);
                        SameLine();
                        if (RadioButton(name.c_str(), &element_selection_mode, int(element))) {
                            SetEditingElement({element, -1});
                        }
                    }
                    Text("Editing %s: %s", to_string(EditingElement.Element).c_str(), EditingElement.is_valid() ? std::to_string(EditingElement.idx()).c_str() : "None");
                    if (EditingElement.Element == MeshElement::Vertex && EditingElement.is_valid() && active_entity != entt::null) {
                        const auto &mesh = GetActiveMesh();
                        const auto pos = mesh.GetPosition(Mesh::VH{EditingElement.idx()});
                        Text("Vertex %d: (%.4f, %.4f, %.4f)", EditingElement.idx(), pos.x, pos.y, pos.z);
                    }
                }
                PopID();
            }
            if (active_entity != entt::null) {
                PushID(uint(active_entity));
                Text("Active entity: %s", GetName(R, active_entity).c_str());
                Indent();

                const auto &node = R.get<SceneNode>(active_entity);
                if (auto parent_entity = node.Parent; parent_entity != entt::null) {
                    AlignTextToFramePadding();
                    Text("Parent: %s", GetName(R, parent_entity).c_str());
                    SameLine();
                    if (Button("Select")) SetActive(parent_entity);
                }
                if (!node.Children.empty() && CollapsingHeader("Children")) {
                    RenderEntitiesTable("Children", node.Children);
                }
                const auto &active_mesh = R.get<const Mesh>(active_mesh_entity);
                TextUnformatted(
                    std::format("Vertices | Edges | Faces: {:L} | {:L} | {:L}", active_mesh.GetVertexCount(), active_mesh.GetEdgeCount(), active_mesh.GetFaceCount()).c_str()
                );
                Text("Model buffer index: %s", GetModelBufferIndex(active_entity) ? std::to_string(*GetModelBufferIndex(active_entity)).c_str() : "None");
                Unindent();
                bool visible = R.all_of<Visible>(active_entity);
                if (Checkbox("Visible", &visible)) {
                    SetVisible(active_entity, visible);
                    InvalidateCommandBuffer();
                }
                if (Button("Add instance")) AddInstance(active_mesh_entity);
                if (CollapsingHeader("Transform")) {
                    auto pos = R.get<Position>(active_entity).Value;
                    auto scale = R.get<Scale>(active_entity).Value;
                    auto rot = R.get<Rotation>(active_entity).Value;
                    bool model_changed = false;
                    model_changed |= DragFloat3("Position", &pos[0], 0.01f);
                    model_changed |= DragFloat4("Rotation (quat WXYZ)", &rot[0], 0.01f);

                    const bool frozen = R.all_of<Frozen>(active_entity);
                    if (frozen) BeginDisabled();
                    const auto label = std::format("Scale{}", frozen ? " (frozen)" : "");
                    model_changed |= DragFloat3(label.c_str(), &scale[0], 0.01f, 0.01f, 10);
                    if (frozen) EndDisabled();
                    if (model_changed) SetModel(active_entity, pos, rot, scale);

                    using namespace ModelGizmo;
                    const bool scale_enabled = !frozen;
                    if (!scale_enabled && MGizmo.Op == Op::Scale) MGizmo.Op = Op::Translate;

                    Checkbox("Gizmo", &MGizmo.Show);
                    if (MGizmo.Show) {
                        if (const auto label = ToString(CurrentOp()); label != "") Text("Op: %s", label.data());
                        if (IsActive()) Text("Active");

                        auto &op = MGizmo.Op;
                        if (IsKeyPressed(ImGuiKey_T)) op = Op::Translate;
                        if (IsKeyPressed(ImGuiKey_R)) op = Op::Rotate;
                        if (scale_enabled && IsKeyPressed(ImGuiKey_S)) op = Op::Scale;
                        if (RadioButton("Translate (T)", op == Op::Translate)) op = Op::Translate;
                        if (RadioButton("Rotate (R)", op == Op::Rotate)) op = Op::Rotate;
                        if (!scale_enabled) BeginDisabled();
                        const auto label = std::format("Scale (S){}", !scale_enabled ? " (frozen)" : "");
                        if (RadioButton(label.c_str(), op == Op::Scale)) op = Op::Scale;
                        if (!scale_enabled) EndDisabled();
                        if (RadioButton("Universal", op == Op::Universal)) op = Op::Universal;
                        Spacing();
                        Checkbox("Snap", &MGizmo.Snap);
                        if (MGizmo.Snap) {
                            SameLine();
                            // todo link/unlink snap values
                            DragFloat3("Snap", &MGizmo.SnapValue.x, 1.f, 0.01f, 100.f);
                        }
                    }
                    if (TreeNode("Model transform")) {
                        TextUnformatted("Transform");
                        const auto &model = R.get<Model>(active_entity);
                        RenderMat4(model.Transform);
                        Spacing();
                        TextUnformatted("Inverse transform");
                        RenderMat4(model.InvTransform);
                        TreePop();
                    }
                }
                if (const auto *primitive = R.try_get<Primitive>(active_mesh_entity)) {
                    if (CollapsingHeader("Update primitive")) {
                        if (auto new_mesh = PrimitiveEditor(*primitive, false)) {
                            ReplaceMesh(active_mesh_entity, std::move(*new_mesh));
                            InvalidateCommandBuffer();
                        }
                    }
                }
                PopID();
            } else {
                Text("Selected object: None");
            }

            if (CollapsingHeader("Add primitive")) {
                PushID("AddPrimitive");
                static int select_primitive = int(Primitive::Cube);
                for (uint i = 0; i < AllPrimitives.size(); ++i) {
                    if (i % 3 != 0) SameLine();
                    const auto primitive = AllPrimitives[i];
                    RadioButton(to_string(primitive).c_str(), &select_primitive, int(primitive));
                }
                if (auto mesh = PrimitiveEditor(Primitive(select_primitive), true)) {
                    R.emplace<Primitive>(AddMesh(std::move(*mesh), {.Name = to_string(Primitive(select_primitive))}), Primitive(select_primitive));
                }
                PopID();
            }

            if (CollapsingHeader("All objects")) {
                std::vector<entt::entity> table_entities;
                for (const auto &[entity, node] : R.view<const SceneNode>().each()) {
                    if (node.Parent == entt::null) table_entities.emplace_back(entity);
                }
                RenderEntitiesTable("All objects", table_entities);
            }
            EndTabItem();
        }

        if (BeginTabItem("Render")) {
            if (ColorEdit3("Background color", BackgroundColor.float32)) InvalidateCommandBuffer();
            if (Checkbox("Show grid", &ShowGrid)) InvalidateCommandBuffer();
            if (Button("Recompile shaders")) {
                CompileShaders();
                InvalidateCommandBuffer();
            }
            SeparatorText("Render mode");
            PushID("RenderMode");
            int render_mode = int(RenderMode);
            bool render_mode_changed = RadioButton("Vertices", &render_mode, int(RenderMode::Vertices));
            SameLine();
            render_mode_changed |= RadioButton("Edges", &render_mode, int(RenderMode::Edges));
            SameLine();
            render_mode_changed |= RadioButton("Faces", &render_mode, int(RenderMode::Faces));
            SameLine();
            render_mode_changed |= RadioButton("Faces and edges", &render_mode, int(RenderMode::FacesAndEdges));
            PopID();

            int color_mode = int(ColorMode);
            bool color_mode_changed = false;
            if (RenderMode != RenderMode::Edges) {
                SeparatorText("Fill color mode");
                PushID("ColorMode");
                color_mode_changed |= RadioButton("Mesh", &color_mode, int(ColorMode::Mesh));
                color_mode_changed |= RadioButton("Normals", &color_mode, int(ColorMode::Normals));
                PopID();
            }
            if (render_mode_changed || color_mode_changed) {
                RenderMode = ::RenderMode(render_mode);
                ColorMode = ::ColorMode(color_mode);
                UpdateEdgeColors(); // Different modes use different edge colors for better visibility.
            }
            if (RenderMode == RenderMode::FacesAndEdges || RenderMode == RenderMode::Edges) {
                auto &edge_color = RenderMode == RenderMode::FacesAndEdges ? MeshEdgeColor : EdgeColor;
                if (ColorEdit3("Edge color", &edge_color.x)) UpdateEdgeColors();
            }
            SeparatorText("Indicators");
            // todo go back to storing normal settings in a map of element type to bool,
            //   and ensure meshes/instances are created with the current normals
            if (active_entity != entt::null) {
                AlignTextToFramePadding();
                TextUnformatted("Normals:");
                const auto &mesh = R.get<Mesh>(active_mesh_entity);
                auto &normals = MeshVkData->NormalIndicators.at(active_mesh_entity);
                for (const auto element : AllNormalElements) {
                    SameLine();
                    bool has_normals = normals.contains(element);
                    auto name = to_string(element);
                    Capitalize(name);
                    if (Checkbox(name.c_str(), &has_normals)) {
                        if (has_normals) {
                            normals.emplace(element, CreateRenderBuffers(mesh.CreateNormalVertices(element), mesh.CreateNormalIndices(element)));
                        } else {
                            normals.erase(element);
                        }
                        InvalidateCommandBuffer();
                    }
                }
                if (Checkbox("BVH boxes", &ShowBvhBoxes)) {
                    auto &buffers = MeshVkData->BvhBoxes;
                    if (ShowBvhBoxes) buffers.emplace(active_mesh_entity, CreateRenderBuffers(mesh.CreateBvhBuffers(EdgeColor)));
                    else buffers.erase(active_mesh_entity);
                    InvalidateCommandBuffer();
                }
                SameLine(); // For Bounding boxes checkbox
            }
            if (Checkbox("Bounding boxes", &ShowBoundingBoxes)) {
                auto &buffers = MeshVkData->Boxes;
                for (const auto &[entity, mesh] : R.view<const Mesh>().each()) {
                    if (ShowBoundingBoxes) buffers.emplace(entity, CreateRenderBuffers(CreateBoxVertices(mesh.BoundingBox, EdgeColor), BBox::EdgeIndices));
                    else buffers.erase(entity);
                }
                InvalidateCommandBuffer();
            }
            SeparatorText("Silhouette");
            if (ColorEdit4("Color", &ActiveSilhouetteColor[0])) {
                UpdateBuffer(*SilhouetteDisplayBuffer, &ActiveSilhouetteColor, 0, sizeof(ActiveSilhouetteColor));
                InvalidateCommandBuffer();
            }
            EndTabItem();
        }

        if (BeginTabItem("Camera")) {
            bool camera_changed = false;
            if (Button("Reset camera")) {
                Camera = CreateDefaultCamera();
                camera_changed = true;
            }
            // camera_changed |= SliderFloat3("Position", &Camera.Position.x, -10, 10);
            camera_changed |= SliderFloat3("Target", &Camera.Target.x, -10, 10);
            camera_changed |= SliderFloat("Field of view (deg)", &Camera.FieldOfView, 1, 180);
            camera_changed |= SliderFloat("Near clip", &Camera.NearClip, 0.001f, 10, "%.3f", ImGuiSliderFlags_Logarithmic);
            camera_changed |= SliderFloat("Far clip", &Camera.FarClip, 10, 1000, "%.1f", ImGuiSliderFlags_Logarithmic);
            if (camera_changed) {
                Camera.StopMoving();
                UpdateTransformBuffers();
            }
            EndTabItem();
        }

        if (BeginTabItem("Lights")) {
            bool light_changed = false;
            SeparatorText("View light");
            light_changed |= ColorEdit3("Color##View", &Lights.ViewColorAndAmbient[0]);
            SeparatorText("Ambient light");
            light_changed |= SliderFloat("Intensity##Ambient", &Lights.ViewColorAndAmbient[3], 0, 1);
            SeparatorText("Directional light");
            light_changed |= SliderFloat3("Direction##Directional", &Lights.Direction[0], -1, 1);
            light_changed |= ColorEdit3("Color##Directional", &Lights.DirectionalColorAndIntensity[0]);
            light_changed |= SliderFloat("Intensity##Directional", &Lights.DirectionalColorAndIntensity[3], 0, 1);
            if (light_changed) {
                UpdateBuffer(*LightsBuffer, &Lights, 0, sizeof(Lights));
                InvalidateCommandBuffer();
            }
            EndTabItem();
        }
        EndTabBar();
    }
}

void Scene::RenderEntitiesTable(std::string name, const std::vector<entt::entity> &entities) {
    if (MeshEditor::BeginTable(name.c_str(), 3)) {
        static const float CharWidth = CalcTextSize("A").x;
        TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
        TableSetupColumn("Name");
        TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, CharWidth * 16);
        TableHeadersRow();
        entt::entity toggle_active = entt::null, delete_entity = entt::null, toggle_selected = entt::null;
        for (const auto entity : entities) {
            PushID(uint(entity));
            TableNextColumn();
            AlignTextToFramePadding();
            if (R.all_of<Active>(entity)) TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImGuiCol_TextSelectedBg));
            TextUnformatted(IdString(entity).c_str());
            TableNextColumn();
            TextUnformatted(R.get<Name>(entity).Value.c_str());
            TableNextColumn();
            {
                const bool is_active = R.all_of<Active>(entity);
                if (Button(is_active ? "Deactivate" : "Activate")) toggle_active = entity;
            }
            {
                const bool is_selected = R.all_of<Selected>(entity);
                if (Button(is_selected ? "Deselect" : "Select")) toggle_selected = entity;
            }

            SameLine();
            if (Button("Delete")) delete_entity = entity;
            PopID();
        }
        if (toggle_active != entt::null) {
            SetActive(R.all_of<Active>(toggle_active) ? entt::null : toggle_active);
        } else if (toggle_selected != entt::null) ToggleSelected(toggle_selected);
        if (delete_entity != entt::null) DestroyEntity(delete_entity);
        EndTable();
    }
}