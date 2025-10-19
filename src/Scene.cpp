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
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <imgui_internal.h>

#include <format>
#include <ranges>
#include <variant>

using std::ranges::find, std::ranges::find_if, std::ranges::fold_left, std::ranges::to;
using std::views::transform;

template<class... Ts> struct overloaded : Ts... {
    using Ts::operator()...;
};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// No built-in way to default-construct a variant by runtime index.
// Variants are the best, but they are the absolute worst to actually work with...
template<typename V> V CreateVariantByIndex(size_t i) {
    constexpr auto N = std::variant_size_v<V>;
    assert(i < N);
    return [&]<size_t... Is>(std::index_sequence<Is...>) -> V {
        static constexpr V table[]{V{std::in_place_index<Is>}...};
        return table[i]; // copy out
    }(std::make_index_sequence<N>{});
}

struct CameraUBO {
    mat4 View{1}, Proj{1};
    vec3 Position{0, 0, 0};
};

struct ViewProjNearFar {
    mat4 View{1}, Projection{1};
    float Near, Far;
};

// Stored on parent entities.
// Holds the `Model` of the parent entity at position 0, and children at 1+.
struct ModelsBuffer {
    mvk::UniqueBuffers Buffer;
};
struct BoundingBoxesBuffers {
    mvk::RenderBuffers Buffers;
};
struct BvhBoxesBuffers {
    mvk::RenderBuffers Buffers;
};
using MeshElementBuffers = std::unordered_map<MeshElement, mvk::RenderBuffers>;
struct MeshBuffers {
    MeshBuffers(MeshElementBuffers &&mesh, MeshElementBuffers &&normal_indicators)
        : Mesh{std::move(mesh)}, NormalIndicators{std::move(normal_indicators)} {}
    MeshBuffers(const MeshBuffers &) = delete;
    MeshBuffers &operator=(const MeshBuffers &) = delete;

    MeshElementBuffers Mesh, NormalIndicators;
};

entt::entity Scene::GetParentEntity(entt::entity e) const { return ::GetParentEntity(R, e); }
const Mesh &Scene::GetActiveMesh() const { return R.get<Mesh>(GetParentEntity(FindActiveEntity(R))); }

void Scene::Select(entt::entity e) {
    if (e != entt::null && R.all_of<Selected>(e)) return;

    R.clear<Selected>();
    if (e != entt::null) {
        R.clear<Active>();
        R.emplace<Active>(e);
        R.emplace<Selected>(e);
    }
    InvalidateCommandBuffer();
}
void Scene::ToggleSelected(entt::entity e) {
    if (e == entt::null) return;

    if (R.all_of<Selected>(e)) R.remove<Selected>(e);
    else R.emplace_or_replace<Selected>(e);
    InvalidateCommandBuffer();
}

std::vector<Vertex3D> CreateBoxVertices(const BBox &box, const vec4 &color) {
    return box.Corners() |
        // Normals don't matter for wireframes.
        transform([&color](const auto &corner) { return Vertex3D{corner, vec3{}, color}; }) |
        to<std::vector>();
}

const vk::ClearColorValue Transparent{0, 0, 0, 0};

namespace Format {
constexpr auto Color = vk::Format::eB8G8R8A8Unorm;
constexpr auto Depth = vk::Format::eD32Sfloat;
constexpr auto Float = vk::Format::eR32Sfloat;
constexpr auto Float2 = vk::Format::eR32G32Sfloat;
constexpr auto Float3 = vk::Format::eR32G32B32Sfloat;
constexpr auto Float4 = vk::Format::eR32G32B32A32Sfloat;
} // namespace Format

namespace {
// Mutually exclusive structs to track rotation representation.
// Note: `Rotation` is still the source of truth transformation component. These are for slider values only.
struct RotationQuat {
    quat Value; // wxyz
};
struct RotationEuler {
    vec3 Value; // xyz degrees
};
struct RotationAxisAngle {
    vec4 Value; // axis (xyz), angle (degrees)
};
using RotationUiVariant = std::variant<RotationQuat, RotationEuler, RotationAxisAngle>;

Transform GetTransform(const entt::registry &r, entt::entity e) {
    return {r.get<Position>(e).Value, r.get<Rotation>(e).Value, r.all_of<Scale>(e) ? r.get<Scale>(e).Value : vec3{1}};
}

void SetRotation(entt::registry &r, entt::entity e, const quat &v) {
    r.emplace_or_replace<Rotation>(e, v);
    if (!r.all_of<RotationUiVariant>(e)) {
        r.emplace<RotationUiVariant>(e, RotationQuat{v});
        return;
    }

    std::visit(
        overloaded{
            [&](RotationQuat &v_ui) { v_ui.Value = v; },
            [&](RotationEuler &v_ui) {
                float x, y, z;
                glm::extractEulerAngleXYZ(glm::mat4_cast(v), x, y, z);
                v_ui.Value = glm::degrees(vec3{x, y, z});
            },
            [&](RotationAxisAngle &v_ui) {
                const auto q = glm::normalize(v);
                v_ui.Value = {glm::axis(q), glm::degrees(glm::angle(q))};
            },
        },
        r.get<RotationUiVariant>(e)
    );
}

void UpdateModel(entt::registry &r, entt::entity e) {
    const auto t = GetTransform(r, e);
    r.emplace_or_replace<Model>(e, glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S));
}

void UpdateTransform(entt::registry &r, entt::entity e, const Transform &t) {
    r.emplace_or_replace<Position>(e, t.P);
    // Avoid replacing rotation UI slider values if the value hasn't changed.
    if (!r.all_of<Rotation>(e) || r.get<Rotation>(e).Value != t.R) SetRotation(r, e, t.R);
    // Frozen entities can't have their scale changed.
    if (!r.all_of<Frozen>(e)) r.emplace_or_replace<Scale>(e, t.S);

    UpdateModel(r, e);
}

vk::PipelineVertexInputStateCreateInfo CreateVertexInputState() {
    static const std::vector<vk::VertexInputBindingDescription> bindings{
        {0, sizeof(Vertex3D), vk::VertexInputRate::eVertex},
        {1, 2 * sizeof(mat4), vk::VertexInputRate::eInstance},
    };
    static const std::vector<vk::VertexInputAttributeDescription> attrs{
        {0, 0, Format::Float3, offsetof(Vertex3D, Position)},
        {1, 0, Format::Float3, offsetof(Vertex3D, Normal)},
        {2, 0, Format::Float4, offsetof(Vertex3D, Color)},
        // Model mat4, one vec4 per row
        {3, 1, Format::Float4, 0},
        {4, 1, Format::Float4, sizeof(vec4)},
        {5, 1, Format::Float4, 2 * sizeof(vec4)},
        {6, 1, Format::Float4, 3 * sizeof(vec4)},
        // Inverse model mat4, one vec4 per row
        {7, 1, Format::Float4, 4 * sizeof(vec4)},
        {8, 1, Format::Float4, 5 * sizeof(vec4)},
        {9, 1, Format::Float4, 6 * sizeof(vec4)},
        {10, 1, Format::Float4, 7 * sizeof(vec4)},
    };
    return {{}, bindings, attrs};
}

// todo: consider updating `drawIndexed` to use a different strategy:
//   -  https://www.reddit.com/r/vulkan/comments/b7u2hu/way_to_draw_multiple_meshes_with_different/
//      vkCmdDrawIndexedIndirectCount & put the offsets in a UBO indexed with gl_DrawId.
void Bind(vk::CommandBuffer cb, const ShaderPipeline &shader_pipeline, const mvk::RenderBuffers &render_buffers, const mvk::UniqueBuffers &models) {
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *shader_pipeline.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shader_pipeline.PipelineLayout, 0, *shader_pipeline.DescriptorSet, {});

    // Bind buffers
    static constexpr vk::DeviceSize vertex_buffer_offsets[] = {0}, models_buffer_offsets[] = {0};
    cb.bindVertexBuffers(0, {*render_buffers.Vertices.DeviceBuffer}, vertex_buffer_offsets);
    cb.bindIndexBuffer(*render_buffers.Indices.DeviceBuffer, 0, vk::IndexType::eUint32);
    cb.bindVertexBuffers(1, {*models.DeviceBuffer}, models_buffer_offsets);
}

void DrawIndexed(vk::CommandBuffer cb, const mvk::UniqueBuffers &indices, const mvk::UniqueBuffers &models, std::optional<uint> model_index = std::nullopt) {
    const uint index_count = indices.UsedSize / sizeof(uint);
    const uint first_instance = model_index.value_or(0);
    const uint instance_count = model_index.has_value() ? 1 : models.UsedSize / sizeof(Model);
    cb.drawIndexed(index_count, instance_count, 0, 0, first_instance);
}

vk::Extent2D ToExtent2D(vk::Extent3D extent) { return {extent.width, extent.height}; }

constexpr vk::ImageSubresourceRange DepthSubresourceRange{vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
constexpr vk::ImageSubresourceRange ColorSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
} // namespace

void PipelineRenderer::CompileShaders() {
    for (auto &shader_pipeline : std::views::values(ShaderPipelines)) shader_pipeline.Compile(*RenderPass);
}

void PipelineRenderer::Render(vk::CommandBuffer cb, SPT spt, const mvk::RenderBuffers &render_buffers, const mvk::UniqueBuffers &models, std::optional<uint> model_index) const {
    Bind(cb, ShaderPipelines.at(spt), render_buffers, models);
    DrawIndexed(cb, render_buffers.Indices, models, model_index);
}

mvk::ImageResource Scene::RenderBitmapToImage(std::span<const std::byte> data, uint32_t width, uint32_t height) const {
    auto image = mvk::CreateImage(
        Vk.Device, Vk.PhysicalDevice,
        {{},
         vk::ImageType::e2D,
         Format::Color,
         {width, height, 1},
         1,
         1,
         vk::SampleCountFlagBits::e1,
         vk::ImageTiling::eOptimal,
         vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
         vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, Format::Color, {}, ColorSubresourceRange}
    );

    const auto cb = *BufferContext.TransferCb;
    {
        // Write the bitmap into a temporary staging buffer.
        mvk::UniqueBuffer staging_buffer(*BufferContext.Allocator, as_bytes(data), mvk::MemoryUsage::CpuOnly);

        // Transition the image layout to be ready for data transfer.
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {}, // Dependency flags, memory barriers, buffer memory barriers
            vk::ImageMemoryBarrier{
                {}, // srcAccessMask
                vk::AccessFlagBits::eTransferWrite, // dstAccessMask
                {}, // oldLayout
                vk::ImageLayout::eTransferDstOptimal, // newLayout
                {}, // srcQueueFamilyIndex
                {}, // dstQueueFamilyIndex
                *image.Image, // image
                ColorSubresourceRange // subresourceRange
            }
        );

        // Copy buffer to image.
        cb.copyBufferToImage(
            *staging_buffer, *image.Image, vk::ImageLayout::eTransferDstOptimal,
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
                {}, // srcQueueFamilyIndex
                {}, // dstQueueFamilyIndex
                *image.Image, // image
                ColorSubresourceRange // subresourceRange
            }
        );
        cb.end();

        vk::SubmitInfo submit;
        submit.setCommandBuffers(cb);
        Vk.Queue.submit(submit, *TransferFence);
        WaitFor(*TransferFence);
    } // staging buffer is destroyed here

    BufferContext.Reclaimer.Reclaim();
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    return image;
}

std::vector<vk::WriteDescriptorSet> PipelineRenderer::GetDescriptors(std::vector<ShaderBindingDescriptor> &&descriptors) const {
    std::vector<vk::WriteDescriptorSet> write_descriptor_sets;
    for (const auto &descriptor : descriptors) {
        const auto &sp = ShaderPipelines.at(descriptor.PipelineType);
        if (auto ds = sp.CreateWriteDescriptorSet(descriptor.BindingName, descriptor.BufferInfo, descriptor.ImageInfo)) {
            write_descriptor_sets.push_back(*ds);
        }
    }
    return write_descriptor_sets;
}

// Pipeline definitions
namespace {
struct MainPipeline {
    static PipelineRenderer CreateRenderer(vk::Device d, vk::DescriptorPool descriptor_pool, vk::SampleCountFlagBits msaa_samples) {
        const std::vector<vk::AttachmentDescription> attachments{
            // Depth attachment.
            {{}, Format::Depth, msaa_samples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
            // Multisampled offscreen image.
            {{}, Format::Color, msaa_samples, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
            // Single-sampled resolve.
            {{}, Format::Color, vk::SampleCountFlagBits::e1, {}, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
        };
        const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
        const vk::AttachmentReference resolve_attachment_ref{2, vk::ImageLayout::eColorAttachmentOptimal};
        const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, &resolve_attachment_ref, &depth_attachment_ref};

        // Can't construct this map in-place with pairs because `ShaderPipeline` doesn't have a copy constructor.
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
                CreateColorBlendAttachment(true), CreateDepthStencil(), msaa_samples
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
                CreateColorBlendAttachment(true), CreateDepthStencil(), msaa_samples
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
                CreateColorBlendAttachment(true), CreateDepthStencil(true, false), msaa_samples
            }
        );

        // Render the silhouette edge depth regardless of the tested depth value.
        // We should be able to just disable depth tests and enable depth writes, but it seems that some GPUs or drivers
        // optimize out depth writes when depth testing is disabled, so instead we configure a depth test that always passes.
        pipelines.emplace(
            SPT::SilhouetteEdgeDepth,
            ShaderPipeline{
                d, descriptor_pool,
                Shaders{
                    {{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SampleDepth.frag"}}
                },
                vk::PipelineVertexInputStateCreateInfo{},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
                CreateColorBlendAttachment(true), CreateDepthStencil(true, true, vk::CompareOp::eAlways), msaa_samples
            }
        );
        // Render silhouette edge color regardless of the tested depth value.
        pipelines.emplace(
            SPT::SilhouetteEdgeColor,
            ShaderPipeline{
                d,
                descriptor_pool,
                Shaders{
                    {{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SilhouetteEdgeColor.frag"}}
                },
                vk::PipelineVertexInputStateCreateInfo{},
                vk::PolygonMode::eFill,
                vk::PrimitiveTopology::eTriangleStrip,
                CreateColorBlendAttachment(true),
                CreateDepthStencil(false, false),
                msaa_samples,
                vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)} // Manipulating flag (push constant has min 4 bytes)
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
                CreateColorBlendAttachment(true), CreateDepthStencil(), msaa_samples
            }
        );
        return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
    }

    struct ResourcesT {
        ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::SampleCountFlagBits msaa_samples, vk::RenderPass render_pass)
            : DepthImage{mvk::CreateImage(
                  d, pd,
                  {{},
                   vk::ImageType::e2D,
                   Format::Depth,
                   vk::Extent3D{extent, 1},
                   1,
                   1,
                   msaa_samples,
                   vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eDepthStencilAttachment,
                   vk::SharingMode::eExclusive},
                  {{}, {}, vk::ImageViewType::e2D, Format::Depth, {}, DepthSubresourceRange}
              )},
              OffscreenImage{mvk::CreateImage(
                  d, pd,
                  {{},
                   vk::ImageType::e2D,
                   Format::Color,
                   vk::Extent3D{extent, 1},
                   1,
                   1,
                   msaa_samples,
                   vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
                   vk::SharingMode::eExclusive},
                  {{}, {}, vk::ImageViewType::e2D, Format::Color, {}, ColorSubresourceRange}
              )},
              ResolveImage{mvk::CreateImage(
                  d, pd,
                  {
                      {},
                      vk::ImageType::e2D,
                      Format::Color,
                      vk::Extent3D{extent, 1},
                      1,
                      1,
                      vk::SampleCountFlagBits::e1,
                      vk::ImageTiling::eOptimal,
                      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
                      vk::SharingMode::eExclusive,
                  },
                  {{}, {}, vk::ImageViewType::e2D, Format::Color, {}, ColorSubresourceRange}
              )} {
            const std::array image_views{*DepthImage.View, *OffscreenImage.View, *ResolveImage.View};
            Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
        }

        // Perform depth testing, render into a multisampled offscreen image, and resolve into a single-sampled image.
        mvk::ImageResource DepthImage, OffscreenImage, ResolveImage;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::SampleCountFlagBits msaa_samples) {
        Resources = std::make_unique<ResourcesT>(extent, d, pd, msaa_samples, *Renderer.RenderPass);
    }
    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
};

struct SilhouettePipeline {
    static PipelineRenderer CreateRenderer(vk::Device d, vk::DescriptorPool descriptor_pool) {
        const std::vector<vk::AttachmentDescription> attachments{
            // We need to test and write depth since we want silhouette edges to respect mutual occlusion when multiple meshes are selected.
            {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
            // Single-sampled offscreen "image" of two channels: depth and object ID.
            {{}, Format::Float2, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        };
        const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
        const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, &depth_attachment_ref};
        std::unordered_map<SPT, ShaderPipeline> pipelines;
        pipelines.emplace(
            SPT::SilhouetteDepthObject,
            ShaderPipeline{
                d,
                descriptor_pool,
                Shaders{
                    {{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "DepthObject.frag"}}
                },
                CreateVertexInputState(),
                vk::PolygonMode::eFill,
                vk::PrimitiveTopology::eTriangleList,
                CreateColorBlendAttachment(false),
                CreateDepthStencil(),
                vk::SampleCountFlagBits::e1,
                vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)},
            }
        );
        return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
    }

    struct ResourcesT {
        ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass)
            : DepthImage{mvk::CreateImage(
                  d, pd,
                  {{},
                   vk::ImageType::e2D,
                   Format::Depth,
                   vk::Extent3D{extent, 1},
                   1,
                   1,
                   vk::SampleCountFlagBits::e1,
                   vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eDepthStencilAttachment,
                   vk::SharingMode::eExclusive},
                  {{}, {}, vk::ImageViewType::e2D, Format::Depth, {}, DepthSubresourceRange}
              )},
              OffscreenImage{mvk::CreateImage(
                  d, pd,
                  {{},
                   vk::ImageType::e2D,
                   Format::Float2,
                   vk::Extent3D{extent, 1},
                   1,
                   1,
                   vk::SampleCountFlagBits::e1,
                   vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
                   vk::SharingMode::eExclusive},
                  {{}, {}, vk::ImageViewType::e2D, Format::Float2, {}, ColorSubresourceRange}
              )} {
            const std::array image_views{*DepthImage.View, *OffscreenImage.View};
            Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
            ImageSampler = d.createSamplerUnique({
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
        }

        mvk::ImageResource DepthImage, OffscreenImage;
        vk::UniqueSampler ImageSampler;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd) {
        Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass);
    }

    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
};

struct SilhouetteEdgePipeline {
    static PipelineRenderer CreateRenderer(vk::Device d, vk::DescriptorPool descriptor_pool) {
        const std::vector<vk::AttachmentDescription> attachments{
            {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilReadOnlyOptimal},
            {{}, Format::Float, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        };
        const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
        const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, &depth_attachment_ref};
        std::unordered_map<SPT, ShaderPipeline> pipelines;
        pipelines.emplace(
            SPT::SilhouetteEdgeDepthObject,
            ShaderPipeline{
                d,
                descriptor_pool,
                Shaders{
                    {{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SilhouetteEdgeDepthObject.frag"}}
                },
                vk::PipelineVertexInputStateCreateInfo{},
                vk::PolygonMode::eFill,
                vk::PrimitiveTopology::eTriangleStrip,
                CreateColorBlendAttachment(false),
                CreateDepthStencil(),
                vk::SampleCountFlagBits::e1,
                vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(float)},
            }
        );
        return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
    }

    struct ResourcesT {
        ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass)
            : DepthImage{mvk::CreateImage(
                  d, pd,
                  {{},
                   vk::ImageType::e2D,
                   Format::Depth,
                   vk::Extent3D{extent, 1},
                   1,
                   1,
                   vk::SampleCountFlagBits::e1,
                   vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eDepthStencilAttachment,
                   vk::SharingMode::eExclusive},
                  {{}, {}, vk::ImageViewType::e2D, Format::Depth, {}, DepthSubresourceRange}
              )},
              OffscreenImage{mvk::CreateImage(
                  d, pd,
                  {{},
                   vk::ImageType::e2D,
                   Format::Float,
                   vk::Extent3D{extent, 1},
                   1,
                   1,
                   vk::SampleCountFlagBits::e1,
                   vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
                   vk::SharingMode::eExclusive},
                  {{}, {}, vk::ImageViewType::e2D, Format::Float, {}, ColorSubresourceRange}
              )} {
            const std::array image_views{*DepthImage.View, *OffscreenImage.View};
            Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
            ImageSampler = d.createSamplerUnique(vk::SamplerCreateInfo{});
            DepthSampler = d.createSamplerUnique(vk::SamplerCreateInfo{});
        }

        mvk::ImageResource DepthImage, OffscreenImage;
        vk::UniqueSampler ImageSampler, DepthSampler;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd) {
        Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass);
    }

    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
};
} // namespace

struct ScenePipelines {
    ScenePipelines(vk::Device d, vk::PhysicalDevice pd, vk::DescriptorPool dp)
        : d(d), pd(pd), Samples{GetMaxUsableSampleCount(pd)},
          Main{MainPipeline::CreateRenderer(d, dp, Samples), nullptr},
          Silhouette{SilhouettePipeline::CreateRenderer(d, dp), nullptr},
          SilhouetteEdge{SilhouetteEdgePipeline::CreateRenderer(d, dp), nullptr} {}

    vk::Device d;
    vk::PhysicalDevice pd;
    vk::SampleCountFlagBits Samples;

    MainPipeline Main;
    SilhouettePipeline Silhouette;
    SilhouetteEdgePipeline SilhouetteEdge;

    void SetExtent(vk::Extent2D);
    // These do _not_ re-submit the command buffer. Callers must do so manually if needed.
    void CompileShaders() {
        Main.Renderer.CompileShaders();
        Silhouette.Renderer.CompileShaders();
        SilhouetteEdge.Renderer.CompileShaders();
    }
};

Scene::Scene(SceneVulkanResources vc, entt::registry &r)
    : Vk(vc),
      R(r),
      CommandPool(Vk.Device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, Vk.QueueFamily})),
      RenderCommandBuffer(std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())),
      RenderFence(Vk.Device.createFenceUnique({})),
      TransferFence(Vk.Device.createFenceUnique({})),
      Pipelines(std::make_unique<ScenePipelines>(Vk.Device, Vk.PhysicalDevice, Vk.DescriptorPool)),
      BufferContext(Vk.PhysicalDevice, Vk.Device, Vk.Instance, *CommandPool),
      CameraUBOBuffer(BufferContext, sizeof(CameraUBO), vk::BufferUsageFlagBits::eUniformBuffer),
      ViewProjNearFarBuffer(BufferContext, sizeof(ViewProjNearFar), vk::BufferUsageFlagBits::eUniformBuffer),
      LightsBuffer(BufferContext, as_bytes(Lights), vk::BufferUsageFlagBits::eUniformBuffer),
      SilhouetteColorsBuffer(BufferContext, as_bytes(Colors), vk::BufferUsageFlagBits::eUniformBuffer) {
    // EnTT listeners
    R.on_construct<Selected>().connect<&Scene::OnCreateSelected>(*this);
    R.on_destroy<Selected>().connect<&Scene::OnDestroySelected>(*this);

    R.on_construct<Excitable>().connect<&Scene::OnCreateExcitable>(*this);
    R.on_update<Excitable>().connect<&Scene::OnUpdateExcitable>(*this);
    R.on_destroy<Excitable>().connect<&Scene::OnDestroyExcitable>(*this);

    R.on_construct<ExcitedVertex>().connect<&Scene::OnCreateExcitedVertex>(*this);
    R.on_destroy<ExcitedVertex>().connect<&Scene::OnDestroyExcitedVertex>(*this);

    UpdateEdgeColors();
    UpdateTransformBuffers();

    const auto transform_buffer = CameraUBOBuffer.GetDescriptor();
    const auto lights_buffer = LightsBuffer.GetDescriptor();
    const auto view_proj_near_far_buffer = ViewProjNearFarBuffer.GetDescriptor();
    const auto silhouette_display_buffer = SilhouetteColorsBuffer.GetDescriptor();
    auto descriptors = Pipelines->Main.Renderer.GetDescriptors({
        {SPT::Fill, "CameraUBO", &transform_buffer},
        {SPT::Fill, "LightsUBO", &lights_buffer},
        {SPT::Line, "CameraUBO", &transform_buffer},
        {SPT::Grid, "ViewProjNearFarUBO", &view_proj_near_far_buffer},
        {SPT::SilhouetteEdgeColor, "SilhouetteColorsUBO", &silhouette_display_buffer},
        {SPT::DebugNormals, "CameraUBO", &transform_buffer},
    });
    descriptors.append_range(
        Pipelines->Silhouette.Renderer.GetDescriptors({
            {SPT::SilhouetteDepthObject, "CameraUBO", &transform_buffer},
        })
    );
    Vk.Device.updateDescriptorSets(descriptors, {});

    Pipelines->CompileShaders();

    AddPrimitive(Primitive::Cube, {.Select = true, .Visible = true});
}

Scene::~Scene() {}; // Using unique handles, so no need to manually destroy anything.

void Scene::OnCreateSelected(entt::registry &, entt::entity e) {
    UpdateEntitySelectionOverlays(e);
}
void Scene::OnDestroySelected(entt::registry &, entt::entity e) {
    RemoveEntitySelectionOverlays(e);
}

void Scene::OnCreateExcitable(entt::registry &r, entt::entity e) {
    SelectionModes.insert(SelectionMode::Excite);
    SetSelectionMode(SelectionMode::Excite);
    UpdateHighlightedVertices(e, r.get<Excitable>(e));
}
void Scene::OnUpdateExcitable(entt::registry &r, entt::entity e) {
    UpdateHighlightedVertices(e, r.get<Excitable>(e));
}
void Scene::OnDestroyExcitable(entt::registry &r, entt::entity e) {
    // The last excitable entity is being destroyed.
    if (r.storage<Excitable>().size() == 1) {
        if (SelectionMode == SelectionMode::Excite) SetSelectionMode(*SelectionModes.begin());
        SelectionModes.erase(SelectionMode::Excite);
    }

    static constexpr Excitable EmptyExcitable{};
    UpdateHighlightedVertices(e, EmptyExcitable);
}

void Scene::OnCreateExcitedVertex(entt::registry &r, entt::entity e) {
    auto &excited_vertex = r.get<ExcitedVertex>(e);
    // Orient the camera towards the excited vertex.
    const auto vh = Mesh::VH(excited_vertex.Vertex);
    const auto &mesh = r.get<Mesh>(e);
    const auto &transform = r.get<Model>(e).Transform;
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
         .Transform = {
             .P = vertex_pos + 0.05f * scale_factor * normal,
             .R = glm::rotation(World.Up, normal),
             .S = vec3{scale_factor},
         },
         .Select = false}
    );
}
void Scene::OnDestroyExcitedVertex(entt::registry &r, entt::entity e) {
    DestroyEntity(r.get<ExcitedVertex>(e).IndicatorEntity);
}

vk::ImageView Scene::GetViewportImageView() const { return *Pipelines->Main.Resources->ResolveImage.View; }

void Scene::SetVisible(entt::entity entity, bool visible) {
    const bool already_visible = R.all_of<Visible>(entity);
    if ((visible && already_visible) || (!visible && !already_visible)) return;

    const auto parent = GetParentEntity(entity);
    auto &node = R.get<SceneNode>(entity);
    auto &model_buffer = R.get<ModelsBuffer>(parent).Buffer;
    if (visible) {
        node.ModelBufferIndex = model_buffer.UsedSize / sizeof(Model);
        model_buffer.Insert(as_bytes(R.get<Model>(entity)), model_buffer.UsedSize);
        R.emplace<Visible>(entity);
    } else {
        R.remove<Visible>(entity);
        const uint old_model_index = node.ModelBufferIndex;
        model_buffer.Erase(old_model_index * sizeof(Model), sizeof(Model));
        auto &parent_node = R.get<SceneNode>(parent);
        for (auto child : parent_node.Children) {
            if (auto &child_node = R.get<SceneNode>(child); child_node.ModelBufferIndex > old_model_index) {
                --child_node.ModelBufferIndex;
            }
        }
    }
}

mvk::RenderBuffers Scene::CreateRenderBuffers(RenderBuffers &&buffers) {
    return {
        mvk::UniqueBuffers{BufferContext, as_bytes(buffers.Vertices), vk::BufferUsageFlagBits::eVertexBuffer},
        mvk::UniqueBuffers{BufferContext, as_bytes(buffers.Indices), vk::BufferUsageFlagBits::eIndexBuffer}
    };
}

entt::entity Scene::AddMesh(Mesh &&mesh, MeshCreateInfo info) {
    const auto e = R.create();

    auto node = R.emplace<SceneNode>(e); // No parent or children.
    UpdateTransform(R, e, info.Transform);
    R.emplace<Name>(e, CreateName(R, info.Name));
    R.emplace<ModelsBuffer>(e, mvk::UniqueBuffers{BufferContext, sizeof(Model), vk::BufferUsageFlagBits::eVertexBuffer});
    SetVisible(e, true); // Always set visibility to true first, since this sets up the model buffer/indices.
    if (!info.Visible) SetVisible(e, false);

    MeshElementBuffers element_buffers{};
    for (auto element : AllElements) { // todo only create buffers for viewed elements.
        element_buffers.emplace(element, CreateRenderBuffers(mesh.CreateVertices(element), mesh.CreateIndices(element)));
    }
    R.emplace<MeshBuffers>(e, std::move(element_buffers), MeshElementBuffers{});

    if (ShowBoundingBoxes) {
        R.emplace<BoundingBoxesBuffers>(e, CreateRenderBuffers(CreateBoxVertices(mesh.BoundingBox, EdgeColor), BBox::EdgeIndices));
    }

    R.emplace<Mesh>(e, std::move(mesh));

    if (info.Select) Select(e);
    else if (R.view<Active>().empty()) R.emplace<Active>(e); // If this is the first mesh, set it active by default.

    InvalidateCommandBuffer();

    return e;
}

entt::entity Scene::AddMesh(const fs::path &path, MeshCreateInfo info) {
    auto polymesh = LoadPolyMesh(path);
    if (!polymesh) throw std::runtime_error(std::format("Failed to load mesh: {}", path.string()));
    const auto e = AddMesh({std::move(*polymesh)}, std::move(info));
    R.emplace<Path>(e, path);
    return e;
}

entt::entity Scene::AddPrimitive(Primitive primitive, MeshCreateInfo info) {
    if (info.Name.empty()) info.Name = to_string(primitive);
    const auto e = AddMesh(CreateDefaultPrimitive(primitive), std::move(info));
    R.emplace<Primitive>(e, primitive);
    return e;
}

void Scene::ClearMeshes() {
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Mesh>()) entities.emplace_back(e);
    for (const auto e : entities) DestroyEntity(e);
    InvalidateCommandBuffer();
}

void Scene::ReplaceMesh(entt::entity e, Mesh &&mesh) {
    auto &mesh_buffers = R.get<MeshBuffers>(e);
    for (auto &[element, buffers] : mesh_buffers.Mesh) {
        buffers.Vertices.Update(mesh.CreateVertices(element));
        buffers.Indices.Update(mesh.CreateIndices(element));
    }
    for (auto &[element, buffers] : mesh_buffers.NormalIndicators) {
        buffers.Vertices.Update(mesh.CreateNormalVertices(element));
        buffers.Indices.Update(mesh.CreateNormalIndices(element));
    }
    if (auto buffers = R.try_get<BoundingBoxesBuffers>(e)) {
        buffers->Buffers.Vertices.Update(CreateBoxVertices(mesh.BoundingBox, EdgeColor));
        // Box indices are always the same.
    }
    R.replace<Mesh>(e, std::move(mesh));
}

entt::entity Scene::AddInstance(entt::entity parent, MeshCreateInfo info) {
    const auto e = R.create();
    // For now, we assume one-level deep hierarchy, so we don't allocate a models buffer for the instance.
    R.emplace<SceneNode>(e, parent);
    auto &parent_node = R.get<SceneNode>(parent);
    parent_node.Children.emplace_back(e);
    UpdateTransform(R, e, info.Transform);
    R.emplace<Name>(e, info.Name.empty() ? std::format("{}_instance_{}", GetName(R, parent), parent_node.Children.size()) : CreateName(R, info.Name));
    auto &model_buffer = R.get<ModelsBuffer>(parent).Buffer;
    model_buffer.Reserve(model_buffer.UsedSize + sizeof(Model));
    SetVisible(e, info.Visible);
    if (info.Select) Select(e);
    InvalidateCommandBuffer();

    return e;
}
void Scene::DestroyInstance(entt::entity instance) {
    SetVisible(instance, false);
    std::erase(R.get<SceneNode>(R.get<SceneNode>(instance).Parent).Children, instance);
    R.destroy(instance);
    InvalidateCommandBuffer();
}

void Scene::DestroyEntity(entt::entity e) {
    if (const auto parent_entity = GetParentEntity(e); parent_entity != e) return DestroyInstance(e);

    R.erase<ModelsBuffer>(e);
    R.erase<MeshBuffers>(e);
    R.remove<BoundingBoxesBuffers>(e);
    R.remove<BvhBoxesBuffers>(e);

    const auto &node = R.get<SceneNode>(e);
    for (const auto child : node.Children) R.destroy(child);
    R.destroy(e);
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
    const auto e = FindActiveEntity(R);
    if (auto excitable = R.try_get<Excitable>(e)) {
        UpdateHighlightedVertices(e, *excitable);
    }
}
void Scene::SetEditingElement(MeshElementIndex element) {
    if (R.storage<Active>().empty()) return;

    EditingElement = element;
    UpdateRenderBuffers(GetParentEntity(FindActiveEntity(R)));
}

void Scene::SetTransform(entt::entity e, Transform transform) {
    UpdateTransform(R, e, transform);
    UpdateModelBuffer(e);
    InvalidateCommandBuffer();
}

void Scene::WaitFor(vk::Fence fence) const {
    if (auto wait_result = Vk.Device.waitForFences(fence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    Vk.Device.resetFences(fence);
}

// Get the model VK buffer index.
// Returns `std::nullopt` if the entity is not visible (and thus does not have a rendered model).
std::optional<uint> Scene::GetModelBufferIndex(entt::entity e) {
    if (e == entt::null || !R.all_of<Visible>(e)) return std::nullopt;
    return R.get<SceneNode>(e).ModelBufferIndex;
}

void Scene::UpdateRenderBuffers(entt::entity e) {
    if (const auto *mesh = R.try_get<Mesh>(e)) {
        auto &mesh_buffers = R.get<MeshBuffers>(e);
        const bool is_active = GetParentEntity(FindActiveEntity(R)) == e;
        const Mesh::ElementIndex selected_element{
            is_active && SelectionMode == SelectionMode::Edit       ? EditingElement :
                is_active && SelectionMode == SelectionMode::Excite ? MeshElementIndex{MeshElement::Vertex, int(R.get<Excitable>(e).SelectedVertex())} :
                                                                      MeshElementIndex{}
        };
        for (auto element : AllElements) { // todo only update buffers for viewed elements.
            mesh_buffers.Mesh.at(element).Vertices.Update(mesh->CreateVertices(element, selected_element));
        }
        InvalidateCommandBuffer();
    };
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

void Scene::RecordRenderCommandBuffer() {
    const auto &cb = *RenderCommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(Extent.width), float(Extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, Extent});

    const auto &main = Pipelines->Main;
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, {}, {}, // Dependency flags, memory barriers, buffer memory barriers
        vk::ImageMemoryBarrier{
            {}, // srcAccessMask
            {}, // dstAccessMask
            {}, // oldLayout
            vk::ImageLayout::eColorAttachmentOptimal, // newLayout
            {}, // srcQueueFamilyIndex
            {}, // dstQueueFamilyIndex
            *main.Resources->ResolveImage.Image,
            ColorSubresourceRange
        }
    );

    const auto active_entity = FindActiveEntity(R);
    const auto active_model_buffer_index = GetModelBufferIndex(active_entity);
    const bool render_silhouette = active_model_buffer_index && SelectionMode == SelectionMode::Object;
    if (render_silhouette) {
        // Render all selected mesh instances into a depth/object ID texture.
        const auto &silhouette = Pipelines->Silhouette;
        {
            static const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
            const vk::Rect2D rect{{0, 0}, ToExtent2D(silhouette.Resources->OffscreenImage.Extent)};
            cb.beginRenderPass({*silhouette.Renderer.RenderPass, *silhouette.Resources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);
        }
        {
            static constexpr uint32_t active_id = 1;
            uint32_t selected_id = 2; // 2+
            for (const auto selected_entity : R.view<Selected>()) {
                const auto &shader_pipeline = silhouette.Renderer.ShaderPipelines.at(SPT::SilhouetteDepthObject);
                const auto mesh_entity = GetParentEntity(selected_entity);
                const auto &render_buffers = R.get<MeshBuffers>(mesh_entity).Mesh.at(MeshElement::Vertex);
                const auto &models = R.get<ModelsBuffer>(mesh_entity).Buffer;
                Bind(cb, shader_pipeline, render_buffers, models);
                const auto object_id = R.all_of<Active>(selected_entity) ? active_id : selected_id++;
                cb.pushConstants(*shader_pipeline.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(object_id), &object_id);
                DrawIndexed(cb, render_buffers.Indices, models, *GetModelBufferIndex(selected_entity));
            }
            cb.endRenderPass();
        }
        {
            const auto &silhouette_edge = Pipelines->SilhouetteEdge;
            static const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
            const vk::Rect2D rect{{0, 0}, ToExtent2D(silhouette_edge.Resources->OffscreenImage.Extent)};
            cb.beginRenderPass({*silhouette_edge.Renderer.RenderPass, *silhouette_edge.Resources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);
            const auto &silhouette_edo = silhouette_edge.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepthObject);
            cb.pushConstants(*silhouette_edo.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(SilhouetteEdgeWidth), &SilhouetteEdgeWidth);
            silhouette_edo.RenderQuad(cb);
            cb.endRenderPass();
        }
    }

    // Main rendering pass
    {
        const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {BackgroundColor}};
        const vk::Rect2D rect{{0, 0}, ToExtent2D(main.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*main.Renderer.RenderPass, *main.Resources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);
    }

    // Silhouette edge depth (not color! we render it before mesh depth to avoid overwriting closer depths with further ones)
    if (render_silhouette) main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepth).RenderQuad(cb);

    // Meshes
    for (const auto [_, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
        for (const auto [pipeline, element] : GetPipelineElements(RenderMode, ColorMode)) {
            main.Renderer.Render(cb, pipeline, mesh_buffers.Mesh.at(element), models.Buffer);
        }
    }

    // Silhouette edge color (rendered ontop of meshes)
    if (render_silhouette) {
        const auto &silhouette_edc = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeColor);
        const uint32_t manipulating = TransformGizmo::IsUsing();
        cb.pushConstants(*silhouette_edc.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t), &manipulating);
        silhouette_edc.RenderQuad(cb);
    }

    // Selection overlays
    for (const auto &[_, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
        for (const auto &[element, buffers] : mesh_buffers.NormalIndicators) {
            main.Renderer.Render(cb, SPT::Line, buffers, models.Buffer);
        }
    }
    for (const auto &[_, bvh_boxes, models] : R.view<BvhBoxesBuffers, ModelsBuffer>().each()) {
        main.Renderer.Render(cb, SPT::Line, bvh_boxes.Buffers, models.Buffer);
    }
    for (const auto &[_, bounding_boxes, models] : R.view<BoundingBoxesBuffers, ModelsBuffer>().each()) {
        main.Renderer.Render(cb, SPT::Line, bounding_boxes.Buffers, models.Buffer);
    }

    // Grid lines texture
    if (ShowGrid) main.Renderer.ShaderPipelines.at(SPT::Grid).RenderQuad(cb);

    cb.endRenderPass();
    cb.end();
}

void Scene::InvalidateCommandBuffer() {
    CommandBufferDirty = true;
}

void Scene::UpdateEdgeColors() {
    Mesh::EdgeColor = RenderMode == RenderMode::FacesAndEdges ? MeshEdgeColor : EdgeColor;
    for (const auto e : R.view<Mesh>()) UpdateRenderBuffers(e);
}

void Scene::UpdateTransformBuffers() {
    const float aspect_ratio = Extent.width == 0 || Extent.height == 0 ? 1.f : float(Extent.width) / float(Extent.height);
    const CameraUBO camera_ubo{Camera.View(), Camera.Projection(aspect_ratio), Camera.Position()};
    CameraUBOBuffer.Update(as_bytes(camera_ubo));

    const ViewProjNearFar vpnf{camera_ubo.View, camera_ubo.Proj, Camera.NearClip, Camera.FarClip};
    ViewProjNearFarBuffer.Update(as_bytes(vpnf));
    InvalidateCommandBuffer();
}

void Scene::UpdateModelBuffer(entt::entity e) {
    if (const auto buffer_index = GetModelBufferIndex(e)) {
        const auto &model = R.get<Model>(e);
        R.get<ModelsBuffer>(GetParentEntity(e)).Buffer.Update(as_bytes(model), *buffer_index * sizeof(Model));
    }
}

void Scene::UpdateHighlightedVertices(entt::entity e, const Excitable &excitable) {
    if (auto *mesh = R.try_get<Mesh>(e)) {
        mesh->ClearHighlights();
        if (SelectionMode == SelectionMode::Excite) {
            for (const auto vertex : excitable.ExcitableVertices) {
                mesh->HighlightVertex(Mesh::VH(vertex));
            }
        }
        UpdateRenderBuffers(e);
    }
}

// todo selection overlays for _only selected instances_ (currently all instances of selected meshes)
void Scene::UpdateEntitySelectionOverlays(entt::entity instance_entity) {
    const auto e = GetParentEntity(instance_entity);
    const auto &mesh = R.get<const Mesh>(e);
    auto &mesh_buffers = R.get<MeshBuffers>(e);
    for (const auto element : NormalElements) {
        if (ShownNormalElements.contains(element) && !mesh_buffers.NormalIndicators.contains(element)) {
            mesh_buffers.NormalIndicators.emplace(element, CreateRenderBuffers(mesh.CreateNormalVertices(element), mesh.CreateNormalIndices(element)));
        } else if (!ShownNormalElements.contains(element) && mesh_buffers.NormalIndicators.contains(element)) {
            mesh_buffers.NormalIndicators.erase(element);
        }
    }
    if (ShowBoundingBoxes && !R.all_of<BoundingBoxesBuffers>(e)) {
        R.emplace<BoundingBoxesBuffers>(e, CreateRenderBuffers(CreateBoxVertices(mesh.BoundingBox, EdgeColor), BBox::EdgeIndices));
    } else if (!ShowBoundingBoxes && R.all_of<BoundingBoxesBuffers>(e)) {
        R.remove<BoundingBoxesBuffers>(e);
    }
    if (ShowBvhBoxes && !R.all_of<BvhBoxesBuffers>(e)) {
        R.emplace<BvhBoxesBuffers>(e, CreateRenderBuffers(mesh.CreateBvhBuffers(EdgeColor)));
    } else if (!ShowBvhBoxes && R.all_of<BvhBoxesBuffers>(e)) {
        R.remove<BvhBoxesBuffers>(e);
    }
}

void Scene::RemoveEntitySelectionOverlays(entt::entity instance_entity) {
    const auto e = GetParentEntity(instance_entity);
    if (auto *buffers = R.try_get<MeshBuffers>(e)) buffers->NormalIndicators.clear();
    R.remove<BoundingBoxesBuffers>(e);
    R.remove<BvhBoxesBuffers>(e);
}

using namespace ImGui;

namespace {
constexpr vec2 ToGlm(ImVec2 v) { return std::bit_cast<vec2>(v); }
constexpr vk::Extent2D ToExtent(vec2 e) { return {uint(e.x), uint(e.y)}; }

constexpr std::string Capitalize(std::string_view str) {
    if (str.empty()) return {};

    std::string result{str};
    char &c = result[0];
    if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
    return result;
}

// We already cache the transpose of the inverse transform for all models,
// so save some compute by using that.
ray WorldToLocal(const ray &r, const mat4 &model_i_t) {
    const auto model_i = glm::transpose(model_i_t);
    return {{model_i * vec4{r.o, 1}}, glm::normalize(vec3{model_i * vec4{r.d, 0}})};
}

std::multimap<float, entt::entity> IntersectedEntitiesByDistance(const entt::registry &r, const ray &world_ray) {
    std::multimap<float, entt::entity> entities_by_distance;
    for (const auto &[e, model] : r.view<const Model, const Visible>().each()) {
        const auto &mesh = r.get<const Mesh>(GetParentEntity(r, e));
        if (auto intersection = mesh.Intersect(WorldToLocal(world_ray, model.InvTransform))) {
            entities_by_distance.emplace(intersection->Distance, e);
        }
    }
    return entities_by_distance;
}

entt::entity CycleIntersectedEntity(const entt::registry &r, entt::entity active_entity, const ray &mouse_ray_ws) {
    if (const auto entities_by_distance = IntersectedEntitiesByDistance(r, mouse_ray_ws); !entities_by_distance.empty()) {
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
    for (const auto &[e, model] : r.view<const Model, const Visible>().each()) {
        const auto &mesh = r.get<const Mesh>(GetParentEntity(r, e));
        const auto local_ray = WorldToLocal(world_ray, model.InvTransform);
        if (auto intersection = mesh.Intersect(local_ray);
            intersection && intersection->Distance < nearest_distance) {
            nearest_distance = intersection->Distance;
            nearest = {e, *intersection, local_ray(nearest_distance)};
        }
    }
    return nearest;
}

void WrapMousePos(const ImRect &wrap_rect, vec2 &accumulated_wrap_mouse_delta) {
    const auto &g = *GImGui;
    ImVec2 mouse_delta{0, 0};
    for (int axis = 0; axis < 2; ++axis) {
        if (g.IO.MousePos[axis] >= wrap_rect.Max[axis]) mouse_delta[axis] = -wrap_rect.GetSize()[axis] + 1;
        else if (g.IO.MousePos[axis] <= wrap_rect.Min[axis]) mouse_delta[axis] = wrap_rect.GetSize()[axis] - 1;
    }
    if (mouse_delta != ImVec2{0, 0}) {
        accumulated_wrap_mouse_delta -= ToGlm(mouse_delta);
        TeleportMousePos(g.IO.MousePos + mouse_delta);
    }
}

bool IsSingleClicked(ImGuiMouseButton button) {
    static bool EscapePressed = false; // Escape cancels click
    if (IsMouseClicked(button)) EscapePressed = false;
    if (IsKeyPressed(ImGuiKey_Escape)) EscapePressed = true;
    if (IsMouseReleased(button)) {
        const bool was_escape_pressed = EscapePressed;
        EscapePressed = false;
        if (was_escape_pressed) return false;
    }
    return IsMouseReleased(button) && !IsMouseDragPastThreshold(button);
}
} // namespace

void Scene::Interact() {
    if (Extent.width == 0 || Extent.height == 0) return;

    const auto active_entity = FindActiveEntity(R);
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
    if (!IsMouseDown(ImGuiMouseButton_Left)) {
        R.remove<ExcitedVertex>(active_entity);
        AccumulatedWrapMouseDelta = {0, 0};
    }
    if (TransformGizmo::IsUsing()) {
        // TransformGizmo overrides this mouse cursor during some actions - this is a default.
        SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        WrapMousePos(GetCurrentWindowRead()->InnerClipRect, AccumulatedWrapMouseDelta);
    }
    if (!IsWindowHovered()) return;

    // Mouse wheel for camera rotation, Cmd+wheel to zoom.
    const auto &io = GetIO();
    if (const vec2 wheel{io.MouseWheelH, io.MouseWheel}; wheel != vec2{0, 0}) {
        if (io.KeyCtrl || io.KeySuper) {
            Camera.SetTargetDistance(std::max(Camera.Distance * (1 - wheel.y / 16.f), 0.01f));
        } else {
            Camera.SetTargetYawPitch(Camera.YawPitch + wheel * 0.15f);
        }
    }
    if (!IsSingleClicked(ImGuiMouseButton_Left) || TransformGizmo::IsUsing() || OrientationGizmo::IsActive()) return;

    // Handle mouse selection.
    const auto size = GetContentRegionAvail();
    const auto mouse_pos = (GetMousePos() - GetCursorScreenPos()) / size;
    const auto mouse_ray_ws = Camera.NdcToWorldRay({2 * mouse_pos.x - 1, 1 - 2 * mouse_pos.y}, size.x / size.y);
    if (SelectionMode == SelectionMode::Edit) {
        if (EditingElement.Element != MeshElement::None && active_entity != entt::null && R.all_of<Visible>(active_entity)) {
            const auto &model = R.get<Model>(active_entity);
            const auto mouse_ray = WorldToLocal(mouse_ray_ws, model.InvTransform);
            const auto &mesh = GetActiveMesh();
            {
                const auto nearest_vertex = mesh.FindNearestVertex(mouse_ray);
                if (EditingElement.Element == MeshElement::Vertex) SetEditingElement(Mesh::ElementIndex{nearest_vertex});
                else if (EditingElement.Element == MeshElement::Edge) SetEditingElement(Mesh::ElementIndex{mesh.FindNearestEdge(mouse_ray)});
                else if (EditingElement.Element == MeshElement::Face) SetEditingElement(Mesh::ElementIndex{mesh.FindNearestIntersectingFace(mouse_ray)});
            }
        }
    } else if (SelectionMode == SelectionMode::Object) {
        const auto intersected = CycleIntersectedEntity(R, active_entity, mouse_ray_ws);
        if (intersected != entt::null && IsKeyDown(ImGuiMod_Shift)) {
            if (active_entity == intersected) {
                ToggleSelected(intersected);
            } else {
                R.clear<Active>();
                R.emplace<Active>(intersected);
                R.emplace_or_replace<Selected>(intersected);
                InvalidateCommandBuffer();
            }
        } else {
            Select(intersected);
        }
    } else if (SelectionMode == SelectionMode::Excite) {
        // Excite the nearest entity if it's excitable.
        if (const auto nearest = IntersectNearest(R, mouse_ray_ws)) {
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

void ScenePipelines::SetExtent(vk::Extent2D extent) {
    Main.SetExtent(extent, d, pd, Samples);
    Silhouette.SetExtent(extent, d, pd);
    SilhouetteEdge.SetExtent(extent, d, pd);

    const auto silhouette_info = vk::DescriptorImageInfo{*Silhouette.Resources->ImageSampler, *Silhouette.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
    auto ds = SilhouetteEdge.Renderer.GetDescriptors({
        {SPT::SilhouetteEdgeDepthObject, "SilhouetteSampler", nullptr, &silhouette_info},
    });
    const auto object_id_info = vk::DescriptorImageInfo{*SilhouetteEdge.Resources->ImageSampler, *SilhouetteEdge.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
    const auto depth_info = vk::DescriptorImageInfo{*SilhouetteEdge.Resources->DepthSampler, *SilhouetteEdge.Resources->DepthImage.View, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
    ds.append_range(Main.Renderer.GetDescriptors({
        {SPT::SilhouetteEdgeDepth, "DepthSampler", nullptr, &depth_info},
        {SPT::SilhouetteEdgeColor, "ObjectIdSampler", nullptr, &object_id_info},
    }));
    d.updateDescriptorSets(ds, {});
};

bool Scene::RenderViewport() {
    const auto content_region = ToGlm(GetContentRegionAvail());
    const bool extent_changed = Extent.width != content_region.x || Extent.height != content_region.y;
    if (!extent_changed && !CommandBufferDirty) return false;

    CommandBufferDirty = false;

    const auto transfer_cb = *BufferContext.TransferCb;
    vk::SubmitInfo submit;
    if (extent_changed) {
        Extent = ToExtent(content_region);
        // Must submit the transfer command buffer before updating the pipelines,
        // so we need two submits for the extent change.
        UpdateTransformBuffers(); // Depends on the aspect ratio.
        transfer_cb.end();
        submit.setCommandBuffers(transfer_cb);
        Vk.Queue.submit(submit, *TransferFence);
        WaitFor(*TransferFence);
        Pipelines->SetExtent(Extent);
        RecordRenderCommandBuffer();
        submit.setCommandBuffers(*RenderCommandBuffer);
    } else {
        transfer_cb.end();
        RecordRenderCommandBuffer();
        submit.setCommandBuffers(CommandBuffers);
    }

    Vk.Queue.submit(submit, *RenderFence);
    // The caller may use the resolve image and sampler immediately after `Scene::Render` returns.
    // Returning `true` indicates that the resolve image/sampler have been recreated.
    WaitFor(*RenderFence);
    BufferContext.Reclaimer.Reclaim();
    transfer_cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    return extent_changed;
}

void Scene::RenderOverlay() {
    const auto window_pos = ToGlm(GetWindowPos());
    if (!R.view<Active>().empty()) { // Draw center-dot for active/selected entities
        const auto size = ToGlm(GetContentRegionAvail());
        const auto vp = Camera.Projection(size.x / size.y) * Camera.View();
        for (const auto [e, p] : R.view<const Position, const Visible>().each()) {
            if (!R.any_of<Active, Selected>(e)) continue;

            const auto p_cs = vp * vec4{p.Value, 1}; // World to clip space
            const auto p_ndc = fabsf(p_cs.w) > FLT_EPSILON ? vec3{p_cs} / p_cs.w : vec3{p_cs}; // Clip space to NDC
            const auto p_uv = vec2{p_ndc.x + 1, 1 - p_ndc.y} * 0.5f; // NDC to UV [0,1] (top-left origin)
            const auto p_px = std::bit_cast<ImVec2>(window_pos + p_uv * size); // UV to px
            auto &dl = *GetWindowDrawList();
            dl.AddCircleFilled(p_px, 3.5f, ColorConvertFloat4ToU32(std::bit_cast<ImVec4>(R.all_of<Active>(e) ? Colors.Active : Colors.Selected)), 10);
            dl.AddCircle(p_px, 3.5f, IM_COL32(0, 0, 0, 255), 10, 1.f);
        }
    }

    const auto selected_view = R.view<Selected>();
    if (MGizmo.Show && !selected_view.empty()) {
        // Transform all selected entities around their average position, using the active entity's rotation/scale.
        struct StartTransform {
            Transform T;
        };
        const auto start_transform_view = R.view<const StartTransform>();

        const auto active_transform = GetTransform(R, FindActiveEntity(R));
        const auto p = fold_left(selected_view | transform([&](auto e) { return R.get<Position>(e).Value; }), vec3{}, std::plus{}) / float(selected_view.size());
        if (auto start_delta = TransformGizmo::Draw(
                {{.P = p, .R = active_transform.R, .S = active_transform.S}, MGizmo.Mode},
                MGizmo.Config, Camera, window_pos, ToGlm(GetContentRegionAvail()), ToGlm(GetIO().MousePos) + AccumulatedWrapMouseDelta
            )) {
            const auto &[ts, td] = *start_delta;
            if (start_transform_view.empty()) {
                for (const auto e : selected_view) R.emplace<StartTransform>(e, GetTransform(R, e));
            }
            // Compute delta transform from drag start
            const auto r = ts.R, rT = glm::conjugate(r);
            for (const auto &[e, ts_e_comp] : start_transform_view.each()) {
                const auto &ts_e = ts_e_comp.T;
                const bool frozen = R.all_of<Frozen>(e);
                const auto offset = ts_e.P - ts.P;
                SetTransform(
                    e,
                    {
                        .P = td.P + ts.P + glm::rotate(td.R, frozen ? offset : r * (rT * offset * td.S)),
                        .R = glm::normalize(td.R * ts_e.R),
                        .S = frozen ? ts_e.S : td.S * ts_e.S,
                    }
                );
            }

            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
            WrapMousePos(GetCurrentWindowRead()->InnerClipRect, AccumulatedWrapMouseDelta);
        } else {
            R.clear<StartTransform>();
        }
    }
    { // Orientation gizmo
        static constexpr float OGizmoSize{90};
        const float padding = GetTextLineHeightWithSpacing();
        const auto pos = window_pos + vec2{GetWindowContentRegionMax().x, GetWindowContentRegionMin().y} - vec2{OGizmoSize, 0} + vec2{-padding, padding};
        OrientationGizmo::Draw(pos, OGizmoSize, Camera);
        if (Camera.Tick()) UpdateTransformBuffers();
    }
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

bool SliderUInt(const char *label, uint32_t *v, uint32_t v_min, uint32_t v_max, const char *format = nullptr, ImGuiSliderFlags flags = 0) {
    return SliderScalar(label, ImGuiDataType_U32, v, &v_min, &v_max, format, flags);
}
} // namespace

void Scene::RenderEntityControls(entt::entity active_entity) {
    if (active_entity == entt::null) {
        TextUnformatted("Active object: None");
        return;
    }

    PushID("EntityControls");
    Text("Active entity: %s", GetName(R, active_entity).c_str());
    Indent();

    entt::entity activate_entity = entt::null, delete_entity = entt::null, toggle_selected = entt::null;
    const auto &node = R.get<SceneNode>(active_entity);
    if (Button("Delete")) delete_entity = active_entity;
    if (auto parent_entity = node.Parent; parent_entity != entt::null) {
        AlignTextToFramePadding();
        Text("Parent: %s", GetName(R, parent_entity).c_str());
        SameLine();
        if (active_entity != parent_entity && Button("Activate")) activate_entity = parent_entity;
        SameLine();
        if (Button(R.all_of<Selected>(parent_entity) ? "Deselect" : "Select")) toggle_selected = parent_entity;
    }
    if (!node.Children.empty() && CollapsingHeader("Children")) {
        RenderEntitiesTable("Children", node.Children);
    }

    const auto active_mesh_entity = GetParentEntity(active_entity);
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
        bool model_changed = DragFloat3("Position", &R.get<Position>(active_entity).Value[0], 0.01f);
        // Rotation editor
        {
            int mode_i = R.get<RotationUiVariant>(active_entity).index();
            const char *modes[]{"Quat (WXYZ)", "XYZ Euler", "Axis Angle"};
            if (ImGui::Combo("Rotation Mode", &mode_i, modes, IM_ARRAYSIZE(modes))) {
                R.replace<RotationUiVariant>(active_entity, CreateVariantByIndex<RotationUiVariant>(mode_i));
                SetRotation(R, active_entity, R.get<Rotation>(active_entity).Value);
            }
        }
        model_changed |= std::visit(
            overloaded{
                [&](RotationQuat &v) {
                    if (DragFloat4("Rotation (quat WXYZ)", &v.Value[0], 0.01f)) {
                        R.replace<Rotation>(active_entity, glm::normalize(v.Value));
                        return true;
                    }
                    return false;
                },
                [&](RotationEuler &v) {
                    if (DragFloat3("Rotation (XYZ Euler, deg)", &v.Value[0], 1.f)) {
                        const auto rads = glm::radians(v.Value);
                        R.replace<Rotation>(active_entity, glm::normalize(glm::quat_cast(glm::eulerAngleXYZ(rads.x, rads.y, rads.z))));
                        return true;
                    }
                    return false;
                },
                [&](RotationAxisAngle &v) {
                    bool changed = DragFloat3("Rotation axis (XYZ)", &v.Value[0], 0.01f);
                    changed |= DragFloat("Angle (deg)", &v.Value.w, 1.f);
                    if (changed) {
                        const auto axis = glm::normalize(vec3{v.Value});
                        const auto angle = glm::radians(v.Value.w);
                        R.replace<Rotation>(active_entity, glm::normalize(quat{std::cos(angle / 2), axis * std::sin(angle / 2)}));
                        return true;
                    }
                    return false;
                },
            },
            R.get<RotationUiVariant>(active_entity)
        );

        const bool frozen = R.all_of<Frozen>(active_entity);
        if (frozen) BeginDisabled();
        const auto label = std::format("Scale{}", frozen ? " (frozen)" : "");
        model_changed |= DragFloat3(label.c_str(), &R.get<Scale>(active_entity).Value[0], 0.01f, 0.01f, 10);
        if (frozen) EndDisabled();
        if (model_changed) {
            UpdateModel(R, active_entity);
            UpdateModelBuffer(active_entity);
            InvalidateCommandBuffer();
        }

        using enum TransformGizmo::Type;
        auto &type = MGizmo.Config.Type;
        const bool scale_enabled = !frozen;
        if (!scale_enabled && type == Scale) type = Translate;

        Spacing();
        Checkbox("Gizmo", &MGizmo.Show);
        if (MGizmo.Show) {
            Indent();
            if (TransformGizmo::IsUsing()) {
                SameLine();
                Text("Using");
            }
            if (const auto label = TransformGizmo::ToString(); label != "") {
                SameLine();
                Text("Op: %s", label.data());
            }
            {
                using enum TransformGizmo::Mode;
                auto &mode = MGizmo.Mode;
                AlignTextToFramePadding();
                Text("Mode:");
                SameLine();
                if (RadioButton("Local", mode == Local)) mode = Local;
                SameLine();
                if (RadioButton("World", mode == World)) mode = World;
            }
            if (IsKeyPressed(ImGuiKey_T)) type = Translate;
            if (IsKeyPressed(ImGuiKey_R)) type = Rotate;
            if (scale_enabled && IsKeyPressed(ImGuiKey_S)) type = Scale;
            if (RadioButton("Translate (T)", type == Translate)) type = Translate;
            if (RadioButton("Rotate (R)", type == Rotate)) type = Rotate;
            if (!scale_enabled) BeginDisabled();
            const auto label = std::format("Scale (S){}", !scale_enabled ? " (frozen)" : "");
            if (RadioButton(label.c_str(), type == Scale)) type = Scale;
            if (!scale_enabled) EndDisabled();
            if (RadioButton("Universal", type == Universal)) type = Universal;
            Spacing();
            Checkbox("Snap", &MGizmo.Config.Snap);
            if (MGizmo.Config.Snap) {
                SameLine();
                // todo link/unlink snap values
                DragFloat3("Snap", &MGizmo.Config.SnapValue.x, 1.f, 0.01f, 100.f);
            }
            Unindent();
        }
        Spacing();
        if (TreeNode("Transform matrix")) {
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

    if (delete_entity != entt::null) DestroyEntity(delete_entity);
    else if (activate_entity != entt::null) Select(activate_entity);
    else if (toggle_selected != entt::null) ToggleSelected(toggle_selected);
}

void Scene::RenderControls() {
    if (BeginTabBar("Scene controls")) {
        const auto active_entity = FindActiveEntity(R);
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
                    auto element_selection_mode = int(EditingElement.Element);
                    for (const auto element : AllElements) {
                        auto name = Capitalize(to_string(element));
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
            RenderEntityControls(active_entity);

            if (CollapsingHeader("Add primitive")) {
                PushID("AddPrimitive");
                static auto select_primitive = int(Primitive::Cube);
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
                Pipelines->CompileShaders();
                InvalidateCommandBuffer();
            }
            SeparatorText("Render mode");
            PushID("RenderMode");
            auto render_mode = int(RenderMode);
            bool render_mode_changed = RadioButton("Vertices", &render_mode, int(RenderMode::Vertices));
            SameLine();
            render_mode_changed |= RadioButton("Edges", &render_mode, int(RenderMode::Edges));
            SameLine();
            render_mode_changed |= RadioButton("Faces", &render_mode, int(RenderMode::Faces));
            SameLine();
            render_mode_changed |= RadioButton("Faces and edges", &render_mode, int(RenderMode::FacesAndEdges));
            PopID();

            auto color_mode = int(ColorMode);
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
            {
                SeparatorText("Active/Selected");
                bool color_changed = ColorEdit3("Active color", &Colors.Active[0]);
                color_changed |= ColorEdit3("Selected color", &Colors.Selected[0]);
                if (color_changed) {
                    SilhouetteColorsBuffer.Update(as_bytes(Colors));
                    InvalidateCommandBuffer();
                }
                if (SliderUInt("Edge width", &SilhouetteEdgeWidth, 1, 4)) InvalidateCommandBuffer();
            }
            if (active_entity != entt::null) {
                SeparatorText("Selection overlays");
                AlignTextToFramePadding();
                TextUnformatted("Normals:");
                bool changed = false;
                for (const auto element : NormalElements) {
                    SameLine();
                    bool show_normal_element = ShownNormalElements.contains(element);
                    auto element_name = Capitalize(to_string(element));
                    if (Checkbox(element_name.c_str(), &show_normal_element)) {
                        if (show_normal_element) ShownNormalElements.insert(element);
                        else ShownNormalElements.erase(element);
                        changed = true;
                    }
                }
                if (Checkbox("BVH boxes", &ShowBvhBoxes)) changed = true;
                SameLine();
                if (Checkbox("Bounding boxes", &ShowBoundingBoxes)) changed = true;
                if (changed) {
                    for (auto selected_entity : R.view<Selected>()) {
                        UpdateEntitySelectionOverlays(selected_entity);
                    }
                    InvalidateCommandBuffer();
                }
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
            float fov_deg = glm::degrees(Camera.FieldOfViewRad);
            if (SliderFloat("Field of view (deg)", &fov_deg, 1, 180)) {
                Camera.FieldOfViewRad = glm::radians(fov_deg);
                camera_changed = true;
            }
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
                LightsBuffer.Update(as_bytes(Lights));
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
        entt::entity activate_entity = entt::null, toggle_selected = entt::null;
        for (const auto e : entities) {
            PushID(uint(e));
            TableNextColumn();
            AlignTextToFramePadding();
            if (R.all_of<Active>(e)) TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImGuiCol_TextSelectedBg));
            TextUnformatted(IdString(e).c_str());
            TableNextColumn();
            TextUnformatted(R.get<Name>(e).Value.c_str());
            TableNextColumn();
            if (!R.all_of<Active>(e) && Button("Activate")) activate_entity = e;
            SameLine();
            if (Button(R.all_of<Selected>(e) ? "Deselect" : "Select")) toggle_selected = e;
            PopID();
        }
        if (activate_entity != entt::null) Select(activate_entity);
        else if (toggle_selected != entt::null) ToggleSelected(toggle_selected);
        EndTable();
    }
}
