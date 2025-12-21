#include "Scene.h"
#include "Widgets.h" // imgui

#include "BVH.h"
#include "Bindless.h"
#include "Entity.h"
#include "Excitable.h"
#include "OrientationGizmo.h"
#include "SceneTree.h"
#include "Shader.h"
#include "SvgResource.h"
#include "mesh/Arrow.h"
#include "mesh/MeshIntersection.h"
#include "mesh/MeshRender.h"
#include "mesh/Primitives.h"

#include <entt/entity/registry.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <imgui_internal.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <format>
#include <limits>
#include <print>
#include <ranges>
#include <unordered_map>
#include <variant>

using std::ranges::all_of, std::ranges::distance, std::ranges::find, std::ranges::find_if, std::ranges::fold_left, std::ranges::to;
using std::views::transform;

using namespace he;

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

struct SceneUBO {
    mat4 View{1}, Proj{1};
    vec4 CameraPositionNear{0, 0, 0, 0}; // xyz = camera pos, w = near
    vec4 ViewColorAndAmbient{0, 0, 0, 0};
    vec4 DirectionalColorAndIntensity{0, 0, 0, 0};
    vec4 LightDirectionFar{0, 0, 0, 0}; // xyz = light dir, w = far
    vec4 SilhouetteActive{0, 0, 0, 0};
    vec4 SilhouetteSelected{0, 0, 0, 0};
};

enum class ShaderPipelineType {
    Fill,
    Line,
    Point,
    Grid,
    SilhouetteDepthObject,
    SilhouetteEdgeDepthObject,
    SilhouetteEdgeDepth,
    SilhouetteEdgeColor,
    SelectionFragment,
    DebugNormals,
};
using SPT = ShaderPipelineType;

struct DrawPushConstants {
    uint32_t VertexSlot;
    uint32_t IndexSlot;
    uint32_t ModelSlot;
    uint32_t FirstInstance;
    uint32_t ObjectId;
    uint32_t VertexCountOrHeadImageSlot; // HeadImageSlot for selection fragment
    uint32_t SelectionNodesSlot;
    uint32_t SelectionCounterSlot;
};

struct PipelineRenderer {
    vk::UniqueRenderPass RenderPass;
    std::unordered_map<SPT, ShaderPipeline> ShaderPipelines;

    void CompileShaders();

    // If `model_index` is set, only the model at that index is rendered.
    // Otherwise, all models are rendered.
    void Render(
        vk::CommandBuffer, SPT, const RenderBuffers &, const ModelsBuffer &, DrawPushConstants, std::optional<uint> model_index = std::nullopt
    ) const;
};

struct MeshSelection {
    Element Element{Element::None};
    std::vector<uint32_t> Handles;
    std::optional<uint32_t> ActiveHandle; // Most recently selected element (always in Handles if set)
};

// Component: Vertices highlighted for rendering (in addition to selected elements)
struct MeshHighlightedVertices {
    std::unordered_set<VH> Vertices;
};

entt::entity Scene::GetMeshEntity(entt::entity e) const {
    if (const auto *mesh_instance = R.try_get<MeshInstance>(e)) {
        return mesh_instance->MeshEntity;
    }
    return entt::null;
}
entt::entity Scene::GetActiveMeshEntity() const {
    if (const auto active_entity = FindActiveEntity(R); active_entity != entt::null) {
        return GetMeshEntity(active_entity);
    }
    return entt::null;
}

void Scene::Select(entt::entity e) {
    R.clear<Selected>();
    R.clear<Active>();
    if (e != entt::null) {
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

std::vector<uint32_t> ConvertSelectionElement(const MeshSelection &selection, const Mesh &mesh, Element new_element) {
    if (selection.Element == Element::None || selection.Handles.empty()) return {};
    if (selection.Element == new_element) return selection.Handles;

    const auto handles = selection.Handles | to<std::unordered_set>();
    std::unordered_set<uint32_t> new_handles;
    if (selection.Element == Element::Face) {
        if (new_element == Element::Edge) {
            for (auto f : handles) {
                for (const auto heh : mesh.fh_range(FH{f})) new_handles.emplace(*mesh.GetEdge(heh));
            }
        } else if (new_element == Element::Vertex) {
            for (auto f : handles) {
                for (const auto vh : mesh.fv_range(FH{f})) new_handles.emplace(*vh);
            }
        }
    } else if (selection.Element == Element::Edge) {
        if (new_element == Element::Vertex) {
            for (auto eh_raw : handles) {
                const auto heh = mesh.GetHalfedge(EH{eh_raw}, 0);
                new_handles.emplace(*mesh.GetFromVertex(heh));
                new_handles.emplace(*mesh.GetToVertex(heh));
            }
        } else if (new_element == Element::Face) {
            for (const auto fh : mesh.faces()) {
                const bool all_selected = all_of(mesh.fh_range(fh), [&](auto heh) {
                    return handles.contains(*mesh.GetEdge(heh));
                });
                if (all_selected) new_handles.emplace(*fh);
            }
        }
    } else if (selection.Element == Element::Vertex) {
        if (new_element == Element::Edge) {
            for (const auto eh : mesh.edges()) {
                const auto heh = mesh.GetHalfedge(eh, 0);
                if (handles.contains(*mesh.GetFromVertex(heh)) && handles.contains(*mesh.GetToVertex(heh))) {
                    new_handles.emplace(*eh);
                }
            }
        } else if (new_element == Element::Face) {
            for (const auto fh : mesh.faces()) {
                const bool all_selected = all_of(mesh.fv_range(fh), [&](auto vh) { return handles.contains(*vh); });
                if (all_selected) new_handles.emplace(*fh);
            }
        }
    }
    return new_handles | to<std::vector>();
}

const vk::ClearColorValue Transparent{0, 0, 0, 0};

namespace Format {
constexpr auto Color = vk::Format::eB8G8R8A8Unorm;
constexpr auto Depth = vk::Format::eD32Sfloat;
constexpr auto Float = vk::Format::eR32Sfloat;
constexpr auto Float2 = vk::Format::eR32G32Sfloat;
constexpr auto Uint = vk::Format::eR32Uint;
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

void UpdateTransform(entt::registry &r, entt::entity e, const Transform &t) {
    r.emplace_or_replace<Position>(e, t.P);
    // Avoid replacing rotation UI slider values if the value hasn't changed.
    if (!r.all_of<Rotation>(e) || r.get<Rotation>(e).Value != t.R) SetRotation(r, e, t.R);
    // Frozen entities can't have their scale changed.
    if (!r.all_of<Frozen>(e)) r.emplace_or_replace<Scale>(e, t.S);

    UpdateWorldMatrix(r, e);
}

void Bind(vk::CommandBuffer cb, const ShaderPipeline &shader_pipeline) {
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *shader_pipeline.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *shader_pipeline.PipelineLayout, 0, shader_pipeline.GetDescriptorSet(), {});
}

vk::Extent2D ToExtent2D(vk::Extent3D extent) { return {extent.width, extent.height}; }

constexpr vk::ImageSubresourceRange DepthSubresourceRange{vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
constexpr vk::ImageSubresourceRange ColorSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

BindlessConfig MakeBindlessConfig(vk::PhysicalDevice pd) {
    const auto p2 = pd.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDescriptorIndexingProperties>();
    const auto &props = p2.get<vk::PhysicalDeviceDescriptorIndexingProperties>();
    return {
        .MaxBuffers = std::clamp(
            std::min(props.maxDescriptorSetUpdateAfterBindStorageBuffers, props.maxPerStageDescriptorUpdateAfterBindStorageBuffers),
            1u, 32768u
        ),
        .MaxImages = std::clamp(
            std::min(props.maxDescriptorSetUpdateAfterBindStorageImages, props.maxPerStageDescriptorUpdateAfterBindStorageImages),
            1u, 1024u
        ),
        .MaxUniforms = std::clamp(
            std::min(props.maxDescriptorSetUpdateAfterBindUniformBuffers, props.maxPerStageDescriptorUpdateAfterBindUniformBuffers),
            1u, 64u
        ),
        .MaxSamplers = std::clamp(
            std::min(props.maxDescriptorSetUpdateAfterBindSampledImages, props.maxPerStageDescriptorUpdateAfterBindSamplers),
            1u, 256u
        )
    };
}

struct SelectionNode {
    uint32_t ObjectId;
    float Depth;
    uint32_t Next;
};

struct SelectionCounters {
    uint32_t Count;
    uint32_t Overflow;
};

struct ClickHit {
    float Depth;
    uint32_t ObjectId;
};

struct ClickResult {
    uint32_t Count;
    std::array<ClickHit, 64> Hits;
};

struct ClickSelectPushConstants {
    glm::uvec2 TargetPx;
    uint32_t HeadImageIndex;
    uint32_t SelectionNodesIndex;
    uint32_t ClickResultIndex;
};

struct BoxSelectPushConstants {
    glm::uvec2 BoxMin;
    glm::uvec2 BoxMax;
    uint32_t ObjectCount;
    uint32_t HeadImageIndex;
    uint32_t SelectionNodesIndex;
    uint32_t BoxResultIndex;
};
} // namespace

struct SceneUniqueBuffers {
    static constexpr uint32_t MaxSelectionNodes = 4'000'000; // Matches prior fragment cap
    static constexpr uint32_t MaxSelectableObjects = MaxSelectionNodes;
    static constexpr uint32_t BoxSelectBitsetWords = (MaxSelectableObjects + 31) / 32;

    SceneUniqueBuffers(vk::PhysicalDevice pd, vk::Device d, VkInstance instance, vk::CommandPool command_pool)
        : Ctx{pd, d, instance, command_pool},
          SceneUBO{Ctx, sizeof(SceneUBO), vk::BufferUsageFlagBits::eUniformBuffer},
          SelectionNodeBuffer{Ctx, MaxSelectionNodes * sizeof(SelectionNode), vk::BufferUsageFlagBits::eStorageBuffer},
          SelectionCounterBuffer{*Ctx.Allocator, sizeof(SelectionCounters), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ClickResultBuffer{*Ctx.Allocator, sizeof(ClickResult), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          BoxSelectBitsetBuffer{*Ctx.Allocator, BoxSelectBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer} {
    }

    const ClickResult &GetClickResult() const { return *reinterpret_cast<const ClickResult *>(ClickResultBuffer.GetData().data()); }
    vk::DescriptorBufferInfo GetBoxSelectBitsetDescriptor() const { return {*BoxSelectBitsetBuffer, 0, BoxSelectBitsetWords * sizeof(uint32_t)}; }

    RenderBuffers CreateRenderBuffers(std::vector<Vertex3D> &&vertices, std::vector<uint> &&indices) const {
        const auto vertex_usage = vk::BufferUsageFlagBits::eStorageBuffer;
        const auto index_usage = vk::BufferUsageFlagBits::eStorageBuffer;
        return {
            {Ctx, as_bytes(vertices), vertex_usage},
            {Ctx, as_bytes(indices), index_usage}
        };
    }
    template<size_t N>
    RenderBuffers CreateRenderBuffers(std::vector<Vertex3D> &&vertices, const std::array<uint, N> &indices) const {
        const auto vertex_usage = vk::BufferUsageFlagBits::eStorageBuffer;
        const auto index_usage = vk::BufferUsageFlagBits::eStorageBuffer;
        return {
            {Ctx, as_bytes(vertices), vertex_usage},
            {Ctx, as_bytes(indices), index_usage}
        };
    }

    mvk::BufferContext Ctx;
    mvk::UniqueBuffers SceneUBO;
    mvk::UniqueBuffers SelectionNodeBuffer;
    mvk::UniqueBuffer SelectionCounterBuffer;
    mvk::UniqueBuffer ClickResultBuffer;
    mvk::UniqueBuffer BoxSelectBitsetBuffer;
};

void PipelineRenderer::CompileShaders() {
    for (auto &shader_pipeline : std::views::values(ShaderPipelines)) shader_pipeline.Compile(*RenderPass);
}

void PipelineRenderer::Render(
    vk::CommandBuffer cb, SPT spt, const RenderBuffers &render_buffers, const ModelsBuffer &models, DrawPushConstants pc, std::optional<uint> model_index
) const {
    Bind(cb, ShaderPipelines.at(spt));
    pc.FirstInstance = model_index.value_or(0);
    const auto instance_count = model_index.has_value() ? 1 : models.Buffer.UsedSize / sizeof(WorldMatrix);
    cb.pushConstants(*ShaderPipelines.at(spt).PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPushConstants), &pc);
    cb.draw(render_buffers.Indices.UsedSize / sizeof(uint32_t), instance_count, 0, 0);
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

    const auto cb = *UniqueBuffers->Ctx.TransferCb;
    {
        // Write the bitmap into a temporary staging buffer.
        mvk::UniqueBuffer staging_buffer{*UniqueBuffers->Ctx.Allocator, as_bytes(data), mvk::MemoryUsage::CpuOnly};

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

    UniqueBuffers->Ctx.Reclaimer.Reclaim();
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    return image;
}

// Pipeline definitions
namespace {
struct MainPipeline {
    static PipelineRenderer CreateRenderer(
        vk::Device d, vk::SampleCountFlagBits msaa_samples,
        vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}
    ) {
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

        const PipelineContext ctx{d, shared_layout, shared_set, msaa_samples};
        const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPushConstants)};

        // Can't construct this map in-place with pairs because `ShaderPipeline` doesn't have a copy constructor.
        std::unordered_map<SPT, ShaderPipeline> pipelines;
        pipelines.emplace(
            SPT::Fill,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "Lighting.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                CreateColorBlendAttachment(true), CreateDepthStencil(), draw_pc
            )
        );
        pipelines.emplace(
            SPT::Line,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "VertexColor.frag"}}},
                {},
                vk::PolygonMode::eLine, vk::PrimitiveTopology::eLineList,
                CreateColorBlendAttachment(true), CreateDepthStencil(), draw_pc
            )
        );
        pipelines.emplace(
            SPT::Point,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "VertexPoint.vert"}, {ShaderType::eFragment, "VertexPoint.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
                CreateColorBlendAttachment(true), CreateDepthStencil(), draw_pc, -1.0f
            )
        );
        pipelines.emplace(
            SPT::Grid,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "GridLines.vert"}, {ShaderType::eFragment, "GridLines.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
                CreateColorBlendAttachment(true), CreateDepthStencil(true, false)
            )
        );

        // Render the silhouette edge depth regardless of the tested depth value.
        // We should be able to just disable depth tests and enable depth writes, but it seems that some GPUs or drivers
        // optimize out depth writes when depth testing is disabled, so instead we configure a depth test that always passes.
        pipelines.emplace(
            SPT::SilhouetteEdgeDepth,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SampleDepth.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
                CreateColorBlendAttachment(true), CreateDepthStencil(true, true, vk::CompareOp::eAlways),
                vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)}
            )
        );
        // Render silhouette edge color regardless of the tested depth value.
        pipelines.emplace(
            SPT::SilhouetteEdgeColor,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SilhouetteEdgeColor.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
                CreateColorBlendAttachment(true), CreateDepthStencil(false, false),
                vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t) * 2} // Manipulating flag + sampler index
            )
        );
        pipelines.emplace(
            SPT::DebugNormals,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "VertexTransform.vert"}, {ShaderType::eFragment, "Normals.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                CreateColorBlendAttachment(true), CreateDepthStencil(), draw_pc
            )
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
    static PipelineRenderer CreateRenderer(vk::Device d, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}) {
        const std::vector<vk::AttachmentDescription> attachments{
            // We need to test and write depth since we want silhouette edges to respect mutual occlusion when multiple meshes are selected.
            {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
            // Single-sampled offscreen "image" of two channels: depth and object ID.
            {{}, Format::Float2, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        };
        const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
        const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, &depth_attachment_ref};

        const PipelineContext ctx{d, shared_layout, shared_set, vk::SampleCountFlagBits::e1};
        const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPushConstants)};
        std::unordered_map<SPT, ShaderPipeline> pipelines;
        pipelines.emplace(
            SPT::SilhouetteDepthObject,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "DepthObject.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                CreateColorBlendAttachment(false), CreateDepthStencil(), draw_pc
            )
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
    static PipelineRenderer CreateRenderer(vk::Device d, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}) {
        const std::vector<vk::AttachmentDescription> attachments{
            {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilReadOnlyOptimal},
            {{}, Format::Float, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
        };
        const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        const vk::AttachmentReference color_attachment_ref{1, vk::ImageLayout::eColorAttachmentOptimal};
        const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref, nullptr, &depth_attachment_ref};

        const PipelineContext ctx{d, shared_layout, shared_set, vk::SampleCountFlagBits::e1};
        std::unordered_map<SPT, ShaderPipeline> pipelines;
        pipelines.emplace(
            SPT::SilhouetteEdgeDepthObject,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "TexQuad.vert"}, {ShaderType::eFragment, "SilhouetteEdgeDepthObject.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleStrip,
                CreateColorBlendAttachment(false), CreateDepthStencil(),
                vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t) * 2}
            )
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

struct SelectionFragmentPipeline {
    static PipelineRenderer CreateRenderer(vk::Device d, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}) {
        const std::vector<vk::AttachmentDescription> attachments{
            // Only need depth for depth testing, don't need to store it
            {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        };
        const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 0, nullptr, nullptr, &depth_attachment_ref};

        const PipelineContext ctx{d, shared_layout, shared_set, vk::SampleCountFlagBits::e1};
        const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPushConstants)};
        std::unordered_map<SPT, ShaderPipeline> pipelines;
        pipelines.emplace(
            SPT::SelectionFragment,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "SelectionFragment.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                {}, CreateDepthStencil(), draw_pc // No color attachment
            )
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
              HeadImage{mvk::CreateImage(
                  d, pd,
                  {{},
                   vk::ImageType::e2D,
                   Format::Uint,
                   vk::Extent3D{extent, 1},
                   1,
                   1,
                   vk::SampleCountFlagBits::e1,
                   vk::ImageTiling::eOptimal,
                   vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
                   vk::SharingMode::eExclusive},
                  {{}, {}, vk::ImageViewType::e2D, Format::Uint, {}, ColorSubresourceRange}
              )} {
            const std::array image_views{*DepthImage.View};
            Framebuffer = d.createFramebufferUnique({{}, render_pass, image_views, extent.width, extent.height, 1});
        }

        mvk::ImageResource DepthImage, HeadImage;
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
    ScenePipelines(
        vk::Device d, vk::PhysicalDevice pd,
        vk::DescriptorSetLayout selection_layout = {}, vk::DescriptorSet selection_set = {}
    )
        : Device(d), PhysicalDevice(pd), Samples{GetMaxUsableSampleCount(pd)},
          Main{MainPipeline::CreateRenderer(d, Samples, selection_layout, selection_set), nullptr},
          Silhouette{SilhouettePipeline::CreateRenderer(d, selection_layout, selection_set), nullptr},
          SilhouetteEdge{SilhouetteEdgePipeline::CreateRenderer(d, selection_layout, selection_set), nullptr},
          SelectionFragment{SelectionFragmentPipeline::CreateRenderer(d, selection_layout, selection_set), nullptr},
          ClickSelect{
              d, Shaders{{{ShaderType::eCompute, "ClickSelect.comp"}}},
              vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ClickSelectPushConstants)},
              selection_layout,
              selection_set
          },
          BoxSelect{
              d, Shaders{{{ShaderType::eCompute, "BoxSelect.comp"}}},
              vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(BoxSelectPushConstants)},
              selection_layout,
              selection_set
          } {}

    vk::Device Device;
    vk::PhysicalDevice PhysicalDevice;
    vk::SampleCountFlagBits Samples;

    MainPipeline Main;
    SilhouettePipeline Silhouette;
    SilhouetteEdgePipeline SilhouetteEdge;
    SelectionFragmentPipeline SelectionFragment;
    ComputePipeline ClickSelect;
    ComputePipeline BoxSelect;

    void SetExtent(vk::Extent2D);
    // These do _not_ re-submit the command buffer. Callers must do so manually if needed.
    void CompileShaders() {
        Main.Renderer.CompileShaders();
        Silhouette.Renderer.CompileShaders();
        SilhouetteEdge.Renderer.CompileShaders();
        SelectionFragment.Renderer.CompileShaders();
        ClickSelect.Compile();
        BoxSelect.Compile();
    }
};

struct Scene::SelectionBindlessHandles {
    explicit SelectionBindlessHandles(BindlessAllocator &alloc) {
        HeadImage = alloc.Allocate(SlotType::Image);
        SelectionNodes = alloc.Allocate(SlotType::Buffer);
        SelectionCounter = alloc.Allocate(SlotType::Buffer);
        ClickResult = alloc.Allocate(SlotType::Buffer);
        BoxResult = alloc.Allocate(SlotType::Buffer);
        ObjectIdSampler = alloc.Allocate(SlotType::Sampler);
        DepthSampler = alloc.Allocate(SlotType::Sampler);
        SilhouetteSampler = alloc.Allocate(SlotType::Sampler);
    }

    uint32_t HeadImage{InvalidSlot};
    uint32_t SelectionNodes{InvalidSlot};
    uint32_t SelectionCounter{InvalidSlot};
    uint32_t ClickResult{InvalidSlot};
    uint32_t BoxResult{InvalidSlot};
    uint32_t ObjectIdSampler{InvalidSlot};
    uint32_t DepthSampler{InvalidSlot};
    uint32_t SilhouetteSampler{InvalidSlot};

    void Release(BindlessAllocator &alloc) {
        if (HeadImage != InvalidSlot) alloc.Release(SlotType::Image, HeadImage);
        if (SelectionNodes != InvalidSlot) alloc.Release(SlotType::Buffer, SelectionNodes);
        if (SelectionCounter != InvalidSlot) alloc.Release(SlotType::Buffer, SelectionCounter);
        if (ClickResult != InvalidSlot) alloc.Release(SlotType::Buffer, ClickResult);
        if (BoxResult != InvalidSlot) alloc.Release(SlotType::Buffer, BoxResult);
        if (ObjectIdSampler != InvalidSlot) alloc.Release(SlotType::Sampler, ObjectIdSampler);
        if (DepthSampler != InvalidSlot) alloc.Release(SlotType::Sampler, DepthSampler);
        if (SilhouetteSampler != InvalidSlot) alloc.Release(SlotType::Sampler, SilhouetteSampler);
    }
};

Scene::Scene(SceneVulkanResources vc, entt::registry &r)
    : Vk{vc},
      R{r},
      CommandPool{Vk.Device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, Vk.QueueFamily})},
      RenderCommandBuffer{std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())},
      RenderFence{Vk.Device.createFenceUnique({})},
      TransferFence{Vk.Device.createFenceUnique({})},
      ClickCommandBuffer{std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())},
      Bindless{std::make_unique<BindlessResources>(Vk.Device, MakeBindlessConfig(Vk.PhysicalDevice))},
      BindlessAlloc{std::make_unique<BindlessAllocator>(*Bindless)},
      SelectionHandles{std::make_unique<SelectionBindlessHandles>(*BindlessAlloc)},
      Pipelines{std::make_unique<ScenePipelines>(
          Vk.Device, Vk.PhysicalDevice,
          Bindless->SetLayout.get(), *Bindless->DescriptorSet
      )},
      UniqueBuffers{std::make_unique<SceneUniqueBuffers>(Vk.PhysicalDevice, Vk.Device, Vk.Instance, *CommandPool)} {
    // EnTT listeners
    R.on_construct<Selected>().connect<&Scene::OnCreateSelected>(*this);
    R.on_destroy<Selected>().connect<&Scene::OnDestroySelected>(*this);

    R.on_construct<Excitable>().connect<&Scene::OnCreateExcitable>(*this);
    R.on_update<Excitable>().connect<&Scene::OnUpdateExcitable>(*this);
    R.on_destroy<Excitable>().connect<&Scene::OnDestroyExcitable>(*this);

    R.on_construct<ExcitedVertex>().connect<&Scene::OnCreateExcitedVertex>(*this);
    R.on_destroy<ExcitedVertex>().connect<&Scene::OnDestroyExcitedVertex>(*this);

    R.on_destroy<MeshBuffers>().connect<&Scene::OnDestroyMeshBuffers>(*this);
    R.on_destroy<ModelsBuffer>().connect<&Scene::OnDestroyModelsBuffer>(*this);
    R.on_destroy<BoundingBoxesBuffers>().connect<&Scene::OnDestroyBoundingBoxesBuffers>(*this);
    R.on_destroy<BvhBoxesBuffers>().connect<&Scene::OnDestroyBvhBoxesBuffers>(*this);

    UpdateEdgeColors();
    UpdateTransformBuffers();
    BoxSelectZeroBits.assign(SceneUniqueBuffers::BoxSelectBitsetWords, 0);

    Pipelines->CompileShaders();

    { // Default scene content
        static constexpr int kGrid = 10;
        static constexpr float kSpacing = 2.0f;
        int count = 0;
        for (int z = 0; z < kGrid; ++z) {
            for (int y = 0; y < kGrid; ++y) {
                for (int x = 0; x < kGrid; ++x) {
                    if (count >= kGrid * kGrid * kGrid) break;
                    const Transform t{.P = vec3{float(x) * kSpacing, float(y) * kSpacing, float(z) * kSpacing}};
                    const auto e = AddMesh(
                        CreateDefaultPrimitive(PrimitiveType::Cube),
                        {.Name = ToString(PrimitiveType::Cube), .Transform = t, .Select = MeshCreateInfo::SelectBehavior::None}
                    );
                    R.emplace<PrimitiveType>(e, PrimitiveType::Cube);
                    ++count;
                }
            }
        }
        const float extent = float(kGrid - 1) * kSpacing;
        const vec3 center{extent * 0.5f, extent * 0.5f, extent * 0.5f};
        const vec3 position = center + vec3{extent, extent, extent};
        Camera = ::Camera{position, center, glm::radians(60.f), 0.01f, 500.f};
    }
}

Scene::~Scene() {
    if (BindlessAlloc && SelectionHandles) SelectionHandles->Release(*BindlessAlloc);
}

void Scene::LoadIcons(vk::Device device) {
    const auto RenderBitmap = [this](std::span<const std::byte> data, uint32_t width, uint32_t height) {
        return RenderBitmapToImage(data, width, height);
    };

    device.waitIdle();
    Icons.Select = std::make_unique<SvgResource>(device, RenderBitmap, "res/svg/select.svg");
    Icons.SelectBox = std::make_unique<SvgResource>(device, RenderBitmap, "res/svg/select_box.svg");
    Icons.Move = std::make_unique<SvgResource>(device, RenderBitmap, "res/svg/move.svg");
    Icons.Rotate = std::make_unique<SvgResource>(device, RenderBitmap, "res/svg/rotate.svg");
    Icons.Scale = std::make_unique<SvgResource>(device, RenderBitmap, "res/svg/scale.svg");
    Icons.Universal = std::make_unique<SvgResource>(device, RenderBitmap, "res/svg/transform.svg");
}

void Scene::OnCreateSelected(entt::registry &, entt::entity e) {
    UpdateEntitySelectionOverlays(e);
}
void Scene::OnDestroySelected(entt::registry &r, entt::entity e) {
    if (const auto *mesh_instance = r.try_get<MeshInstance>(e)) {
        const auto mesh_entity = mesh_instance->MeshEntity;
        if (auto *buffers = r.try_get<MeshBuffers>(mesh_entity)) {
            for (auto &[_, indicator] : buffers->NormalIndicators) ReleaseRenderBufferBindless(indicator);
            buffers->NormalIndicators.clear();
        }
        r.remove<BoundingBoxesBuffers>(mesh_entity);
        r.remove<BvhBoxesBuffers>(mesh_entity);
    }
}

void Scene::OnCreateExcitable(entt::registry &r, entt::entity e) {
    InteractionModes.insert(InteractionMode::Excite);
    SetInteractionMode(InteractionMode::Excite);
    UpdateHighlightedVertices(r.get<MeshInstance>(e).MeshEntity, r.get<Excitable>(e));
}
void Scene::OnUpdateExcitable(entt::registry &r, entt::entity e) {
    UpdateHighlightedVertices(r.get<MeshInstance>(e).MeshEntity, r.get<Excitable>(e));
}
void Scene::OnDestroyExcitable(entt::registry &r, entt::entity e) {
    // The last excitable entity is being destroyed.
    if (r.storage<Excitable>().size() == 1) {
        if (InteractionMode == InteractionMode::Excite) SetInteractionMode(*InteractionModes.begin());
        InteractionModes.erase(InteractionMode::Excite);
    }

    static constexpr Excitable EmptyExcitable{};
    UpdateHighlightedVertices(e, EmptyExcitable);
}

void Scene::OnCreateExcitedVertex(entt::registry &r, entt::entity e) {
    const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
    auto &excited_vertex = r.get<ExcitedVertex>(e);
    // Orient the camera towards the excited vertex.
    const auto vh = VH(excited_vertex.Vertex);
    const auto &mesh = r.get<Mesh>(mesh_entity);
    const auto &bbox = r.get<BBox>(mesh_entity);
    const auto &transform = r.get<WorldMatrix>(e).M;
    const vec3 vertex_pos{transform * vec4{mesh.GetPosition(vh), 1}};
    Camera.SetTargetDirection(glm::normalize(vertex_pos - Camera.Target));

    // Create vertex indicator arrow pointing at the excited vertex.
    const vec3 normal{transform * vec4{mesh.GetNormal(vh), 0}};
    const float scale_factor = 0.1f * bbox.DiagonalLength();
    auto vertex_indicator_mesh = Arrow();
    vertex_indicator_mesh.SetColor({1, 0, 0, 1});
    excited_vertex.IndicatorEntity = AddMesh(
        std::move(vertex_indicator_mesh),
        {.Name = "Excite vertex indicator",
         .Transform = {
             .P = vertex_pos + 0.05f * scale_factor * normal,
             .R = glm::rotation(World.Up, normal),
             .S = vec3{scale_factor},
         },
         .Select = MeshCreateInfo::SelectBehavior::None}
    );
}
void Scene::OnDestroyExcitedVertex(entt::registry &r, entt::entity e) {
    Destroy(r.get<ExcitedVertex>(e).IndicatorEntity);
}

void Scene::OnDestroyMeshBuffers(entt::registry &r, entt::entity e) {
    if (auto *mesh_buffers = r.try_get<MeshBuffers>(e)) {
        ReleaseRenderBufferBindless(mesh_buffers->Faces);
        ReleaseRenderBufferBindless(mesh_buffers->Edges);
        ReleaseRenderBufferBindless(mesh_buffers->Vertices);
        for (auto &[_, buffers] : mesh_buffers->NormalIndicators) ReleaseRenderBufferBindless(buffers);
    }
}

void Scene::OnDestroyModelsBuffer(entt::registry &r, entt::entity e) {
    if (auto *models = r.try_get<ModelsBuffer>(e)) ReleaseModelBufferBindless(*models);
}

void Scene::OnDestroyBoundingBoxesBuffers(entt::registry &r, entt::entity e) {
    if (auto *bbox = r.try_get<BoundingBoxesBuffers>(e)) ReleaseRenderBufferBindless(bbox->Buffers);
}

void Scene::OnDestroyBvhBoxesBuffers(entt::registry &r, entt::entity e) {
    if (auto *bvh = r.try_get<BvhBoxesBuffers>(e)) ReleaseRenderBufferBindless(bvh->Buffers);
}

vk::ImageView Scene::GetViewportImageView() const { return *Pipelines->Main.Resources->ResolveImage.View; }

void Scene::SetVisible(entt::entity entity, bool visible) {
    const bool already_visible = R.all_of<Visible>(entity);
    if ((visible && already_visible) || (!visible && !already_visible)) return;

    const auto mesh_entity = R.get<MeshInstance>(entity).MeshEntity;
    auto &models = R.get<ModelsBuffer>(mesh_entity);
    auto &model_buffer = models.Buffer;
    if (visible) {
        auto &render_instance = R.emplace_or_replace<RenderInstance>(entity);
        render_instance.BufferIndex = model_buffer.UsedSize / sizeof(WorldMatrix);
        model_buffer.Insert(as_bytes(R.get<WorldMatrix>(entity)), model_buffer.UsedSize);
        R.emplace<Visible>(entity);
    } else {
        R.remove<Visible>(entity);
        const uint old_model_index = R.get<RenderInstance>(entity).BufferIndex;
        model_buffer.Erase(old_model_index * sizeof(WorldMatrix), sizeof(WorldMatrix));
        // Update buffer indices for all instances of this mesh that have higher indices
        for (const auto [other_entity, mesh_instance, render_instance] : R.view<MeshInstance, RenderInstance>().each()) {
            if (mesh_instance.MeshEntity == mesh_entity && render_instance.BufferIndex > old_model_index) {
                --render_instance.BufferIndex;
            }
        }
        // Also check if the mesh entity itself is visible (it might not have MeshInstance)
        if (mesh_entity != entity) {
            if (auto *mesh_render_instance = R.try_get<RenderInstance>(mesh_entity)) {
                if (mesh_render_instance->BufferIndex > old_model_index) {
                    --mesh_render_instance->BufferIndex;
                }
            }
        }
    }
    UpdateModelBufferBindless(models);
    InvalidateCommandBuffer();
}

entt::entity Scene::AddMesh(Mesh &&mesh, MeshCreateInfo info) {
    const auto mesh_entity = R.create();
    { // Mesh data
        const auto bbox = MeshRender::ComputeBoundingBox(mesh);
        R.emplace<BBox>(mesh_entity, bbox);
        R.emplace<BVH>(mesh_entity, MeshRender::CreateFaceBoundingBoxes(mesh));

        auto face_buffers = UniqueBuffers->CreateRenderBuffers(
            MeshRender::CreateFaceVertices(mesh, SmoothShading),
            MeshRender::CreateFaceIndices(mesh)
        );
        auto edge_buffers = UniqueBuffers->CreateRenderBuffers(
            MeshRender::CreateEdgeVertices(mesh, Element::Vertex),
            MeshRender::CreateEdgeIndices(mesh)
        );
        auto vertex_buffers = UniqueBuffers->CreateRenderBuffers(
            MeshRender::CreateVertexPoints(mesh, Element::None, {}),
            MeshRender::CreateVertexIndices(mesh)
        );

        R.emplace<Mesh>(mesh_entity, std::move(mesh));
        R.emplace<MeshSelection>(mesh_entity);
        R.emplace<MeshHighlightedVertices>(mesh_entity);
        R.emplace<ModelsBuffer>(
            mesh_entity,
            mvk::UniqueBuffers{UniqueBuffers->Ctx, sizeof(WorldMatrix), vk::BufferUsageFlagBits::eStorageBuffer}
        );
        R.emplace<MeshBuffers>(mesh_entity, std::move(face_buffers), std::move(edge_buffers), std::move(vertex_buffers));
        UpdateRenderBufferBindless(R.get<MeshBuffers>(mesh_entity).Faces);
        UpdateRenderBufferBindless(R.get<MeshBuffers>(mesh_entity).Edges);
        UpdateRenderBufferBindless(R.get<MeshBuffers>(mesh_entity).Vertices);
        UpdateModelBufferBindless(R.get<ModelsBuffer>(mesh_entity));
        if (ShowBoundingBoxes) {
            auto buffers = UniqueBuffers->CreateRenderBuffers(CreateBoxVertices(bbox, EdgeColor), BBox::EdgeIndices);
            UpdateRenderBufferBindless(buffers);
            R.emplace<BoundingBoxesBuffers>(mesh_entity, std::move(buffers));
        }
    }

    const auto instance_entity = R.create();
    R.emplace<MeshInstance>(instance_entity, mesh_entity);
    UpdateTransform(R, instance_entity, info.Transform);
    R.emplace<Name>(instance_entity, CreateName(R, info.Name));

    auto &model_buffer = R.get<ModelsBuffer>(mesh_entity).Buffer;
    model_buffer.Reserve(model_buffer.UsedSize + sizeof(WorldMatrix));
    SetVisible(instance_entity, true); // Always set visibility to true first, since this sets up the model buffer/indices.
    if (!info.Visible) SetVisible(instance_entity, false);

    switch (info.Select) {
        case MeshCreateInfo::SelectBehavior::Exclusive:
            Select(instance_entity);
            break;
        case MeshCreateInfo::SelectBehavior::Additive:
            R.emplace<Selected>(instance_entity);
            // Fallthrough
        case MeshCreateInfo::SelectBehavior::None:
            // If this is the first mesh, set it active by default.
            if (R.storage<Active>().empty()) R.emplace<Active>(instance_entity);
            break;
    }

    InvalidateCommandBuffer();
    return instance_entity;
}

entt::entity Scene::AddMesh(const fs::path &path, MeshCreateInfo info) {
    auto mesh = Mesh::Load(path);
    if (!mesh) throw std::runtime_error(std::format("Failed to load mesh: {}", path.string()));
    const auto e = AddMesh(std::move(*mesh), std::move(info));
    R.emplace<Path>(e, path);
    return e;
}

entt::entity Scene::Duplicate(entt::entity e, std::optional<MeshCreateInfo> info) {
    const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
    const auto e_new = AddMesh(
        Mesh{R.get<const Mesh>(mesh_entity)},
        info.value_or(MeshCreateInfo{
            .Name = std::format("{}_copy", GetName(R, e)),
            .Transform = GetTransform(R, e),
            .Select = R.all_of<Selected>(e) ? MeshCreateInfo::SelectBehavior::Additive : MeshCreateInfo::SelectBehavior::None,
            .Visible = R.all_of<Visible>(e),
        })
    );
    if (auto primitive_type = R.try_get<PrimitiveType>(mesh_entity)) R.emplace<PrimitiveType>(e_new, *primitive_type);
    return e_new;
}

entt::entity Scene::DuplicateLinked(entt::entity e, std::optional<MeshCreateInfo> info) {
    const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
    const auto e_new = R.create();
    {
        uint instance_count = 0; // Count instances for naming (first duplicated instance is _1, etc.)
        for (const auto [_, mesh_instance] : R.view<MeshInstance>().each()) {
            if (mesh_instance.MeshEntity == mesh_entity) ++instance_count;
        }
        R.emplace<Name>(e_new, !info || info->Name.empty() ? std::format("{}_{}", GetName(R, e), instance_count) : CreateName(R, info->Name));
    }
    R.emplace<MeshInstance>(e_new, mesh_entity);
    {
        auto &model_buffer = R.get<ModelsBuffer>(mesh_entity).Buffer;
        model_buffer.Reserve(model_buffer.UsedSize + sizeof(WorldMatrix));
    }
    UpdateTransform(R, e_new, info ? info->Transform : GetTransform(R, e));
    SetVisible(e_new, !info || info->Visible);

    if (!info || info->Select == MeshCreateInfo::SelectBehavior::Additive) R.emplace<Selected>(e_new);
    else if (info->Select == MeshCreateInfo::SelectBehavior::Exclusive) Select(e_new);

    InvalidateCommandBuffer();
    return e_new;
}

void Scene::Delete() {
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) entities.emplace_back(e);
    for (const auto e : entities) Destroy(e);
}

void Scene::Duplicate() {
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) entities.emplace_back(e);
    for (const auto e : entities) {
        const auto new_e = Duplicate(e);
        if (R.all_of<Active>(e)) {
            R.remove<Active>(e);
            R.emplace<Active>(new_e);
        }
        R.remove<Selected>(e);
    }
    StartScreenTransform = TransformGizmo::TransformType::Translate;
}
void Scene::DuplicateLinked() {
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) entities.emplace_back(e);
    for (const auto e : entities) {
        const auto new_e = DuplicateLinked(e);
        if (R.all_of<Active>(e)) {
            R.remove<Active>(e);
            R.emplace<Active>(new_e);
        }
        R.remove<Selected>(e);
    }
    StartScreenTransform = TransformGizmo::TransformType::Translate;
}

void Scene::ClearMeshes() {
    std::vector<entt::entity> entities;
    for (const auto e : R.view<MeshInstance>()) entities.emplace_back(e);
    for (const auto e : entities) Destroy(e);
    InvalidateCommandBuffer();
}

void Scene::ReplaceMesh(entt::entity e, Mesh &&mesh) {
    // Update components
    const auto bbox = MeshRender::ComputeBoundingBox(mesh);
    R.replace<BBox>(e, bbox);
    R.replace<BVH>(e, BVH{MeshRender::CreateFaceBoundingBoxes(mesh)});
    R.replace<Mesh>(e, std::move(mesh));

    const auto &pm = R.get<Mesh>(e);
    auto &mesh_buffers = R.get<MeshBuffers>(e);
    mesh_buffers.Faces.Vertices.Update(MeshRender::CreateFaceVertices(pm, SmoothShading));
    mesh_buffers.Faces.Indices.Update(MeshRender::CreateFaceIndices(pm));
    mesh_buffers.Edges.Vertices.Update(MeshRender::CreateEdgeVertices(pm, Element::Vertex));
    mesh_buffers.Edges.Indices.Update(MeshRender::CreateEdgeIndices(pm));

    for (auto &[element, buffers] : mesh_buffers.NormalIndicators) {
        buffers.Vertices.Update(MeshRender::CreateNormalVertices(pm, element));
        buffers.Indices.Update(MeshRender::CreateNormalIndices(pm, element));
    }
    if (auto buffers = R.try_get<BoundingBoxesBuffers>(e)) {
        buffers->Buffers.Vertices.Update(CreateBoxVertices(bbox, EdgeColor));
        // Box indices are always the same.
    }
}

void Scene::Destroy(entt::entity e) {
    { // Clear relationships
        ClearParent(R, e);
        std::vector<entt::entity> children;
        for (auto child : Children{&R, e}) children.emplace_back(child);
        for (const auto child : children) ClearParent(R, child);
    }

    // Track mesh entity if this is an instance
    entt::entity mesh_entity = entt::null;
    if (R.all_of<MeshInstance>(e)) {
        mesh_entity = R.get<MeshInstance>(e).MeshEntity;
        SetVisible(e, false);
    }

    if (R.valid(e)) R.destroy(e);

    // If this was the last instance, destroy the mesh
    if (R.valid(mesh_entity)) {
        bool has_instances = false;
        for (const auto [_, mi] : R.view<MeshInstance>().each()) {
            if (mi.MeshEntity == mesh_entity) {
                has_instances = true;
                break;
            }
        }
        if (!has_instances) R.destroy(mesh_entity);
    }

    InvalidateCommandBuffer();
}

void Scene::SetInteractionMode(::InteractionMode mode) {
    if (InteractionMode == mode) return;

    InteractionMode = mode;
    for (const auto &[entity, mesh] : R.view<Mesh>().each()) {
        const bool highlight_faces = InteractionMode == InteractionMode::Excite && R.try_get<Excitable>(entity);
        mesh.SetColor(highlight_faces ? MeshRender::HighlightedFaceColor : Mesh::DefaultMeshColor);
        UpdateRenderBuffers(entity);
    }
    const auto e = FindActiveEntity(R);
    if (auto excitable = R.try_get<Excitable>(e)) {
        UpdateHighlightedVertices(e, *excitable);
    }
}
void Scene::SetEditMode(Element mode) {
    if (EditMode == mode) return;
    EditMode = mode;
    for (const auto mesh_entity : R.view<MeshSelection, Mesh>()) {
        R.patch<MeshSelection>(mesh_entity, [&](auto &s) {
            s = {EditMode, ConvertSelectionElement(s, R.get<Mesh>(mesh_entity), EditMode), std::nullopt};
        });
        UpdateRenderBuffers(mesh_entity);
    }
}

void Scene::SelectElement(entt::entity mesh_entity, AnyHandle element, bool toggle) {
    auto &selection = R.get<MeshSelection>(mesh_entity);
    const auto new_element = element ? element.Element : Element::None;
    if (!toggle || selection.Element != new_element) {
        selection.Element = new_element;
        selection.Handles.clear();
        selection.ActiveHandle = {};
    }

    if (element.Element != Element::None && element) {
        const auto handle = *element;
        if (auto it = find(selection.Handles, handle); toggle && it != selection.Handles.end()) {
            selection.Handles.erase(it);
            if (selection.ActiveHandle == handle) selection.ActiveHandle = {};
        } else {
            selection.Handles.emplace_back(handle);
            selection.ActiveHandle = handle;
        }
    }
    UpdateRenderBuffers(mesh_entity);
}

void Scene::WaitFor(vk::Fence fence) const {
    if (auto wait_result = Vk.Device.waitForFences(fence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    Vk.Device.resetFences(fence);
}

void Scene::UpdateRenderBuffers(entt::entity e) {
    if (const auto *mesh = R.try_get<Mesh>(e)) {
        std::unordered_set<VH> SelectedVertices, ActiveVertex;
        std::unordered_set<EH> SelectedEdges, ActiveEdge;
        std::unordered_set<FH> SelectedFaces, ActiveFace;
        const auto &selection = R.get<MeshSelection>(e);
        if (selection.Element == Element::Vertex) {
            for (auto h : selection.Handles) {
                SelectedVertices.emplace(h);
                if (selection.ActiveHandle && *selection.ActiveHandle == h) ActiveVertex.emplace(h);
            }
        } else if (selection.Element == Element::Edge) {
            for (auto h : selection.Handles) {
                SelectedEdges.emplace(h);
                const auto heh = mesh->GetHalfedge(EH{h}, 0);
                SelectedVertices.emplace(mesh->GetFromVertex(heh));
                SelectedVertices.emplace(mesh->GetToVertex(heh));
                if (selection.ActiveHandle && *selection.ActiveHandle == h) ActiveEdge.emplace(h);
            }
        } else if (selection.Element == Element::Face) {
            for (auto h : selection.Handles) {
                SelectedFaces.emplace(h);
                for (const auto heh : mesh->fh_range(FH{h})) SelectedEdges.emplace(mesh->GetEdge(heh));
                if (selection.ActiveHandle && *selection.ActiveHandle == h) ActiveFace.emplace(h);
            }
        }

        const auto &highlighted = R.get<MeshHighlightedVertices>(e).Vertices;
        const bool edit_mode = InteractionMode == InteractionMode::Edit;
        const auto element = edit_mode && selection.Element == EditMode ? selection.Element : Element::None;
        R.patch<MeshBuffers>(e, [&](auto &mb) {
            mb.Faces.Vertices.Update(
                MeshRender::CreateFaceVertices(
                    *mesh, SmoothShading,
                    highlighted,
                    element == Element::Face ? SelectedFaces : std::unordered_set<FH>{},
                    element == Element::Face ? ActiveFace : std::unordered_set<FH>{}
                )
            );
            mb.Edges.Vertices.Update(
                MeshRender::CreateEdgeVertices(
                    *mesh, EditMode,
                    element == Element::Vertex ? SelectedVertices : std::unordered_set<VH>{},
                    element == Element::Edge || element == Element::Face ? SelectedEdges : std::unordered_set<EH>{},
                    element == Element::Vertex ? ActiveVertex : std::unordered_set<VH>{},
                    element == Element::Edge ? ActiveEdge : std::unordered_set<EH>{}
                )
            );
            mb.Vertices.Vertices.Update(
                MeshRender::CreateVertexPoints(
                    *mesh, element,
                    element == Element::Vertex ? SelectedVertices : std::unordered_set<VH>{},
                    element == Element::Vertex ? ActiveVertex : std::unordered_set<VH>{}
                )
            );
        });
        UpdateRenderBufferBindless(R.get<MeshBuffers>(e).Faces);
        UpdateRenderBufferBindless(R.get<MeshBuffers>(e).Edges);
        UpdateRenderBufferBindless(R.get<MeshBuffers>(e).Vertices);
        InvalidateCommandBuffer();
    };
}

std::string Scene::DebugBufferHeapUsage() const { return UniqueBuffers->Ctx.Allocator.DebugHeapUsage(); }

namespace {
void SetTransform(entt::registry &r, entt::entity e, Transform &&t) {
    UpdateTransform(r, e, std::move(t));
}
} // namespace

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
    const bool render_silhouette = GetModelBufferIndex(R, active_entity) && InteractionMode == InteractionMode::Object;

    const auto make_pc = [this](RenderBuffers &render_buffers, ModelsBuffer &models, uint32_t first_instance = 0) {
        if (render_buffers.VertexSlot == InvalidSlot || render_buffers.IndexSlot == InvalidSlot) UpdateRenderBufferBindless(render_buffers);
        if (models.Slot == InvalidSlot) UpdateModelBufferBindless(models);
        const uint32_t vertex_count = static_cast<uint32_t>(render_buffers.Vertices.UsedSize / sizeof(Vertex3D));
        return DrawPushConstants{
            render_buffers.VertexSlot,
            render_buffers.IndexSlot,
            models.Slot,
            first_instance,
            0, // ObjectId
            vertex_count,
            0, // SelectionNodesSlot
            0 // SelectionCounterSlot
        };
    };

    // Helper to render meshes with object IDs
    auto render_meshes_with_ids = [&](const auto &entity_view, const ShaderPipeline &shader_pipeline, auto &&get_object_id, auto &&push_extra) {
        Bind(cb, shader_pipeline);
        for (const auto e : entity_view) {
            const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
            auto &render_buffers = R.get<MeshBuffers>(mesh_entity).Faces;
            auto &models = R.get<ModelsBuffer>(mesh_entity);

            auto pc = make_pc(render_buffers, models, *GetModelBufferIndex(R, e));
            pc.ObjectId = get_object_id(e);
            push_extra(pc);
            cb.pushConstants(*shader_pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
            cb.draw(render_buffers.Indices.UsedSize / sizeof(uint32_t), 1, 0, 0);
        }
    };

    // Pass 1: Silhouette depth/object rendering (selected meshes only, for visual outline)
    if (render_silhouette) {
        const auto &silhouette = Pipelines->Silhouette;
        static const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
        const vk::Rect2D rect{{0, 0}, ToExtent2D(silhouette.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette.Renderer.RenderPass, *silhouette.Resources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);

        static constexpr uint32_t ActiveId{1};
        uint32_t selected_id = 2;
        render_meshes_with_ids(
            R.view<Selected>(),
            silhouette.Renderer.ShaderPipelines.at(SPT::SilhouetteDepthObject),
            [&](entt::entity e) { return R.all_of<Active>(e) ? ActiveId : selected_id++; },
            [&](DrawPushConstants &) {}
        );
        cb.endRenderPass();

        const auto &silhouette_edge = Pipelines->SilhouetteEdge;
        static const std::vector<vk::ClearValue> edge_clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
        const vk::Rect2D edge_rect{{0, 0}, ToExtent2D(silhouette_edge.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette_edge.Renderer.RenderPass, *silhouette_edge.Resources->Framebuffer, edge_rect, edge_clear_values}, vk::SubpassContents::eInline);
        const auto &silhouette_edo = silhouette_edge.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepthObject);
        struct SilhouetteEdgeDepthObjectPushConstants {
            uint32_t SilhouetteEdgeWidth;
            uint32_t SilhouetteSamplerIndex;
        } edge_pc{SilhouetteEdgeWidth, SelectionHandles->SilhouetteSampler};
        cb.pushConstants(*silhouette_edo.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(edge_pc), &edge_pc);
        silhouette_edo.RenderQuad(cb);
        cb.endRenderPass();
    }

    // Pass 2: Selection fragment list (all visible meshes, for click selection)
    if (InteractionMode == InteractionMode::Object) {
        const auto &selection = Pipelines->SelectionFragment;
        static const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}};
        const vk::Rect2D rect{{0, 0}, ToExtent2D(selection.Resources->DepthImage.Extent)};
        cb.beginRenderPass({*selection.Renderer.RenderPass, *selection.Resources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);

        uint32_t object_id = 1;
        render_meshes_with_ids(
            R.view<Visible>(),
            selection.Renderer.ShaderPipelines.at(SPT::SelectionFragment),
            [&](entt::entity) { return object_id++; },
            [&](DrawPushConstants &pc) {
                pc.VertexCountOrHeadImageSlot = SelectionHandles->HeadImage; // Reused as HeadImageSlot
                pc.SelectionNodesSlot = SelectionHandles->SelectionNodes;
                pc.SelectionCounterSlot = SelectionHandles->SelectionCounter;
            }
        );
        cb.endRenderPass();
    }

    // Main rendering pass
    {
        const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {BackgroundColor}};
        const vk::Rect2D rect{{0, 0}, ToExtent2D(main.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*main.Renderer.RenderPass, *main.Resources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);
    }

    // Silhouette edge depth (not color! we render it before mesh depth to avoid overwriting closer depths with further ones)
    if (render_silhouette) {
        const auto &silhouette_depth = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepth);
        const uint32_t depth_sampler_index = SelectionHandles->DepthSampler;
        cb.pushConstants(*silhouette_depth.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(depth_sampler_index), &depth_sampler_index);
        silhouette_depth.RenderQuad(cb);
    }

    { // Meshes
        const SPT fill_pipeline = ColorMode == ColorMode::Mesh ? SPT::Fill : SPT::DebugNormals;
        std::unordered_set<entt::entity> selected_mesh_entities;
        for (const auto [_, mi] : R.view<const MeshInstance, const Selected>().each()) selected_mesh_entities.emplace(mi.MeshEntity);

        for (auto [entity, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
            const bool show_wireframe_overlay = InteractionMode == InteractionMode::Edit && selected_mesh_entities.contains(entity);
            if (ViewportShading == ViewportShadingMode::Solid) {
                main.Renderer.Render(cb, fill_pipeline, mesh_buffers.Faces, models, make_pc(mesh_buffers.Faces, models));
            }
            if (ViewportShading == ViewportShadingMode::Wireframe || show_wireframe_overlay) {
                main.Renderer.Render(cb, SPT::Line, mesh_buffers.Edges, models, make_pc(mesh_buffers.Edges, models));
            }
            if (show_wireframe_overlay && EditMode == Element::Vertex) {
                main.Renderer.Render(cb, SPT::Point, mesh_buffers.Vertices, models, make_pc(mesh_buffers.Vertices, models));
            }
        }
    }

    // Silhouette edge color (rendered ontop of meshes)
    if (render_silhouette) {
        const auto &silhouette_edc = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeColor);
        struct SilhouetteEdgeColorPushConstants {
            uint32_t Manipulating;
            uint32_t ObjectSamplerIndex;
        } pc{TransformGizmo::IsUsing(), SelectionHandles->ObjectIdSampler};
        cb.pushConstants(*silhouette_edc.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        silhouette_edc.RenderQuad(cb);
    }

    // Selection overlays
    for (auto [_, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
        for (auto &[element, buffers] : mesh_buffers.NormalIndicators) {
            main.Renderer.Render(cb, SPT::Line, buffers, models, make_pc(buffers, models));
        }
    }
    for (auto [_, bvh_boxes, models] : R.view<BvhBoxesBuffers, ModelsBuffer>().each()) {
        main.Renderer.Render(cb, SPT::Line, bvh_boxes.Buffers, models, make_pc(bvh_boxes.Buffers, models));
    }
    for (auto [_, bounding_boxes, models] : R.view<BoundingBoxesBuffers, ModelsBuffer>().each()) {
        main.Renderer.Render(cb, SPT::Line, bounding_boxes.Buffers, models, make_pc(bounding_boxes.Buffers, models));
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
    MeshRender::EdgeColor = ViewportShading == ViewportShadingMode::Solid ? MeshEdgeColor : EdgeColor;
    for (const auto e : R.view<Mesh>()) UpdateRenderBuffers(e);
}

void Scene::UpdateTransformBuffers() {
    const float aspect_ratio = Extent.width == 0 || Extent.height == 0 ? 1.f : float(Extent.width) / float(Extent.height);
    SceneUBO scene_ubo{};
    scene_ubo.View = Camera.View();
    scene_ubo.Proj = Camera.Projection(aspect_ratio);
    scene_ubo.CameraPositionNear = vec4{Camera.Position(), Camera.NearClip};
    scene_ubo.ViewColorAndAmbient = Lights.ViewColorAndAmbient;
    scene_ubo.DirectionalColorAndIntensity = Lights.DirectionalColorAndIntensity;
    scene_ubo.LightDirectionFar = vec4{Lights.Direction, Camera.FarClip};
    scene_ubo.SilhouetteActive = Colors.Active;
    scene_ubo.SilhouetteSelected = Colors.Selected;
    UniqueBuffers->SceneUBO.Update(as_bytes(scene_ubo));
    InvalidateCommandBuffer();
}

void Scene::UpdateHighlightedVertices(entt::entity e, const Excitable &excitable) {
    if (auto *highlighted = R.try_get<MeshHighlightedVertices>(e)) {
        highlighted->Vertices.clear();
        if (InteractionMode == InteractionMode::Excite) {
            for (const auto vertex : excitable.ExcitableVertices) {
                highlighted->Vertices.emplace(VH(vertex));
            }
        }
        UpdateRenderBuffers(e);
    }
}

// todo selection overlays for _only selected instances_ (currently all instances of selected meshes)
void Scene::UpdateEntitySelectionOverlays(entt::entity instance_entity) {
    const auto mesh_entity = R.get<MeshInstance>(instance_entity).MeshEntity;
    const auto &mesh = R.get<const Mesh>(mesh_entity);
    auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
    for (const auto element : NormalElements) {
        if (ShownNormalElements.contains(element) && !mesh_buffers.NormalIndicators.contains(element)) {
            auto buffers = UniqueBuffers->CreateRenderBuffers(MeshRender::CreateNormalVertices(mesh, element), MeshRender::CreateNormalIndices(mesh, element));
            UpdateRenderBufferBindless(buffers);
            mesh_buffers.NormalIndicators.emplace(element, std::move(buffers));
        } else if (!ShownNormalElements.contains(element) && mesh_buffers.NormalIndicators.contains(element)) {
            ReleaseRenderBufferBindless(mesh_buffers.NormalIndicators.at(element));
            mesh_buffers.NormalIndicators.erase(element);
        }
    }
    if (ShowBoundingBoxes && !R.all_of<BoundingBoxesBuffers>(mesh_entity)) {
        const auto &bbox = R.get<BBox>(mesh_entity);
        auto buffers = UniqueBuffers->CreateRenderBuffers(CreateBoxVertices(bbox, EdgeColor), BBox::EdgeIndices);
        UpdateRenderBufferBindless(buffers);
        R.emplace<BoundingBoxesBuffers>(mesh_entity, std::move(buffers));
    } else if (!ShowBoundingBoxes && R.all_of<BoundingBoxesBuffers>(mesh_entity)) {
        R.remove<BoundingBoxesBuffers>(mesh_entity);
    }
    if (ShowBvhBoxes && !R.all_of<BvhBoxesBuffers>(mesh_entity)) {
        const auto &bvh = R.get<BVH>(mesh_entity);
        auto bvh_buffers = MeshRender::CreateBvhBuffers(bvh, EdgeColor);
        auto buffers = UniqueBuffers->CreateRenderBuffers(std::move(bvh_buffers.Vertices), std::move(bvh_buffers.Indices));
        UpdateRenderBufferBindless(buffers);
        R.emplace<BvhBoxesBuffers>(mesh_entity, std::move(buffers));
    } else if (!ShowBvhBoxes && R.all_of<BvhBoxesBuffers>(mesh_entity)) {
        R.remove<BvhBoxesBuffers>(mesh_entity);
    }
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
ray WorldToLocal(const ray &r, const mat4 &world_inv_transp) {
    const auto world_inv = glm::transpose(world_inv_transp);
    return {{world_inv * vec4{r.o, 1}}, glm::normalize(vec3{world_inv * vec4{r.d, 0}})};
}

// Nearest intersection across all meshes.
struct EntityIntersection {
    entt::entity Entity;
    Intersection Intersection;
    vec3 Position;
};

template<typename... FilterTags>
std::vector<EntityIntersection> IntersectVisible(const entt::registry &r, const ray &world_ray) {
    std::vector<EntityIntersection> results;
    const auto compute_intersection = [&](entt::entity e, const WorldMatrix &world_matrix) {
        const auto mesh_entity = r.get<MeshInstance>(e).MeshEntity;
        const auto &bvh = r.get<const BVH>(mesh_entity);
        const auto &mesh = r.get<const Mesh>(mesh_entity);
        const auto local_ray = WorldToLocal(world_ray, world_matrix.MInv);
        if (auto intersection = MeshIntersection::Intersect(bvh, mesh, local_ray)) {
            results.emplace_back(e, *intersection, local_ray(intersection->Distance));
        }
    };

    for (const auto &[e, world_matrix] : r.view<const WorldMatrix, const Visible, FilterTags...>().each()) {
        compute_intersection(e, world_matrix);
    }
    std::ranges::sort(results, {}, [](const auto &result) { return result.Intersection.Distance; });
    return results;
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
    if (IsKeyPressed(ImGuiKey_Escape, false)) EscapePressed = true;
    if (IsMouseReleased(button)) {
        const bool was_escape_pressed = EscapePressed;
        EscapePressed = false;
        if (was_escape_pressed) return false;
    }
    return IsMouseReleased(button) && !IsMouseDragPastThreshold(button);
}

void BarrierSelectionData(vk::CommandBuffer cb, const SceneUniqueBuffers &buffers, const mvk::ImageResource &head_image) {
    const vk::BufferMemoryBarrier node_barrier{
        vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead,
        {},
        {},
        *buffers.SelectionNodeBuffer.DeviceBuffer,
        0,
        VK_WHOLE_SIZE
    };
    const vk::ImageMemoryBarrier head_barrier{
        vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eGeneral,
        vk::ImageLayout::eGeneral,
        {},
        {},
        *head_image.Image,
        ColorSubresourceRange
    };
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eFragmentShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, {node_barrier}, {head_barrier}
    );
}

} // namespace

std::vector<entt::entity> Scene::RunClickSelect(glm::uvec2 mouse_px) {
    auto cb = *ClickCommandBuffer;
    cb.reset({});

    const auto *counters = reinterpret_cast<const SelectionCounters *>(UniqueBuffers->SelectionCounterBuffer.GetData().data());
    const auto node_count = std::min<uint32_t>(counters->Count, SceneUniqueBuffers::MaxSelectionNodes);
    if (node_count == 0) return {};

    // Dispatch click-reduction compute writing directly to host-visible buffer (no copy needed).
    UniqueBuffers->ClickResultBuffer.Write(as_bytes(ClickResult{})); // Reset count/hits.
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    BarrierSelectionData(cb, *UniqueBuffers, Pipelines->SelectionFragment.Resources->HeadImage);

    const auto &compute = Pipelines->ClickSelect;
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, *compute.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *compute.PipelineLayout, 0, compute.GetDescriptorSet(), {});
    const ClickSelectPushConstants pc{
        .TargetPx = mouse_px,
        .HeadImageIndex = SelectionHandles->HeadImage,
        .SelectionNodesIndex = SelectionHandles->SelectionNodes,
        .ClickResultIndex = SelectionHandles->ClickResult
    };
    cb.pushConstants(*compute.PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
    cb.dispatch(1, 1, 1);

    const vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead};
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, barrier, {}, {});
    cb.end();
    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    Vk.Queue.submit(submit, *TransferFence);
    WaitFor(*TransferFence);

    // Step 3: convert click hits to entities.
    const auto &result = UniqueBuffers->GetClickResult();
    const auto visible_entities = R.view<Visible>() | to<std::vector>();
    const uint32_t hit_count = std::min<uint32_t>(result.Count, result.Hits.size());
    std::vector<std::pair<float, uint32_t>> hits;
    hits.reserve(hit_count);
    for (uint32_t i = 0; i < hit_count; ++i) {
        const auto &hit = result.Hits[i];
        if (hit.ObjectId == 0 || hit.ObjectId > visible_entities.size()) continue;
        hits.emplace_back(hit.Depth, hit.ObjectId);
    }
    std::ranges::sort(hits);

    std::vector<entt::entity> entities;
    std::unordered_set<uint32_t> seen_object_ids;
    for (const auto &[_, object_id] : hits) {
        if (seen_object_ids.insert(object_id).second) {
            entities.push_back(visible_entities[object_id - 1]);
        }
    }
    return entities;
}

std::vector<entt::entity> Scene::RunBoxSelect(glm::uvec2 box_min, glm::uvec2 box_max) {
    if (box_min.x >= box_max.x || box_min.y >= box_max.y) return {};
    const auto *counters = reinterpret_cast<const SelectionCounters *>(UniqueBuffers->SelectionCounterBuffer.GetData().data());
    const auto node_count = std::min<uint32_t>(counters->Count, SceneUniqueBuffers::MaxSelectionNodes);
    if (node_count == 0) return {};

    const auto visible_entities = R.view<Visible>() | to<std::vector>();
    const auto object_count = static_cast<uint32_t>(visible_entities.size());
    if (object_count == 0) return {};

    const uint32_t bitset_words = (object_count + 31) / 32;
    if (bitset_words > BoxSelectZeroBits.size()) return {};
    const std::span<const uint32_t> zero_bits{BoxSelectZeroBits.data(), bitset_words};
    UniqueBuffers->BoxSelectBitsetBuffer.Write(std::as_bytes(zero_bits));

    auto cb = *ClickCommandBuffer;
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    BarrierSelectionData(cb, *UniqueBuffers, Pipelines->SelectionFragment.Resources->HeadImage);

    const auto &compute = Pipelines->BoxSelect;
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, *compute.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *compute.PipelineLayout, 0, compute.GetDescriptorSet(), {});
    const BoxSelectPushConstants pc{
        .BoxMin = box_min,
        .BoxMax = box_max,
        .ObjectCount = object_count,
        .HeadImageIndex = SelectionHandles->HeadImage,
        .SelectionNodesIndex = SelectionHandles->SelectionNodes,
        .BoxResultIndex = SelectionHandles->BoxResult
    };
    cb.pushConstants(*compute.PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
    const uint32_t width = box_max.x - box_min.x;
    const uint32_t height = box_max.y - box_min.y;
    const uint32_t group_count_x = (width + 15) / 16;
    const uint32_t group_count_y = (height + 15) / 16;
    cb.dispatch(group_count_x, group_count_y, 1);

    const vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead};
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, barrier, {}, {});
    cb.end();
    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    Vk.Queue.submit(submit, *TransferFence);
    WaitFor(*TransferFence);

    const auto bitset_data = UniqueBuffers->BoxSelectBitsetBuffer.GetData();
    const auto *bits = reinterpret_cast<const uint32_t *>(bitset_data.data());
    std::vector<entt::entity> entities;
    entities.reserve(object_count);
    for (uint32_t i = 0; i < object_count; ++i) {
        const uint32_t mask = 1u << (i % 32);
        if (bits[i / 32] & mask) entities.push_back(visible_entities[i]);
    }
    return entities;
}

void Scene::UpdateSelectionBindlessDescriptors() {
    if (!Pipelines || !Pipelines->SelectionFragment.Resources) return;

    auto &handles = *SelectionHandles;
    auto &alloc = *BindlessAlloc;
    std::vector<vk::WriteDescriptorSet> writes;

    const auto head_image_info = vk::DescriptorImageInfo{
        nullptr,
        *Pipelines->SelectionFragment.Resources->HeadImage.View,
        vk::ImageLayout::eGeneral
    };
    writes.push_back(alloc.MakeImageWrite(handles.HeadImage, head_image_info));

    const auto selection_nodes = UniqueBuffers->SelectionNodeBuffer.GetDescriptor();
    const auto selection_counter = vk::DescriptorBufferInfo{*UniqueBuffers->SelectionCounterBuffer, 0, sizeof(SelectionCounters)};
    const auto click_result = vk::DescriptorBufferInfo{*UniqueBuffers->ClickResultBuffer, 0, sizeof(ClickResult)};
    const auto box_result = UniqueBuffers->GetBoxSelectBitsetDescriptor();
    const auto scene_ubo = UniqueBuffers->SceneUBO.GetDescriptor();

    writes.push_back(alloc.MakeUniformWrite(SceneUBOSlot, scene_ubo));
    writes.push_back(alloc.MakeBufferWrite(SlotType::Buffer, handles.SelectionNodes, selection_nodes));
    writes.push_back(alloc.MakeBufferWrite(SlotType::Buffer, handles.SelectionCounter, selection_counter));
    writes.push_back(alloc.MakeBufferWrite(SlotType::Buffer, handles.ClickResult, click_result));
    writes.push_back(alloc.MakeBufferWrite(SlotType::Buffer, handles.BoxResult, box_result));
    // Samplers
    const auto &sil = Pipelines->Silhouette;
    const auto &sil_edge = Pipelines->SilhouetteEdge;
    const auto object_id_sampler = vk::DescriptorImageInfo{*sil_edge.Resources->ImageSampler, *sil_edge.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
    const auto depth_sampler = vk::DescriptorImageInfo{*sil_edge.Resources->DepthSampler, *sil_edge.Resources->DepthImage.View, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
    const auto silhouette_sampler = vk::DescriptorImageInfo{*sil.Resources->ImageSampler, *sil.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
    writes.push_back(alloc.MakeSamplerWrite(handles.ObjectIdSampler, object_id_sampler));
    writes.push_back(alloc.MakeSamplerWrite(handles.DepthSampler, depth_sampler));
    writes.push_back(alloc.MakeSamplerWrite(handles.SilhouetteSampler, silhouette_sampler));

    Vk.Device.updateDescriptorSets(writes, {});
}

void Scene::UpdateRenderBufferBindless(RenderBuffers &rb) {
    if (!BindlessAlloc) return;

    auto &alloc = *BindlessAlloc;
    if (rb.VertexSlot == InvalidSlot) rb.VertexSlot = alloc.Allocate(SlotType::VertexBuffer);
    if (rb.IndexSlot == InvalidSlot) rb.IndexSlot = alloc.Allocate(SlotType::IndexBuffer);

    const auto vertex_info = rb.Vertices.GetDescriptor();
    const auto index_info = rb.Indices.GetDescriptor();
    const std::array writes{
        alloc.MakeBufferWrite(SlotType::VertexBuffer, rb.VertexSlot, vertex_info),
        alloc.MakeBufferWrite(SlotType::IndexBuffer, rb.IndexSlot, index_info),
    };
    Vk.Device.updateDescriptorSets(writes, {});
}

void Scene::UpdateModelBufferBindless(ModelsBuffer &mb) {
    if (!BindlessAlloc) return;

    auto &alloc = *BindlessAlloc;
    if (mb.Slot == InvalidSlot) mb.Slot = alloc.Allocate(SlotType::ModelBuffer);
    const auto info = mb.Buffer.GetDescriptor();
    Vk.Device.updateDescriptorSets(alloc.MakeBufferWrite(SlotType::ModelBuffer, mb.Slot, info), {});
}

void Scene::ReleaseRenderBufferBindless(RenderBuffers &rb) {
    if (!BindlessAlloc) return;
    if (rb.VertexSlot != InvalidSlot) {
        BindlessAlloc->Release(SlotType::VertexBuffer, rb.VertexSlot);
        rb.VertexSlot = InvalidSlot;
    }
    if (rb.IndexSlot != InvalidSlot) {
        BindlessAlloc->Release(SlotType::IndexBuffer, rb.IndexSlot);
        rb.IndexSlot = InvalidSlot;
    }
}

void Scene::ReleaseModelBufferBindless(ModelsBuffer &mb) {
    if (!BindlessAlloc) return;
    if (mb.Slot != InvalidSlot) {
        BindlessAlloc->Release(SlotType::ModelBuffer, mb.Slot);
        mb.Slot = InvalidSlot;
    }
}

void Scene::Interact() {
    if (Extent.width == 0 || Extent.height == 0) return;

    const auto active_entity = FindActiveEntity(R);
    // Handle keyboard input.
    if (IsWindowFocused()) {
        if (IsKeyPressed(ImGuiKey_Tab)) {
            // Cycle to the next interaction mode, wrapping around to the first.
            auto it = find(InteractionModes, InteractionMode);
            SetInteractionMode(++it != InteractionModes.end() ? *it : *InteractionModes.begin());
        }
        if (InteractionMode == InteractionMode::Edit) {
            if (IsKeyPressed(ImGuiKey_1, false)) SetEditMode(Element::Vertex);
            else if (IsKeyPressed(ImGuiKey_2, false)) SetEditMode(Element::Edge);
            else if (IsKeyPressed(ImGuiKey_3, false)) SetEditMode(Element::Face);
        }
        if (!R.storage<Selected>().empty()) {
            if (IsKeyPressed(ImGuiKey_D, false) && GetIO().KeyShift) Duplicate();
            else if (IsKeyPressed(ImGuiKey_D, false) && GetIO().KeyAlt) DuplicateLinked();
            else if (IsKeyPressed(ImGuiKey_Delete, false) || IsKeyPressed(ImGuiKey_Backspace, false)) Delete();
            else if (IsKeyPressed(ImGuiKey_G, false)) StartScreenTransform = TransformGizmo::TransformType::Translate;
            else if (IsKeyPressed(ImGuiKey_R, false)) StartScreenTransform = TransformGizmo::TransformType::Rotate;
            else if (IsKeyPressed(ImGuiKey_S, false)) StartScreenTransform = TransformGizmo::TransformType::Scale;
            else if (IsKeyPressed(ImGuiKey_H, false)) {
                for (const auto e : R.view<Selected>()) SetVisible(e, !R.all_of<Visible>(e));
            } else if (IsKeyPressed(ImGuiKey_P, false) && GetIO().KeyCtrl) {
                if (active_entity != entt::null) {
                    for (const auto e : R.view<Selected>()) {
                        if (e != active_entity) SetParent(R, e, active_entity);
                    }
                }
            } else if (IsKeyPressed(ImGuiKey_P, false) && GetIO().KeyAlt) {
                for (const auto e : R.view<Selected>()) ClearParent(R, e);
            }
        }
    }

    // Handle mouse input.
    if (!IsMouseDown(ImGuiMouseButton_Left)) {
        R.remove<ExcitedVertex>(active_entity);
    }
    if (TransformGizmo::IsUsing()) {
        // TransformGizmo overrides this mouse cursor during some actions - this is a default.
        SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        WrapMousePos(GetCurrentWindowRead()->InnerClipRect, AccumulatedWrapMouseDelta);
    } else {
        AccumulatedWrapMouseDelta = {0, 0};
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

    if (TransformGizmo::IsUsing() || OrientationGizmo::IsActive() || TransformModePillsHovered) return;

    // todo box selection for edit mode (select vertices/edges/faces within box)
    if (SelectionMode == SelectionMode::Box && InteractionMode == InteractionMode::Object) {
        const auto mouse_pos = ToGlm(GetMousePos());
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            BoxSelectStart = mouse_pos;
            BoxSelectEnd = mouse_pos;
        } else if (IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart.has_value()) {
            BoxSelectEnd = mouse_pos;
            static constexpr float drag_threshold{2};
            if (glm::distance(*BoxSelectStart, *BoxSelectEnd) > drag_threshold) {
                const auto window_pos = ToGlm(GetCursorScreenPos());
                const vec2 extent_size{float(Extent.width), float(Extent.height)};
                const auto box_min = glm::min(*BoxSelectStart, *BoxSelectEnd) - window_pos;
                const auto box_max = glm::max(*BoxSelectStart, *BoxSelectEnd) - window_pos;
                const auto local_min = glm::clamp(glm::min(box_min, box_max), vec2{0}, extent_size);
                const auto local_max = glm::clamp(glm::max(box_min, box_max), vec2{0}, extent_size);
                const glm::uvec2 box_min_px{
                    static_cast<uint32_t>(glm::floor(local_min.x)),
                    static_cast<uint32_t>(glm::floor(extent_size.y - local_max.y))
                };
                const glm::uvec2 box_max_px{
                    static_cast<uint32_t>(glm::ceil(local_max.x)),
                    static_cast<uint32_t>(glm::ceil(extent_size.y - local_min.y))
                };
                const auto selected_entities = RunBoxSelect(box_min_px, box_max_px);
                R.clear<Selected>();
                for (const auto e : selected_entities) R.emplace<Selected>(e);
                InvalidateCommandBuffer();
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart.has_value()) {
            BoxSelectStart.reset();
            BoxSelectEnd.reset();
        }
        if (BoxSelectStart.has_value()) return;
    }

    if (!IsSingleClicked(ImGuiMouseButton_Left)) return;

    // Handle mouse selection.
    const auto size = GetContentRegionAvail();
    const auto mouse_pos = (GetMousePos() - GetCursorScreenPos()) / size;
    const auto mouse_ray_ws = Camera.NdcToWorldRay({2 * mouse_pos.x - 1, 1 - 2 * mouse_pos.y}, size.x / size.y);
    if (InteractionMode == InteractionMode::Edit) {
        if (EditMode != Element::None) {
            const bool shift_down = IsKeyDown(ImGuiMod_Shift);
            const bool ctrl_down = IsKeyDown(ImGuiMod_Ctrl) || IsKeyDown(ImGuiMod_Super);
            const bool toggle = shift_down || ctrl_down;
            if (const auto intersections = IntersectVisible<const Selected>(R, mouse_ray_ws); !intersections.empty()) {
                const auto &hit = intersections.front();
                const auto &world_matrix = R.get<WorldMatrix>(hit.Entity);
                const auto mesh_entity = R.get<MeshInstance>(hit.Entity).MeshEntity;
                if (!toggle) {
                    for (const auto [e, selection] : R.view<MeshSelection>().each()) {
                        if (selection.Handles.empty()) continue;
                        R.patch<MeshSelection>(e, [](auto &s) { s.Handles.clear(); s.Element = Element::None; });
                        UpdateRenderBuffers(e);
                    }
                }
                const auto &bvh = R.get<BVH>(mesh_entity);
                const auto &mesh = R.get<Mesh>(mesh_entity);
                const auto mouse_ray = WorldToLocal(mouse_ray_ws, world_matrix.MInv);

                using namespace MeshIntersection;
                if (EditMode == Element::Vertex) SelectElement(mesh_entity, FindNearestVertex(bvh, mesh, mouse_ray), toggle);
                else if (EditMode == Element::Edge) SelectElement(mesh_entity, FindNearestEdge(bvh, mesh, mouse_ray), toggle);
                else if (EditMode == Element::Face) SelectElement(mesh_entity, FindNearestIntersectingFace(bvh, mesh, mouse_ray), toggle);
            }
        }
    } else if (InteractionMode == InteractionMode::Object) {
        const auto mouse_pos_rel = GetMousePos() - GetCursorScreenPos();
        // Flip y-coordinate: ImGui uses top-left origin, but Vulkan gl_FragCoord uses bottom-left origin
        const auto objects = RunClickSelect({uint32_t(mouse_pos_rel.x), uint32_t(Extent.height - mouse_pos_rel.y)});
        entt::entity intersected = entt::null;
        if (!objects.empty()) {
            auto it = find(objects, active_entity);
            if (it != objects.end()) ++it;
            if (it == objects.end()) it = objects.begin();
            intersected = *it;
        }

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
    } else if (InteractionMode == InteractionMode::Excite) {
        // Excite the nearest entity if it's excitable.
        if (const auto nearest = IntersectVisible<>(R, mouse_ray_ws); !nearest.empty()) {
            const auto &hit = nearest.front();
            if (const auto *excitable = R.try_get<Excitable>(hit.Entity)) {
                // Find the nearest excitable vertex.
                std::optional<uint> nearest_excite_vertex;
                float min_dist_sq = std::numeric_limits<float>::max();
                const auto &mesh = R.get<Mesh>(R.get<MeshInstance>(hit.Entity).MeshEntity);
                const auto p = hit.Position;
                for (uint excite_vertex : excitable->ExcitableVertices) {
                    const auto diff = p - mesh.GetPosition(VH(excite_vertex));
                    if (float dist_sq = glm::dot(diff, diff); dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        nearest_excite_vertex = excite_vertex;
                    }
                }
                if (nearest_excite_vertex) {
                    R.emplace<ExcitedVertex>(hit.Entity, *nearest_excite_vertex, 1.f);
                }
            }
        }
    }
}

void ScenePipelines::SetExtent(vk::Extent2D extent) {
    Main.SetExtent(extent, Device, PhysicalDevice, Samples);
    Silhouette.SetExtent(extent, Device, PhysicalDevice);
    SilhouetteEdge.SetExtent(extent, Device, PhysicalDevice);
    SelectionFragment.SetExtent(extent, Device, PhysicalDevice);

    // Using bindless for samplers; no per-descriptor updates needed.
};

bool Scene::RenderViewport() {
    const auto content_region = ToGlm(GetContentRegionAvail());
    const bool extent_changed = Extent.width != content_region.x || Extent.height != content_region.y;
    if (!extent_changed && !CommandBufferDirty) return false;

    struct RenderViewportTimer {
        std::chrono::steady_clock::time_point Start{std::chrono::steady_clock::now()};
        ~RenderViewportTimer() {
            const auto end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(end - Start).count();
            std::println("RenderViewport: ms={:.3f}", ms);
        }
    };

    const RenderViewportTimer timer{};

    CommandBufferDirty = false;

    if (extent_changed) {
        Extent = ToExtent(content_region);
        UpdateTransformBuffers();
        Vk.Device.waitIdle(); // Ensure GPU work is done before destroying old pipeline resources
        Pipelines->SetExtent(Extent);
        UpdateSelectionBindlessDescriptors();
    }

    const auto transfer_cb = *UniqueBuffers->Ctx.TransferCb;
    // transfer_cb is kept recording between frames by BufferContext and our end-of-frame begin().
    if (InteractionMode == InteractionMode::Object) {
        UniqueBuffers->SelectionCounterBuffer.Write(as_bytes(SelectionCounters{}));

        const auto &head_image = Pipelines->SelectionFragment.Resources->HeadImage;
        const vk::ImageMemoryBarrier to_general{
            {}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {}, {}, *head_image.Image, ColorSubresourceRange
        };
        transfer_cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, to_general);

        const vk::ClearColorValue clear_value{std::array<uint32_t, 4>{InvalidSlot, 0, 0, 0}};
        transfer_cb.clearColorImage(*head_image.Image, vk::ImageLayout::eGeneral, clear_value, ColorSubresourceRange);

        const vk::ImageMemoryBarrier to_shader{
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            vk::ImageLayout::eGeneral,
            vk::ImageLayout::eGeneral,
            {},
            {},
            *head_image.Image,
            ColorSubresourceRange
        };
        transfer_cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, to_shader);
    }
    // Ensure buffer writes (staging copies) are visible to shader reads.
    const vk::MemoryBarrier buffer_barrier{
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead
    };
    transfer_cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader,
        {}, buffer_barrier, {}, {}
    );

    transfer_cb.end();

    RecordRenderCommandBuffer();

    // Submit transfer and render commands together
    const std::array command_buffers{transfer_cb, *RenderCommandBuffer};
    vk::SubmitInfo submit;
    submit.setCommandBuffers(command_buffers);
    Vk.Queue.submit(submit, *RenderFence);
    WaitFor(*RenderFence);

    // Read back selection node count for dispatch sizing (host-visible buffer).
    // No copy needed; buffer is host-visible and GPU work is done by this point.

    UniqueBuffers->Ctx.Reclaimer.Reclaim();
    // Leave transfer_cb recording for next frame's staging.
    transfer_cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    return extent_changed;
}

void Scene::RenderOverlay() {
    const auto window_pos = ToGlm(GetWindowPos());
    { // Transform mode pill buttons (top-left overlay)
        struct ButtonInfo {
            const SvgResource &Icon;
            TransformGizmo::Type ButtonType;
            ImDrawFlags Corners;
            bool Enabled;
        };

        using enum TransformGizmo::Type;
        const auto v = R.view<Selected, Frozen>();
        const bool scale_enabled = v.begin() == v.end();
        const ButtonInfo buttons[]{
            {*Icons.Select, None, ImDrawFlags_RoundCornersTop, true},
            {*Icons.SelectBox, None, ImDrawFlags_RoundCornersBottom, true},
            {*Icons.Move, Translate, ImDrawFlags_RoundCornersTop, true},
            {*Icons.Rotate, Rotate, ImDrawFlags_RoundCornersNone, true},
            {*Icons.Scale, Scale, ImDrawFlags_RoundCornersNone, scale_enabled},
            {*Icons.Universal, Universal, ImDrawFlags_RoundCornersBottom, true},
        };

        auto &element = MGizmo.Config.Type;
        if (!scale_enabled && element == Scale) element = Translate;

        const float padding = GetTextLineHeightWithSpacing() / 2.f;
        const auto start_pos = std::bit_cast<ImVec2>(window_pos) + GetWindowContentRegionMin() + ImVec2{padding, padding};
        const auto saved_cursor_pos = GetCursorScreenPos();

        auto &dl = *GetWindowDrawList();
        TransformModePillsHovered = false;
        static constexpr ImVec2 button_size{36, 30};
        static constexpr float gap{4}; // Gap between select buttons and transform buttons
        for (uint i = 0; i < 6; ++i) {
            const auto &[icon, button_type, corners, enabled] = buttons[i];
            static constexpr ImVec2 padding{0.5f, 0.5f};
            static constexpr float icon_dim{button_size.y * 0.75f};
            static constexpr ImVec2 icon_size{icon_dim, icon_dim};
            const float y_offset = i < 2 ? i * button_size.y : 2 * button_size.y + gap + (i - 2) * button_size.y;
            SetCursorScreenPos({start_pos.x, start_pos.y + y_offset});

            if (!enabled) BeginDisabled();
            PushID(i);
            const bool clicked = InvisibleButton("##icon", button_size);
            PopID();
            if (!enabled) EndDisabled();

            const bool hovered = IsItemHovered();
            if (hovered) TransformModePillsHovered = true;

            if (clicked) {
                if (i == 0) {
                    SelectionMode = SelectionMode::Click;
                    element = None;
                } else if (i == 1) {
                    SelectionMode = SelectionMode::Box;
                    element = None;
                } else { // Transform buttons
                    element = button_type;
                }
            }

            const bool is_active = i < 2 ?
                (element == None && ((i == 0 && SelectionMode == SelectionMode::Click) || (i == 1 && SelectionMode == SelectionMode::Box))) :
                element == button_type;
            const auto bg_color = GetColorU32(
                !enabled      ? ImGuiCol_FrameBg :
                    is_active ? ImGuiCol_ButtonActive :
                    hovered   ? ImGuiCol_ButtonHovered :
                                ImGuiCol_Button
            );
            dl.AddRectFilled(GetItemRectMin() + padding, GetItemRectMax() - padding, bg_color, 8.f, corners);
            SetCursorScreenPos(GetItemRectMin() + (button_size - icon_size) * 0.5f);
            icon.RenderIcon(std::bit_cast<vec2>(icon_size));
        }
        SetCursorScreenPos(saved_cursor_pos);
    }

    if (!R.storage<Active>().empty()) { // Draw center-dot for active/selected entities
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

    if (const auto selected_view = R.view<Selected>(); !selected_view.empty()) {
        // Transform all root selected entities (whose parent is not also selected) around their average position,
        // using the active entity's rotation/scale.
        // Non-root selected entities already follow their parent's transform.
        struct StartTransform {
            Transform T;
        };
        const auto start_transform_view = R.view<const StartTransform>();

        const auto is_parent_selected = [&](entt::entity e) {
            if (const auto *node = R.try_get<SceneNode>(e)) {
                return node->Parent != entt::null && R.all_of<Selected>(node->Parent);
            }
            return false;
        };
        auto root_selected = selected_view | std::views::filter([&](auto e) { return !is_parent_selected(e); });
        const auto root_count = distance(root_selected);

        const auto active_transform = GetTransform(R, FindActiveEntity(R));
        const auto p = fold_left(root_selected | transform([&](auto e) { return R.get<Position>(e).Value; }), vec3{}, std::plus{}) / float(root_count);
        if (auto start_delta = TransformGizmo::Draw(
                {{.P = p, .R = active_transform.R, .S = active_transform.S}, MGizmo.Mode},
                MGizmo.Config, Camera, window_pos, ToGlm(GetContentRegionAvail()), ToGlm(GetIO().MousePos) + AccumulatedWrapMouseDelta,
                StartScreenTransform
            )) {
            const auto &[ts, td] = *start_delta;
            if (start_transform_view.empty()) {
                for (const auto e : root_selected) R.emplace<StartTransform>(e, GetTransform(R, e));
            }
            // Compute delta transform from drag start
            const auto r = ts.R, rT = glm::conjugate(r);
            for (const auto &[e, ts_e_comp] : start_transform_view.each()) {
                const auto &ts_e = ts_e_comp.T;
                const bool frozen = R.all_of<Frozen>(e);
                const auto offset = ts_e.P - ts.P;
                SetTransform(
                    R, e,
                    {
                        .P = td.P + ts.P + glm::rotate(td.R, frozen ? offset : r * (rT * offset * td.S)),
                        .R = glm::normalize(td.R * ts_e.R),
                        .S = frozen ? ts_e.S : td.S * ts_e.S,
                    }
                );
            }
            InvalidateCommandBuffer();
        } else if (!start_transform_view.empty()) {
            R.clear<StartTransform>();
            InvalidateCommandBuffer(); // No longer manipulating - silhouette color changes.
        }
    }
    { // Orientation gizmo
        static constexpr float OGizmoSize{90};
        const float padding = GetTextLineHeightWithSpacing();
        const auto pos = window_pos + vec2{GetWindowContentRegionMax().x, GetWindowContentRegionMin().y} - vec2{OGizmoSize, 0} + vec2{-padding, padding};
        OrientationGizmo::Draw(pos, OGizmoSize, Camera);
        if (Camera.Tick()) UpdateTransformBuffers();
    }

    if (BoxSelectStart.has_value() && BoxSelectEnd.has_value()) {
        auto &dl = *GetWindowDrawList();
        const auto box_min = glm::min(*BoxSelectStart, *BoxSelectEnd);
        const auto box_max = glm::max(*BoxSelectStart, *BoxSelectEnd);
        dl.AddRectFilled(std::bit_cast<ImVec2>(box_min), std::bit_cast<ImVec2>(box_max), IM_COL32(255, 255, 255, 30));

        // Dashed outline
        static constexpr auto outline_color{IM_COL32(255, 255, 255, 200)};
        static constexpr float dash_size{4};
        static constexpr float gap_size{4};
        // Top
        for (float x = box_min.x; x < box_max.x; x += dash_size + gap_size) {
            dl.AddLine({x, box_min.y}, {glm::min(x + dash_size, box_max.x), box_min.y}, outline_color, 1.0f);
        }
        // Bottom
        for (float x = box_min.x; x < box_max.x; x += dash_size + gap_size) {
            dl.AddLine({x, box_max.y}, {glm::min(x + dash_size, box_max.x), box_max.y}, outline_color, 1.0f);
        }
        // Left
        for (float y = box_min.y; y < box_max.y; y += dash_size + gap_size) {
            dl.AddLine({box_min.x, y}, {box_min.x, glm::min(y + dash_size, box_max.y)}, outline_color, 1.0f);
        }
        // Right
        for (float y = box_min.y; y < box_max.y; y += dash_size + gap_size) {
            dl.AddLine({box_max.x, y}, {box_max.x, glm::min(y + dash_size, box_max.y)}, outline_color, 1.0f);
        }
    }

    StartScreenTransform = {};
}

namespace {
std::optional<Mesh> PrimitiveEditor(PrimitiveType type, bool is_create = true) {
    const char *create_label = is_create ? "Add" : "Update";
    if (type == PrimitiveType::Rect) {
        static vec2 size{1, 1};
        InputFloat2("Size", &size.x);
        if (Button(create_label)) return Rect(size / 2.f);
    } else if (type == PrimitiveType::Circle) {
        static float r = 0.5;
        InputFloat("Radius", &r);
        if (Button(create_label)) return Circle(r);
    } else if (type == PrimitiveType::Cube) {
        static vec3 size{1.0, 1.0, 1.0};
        InputFloat3("Size", &size.x);
        if (Button(create_label)) return Cuboid(size / 2.f);
    } else if (type == PrimitiveType::IcoSphere) {
        static float r = 0.5;
        static int subdivisions = 3;
        InputFloat("Radius", &r);
        InputInt("Subdivisions", &subdivisions);
        if (Button(create_label)) return IcoSphere(r, uint(subdivisions));
    } else if (type == PrimitiveType::UVSphere) {
        static float r = 0.5;
        InputFloat("Radius", &r);
        if (Button(create_label)) return UVSphere(r);
    } else if (type == PrimitiveType::Torus) {
        static vec2 radii{0.5, 0.2};
        static glm::ivec2 n_segments = {32, 16};
        InputFloat2("Major/minor radius", &radii.x);
        InputInt2("Major/minor segments", &n_segments.x);
        if (Button(create_label)) return Torus(radii.x, radii.y, uint(n_segments.x), uint(n_segments.y));
    } else if (type == PrimitiveType::Cylinder) {
        static float r = 1, h = 1;
        InputFloat("Radius", &r);
        InputFloat("Height", &h);
        if (Button(create_label)) return Cylinder(r, h);
    } else if (type == PrimitiveType::Cone) {
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

    entt::entity activate_entity = entt::null, toggle_selected = entt::null;

    // Scene graph hierarchy (for debugging/future Ctrl+P parenting feature)
    if (const auto *node = R.try_get<SceneNode>(active_entity)) {
        if (auto parent_entity = node->Parent; parent_entity != entt::null) {
            AlignTextToFramePadding();
            Text("Parent: %s", GetName(R, parent_entity).c_str());
            SameLine();
            if (active_entity != parent_entity && Button("Activate##Parent")) activate_entity = parent_entity;
            SameLine();
            if (Button(R.all_of<Selected>(parent_entity) ? "Deselect##Parent" : "Select##Parent")) toggle_selected = parent_entity;
        }
        if (node->FirstChild != entt::null && CollapsingHeader("Children")) {
            RenderEntitiesTable("Children", active_entity);
        }
    }

    if (const auto *mesh_instance = R.try_get<MeshInstance>(active_entity)) {
        Text("Mesh entity: %s", GetName(R, mesh_instance->MeshEntity).c_str());
    }
    {
        const auto model_buffer_index = GetModelBufferIndex(R, active_entity);
        Text("Model buffer index: %s", model_buffer_index ? std::to_string(*model_buffer_index).c_str() : "None");
    }
    const auto active_mesh_entity = R.get<MeshInstance>(active_entity).MeshEntity;
    const auto &active_mesh = R.get<const Mesh>(active_mesh_entity);
    TextUnformatted(
        std::format("Vertices | Edges | Faces: {:L} | {:L} | {:L}", active_mesh.VertexCount(), active_mesh.EdgeCount(), active_mesh.FaceCount()).c_str()
    );
    Unindent();
    bool visible = R.all_of<Visible>(active_entity);
    if (Checkbox("Visible", &visible)) SetVisible(active_entity, visible);
    if (CollapsingHeader("Transform")) {
        bool model_changed = DragFloat3("Position", &R.get<Position>(active_entity).Value[0], 0.01f);
        // Rotation editor
        {
            int mode_i = R.get<RotationUiVariant>(active_entity).index();
            const char *modes[]{"Quat (WXYZ)", "XYZ Euler", "Axis Angle"};
            if (ImGui::Combo("Rotation mode", &mode_i, modes, IM_ARRAYSIZE(modes))) {
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
            UpdateWorldMatrix(R, active_entity);
            InvalidateCommandBuffer();
        }
        Spacing();
        {
            AlignTextToFramePadding();
            Text("Mode:");
            SameLine();
            using enum TransformGizmo::Mode;
            auto &mode = MGizmo.Mode;
            if (RadioButton("Local", mode == Local)) mode = Local;
            SameLine();
            if (RadioButton("World", mode == World)) mode = World;
            Spacing();
            Checkbox("Snap", &MGizmo.Config.Snap);
            if (MGizmo.Config.Snap) {
                SameLine();
                // todo link/unlink snap values
                DragFloat3("Snap", &MGizmo.Config.SnapValue.x, 1.f, 0.01f, 100.f);
            }
        }
        Spacing();
        if (TreeNode("Debug")) {
            if (const auto label = TransformGizmo::ToString(); label != "") {
                Text("%s op: %s", TransformGizmo::IsUsing() ? "Active" : "Hovered", label.data());
            } else {
                TextUnformatted("Not hovering");
            }
            TreePop();
        }
        if (TreeNode("World matrix")) {
            TextUnformatted("M");
            const auto &world_matrix = R.get<WorldMatrix>(active_entity);
            RenderMat4(world_matrix.M);
            Spacing();
            TextUnformatted("MInv");
            RenderMat4(world_matrix.MInv);
            TreePop();
        }
    }
    if (const auto *primitive_type = R.try_get<PrimitiveType>(active_mesh_entity)) {
        if (CollapsingHeader("Update primitive")) {
            if (auto new_mesh = PrimitiveEditor(*primitive_type, false)) {
                ReplaceMesh(active_mesh_entity, std::move(*new_mesh));
                InvalidateCommandBuffer();
            }
        }
    }
    PopID();

    if (activate_entity != entt::null) Select(activate_entity);
    else if (toggle_selected != entt::null) ToggleSelected(toggle_selected);
}

void Scene::RenderControls() {
    if (BeginTabBar("Scene controls")) {
        if (BeginTabItem("Object")) {
            {
                PushID("InteractionMode");
                AlignTextToFramePadding();
                TextUnformatted("Interaction mode:");
                int interaction_mode = int(InteractionMode);
                bool interaction_mode_changed = false;
                for (const auto mode : InteractionModes) {
                    SameLine();
                    interaction_mode_changed |= RadioButton(to_string(mode).c_str(), &interaction_mode, int(mode));
                }
                if (interaction_mode_changed) SetInteractionMode(::InteractionMode(interaction_mode));
                if (InteractionMode == InteractionMode::Edit) {
                    AlignTextToFramePadding();
                    TextUnformatted("Edit mode:");
                    auto type_interaction_mode = int(EditMode);
                    for (const auto element : Elements) {
                        auto name = Capitalize(label(element));
                        SameLine();
                        if (RadioButton(name.c_str(), &type_interaction_mode, int(element))) {
                            SetEditMode(element);
                        }
                    }
                    const auto active_entity = FindActiveEntity(R);
                    if (active_entity != entt::null) {
                        const auto mesh_entity = R.get<MeshInstance>(active_entity).MeshEntity;
                        const auto &selection = R.get<MeshSelection>(mesh_entity);
                        const bool matching_mode = selection.Element == EditMode;
                        const auto selected_count = matching_mode ? selection.Handles.size() : 0;
                        Text("Editing %s: %zu selected", label(EditMode).data(), selected_count);
                        if (EditMode == Element::Vertex && matching_mode && selected_count > 0) {
                            const auto &mesh = R.get<Mesh>(mesh_entity);
                            for (const auto vh : selection.Handles) {
                                const auto pos = mesh.GetPosition(VH{vh});
                                Text("Vertex %u: (%.4f, %.4f, %.4f)", vh, pos.x, pos.y, pos.z);
                            }
                        }
                    }
                }
                PopID();
            }
            if (!R.storage<Selected>().empty()) { // Selection actions
                SeparatorText("Selection actions");
                if (Button("Duplicate")) {
                    Duplicate();
                }
                SameLine();
                if (Button("Duplicate linked")) {
                    DuplicateLinked();
                }
                if (Button("Delete")) Delete();
            }
            RenderEntityControls(FindActiveEntity(R));

            if (CollapsingHeader("Add primitive")) {
                PushID("AddPrimitive");
                static auto selected_type_i = int(PrimitiveType::Cube);
                for (uint i = 0; i < PrimitiveTypes.size(); ++i) {
                    if (i % 3 != 0) SameLine();
                    RadioButton(ToString(PrimitiveTypes[i]).c_str(), &selected_type_i, i);
                }
                const auto selected_type = PrimitiveType(selected_type_i);
                if (auto mesh = PrimitiveEditor(selected_type, true)) {
                    R.emplace<PrimitiveType>(AddMesh(std::move(*mesh), {.Name = ToString(selected_type)}), selected_type);
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
                PopID();
            }

            if (CollapsingHeader("All objects")) {
                RenderEntitiesTable("All objects", entt::null);
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
            SeparatorText("Viewport shading");
            PushID("ViewportShading");
            auto viewport_shading = int(ViewportShading);
            bool viewport_shading_changed = RadioButton("Solid", &viewport_shading, int(ViewportShadingMode::Solid));
            SameLine();
            viewport_shading_changed |= RadioButton("Wireframe", &viewport_shading, int(ViewportShadingMode::Wireframe));
            PopID();

            bool smooth_shading_changed = Checkbox("Smooth shading", &SmoothShading);

            auto color_mode = int(ColorMode);
            bool color_mode_changed = false;
            if (ViewportShading == ViewportShadingMode::Solid) {
                SeparatorText("Fill color mode");
                PushID("ColorMode");
                color_mode_changed |= RadioButton("Mesh", &color_mode, int(ColorMode::Mesh));
                color_mode_changed |= RadioButton("Normals", &color_mode, int(ColorMode::Normals));
                PopID();
            }
            if (viewport_shading_changed || color_mode_changed || smooth_shading_changed) {
                ViewportShading = ::ViewportShadingMode(viewport_shading);
                ColorMode = ::ColorMode(color_mode);
                UpdateEdgeColors();
            }
            if (ViewportShading == ViewportShadingMode::Wireframe) {
                if (ColorEdit3("Edge color", &EdgeColor.x)) UpdateEdgeColors();
            }
            {
                SeparatorText("Active/Selected");
                bool color_changed = ColorEdit3("Active color", &Colors.Active[0]);
                color_changed |= ColorEdit3("Selected color", &Colors.Selected[0]);
                if (color_changed) {
                    UpdateTransformBuffers();
                }
                if (SliderUInt("Edge width", &SilhouetteEdgeWidth, 1, 4)) InvalidateCommandBuffer();
            }
            if (!R.view<Selected>().empty()) {
                SeparatorText("Selection overlays");
                AlignTextToFramePadding();
                TextUnformatted("Normals:");
                bool changed = false;
                for (const auto element : NormalElements) {
                    SameLine();
                    bool show = ShownNormalElements.contains(element);
                    const auto type_name = Capitalize(label(element));
                    if (Checkbox(type_name.c_str(), &show)) {
                        if (show) ShownNormalElements.insert(element);
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
                UpdateTransformBuffers();
            }
            EndTabItem();
        }
        EndTabBar();
    }
}

void Scene::RenderEntitiesTable(std::string name, entt::entity parent) {
    if (MeshEditor::BeginTable(name.c_str(), 3)) {
        static const float CharWidth = CalcTextSize("A").x;
        TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
        TableSetupColumn("Name");
        TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, CharWidth * 16);
        TableHeadersRow();
        entt::entity activate_entity = entt::null, toggle_selected = entt::null;

        auto render_entity = [&](entt::entity e) {
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
        };

        if (parent == entt::null) { // Iterate root entities
            for (const auto &[entity, node] : R.view<const SceneNode>().each()) {
                if (node.Parent == entt::null) render_entity(entity);
            }
        } else { // Iterate children
            for (const auto child : Children{&R, parent}) render_entity(child);
        }

        if (activate_entity != entt::null) Select(activate_entity);
        else if (toggle_selected != entt::null) ToggleSelected(toggle_selected);
        EndTable();
    }
}
