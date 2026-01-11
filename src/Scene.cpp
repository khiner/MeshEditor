#include "Scene.h"
#include "Widgets.h" // imgui

#include "Bindless.h"
#include "Entity.h"
#include "Excitable.h"
#include "OrientationGizmo.h"
#include "SceneTree.h"
#include "Shader.h"
#include "SvgResource.h"
#include "Timer.h"
#include "mesh/Arrow.h"
#include "mesh/MeshRender.h"
#include "mesh/Primitives.h"

#include <entt/entity/registry.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <imgui_internal.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <format>
#include <limits>
#include <ranges>
#include <unordered_map>
#include <variant>

using std::ranges::any_of, std::ranges::all_of, std::ranges::distance, std::ranges::find, std::ranges::find_if, std::ranges::fold_left, std::ranges::to;
using std::views::filter, std::views::iota, std::views::take, std::views::transform;

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
    vec4 BaseColor{0, 0, 0, 0};
    vec4 EdgeColor{0, 0, 0, 0};
    vec4 VertexUnselectedColor{0, 0, 0, 0};
    vec4 SelectedColor{0, 0, 0, 0};
    vec4 ActiveColor{0, 0, 0, 0};
};

// Tracks transform at start of gizmo manipulation. If present, actively manipulating.
struct StartTransform {
    Transform T;
};

// Scene render settings singleton component. Changes require command buffer re-recording.
struct SceneSettings {
    ViewportShadingMode ViewportShading{ViewportShadingMode::Solid};
    ColorMode ColorMode{ColorMode::Mesh};
    bool SmoothShading{false};
    bool ShowGrid{true};
    vk::ClearColorValue BackgroundColor{0.25f, 0.25f, 0.25f, 1.f};
    uint32_t SilhouetteEdgeWidth{1};
    InteractionMode InteractionMode{InteractionMode::Object};
    bool ShowBoundingBoxes{false};
    std::unordered_set<he::Element> ShownNormalElements{};
};

// Helper macros for SceneSettings access
#define SETTINGS R.get<const SceneSettings>(SettingsEntity)
#define PATCH_SETTINGS(fn) R.patch<SceneSettings>(SettingsEntity, fn)

enum class ShaderPipelineType {
    Fill,
    Line,
    Point,
    Grid,
    SilhouetteDepthObject,
    SilhouetteEdgeDepthObject,
    SilhouetteEdgeDepth,
    SilhouetteEdgeColor,
    SelectionElementFace,
    SelectionElementEdge,
    SelectionElementVertex,
    SelectionElementFaceXRay,
    SelectionElementEdgeXRay,
    SelectionElementVertexXRay,
    SelectionFragmentXRay,
    SelectionFragment,
    DebugNormals,
};
using SPT = ShaderPipelineType;

struct DrawPushConstants {
    uint32_t VertexSlot;
    uint32_t IndexSlot;
    uint32_t IndexOffset;
    uint32_t ModelSlot;
    uint32_t FirstInstance{0};
    uint32_t ObjectIdSlot{InvalidSlot};
    uint32_t FaceNormalSlot{InvalidSlot};
    uint32_t FaceIdOffset{0};
    uint32_t FaceNormalOffset{0};
    uint32_t VertexCountOrHeadImageSlot{0};
    uint32_t SelectionNodesSlot{0};
    uint32_t SelectionCounterSlot{0};
    uint32_t ElementIdOffset{0};
    uint32_t ElementStateSlot{InvalidSlot};
    uint32_t VertexOffset{0};
    uint32_t Pad0{0};
    vec4 LineColor{0};
};
static_assert(sizeof(DrawPushConstants) % 16 == 0, "DrawPushConstants must be 16-byte aligned.");

struct PipelineRenderer {
    vk::UniqueRenderPass RenderPass;
    std::unordered_map<SPT, ShaderPipeline> ShaderPipelines;

    void CompileShaders() {
        for (auto &shader_pipeline : std::views::values(ShaderPipelines)) shader_pipeline.Compile(*RenderPass);
    }

    const ShaderPipeline &Bind(vk::CommandBuffer cb, SPT spt) const {
        const auto &pipeline = ShaderPipelines.at(spt);
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline.Pipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline.PipelineLayout, 0, pipeline.GetDescriptorSet(), {});
        return pipeline;
    }
};

namespace {
std::vector<uint32_t> MakeElementStates(size_t count) { return std::vector<uint32_t>(std::max<size_t>(count, 1u), 0); }

// Returns primary edit instance per selected mesh: active instance if selected, else first selected instance.
std::unordered_map<entt::entity, entt::entity> ComputePrimaryEditInstances(entt::registry &r) {
    std::unordered_map<entt::entity, entt::entity> primaries;
    const auto active = FindActiveEntity(r);
    for (const auto [e, mi] : r.view<const MeshInstance, const Selected>().each()) {
        auto &primary = primaries[mi.MeshEntity];
        if (primary == entt::entity{} || e == active) primary = e;
    }
    return primaries;
}
} // namespace

struct MeshSelection {
    Element Element{Element::None};
    std::vector<uint32_t> Handles{};
    std::optional<uint32_t> ActiveHandle{}; // Most recently selected element (always in Handles if set)
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
}
void Scene::ToggleSelected(entt::entity e) {
    if (e == entt::null) return;

    if (R.all_of<Selected>(e)) R.remove<Selected>(e);
    else R.emplace_or_replace<Selected>(e);
}

std::vector<Vertex3D> CreateBoxVertices(const BBox &box) {
    return box.Corners() |
        // Normals don't matter for wireframes.
        transform([](const auto &corner) { return Vertex3D{corner, vec3{}}; }) |
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

    r.patch<RotationUiVariant>(e, [&](auto &rotation_ui) {
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
            rotation_ui
        );
    });
}

void SetTransform(entt::registry &r, entt::entity e, const Transform &t) {
    r.emplace_or_replace<Position>(e, t.P);
    // Avoid replacing rotation UI slider values if the value hasn't changed.
    if (!r.all_of<Rotation>(e) || r.get<Rotation>(e).Value != t.R) SetRotation(r, e, t.R);
    // Frozen entities can't have their scale changed.
    if (!r.all_of<Frozen>(e)) r.emplace_or_replace<Scale>(e, t.S);

    UpdateWorldMatrix(r, e);
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
    float Depth;
    uint32_t ObjectId;
    uint32_t Next;
    uint32_t Padding0;
};
static_assert(sizeof(SelectionNode) == 16, "SelectionNode must match std430 layout.");

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

struct ClickElementCandidate {
    uint32_t ObjectId;
    float Depth;
    uint32_t DistanceSq;
    uint32_t Padding;
};
static_assert(sizeof(ClickElementCandidate) == 16, "ClickElementCandidate must match std430 layout.");

struct ClickSelectPushConstants {
    glm::uvec2 TargetPx;
    uint32_t HeadImageIndex;
    uint32_t SelectionNodesIndex;
    uint32_t ClickResultIndex;
};

struct ClickSelectElementPushConstants {
    glm::uvec2 TargetPx;
    uint32_t Radius;
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

constexpr uint32_t ClickSelectRadiusPx = 50;
constexpr uint32_t ClickSelectDiameterPx = ClickSelectRadiusPx * 2 + 1;
constexpr uint32_t ClickSelectPixelCount = ClickSelectDiameterPx * ClickSelectDiameterPx;

void WaitFor(vk::Fence fence, vk::Device device) {
    if (auto wait_result = device.waitForFences(fence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    device.resetFences(fence);
}
} // namespace

// Owns render-only/generated data (e.g., SceneUBO/indicators/overlays/selection fragments).
struct SceneBuffers {
    static constexpr uint32_t MaxSelectableObjects{100'000};
    static constexpr uint32_t BoxSelectBitsetWords{(MaxSelectableObjects + 31) / 32};
    static constexpr uint32_t ClickElementGroupSize{256};
    static constexpr uint32_t ClickSelectElementGroupCount{(ClickSelectPixelCount + ClickElementGroupSize - 1) / ClickElementGroupSize};
    static constexpr uint32_t SelectionNodesPerPixel{10};
    static constexpr uint32_t MaxSelectionNodeBytes{64 * 1024 * 1024};

    SceneBuffers(vk::PhysicalDevice pd, vk::Device d, vk::Instance instance, DescriptorSlots &slots)
        : Ctx{pd, d, instance, slots},
          VertexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer},
          FaceIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          EdgeIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          VertexIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          SceneUBO{Ctx, sizeof(SceneUBO), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::Uniform},
          SelectionNodeBuffer{Ctx, sizeof(SelectionNode), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          SelectionCounterBuffer{Ctx, sizeof(SelectionCounters), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ClickResultBuffer{Ctx, sizeof(ClickResult), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ClickElementResultBuffer{Ctx, ClickSelectElementGroupCount * sizeof(ClickElementCandidate), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          BoxSelectBitsetBuffer{Ctx, BoxSelectBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer} {}

    const ClickResult &GetClickResult() const { return *reinterpret_cast<const ClickResult *>(ClickResultBuffer.GetData().data()); }
    vk::DescriptorBufferInfo GetBoxSelectBitsetDescriptor() const { return {*BoxSelectBitsetBuffer, 0, BoxSelectBitsetWords * sizeof(uint32_t)}; }

    SlottedBufferRange CreateIndices(std::span<const uint> indices, IndexKind index_kind) {
        auto &index_buffer = GetIndexBuffer(index_kind);
        return {index_buffer.Allocate(indices), index_buffer.Buffer.Slot};
    }
    RenderBuffers CreateRenderBuffers(std::span<const Vertex3D> vertices, std::span<const uint> indices, IndexKind index_kind) {
        return {VertexBuffer.Allocate(vertices), CreateIndices(indices, index_kind), index_kind};
    }

    void Release(RenderBuffers &buffers) {
        VertexBuffer.Release(buffers.Vertices);
        buffers.Vertices = {};
        GetIndexBuffer(buffers.IndexType).Release(buffers.Indices.Range);
        buffers.Indices.Range = {};
    }

    void Release(MeshBuffers &buffers) {
        FaceIndexBuffer.Release(buffers.FaceIndices.Range);
        buffers.FaceIndices.Range = {};
        EdgeIndexBuffer.Release(buffers.EdgeIndices.Range);
        buffers.EdgeIndices.Range = {};
        VertexIndexBuffer.Release(buffers.VertexIndices.Range);
        buffers.VertexIndices.Range = {};
        for (auto &[_, rb] : buffers.NormalIndicators) Release(rb);
        buffers.NormalIndicators.clear();
    }

    BufferArena<uint32_t> &GetIndexBuffer(IndexKind kind) {
        switch (kind) {
            case IndexKind::Face: return FaceIndexBuffer;
            case IndexKind::Edge: return EdgeIndexBuffer;
            case IndexKind::Vertex: return VertexIndexBuffer;
        }
    }

    void ResizeSelectionNodeBuffer(vk::Extent2D extent) {
        const uint64_t pixels = uint64_t(extent.width) * extent.height;
        const uint64_t desired_nodes = pixels == 0 ? 1 : pixels * SelectionNodesPerPixel;
        const uint64_t max_nodes = std::max<uint64_t>(1, MaxSelectionNodeBytes / sizeof(SelectionNode));
        const uint32_t final_count = std::min<uint64_t>(std::min(desired_nodes, max_nodes), std::numeric_limits<uint32_t>::max());
        if (final_count == SelectionNodeCapacity) return;
        SelectionNodeCapacity = final_count;
        SelectionNodeBuffer = {Ctx, SelectionNodeCapacity * sizeof(SelectionNode), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer};
    }

    mvk::BufferContext Ctx;
    BufferArena<Vertex3D> VertexBuffer;
    BufferArena<uint32_t> FaceIndexBuffer, EdgeIndexBuffer, VertexIndexBuffer;
    mvk::Buffer SceneUBO;
    uint32_t SelectionNodeCapacity{1};
    mvk::Buffer SelectionNodeBuffer;
    // CPU readback buffers (host-visible)
    mvk::Buffer SelectionCounterBuffer, ClickResultBuffer, ClickElementResultBuffer, BoxSelectBitsetBuffer;
};

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

    auto cb = std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    cb->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    {
        // Write the bitmap into a temporary staging buffer.
        mvk::Buffer staging_buffer{Buffers->Ctx, as_bytes(data), mvk::MemoryUsage::CpuOnly};
        // Transition the image layout to be ready for data transfer.
        cb->pipelineBarrier(
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
        cb->copyBufferToImage(
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
        cb->pipelineBarrier(
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
        cb->end();

        vk::SubmitInfo submit;
        submit.setCommandBuffers(*cb);
        Vk.Queue.submit(submit, *OneShotFence);
        WaitFor(*OneShotFence, Vk.Device);
    } // staging buffer is destroyed here

    Buffers->Ctx.ReclaimRetiredBuffers();

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
            // Single-sampled resolve target. UNDEFINED + DONT_CARE = discard previous contents, let render pass handle transition.
            {{}, Format::Color, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore, {}, {}, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal},
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
                vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t) * 3} // Manipulating flag + sampler index + active object id
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
            // Store depth for reuse by element selection (mutual occlusion between selected meshes).
            {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal},
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
    // Render pass that loads depth from silhouette pass for element occlusion
    static PipelineRenderer CreateRenderer(vk::Device d, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}) {
        const std::vector<vk::AttachmentDescription> attachments{
            {{}, Format::Depth, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eDepthStencilAttachmentOptimal},
        };
        const vk::AttachmentReference depth_attachment_ref{0, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        const vk::SubpassDescription subpass{{}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 0, nullptr, nullptr, &depth_attachment_ref};

        const PipelineContext ctx{d, shared_layout, shared_set, vk::SampleCountFlagBits::e1};
        const vk::PushConstantRange draw_pc{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPushConstants)};
        std::unordered_map<SPT, ShaderPipeline> pipelines;
        pipelines.emplace(
            SPT::SelectionElementFace,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementFace.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                {}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionElementFaceXRay,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementFace.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                {}, CreateDepthStencil(false, false), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionElementEdge,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementEdge.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
                {}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionElementEdgeXRay,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementEdge.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eLineList,
                {}, CreateDepthStencil(false, false), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionElementVertex,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementVertex.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
                {}, CreateDepthStencil(true, false, vk::CompareOp::eLessOrEqual), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionElementVertexXRay,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "SelectionElementVertex.vert"}, {ShaderType::eFragment, "SelectionElement.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::ePointList,
                {}, CreateDepthStencil(false, false), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionFragment,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "SelectionFragment.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                {}, CreateDepthStencil(), draw_pc
            )
        );
        pipelines.emplace(
            SPT::SelectionFragmentXRay,
            ctx.CreateGraphics(
                {{{ShaderType::eVertex, "PositionTransform.vert"}, {ShaderType::eFragment, "SelectionFragment.frag"}}},
                {},
                vk::PolygonMode::eFill, vk::PrimitiveTopology::eTriangleList,
                {}, CreateDepthStencil(false, false), draw_pc
            )
        );
        return {d.createRenderPassUnique({{}, attachments, subpass}), std::move(pipelines)};
    }

    struct ResourcesT {
        ResourcesT(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::RenderPass render_pass, vk::ImageView silhouette_depth_view)
            : HeadImage{mvk::CreateImage(
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
              )},
              Framebuffer{d.createFramebufferUnique({{}, render_pass, silhouette_depth_view, extent.width, extent.height, 1})} {}

        mvk::ImageResource HeadImage;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D extent, vk::Device d, vk::PhysicalDevice pd, vk::ImageView silhouette_depth_view) {
        Resources = std::make_unique<ResourcesT>(extent, d, pd, *Renderer.RenderPass, silhouette_depth_view);
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
          ClickSelectElement{
              d, Shaders{{{ShaderType::eCompute, "ClickSelectElement.comp"}}},
              vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ClickSelectElementPushConstants)},
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
    ComputePipeline ClickSelectElement;
    ComputePipeline BoxSelect;

    void SetExtent(vk::Extent2D);
    void CompileShaders() {
        Main.Renderer.CompileShaders();
        Silhouette.Renderer.CompileShaders();
        SilhouetteEdge.Renderer.CompileShaders();
        SelectionFragment.Renderer.CompileShaders();
        ClickSelect.Compile();
        ClickSelectElement.Compile();
        BoxSelect.Compile();
    }
};

struct Scene::SelectionSlotHandles {
    explicit SelectionSlotHandles(DescriptorSlots &slots)
        : Slots(slots),
          HeadImage(slots.Allocate(SlotType::Image)),
          SelectionCounter(slots.Allocate(SlotType::Buffer)),
          ClickResult(slots.Allocate(SlotType::Buffer)),
          ClickElementResult(slots.Allocate(SlotType::Buffer)),
          BoxResult(slots.Allocate(SlotType::Buffer)),
          ObjectIdSampler(slots.Allocate(SlotType::Sampler)),
          DepthSampler(slots.Allocate(SlotType::Sampler)),
          SilhouetteSampler(slots.Allocate(SlotType::Sampler)) {}

    ~SelectionSlotHandles() {
        Slots.Release({SlotType::Image, HeadImage});
        Slots.Release({SlotType::Buffer, SelectionCounter});
        Slots.Release({SlotType::Buffer, ClickResult});
        Slots.Release({SlotType::Buffer, ClickElementResult});
        Slots.Release({SlotType::Buffer, BoxResult});
        Slots.Release({SlotType::Sampler, ObjectIdSampler});
        Slots.Release({SlotType::Sampler, DepthSampler});
        Slots.Release({SlotType::Sampler, SilhouetteSampler});
    }

    DescriptorSlots &Slots;
    uint32_t HeadImage, SelectionCounter, ClickResult, ClickElementResult, BoxResult, ObjectIdSampler, DepthSampler, SilhouetteSampler;
};

// Unmanaged reactive storage to track entity destruction.
// Unlike managed storage (via R.storage<entt::reactive>()), this keeps entities after destruction
// until manually cleared, allowing ProcessComponentEvents to detect that entities were deleted.
struct Scene::EntityDestroyTracker {
    entt::storage_for_t<entt::reactive> Storage;

    void Bind(entt::registry &r) {
        Storage.bind(r);
        Storage.on_destroy<Visible>();
    }
};

Scene::Scene(SceneVulkanResources vc, entt::registry &r)
    : Vk{vc},
      R{r},
      CommandPool{Vk.Device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, Vk.QueueFamily})},
      RenderCommandBuffer{std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())},
#ifdef MVK_FORCE_STAGED_TRANSFERS
      TransferCommandBuffer{std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())},
#endif
      RenderFence{Vk.Device.createFenceUnique({})},
      OneShotFence{Vk.Device.createFenceUnique({})},
      SelectionReadySemaphore{Vk.Device.createSemaphoreUnique({})},
      ClickCommandBuffer{std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())},
      Slots{std::make_unique<DescriptorSlots>(Vk.Device, MakeBindlessConfig(Vk.PhysicalDevice))},
      SelectionHandles{std::make_unique<SelectionSlotHandles>(*Slots)},
      DestroyTracker{std::make_unique<EntityDestroyTracker>()},
      Pipelines{std::make_unique<ScenePipelines>(
          Vk.Device, Vk.PhysicalDevice,
          Slots->GetSetLayout(), Slots->GetSet()
      )},
      Buffers{std::make_unique<SceneBuffers>(Vk.PhysicalDevice, Vk.Device, Vk.Instance, *Slots)},
      Meshes{Buffers->Ctx} {
    // Reactive storage subscriptions for deferred once-per-frame processing
    using namespace entt::literals;

    R.storage<entt::reactive>("selected_changes"_hs)
        .on_construct<Selected>()
        .on_destroy<Selected>();
    R.storage<entt::reactive>("visible_changes"_hs)
        .on_construct<Visible>()
        .on_destroy<Visible>();
    R.storage<entt::reactive>("active_changes"_hs)
        .on_construct<Active>()
        .on_destroy<Active>();

    R.storage<entt::reactive>("mesh_selection_changes"_hs)
        .on_construct<MeshSelection>()
        .on_update<MeshSelection>();
    R.storage<entt::reactive>("excitable_changes"_hs)
        .on_construct<Excitable>()
        .on_update<Excitable>()
        .on_destroy<Excitable>();
    R.storage<entt::reactive>("excited_vertex_changes"_hs)
        .on_construct<ExcitedVertex>()
        .on_destroy<ExcitedVertex>();
    R.storage<entt::reactive>("models_buffer_changes"_hs)
        .on_update<ModelsBuffer>();
    R.storage<entt::reactive>("start_transform_changes"_hs)
        .on_construct<StartTransform>()
        .on_destroy<StartTransform>();
    R.storage<entt::reactive>("scene_settings_changes"_hs)
        .on_update<SceneSettings>();
    // temp: Keep immediate handler for ExcitedVertex destroy (needs to capture indicator entity before removal)
    R.on_destroy<ExcitedVertex>().connect<&Scene::OnDestroyExcitedVertex>(*this);

    DestroyTracker->Bind(R);

    SettingsEntity = R.create();
    R.emplace<SceneSettings>(SettingsEntity);

    UpdateEdgeColors();
    UpdateSceneUBO();
    BoxSelectZeroBits.assign(SceneBuffers::BoxSelectBitsetWords, 0);

    Pipelines->CompileShaders();

    { // Default scene content
        static constexpr int kGrid = 10;
        static constexpr float kSpacing = 2.0f;
        for (int z = 0; z < kGrid; ++z) {
            for (int y = 0; y < kGrid; ++y) {
                for (int x = 0; x < kGrid; ++x) {
                    const auto e = AddMesh(
                        CreateDefaultPrimitive(PrimitiveType::Cube),
                        {.Name = ToString(PrimitiveType::Cube),
                         .Transform = {.P = vec3{x, y, z} * kSpacing},
                         .Select = MeshCreateInfo::SelectBehavior::None}
                    );
                    R.emplace<PrimitiveType>(e.first, PrimitiveType::Cube);
                }
            }
        }
        const float extent{(kGrid - 1) * kSpacing};
        const vec3 center{extent * 0.5f};
        Camera = {center + vec3{extent}, center, glm::radians(60.f), 0.01f, 500.f};
    }
}

Scene::~Scene() {
    R.clear<Mesh>();
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

void Scene::ProcessComponentEvents() {
    using namespace entt::literals;

    std::unordered_set<entt::entity> dirty_overlay_meshes, dirty_element_state_meshes;

    { // Selected changes
        auto &selected_tracker = R.storage<entt::reactive>("selected_changes"_hs);
        if (!selected_tracker.empty()) CommandBufferDirty = NeedsRender = true;
        for (auto instance_entity : selected_tracker) {
            if (auto *mi = R.try_get<MeshInstance>(instance_entity)) {
                const auto mesh_entity = mi->MeshEntity;
                if (R.all_of<Selected>(instance_entity)) {
                    // Construct: mark mesh for overlay update
                    dirty_overlay_meshes.insert(mesh_entity);
                } else {
                    // Destroy: check if mesh has no more selected instances
                    bool has_selected = false;
                    for (const auto [e, other_mi] : R.view<MeshInstance, Selected>().each()) {
                        if (other_mi.MeshEntity == mesh_entity) {
                            has_selected = true;
                            break;
                        }
                    }
                    if (!has_selected) {
                        // Clean up overlays for this mesh
                        if (auto *buffers = R.try_get<MeshBuffers>(mesh_entity)) {
                            for (auto &[_, rb] : buffers->NormalIndicators) Buffers->Release(rb);
                            buffers->NormalIndicators.clear();
                        }
                        if (auto *bbox = R.try_get<BoundingBoxesBuffers>(mesh_entity)) {
                            Buffers->Release(bbox->Buffers);
                        }
                        R.remove<BoundingBoxesBuffers>(mesh_entity);
                    }
                }
            }
        }
        selected_tracker.clear();
    }
    { // Visible changes
        auto &visible_tracker = R.storage<entt::reactive>("visible_changes"_hs);
        if (!visible_tracker.empty()) CommandBufferDirty = NeedsRender = true;
        visible_tracker.clear();
    }
    { // Active changes
        auto &active_tracker = R.storage<entt::reactive>("active_changes"_hs);
        if (!active_tracker.empty()) CommandBufferDirty = NeedsRender = true;
        active_tracker.clear();
    }
    { // Entity destruction
        if (!DestroyTracker->Storage.empty()) CommandBufferDirty = NeedsRender = true;
        DestroyTracker->Storage.clear();
    }
    { // MeshSelection changes
        auto &mesh_selection_tracker = R.storage<entt::reactive>("mesh_selection_changes"_hs);
        for (auto mesh_entity : mesh_selection_tracker) {
            if (R.all_of<MeshSelection>(mesh_entity)) {
                if (!R.all_of<MeshElementStateBuffers>(mesh_entity)) {
                    const auto &mesh = R.get<Mesh>(mesh_entity);
                    R.emplace<MeshElementStateBuffers>(
                        mesh_entity,
                        mvk::Buffer{Buffers->Ctx, as_bytes(MakeElementStates(mesh.FaceCount())), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
                        mvk::Buffer{Buffers->Ctx, as_bytes(MakeElementStates(mesh.EdgeCount() * 2)), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
                        mvk::Buffer{Buffers->Ctx, as_bytes(MakeElementStates(mesh.VertexCount())), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer}
                    );
                }
                dirty_element_state_meshes.insert(mesh_entity);
            }
        }
        mesh_selection_tracker.clear();
    }
    { // Excitable changes
        auto &excitable_tracker = R.storage<entt::reactive>("excitable_changes"_hs);
        for (auto instance_entity : excitable_tracker) {
            if (auto *mi = R.try_get<MeshInstance>(instance_entity)) {
                dirty_element_state_meshes.insert(mi->MeshEntity);
            }
            if (R.all_of<Excitable>(instance_entity)) {
                InteractionModes.insert(InteractionMode::Excite);
            }
        }
        if (R.storage<Excitable>().empty()) {
            if (SETTINGS.InteractionMode == InteractionMode::Excite) SetInteractionMode(*InteractionModes.begin());
            InteractionModes.erase(InteractionMode::Excite);
        } else if (!excitable_tracker.empty()) {
            SetInteractionMode(InteractionMode::Excite); // If we just added excitables, switch to excite mode
        }
        excitable_tracker.clear();
    }
    { // ExcitedVertex changes
        auto &excited_vertex_tracker = R.storage<entt::reactive>("excited_vertex_changes"_hs);
        for (auto instance_entity : excited_vertex_tracker) {
            if (auto *mi = R.try_get<MeshInstance>(instance_entity)) {
                dirty_element_state_meshes.insert(mi->MeshEntity);
            }

            if (auto *ev = R.try_get<ExcitedVertex>(instance_entity)) {
                // Construct: create indicator if not exists
                if (ev->IndicatorEntity == entt::null) {
                    const auto mesh_entity = R.get<MeshInstance>(instance_entity).MeshEntity;
                    const auto &mesh = R.get<Mesh>(mesh_entity);
                    const auto &transform = R.get<WorldMatrix>(instance_entity).M;
                    const auto vh = VH(ev->Vertex);
                    const vec3 vertex_pos{transform * vec4{mesh.GetPosition(vh), 1}};

                    // Orient camera towards excited vertex
                    Camera.SetTargetDirection(glm::normalize(vertex_pos - Camera.Target));

                    // Create indicator arrow
                    const auto bbox = MeshRender::ComputeBoundingBox(mesh);
                    const vec3 normal{transform * vec4{mesh.GetNormal(vh), 0}};
                    const float scale_factor = 0.1f * glm::length(bbox.Max - bbox.Min);
                    const auto [_, indicator_entity] = AddMesh(
                        Meshes.CreateMesh(Arrow()),
                        {.Name = "Excite vertex indicator",
                         .Transform = {
                             .P = vertex_pos + 0.05f * scale_factor * normal,
                             .R = glm::rotation(World.Up, normal),
                             .S = vec3{scale_factor},
                         },
                         .Select = MeshCreateInfo::SelectBehavior::None}
                    );
                    R.patch<ExcitedVertex>(instance_entity, [indicator_entity](auto &e) { e.IndicatorEntity = indicator_entity; });
                }
            }
            // Note: ExcitedVertex destroy is handled by immediate callback (OnDestroyExcitedVertex)
            // which captures the indicator entity before component removal
        }
        excited_vertex_tracker.clear();
    }

    { // ModelsBuffer changes (buffer data update, not structure)
        auto &models_tracker = R.storage<entt::reactive>("models_buffer_changes"_hs);
        if (!models_tracker.empty()) NeedsRender = true;
        models_tracker.clear();
    }
    { // StartTransform changes (manipulation state - silhouette push constant requires re-record)
        auto &start_transform_tracker = R.storage<entt::reactive>("start_transform_changes"_hs);
        if (!start_transform_tracker.empty()) CommandBufferDirty = NeedsRender = true;
        start_transform_tracker.clear();
    }
    { // SceneSettings changes
        auto &settings_tracker = R.storage<entt::reactive>("scene_settings_changes"_hs);
        if (!settings_tracker.empty()) CommandBufferDirty = NeedsRender = true;
        settings_tracker.clear();
    }

    // Apply batched updates
    for (const auto mesh_entity : dirty_overlay_meshes) UpdateEntitySelectionOverlays(mesh_entity);
    for (const auto mesh_entity : dirty_element_state_meshes) UpdateMeshElementStateBuffers(mesh_entity);

    if (!dirty_overlay_meshes.empty() || !dirty_element_state_meshes.empty()) {
        CommandBufferDirty = NeedsRender = true;
    }
}

void Scene::OnDestroyExcitedVertex(entt::registry &r, entt::entity e) {
    if (const auto indicator = r.get<ExcitedVertex>(e).IndicatorEntity; indicator != entt::null) {
        Destroy(indicator);
    }
}

vk::ImageView Scene::GetViewportImageView() const { return *Pipelines->Main.Resources->ResolveImage.View; }

namespace {
void UpdateVisibleObjectIds(entt::registry &r) {
    std::unordered_map<entt::entity, std::vector<uint32_t>> ids_by_mesh;
    uint32_t object_id = 1;
    for (const auto e : r.view<Visible>()) {
        const auto mesh_entity = r.get<MeshInstance>(e).MeshEntity;
        const auto &models = r.get<const ModelsBuffer>(mesh_entity);
        const uint32_t instance_count = models.Buffer.UsedSize / sizeof(WorldMatrix);
        if (instance_count == 0) continue;
        auto &ids = ids_by_mesh[mesh_entity];
        if (ids.empty()) ids.resize(instance_count, 0);
        const auto buffer_index = r.get<const RenderInstance>(e).BufferIndex;
        if (buffer_index < ids.size()) ids[buffer_index] = object_id;
        r.patch<RenderInstance>(e, [object_id](auto &ri) { ri.ObjectId = object_id; });
        ++object_id;
    }
    for (const auto &[mesh_entity, ids] : ids_by_mesh) {
        r.patch<ModelsBuffer>(mesh_entity, [&ids](auto &mb) { mb.ObjectIds.Update(as_bytes(ids)); });
    }
}
} // namespace

void Scene::SetVisible(entt::entity entity, bool visible) {
    const bool already_visible = R.all_of<Visible>(entity);
    if ((visible && already_visible) || (!visible && !already_visible)) return;

    const auto mesh_entity = R.get<MeshInstance>(entity).MeshEntity;
    if (visible) {
        const auto buffer_index = R.get<const ModelsBuffer>(mesh_entity).Buffer.UsedSize / sizeof(WorldMatrix);
        R.emplace_or_replace<RenderInstance>(entity, buffer_index, 0u);
        R.patch<ModelsBuffer>(mesh_entity, [&](auto &mb) {
            mb.Buffer.Insert(as_bytes(R.get<WorldMatrix>(entity)), mb.Buffer.UsedSize);
            mb.ObjectIds.Insert(as_bytes(uint32_t{0}), mb.ObjectIds.UsedSize); // Placeholder; actual IDs set on-demand.
        });
        R.emplace<Visible>(entity);
    } else {
        R.remove<Visible>(entity);
        const uint old_model_index = R.get<const RenderInstance>(entity).BufferIndex;
        R.patch<RenderInstance>(entity, [](auto &ri) { ri.ObjectId = 0; });
        R.patch<ModelsBuffer>(mesh_entity, [old_model_index](auto &mb) {
            mb.Buffer.Erase(old_model_index * sizeof(WorldMatrix), sizeof(WorldMatrix));
            mb.ObjectIds.Erase(old_model_index * sizeof(uint32_t), sizeof(uint32_t));
        });
        // Update buffer indices for all instances of this mesh that have higher indices
        for (const auto [other_entity, mesh_instance, ri] : R.view<MeshInstance, const RenderInstance>().each()) {
            if (mesh_instance.MeshEntity == mesh_entity && ri.BufferIndex > old_model_index) {
                R.patch<RenderInstance>(other_entity, [](auto &ri) { --ri.BufferIndex; });
            }
        }
        // Also check if the mesh entity itself is visible (it might not have MeshInstance)
        if (mesh_entity != entity) {
            if (const auto *mesh_ri = R.try_get<const RenderInstance>(mesh_entity)) {
                if (mesh_ri->BufferIndex > old_model_index) {
                    R.patch<RenderInstance>(mesh_entity, [](auto &ri) { --ri.BufferIndex; });
                }
            }
        }
    }
    UpdateVisibleObjectIds(R);
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(Mesh &&mesh, MeshCreateInfo info) {
    const auto mesh_entity = R.create();
    { // Mesh data
        SlottedBufferRange vertices{Meshes.GetVerticesRange(mesh.GetStoreId()), Meshes.GetVerticesSlot()};
        auto face_indices = Buffers->CreateIndices(mesh.CreateTriangleIndices(), IndexKind::Face);
        auto edge_indices = Buffers->CreateIndices(mesh.CreateEdgeIndices(), IndexKind::Edge);
        auto vertex_indices = Buffers->CreateIndices(MeshRender::CreateVertexIndices(mesh), IndexKind::Vertex);

        R.emplace<Mesh>(mesh_entity, std::move(mesh));
        R.emplace<ModelsBuffer>(
            mesh_entity,
            mvk::Buffer{Buffers->Ctx, sizeof(WorldMatrix), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ModelBuffer},
            mvk::Buffer{Buffers->Ctx, sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer}
        );
        R.emplace<MeshBuffers>(mesh_entity, vertices, face_indices, edge_indices, vertex_indices);
        R.emplace<MeshSelection>(mesh_entity);
    }

    const auto instance_entity = R.create();
    R.emplace<MeshInstance>(instance_entity, mesh_entity);
    SetTransform(R, instance_entity, info.Transform);
    R.emplace<Name>(instance_entity, CreateName(R, info.Name));

    R.patch<ModelsBuffer>(mesh_entity, [](auto &mb) {
        mb.Buffer.Reserve(mb.Buffer.UsedSize + sizeof(WorldMatrix));
        mb.ObjectIds.Reserve(mb.ObjectIds.UsedSize + sizeof(uint32_t));
    });
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
            // If no mesh is active yet, activate the new one.
            if (R.storage<Active>().empty()) {
                R.emplace<Active>(instance_entity);
                R.emplace_or_replace<Selected>(instance_entity);
            }
            break;
    }

    return {mesh_entity, instance_entity};
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(MeshData &&data, MeshCreateInfo info) {
    return AddMesh(Meshes.CreateMesh(std::move(data)), std::move(info));
}

std::pair<entt::entity, entt::entity> Scene::AddMesh(const fs::path &path, MeshCreateInfo info) {
    auto mesh = Meshes.LoadMesh(path);
    if (!mesh) throw std::runtime_error(std::format("Failed to load mesh: {}", path.string()));
    const auto e = AddMesh(std::move(*mesh), std::move(info));
    R.emplace<Path>(e.first, path);
    return e;
}

entt::entity Scene::Duplicate(entt::entity e, std::optional<MeshCreateInfo> info) {
    const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
    const auto e_new = AddMesh(
        Meshes.CloneMesh(R.get<const Mesh>(mesh_entity)),
        info.value_or(MeshCreateInfo{
            .Name = std::format("{}_copy", GetName(R, e)),
            .Transform = GetTransform(R, e),
            .Select = R.all_of<Selected>(e) ? MeshCreateInfo::SelectBehavior::Additive : MeshCreateInfo::SelectBehavior::None,
            .Visible = R.all_of<Visible>(e),
        })
    );
    if (auto primitive_type = R.try_get<PrimitiveType>(mesh_entity)) R.emplace<PrimitiveType>(e_new.first, *primitive_type);
    return e_new.second;
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
    R.patch<ModelsBuffer>(mesh_entity, [](auto &mb) {
        mb.Buffer.Reserve(mb.Buffer.UsedSize + sizeof(WorldMatrix));
        mb.ObjectIds.Reserve(mb.ObjectIds.UsedSize + sizeof(uint32_t));
    });
    SetTransform(R, e_new, info ? info->Transform : GetTransform(R, e));
    SetVisible(e_new, !info || info->Visible);

    if (!info || info->Select == MeshCreateInfo::SelectBehavior::Additive) R.emplace<Selected>(e_new);
    else if (info->Select == MeshCreateInfo::SelectBehavior::Exclusive) Select(e_new);

    return e_new;
}

void Scene::Delete() {
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) entities.emplace_back(e);
    for (const auto e : entities) Destroy(e);
}

void Scene::Duplicate() {
    const Timer timer{"Duplicate"};
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) entities.emplace_back(e);
    for (const auto e : entities) {
        const auto new_e = Duplicate(e);
        if (R.all_of<Active>(e)) {
            R.remove<Active>(e);
            R.emplace<Active>(new_e);
            R.emplace_or_replace<Selected>(new_e);
        }
        R.remove<Selected>(e);
    }
    StartScreenTransform = TransformGizmo::TransformType::Translate;
}
void Scene::DuplicateLinked() {
    const Timer timer{"DuplicateLinked"};
    std::vector<entt::entity> entities;
    for (const auto e : R.view<Selected>()) entities.emplace_back(e);
    for (const auto e : entities) {
        const auto new_e = DuplicateLinked(e);
        if (R.all_of<Active>(e)) {
            R.remove<Active>(e);
            R.emplace<Active>(new_e);
            R.emplace_or_replace<Selected>(new_e);
        }
        R.remove<Selected>(e);
    }
    StartScreenTransform = TransformGizmo::TransformType::Translate;
}

void Scene::ClearMeshes() {
    std::vector<entt::entity> entities;
    for (const auto e : R.view<MeshInstance>()) entities.emplace_back(e);
    for (const auto e : entities) Destroy(e);
}

void Scene::SetMeshPositions(entt::entity e, std::span<const vec3> positions) {
    const auto &mesh = R.get<const Mesh>(e);
    Meshes.SetPositions(mesh, positions);
    UpdateEntitySelectionOverlays(e);
    UpdateMeshElementStateBuffers(e);
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
        const auto has_instances = any_of(
            R.view<MeshInstance>().each(),
            [mesh_entity](const auto &entry) { return std::get<1>(entry).MeshEntity == mesh_entity; }
        );
        if (!has_instances) {
            if (auto *mesh_buffers = R.try_get<MeshBuffers>(mesh_entity)) Buffers->Release(*mesh_buffers);
            R.destroy(mesh_entity);
        }
    }
}

void Scene::SetInteractionMode(InteractionMode mode) {
    if (SETTINGS.InteractionMode == mode) return;

    PATCH_SETTINGS([mode](auto &s) { s.InteractionMode = mode; });
    for (const auto &entity : R.view<Mesh>()) UpdateMeshElementStateBuffers(entity);
}
void Scene::SetEditMode(Element mode) {
    if (EditMode == mode) return;

    EditMode = mode;
    for (const auto &[e, selection, mesh] : R.view<MeshSelection, Mesh>().each()) {
        R.replace<MeshSelection>(e, MeshSelection{EditMode, ConvertSelectionElement(selection, mesh, EditMode), std::nullopt});
    }
}

void Scene::SelectElement(entt::entity mesh_entity, AnyHandle element, bool toggle) {
    const auto new_element = element ? element.Element : Element::None;
    R.patch<MeshSelection>(mesh_entity, [&](auto &selection) {
        if (!toggle || selection.Element != new_element) {
            selection = {.Element = new_element};
        }
        if (!element) return;

        const auto handle = *element;
        if (auto it = find(selection.Handles, handle); toggle && it != selection.Handles.end()) {
            selection.Handles.erase(it);
            if (selection.ActiveHandle == handle) selection.ActiveHandle = {};
        } else {
            selection.Handles.emplace_back(handle);
            selection.ActiveHandle = handle;
        }
    });
    UpdateMeshElementStateBuffers(mesh_entity);
}

void Scene::UpdateMeshElementStateBuffers(entt::entity e) {
    if (!R.all_of<Mesh>(e)) return;

    const auto &mesh = R.get<Mesh>(e);
    std::unordered_set<VH> selected_vertices;
    std::unordered_set<EH> selected_edges;
    std::unordered_set<FH> selected_faces;

    auto element{Element::None};
    std::vector<uint32_t> handles;
    std::optional<uint32_t> active_handle;
    if (SETTINGS.InteractionMode == InteractionMode::Excite) {
        element = Element::Vertex;
        for (auto [entity, mi, excitable] : R.view<const MeshInstance, const Excitable>().each()) {
            if (mi.MeshEntity != e) continue;
            handles = excitable.ExcitableVertices;
            selected_vertices.insert(excitable.ExcitableVertices.begin(), excitable.ExcitableVertices.end());
            if (const auto *excited_vertex = R.try_get<ExcitedVertex>(entity)) {
                active_handle = excited_vertex->Vertex;
            }
            break;
        }
    } else if (const auto &selection = R.get<MeshSelection>(e);
               SETTINGS.InteractionMode == InteractionMode::Edit && selection.Element == EditMode) {
        element = selection.Element;
        handles = selection.Handles;
        active_handle = selection.ActiveHandle;
        if (element == Element::Vertex) {
            selected_vertices.insert(selection.Handles.begin(), selection.Handles.end());
        } else if (element == Element::Edge) {
            selected_edges.insert(selection.Handles.begin(), selection.Handles.end());
        } else if (element == Element::Face) {
            for (auto h : selection.Handles) {
                selected_faces.emplace(h);
                for (const auto heh : mesh.fh_range(FH{h})) selected_edges.emplace(mesh.GetEdge(heh));
            }
        }
    }

    auto face_states = MakeElementStates(mesh.FaceCount());
    auto edge_states = MakeElementStates(mesh.EdgeCount() * 2);
    auto vertex_states = MakeElementStates(mesh.VertexCount());
    if (element == Element::Face) {
        for (const auto fh : selected_faces) face_states[*fh] |= MeshRender::ElementStateSelected;
        if (active_handle) face_states[*active_handle] |= MeshRender::ElementStateActive;
    }

    if (element == Element::Edge || element == Element::Face) {
        for (uint32_t ei = 0; ei < mesh.EdgeCount(); ++ei) {
            uint32_t state = 0;
            if (selected_edges.contains(EH{ei})) state |= MeshRender::ElementStateSelected;
            if (element == Element::Edge && active_handle == ei) state |= MeshRender::ElementStateActive;
            edge_states[2 * ei] = edge_states[2 * ei + 1] = state;
        }
    } else if (element == Element::Vertex) {
        for (uint32_t ei = 0; ei < mesh.EdgeCount(); ++ei) {
            const auto heh = mesh.GetHalfedge(EH{ei}, 0);
            const auto v_from = mesh.GetFromVertex(heh), v_to = mesh.GetToVertex(heh);
            auto get_state = [&](auto vh) {
                uint32_t state = 0;
                if (selected_vertices.contains(vh)) state |= MeshRender::ElementStateSelected;
                if (active_handle == *vh) state |= MeshRender::ElementStateActive;
                return state;
            };
            edge_states[2 * ei] = get_state(v_from);
            edge_states[2 * ei + 1] = get_state(v_to);
        }
    }

    if (element == Element::Vertex) {
        for (const auto vh : selected_vertices) vertex_states[*vh] |= MeshRender::ElementStateSelected;
        if (active_handle) vertex_states[*active_handle] |= MeshRender::ElementStateActive;
    }

    R.patch<MeshElementStateBuffers>(e, [&](auto &states) {
        states.Faces.Update(as_bytes(face_states));
        states.Edges.Update(as_bytes(edge_states));
        states.Vertices.Update(as_bytes(vertex_states));
    });

    NeedsRender = SelectionStale = true;
}

std::string Scene::DebugBufferHeapUsage() const { return Buffers->Ctx.DebugHeapUsage(); }

namespace {

// If `model_index` is set, only the model at that index is rendered. Otherwise, all models are rendered.
void Draw(
    vk::CommandBuffer cb, const ShaderPipeline &pipeline, uint index_count, const ModelsBuffer &models,
    DrawPushConstants pc, std::optional<uint> model_index = {}
) {
    pc.FirstInstance = model_index.value_or(0);
    const auto instance_count = model_index.has_value() ? 1 : models.Buffer.UsedSize / sizeof(WorldMatrix);
    cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPushConstants), &pc);
    cb.draw(index_count, instance_count, 0, 0);
}

void Draw(
    vk::CommandBuffer cb, const ShaderPipeline &pipeline, const SlottedBufferRange &indices, const ModelsBuffer &models,
    DrawPushConstants pc, std::optional<uint> model_index = {}
) {
    Draw(cb, pipeline, indices.Range.Count, models, pc, model_index);
}

DrawPushConstants MakeDrawPc(uint32_t vertex_slot, BufferRange vertices, const SlottedBufferRange &indices, uint32_t model_slot) {
    return {
        .VertexSlot = vertex_slot,
        .IndexSlot = indices.Slot,
        .IndexOffset = indices.Range.Offset,
        .ModelSlot = model_slot,
        .FirstInstance = 0,
        .ObjectIdSlot = InvalidSlot,
        .FaceNormalSlot = InvalidSlot,
        .FaceIdOffset = 0,
        .FaceNormalOffset = 0,
        .VertexCountOrHeadImageSlot = vertices.Count,
        .SelectionNodesSlot = 0,
        .SelectionCounterSlot = 0,
        .ElementIdOffset = 0,
        .ElementStateSlot = InvalidSlot,
        .VertexOffset = vertices.Offset,
        .Pad0 = 0,
        .LineColor = vec4{0, 0, 0, 0},
    };
}
DrawPushConstants MakeDrawPc(const SlottedBufferRange &vertices, const SlottedBufferRange &indices, const ModelsBuffer &mb) {
    return MakeDrawPc(vertices.Slot, vertices.Range, indices, mb.Buffer.Slot);
}
DrawPushConstants MakeDrawPc(const RenderBuffers &rb, uint32_t vertex_slot, const ModelsBuffer &mb) {
    return MakeDrawPc(vertex_slot, rb.Vertices, rb.Indices, mb.Buffer.Slot);
}
} // namespace

void Scene::RecordRenderCommandBuffer() {
    SelectionStale = true;
    const auto &cb = *RenderCommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(Extent.width), float(Extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, Extent});

    // In Edit mode, only primary edit instance per selected mesh gets Edit visuals.
    // Other selected instances render normally with silhouettes.
    const auto &settings = SETTINGS;
    const bool is_edit_mode = settings.InteractionMode == InteractionMode::Edit;
    const auto primary_edit_instances = is_edit_mode ? ComputePrimaryEditInstances(R) : std::unordered_map<entt::entity, entt::entity>{};
    std::unordered_set<entt::entity> silhouette_instances;
    if (is_edit_mode) {
        for (const auto [e, mi] : R.view<const MeshInstance, const Selected>().each()) {
            if (primary_edit_instances.at(mi.MeshEntity) != e) silhouette_instances.insert(e);
        }
    }

    auto render_silhouette_instances = [&](const PipelineRenderer &renderer, SPT spt) {
        const auto &pipeline = renderer.Bind(cb, spt);
        auto render = [&](entt::entity e) {
            const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
            auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
            auto &models = R.get<ModelsBuffer>(mesh_entity);
            auto pc = MakeDrawPc(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
            pc.ObjectIdSlot = models.ObjectIds.Slot;
            Draw(cb, pipeline, mesh_buffers.FaceIndices, models, pc, R.get<RenderInstance>(e).BufferIndex);
        };
        if (is_edit_mode) {
            for (const auto e : silhouette_instances) render(e);
        } else {
            for (const auto e : R.view<Selected>()) render(e);
        }
    };

    const bool render_silhouette = !R.view<Selected>().empty() &&
        (settings.InteractionMode == InteractionMode::Object || !silhouette_instances.empty());
    if (render_silhouette) { // Silhouette depth/object pass
        const auto &silhouette = Pipelines->Silhouette;
        static const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
        const vk::Rect2D rect{{0, 0}, ToExtent2D(silhouette.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette.Renderer.RenderPass, *silhouette.Resources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);
        render_silhouette_instances(silhouette.Renderer, SPT::SilhouetteDepthObject);
        cb.endRenderPass();

        const auto &silhouette_edge = Pipelines->SilhouetteEdge;
        static const std::vector<vk::ClearValue> edge_clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
        const vk::Rect2D edge_rect{{0, 0}, ToExtent2D(silhouette_edge.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette_edge.Renderer.RenderPass, *silhouette_edge.Resources->Framebuffer, edge_rect, edge_clear_values}, vk::SubpassContents::eInline);
        const auto &silhouette_edo = silhouette_edge.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepthObject);
        struct SilhouetteEdgeDepthObjectPushConstants {
            uint32_t SilhouetteEdgeWidth;
            uint32_t SilhouetteSamplerIndex;
        } edge_pc{settings.SilhouetteEdgeWidth, SelectionHandles->SilhouetteSampler};
        cb.pushConstants(*silhouette_edo.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(edge_pc), &edge_pc);
        silhouette_edo.RenderQuad(cb);
        cb.endRenderPass();
    }

    const auto &main = Pipelines->Main;
    // Main rendering pass
    {
        const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {settings.BackgroundColor}};
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
        const SPT fill_pipeline = settings.ColorMode == ColorMode::Mesh ? SPT::Fill : SPT::DebugNormals;
        const bool is_excite_mode = settings.InteractionMode == InteractionMode::Excite;
        const bool show_solid = settings.ViewportShading == ViewportShadingMode::Solid;
        const bool show_wireframe = settings.ViewportShading == ViewportShadingMode::Wireframe;

        std::unordered_set<entt::entity> excitable_mesh_entities;
        if (is_excite_mode) {
            for (const auto [e, mi, excitable] : R.view<const MeshInstance, const Excitable>().each())
                excitable_mesh_entities.emplace(mi.MeshEntity);
        }

        // Solid faces
        if (show_solid) {
            const auto &pipeline = main.Renderer.Bind(cb, fill_pipeline);
            for (auto [entity, mesh_buffers, models, mesh, state_buffers] :
                 R.view<MeshBuffers, ModelsBuffer, Mesh, MeshElementStateBuffers>().each()) {
                auto pc = MakeDrawPc(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
                const auto face_id_range = Meshes.GetFaceIdRange(mesh.GetStoreId());
                const auto face_normal_range = Meshes.GetFaceNormalRange(mesh.GetStoreId());
                pc.ObjectIdSlot = Meshes.GetFaceIdSlot();
                pc.FaceIdOffset = face_id_range.Offset;
                pc.FaceNormalSlot = settings.SmoothShading ? InvalidSlot : Meshes.GetFaceNormalSlot();
                pc.FaceNormalOffset = face_normal_range.Offset;
                if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                    // Draw primary with element state first, then all without (depth LESS won't overwrite)
                    pc.ElementStateSlot = state_buffers.Faces.Slot;
                    Draw(cb, pipeline, mesh_buffers.FaceIndices, models, pc, R.get<RenderInstance>(it->second).BufferIndex);
                    pc.ElementStateSlot = InvalidSlot;
                    Draw(cb, pipeline, mesh_buffers.FaceIndices, models, pc);
                } else {
                    pc.ElementStateSlot = state_buffers.Faces.Slot;
                    Draw(cb, pipeline, mesh_buffers.FaceIndices, models, pc);
                }
            }
        }

        // Wireframe edges
        if (show_wireframe || is_edit_mode || is_excite_mode) {
            const auto &pipeline = main.Renderer.Bind(cb, SPT::Line);
            for (auto [entity, mesh_buffers, models, state_buffers] :
                 R.view<MeshBuffers, ModelsBuffer, MeshElementStateBuffers>().each()) {
                auto pc = MakeDrawPc(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, models);
                pc.ElementStateSlot = state_buffers.Edges.Slot;
                if (show_wireframe) {
                    Draw(cb, pipeline, mesh_buffers.EdgeIndices, models, pc);
                } else if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                    Draw(cb, pipeline, mesh_buffers.EdgeIndices, models, pc, R.get<RenderInstance>(it->second).BufferIndex);
                } else if (excitable_mesh_entities.contains(entity)) {
                    Draw(cb, pipeline, mesh_buffers.EdgeIndices, models, pc);
                }
            }
        }

        // Vertex points
        if ((is_edit_mode && EditMode == Element::Vertex) || is_excite_mode) {
            const auto &pipeline = main.Renderer.Bind(cb, SPT::Point);
            for (auto [entity, mesh_buffers, models, state_buffers] :
                 R.view<MeshBuffers, ModelsBuffer, MeshElementStateBuffers>().each()) {
                auto pc = MakeDrawPc(mesh_buffers.Vertices, mesh_buffers.VertexIndices, models);
                pc.ElementStateSlot = state_buffers.Vertices.Slot;
                if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                    Draw(cb, pipeline, mesh_buffers.VertexIndices, models, pc, R.get<RenderInstance>(it->second).BufferIndex);
                } else if (excitable_mesh_entities.contains(entity)) {
                    Draw(cb, pipeline, mesh_buffers.VertexIndices, models, pc);
                }
            }
        }
    }

    // Silhouette edge color (rendered ontop of meshes)
    if (render_silhouette) {
        const auto &silhouette_edc = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeColor);
        // In Edit mode, never show active silhouette - only selected (non-active) silhouettes
        uint32_t active_object_id = 0;
        if (!is_edit_mode) {
            const auto active_entity = FindActiveEntity(R);
            active_object_id = active_entity != entt::null && R.all_of<Visible>(active_entity) ?
                R.get<RenderInstance>(active_entity).ObjectId :
                0;
        }
        struct SilhouetteEdgeColorPushConstants {
            uint32_t Manipulating;
            uint32_t ObjectSamplerIndex;
            uint32_t ActiveObjectId;
        } pc{TransformGizmo::IsUsing(), SelectionHandles->ObjectIdSampler, active_object_id};
        cb.pushConstants(*silhouette_edc.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        silhouette_edc.RenderQuad(cb);
    }

    { // Selection overlays
        const auto &pipeline = main.Renderer.Bind(cb, SPT::Line);
        const auto vertex_slot = Buffers->VertexBuffer.Buffer.Slot;
        for (auto [_, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
            for (auto &[element, buffers] : mesh_buffers.NormalIndicators) {
                auto pc = MakeDrawPc(buffers, vertex_slot, models);
                pc.LineColor = element == Element::Face ? MeshRender::FaceNormalIndicatorColor : MeshRender::VertexNormalIndicatorColor;
                Draw(cb, pipeline, buffers.Indices, models, pc);
            }
        }
        for (auto [_, bounding_boxes, models] : R.view<BoundingBoxesBuffers, ModelsBuffer>().each()) {
            auto pc = MakeDrawPc(bounding_boxes.Buffers, vertex_slot, models);
            pc.LineColor = MeshRender::EdgeColor;
            Draw(cb, pipeline, bounding_boxes.Buffers.Indices, models, pc);
        }
    }

    // Grid lines texture
    if (settings.ShowGrid) main.Renderer.ShaderPipelines.at(SPT::Grid).RenderQuad(cb);

    cb.endRenderPass();
    cb.end();
}

#ifdef MVK_FORCE_STAGED_TRANSFERS
void Scene::RecordTransferCommandBuffer() {
    const Timer timer{"RecordTransferCommandBuffer"};
    TransferCommandBuffer->reset({});
    TransferCommandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    if (auto deferred_copies = Buffers->Ctx.TakeDeferredCopies(); !deferred_copies.empty()) {
        for (const auto &[buffers, ranges] : deferred_copies) {
            auto regions = ranges | transform([](const auto &r) {
                               const auto &[start, end] = r;
                               return vk::BufferCopy{start, start, end - start};
                           }) |
                to<std::vector>();
            TransferCommandBuffer->copyBuffer(buffers.Src, buffers.Dst, regions);
        }
        // Ensure buffer writes (staging copies) are visible to shader reads.
        const vk::MemoryBarrier buffer_barrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead};
        TransferCommandBuffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader,
            {}, buffer_barrier, {}, {}
        );
    }
    TransferCommandBuffer->end();
}
#endif

void Scene::UpdateEdgeColors() {
    MeshRender::EdgeColor = SETTINGS.ViewportShading == ViewportShadingMode::Solid ? MeshEdgeColor : EdgeColor;
    UpdateSceneUBO();
}

void Scene::UpdateSceneUBO() {
    const float aspect_ratio = Extent.width == 0 || Extent.height == 0 ? 1.f : float(Extent.width) / float(Extent.height);
    Buffers->SceneUBO.Update(as_bytes(SceneUBO{
        .View = Camera.View(),
        .Proj = Camera.Projection(aspect_ratio),
        .CameraPositionNear = vec4{Camera.Position(), Camera.NearClip},
        .ViewColorAndAmbient = Lights.ViewColorAndAmbient,
        .DirectionalColorAndIntensity = Lights.DirectionalColorAndIntensity,
        .LightDirectionFar = vec4{Lights.Direction, Camera.FarClip},
        .SilhouetteActive = Colors.Active,
        .SilhouetteSelected = Colors.Selected,
        .BaseColor = Mesh::DefaultMeshColor,
        .EdgeColor = MeshRender::EdgeColor,
        .VertexUnselectedColor = MeshRender::UnselectedVertexEditColor,
        .SelectedColor = MeshRender::SelectedColor,
        .ActiveColor = MeshRender::ActiveColor,
    }));
    NeedsRender = true;
}

void Scene::UpdateEntitySelectionOverlays(entt::entity mesh_entity) {
    const auto &mesh = R.get<const Mesh>(mesh_entity);
    R.patch<MeshBuffers>(mesh_entity, [&](auto &mesh_buffers) {
        for (const auto element : NormalElements) {
            if (SETTINGS.ShownNormalElements.contains(element)) {
                if (!mesh_buffers.NormalIndicators.contains(element)) {
                    const auto index_kind = element == Element::Face ? IndexKind::Face : IndexKind::Vertex;
                    mesh_buffers.NormalIndicators.emplace(
                        element,
                        Buffers->CreateRenderBuffers(MeshRender::CreateNormalVertices(mesh, element), MeshRender::CreateNormalIndices(mesh, element), index_kind)
                    );
                } else {
                    Buffers->VertexBuffer.Update(mesh_buffers.NormalIndicators.at(element).Vertices, MeshRender::CreateNormalVertices(mesh, element));
                }
            } else if (mesh_buffers.NormalIndicators.contains(element)) {
                Buffers->Release(mesh_buffers.NormalIndicators.at(element));
                mesh_buffers.NormalIndicators.erase(element);
            }
        }
    });

    if (SETTINGS.ShowBoundingBoxes) {
        if (!R.all_of<BoundingBoxesBuffers>(mesh_entity)) {
            R.emplace<BoundingBoxesBuffers>(mesh_entity, Buffers->CreateRenderBuffers(CreateBoxVertices(MeshRender::ComputeBoundingBox(mesh)), BBox::EdgeIndices, IndexKind::Edge));
        } else {
            Buffers->VertexBuffer.Update(R.get<BoundingBoxesBuffers>(mesh_entity).Buffers.Vertices, CreateBoxVertices(MeshRender::ComputeBoundingBox(mesh)));
        }
    } else if (R.all_of<BoundingBoxesBuffers>(mesh_entity)) {
        Buffers->Release(R.get<BoundingBoxesBuffers>(mesh_entity).Buffers);
        R.remove<BoundingBoxesBuffers>(mesh_entity);
    }
}

using namespace ImGui;

namespace {
constexpr vec2 ToGlm(ImVec2 v) { return std::bit_cast<vec2>(v); }
constexpr vk::Extent2D ToExtent(vec2 e) { return {uint(e.x), uint(e.y)}; }

std::optional<std::pair<glm::uvec2, glm::uvec2>> ComputeBoxSelectPixels(vec2 start, vec2 end, vec2 window_pos, vk::Extent2D extent, float drag_threshold) {
    if (glm::distance(start, end) <= drag_threshold) return {};

    const vec2 extent_size{float(extent.width), float(extent.height)};
    const auto box_min = glm::min(start, end) - window_pos;
    const auto box_max = glm::max(start, end) - window_pos;
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
    return std::pair{box_min_px, box_max_px};
}

constexpr std::string Capitalize(std::string_view str) {
    if (str.empty()) return {};

    std::string result{str};
    char &c = result[0];
    if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
    return result;
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

} // namespace

void Scene::RenderSelectionPass(vk::Semaphore signal_semaphore) {
    const Timer timer{"RenderSelectionPass"};

    const auto primary_edit_instances = SETTINGS.InteractionMode == InteractionMode::Edit ?
        ComputePrimaryEditInstances(R) :
        std::unordered_map<entt::entity, entt::entity>{};

    // Object selection never uses depth testing - we want all visible pixels regardless of occlusion
    RenderSelectionPassWith(false, [&](vk::CommandBuffer cb, const PipelineRenderer &renderer) {
        const auto &pipeline = renderer.Bind(cb, SPT::SelectionFragmentXRay);
        for (auto [mesh_entity, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
            const uint32_t instance_count = models.Buffer.UsedSize / sizeof(WorldMatrix);
            if (instance_count == 0) continue;

            auto pc = MakeDrawPc(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
            pc.ObjectIdSlot = models.ObjectIds.Slot;
            pc.VertexCountOrHeadImageSlot = SelectionHandles->HeadImage;
            pc.SelectionNodesSlot = Buffers->SelectionNodeBuffer.Slot;
            pc.SelectionCounterSlot = SelectionHandles->SelectionCounter;
            if (auto it = primary_edit_instances.find(mesh_entity); it != primary_edit_instances.end()) {
                Draw(cb, pipeline, mesh_buffers.FaceIndices, models, pc, R.get<RenderInstance>(it->second).BufferIndex);
            } else { // todo can we guarantee only selected instances are rendered here and thus we can drop this check?
                Draw(cb, pipeline, mesh_buffers.FaceIndices, models, pc);
            }
        } }, signal_semaphore);

    SelectionStale = false;
}

void Scene::RenderSilhouetteDepth(vk::CommandBuffer cb) {
    // Render selected meshes to silhouette depth buffer for element occlusion
    const auto &silhouette = Pipelines->Silhouette;
    static const std::vector<vk::ClearValue> clear_values{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
    const vk::Rect2D rect{{0, 0}, ToExtent2D(silhouette.Resources->OffscreenImage.Extent)};
    cb.beginRenderPass({*silhouette.Renderer.RenderPass, *silhouette.Resources->Framebuffer, rect, clear_values}, vk::SubpassContents::eInline);

    const auto &pipeline = silhouette.Renderer.Bind(cb, SPT::SilhouetteDepthObject);
    for (const auto e : R.view<Selected>()) {
        const auto mesh_entity = R.get<MeshInstance>(e).MeshEntity;
        auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
        auto &models = R.get<ModelsBuffer>(mesh_entity);
        if (const auto model_index = GetModelBufferIndex(R, e)) {
            auto pc = MakeDrawPc(mesh_buffers.Vertices, mesh_buffers.FaceIndices, models);
            pc.ObjectIdSlot = models.ObjectIds.Slot;
            Draw(cb, pipeline, mesh_buffers.FaceIndices, models, pc, *model_index);
        }
    }
    cb.endRenderPass();
}

void Scene::RenderSelectionPassWith(bool render_depth, const std::function<void(vk::CommandBuffer, const PipelineRenderer &)> &draw_fn, vk::Semaphore signal_semaphore) {
    const Timer timer{"RenderSelectionPassWith"};
    auto cb = *ClickCommandBuffer;
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Reset selection counter.
    Buffers->SelectionCounterBuffer.Write(as_bytes(SelectionCounters{}));

    // Transition head image to general layout and clear.
    const auto &head_image = Pipelines->SelectionFragment.Resources->HeadImage;
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
        vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {}, {}, *head_image.Image, ColorSubresourceRange}
    );
    cb.clearColorImage(*head_image.Image, vk::ImageLayout::eGeneral, vk::ClearColorValue{std::array<uint32_t, 4>{InvalidSlot, 0, 0, 0}}, ColorSubresourceRange);
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
        vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, {}, {}, *head_image.Image, ColorSubresourceRange}
    );

    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(Extent.width), float(Extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, Extent});

    if (render_depth) RenderSilhouetteDepth(cb);

    const auto &selection = Pipelines->SelectionFragment;
    const vk::Rect2D rect{{0, 0}, ToExtent2D(Pipelines->Silhouette.Resources->DepthImage.Extent)};
    cb.beginRenderPass({*selection.Renderer.RenderPass, *selection.Resources->Framebuffer, rect, {}}, vk::SubpassContents::eInline);
    draw_fn(cb, selection.Renderer);
    cb.endRenderPass();

    cb.end();
    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    if (signal_semaphore) submit.setSignalSemaphores(signal_semaphore);
    Vk.Queue.submit(submit, *OneShotFence);
    WaitFor(*OneShotFence, Vk.Device);
}

namespace {
void RunSelectionCompute(
    vk::CommandBuffer cb, vk::Queue queue, vk::Fence fence, vk::Device device,
    const auto &compute, const auto &pc, auto &&dispatch, vk::Semaphore wait_semaphore = {}
) {
    const Timer timer{"RunSelectionCompute"};
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, *compute.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *compute.PipelineLayout, 0, compute.GetDescriptorSet(), {});
    cb.pushConstants(*compute.PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
    dispatch(cb);

    const vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead};
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, barrier, {}, {});
    cb.end();

    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    static constexpr vk::PipelineStageFlags wait_stage{vk::PipelineStageFlagBits::eComputeShader};
    if (wait_semaphore) {
        submit.setWaitSemaphores(wait_semaphore);
        submit.setWaitDstStageMask(wait_stage);
    }
    queue.submit(submit, fence);
    WaitFor(fence, device);
}

uint32_t GetElementCount(const Mesh &mesh, Element element) {
    if (element == Element::Vertex) return mesh.VertexCount();
    if (element == Element::Edge) return mesh.EdgeCount();
    if (element == Element::Face) return mesh.FaceCount();
    return 0;
}

SPT SelectionPipelineForElement(Element element, bool xray) {
    if (element == Element::Vertex) return xray ? SPT::SelectionElementVertexXRay : SPT::SelectionElementVertex;
    if (element == Element::Edge) return xray ? SPT::SelectionElementEdgeXRay : SPT::SelectionElementEdge;
    return xray ? SPT::SelectionElementFaceXRay : SPT::SelectionElementFace;
}

// After rendering elements to selection buffer, dispatch compute shader to find the nearest element to mouse_px.
// Returns 0-based element index, or nullopt if no element found.
std::optional<uint32_t> FindNearestSelectionElement(
    const SceneBuffers &buffers, const ComputePipeline &compute, vk::CommandBuffer cb,
    vk::Queue queue, vk::Fence fence, vk::Device device,
    uint32_t head_image_index, uint32_t selection_nodes_slot, uint32_t click_result_slot,
    glm::uvec2 mouse_px, uint32_t max_element_id, Element element,
    vk::Semaphore wait_semaphore
) {
    const uint32_t radius = element == Element::Face ? 0u : ClickSelectRadiusPx;
    const uint32_t group_count = element == Element::Face ? 1u : SceneBuffers::ClickSelectElementGroupCount;
    RunSelectionCompute(
        cb, queue, fence, device, compute,
        ClickSelectElementPushConstants{
            .TargetPx = mouse_px,
            .Radius = radius,
            .HeadImageIndex = head_image_index,
            .SelectionNodesIndex = selection_nodes_slot,
            .ClickResultIndex = click_result_slot,
        },
        [group_count](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(group_count, 1, 1); },
        wait_semaphore
    );

    const auto *candidates = reinterpret_cast<const ClickElementCandidate *>(buffers.ClickElementResultBuffer.GetData().data());
    ClickElementCandidate best{.ObjectId = 0, .Depth = 1.0f, .DistanceSq = std::numeric_limits<uint32_t>::max(), .Padding = 0};
    for (uint32_t i = 0; i < group_count; ++i) {
        const auto &candidate = candidates[i];
        if (candidate.ObjectId == 0) continue;
        if (candidate.DistanceSq < best.DistanceSq || (candidate.DistanceSq == best.DistanceSq && candidate.Depth < best.Depth)) {
            best = candidate;
        }
    }

    if (best.ObjectId == 0 || best.ObjectId > max_element_id) return {};
    return best.ObjectId - 1;
}
} // namespace

void Scene::RenderEditSelectionPass(std::span<const ElementRange> ranges, Element element, vk::Semaphore signal_semaphore) {
    if (ranges.empty() || element == Element::None) return;

    const auto primary_edit_instances = ComputePrimaryEditInstances(R);
    const Timer timer{"RenderEditSelectionPass"};
    const bool xray_selection = SelectionXRay || SETTINGS.ViewportShading == ViewportShadingMode::Wireframe;
    RenderSelectionPassWith(!xray_selection, [&](vk::CommandBuffer cb, const PipelineRenderer &renderer) {
        const auto &pipeline = renderer.Bind(cb, SelectionPipelineForElement(element, xray_selection));
        for (const auto &r : ranges) {
            auto &mesh_buffers = R.get<MeshBuffers>(r.MeshEntity);
            auto &models = R.get<ModelsBuffer>(r.MeshEntity);
            const auto &mesh = R.get<Mesh>(r.MeshEntity);
            const auto &indices = element == Element::Vertex ? mesh_buffers.VertexIndices :
                element == Element::Edge                     ? mesh_buffers.EdgeIndices :
                                                               mesh_buffers.FaceIndices;
            auto pc = MakeDrawPc(mesh_buffers.Vertices, indices, models);
            pc.ObjectIdSlot = element == Element::Face ? Meshes.GetFaceIdSlot() : InvalidSlot;
            pc.FaceIdOffset = element == Element::Face ? Meshes.GetFaceIdRange(mesh.GetStoreId()).Offset : 0;
            pc.VertexCountOrHeadImageSlot = SelectionHandles->HeadImage;
            pc.SelectionNodesSlot = Buffers->SelectionNodeBuffer.Slot;
            pc.SelectionCounterSlot = SelectionHandles->SelectionCounter;
            pc.ElementIdOffset = r.Offset;
            if (auto it = primary_edit_instances.find(r.MeshEntity); it != primary_edit_instances.end()) {
                Draw(cb, pipeline, indices, models, pc, R.get<RenderInstance>(it->second).BufferIndex);
            } else { // todo can we guarantee only selected instances are rendered here and thus we can drop this check?
                Draw(cb, pipeline, indices, models, pc);
            }
        } }, signal_semaphore);

    // Edit selection pass overwrites the shared head image used for object selection.
    SelectionStale = true;
}

std::vector<std::vector<uint32_t>> Scene::RunBoxSelectElements(std::span<const ElementRange> ranges, Element element, std::pair<glm::uvec2, glm::uvec2> box_px) {
    if (ranges.empty() || element == Element::None) return {};

    std::vector<std::vector<uint32_t>> results(ranges.size());
    const auto [box_min, box_max] = box_px;
    if (box_min.x >= box_max.x || box_min.y >= box_max.y) return results;

    const Timer timer{"RunBoxSelectElements"};
    const auto element_count = fold_left(
        ranges, uint32_t{0},
        [](uint32_t total, const auto &range) { return std::max(total, range.Offset + range.Count); }
    );
    if (element_count == 0) return results;

    const uint32_t bitset_words = (element_count + 31) / 32;
    if (bitset_words > BoxSelectZeroBits.size()) return results;

    RenderEditSelectionPass(ranges, element, *SelectionReadySemaphore);

    const std::span<const uint32_t> zero_bits{BoxSelectZeroBits.data(), bitset_words};
    Buffers->BoxSelectBitsetBuffer.Write(std::as_bytes(zero_bits));

    auto cb = *ClickCommandBuffer;
    const uint32_t group_count_x = (box_max.x - box_min.x + 15) / 16;
    const uint32_t group_count_y = (box_max.y - box_min.y + 15) / 16;
    RunSelectionCompute(
        cb, Vk.Queue, *OneShotFence, Vk.Device, Pipelines->BoxSelect,
        BoxSelectPushConstants{
            .BoxMin = box_min,
            .BoxMax = box_max,
            .ObjectCount = element_count,
            .HeadImageIndex = SelectionHandles->HeadImage,
            .SelectionNodesIndex = Buffers->SelectionNodeBuffer.Slot,
            .BoxResultIndex = SelectionHandles->BoxResult,
        },
        [group_count_x, group_count_y](vk::CommandBuffer dispatch_cb) {
            dispatch_cb.dispatch(group_count_x, group_count_y, 1);
        },
        *SelectionReadySemaphore
    );

    const auto *bits = reinterpret_cast<const uint32_t *>(Buffers->BoxSelectBitsetBuffer.GetData().data());
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto &range = ranges[i];
        results[i] = iota(range.Offset, range.Offset + range.Count) //
            | filter([&](uint32_t idx) {
                         const uint32_t mask = 1u << (idx % 32);
                         return (bits[idx / 32] & mask) != 0;
                     }) //
            | transform([offset = range.Offset](uint32_t idx) { return idx - offset; }) //
            | to<std::vector>();
    }
    return results;
}

std::optional<AnyHandle> Scene::RunClickSelectElement(entt::entity mesh_entity, Element element, glm::uvec2 mouse_px) {
    const auto &mesh = R.get<Mesh>(mesh_entity);
    const uint32_t element_count = GetElementCount(mesh, element);
    if (element_count == 0 || element == Element::None) return {};

    const Timer timer{"RunClickSelectElement"};
    const ElementRange range{mesh_entity, 0, element_count};
    RenderEditSelectionPass(std::span{&range, 1}, element, *SelectionReadySemaphore);
    if (const auto index = FindNearestSelectionElement(
            *Buffers, Pipelines->ClickSelectElement, *ClickCommandBuffer,
            Vk.Queue, *OneShotFence, Vk.Device,
            SelectionHandles->HeadImage, Buffers->SelectionNodeBuffer.Slot, SelectionHandles->ClickElementResult,
            mouse_px, element_count, element,
            *SelectionReadySemaphore
        )) {
        return AnyHandle{element, *index};
    }
    return {};
}

std::optional<uint32_t> Scene::RunClickSelectExcitableVertex(entt::entity instance_entity, glm::uvec2 mouse_px) {
    if (!R.all_of<Excitable>(instance_entity)) return {};

    const Timer timer{"RunClickSelectExcitableVertex"};
    const auto mesh_entity = R.get<MeshInstance>(instance_entity).MeshEntity;
    const auto &mesh = R.get<Mesh>(mesh_entity);
    const uint32_t vertex_count = mesh.VertexCount();
    if (vertex_count == 0) return {};

    auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
    auto &models = R.get<ModelsBuffer>(mesh_entity);
    auto &state_buffers = R.get<MeshElementStateBuffers>(mesh_entity);
    const auto model_index = R.get<RenderInstance>(instance_entity).BufferIndex;
    RenderSelectionPassWith(true, [&](vk::CommandBuffer cb, const PipelineRenderer &renderer) {
        const auto &pipeline = renderer.Bind(cb, SPT::SelectionElementVertex);
        auto pc = MakeDrawPc(mesh_buffers.Vertices, mesh_buffers.VertexIndices, models);
        pc.VertexCountOrHeadImageSlot = SelectionHandles->HeadImage;
        pc.SelectionNodesSlot = Buffers->SelectionNodeBuffer.Slot;
        pc.SelectionCounterSlot = SelectionHandles->SelectionCounter;
        pc.ElementStateSlot = state_buffers.Vertices.Slot;
        Draw(cb, pipeline, mesh_buffers.VertexIndices, models, pc, model_index); }, *SelectionReadySemaphore);
    SelectionStale = true;

    return FindNearestSelectionElement(
        *Buffers, Pipelines->ClickSelectElement, *ClickCommandBuffer,
        Vk.Queue, *OneShotFence, Vk.Device,
        SelectionHandles->HeadImage, Buffers->SelectionNodeBuffer.Slot, SelectionHandles->ClickElementResult,
        mouse_px, vertex_count, Element::Vertex,
        *SelectionReadySemaphore
    );
}

// Returns entities hit at mouse_px, sorted by depth (near-to-far), with duplicates removed.
std::vector<entt::entity> Scene::RunClickSelect(glm::uvec2 mouse_px) {
    const bool selection_rendered = SelectionStale;
    if (selection_rendered) RenderSelectionPass(*SelectionReadySemaphore);

    const Timer timer{"RunClickSelect"};
    Buffers->ClickResultBuffer.Write(as_bytes(ClickResult{}));
    auto cb = *ClickCommandBuffer;
    const auto &compute = Pipelines->ClickSelect;
    RunSelectionCompute(
        cb, Vk.Queue, *OneShotFence, Vk.Device, compute,
        ClickSelectPushConstants{
            .TargetPx = mouse_px,
            .HeadImageIndex = SelectionHandles->HeadImage,
            .SelectionNodesIndex = Buffers->SelectionNodeBuffer.Slot,
            .ClickResultIndex = SelectionHandles->ClickResult,
        },
        [](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(1, 1, 1); },
        selection_rendered ? *SelectionReadySemaphore : vk::Semaphore{}
    );

    // Convert click hits to entities.
    const auto &result = Buffers->GetClickResult();
    const auto visible_entities = R.view<Visible>() | to<std::vector>();
    auto hits = result.Hits //
        | take(std::min<uint32_t>(result.Count, result.Hits.size())) //
        | filter([&](const auto &hit) { return hit.ObjectId != 0 && hit.ObjectId <= visible_entities.size(); }) //
        | transform([](const auto &hit) { return std::pair{hit.Depth, hit.ObjectId}; }) //
        | to<std::vector>();
    std::ranges::sort(hits);

    std::vector<entt::entity> entities;
    entities.reserve(hits.size());
    std::unordered_set<uint32_t> seen_object_ids;
    for (const auto &[_, object_id] : hits) {
        if (seen_object_ids.insert(object_id).second) {
            entities.emplace_back(visible_entities[object_id - 1]);
        }
    }
    return entities;
}

std::vector<entt::entity> Scene::RunBoxSelect(std::pair<glm::uvec2, glm::uvec2> box_px) {
    const auto [box_min, box_max] = box_px;
    if (box_min.x >= box_max.x || box_min.y >= box_max.y) return {};

    const auto visible_entities = R.view<Visible>() | to<std::vector>();
    const auto object_count = static_cast<uint32_t>(visible_entities.size());
    if (object_count == 0) return {};

    const uint32_t bitset_words = (object_count + 31) / 32;
    if (bitset_words > BoxSelectZeroBits.size()) return {};

    const Timer timer{"RunBoxSelect"};
    const bool selection_rendered = SelectionStale;
    if (selection_rendered) RenderSelectionPass(*SelectionReadySemaphore);

    const std::span<const uint32_t> zero_bits{BoxSelectZeroBits.data(), bitset_words};
    Buffers->BoxSelectBitsetBuffer.Write(std::as_bytes(zero_bits));

    auto cb = *ClickCommandBuffer;
    const auto &compute = Pipelines->BoxSelect;
    const uint32_t group_count_x = (box_max.x - box_min.x + 15) / 16;
    const uint32_t group_count_y = (box_max.y - box_min.y + 15) / 16;
    RunSelectionCompute(
        cb, Vk.Queue, *OneShotFence, Vk.Device, compute,
        BoxSelectPushConstants{
            .BoxMin = box_min,
            .BoxMax = box_max,
            .ObjectCount = object_count,
            .HeadImageIndex = SelectionHandles->HeadImage,
            .SelectionNodesIndex = Buffers->SelectionNodeBuffer.Slot,
            .BoxResultIndex = SelectionHandles->BoxResult,
        },
        [group_count_x, group_count_y](auto dispatch_cb) { dispatch_cb.dispatch(group_count_x, group_count_y, 1); },
        selection_rendered ? *SelectionReadySemaphore : vk::Semaphore{}
    );

    const auto *bits = reinterpret_cast<const uint32_t *>(Buffers->BoxSelectBitsetBuffer.GetData().data());
    return iota(uint32_t{0}, object_count) | filter([&](uint32_t i) {
               const uint32_t mask = 1u << (i % 32);
               return (bits[i / 32] & mask) != 0;
           }) |
        transform([&](uint32_t i) { return visible_entities[i]; }) | to<std::vector>();
}

void Scene::Interact() {
    if (Extent.width == 0 || Extent.height == 0) return;

    const auto active_entity = FindActiveEntity(R);
    // Handle keyboard input.
    if (IsWindowFocused()) {
        if (IsKeyPressed(ImGuiKey_Tab)) {
            // Cycle to the next interaction mode, wrapping around to the first.
            auto it = find(InteractionModes, SETTINGS.InteractionMode);
            SetInteractionMode(++it != InteractionModes.end() ? *it : *InteractionModes.begin());
        }
        if (SETTINGS.InteractionMode == InteractionMode::Edit) {
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
        R.clear<ExcitedVertex>();
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

    if (SelectionMode == SelectionMode::Box && SETTINGS.InteractionMode == InteractionMode::Edit) {
        const auto mouse_pos = ToGlm(GetMousePos());
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            BoxSelectStart = mouse_pos;
            BoxSelectEnd = mouse_pos;
        } else if (IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart.has_value()) {
            BoxSelectEnd = mouse_pos;
            static constexpr float drag_threshold{2};
            if (const auto box_px = ComputeBoxSelectPixels(*BoxSelectStart, *BoxSelectEnd, ToGlm(GetCursorScreenPos()), Extent, drag_threshold)) {
                Timer timer{"BoxSelectElements (all)"};
                // todo don't double-update MeshSelection
                for (const auto e : R.view<MeshSelection>()) {
                    R.patch<MeshSelection>(e, [](auto &s) { s.Handles.clear(); s.ActiveHandle = {}; });
                }

                std::unordered_set<entt::entity> mesh_entities; // Meshs of selected instances
                for (const auto [e, mi] : R.view<const MeshInstance, const Selected>().each()) {
                    mesh_entities.insert(mi.MeshEntity);
                }
                std::vector<ElementRange> ranges;
                ranges.reserve(mesh_entities.size());
                uint32_t offset = 0;
                for (const auto mesh_entity : mesh_entities) {
                    if (const uint32_t count = GetElementCount(R.get<Mesh>(mesh_entity), EditMode); count > 0) {
                        ranges.emplace_back(mesh_entity, offset, count);
                        offset += count;
                    }
                }

                auto results = RunBoxSelectElements(ranges, EditMode, *box_px);
                for (size_t i = 0; i < ranges.size(); ++i) {
                    const auto e = ranges[i].MeshEntity;
                    R.replace<MeshSelection>(e, MeshSelection{EditMode, i < results.size() ? std::move(results[i]) : std::vector<uint32_t>{}});
                }
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart.has_value()) {
            BoxSelectStart.reset();
            BoxSelectEnd.reset();
        }
        if (BoxSelectStart.has_value()) return;
    }

    if (SelectionMode == SelectionMode::Box && SETTINGS.InteractionMode == InteractionMode::Object) {
        const auto mouse_pos = ToGlm(GetMousePos());
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            BoxSelectStart = mouse_pos;
            BoxSelectEnd = mouse_pos;
        } else if (IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart.has_value()) {
            BoxSelectEnd = mouse_pos;
            static constexpr float drag_threshold{2};
            if (const auto box_px = ComputeBoxSelectPixels(*BoxSelectStart, *BoxSelectEnd, ToGlm(GetCursorScreenPos()), Extent, drag_threshold)) {
                const auto selected_entities = RunBoxSelect(*box_px);
                R.clear<Selected>();
                for (const auto e : selected_entities) R.emplace<Selected>(e);
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart.has_value()) {
            BoxSelectStart.reset();
            BoxSelectEnd.reset();
        }
        if (BoxSelectStart.has_value()) return;
    }

    if (!IsSingleClicked(ImGuiMouseButton_Left)) return;

    // Handle mouse selection.
    const auto mouse_pos_rel = GetMousePos() - GetCursorScreenPos();
    // Flip y-coordinate: ImGui uses top-left origin, but Vulkan gl_FragCoord uses bottom-left origin
    const glm::uvec2 mouse_px{uint32_t(mouse_pos_rel.x), uint32_t(Extent.height - mouse_pos_rel.y)};
    if (SETTINGS.InteractionMode == InteractionMode::Edit) {
        if (EditMode != Element::None) {
            const auto hit_entities = RunClickSelect(mouse_px);
            const auto hit_it = find_if(hit_entities, [&](auto e) { return R.all_of<Selected>(e); });
            const bool toggle = IsKeyDown(ImGuiMod_Shift) || IsKeyDown(ImGuiMod_Ctrl) || IsKeyDown(ImGuiMod_Super);
            if (!toggle) {
                for (const auto [e, selection] : R.view<MeshSelection>().each()) {
                    if (!selection.Handles.empty() || selection.Element != Element::None) {
                        R.replace<MeshSelection>(e, MeshSelection{});
                    }
                }
            }
            if (hit_it != hit_entities.end()) {
                const auto mesh_entity = R.get<MeshInstance>(*hit_it).MeshEntity;
                if (const auto element = RunClickSelectElement(mesh_entity, EditMode, mouse_px)) {
                    SelectElement(mesh_entity, *element, toggle);
                }
            }
        }
    } else if (SETTINGS.InteractionMode == InteractionMode::Object) {
        const auto hit_entities = RunClickSelect(mouse_px);
        // Cycle through hit entities.
        entt::entity intersected = entt::null;
        if (!hit_entities.empty()) {
            auto it = find(hit_entities, active_entity);
            if (it != hit_entities.end()) ++it;
            if (it == hit_entities.end()) it = hit_entities.begin();
            intersected = *it;
        }
        if (intersected != entt::null && IsKeyDown(ImGuiMod_Shift)) {
            if (active_entity == intersected) {
                ToggleSelected(intersected);
            } else {
                R.clear<Active>();
                R.emplace<Active>(intersected);
                R.emplace_or_replace<Selected>(intersected);
            }
        } else if (intersected != entt::null || !IsKeyDown(ImGuiMod_Shift)) {
            Select(intersected);
        }
    } else if (SETTINGS.InteractionMode == InteractionMode::Excite) {
        // Excite the nearest excitable vertex using screen-space selection.
        if (const auto hit_entities = RunClickSelect(mouse_px); !hit_entities.empty()) {
            if (const auto hit_entity = hit_entities.front(); R.all_of<Excitable>(hit_entity)) {
                if (const auto vertex = RunClickSelectExcitableVertex(hit_entity, mouse_px)) {
                    R.remove<ExcitedVertex>(hit_entity);
                    R.emplace<ExcitedVertex>(hit_entity, *vertex, 1.f);
                }
            }
        }
    }
}

void ScenePipelines::SetExtent(vk::Extent2D extent) {
    Main.SetExtent(extent, Device, PhysicalDevice, Samples);
    Silhouette.SetExtent(extent, Device, PhysicalDevice);
    SilhouetteEdge.SetExtent(extent, Device, PhysicalDevice);
    SelectionFragment.SetExtent(extent, Device, PhysicalDevice, *Silhouette.Resources->DepthImage.View);
};

bool Scene::RenderViewport() {
    ProcessComponentEvents();

    if (auto descriptor_updates = Buffers->Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        const Timer timer{"RenderViewport->UpdateBufferDescriptorSets"};
        Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
        Buffers->Ctx.ClearDeferredDescriptorUpdates();
    }
    const auto content_region = ToGlm(GetContentRegionAvail());
    const bool extent_changed = Extent.width != content_region.x || Extent.height != content_region.y;
    if (!extent_changed && !CommandBufferDirty && !NeedsRender) return false;

    const Timer timer{"RenderViewport"};
    if (extent_changed) {
        Extent = ToExtent(content_region);
        UpdateSceneUBO();
        Vk.Device.waitIdle(); // Ensure GPU work is done before destroying old pipeline resources
        Pipelines->SetExtent(Extent);
        Buffers->ResizeSelectionNodeBuffer(Extent);
        {
            const Timer timer{"RenderViewport->UpdateSelectionDescriptorSets"};
            const auto head_image_info = vk::DescriptorImageInfo{
                nullptr,
                *Pipelines->SelectionFragment.Resources->HeadImage.View,
                vk::ImageLayout::eGeneral
            };
            const vk::DescriptorBufferInfo selection_counter{*Buffers->SelectionCounterBuffer, 0, sizeof(SelectionCounters)};
            const vk::DescriptorBufferInfo click_result{*Buffers->ClickResultBuffer, 0, sizeof(ClickResult)};
            const vk::DescriptorBufferInfo click_element_result{*Buffers->ClickElementResultBuffer, 0, SceneBuffers::ClickSelectElementGroupCount * sizeof(ClickElementCandidate)};
            const auto &sil = Pipelines->Silhouette;
            const auto &sil_edge = Pipelines->SilhouetteEdge;
            const vk::DescriptorImageInfo silhouette_sampler{*sil.Resources->ImageSampler, *sil.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo object_id_sampler{*sil_edge.Resources->ImageSampler, *sil_edge.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo depth_sampler{*sil_edge.Resources->DepthSampler, *sil_edge.Resources->DepthImage.View, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
            const auto box_result = Buffers->GetBoxSelectBitsetDescriptor();
            Vk.Device.updateDescriptorSets(
                {
                    Slots->MakeImageWrite(SelectionHandles->HeadImage, head_image_info),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->SelectionCounter}, selection_counter),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->ClickResult}, click_result),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->ClickElementResult}, click_element_result),
                    Slots->MakeBufferWrite({SlotType::Buffer, SelectionHandles->BoxResult}, box_result),
                    Slots->MakeSamplerWrite(SelectionHandles->ObjectIdSampler, object_id_sampler),
                    Slots->MakeSamplerWrite(SelectionHandles->DepthSampler, depth_sampler),
                    Slots->MakeSamplerWrite(SelectionHandles->SilhouetteSampler, silhouette_sampler),
                },
                {}
            );
        }
        if (auto descriptor_updates = Buffers->Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
            Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
            Buffers->Ctx.ClearDeferredDescriptorUpdates();
        }
        CommandBufferDirty = true;
    }

#ifdef MVK_FORCE_STAGED_TRANSFERS
    RecordTransferCommandBuffer();
#endif

    if (CommandBufferDirty) {
        RecordRenderCommandBuffer();
        CommandBufferDirty = false;
    }

    vk::SubmitInfo submit;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    const std::array command_buffers{*TransferCommandBuffer, *RenderCommandBuffer};
    submit.setCommandBuffers(command_buffers);
#else
    submit.setCommandBuffers(*RenderCommandBuffer);
#endif
    Vk.Queue.submit(submit, *RenderFence);
    {
        const Timer timer{"RenderViewport->WaitForGPU"};
        WaitFor(*RenderFence, Vk.Device);
    }

    Buffers->Ctx.ReclaimRetiredBuffers();

    NeedsRender = false;
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

    if (!R.storage<Selected>().empty()) { // Draw center-dot for active/selected entities
        const auto size = ToGlm(GetContentRegionAvail());
        const auto vp = Camera.Projection(size.x / size.y) * Camera.View();
        for (const auto [e, wm] : R.view<const WorldMatrix, const Visible>().each()) {
            if (!R.any_of<Active, Selected>(e)) continue;

            const auto p_cs = vp * wm.M[3]; // World to clip space (4th column is translation)
            const auto p_ndc = fabsf(p_cs.w) > FLT_EPSILON ? vec3{p_cs} / p_cs.w : vec3{p_cs}; // Clip space to NDC
            const auto p_uv = vec2{p_ndc.x + 1, 1 - p_ndc.y} * 0.5f; // NDC to UV [0,1] (top-left origin)
            const auto p_px = std::bit_cast<ImVec2>(window_pos + p_uv * size); // UV to px
            auto &dl = *GetWindowDrawList();
            dl.AddCircleFilled(p_px, 3.5f, ColorConvertFloat4ToU32(std::bit_cast<ImVec4>(R.all_of<Active>(e) ? Colors.Active : Colors.Selected)), 10);
            dl.AddCircle(p_px, 3.5f, IM_COL32(0, 0, 0, 255), 10, 1.f);
        }
    }

    if (const auto selected_view = R.view<const Selected>(); !selected_view.empty()) { // Transform gizmo
        // Transform all root selected entities (whose parent is not also selected) around their average position,
        // using the active entity's rotation/scale.
        // Non-root selected entities already follow their parent's transform.
        const auto is_parent_selected = [&](entt::entity e) {
            if (const auto *node = R.try_get<SceneNode>(e)) {
                return node->Parent != entt::null && R.all_of<Selected>(node->Parent);
            }
            return false;
        };

        auto root_selected = selected_view | filter([&](auto e) { return !is_parent_selected(e); });
        const auto root_count = distance(root_selected);

        const auto active_entity = FindActiveEntity(R);
        const auto active_transform = active_entity != entt::null ? GetTransform(R, active_entity) : Transform{};
        const auto p = fold_left(root_selected | transform([&](auto e) { return R.get<Position>(e).Value; }), vec3{}, std::plus{}) / float(root_count);
        const auto start_transform_view = R.view<const StartTransform>();
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
        } else if (!start_transform_view.empty()) {
            R.clear<StartTransform>();
        }
    }
    { // Orientation gizmo
        static constexpr float OGizmoSize{90};
        const float padding = GetTextLineHeightWithSpacing();
        const auto pos = window_pos + vec2{GetWindowContentRegionMax().x, GetWindowContentRegionMin().y} - vec2{OGizmoSize, 0} + vec2{-padding, padding};
        OrientationGizmo::Draw(pos, OGizmoSize, Camera);
        if (Camera.Tick()) UpdateSceneUBO();
    }

    if (BoxSelectStart.has_value() && BoxSelectEnd.has_value()) {
        auto &dl = *GetWindowDrawList();
        const auto box_min = glm::min(*BoxSelectStart, *BoxSelectEnd);
        const auto box_max = glm::max(*BoxSelectStart, *BoxSelectEnd);
        dl.AddRectFilled(std::bit_cast<ImVec2>(box_min), std::bit_cast<ImVec2>(box_max), IM_COL32(255, 255, 255, 30));

        // Dashed outline
        static constexpr auto outline_color{IM_COL32(255, 255, 255, 200)};
        static constexpr float dash_size{4}, gap_size{4};
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
std::optional<MeshData> PrimitiveEditor(PrimitiveType type, bool is_create = true) {
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

    return {};
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
        auto position = R.get<const Position>(active_entity).Value;
        bool model_changed = DragFloat3("Position", &position[0], 0.01f);
        if (model_changed) R.patch<Position>(active_entity, [&](auto &p) { p.Value = position; });
        // Rotation editor
        {
            int mode_i = R.get<const RotationUiVariant>(active_entity).index();
            const char *modes[]{"Quat (WXYZ)", "XYZ Euler", "Axis Angle"};
            if (ImGui::Combo("Rotation mode", &mode_i, modes, IM_ARRAYSIZE(modes))) {
                R.replace<RotationUiVariant>(active_entity, CreateVariantByIndex<RotationUiVariant>(mode_i));
                SetRotation(R, active_entity, R.get<const Rotation>(active_entity).Value);
            }
        }
        auto rotation_ui = R.get<const RotationUiVariant>(active_entity);
        const bool rotation_changed = std::visit(
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
            rotation_ui
        );
        if (rotation_changed) {
            R.patch<RotationUiVariant>(active_entity, [&](auto &v) { v = rotation_ui; });
        }
        model_changed |= rotation_changed;

        const bool frozen = R.all_of<Frozen>(active_entity);
        if (frozen) BeginDisabled();
        const auto label = std::format("Scale{}", frozen ? " (frozen)" : "");
        auto scale = R.get<const Scale>(active_entity).Value;
        const bool scale_changed = DragFloat3(label.c_str(), &scale[0], 0.01f, 0.01f, 10);
        if (scale_changed) R.patch<Scale>(active_entity, [&](auto &s) { s.Value = scale; });
        model_changed |= scale_changed;
        if (frozen) EndDisabled();
        if (model_changed) {
            UpdateWorldMatrix(R, active_entity);
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
                SetMeshPositions(active_mesh_entity, new_mesh->Positions);
                NeedsRender = true;
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
                auto interaction_mode = int(SETTINGS.InteractionMode);
                bool interaction_mode_changed = false;
                for (const auto mode : InteractionModes) {
                    SameLine();
                    interaction_mode_changed |= RadioButton(to_string(mode).c_str(), &interaction_mode, int(mode));
                }
                if (interaction_mode_changed) SetInteractionMode(InteractionMode(interaction_mode));
                const bool wireframe_xray = SETTINGS.ViewportShading == ViewportShadingMode::Wireframe;
                if (interaction_mode == int(InteractionMode::Edit)) {
                    Checkbox("X-ray selection", &SelectionXRay);
                }
                if (wireframe_xray) {
                    SameLine();
                    TextDisabled("(wireframe)");
                }
                if (SETTINGS.InteractionMode == InteractionMode::Edit) {
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
                    R.emplace<PrimitiveType>(AddMesh(std::move(*mesh), {.Name = ToString(selected_type)}).first, selected_type);
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
            const auto &settings = SETTINGS;
            std::array bg_color{settings.BackgroundColor.float32[0], settings.BackgroundColor.float32[1], settings.BackgroundColor.float32[2]};
            if (ColorEdit3("Background color", bg_color.data())) {
                R.patch<SceneSettings>(SettingsEntity, [&bg_color](auto &s) { s.BackgroundColor = {bg_color[0], bg_color[1], bg_color[2], 1.f}; });
            }
            bool show_grid = settings.ShowGrid;
            if (Checkbox("Show grid", &show_grid)) {
                PATCH_SETTINGS([show_grid](auto &s) { s.ShowGrid = show_grid; });
            }
            if (Button("Recompile shaders")) {
                Pipelines->CompileShaders();
                CommandBufferDirty = NeedsRender = true;
            }
            SeparatorText("Viewport shading");
            PushID("ViewportShading");
            auto viewport_shading = int(settings.ViewportShading);
            bool viewport_shading_changed = RadioButton("Solid", &viewport_shading, int(ViewportShadingMode::Solid));
            SameLine();
            viewport_shading_changed |= RadioButton("Wireframe", &viewport_shading, int(ViewportShadingMode::Wireframe));
            PopID();

            bool smooth_shading_changed = false;
            bool smooth_shading = settings.SmoothShading;
            if (settings.ViewportShading == ViewportShadingMode::Solid) {
                smooth_shading_changed = Checkbox("Smooth shading", &smooth_shading);
            }

            auto color_mode = int(settings.ColorMode);
            bool color_mode_changed = false;
            if (settings.ViewportShading == ViewportShadingMode::Solid) {
                SeparatorText("Fill color mode");
                PushID("ColorMode");
                color_mode_changed |= RadioButton("Mesh", &color_mode, int(ColorMode::Mesh));
                color_mode_changed |= RadioButton("Normals", &color_mode, int(ColorMode::Normals));
                PopID();
            }
            if (viewport_shading_changed || color_mode_changed || smooth_shading_changed) {
                R.patch<SceneSettings>(SettingsEntity, [viewport_shading, color_mode, smooth_shading](auto &s) {
                    s.ViewportShading = ViewportShadingMode(viewport_shading);
                    s.ColorMode = ColorMode(color_mode);
                    s.SmoothShading = smooth_shading;
                });
                UpdateEdgeColors();
            }
            if (settings.ViewportShading == ViewportShadingMode::Wireframe) {
                if (ColorEdit3("Edge color", &EdgeColor.x)) UpdateEdgeColors();
            }
            {
                SeparatorText("Active/Selected");
                bool color_changed = ColorEdit3("Active color", &Colors.Active[0]);
                color_changed |= ColorEdit3("Selected color", &Colors.Selected[0]);
                if (color_changed) {
                    UpdateSceneUBO();
                }
                uint32_t edge_width = settings.SilhouetteEdgeWidth;
                if (SliderUInt("Edge width", &edge_width, 1, 4)) {
                    PATCH_SETTINGS([edge_width](auto &s) { s.SilhouetteEdgeWidth = edge_width; });
                }
            }
            if (!R.view<Selected>().empty()) {
                SeparatorText("Selection overlays");
                AlignTextToFramePadding();
                TextUnformatted("Normals:");
                bool changed = false;
                for (const auto element : NormalElements) {
                    SameLine();
                    bool show = settings.ShownNormalElements.contains(element);
                    const auto type_name = Capitalize(label(element));
                    if (Checkbox(type_name.c_str(), &show)) {
                        R.patch<SceneSettings>(SettingsEntity, [element, show](auto &s) {
                            if (show) s.ShownNormalElements.insert(element);
                            else s.ShownNormalElements.erase(element);
                        });
                        changed = true;
                    }
                }
                bool show_bboxes = settings.ShowBoundingBoxes;
                if (Checkbox("Bounding boxes", &show_bboxes)) {
                    PATCH_SETTINGS([show_bboxes](auto &s) { s.ShowBoundingBoxes = show_bboxes; });
                    changed = true;
                }
                if (changed) {
                    for (auto selected_entity : R.view<Selected>()) {
                        UpdateEntitySelectionOverlays(R.get<MeshInstance>(selected_entity).MeshEntity);
                    }
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
                UpdateSceneUBO();
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
                UpdateSceneUBO();
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
            for (const auto &[entity, name] : R.view<const Name>().each()) {
                const auto *node = R.try_get<SceneNode>(entity);
                if (!node || node->Parent == entt::null) render_entity(entity);
            }
        } else { // Iterate children
            for (const auto child : Children{&R, parent}) render_entity(child);
        }

        if (activate_entity != entt::null) Select(activate_entity);
        else if (toggle_selected != entt::null) ToggleSelected(toggle_selected);
        EndTable();
    }
}
