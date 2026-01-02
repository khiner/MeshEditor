#include "Scene.h"
#include "Widgets.h" // imgui

#include "Bindless.h"
#include "Entity.h"
#include "Excitable.h"
#include "OrientationGizmo.h"
#include "SceneTree.h"
#include "Shader.h"
#include "SvgResource.h"
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
#include <chrono>
#include <cstddef>
#include <format>
#include <limits>
#include <print>
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
struct Timer {
    std::string_view Name;
    std::chrono::steady_clock::time_point Start{std::chrono::steady_clock::now()};

    Timer(const std::string_view name) : Name{name} {}
    ~Timer() {
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - Start).count();
        std::println("{}: ms={:.3f}", Name, ms);
    }
};

std::vector<uint32_t> MakeElementStates(size_t count) { return std::vector<uint32_t>(std::max<size_t>(count, 1u), 0); }
ElementStateBuffer CreateElementStateBuffer(mvk::BufferContext &ctx, size_t count) {
    auto states = MakeElementStates(count);
    return {mvk::Buffer{ctx, as_bytes(states), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer}};
}
void ResetElementStateBuffer(ElementStateBuffer &buffer, size_t count) {
    buffer.Buffer.Update(as_bytes(MakeElementStates(count)));
}

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
    std::vector<uint32_t> Handles;
    std::optional<uint32_t> ActiveHandle; // Most recently selected element (always in Handles if set)
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
    glm::uvec2 Pixel;
    float Depth;
    uint32_t ObjectId;
    uint32_t Next;
    uint32_t Padding0;
};
static_assert(sizeof(SelectionNode) == 24, "SelectionNode must match std430 layout.");

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
    uint32_t NodeCount;
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

void WaitFor(vk::Fence fence, vk::Device device) {
    if (auto wait_result = device.waitForFences(fence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    device.resetFences(fence);
}
} // namespace

// Owns render-only/generated data (e.g., SceneUBO/indicators/overlays/selection fragments) separate from mesh storage.
struct SceneBuffers {
    static constexpr uint32_t MaxSelectionNodes = 4'000'000;
    static constexpr uint32_t MaxSelectableObjects = MaxSelectionNodes;
    static constexpr uint32_t BoxSelectBitsetWords = (MaxSelectableObjects + 31) / 32;
    static constexpr uint32_t ClickElementGroupSize = 256;
    static constexpr uint32_t MaxClickElementGroups = (MaxSelectionNodes + ClickElementGroupSize - 1) / ClickElementGroupSize;

    SceneBuffers(vk::PhysicalDevice pd, vk::Device d, vk::Instance instance, DescriptorSlots &slots)
        : Ctx{pd, d, instance, slots},
          VertexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::VertexBuffer},
          FaceIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          EdgeIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          VertexIndexBuffer{Ctx, vk::BufferUsageFlagBits::eStorageBuffer, SlotType::IndexBuffer},
          SceneUBO{Ctx, sizeof(SceneUBO), vk::BufferUsageFlagBits::eUniformBuffer, SlotType::Uniform},
          SelectionNodeBuffer{Ctx, MaxSelectionNodes * sizeof(SelectionNode), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::Buffer},
          SelectionCounterBuffer{Ctx, sizeof(SelectionCounters), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ClickResultBuffer{Ctx, sizeof(ClickResult), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          ClickElementResultBuffer{Ctx, MaxClickElementGroups * sizeof(ClickElementCandidate), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer},
          BoxSelectBitsetBuffer{Ctx, BoxSelectBitsetWords * sizeof(uint32_t), mvk::MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eStorageBuffer} {}

    const ClickResult &GetClickResult() const { return *reinterpret_cast<const ClickResult *>(ClickResultBuffer.GetData().data()); }
    vk::DescriptorBufferInfo GetBoxSelectBitsetDescriptor() const { return {*BoxSelectBitsetBuffer, 0, BoxSelectBitsetWords * sizeof(uint32_t)}; }

    RenderBuffers CreateRenderBuffers(std::vector<Vertex3D> &&vertices, std::vector<uint> &&indices, IndexKind index_kind) {
        const auto vertex_range = VertexBuffer.Allocate(vertices);
        auto &index_buffer = GetIndexBuffer(index_kind);
        const auto index_range = index_buffer.Allocate(indices);
        return {vertex_range, {index_range, index_buffer.Buffer.Slot}, index_kind};
    }
    RenderBuffers CreateRenderBuffers(SlottedBufferRange vertex_range, std::vector<uint> &&indices, IndexKind index_kind) {
        auto &index_buffer = GetIndexBuffer(index_kind);
        const auto index_range = index_buffer.Allocate(indices);
        return {vertex_range, {index_range, index_buffer.Buffer.Slot}, index_kind};
    }
    template<size_t N>
    RenderBuffers CreateRenderBuffers(std::vector<Vertex3D> &&vertices, const std::array<uint, N> &indices, IndexKind index_kind) {
        const auto vertex_range = VertexBuffer.Allocate(vertices);
        auto &index_buffer = GetIndexBuffer(index_kind);
        const auto index_range = index_buffer.Allocate(indices);
        return {vertex_range, {index_range, index_buffer.Buffer.Slot}, index_kind};
    }

    void UpdateRenderVertices(RenderBuffers &buffers, std::vector<Vertex3D> &&vertices) {
        if (!buffers.Vertices.OwnsRange()) return;
        VertexBuffer.Update(buffers.Vertices.Range, vertices);
    }

    void ReleaseRenderVertices(RenderBuffers &buffers) {
        if (!buffers.Vertices.OwnsRange()) return;
        VertexBuffer.Release(buffers.Vertices.Range);
        buffers.Vertices.Range = {};
    }

    void UpdateRenderIndices(RenderBuffers &buffers, std::span<const uint32_t> indices) {
        GetIndexBuffer(buffers.IndexType).Update(buffers.Indices.Range, indices);
    }

    void ReleaseRenderIndices(RenderBuffers &buffers) {
        if (buffers.Indices.Range.Count == 0) return;
        GetIndexBuffer(buffers.IndexType).Release(buffers.Indices.Range);
        buffers.Indices.Range = {};
    }

    BufferArena<uint32_t> &GetIndexBuffer(IndexKind kind) {
        switch (kind) {
            case IndexKind::Face: return FaceIndexBuffer;
            case IndexKind::Edge: return EdgeIndexBuffer;
            case IndexKind::Vertex: return VertexIndexBuffer;
        }
    }

    mvk::BufferContext Ctx;
    BufferArena<Vertex3D> VertexBuffer;
    BufferArena<uint32_t> FaceIndexBuffer, EdgeIndexBuffer, VertexIndexBuffer;
    mvk::Buffer SceneUBO;
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

    const auto cb = *TransferCommandBuffer;
    {
        // Write the bitmap into a temporary staging buffer.
        mvk::Buffer staging_buffer{Buffers->Ctx, as_bytes(data), mvk::MemoryUsage::CpuOnly};
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
        WaitFor(*TransferFence, Vk.Device);
    } // staging buffer is destroyed here

    Buffers->Ctx.ReclaimRetiredBuffers();
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

Scene::Scene(SceneVulkanResources vc, entt::registry &r)
    : Vk{vc},
      R{r},
      CommandPool{Vk.Device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, Vk.QueueFamily})},
      RenderCommandBuffer{std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())},
      TransferCommandBuffer{std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())},
      RenderFence{Vk.Device.createFenceUnique({})},
      TransferFence{Vk.Device.createFenceUnique({})},
      ClickCommandBuffer{std::move(Vk.Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front())},
      Slots{std::make_unique<DescriptorSlots>(Vk.Device, MakeBindlessConfig(Vk.PhysicalDevice))},
      SelectionHandles{std::make_unique<SelectionSlotHandles>(*Slots)},
      Pipelines{std::make_unique<ScenePipelines>(
          Vk.Device, Vk.PhysicalDevice,
          Slots->GetSetLayout(), Slots->GetSet()
      )},
      Buffers{std::make_unique<SceneBuffers>(Vk.PhysicalDevice, Vk.Device, Vk.Instance, *Slots)} {
    TransferCommandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    Meshes.Init(Buffers->Ctx);
    // EnTT listeners
    R.on_construct<Selected>().connect<&Scene::OnCreateSelected>(*this);
    R.on_destroy<Selected>().connect<&Scene::OnDestroySelected>(*this);
    R.on_construct<Excitable>().connect<&Scene::OnCreateExcitable>(*this);
    R.on_update<Excitable>().connect<&Scene::OnUpdateExcitable>(*this);
    R.on_destroy<Excitable>().connect<&Scene::OnDestroyExcitable>(*this);

    R.on_construct<ExcitedVertex>().connect<&Scene::OnCreateExcitedVertex>(*this);
    R.on_destroy<ExcitedVertex>().connect<&Scene::OnDestroyExcitedVertex>(*this);

    UpdateEdgeColors();
    UpdateSceneUBO();
    BoxSelectZeroBits.assign(SceneBuffers::BoxSelectBitsetWords, 0);

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
    R.clear<Mesh>();
    TransferCommandBuffer->end();
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
            for (auto &[_, buffers] : buffers->NormalIndicators) {
                Buffers->ReleaseRenderVertices(buffers);
                Buffers->ReleaseRenderIndices(buffers);
            }
            buffers->NormalIndicators.clear();
        }
        if (auto *bbox_buffers = r.try_get<BoundingBoxesBuffers>(mesh_entity)) {
            Buffers->ReleaseRenderVertices(bbox_buffers->Buffers);
            Buffers->ReleaseRenderIndices(bbox_buffers->Buffers);
        }
        r.remove<BoundingBoxesBuffers>(mesh_entity);
    }
}

void Scene::OnCreateExcitable(entt::registry &r, entt::entity e) {
    InteractionModes.insert(InteractionMode::Excite);
    SetInteractionMode(InteractionMode::Excite);
    UpdateRenderBuffers(r.get<MeshInstance>(e).MeshEntity);
}
void Scene::OnUpdateExcitable(entt::registry &r, entt::entity e) {
    UpdateRenderBuffers(r.get<MeshInstance>(e).MeshEntity);
}
void Scene::OnDestroyExcitable(entt::registry &r, entt::entity e) {
    if (r.storage<Excitable>().size() == 1) {
        if (InteractionMode == InteractionMode::Excite) SetInteractionMode(*InteractionModes.begin());
        InteractionModes.erase(InteractionMode::Excite);
    }
    if (const auto *mesh_instance = r.try_get<MeshInstance>(e)) {
        UpdateRenderBuffers(mesh_instance->MeshEntity);
    }
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
    const float scale_factor = 0.1f * glm::length(bbox.Max - bbox.Min);
    auto vertex_indicator_mesh = Meshes.CreateMesh(Arrow());
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
    UpdateRenderBuffers(mesh_entity);
}
void Scene::OnDestroyExcitedVertex(entt::registry &r, entt::entity e) {
    const auto indicator = r.get<ExcitedVertex>(e).IndicatorEntity;
    if (indicator != entt::null) {
        Destroy(indicator);
    }
    if (const auto *mesh_instance = r.try_get<MeshInstance>(e)) {
        UpdateRenderBuffers(mesh_instance->MeshEntity);
    }
}

vk::ImageView Scene::GetViewportImageView() const { return *Pipelines->Main.Resources->ResolveImage.View; }

namespace {
void UpdateVisibleObjectIds(entt::registry &r) {
    std::unordered_map<entt::entity, std::vector<uint32_t>> ids_by_mesh;
    uint32_t object_id = 1;
    for (const auto e : r.view<Visible>()) {
        const auto mesh_entity = r.get<MeshInstance>(e).MeshEntity;
        auto &models = r.get<ModelsBuffer>(mesh_entity);
        const uint32_t instance_count = models.Buffer.UsedSize / sizeof(WorldMatrix);
        if (instance_count == 0) continue;
        auto &ids = ids_by_mesh[mesh_entity];
        if (ids.empty()) ids.resize(instance_count, 0);
        const auto buffer_index = r.get<RenderInstance>(e).BufferIndex;
        if (buffer_index < ids.size()) ids[buffer_index] = object_id;
        r.get<RenderInstance>(e).ObjectId = object_id;
        ++object_id;
    }

    for (auto &[mesh_entity, ids] : ids_by_mesh) {
        r.get<ModelsBuffer>(mesh_entity).ObjectIds.Update(as_bytes(ids));
    }
}
} // namespace

void Scene::SetVisible(entt::entity entity, bool visible) {
    const bool already_visible = R.all_of<Visible>(entity);
    if ((visible && already_visible) || (!visible && !already_visible)) return;

    const auto mesh_entity = R.get<MeshInstance>(entity).MeshEntity;
    auto &models = R.get<ModelsBuffer>(mesh_entity);
    auto &model_buffer = models.Buffer;
    auto &object_ids = models.ObjectIds;
    if (visible) {
        auto &render_instance = R.emplace_or_replace<RenderInstance>(entity);
        render_instance.BufferIndex = model_buffer.UsedSize / sizeof(WorldMatrix);
        render_instance.ObjectId = 0;
        model_buffer.Insert(as_bytes(R.get<WorldMatrix>(entity)), model_buffer.UsedSize);
        object_ids.Insert(as_bytes(uint32_t{0}), object_ids.UsedSize); // Placeholder; actual IDs set on-demand.
        R.emplace<Visible>(entity);
    } else {
        R.remove<Visible>(entity);
        auto &render_instance = R.get<RenderInstance>(entity);
        const uint old_model_index = render_instance.BufferIndex;
        render_instance.ObjectId = 0;
        model_buffer.Erase(old_model_index * sizeof(WorldMatrix), sizeof(WorldMatrix));
        object_ids.Erase(old_model_index * sizeof(uint32_t), sizeof(uint32_t));
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
    UpdateVisibleObjectIds(R);
    InvalidateCommandBuffer();
}

entt::entity Scene::AddMesh(Mesh &&mesh, MeshCreateInfo info) {
    const auto mesh_entity = R.create();
    { // Mesh data
        const auto bbox = MeshRender::ComputeBoundingBox(mesh);
        R.emplace<BBox>(mesh_entity, bbox);

        const auto vertex_range = Meshes.GetVerticesRange(mesh.GetStoreId());
        const auto vertex_slot = Meshes.GetVerticesSlot();
        auto face_buffers = Buffers->CreateRenderBuffers({vertex_range, vertex_slot}, MeshRender::CreateFaceIndices(mesh), IndexKind::Face);
        auto edge_buffers = Buffers->CreateRenderBuffers({vertex_range, vertex_slot}, MeshRender::CreateEdgeIndices(mesh), IndexKind::Edge);
        auto vertex_buffers = Buffers->CreateRenderBuffers({vertex_range, vertex_slot}, MeshRender::CreateVertexIndices(mesh), IndexKind::Vertex);

        R.emplace<Mesh>(mesh_entity, std::move(mesh));
        R.emplace<MeshSelection>(mesh_entity);
        R.emplace<ModelsBuffer>(
            mesh_entity,
            mvk::Buffer{Buffers->Ctx, sizeof(WorldMatrix), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ModelBuffer},
            mvk::Buffer{Buffers->Ctx, sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, SlotType::ObjectIdBuffer}
        );
        R.emplace<MeshBuffers>(mesh_entity, std::move(face_buffers), std::move(edge_buffers), std::move(vertex_buffers));
        {
            R.emplace<MeshElementStateBuffers>(
                mesh_entity,
                CreateElementStateBuffer(Buffers->Ctx, mesh.FaceCount()),
                CreateElementStateBuffer(Buffers->Ctx, mesh.EdgeCount() * 2),
                CreateElementStateBuffer(Buffers->Ctx, mesh.VertexCount())
            );
        }
        UpdateRenderBuffers(mesh_entity);
        if (ShowBoundingBoxes) {
            R.emplace<BoundingBoxesBuffers>(
                mesh_entity,
                Buffers->CreateRenderBuffers(CreateBoxVertices(bbox), BBox::EdgeIndices, IndexKind::Edge)
            );
        }
    }

    const auto instance_entity = R.create();
    R.emplace<MeshInstance>(instance_entity, mesh_entity);
    UpdateTransform(R, instance_entity, info.Transform);
    R.emplace<Name>(instance_entity, CreateName(R, info.Name));

    auto &models = R.get<ModelsBuffer>(mesh_entity);
    models.Buffer.Reserve(models.Buffer.UsedSize + sizeof(WorldMatrix));
    models.ObjectIds.Reserve(models.ObjectIds.UsedSize + sizeof(uint32_t));
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

    InvalidateCommandBuffer();
    return instance_entity;
}

entt::entity Scene::AddMesh(MeshData &&data, MeshCreateInfo info) {
    return AddMesh(Meshes.CreateMesh(std::move(data)), std::move(info));
}

entt::entity Scene::AddMesh(const fs::path &path, MeshCreateInfo info) {
    auto mesh = Meshes.LoadMesh(path);
    if (!mesh) throw std::runtime_error(std::format("Failed to load mesh: {}", path.string()));
    const auto e = AddMesh(std::move(*mesh), std::move(info));
    R.emplace<Path>(e, path);
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
        auto &models = R.get<ModelsBuffer>(mesh_entity);
        models.Buffer.Reserve(models.Buffer.UsedSize + sizeof(WorldMatrix));
        models.ObjectIds.Reserve(models.ObjectIds.UsedSize + sizeof(uint32_t));
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
    InvalidateCommandBuffer();
}

void Scene::ReplaceMesh(entt::entity e, MeshData &&data) {
    auto mesh = Meshes.CreateMesh(std::move(data));
    // Update components
    const auto bbox = MeshRender::ComputeBoundingBox(mesh);
    R.replace<BBox>(e, bbox);
    R.replace<Mesh>(e, std::move(mesh));

    const auto &pm = R.get<Mesh>(e);
    auto &mesh_buffers = R.get<MeshBuffers>(e);
    const auto vertex_range = Meshes.GetVerticesRange(pm.GetStoreId());
    const auto vertex_slot = Meshes.GetVerticesSlot();
    const auto reset_buffers = [&](RenderBuffers &buffers, const std::vector<uint> &indices) {
        Buffers->ReleaseRenderVertices(buffers);
        buffers.Vertices = {vertex_range, vertex_slot};
        Buffers->UpdateRenderIndices(buffers, indices);
    };
    const auto face_indices = MeshRender::CreateFaceIndices(pm);
    const auto edge_indices = MeshRender::CreateEdgeIndices(pm);
    const auto vertex_indices = MeshRender::CreateVertexIndices(pm);
    reset_buffers(mesh_buffers.Faces, face_indices);
    reset_buffers(mesh_buffers.Edges, edge_indices);
    reset_buffers(mesh_buffers.Vertices, vertex_indices);
    if (auto *state_buffers = R.try_get<MeshElementStateBuffers>(e)) {
        ResetElementStateBuffer(state_buffers->Faces, pm.FaceCount());
        ResetElementStateBuffer(state_buffers->Edges, pm.EdgeCount() * 2);
        ResetElementStateBuffer(state_buffers->Vertices, pm.VertexCount());
    }
    for (auto &[element, buffers] : mesh_buffers.NormalIndicators) {
        Buffers->UpdateRenderVertices(buffers, MeshRender::CreateNormalVertices(pm, element));
        Buffers->UpdateRenderIndices(buffers, MeshRender::CreateNormalIndices(pm, element));
    }
    if (auto buffers = R.try_get<BoundingBoxesBuffers>(e)) {
        Buffers->UpdateRenderVertices(buffers->Buffers, CreateBoxVertices(bbox));
    }
    UpdateRenderBuffers(e);
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
            if (auto *mesh_buffers = R.try_get<MeshBuffers>(mesh_entity)) {
                Buffers->ReleaseRenderVertices(mesh_buffers->Faces);
                Buffers->ReleaseRenderVertices(mesh_buffers->Edges);
                Buffers->ReleaseRenderVertices(mesh_buffers->Vertices);
                Buffers->ReleaseRenderIndices(mesh_buffers->Faces);
                Buffers->ReleaseRenderIndices(mesh_buffers->Edges);
                Buffers->ReleaseRenderIndices(mesh_buffers->Vertices);
                for (auto &[_, buffers] : mesh_buffers->NormalIndicators) {
                    Buffers->ReleaseRenderVertices(buffers);
                    Buffers->ReleaseRenderIndices(buffers);
                }
            }
            if (auto *buffers = R.try_get<BoundingBoxesBuffers>(mesh_entity)) {
                Buffers->ReleaseRenderVertices(buffers->Buffers);
                Buffers->ReleaseRenderIndices(buffers->Buffers);
            }
            R.destroy(mesh_entity);
        }
    }

    InvalidateCommandBuffer();
}

void Scene::SetInteractionMode(::InteractionMode mode) {
    if (InteractionMode == mode) return;

    InteractionMode = mode;
    for (const auto &entity : R.view<Mesh>()) UpdateRenderBuffers(entity);
    const auto e = FindActiveEntity(R);
    if (e != entt::null && R.all_of<Excitable>(e)) {
        UpdateRenderBuffers(R.get<MeshInstance>(e).MeshEntity);
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

void Scene::UpdateRenderBuffers(entt::entity e) {
    if (const auto *mesh = R.try_get<Mesh>(e)) {
        std::unordered_set<VH> selected_vertices;
        std::unordered_set<EH> selected_edges;
        std::unordered_set<FH> selected_faces;
        const auto &selection = R.get<MeshSelection>(e);
        const bool edit_mode = InteractionMode == InteractionMode::Edit;
        const bool excite_mode = InteractionMode == InteractionMode::Excite;
        Element element = Element::None;
        std::optional<uint32_t> active_handle;
        if (excite_mode) {
            element = Element::Vertex;
            for (auto [entity, mi, excitable] : R.view<const MeshInstance, const Excitable>().each()) {
                if (mi.MeshEntity != e) continue;
                for (const auto vertex : excitable.ExcitableVertices) {
                    selected_vertices.emplace(VH(vertex));
                }
                if (R.all_of<ExcitedVertex>(entity)) {
                    active_handle = R.get<ExcitedVertex>(entity).Vertex;
                }
                break;
            }
        } else if (edit_mode && selection.Element == EditMode) {
            element = selection.Element;
            if (element == Element::Vertex) {
                for (auto h : selection.Handles) {
                    selected_vertices.emplace(h);
                }
            } else if (element == Element::Edge) {
                for (auto h : selection.Handles) {
                    selected_edges.emplace(h);
                }
            } else if (element == Element::Face) {
                for (auto h : selection.Handles) {
                    selected_faces.emplace(h);
                    for (const auto heh : mesh->fh_range(FH{h})) selected_edges.emplace(mesh->GetEdge(heh));
                }
            }
            active_handle = selection.ActiveHandle;
        }

        auto face_states = MakeElementStates(mesh->FaceCount());
        auto edge_states = MakeElementStates(mesh->EdgeCount() * 2);
        auto vertex_states = MakeElementStates(mesh->VertexCount());
        if (element == Element::Face) {
            for (const auto fh : selected_faces) face_states[*fh] |= MeshRender::ElementStateSelected;
            if (active_handle) face_states[*active_handle] |= MeshRender::ElementStateActive;
        }
        if (element == Element::Edge || element == Element::Face) {
            const bool has_active_edge = element == Element::Edge && active_handle.has_value();
            for (uint32_t ei = 0; ei < mesh->EdgeCount(); ++ei) {
                const EH eh{ei};
                uint32_t state = 0;
                if (selected_edges.contains(eh)) state |= MeshRender::ElementStateSelected;
                if (has_active_edge && *active_handle == ei) state |= MeshRender::ElementStateActive;
                edge_states[2 * ei] = state;
                edge_states[2 * ei + 1] = state;
            }
        } else if (element == Element::Vertex) {
            const bool has_active_vertex = active_handle.has_value();
            for (uint32_t ei = 0; ei < mesh->EdgeCount(); ++ei) {
                const auto heh = mesh->GetHalfedge(EH{ei}, 0);
                const auto v_from = mesh->GetFromVertex(heh);
                const auto v_to = mesh->GetToVertex(heh);
                uint32_t state_from = 0;
                uint32_t state_to = 0;
                if (selected_vertices.contains(v_from)) state_from |= MeshRender::ElementStateSelected;
                if (selected_vertices.contains(v_to)) state_to |= MeshRender::ElementStateSelected;
                if (has_active_vertex && *active_handle == *v_from) state_from |= MeshRender::ElementStateActive;
                if (has_active_vertex && *active_handle == *v_to) state_to |= MeshRender::ElementStateActive;
                edge_states[2 * ei] = state_from;
                edge_states[2 * ei + 1] = state_to;
            }
        }
        if (element == Element::Vertex) {
            for (const auto vh : selected_vertices) vertex_states[*vh] |= MeshRender::ElementStateSelected;
            if (active_handle) vertex_states[*active_handle] |= MeshRender::ElementStateActive;
        }

        auto &states = R.get<MeshElementStateBuffers>(e);
        states.Faces.Buffer.Update(as_bytes(face_states));
        states.Edges.Buffer.Update(as_bytes(edge_states));
        states.Vertices.Buffer.Update(as_bytes(vertex_states));
        InvalidateCommandBuffer();
    };
}

std::string Scene::DebugBufferHeapUsage() const { return Buffers->Ctx.DebugHeapUsage(); }

namespace {
void SetTransform(entt::registry &r, entt::entity e, Transform &&t) {
    UpdateTransform(r, e, std::move(t));
}

// If `model_index` is set, only the model at that index is rendered. Otherwise, all models are rendered.
void Draw(
    vk::CommandBuffer cb, const ShaderPipeline &pipeline, const RenderBuffers &buffers, const ModelsBuffer &models,
    DrawPushConstants pc, std::optional<uint> model_index = {}
) {
    pc.FirstInstance = model_index.value_or(0);
    const auto instance_count = model_index.has_value() ? 1 : models.Buffer.UsedSize / sizeof(WorldMatrix);
    cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawPushConstants), &pc);
    cb.draw(buffers.Indices.Range.Count, instance_count, 0, 0);
}

DrawPushConstants MakeDrawPc(const RenderBuffers &rb, const ModelsBuffer &mb) {
    return {
        .VertexSlot = rb.Vertices.Slot,
        .IndexSlot = rb.Indices.Slot,
        .IndexOffset = rb.Indices.Range.Offset,
        .ModelSlot = mb.Buffer.Slot,
        .FirstInstance = 0,
        .ObjectIdSlot = InvalidSlot,
        .FaceNormalSlot = InvalidSlot,
        .FaceIdOffset = 0,
        .FaceNormalOffset = 0,
        .VertexCountOrHeadImageSlot = rb.Vertices.Range.Count,
        .SelectionNodesSlot = 0,
        .SelectionCounterSlot = 0,
        .ElementIdOffset = 0,
        .ElementStateSlot = InvalidSlot,
        .VertexOffset = rb.Vertices.Range.Offset,
        .Pad0 = 0,
        .LineColor = vec4{0, 0, 0, 0},
    };
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
    const bool is_edit_mode = InteractionMode == InteractionMode::Edit;
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
            auto &buffers = R.get<MeshBuffers>(mesh_entity).Faces;
            auto &models = R.get<ModelsBuffer>(mesh_entity);
            auto pc = MakeDrawPc(buffers, models);
            pc.ObjectIdSlot = models.ObjectIds.Slot;
            Draw(cb, pipeline, buffers, models, pc, *GetModelBufferIndex(R, e));
        };
        if (is_edit_mode) {
            for (const auto e : silhouette_instances) render(e);
        } else {
            for (const auto e : R.view<Selected>()) render(e);
        }
    };

    const bool render_silhouette = !R.view<Selected>().empty() &&
        (InteractionMode == InteractionMode::Object || !silhouette_instances.empty());
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
        } edge_pc{SilhouetteEdgeWidth, SelectionHandles->SilhouetteSampler};
        cb.pushConstants(*silhouette_edo.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(edge_pc), &edge_pc);
        silhouette_edo.RenderQuad(cb);
        cb.endRenderPass();
    }

    const auto &main = Pipelines->Main;
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
        const bool is_excite_mode = InteractionMode == InteractionMode::Excite;
        const bool show_solid = ViewportShading == ViewportShadingMode::Solid;
        const bool show_wireframe = ViewportShading == ViewportShadingMode::Wireframe;

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
                auto pc = MakeDrawPc(mesh_buffers.Faces, models);
                const auto face_id_range = Meshes.GetFaceIdRange(mesh.GetStoreId());
                const auto face_normal_range = Meshes.GetFaceNormalRange(mesh.GetStoreId());
                pc.ObjectIdSlot = Meshes.GetFaceIdSlot();
                pc.FaceIdOffset = face_id_range.Offset;
                pc.FaceNormalSlot = SmoothShading ? InvalidSlot : Meshes.GetFaceNormalSlot();
                pc.FaceNormalOffset = face_normal_range.Offset;
                if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                    // Draw primary with element state first, then all without (depth LESS won't overwrite)
                    pc.ElementStateSlot = state_buffers.Faces.Buffer.Slot;
                    Draw(cb, pipeline, mesh_buffers.Faces, models, pc, *GetModelBufferIndex(R, it->second));
                    pc.ElementStateSlot = InvalidSlot;
                    Draw(cb, pipeline, mesh_buffers.Faces, models, pc);
                } else {
                    pc.ElementStateSlot = state_buffers.Faces.Buffer.Slot;
                    Draw(cb, pipeline, mesh_buffers.Faces, models, pc);
                }
            }
        }

        // Wireframe edges
        if (show_wireframe || is_edit_mode || is_excite_mode) {
            const auto &pipeline = main.Renderer.Bind(cb, SPT::Line);
            for (auto [entity, mesh_buffers, models, state_buffers] :
                 R.view<MeshBuffers, ModelsBuffer, MeshElementStateBuffers>().each()) {
                auto pc = MakeDrawPc(mesh_buffers.Edges, models);
                pc.ElementStateSlot = state_buffers.Edges.Buffer.Slot;
                if (show_wireframe) {
                    Draw(cb, pipeline, mesh_buffers.Edges, models, pc);
                } else if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                    Draw(cb, pipeline, mesh_buffers.Edges, models, pc, *GetModelBufferIndex(R, it->second));
                } else if (excitable_mesh_entities.contains(entity)) {
                    Draw(cb, pipeline, mesh_buffers.Edges, models, pc);
                }
            }
        }

        // Vertex points
        if ((is_edit_mode && EditMode == Element::Vertex) || is_excite_mode) {
            const auto &pipeline = main.Renderer.Bind(cb, SPT::Point);
            for (auto [entity, mesh_buffers, models, state_buffers] :
                 R.view<MeshBuffers, ModelsBuffer, MeshElementStateBuffers>().each()) {
                auto pc = MakeDrawPc(mesh_buffers.Vertices, models);
                pc.ElementStateSlot = state_buffers.Vertices.Buffer.Slot;
                if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                    Draw(cb, pipeline, mesh_buffers.Vertices, models, pc, *GetModelBufferIndex(R, it->second));
                } else if (excitable_mesh_entities.contains(entity)) {
                    Draw(cb, pipeline, mesh_buffers.Vertices, models, pc);
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
                auto pc = MakeDrawPc(buffers, models);
                pc.VertexSlot = vertex_slot;
                pc.LineColor = element == Element::Face ? MeshRender::FaceNormalIndicatorColor : MeshRender::VertexNormalIndicatorColor;
                Draw(cb, pipeline, buffers, models, pc);
            }
        }
        for (auto [_, bounding_boxes, models] : R.view<BoundingBoxesBuffers, ModelsBuffer>().each()) {
            auto pc = MakeDrawPc(bounding_boxes.Buffers, models);
            pc.VertexSlot = vertex_slot;
            pc.LineColor = MeshRender::EdgeColor;
            Draw(cb, pipeline, bounding_boxes.Buffers, models, pc);
        }
    }

    // Grid lines texture
    if (ShowGrid) main.Renderer.ShaderPipelines.at(SPT::Grid).RenderQuad(cb);

    cb.endRenderPass();
    cb.end();
}

void Scene::UpdateEdgeColors() {
    MeshRender::EdgeColor = ViewportShading == ViewportShadingMode::Solid ? MeshEdgeColor : EdgeColor;
    UpdateSceneUBO();
}

void Scene::UpdateSceneUBO() {
    const float aspect_ratio = Extent.width == 0 || Extent.height == 0 ? 1.f : float(Extent.width) / float(Extent.height);
    SceneUBO scene_ubo{
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
        .ActiveColor = MeshRender::ActiveColor
    };
    Buffers->SceneUBO.Update(as_bytes(scene_ubo));
    RequestRender();
}

void Scene::UpdateEntitySelectionOverlays(entt::entity instance_entity) {
    const auto mesh_entity = R.get<MeshInstance>(instance_entity).MeshEntity;
    const auto &mesh = R.get<const Mesh>(mesh_entity);
    auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
    for (const auto element : NormalElements) {
        if (ShownNormalElements.contains(element) && !mesh_buffers.NormalIndicators.contains(element)) {
            const auto index_kind = element == Element::Face ? IndexKind::Face : IndexKind::Vertex;
            mesh_buffers.NormalIndicators.emplace(
                element,
                Buffers->CreateRenderBuffers(MeshRender::CreateNormalVertices(mesh, element), MeshRender::CreateNormalIndices(mesh, element), index_kind)
            );
        } else if (!ShownNormalElements.contains(element) && mesh_buffers.NormalIndicators.contains(element)) {
            Buffers->ReleaseRenderVertices(mesh_buffers.NormalIndicators.at(element));
            Buffers->ReleaseRenderIndices(mesh_buffers.NormalIndicators.at(element));
            mesh_buffers.NormalIndicators.erase(element);
        }
    }
    if (ShowBoundingBoxes && !R.all_of<BoundingBoxesBuffers>(mesh_entity)) {
        const auto &bbox = R.get<BBox>(mesh_entity);
        R.emplace<BoundingBoxesBuffers>(mesh_entity, Buffers->CreateRenderBuffers(CreateBoxVertices(bbox), BBox::EdgeIndices, IndexKind::Edge));
    } else if (!ShowBoundingBoxes && R.all_of<BoundingBoxesBuffers>(mesh_entity)) {
        Buffers->ReleaseRenderVertices(R.get<BoundingBoxesBuffers>(mesh_entity).Buffers);
        Buffers->ReleaseRenderIndices(R.get<BoundingBoxesBuffers>(mesh_entity).Buffers);
        R.remove<BoundingBoxesBuffers>(mesh_entity);
    }
}

using namespace ImGui;

namespace {
constexpr vec2 ToGlm(ImVec2 v) { return std::bit_cast<vec2>(v); }
constexpr vk::Extent2D ToExtent(vec2 e) { return {uint(e.x), uint(e.y)}; }

std::optional<std::pair<glm::uvec2, glm::uvec2>> ComputeBoxSelectPixels(
    vec2 start,
    vec2 end,
    vec2 window_pos,
    vk::Extent2D extent,
    float drag_threshold
) {
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

void Scene::RenderSelectionPass() {
    const Timer timer{"RenderSelectionPass"};

    const auto primary_edit_instances = InteractionMode == InteractionMode::Edit ?
        ComputePrimaryEditInstances(R) :
        std::unordered_map<entt::entity, entt::entity>{};

    // Object selection never uses depth testing - we want all visible pixels regardless of occlusion
    RenderSelectionPassWith(false, [&](vk::CommandBuffer cb, const PipelineRenderer &renderer) {
        const auto &pipeline = renderer.Bind(cb, SPT::SelectionFragmentXRay);
        for (auto [mesh_entity, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
            const uint32_t instance_count = models.Buffer.UsedSize / sizeof(WorldMatrix);
            if (instance_count == 0) continue;

            auto pc = MakeDrawPc(mesh_buffers.Faces, models);
            pc.ObjectIdSlot = models.ObjectIds.Slot;
            pc.VertexCountOrHeadImageSlot = SelectionHandles->HeadImage;
            pc.SelectionNodesSlot = Buffers->SelectionNodeBuffer.Slot;
            pc.SelectionCounterSlot = SelectionHandles->SelectionCounter;
            if (auto it = primary_edit_instances.find(mesh_entity); it != primary_edit_instances.end()) {
                Draw(cb, pipeline, mesh_buffers.Faces, models, pc, *GetModelBufferIndex(R, it->second));
            } else { // todo can we guarantee only selected instances are rendered here and thus we can drop this check?
                Draw(cb, pipeline, mesh_buffers.Faces, models, pc);
            }
        }
    });

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
            auto pc = MakeDrawPc(mesh_buffers.Faces, models);
            pc.ObjectIdSlot = models.ObjectIds.Slot;
            Draw(cb, pipeline, mesh_buffers.Faces, models, pc, *model_index);
        }
    }
    cb.endRenderPass();
}

void Scene::RenderSelectionPassWith(bool render_depth, const std::function<void(vk::CommandBuffer, const PipelineRenderer &)> &draw_fn) {
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
    Vk.Queue.submit(submit, *TransferFence);
    WaitFor(*TransferFence, Vk.Device);
}

namespace {
void RunSelectionCompute(vk::CommandBuffer cb, vk::Queue queue, vk::Fence fence, vk::Device device, const auto &compute, const auto &pc, auto &&dispatch) {
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
    queue.submit(submit, fence);
    WaitFor(fence, device);
}

bool HasSelectionNodes(const mvk::Buffer &counter_buffer) {
    const auto *counters = reinterpret_cast<const SelectionCounters *>(counter_buffer.GetData().data());
    return std::min<uint32_t>(counters->Count, SceneBuffers::MaxSelectionNodes) != 0;
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
    uint32_t selection_nodes_slot, uint32_t click_result_slot, glm::uvec2 mouse_px, uint32_t max_element_id
) {
    if (!HasSelectionNodes(buffers.SelectionCounterBuffer)) return {};

    const auto *counters = reinterpret_cast<const SelectionCounters *>(buffers.SelectionCounterBuffer.GetData().data());
    const uint32_t node_count = std::min<uint32_t>(counters->Count, SceneBuffers::MaxSelectionNodes);
    if (node_count == 0) return {};
    const uint32_t group_count = (node_count + SceneBuffers::ClickElementGroupSize - 1) / SceneBuffers::ClickElementGroupSize;
    if (group_count == 0) return {};

    RunSelectionCompute(
        cb, queue, fence, device, compute,
        ClickSelectElementPushConstants{
            .TargetPx = mouse_px,
            .NodeCount = node_count,
            .SelectionNodesIndex = selection_nodes_slot,
            .ClickResultIndex = click_result_slot,
        },
        [group_count](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(group_count, 1, 1); }
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

void Scene::RenderEditSelectionPass(std::span<const ElementRange> ranges, Element element) {
    if (ranges.empty() || element == Element::None) return;

    const auto primary_edit_instances = ComputePrimaryEditInstances(R);
    const Timer timer{"RenderEditSelectionPass"};
    const bool xray_selection = SelectionXRay || ViewportShading == ViewportShadingMode::Wireframe;
    RenderSelectionPassWith(!xray_selection, [&](vk::CommandBuffer cb, const PipelineRenderer &renderer) {
        const auto &pipeline = renderer.Bind(cb, SelectionPipelineForElement(element, xray_selection));
        for (const auto &r : ranges) {
            auto &mesh_buffers = R.get<MeshBuffers>(r.MeshEntity);
            auto &models = R.get<ModelsBuffer>(r.MeshEntity);
            const auto &mesh = R.get<Mesh>(r.MeshEntity);
            auto &buffers = element == Element::Vertex ? mesh_buffers.Vertices :
                element == Element::Edge               ? mesh_buffers.Edges :
                                                         mesh_buffers.Faces;
            auto pc = MakeDrawPc(buffers, models);
            pc.ObjectIdSlot = element == Element::Face ? Meshes.GetFaceIdSlot() : InvalidSlot;
            pc.FaceIdOffset = element == Element::Face ? Meshes.GetFaceIdRange(mesh.GetStoreId()).Offset : 0;
            pc.VertexCountOrHeadImageSlot = SelectionHandles->HeadImage;
            pc.SelectionNodesSlot = Buffers->SelectionNodeBuffer.Slot;
            pc.SelectionCounterSlot = SelectionHandles->SelectionCounter;
            pc.ElementIdOffset = r.Offset;
            if (auto it = primary_edit_instances.find(r.MeshEntity); it != primary_edit_instances.end()) {
                Draw(cb, pipeline, buffers, models, pc, *GetModelBufferIndex(R, it->second));
            } else { // todo can we guarantee only selected instances are rendered here and thus we can drop this check?
                Draw(cb, pipeline, buffers, models, pc);
            }
        }
    });
}

std::vector<std::vector<uint32_t>> Scene::RunBoxSelectElements(std::span<const ElementRange> ranges, Element element, glm::uvec2 box_min, glm::uvec2 box_max) {
    std::vector<std::vector<uint32_t>> results(ranges.size());
    if (ranges.empty() || element == Element::None) return results;
    if (box_min.x >= box_max.x || box_min.y >= box_max.y) return results;

    const Timer timer{"RunBoxSelectElements"};

    RenderEditSelectionPass(ranges, element);
    if (!HasSelectionNodes(Buffers->SelectionCounterBuffer)) return results;

    const auto element_count = fold_left(
        ranges, uint32_t{0},
        [](uint32_t total, const auto &range) { return std::max(total, range.Offset + range.Count); }
    );
    if (element_count == 0) return results;

    const uint32_t bitset_words = (element_count + 31) / 32;
    if (bitset_words > BoxSelectZeroBits.size()) return results;

    const std::span<const uint32_t> zero_bits{BoxSelectZeroBits.data(), bitset_words};
    Buffers->BoxSelectBitsetBuffer.Write(std::as_bytes(zero_bits));

    auto cb = *ClickCommandBuffer;
    const uint32_t group_count_x = (box_max.x - box_min.x + 15) / 16;
    const uint32_t group_count_y = (box_max.y - box_min.y + 15) / 16;
    RunSelectionCompute(
        cb, Vk.Queue, *TransferFence, Vk.Device, Pipelines->BoxSelect,
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
        }
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

    const ElementRange range{mesh_entity, 0, element_count};
    RenderEditSelectionPass(std::span{&range, 1}, element);
    if (const auto index = FindNearestSelectionElement(
            *Buffers, Pipelines->ClickSelectElement, *ClickCommandBuffer,
            Vk.Queue, *TransferFence, Vk.Device,
            Buffers->SelectionNodeBuffer.Slot, SelectionHandles->ClickElementResult, mouse_px, element_count
        )) {
        return AnyHandle{element, *index};
    }
    return {};
}

std::optional<uint32_t> Scene::RunClickSelectExcitableVertex(entt::entity instance_entity, glm::uvec2 mouse_px) {
    if (!R.all_of<Excitable>(instance_entity)) return {};

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
        auto pc = MakeDrawPc(mesh_buffers.Vertices, models);
        pc.VertexCountOrHeadImageSlot = SelectionHandles->HeadImage;
        pc.SelectionNodesSlot = Buffers->SelectionNodeBuffer.Slot;
        pc.SelectionCounterSlot = SelectionHandles->SelectionCounter;
        pc.ElementStateSlot = state_buffers.Vertices.Buffer.Slot;
        Draw(cb, pipeline, mesh_buffers.Vertices, models, pc, model_index);
    });

    return FindNearestSelectionElement(
        *Buffers, Pipelines->ClickSelectElement, *ClickCommandBuffer,
        Vk.Queue, *TransferFence, Vk.Device,
        Buffers->SelectionNodeBuffer.Slot, SelectionHandles->ClickElementResult, mouse_px, vertex_count
    );
}

// Returns entities hit at mouse_px, sorted by depth (near-to-far), with duplicates removed.
std::vector<entt::entity> Scene::RunClickSelect(glm::uvec2 mouse_px) {
    if (SelectionStale) RenderSelectionPass();
    if (!HasSelectionNodes(Buffers->SelectionCounterBuffer)) return {};

    Buffers->ClickResultBuffer.Write(as_bytes(ClickResult{}));
    auto cb = *ClickCommandBuffer;
    const auto &compute = Pipelines->ClickSelect;
    RunSelectionCompute(
        cb, Vk.Queue, *TransferFence, Vk.Device, compute,
        ClickSelectPushConstants{
            .TargetPx = mouse_px,
            .HeadImageIndex = SelectionHandles->HeadImage,
            .SelectionNodesIndex = Buffers->SelectionNodeBuffer.Slot,
            .ClickResultIndex = SelectionHandles->ClickResult,
        },
        [](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(1, 1, 1); }
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

std::vector<entt::entity> Scene::RunBoxSelect(glm::uvec2 box_min, glm::uvec2 box_max) {
    if (box_min.x >= box_max.x || box_min.y >= box_max.y) return {};
    if (SelectionStale) RenderSelectionPass();
    if (!HasSelectionNodes(Buffers->SelectionCounterBuffer)) return {};

    const auto visible_entities = R.view<Visible>() | to<std::vector>();
    const auto object_count = static_cast<uint32_t>(visible_entities.size());
    if (object_count == 0) return {};

    const uint32_t bitset_words = (object_count + 31) / 32;
    if (bitset_words > BoxSelectZeroBits.size()) return {};

    const std::span<const uint32_t> zero_bits{BoxSelectZeroBits.data(), bitset_words};
    Buffers->BoxSelectBitsetBuffer.Write(std::as_bytes(zero_bits));

    auto cb = *ClickCommandBuffer;
    const auto &compute = Pipelines->BoxSelect;
    const uint32_t group_count_x = (box_max.x - box_min.x + 15) / 16;
    const uint32_t group_count_y = (box_max.y - box_min.y + 15) / 16;
    RunSelectionCompute(
        cb, Vk.Queue, *TransferFence, Vk.Device, compute,
        BoxSelectPushConstants{
            .BoxMin = box_min,
            .BoxMax = box_max,
            .ObjectCount = object_count,
            .HeadImageIndex = SelectionHandles->HeadImage,
            .SelectionNodesIndex = Buffers->SelectionNodeBuffer.Slot,
            .BoxResultIndex = SelectionHandles->BoxResult,
        },
        [group_count_x, group_count_y](vk::CommandBuffer dispatch_cb) {
            dispatch_cb.dispatch(group_count_x, group_count_y, 1);
        }
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

    if (SelectionMode == SelectionMode::Box && InteractionMode == InteractionMode::Edit) {
        const auto mouse_pos = ToGlm(GetMousePos());
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            BoxSelectStart = mouse_pos;
            BoxSelectEnd = mouse_pos;
        } else if (IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart.has_value()) {
            BoxSelectEnd = mouse_pos;
            static constexpr float drag_threshold{2};
            if (const auto box_px = ComputeBoxSelectPixels(*BoxSelectStart, *BoxSelectEnd, ToGlm(GetCursorScreenPos()), Extent, drag_threshold)) {
                const auto &[box_min_px, box_max_px] = *box_px;

                for (const auto mesh_entity : R.view<MeshSelection>()) {
                    R.patch<MeshSelection>(mesh_entity, [](auto &s) { s.Handles.clear(); s.ActiveHandle = {}; });
                }

                Timer timer{"BoxSelectElements (all)"};
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

                auto results = RunBoxSelectElements(ranges, EditMode, box_min_px, box_max_px);
                for (size_t i = 0; i < ranges.size(); ++i) {
                    const auto mesh_entity = ranges[i].MeshEntity;
                    R.patch<MeshSelection>(mesh_entity, [&](auto &s) {
                        s.Element = EditMode;
                        s.Handles = i < results.size() ? std::move(results[i]) : std::vector<uint32_t>{};
                        s.ActiveHandle = {};
                    });
                    UpdateRenderBuffers(mesh_entity);
                }
                InvalidateCommandBuffer();
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart.has_value()) {
            BoxSelectStart.reset();
            BoxSelectEnd.reset();
        }
        if (BoxSelectStart.has_value()) return;
    }

    if (SelectionMode == SelectionMode::Box && InteractionMode == InteractionMode::Object) {
        const auto mouse_pos = ToGlm(GetMousePos());
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            BoxSelectStart = mouse_pos;
            BoxSelectEnd = mouse_pos;
        } else if (IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart.has_value()) {
            BoxSelectEnd = mouse_pos;
            static constexpr float drag_threshold{2};
            if (const auto box_px = ComputeBoxSelectPixels(*BoxSelectStart, *BoxSelectEnd, ToGlm(GetCursorScreenPos()), Extent, drag_threshold)) {
                const auto &[box_min_px, box_max_px] = *box_px;
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
    const auto mouse_pos_rel = GetMousePos() - GetCursorScreenPos();
    // Flip y-coordinate: ImGui uses top-left origin, but Vulkan gl_FragCoord uses bottom-left origin
    const glm::uvec2 mouse_px{uint32_t(mouse_pos_rel.x), uint32_t(Extent.height - mouse_pos_rel.y)};
    if (InteractionMode == InteractionMode::Edit) {
        if (EditMode != Element::None) {
            const auto hit_entities = RunClickSelect(mouse_px);
            const auto hit_it = find_if(hit_entities, [&](auto e) { return R.all_of<Selected>(e); });
            const bool toggle = IsKeyDown(ImGuiMod_Shift) || IsKeyDown(ImGuiMod_Ctrl) || IsKeyDown(ImGuiMod_Super);
            if (!toggle) {
                for (const auto [e, selection] : R.view<MeshSelection>().each()) {
                    if (selection.Handles.empty()) continue;
                    R.patch<MeshSelection>(e, [](auto &s) { s.Handles.clear(); s.Element = Element::None; });
                    UpdateRenderBuffers(e);
                }
            }
            if (hit_it != hit_entities.end()) {
                const auto mesh_entity = R.get<MeshInstance>(*hit_it).MeshEntity;
                if (const auto element = RunClickSelectElement(mesh_entity, EditMode, mouse_px)) {
                    SelectElement(mesh_entity, *element, toggle);
                }
            }
        }
    } else if (InteractionMode == InteractionMode::Object) {
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
                InvalidateCommandBuffer();
            }
        } else if (intersected != entt::null || !IsKeyDown(ImGuiMod_Shift)) {
            Select(intersected);
        }
    } else if (InteractionMode == InteractionMode::Excite) {
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
    if (auto descriptor_updates = Buffers->Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        const Timer timer{"RenderViewport->UpdateBufferDescriptorSets"};
        Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
        Buffers->Ctx.ClearDeferredDescriptorUpdates();
    }
#ifdef MVK_FORCE_STAGED_TRANSFERS
    if (auto deferred_copies = Buffers->Ctx.TakeDeferredCopies(); !deferred_copies.empty()) {
        const Timer timer{"RenderViewport->FlushStagedBufferCopies"};
        for (const auto &[buffers, ranges] : deferred_copies) {
            auto regions = ranges | transform([](const auto &r) {
                               const auto &[start, end] = r;
                               return vk::BufferCopy{start, start, end - start};
                           }) |
                to<std::vector>();
            TransferCommandBuffer->copyBuffer(buffers.Src, buffers.Dst, regions);
        }
    }
#endif
    const auto content_region = ToGlm(GetContentRegionAvail());
    const bool extent_changed = Extent.width != content_region.x || Extent.height != content_region.y;
    if (!extent_changed && !CommandBufferDirty && !NeedsRender) return false;

    const Timer timer{"RenderViewport"};
    if (extent_changed) {
        Extent = ToExtent(content_region);
        UpdateSceneUBO();
        Vk.Device.waitIdle(); // Ensure GPU work is done before destroying old pipeline resources
        Pipelines->SetExtent(Extent);
        {
            const Timer timer{"RenderViewport->UpdateSelectionDescriptorSets"};
            const auto head_image_info = vk::DescriptorImageInfo{
                nullptr,
                *Pipelines->SelectionFragment.Resources->HeadImage.View,
                vk::ImageLayout::eGeneral
            };
            const vk::DescriptorBufferInfo selection_counter{*Buffers->SelectionCounterBuffer, 0, sizeof(SelectionCounters)};
            const vk::DescriptorBufferInfo click_result{*Buffers->ClickResultBuffer, 0, sizeof(ClickResult)};
            const vk::DescriptorBufferInfo click_element_result{*Buffers->ClickElementResultBuffer, 0, SceneBuffers::MaxClickElementGroups * sizeof(ClickElementCandidate)};
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
        CommandBufferDirty = true;
    }

    // TransferCommandBuffer is kept recording between frames by BufferContext and our end-of-frame begin().
    // Ensure buffer writes (staging copies) are visible to shader reads.
    const vk::MemoryBarrier buffer_barrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead};
    TransferCommandBuffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader,
        {}, buffer_barrier, {}, {}
    );
    TransferCommandBuffer->end();

    if (CommandBufferDirty) {
        RecordRenderCommandBuffer();
        CommandBufferDirty = false;
    }

    // Submit transfer and render commands together
    const std::array command_buffers{*TransferCommandBuffer, *RenderCommandBuffer};
    vk::SubmitInfo submit;
    submit.setCommandBuffers(command_buffers);
    Vk.Queue.submit(submit, *RenderFence);
    {
        const Timer timer{"RenderViewport->WaitForGPU"};
        WaitFor(*RenderFence, Vk.Device);
    }

    Buffers->Ctx.ReclaimRetiredBuffers();
    // Leave TransferCommandBuffer recording for next frame's staging.
    TransferCommandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

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

        auto root_selected = selected_view | filter([&](auto e) { return !is_parent_selected(e); });
        const auto root_count = distance(root_selected);

        const auto active_entity = FindActiveEntity(R);
        const auto active_transform = active_entity != entt::null ? GetTransform(R, active_entity) : Transform{};
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
        if (Camera.Tick()) UpdateSceneUBO();
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
                const bool wireframe_xray = ViewportShading == ViewportShadingMode::Wireframe;
                if (interaction_mode == int(InteractionMode::Edit)) {
                    Checkbox("X-ray selection", &SelectionXRay);
                }
                if (wireframe_xray) {
                    SameLine();
                    TextDisabled("(wireframe)");
                }
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

            bool smooth_shading_changed = false;
            if (ViewportShading == ViewportShadingMode::Solid) {
                smooth_shading_changed = Checkbox("Smooth shading", &SmoothShading);
            }

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
                InvalidateCommandBuffer();
            }
            if (smooth_shading_changed) {
                InvalidateCommandBuffer();
            }
            if (ViewportShading == ViewportShadingMode::Wireframe) {
                if (ColorEdit3("Edge color", &EdgeColor.x)) UpdateEdgeColors();
            }
            {
                SeparatorText("Active/Selected");
                bool color_changed = ColorEdit3("Active color", &Colors.Active[0]);
                color_changed |= ColorEdit3("Selected color", &Colors.Selected[0]);
                if (color_changed) {
                    UpdateSceneUBO();
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
