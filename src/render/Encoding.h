#pragma once

#include "gpu/ExtrasLinePushConstants.h"
#include "gpu/OverlayDispatch.h"
#include "gpu/OverlayMeshPushConstants.h"
#include "metal/Bindless.h"
#include "metal/PassChain.h"
#include "metal/Shader.h"
#include "render/DrawState.h"
#include "render/GpuBuffers.h"

#include <span>

// Shared draw/dispatch bindings; render stages receive the same buffers.
namespace encode {
inline void BindScene(MTL::RenderCommandEncoder *encoder, const mtl::BindlessSet &slots, const GpuBuffers &buffers, uint32_t view_offset = 0) {
    slots.UseResources(encoder);
    auto *table = slots.Table();
    auto *view = *buffers.SceneViewUBO;
    auto *theme = *buffers.ViewportThemeUBO;
    auto *workspace = *buffers.WorkspaceLightsUBO;
    encoder->setVertexBuffer(table, 0, BufferIndex_Bindless);
    encoder->setVertexBuffer(view, view_offset, BufferIndex_SceneView);
    encoder->setVertexBuffer(theme, 0, BufferIndex_ViewportTheme);
    encoder->setVertexBuffer(workspace, 0, BufferIndex_WorkspaceLights);
    encoder->setMeshBuffer(table, 0, BufferIndex_Bindless);
    encoder->setMeshBuffer(view, view_offset, BufferIndex_SceneView);
    encoder->setMeshBuffer(theme, 0, BufferIndex_ViewportTheme);
    encoder->setMeshBuffer(workspace, 0, BufferIndex_WorkspaceLights);
    encoder->setFragmentBuffer(table, 0, BufferIndex_Bindless);
    encoder->setFragmentBuffer(view, view_offset, BufferIndex_SceneView);
    encoder->setFragmentBuffer(theme, 0, BufferIndex_ViewportTheme);
    encoder->setFragmentBuffer(workspace, 0, BufferIndex_WorkspaceLights);
}

inline void BindScene(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const GpuBuffers &buffers, uint32_t view_offset = 0) {
    slots.UseResources(encoder);
    encoder->setBuffer(slots.Table(), 0, BufferIndex_Bindless);
    encoder->setBuffer(*buffers.SceneViewUBO, view_offset, BufferIndex_SceneView);
    encoder->setBuffer(*buffers.ViewportThemeUBO, 0, BufferIndex_ViewportTheme);
    encoder->setBuffer(*buffers.WorkspaceLightsUBO, 0, BufferIndex_WorkspaceLights);
}

template<typename T>
void SetPushConstants(MTL::RenderCommandEncoder *encoder, const T &pc) {
    encoder->setVertexBytes(&pc, sizeof(T), BufferIndex_PushConstants);
    encoder->setMeshBytes(&pc, sizeof(T), BufferIndex_PushConstants);
    encoder->setFragmentBytes(&pc, sizeof(T), BufferIndex_PushConstants);
}

template<typename T>
void SetMeshPushConstants(MTL::RenderCommandEncoder *encoder, const T &pc) {
    encoder->setMeshBytes(&pc, sizeof(T), BufferIndex_PushConstants);
}

template<typename T>
void SetPushConstants(MTL::ComputeCommandEncoder *encoder, const T &pc) {
    encoder->setBytes(&pc, sizeof(T), BufferIndex_PushConstants);
}

// One dispatch per command over a batch's element list, instances in the grid's second dimension.
// A batch's command holds `indices_per_element` indices for each element it draws.
inline void DispatchMeshBatch(
    MTL::RenderCommandEncoder *encoder, const DrawListBuilder &list, const DrawBatchInfo &batch,
    uint32_t indices_per_element, uint32_t elements_per_group, uint32_t threads_per_group
) {
    const auto first_command = batch.IndirectOffset / IndirectCommandStride;
    for (uint32_t d = 0; d < batch.DrawCount; ++d) {
        const auto &cmd = list.IndirectCommands[first_command + d];
        const auto element_count = cmd.indexCount / indices_per_element;
        SetMeshPushConstants(encoder, OverlayMeshPushConstants{
            .DrawDataIndex = batch.DrawDataSlotOffset + cmd.baseInstance,
            .ElementCount = element_count,
        });
        encoder->drawMeshThreadgroups(
            MTL::Size((element_count + elements_per_group - 1) / elements_per_group, cmd.instanceCount, 1),
            MTL::Size(1, 1, 1), MTL::Size(threads_per_group, 1, 1)
        );
    }
}

// One threadgroup per instance, each emitting that instance's whole primitive.
inline void DispatchInstancedMeshBatch(
    MTL::RenderCommandEncoder *encoder, const DrawListBuilder &list, const DrawBatchInfo &batch,
    uint32_t vertices_per_instance
) {
    const auto first_command = batch.IndirectOffset / IndirectCommandStride;
    for (uint32_t d = 0; d < batch.DrawCount; ++d) {
        const auto &cmd = list.IndirectCommands[first_command + d];
        SetMeshPushConstants(encoder, OverlayMeshPushConstants{
            .DrawDataIndex = batch.DrawDataSlotOffset + cmd.baseInstance,
            .ElementCount = vertices_per_instance,
        });
        encoder->drawMeshThreadgroups(MTL::Size(cmd.instanceCount, 1, 1), MTL::Size(1, 1, 1), MTL::Size(vertices_per_instance, 1, 1));
    }
}

// One dispatch per extras line source, chunked line groups.
inline void DispatchExtrasLines(MTL::RenderCommandEncoder *encoder, std::span<const ExtrasLinePushConstants> extras_lines) {
    constexpr auto lines = uint32_t(OverlayDispatch::LineGroupLines);
    for (const auto &extras_line : extras_lines) {
        SetMeshPushConstants(encoder, extras_line);
        encoder->drawMeshThreadgroups(MTL::Size((extras_line.LineCount + lines - 1) / lines, 1, 1), MTL::Size(1, 1, 1), MTL::Size(lines * 2, 1, 1));
    }
}

inline void SetFullViewport(MTL::RenderCommandEncoder *encoder, mtl::Extent2D extent) {
    encoder->setViewport({0.0, 0.0, double(extent.Width), double(extent.Height), 0.0, 1.0});
    encoder->setScissorRect({0, 0, extent.Width, extent.Height});
}

inline MTL::RenderCommandEncoder *BeginScenePass(
    mtl::PassChain &chain, MTL::RenderPassDescriptor *pass, std::string_view name, std::initializer_list<mtl::Barrier> barriers,
    mtl::Extent2D extent, const mtl::BindlessSet &slots, const GpuBuffers &buffers, uint32_t view_offset = 0
) {
    auto *encoder = chain.BeginRender(pass, name, barriers);
    SetFullViewport(encoder, extent);
    BindScene(encoder, slots, buffers, view_offset);
    return encoder;
}

inline void BindCompute(
    MTL::ComputeCommandEncoder *encoder, const mtl::ComputePipeline &pipeline,
    const mtl::BindlessSet &slots, const GpuBuffers &buffers, uint32_t view_offset = 0
) {
    encoder->setComputePipelineState(pipeline.State());
    BindScene(encoder, slots, buffers, view_offset);
}

// Metal has no indexed-indirect multi-draw, so walk the command buffer.
inline void DrawIndexedIndirect(
    MTL::RenderCommandEncoder *encoder, MTL::PrimitiveType primitive, MTL::Buffer *index_buffer,
    MTL::Buffer *indirect, uint64_t indirect_offset, uint32_t draw_count
) {
    for (uint32_t i = 0; i < draw_count; ++i) {
        encoder->drawIndexedPrimitives(
            primitive, MTL::IndexTypeUInt32, index_buffer, 0,
            indirect, indirect_offset + uint64_t(i) * IndirectCommandStride
        );
    }
}
} // namespace encode
