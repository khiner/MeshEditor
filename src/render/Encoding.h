#pragma once

#include "metal/Shader.h"
#include "render/GpuBuffers.h"
#include "viewport/Viewport.h"

#include <Metal/MTLComputeCommandEncoder.hpp>
#include <print>

namespace encode {
// Decode primitive IDs against the current meshlet list.
inline VisibilityShadingPushConstants MeshletDecodePc(const GpuBuffers &buffers) {
    return {
        .PrimitiveSlot = buffers.Render->Primitives.Buffer.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Slot,
        .MeshletSlot = buffers.Render->Meshlets.Buffer.Slot,
        .Payloads = buffers.Render->PayloadSlots(),
        .VisibleMeshletSlot = buffers.SceneCull.Visible.Slot,
    };
}

// Returns visibility-decode inputs only when the raster and visible-list generations match.
inline VisibilityShadingPushConstants VisibilityDecodePc(const GpuBuffers &buffers) {
    if (buffers.VisibilityGeneration != buffers.MeshletVisibleGeneration) {
        static bool reported = false;
        if (!reported) {
            reported = true;
            std::println(
                stderr, "Visibility ids were rasterized against generation {} and decode {}, so a cull ran between them",
                buffers.VisibilityGeneration, buffers.MeshletVisibleGeneration
            );
        }
        assert(false);
    }
    return MeshletDecodePc(buffers);
}

// Binds the bindless table, the document's theme, and the viewport's view uniforms at `view_offset` and workspace lights.
inline void BindScene(MTL::RenderCommandEncoder *encoder, const mtl::BindlessSet &slots, const GpuBuffers &buffers, const Viewport &viewport, uint32_t view_offset = 0) {
    slots.UseResources(encoder);
    const auto *table = slots.Table();
    const auto *view = *viewport.SceneViewUBO;
    const auto *theme = *buffers.ViewportThemeUBO;
    const auto *workspace = *viewport.WorkspaceLightsUBO;
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

inline void BindScene(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const GpuBuffers &buffers, const Viewport &viewport, uint32_t view_offset = 0) {
    slots.UseResources(encoder);
    encoder->setBuffer(slots.Table(), 0, BufferIndex_Bindless);
    encoder->setBuffer(*viewport.SceneViewUBO, view_offset, BufferIndex_SceneView);
    encoder->setBuffer(*buffers.ViewportThemeUBO, 0, BufferIndex_ViewportTheme);
    encoder->setBuffer(*viewport.WorkspaceLightsUBO, 0, BufferIndex_WorkspaceLights);
}

// Binds the bindless table alone, for document passes that read no view.
inline void BindBindless(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots) {
    slots.UseResources(encoder);
    encoder->setBuffer(slots.Table(), 0, BufferIndex_Bindless);
}

void SetPushConstants(MTL::RenderCommandEncoder *encoder, const auto &pc) {
    encoder->setVertexBytes(&pc, sizeof(pc), BufferIndex_PushConstants);
    encoder->setMeshBytes(&pc, sizeof(pc), BufferIndex_PushConstants);
    encoder->setFragmentBytes(&pc, sizeof(pc), BufferIndex_PushConstants);
}

void SetMeshPushConstants(MTL::RenderCommandEncoder *encoder, const auto &pc) {
    encoder->setMeshBytes(&pc, sizeof(pc), BufferIndex_PushConstants);
}

void SetPushConstants(MTL::ComputeCommandEncoder *encoder, const auto &pc) {
    encoder->setBytes(&pc, sizeof(pc), BufferIndex_PushConstants);
}

inline void SetFullViewport(MTL::RenderCommandEncoder *encoder, mtl::Extent2D extent) {
    encoder->setViewport({0.0, 0.0, double(extent.Width), double(extent.Height), 0.0, 1.0});
    encoder->setScissorRect({0, 0, extent.Width, extent.Height});
}

inline MTL::RenderCommandEncoder *BeginScenePass(
    mtl::PassChain &chain, MTL::RenderPassDescriptor *pass, std::string_view name, std::initializer_list<mtl::Barrier> barriers,
    mtl::Extent2D extent, const mtl::BindlessSet &slots, const GpuBuffers &buffers, const Viewport &viewport, uint32_t view_offset = 0
) {
    auto *encoder = chain.BeginRender(pass, name, barriers);
    SetFullViewport(encoder, extent);
    BindScene(encoder, slots, buffers, viewport, view_offset);
    return encoder;
}

inline void BindCompute(
    MTL::ComputeCommandEncoder *encoder, const mtl::ComputePipeline &pipeline,
    const mtl::BindlessSet &slots, const GpuBuffers &buffers, const Viewport &viewport, uint32_t view_offset = 0
) {
    encoder->setComputePipelineState(pipeline.State());
    BindScene(encoder, slots, buffers, viewport, view_offset);
}

} // namespace encode
