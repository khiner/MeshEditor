#pragma once

#include "metal/PassChain.h"
#include "metal/Shader.h"
#include "render/GpuBuffers.h"
#include "viewport/Viewport.h"

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

// Binds the bindless table, the document's theme, and the viewport's view uniforms at `view_offset` and workspace lights, for every stage of the command's encoders.
template<typename E> void BindScene(mtl::Encoder<E> encoder, const mtl::BindlessSet &slots, const GpuBuffers &buffers, const Viewport &viewport, uint32_t view_offset = 0) {
    auto &command = *encoder.Cmd;
    command.SetBuffer(slots.Table(), 0, BufferIndex_Bindless);
    command.SetBuffer(*viewport.SceneViewUBO, view_offset, BufferIndex_SceneView);
    command.SetBuffer(*buffers.ViewportThemeUBO, 0, BufferIndex_ViewportTheme);
    command.SetBuffer(*viewport.WorkspaceLightsUBO, 0, BufferIndex_WorkspaceLights);
}

// Binds the bindless table alone, for document passes that read no view.
template<typename E> void BindBindless(mtl::Encoder<E> encoder, const mtl::BindlessSet &slots) {
    encoder.Cmd->SetBuffer(slots.Table(), 0, BufferIndex_Bindless);
}

template<typename E> void SetPushConstants(mtl::Encoder<E> encoder, const auto &pc) {
    encoder.Cmd->SetBytes(pc, BufferIndex_PushConstants);
}

void SetMeshPushConstants(mtl::RenderEncoder encoder, const auto &pc) {
    encoder.Cmd->SetGeometryBytes(pc, BufferIndex_PushConstants);
}

void SetFragmentPushConstants(mtl::RenderEncoder encoder, const auto &pc) {
    encoder.Cmd->SetFragmentBytes(pc, BufferIndex_PushConstants);
}

inline void SetFullViewport(mtl::RenderEncoder encoder, mtl::Extent2D extent) {
    encoder->setViewport({0.0, 0.0, double(extent.Width), double(extent.Height), 0.0, 1.0});
    encoder->setScissorRect({0, 0, extent.Width, extent.Height});
}

inline mtl::RenderEncoder BeginScenePass(
    mtl::PassChain &chain, const MTL4::RenderPassDescriptor *pass, std::string_view name, std::initializer_list<mtl::Barrier> barriers,
    mtl::Extent2D extent, const mtl::BindlessSet &slots, const GpuBuffers &buffers, const Viewport &viewport, uint32_t view_offset = 0
) {
    const auto encoder = chain.BeginRender(pass, name, barriers);
    SetFullViewport(encoder, extent);
    BindScene(encoder, slots, buffers, viewport, view_offset);
    return encoder;
}

inline void BindCompute(
    mtl::ComputeEncoder encoder, const mtl::ComputePipeline &pipeline,
    const mtl::BindlessSet &slots, const GpuBuffers &buffers, const Viewport &viewport, uint32_t view_offset = 0
) {
    pipeline.Bind(encoder);
    BindScene(encoder, slots, buffers, viewport, view_offset);
}

} // namespace encode
