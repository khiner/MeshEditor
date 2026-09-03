#pragma once

#include "gpu/VisibilityShadingPushConstants.h"
#include "metal/Bindless.h"
#include "metal/PassChain.h"
#include "metal/Shader.h"
#include "render/GpuBuffers.h"
#include "render/Pipelines.h"

#include <cassert>
#include <print>

namespace encode {
// Returns visibility-decode inputs only when the raster and visible-list generations match.
inline VisibilityShadingPushConstants VisibilityDecodePc(const GpuBuffers &buffers) {
    if (buffers.VisibilityIdGeneration != buffers.MeshletVisibleGeneration) {
        static bool reported = false;
        if (!reported) {
            reported = true;
            std::println(
                stderr, "Visibility ids were rasterized against generation {} and decode {}, so a cull ran between them",
                buffers.VisibilityIdGeneration, buffers.MeshletVisibleGeneration
            );
        }
        assert(false);
    }
    return {
        .PrimitiveSlot = buffers.Primitives.Buffer.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Buffer.Slot,
        .MeshletSlot = buffers.Meshlets.Buffer.Slot,
        .MeshletTriangleSlot = buffers.MeshletTriangleIds.Buffer.Slot,
        .MeshletLocalTriangleSlot = buffers.MeshletLocalTriangles.Buffer.Slot,
        .MeshletVertexSlot = buffers.MeshletVertexCorners.Buffer.Slot,
        .VisibleMeshletSlot = buffers.VisibleMeshlets.Slot,
        .Phase2VisibleMeshletSlot = buffers.MeshletPhase2Visible.Slot,
    };
}

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

// Dispatches one tiled stage after an explicit barrier for bindless-buffer dependencies.
inline void DispatchTiledPass(
    MTL::ComputeCommandEncoder *encoder, const mtl::ComputePipeline &pipeline, const mtl::BindlessSet &slots,
    const GpuBuffers &buffers, auto pc, size_t groups, uint32_t first_tile
) {
    if (groups == 0) return;
    BindCompute(encoder, pipeline, slots, buffers);
    pc.FirstTile = first_tile;
    SetPushConstants(encoder, pc);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BlockScan, 0);
    encoder->dispatchThreadgroups(MTL::Size(groups, 1, 1), ThreadgroupSize::Linear256);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
}

} // namespace encode
