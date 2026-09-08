#pragma once

#include "gpu/BindlessBindings.h"
#include "metal/Bindless.h"
#include "metal/Shader.h"

#include <Metal/MTLComputeCommandEncoder.hpp>

namespace mesh_compute {
// Each stage publishes its bindless-buffer writes before the next stage reads them.
inline void DispatchTiledPass(
    MTL::ComputeCommandEncoder *encoder, const mtl::ComputePipeline &pipeline, const mtl::BindlessSet &slots,
    auto pc, size_t groups, uint32_t first_tile
) {
    if (groups == 0) return;
    encoder->setComputePipelineState(pipeline.State());
    slots.UseResources(encoder);
    encoder->setBuffer(slots.Table(), 0, BufferIndex_Bindless);
    pc.FirstTile = first_tile;
    encoder->setBytes(&pc, sizeof(pc), BufferIndex_PushConstants);
    // Eight simd-group sums and the threadgroup total, padded to Metal's 16-byte granule.
    encoder->setThreadgroupMemoryLength(48, 0);
    encoder->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(256, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
}
} // namespace mesh_compute
