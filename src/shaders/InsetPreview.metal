#include "Bindless.metal"
#include "gpu/InsetPreviewPushConstants.h"
#include "gpu/InsetVertexBasis.h"
#include "gpu/Vertex.h"

kernel void InsetPreviewPositions(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant InsetPreviewPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.Count) return;
    const InsetVertexBasis basis = reinterpret_cast<device const InsetVertexBasis *>(BindlessBuffer(uint,bindless.Buffer,pc.Basis.Slot) + pc.Basis.Offset)[i];
    const float3 position = float3(basis.Base) + float3(basis.Width) * pc.Thickness + float3(basis.Depth) * pc.Depth;
    BindlessBufferMutable(Vertex,bindless.VertexBuffer,pc.VertexSlot)[basis.Handle].Position = packed_float3(position);
}
