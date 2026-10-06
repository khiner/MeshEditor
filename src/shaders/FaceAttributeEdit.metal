#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
#include "gpu/FaceAttributeEditPushConstants.h"

// Each face exclusively owns its corners. Permute in place without an attribute copy.
template<typename T>
void PermuteCorners(device T *values, device const BindlessSet &b, ElementAttributeRef at, uint2 range, uint operation) {
    const auto index = [&](uint h) { return ElementAttributeIndex(b, at, h); };
    if (operation == 2u) {
        for (uint a = range.x, z = range.y - 1u; a < z; ++a, --z) {
            const T saved = values[index(a)];
            values[index(a)] = values[index(z)];
            values[index(z)] = saved;
        }
    } else if (operation == 1u) {
        const T saved = values[index(range.x)];
        for (uint h = range.x; h + 1u < range.y; ++h) values[index(h)] = values[index(h + 1u)];
        values[index(range.y - 1u)] = saved;
    } else {
        const T saved = values[index(range.y - 1u)];
        for (uint h = range.y - 1u; h > range.x; --h) values[index(h)] = values[index(h - 1u)];
        values[index(range.x)] = saved;
    }
}

kernel void EditFaceUvs(uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant FaceAttributeEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.Count) return;
    const uint face = ElementWorkDomain{b, pc.Faces, 0u}.Handle(i);
    const auto range = ConnectivityView{b, pc.Connectivity, pc.FaceCount}.FaceHalfedges(face);
    PermuteCorners(BindlessBufferMutable(packed_float2, b.CornerUvBuffer, pc.Attribute.ValuesSlot), b, pc.Attribute, range, pc.Operation);
    // Zero tangents select the renderer's derivative frame using the material's UV set.
    if (pc.Tangents.ValuesSlot != InvalidSlot) {
        auto tangents = BindlessBufferMutable(packed_float4, b.CornerTangentBuffer, pc.Tangents.ValuesSlot);
        for (uint h = range.x; h < range.y; ++h) tangents[ElementAttributeIndex(b, pc.Tangents, h)] = packed_float4(0.f);
    }
}

kernel void EditFaceColors(uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant FaceAttributeEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.Count) return;
    const uint face = ElementWorkDomain{b, pc.Faces, 0u}.Handle(i);
    const auto range = ConnectivityView{b, pc.Connectivity, pc.FaceCount}.FaceHalfedges(face);
    PermuteCorners(BindlessBufferMutable(packed_float4, b.CornerColorBuffer, pc.Attribute.ValuesSlot), b, pc.Attribute, range, pc.Operation);
}
