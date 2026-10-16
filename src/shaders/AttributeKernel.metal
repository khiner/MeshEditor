#include "AttributeNodes.metal"
#include "Bindless.metal"

inline uint4 ReadAttributeInput(device const BindlessSet &b, AttributeInput input, uint address, uint handle, uint ordinal) {
    switch (input.Source) {
        case AttributeSource::Constant: return uint4(input.Value[0], input.Value[1], input.Value[2], input.Value[3]);
        case AttributeSource::Named: return ReadNamedAttribute(b, input.Attribute, input.Type, handle);
        case AttributeSource::Positions: return uint4(as_type<uint3>(float3(BindlessBuffer(Vertex, b.VertexBuffer, input.Slot)[address].Position)), 0u);
        case AttributeSource::Vectors: return uint4(as_type<uint3>(float3(BindlessBuffer(packed_float3, b.Buffer, input.Slot)[address])), 0u);
        case AttributeSource::Ordinal: return uint4(ordinal, 0u, 0u, 0u);
    }
    return uint4(0u);
}

// Thread i reads slot i % 256 of the record's block i / 256, and a live element there writes `evaluate` of its inputs to the output.
template<typename Evaluate>
inline void RunAttributeNode(uint i, device const BindlessSet &b, constant AttributeNodePushConstants &pc, Evaluate evaluate) {
    const uint k = i / MeshElementBlockSize, lane = i % MeshElementBlockSize;
    if (k >= pc.BlockCount) return;
    device const uint *blocks = BindlessBuffer(uint, b.Buffer, pc.Blocks.Slot) + pc.Blocks.Offset;
    const uint block = blocks[k];
    device const MeshElementBlock &membership = BindlessBuffer(MeshElementBlock, b.Buffer, pc.MembershipSlot)[block];
    const uint word = lane / 32u, bit = 1u << (lane % 32u);
    if (!(membership.Live[word] & bit)) return;
    uint ordinal = (k ? blocks[pc.BlockCount + k - 1u] : 0u) + popcount(membership.Live[word] & (bit - 1u));
    for (uint w = 0u; w < word; ++w) ordinal += popcount(membership.Live[w]);
    const uint address = block * MeshElementBlockSize + lane, handle = membership.Local * MeshElementBlockSize + lane;
    const uint4 value = evaluate(ReadAttributeInput(b, pc.A, address, handle, ordinal), ReadAttributeInput(b, pc.B, address, handle, ordinal));
    WriteNamedAttribute(b, pc.Output, pc.OutputType, handle, value);
}

// One entry point per node function.
// AttributeRead writes input A unchanged, for the nodes that read each element's position, normal or rank.
kernel void AttributeRead(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeNodePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    RunAttributeNode(i, b, pc, [&](uint4 a, uint4) { return a; });
}
kernel void AttributeVectorMath(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeNodePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    RunAttributeNode(i, b, pc, [&](uint4 a, uint4 v) { return VectorMathNode(a, v, VectorMathOp(pc.Operation)); });
}
kernel void AttributeFloatMath(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeNodePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    RunAttributeNode(i, b, pc, [&](uint4 a, uint4 v) { return FloatMathNode(a, v, FloatMathOp(pc.Operation)); });
}
kernel void AttributeCompare(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeNodePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    RunAttributeNode(i, b, pc, [&](uint4 a, uint4 v) { return CompareNode(a, pc.A.Type, v, pc.B.Type, pc.Component, CompareOp(pc.Operation), pc.Epsilon); });
}
kernel void AttributeBooleanMath(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant AttributeNodePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    RunAttributeNode(i, b, pc, [&](uint4 a, uint4 v) { return BooleanMathNode(a, v, BooleanMathOp(pc.Operation)); });
}

kernel void AttributeSelectionMasks(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant AttributeMaskPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.Words) return;
    const MeshElementBlock membership = BindlessBuffer(MeshElementBlock, b.Buffer, pc.MembershipSlot)[i / MeshElementBlockWords];
    const bool held = pc.Attribute.ValuesSlot != InvalidSlot && membership.Owner == pc.Owner;
    const uint value = held ? BindlessBuffer(uint, b.Buffer, pc.Attribute.ValuesSlot)[(pc.Attribute.Offset + membership.Local * MeshElementBlockSize) / 32u + i % MeshElementBlockWords] : 0u;
    BindlessBufferMutable(uint, b.Buffer, pc.Masks.Slot)[pc.Masks.Offset + i] = value;
}
