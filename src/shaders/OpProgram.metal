#pragma once
// The evaluation point and the reads every op program shares.
#include "gpu/OpPoint.h"

inline float OpLoad(float, device const uint *words, uint index) { return as_type<float>(words[index]); }
template<int N> inline vec<float, N> OpLoad(vec<float, N>, device const uint *words, uint index) {
    vec<float, N> value;
    for (int c = 0; c < N; ++c) value[c] = as_type<float>(words[index * N + c]);
    return value;
}
inline int OpLoad(int, device const uint *words, uint index) { return as_type<int>(words[index]); }
inline bool OpLoad(bool, device const uint *words, uint index) { return (words[index / 32u] >> (index % 32u)) & 1u; }

// The value of the named attribute in the program's slot at an element, zero at InvalidOffset, or `fallback` when the mesh lacks it.
template<typename T> inline T OpElementAttribute(const thread OpPoint &p, uint slot, uint element, T fallback) {
    const OpAttributeRef a = p.Attributes[slot];
    if (a.Values.ValuesSlot == InvalidSlot) return fallback;
    return element == InvalidOffset ? T(0) : OpLoad(fallback, reinterpret_cast<device const uint *>(p.Buffers[a.Values.ValuesSlot]), a.Values.Offset + element);
}
// The element of the point a read of one value takes: its own element or its first corner, a corner's vertex or its first corner's vertex, or a corner's face, which is InvalidOffset for a wire corner's.
inline uint OpFirstElement(const thread OpPoint &p, OpElement element) {
    if (element == OpElement::Face) return p.Face;
    return element == OpElement::Vertex || element == OpElement::Vertices ? p.Vertex : p.Elements[0];
}

// The named attribute in the program's slot at the point, or `fallback` when the mesh lacks it.
// A read of a weighted point's corners or their vertices weights their values as the point weights its corners, and any other read takes one element's value.
template<typename T> inline T OpAttribute(const thread OpPoint &p, uint slot, T fallback) {
    const OpAttributeRef a = p.Attributes[slot];
    if (a.Values.ValuesSlot == InvalidSlot) return fallback;
    if (a.Element != OpElement::Corners && a.Element != OpElement::Vertices) return OpElementAttribute(p, slot, OpFirstElement(p, a.Element), fallback);
    device const uint *words = reinterpret_cast<device const uint *>(p.Buffers[a.Values.ValuesSlot]);
    const thread uint *elements = a.Element == OpElement::Vertices ? p.Vertices : p.Elements;
    T value = p.Weights[0] * OpLoad(fallback, words, a.Values.Offset + elements[0]);
    for (uint e = 1u; e < 3u; ++e) value += p.Weights[e] * OpLoad(fallback, words, a.Values.Offset + elements[e]);
    return value;
}
// An integer or boolean reads one element, the first corner's at a weighted point.
inline int OpAttribute(const thread OpPoint &p, uint slot, int fallback) { return OpElementAttribute(p, slot, OpFirstElement(p, p.Attributes[slot].Element), fallback); }
inline bool OpAttribute(const thread OpPoint &p, uint slot, bool fallback) { return OpElementAttribute(p, slot, OpFirstElement(p, p.Attributes[slot].Element), fallback); }
// The named attribute at a texel's corner of the largest weight, its vertex or its face.
template<typename T> inline T OpLargestAttribute(const thread OpPoint &p, uint slot, T fallback) {
    const OpElement element = p.Attributes[slot].Element;
    const uint corner = p.Weights[1] > p.Weights[0] ? (p.Weights[2] > p.Weights[1] ? 2u : 1u) : (p.Weights[2] > p.Weights[0] ? 2u : 0u);
    return OpElementAttribute(p, slot, element == OpElement::Face ? p.Face : element == OpElement::Vertices ? p.Vertices[corner] : p.Elements[corner], fallback);
}

// Texture coordinate set `index`, and zero for a set the point lacks.
inline float2 OpTexcoord(const thread OpPoint &p, int index) { return index >= 0 && index < 4 ? p.Texcoords[index] : float2(0.0f); }
// Vertex color set `index`, and zero for a set the point lacks.
inline float4 OpColor(const thread OpPoint &p, int index) { return index == 0 ? p.Color : float4(0.0f); }
// A surface point's coordinates in texture coordinate set `index` with V down, and their derivatives along x and y, all zero for a set the point lacks.
inline float2 OpSurfaceTexcoord(const thread OpPoint &p, int index, uint part) { return index >= 0 && index < 4 ? p.SurfaceTexcoords[part * 4u + uint(index)] : float2(0.0f); }
