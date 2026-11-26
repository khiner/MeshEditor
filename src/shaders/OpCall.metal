#pragma once
// The evaluation point the drivers fill at an element, and the table they call programs through.
#include "ConnectivityRead.metal"
#include "CornerNormalOffset.metal"
#include "gpu/CornerClassMode.h"
#include "gpu/NormalSector.h"
#include "gpu/OpDriver.h"

// Every op program has this signature, and a driver calls program `Program` through its table.
using OpProgramFunction = void(const thread OpPoint &, device const uint *, thread uint *);
using OpPrograms = visible_function_table<OpProgramFunction>;

// The rank among its record's live elements of the live element in slot `lane` of block k of the record's blocks.
inline uint ElementOrdinal(device const BindlessSet &b, DomainBlocks blocks, uint k, uint lane) {
    device const uint *list = BindlessBuffer(uint, b.Buffer, blocks.Blocks.Slot) + blocks.Blocks.Offset;
    device const MeshElementBlock &membership = BindlessBuffer(MeshElementBlock, b.Buffer, blocks.MembershipSlot)[list[k]];
    const uint word = lane / 32u;
    uint ordinal = (k ? list[blocks.Count + k - 1u] : 0u) + popcount(membership.Live[word] & ((1u << (lane % 32u)) - 1u));
    for (uint w = 0u; w < word; ++w) ordinal += popcount(membership.Live[w]);
    return ordinal;
}
// The rank among its record's live elements of the live element at an arena address.
inline uint AddressOrdinal(device const BindlessSet &b, DomainBlocks blocks, uint address) {
    const uint k = LowerBound(BindlessBuffer(uint, b.Buffer, blocks.Blocks.Slot) + blocks.Blocks.Offset, blocks.Count, address / MeshElementBlockSize);
    return ElementOrdinal(b, blocks, k, address % MeshElementBlockSize);
}

// An element of a record's domain: whether it is live, its arena address, its record-relative handle, and its rank among the record's live elements.
struct DomainElement {
    bool Live;
    uint Address, Handle, Ordinal;
};
// The element a dispatch over a record's domain blocks visits at thread i, slot i % 256 of block i / 256.
inline DomainElement ElementAt(device const BindlessSet &b, DomainBlocks blocks, uint i) {
    const uint k = i / MeshElementBlockSize, lane = i % MeshElementBlockSize;
    if (k >= blocks.Count) return {false, 0u, 0u, 0u};
    const uint block = BindlessBuffer(uint, b.Buffer, blocks.Blocks.Slot)[blocks.Blocks.Offset + k];
    device const MeshElementBlock &membership = BindlessBuffer(MeshElementBlock, b.Buffer, blocks.MembershipSlot)[block];
    if (!SlotLive(membership, lane)) return {false, 0u, 0u, 0u};
    return {true, block * MeshElementBlockSize + lane, membership.Local * MeshElementBlockSize + lane, ElementOrdinal(b, blocks, k, lane)};
}

// The words from a word offset into a buffer slot, or null for no slot.
inline device const uint *OpWords(device const BindlessSet &b, SlotOffset words) {
    return words.Slot == InvalidSlot ? nullptr : BindlessBuffer(uint, b.Buffer, words.Slot) + words.Offset;
}
// Starts a point of the program's call at the timeline's frame and time, with the program's image table where it samples images.
inline void OpStart(thread OpPoint &p, device const BindlessSet &b, OpProgramRef program, float frame, float time) {
    p.Frame = frame;
    p.Time = time;
    if (program.Needs & OpNeedsImages) p.Images = reinterpret_cast<device const OpImageTexture *>(OpWords(b, program.Images));
}
// Calls the program at the point, which writes its output words to `out`.
inline void OpRun(OpPrograms programs, device const BindlessSet &b, OpProgramRef program, const thread OpPoint &p, thread uint *out) { programs[program.Index](p, OpWords(b, program.Values), out); }
// Writes texel i of `output` as `stride` floats: the first `components` words of a program's output, and zero after them.
inline void OpWriteTexel(device const BindlessSet &b, SlotOffset output, uint i, uint stride, const thread uint *out, uint components) {
    device float *texel = BindlessBufferMutable(float, b.Buffer, output.Slot) + output.Offset + i * stride;
    for (uint c = 0u; c < stride; ++c) texel[c] = c < components ? as_type<float>(out[c]) : 0.0f;
}

// A corner's shading normal as the viewport shades the unposed record: its face's normal on a flat face, and else its sector's or its vertex's, turned by its custom normal offset.
inline float3 OpCornerNormal(device const BindlessSet &b, OpGeometry g, ConnectivityView conn, uint corner, uint corner_vertex, uint face) {
    const bool mixed = g.ClassMode == uint(CornerClassMode::Mixed);
    const uint face_address = face == InvalidOffset ? InvalidOffset : conn.FaceAddress(face);
    const bool flat = face_address != InvalidOffset && (g.ClassMode == uint(CornerClassMode::UniformFace) || (mixed && BindlessBuffer(uchar, b.Buffer, g.FaceSharpness)[face_address] != 0u));
    const uint root = mixed && !flat ? CornerSectorRoot(b, g.CornerSectors, corner) : InvalidOffset;
    float3 normal;
    if (flat) normal = float3(BindlessBuffer(packed_float3, b.Buffer, g.FaceNormals)[face_address]);
    else if (root == InvalidOffset) normal = float3(BindlessBuffer(packed_float3, b.Buffer, g.Normals)[conn.VertexAddress(corner_vertex)]);
    else normal = float3(BindlessBuffer(NormalSector, b.Buffer, g.NormalSectors.Slot)[g.NormalSectors.Offset + root].Normal);
    const float2 offset = CustomNormalOffset(b, g.CustomNormals, corner);
    if (offset.x >= 0.f) {
        const auto position = [&](uint h) { return float3(BindlessBuffer(Vertex, b.VertexBuffer, g.Positions)[conn.VertexAddress(conn.VertexOf(h))].Position); };
        normal = DecodeNormalOffset(offset, ComputeCornerFrame(normal, position(corner), position(conn.Next(corner)), position(conn.Previous(corner))));
    }
    return normal;
}

// A corner's tangent, and zero without tangents.
inline float3 OpCornerTangent(device const BindlessSet &b, OpGeometry g, uint corner) { return g.Tangents.Slot == InvalidSlot ? float3(0.0f) : float3(BindlessBuffer(packed_float4, b.CornerTangentBuffer, g.Tangents.Slot)[g.Tangents.Offset + corner].xyz); }
// A corner's coordinates in the texture coordinate set with V up, and zero for an absent set.
inline float2 OpCornerTexcoord(device const BindlessSet &b, SlotOffset uvs, uint corner) {
    if (uvs.Slot == InvalidSlot) return float2(0.0f);
    const float2 uv = float2(BindlessBuffer(packed_float2, b.CornerUvBuffer, uvs.Slot)[uvs.Offset + corner]);
    return float2(uv.x, 1.0f - uv.y);
}
// The color of an element of the colors' domain, and ColorAbsent without colors.
inline float4 OpElementColor(device const BindlessSet &b, OpGeometry g, uint element) { return g.Colors.Slot == InvalidSlot ? float4(g.ColorAbsent) : float4(BindlessBuffer(packed_float4, b.CornerColorBuffer, g.Colors.Slot)[g.Colors.Offset + element]); }

// The evaluation point at a live element of the call's domain, with the point fields the call's program reads.
// A corner's point also holds its vertex and face, and its position is its vertex's.
inline OpPoint OpPointAt(device const BindlessSet &b, constant OpCall &call, DomainElement e) {
    const OpGeometry g = call.Geometry;
    const uint needs = call.Program.Needs;
    OpPoint p;
    OpStart(p, b, call.Program, call.Frame, call.Time);
    const bool corner = g.Connectivity.CornersSlot != InvalidSlot;
    uint position_address = e.Address;
    if (corner) {
        const ConnectivityView conn{b, g.Connectivity};
        p.Vertex = conn.VertexOf(e.Handle);
        p.Face = conn.HalfedgeFace(e.Handle);
        position_address = conn.VertexAddress(p.Vertex);
        if (needs & OpNeedsNormal) p.Normal = OpCornerNormal(b, g, conn, e.Handle, p.Vertex, p.Face);
    } else if (needs & OpNeedsNormal) {
        p.Normal = float3(BindlessBuffer(packed_float3, b.Buffer, g.Normals)[e.Address]);
    }
    if (needs & OpNeedsPosition) p.Position = float3(BindlessBuffer(Vertex, b.VertexBuffer, g.Positions)[position_address].Position);
    if (needs & OpNeedsTangent) p.Tangent = OpCornerTangent(b, g, e.Handle);
    if (needs & OpNeedsTexcoords) {
        for (uint s = 0u; s < 4u; ++s) p.Texcoords[s] = OpCornerTexcoord(b, g.Texcoords[s], e.Handle);
    }
    if (needs & OpNeedsColor) p.Color = OpElementColor(b, g, e.Handle);
    if (needs & OpNeedsIndex) p.Index = int(e.Ordinal);
    p.Elements[0] = e.Handle;
    p.Buffers = &b.Buffer[0];
    p.Attributes = reinterpret_cast<device const OpAttributeRef *>(OpWords(b, call.Program.Attributes));
    return p;
}

// The point at the weighted corners of a triangle of the call's record, with the point fields the call's program reads, each the weighted value at the corners.
inline OpPoint OpTrianglePoint(device const BindlessSet &b, constant OpCall &call, ConnectivityView conn, uint3 corners, float3 weights) {
    const OpGeometry g = call.Geometry;
    const uint needs = call.Program.Needs;
    OpPoint p;
    OpStart(p, b, call.Program, call.Frame, call.Time);
    p.Face = conn.HalfedgeFace(corners.x);
    for (uint c = 0u; c < 3u; ++c) {
        p.Elements[c] = corners[c];
        p.Weights[c] = weights[c];
        p.Vertices[c] = conn.VertexOf(corners[c]);
    }
    p.Vertex = p.Vertices[0];
    p.Position = p.Normal = p.Tangent = float3(0.0f);
    p.Color = float4(0.0f);
    for (uint s = 0u; s < 4u; ++s) p.Texcoords[s] = float2(0.0f);
    for (uint c = 0u; c < 3u; ++c) {
        const float w = weights[c];
        const uint corner = corners[c], corner_vertex = p.Vertices[c];
        if (needs & OpNeedsPosition) p.Position += w * float3(BindlessBuffer(Vertex, b.VertexBuffer, g.Positions)[conn.VertexAddress(corner_vertex)].Position);
        if (needs & OpNeedsNormal) p.Normal += w * OpCornerNormal(b, g, conn, corner, corner_vertex, p.Face);
        if (needs & OpNeedsTangent && g.Tangents.Slot != InvalidSlot) p.Tangent += w * OpCornerTangent(b, g, corner);
        if (needs & OpNeedsTexcoords) {
            for (uint s = 0u; s < 4u; ++s)
                if (g.Texcoords[s].Slot != InvalidSlot) p.Texcoords[s] += w * OpCornerTexcoord(b, g.Texcoords[s], corner);
        }
        if (needs & OpNeedsColor) p.Color += w * OpElementColor(b, g, corner);
    }
    p.Buffers = &b.Buffer[0];
    p.Attributes = reinterpret_cast<device const OpAttributeRef *>(OpWords(b, call.Program.Attributes));
    return p;
}
