#pragma once

#include "gpu/NamedAttributeRef.h"
#include "gpu/SlotOffset.h"

// The point fields a program reads, which its driver fills.
GPU_CONSTANT uint32_t OpNeedsPosition = 1u << 0;
GPU_CONSTANT uint32_t OpNeedsNormal = 1u << 1;
GPU_CONSTANT uint32_t OpNeedsTangent = 1u << 2;
GPU_CONSTANT uint32_t OpNeedsTexcoords = 1u << 3;
GPU_CONSTANT uint32_t OpNeedsColor = 1u << 4;
GPU_CONSTANT uint32_t OpNeedsIndex = 1u << 5;
GPU_CONSTANT uint32_t OpNeedsImages = 1u << 6;
GPU_CONSTANT uint32_t OpNeedsNeighbors = 1u << 7;

// A program writes at most this many outputs, of at most this many 32-bit words in all.
GPU_CONSTANT uint32_t OpMaxOutputs = 8u;
GPU_CONSTANT uint32_t OpMaxOutputWords = 32u;

// The element of a point a named attribute is read at: the point's own element, a corner's vertex or face, or a weighted point's corners or their vertices.
enum class OpElement : uint32_t {
    Own,
    Vertex,
    Face,
    Corners,
    Vertices,
};
// A named attribute a program reads, and the element of the point it is read at.
struct OpAttributeRef {
    NamedAttributeRef Values DEFAULT();
    OpElement Element DEFAULT();
};
// A call of a program: its index in the program registry, the OpNeeds bits of the point fields it reads, and its value block, its named attribute table and its image table.
struct OpProgramRef {
    uint32_t Index DEFAULT(InvalidOffset), Needs DEFAULT();
    SlotOffset Values DEFAULT(), Attributes DEFAULT(), Images DEFAULT();
};
// An image a program samples, and no texture for none.
#ifndef __METAL_VERSION__
// The CPU writes the texture's 8-byte GPU resource id, which is zero for none.
struct OpImageTexture {
    uint64_t Texture{};
};
#else
struct OpImageTexture; // Defined by the source that samples images

// Where a program evaluates: the standard geometric fields its driver fills, and the element identity named attributes read through.
// A point of a mesh domain is its element, Elements[0].
// A texel or a surface point weights the three corners of its triangle, Elements, whose vertices are Vertices, and a texel's neighbors are the points of the next texel along each axis.
struct OpPoint {
    float3 Position, Normal, Tangent;
    float2 Texcoords[4]; // With V up
    float4 Color;
    int Index; // The element's ordinal among its record's live elements of the domain
    float Frame, Time;
    uint Elements[3];
    float Weights[3];
    uint Vertex, Face; // A corner's vertex and face, and InvalidOffset for a wire corner's face
    device const uchar *device const *Buffers; // The bindless buffer table
    device const OpAttributeRef *Attributes; // The program's named attributes in its order, and no values slot for an absent one
    uint Vertices[3];
    device const OpImageTexture *Images; // The program's image table
    thread const OpPoint *Neighbors;
    // A surface point's texture coordinates in each set with V down, then their derivatives along x in each set, then along y.
    thread const float2 *SurfaceTexcoords;
};
#endif
