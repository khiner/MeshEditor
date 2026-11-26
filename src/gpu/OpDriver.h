#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"
#include "gpu/OpPoint.h"

// A program output the element driver writes to a named attribute.
struct OpOutput {
    NamedAttributeRef Attribute DEFAULT();
    AttributeType Type DEFAULT();
};

// The stored values the drivers fill point fields from, by arena address or record-relative handle.
// A layer without a slot reads zero, and colors read ColorAbsent.
// A corner's point reads its vertex's position and its shading normal, and the record's connectivity gives the corner's vertex and face.
struct OpGeometry {
    uint32_t Positions DEFAULT(InvalidSlot); // Vertex arena, by address
    uint32_t Normals DEFAULT(InvalidSlot); // Vertex or face normals, by address, and vertex normals for corners
    // A corner's shading normal sources: face normals and face sharpness by address, the record's corner class mode, and its sector roots, sector normals and custom normal offsets by handle.
    uint32_t FaceNormals DEFAULT(InvalidSlot), FaceSharpness DEFAULT(InvalidSlot), ClassMode DEFAULT(InvalidOffset);
    SlotOffset CornerSectors DEFAULT(), NormalSectors DEFAULT(), CustomNormals DEFAULT();
    SlotOffset Tangents DEFAULT(), Colors DEFAULT(); // By handle
    GpuArray<SlotOffset, 4> Texcoords DEFAULT(); // By handle, with V down
    vec4 ColorAbsent DEFAULT();
    ConnectivityRef Connectivity DEFAULT(); // A corner domain's record connectivity
};

// A program called at the live elements of a record's domain, whose blocks of the domain are Blocks.
struct OpCall {
    OpProgramRef Program DEFAULT();
    float Frame DEFAULT(), Time DEFAULT();
    DomainBlocks Blocks DEFAULT();
    OpGeometry Geometry DEFAULT();
};

// One program evaluated at every live element of the call's domain, each tile one block of the call's blocks, one thread per element slot.
struct OpElementsJob {
    OpCall Call DEFAULT();
    uint32_t OutputCount DEFAULT();
    GpuArray<OpOutput, OpMaxOutputs> Outputs DEFAULT();
};

// One boolean program evaluated at every live element of the call's domain, each tile one block of the call's blocks, one thread per element slot.
// It sets the mask bit of each element it returns true at, and mask block b covers element block b at word Masks.Offset + 8b.
struct OpMasksJob {
    OpCall Call DEFAULT();
    SlotOffset Masks DEFAULT();
};

// One program evaluated at one point, one thread writing its Words output words to Result.
struct OpPointPushConstants {
    OpProgramRef Program DEFAULT();
    SlotOffset Result DEFAULT();
    float Frame DEFAULT(), Time DEFAULT();
    uint32_t Words DEFAULT();
};

// One program evaluated at each texel of a Width by Height grid over UV set UvSet of the call's record, one thread per texel in tiles of JobTileThreads, writing the first Components words of its first output to Output, Stride words per texel, row by row from the top.
// Texels holds each texel's triangle address plus one, or zero for a texel no triangle reaches.
// A texel evaluates at the triangle's point under its center, or at the triangle's nearest point when the center lies outside it within Margin texels, and holds zero beyond.
// The call's geometry is its record's corner geometry.
struct OpTexelsJob {
    OpCall Call DEFAULT();
    SlotOffset Texels DEFAULT(), Output DEFAULT();
    uint32_t TrianglesSlot DEFAULT(InvalidSlot); // Triangle arena, by address
    uint32_t Width DEFAULT(), Height DEFAULT(), Margin DEFAULT(), UvSet DEFAULT(), Components DEFAULT(), Stride DEFAULT();
};

// One program evaluated at each texel of a grid of Width by Height texels over the unit square of texture coordinates and Apron more beyond each side, one thread per texel, writing the first Components words of its first output to Output, Stride floats per texel, row by row from the top.
// Texel (x, y) has the coordinates ((x + 0.5 - Apron) / Width, 1 - (y + 0.5 - Apron) / Height) in every set.
struct OpGridPushConstants {
    OpProgramRef Program DEFAULT();
    SlotOffset Output DEFAULT();
    float Frame DEFAULT(), Time DEFAULT();
    uint32_t Width DEFAULT(), Height DEFAULT(), Apron DEFAULT(), Components DEFAULT(), Stride DEFAULT();
};
