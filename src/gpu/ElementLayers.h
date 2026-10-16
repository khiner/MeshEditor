#pragma once

#include "gpu/SlotOffset.h"

// A record's element layers, each its values slot and the index of the record's handle 0 there, or no slot for a layer the record lacks.
// Index layer values by offset + handle, up to the domain handle bound.
struct ElementLayers {
    SlotOffset FacePrimitives DEFAULT(), VertexPrimitives DEFAULT();
    SlotOffset Skin DEFAULT(), Morph DEFAULT();
    SlotOffset CornerTangent DEFAULT(), CornerColor DEFAULT(), VertexColor DEFAULT();
    GpuArray<SlotOffset, 4> CornerUvs DEFAULT();
    SlotOffset CustomNormals DEFAULT();
    SlotOffset CornerSectors DEFAULT(), NormalSectors DEFAULT();
};
