#pragma once
#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"

// Each selected face reports whether it flips, and only a closed component's faces flip, to enclose positive volume.
// Per-face runs are indexed by face handle.
struct RecalculateNormalsPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    DomainBlocks Blocks DEFAULT(), HalfedgeBlocks DEFAULT(); // Face and halfedge blocks
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    uint32_t ScratchSlot DEFAULT(InvalidSlot); // The buffer every run below offsets into
    uint32_t Neighbors DEFAULT(); // Per halfedge: the face across its manifold edge shifted left, with the parity bit set where that face runs the same way, or InvalidOffset
    uint32_t Labels DEFAULT(); // Per face: its component's lowest face shifted left, with its parity bit against that face
    uint32_t Components DEFAULT(); // Per face: NormalComponentWords words, used at a component's lowest face
    uint32_t Changed DEFAULT(); // One word a link round sets when a label falls
    uint32_t Selection DEFAULT(), SelectionCount DEFAULT();
    uint32_t Flips DEFAULT(); // Per selected face: whether it flips
};

// The words of a component's record: whether a selected face is in it, whether it is open, and its volume sum low word first.
// Then the complemented ordered bits of its least coordinates and the ordered bits of its greatest.
GPU_CONSTANT uint32_t NormalComponentMarked = 0u, NormalComponentOpen = 1u, NormalComponentVolume = 2u, NormalComponentLeast = 4u, NormalComponentGreatest = 7u, NormalComponentWords = 10u;
