#pragma once

#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Defines one mesh's vertex-weld inputs, outputs, and scratch layout.
// Flags contain one mark per vertex plus a terminator for exclusive-scan index generation.
struct VertexWeldJob {
    // Position range used as the first weld-key channel and compacted output.
    SlotOffset Positions DEFAULT();
    // Corner array remapped to welded indices.
    SlotOffset Corners DEFAULT();
    // Source-vertex skin joints and weights. InvalidSlot denotes no skin.
    SlotOffset Deform DEFAULT();
    // Target-major morph position and normal deltas. InvalidSlot denotes no morph targets.
    SlotOffset Morph DEFAULT();
    uint32_t TargetCount DEFAULT();
    // Target-major three-word morph tangent deltas staged in scratch. InvalidOffset denotes no authored tangents.
    uint32_t TangentOffset DEFAULT(InvalidOffset);
    uint32_t Count DEFAULT();
    uint32_t CornerCount DEFAULT();
    uint32_t TableOffset DEFAULT();
    uint32_t TableMask DEFAULT();
    uint32_t SlotOffset DEFAULT();
    uint32_t FlagsOffset DEFAULT();
    uint32_t BlockOffset DEFAULT();
    uint32_t BlockCount DEFAULT();
    uint32_t RemapOffset DEFAULT();
    uint32_t RepsOffset DEFAULT();
    // Stage welded channels to prevent in-place compaction from overwriting unread source data.
    uint32_t CompactOffset DEFAULT();
    // Word stride of one staged welded vertex across all vertex-domain channels.
    uint32_t RecordWords DEFAULT();
};
static_assert(sizeof(VertexWeldJob) == 88, "VertexWeldJob size");
