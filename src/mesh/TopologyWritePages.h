#pragma once

#include "gpu/MeshTopologyJob.h"
#include "metal/BufferArena.h"
#include "state/Entity.h"

// Capture the canonical pages each topology emission phase writes, after
// allocation and before that phase is submitted.
// Emission writes the output vertices and faces in the ascending blocks given, the corner run and the triangle run.
// Attribute tables must already describe the destination.
void CaptureTopologyEmitWrites(state::Scene &, const MeshTopologyJob &, uint32_t triangle_count, std::span<const uint32_t> vertex_blocks, std::span<const uint32_t> face_blocks);
// Edge ownership, attributes and selection in the ascending retained and new edge blocks.
void CaptureTopologyEdgeWrites(state::Scene &, std::span<const uint32_t> edge_blocks, bool editor_state);
// Custom normals of the corner run and of the retained corners in `retained`.
// Destination normal payloads must be allocated.
void CaptureTopologyNormalWrites(state::Scene &, const MeshTopologyJob &, const BufferArena<uint32_t> &retained);
