#pragma once

#include "gpu/MeshTopologyJob.h"
#include "mesh/GeometrySelection.h"
#include "state/Entity.h"

// Capture the pages each topology emission phase writes into record `id`, after
// allocation and before that phase is submitted.
// Emission writes the output vertices and faces in the arena blocks given, the corner handle run and the triangle handle run, and their hidden and output selection masks.
// Attribute tables must already describe the destination.
void CaptureTopologyEmitWrites(state::Scene &, uint32_t id, const MeshTopologyJob &, uint32_t triangle_count, std::span<const uint32_t> vertex_blocks, std::span<const uint32_t> face_blocks, const std::optional<GeometryMasks> &selection_output);
// Edge ownership, attributes, hidden masks and output selection masks in the retained and new edge arena blocks.
void CaptureTopologyEdgeWrites(state::Scene &, std::span<const uint32_t> edge_blocks, const std::optional<GeometryMasks> &selection_output);
// Custom normals of the corner run and of the retained corners in `retained`.
// Destination normal payloads must be allocated.
void CaptureTopologyNormalWrites(state::Scene &, uint32_t id, const MeshTopologyJob &, const BufferArena<uint32_t> &retained);
