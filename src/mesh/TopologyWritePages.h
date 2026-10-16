#pragma once

#include "gpu/MeshTopologyJob.h"
#include "mesh/GeometrySelection.h"
#include "state/Entity.h"

// Capture destination pages after allocation and before GPU submission.
// Require attribute tables to describe the destination before capturing vertex, face, corner, triangle, hidden, and selection writes.
void CaptureTopologyEmitWrites(state::Scene &, uint32_t id, const MeshTopologyJob &, uint32_t triangle_count, std::span<const uint32_t> vertex_blocks, std::span<const uint32_t> face_blocks, const std::optional<GeometryMasks> &selection_output);
// Edge ownership, attributes, hidden masks and output selection masks in the retained and new edge arena blocks.
void CaptureTopologyEdgeWrites(state::Scene &, std::span<const uint32_t> edge_blocks, const std::optional<GeometryMasks> &selection_output);
// Custom normals of the corner run and of the retained corners in `retained`.
// Destination normal payloads must be allocated.
void CaptureTopologyNormalWrites(state::Scene &, uint32_t id, const MeshTopologyJob &, const BufferArena<uint32_t> &retained);
