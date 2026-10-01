#pragma once

#include "gpu/MeshConnectivityJob.h"
#include "state/Entity.h"

#include <span>

// Capture the canonical pages a normalized connectivity job writes, before its passes are recorded.
// The prepare footprint names blocks covering the job's writable vertices, halfedges and faces.
void CaptureConnectivityPrepareWrites(state::Scene &, const MeshConnectivityJob &, std::span<const uint32_t> vertex_blocks,
                                      std::span<const uint32_t> halfedge_blocks, std::span<const uint32_t> face_blocks);
