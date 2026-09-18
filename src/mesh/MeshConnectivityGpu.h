#pragma once

#include <memory>
#include <span>

#include "state/Entity.h"

namespace MTL {
class ComputeCommandEncoder;
}

// Connectivity builds encoded into a caller's command buffer, finished once that command buffer has completed.
struct PendingConnectivity {
    struct Batches;
    PendingConnectivity();
    PendingConnectivity(PendingConnectivity &&) noexcept;
    ~PendingConnectivity();
    std::unique_ptr<Batches> Chunks;
};

// Captures the listed face meshes' connectivity writes and encodes their halfedge builds into the encoder.
// Requires AllocateConnectivity on each store id, with face starts in place for a mesh whose faces are not all triangles.
PendingConnectivity EncodeConnectivity(state::Scene &, std::span<const uint32_t> store_ids, MTL::ComputeCommandEncoder *);
// Completes each encoded mesh's record from the edge count its build wrote.
void FinishConnectivity(state::Scene &, PendingConnectivity &);
// Builds the listed face meshes' halfedge connectivity on the GPU and completes their records.
// Requires AllocateConnectivity on each store id, with face starts in place for a mesh whose faces are not all triangles.
void BuildConnectivityNow(state::Scene &, std::span<const uint32_t> store_ids);
