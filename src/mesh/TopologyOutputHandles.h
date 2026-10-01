#pragma once

#include "gpu/ElementHandleRange.h"
#include "gpu/ElementWork.h"
#include "gpu/MeshTopologyArenas.h"
#include "gpu/MeshTopologyJob.h"
#include "gpu/TopologyIdentityPushConstants.h"
#include "mesh/MeshStore.h"
#include "metal/Buffer.h"

namespace mtl { struct ComputeChain; }

// Plan canonical output identities from the operator's scanned counts, recorded after its count passes.
// Geometry stays in the arenas. Maps and allocation counts are the only produced data, read once the chain submits.
// Source handles must remain reserved while the edit's source clones are in use.
// The source core contains complete replaced faces. Triangle starts and the
// job's source connectivity must refer to the same source clones.
enum class TopologyIdentityPolicy { Preserve, Fresh };

struct TopologyOutputHandles {
    TopologyOutputHandles(state::Scene &, mtl::ComputeChain &, const MeshTopologyJob &, MeshStore::TopologyCounts bounds,
                          uint32_t scratch_slot, const MeshTopologyArenas &source, TopologyIdentityPolicy = TopologyIdentityPolicy::Preserve);
    void Finish(const mtl::ComputeChain &);
    // Records the writes of newly inserted vertex and face handles, each a run or a list, into the output maps in compact new-element order.
    void Assign(state::Scene &, mtl::ComputeChain &, std::array<ElementHandleRange,2> inserted) const;

    mtl::Buffer Vertices, Faces;
    std::array<ElementWork,2> New{}; // Compact output ordinals without a retained source identity
    std::array<uint32_t,2> NewCounts{};
    std::array<ElementWork,2> Retired{};
    std::array<uint32_t,2> RetiredCounts{};
    std::array<ElementWork,2> Replaced{}; // All source corners and their derived triangles
};
