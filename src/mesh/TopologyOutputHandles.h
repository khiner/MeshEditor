#pragma once

#include "gpu/TopologyIdentityPushConstants.h"
#include "mesh/MeshStore.h"

namespace mtl {
struct ComputeChain;
} // namespace mtl

// Plan output identities from the operator's scanned counts, recorded after its count passes.
// Geometry stays in the arenas.
// Maps and allocation counts are the only produced data, ranges of the chain's scratch read once the chain submits.
// Source handles must remain reserved while the edit's source clones are in use.
// The source core contains complete replaced faces.
// Triangle starts and the job's source connectivity must refer to the same source clones.
enum class TopologyIdentityPolicy { Preserve,
                                    Fresh };

struct TopologyOutputHandles {
    // Use batch scratch containing scanned counts and source_id matching the source work handles.
    TopologyOutputHandles(state::Scene &, mtl::ComputeChain &, uint32_t source_id, const MeshTopologyJob &, MeshStore::TopologyCounts bounds, SlotOffset scratch, const MeshTopologyArenas &source, TopologyIdentityPolicy = TopologyIdentityPolicy::Preserve);
    void Finish(const mtl::ComputeChain &);
    // Records the writes of newly inserted vertex and face handles, each a run or a list, into the output maps in compact new-element order.
    void Assign(state::Scene &, mtl::ComputeChain &, std::array<ElementHandleRange, 2> inserted) const;

    Range Vertices, Faces; // Compact output ordinal -> handle, in the chain's scratch
    std::array<ElementWork, 2> New{}; // Compact output ordinals without a retained source identity
    std::array<uint32_t, 2> NewCounts{};
    std::array<ElementWork, 2> Retired{};
    std::array<uint32_t, 2> RetiredCounts{};
    std::array<ElementWork, 2> Replaced{}; // All source corners and their derived triangles
};
