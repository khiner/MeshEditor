#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/MeshTopologyArenas.h"
#include "gpu/SlotOffset.h"
#include "metal/Buffer.h"
#include "metal/BufferArena.h"

namespace state { struct Scene; }
struct MeshClosure;

// Read-only clones of every canonical page the source neighborhood's topology rules read.
// The neighborhood's work lives in `scratch`.
// Clones preserve bytes, not allocator identities: the caller keeps source
// handles reserved until the clones and their submitted consumers retire.
// Bindings without read pages stay invalid.
struct TopologyReadView {
    TopologyReadView(state::Scene &, uint32_t source_id, const MeshClosure &, const BufferArena<uint32_t> &scratch);

    MeshTopologyArenas Arenas;
    ConnectivityRef Connectivity;
    std::array<SlotOffset,3> Selection; // V, E, F, canonical bit indices
    std::vector<mtl::Buffer> Clones;
};
