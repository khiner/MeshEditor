#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/MeshTopologyArenas.h"
#include "gpu/NamedAttributeRef.h"
#include "gpu/SlotOffset.h"
#include "mesh/PageFootprint.h"

namespace state {
struct Scene;
} // namespace state
struct MeshClosure;
struct MeshStore;

// Read-only clones of every page the source neighborhoods of a batch of in-place edits read, including the sources' table runs.
// Each source reads the clones at its own offsets, so the batch takes one slot per read buffer.
// Clones preserve bytes, not allocator identities: the caller keeps source
// handles reserved until the clones and their submitted consumers retire.
// Bindings without read pages stay invalid.
struct TopologyReadView {
    // Adds the pages the rules of a source's neighborhood read, whose work lives in `scratch`.
    void Add(state::Scene &, uint32_t source_id, const MeshClosure &, const BufferArena<uint32_t> &scratch);
    // Clones the added pages and binds them, after every source of the batch is added.
    void Clone(state::Scene &);
    // A source's connectivity through its table runs in the clones.
    ConnectivityRef SourceConnectivity(const MeshStore &, uint32_t source_id) const;
    // A source's element layers at its ranges in the clones, read before its insertions move any range.
    ElementLayers SourceLayers(const MeshStore &, uint32_t source_id) const;

    MeshTopologyArenas Arenas; // Clone slots, with layers without a source's offsets
    // The clone of each named attribute value arena, by AttributeType.
    std::array<uint32_t, AttributeTypeCount> AttributeValues;
    std::vector<mtl::Buffer> Clones;

private:
    PageFootprint Pages;
    ConnectivityRef Connectivity; // Clone slots, without a source's offsets
};
