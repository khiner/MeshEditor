#pragma once
#include "mesh/MeshStore.h"

namespace state {
struct Scene;
} // namespace state

// Releases the payload runs the retired clusters' records name, then their identities.
// Live draw memberships and other consumers must stop using retired IDs first.
// Every later arena writer must capture reused pages.
// Reject foreign or already-retired IDs before mutation.
void RetireMeshletStorage(state::Scene &, MeshStore::Record &, std::span<const uint32_t> clusters);
