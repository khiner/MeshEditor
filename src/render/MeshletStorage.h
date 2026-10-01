#pragma once
#include <cstdint>
#include <span>

struct MeshBuffers;
namespace state { struct Scene; }

// Releases the payload ranges the retired clusters' host records name, then their identities.
// Live draw memberships and other consumers must stop using retired IDs first.
// Every later arena writer must capture reused pages.
// Reject foreign or already-retired IDs before mutation.
void RetireMeshletStorage(state::Scene &, MeshBuffers &, std::span<const uint32_t> clusters);
