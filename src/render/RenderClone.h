#pragma once

#include <cstdint>
#include <span>

struct CloneCopies;
namespace mtl { struct ComputeChain; }
namespace state { struct Scene; }

// Gives each clone, a store record cloned from its source, a copy of the source's render records with every reference rebased.
// The clone arrives render-ready, so it skips the new-mesh shading and meshlet builds, and its spatial tree builds once the chain submits the copies.
// A source without render records leaves its clone to those builds.
// The clones' mesh entities exist, so the records' display fields rederive with the settle pass.
void CloneRenderRecords(state::Scene &, mtl::ComputeChain &, CloneCopies &, std::span<const uint32_t> source_ids, std::span<const uint32_t> clone_ids);
