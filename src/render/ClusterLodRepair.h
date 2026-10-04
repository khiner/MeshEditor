#pragma once
#include "state/Entity.h"
#include <cstdint>
#include <span>
#include <vector>

struct GpuBuffers;
namespace mtl { struct ComputeChain; }

// Returns the seed groups and every group their proxies feed, transitively, in ascending order.
std::vector<uint32_t> ClusterGroupClosure(const GpuBuffers &, std::span<const uint32_t> seeds);
// Rewrites the member and proxy runs of every group a removed or added cluster names.
// Removed records still carry the group fields they were linked under.
void ReplaceGroupClusters(GpuBuffers &, std::span<const uint32_t> removed, std::span<const uint32_t> added);
// One render owner's seed groups.
struct ClusterGroupSeeds {
    state::Entity Entity;
    std::vector<uint32_t> Groups;
};
// Marks the closure of each entity's seed groups stale on its render owner until the next repair, with one index update for every owner.
// A stale group's error is infinite, so the cut refines through it to current geometry.
// Returns each entity's members of its stale groups in ascending order, whose traversal leaves carry those errors.
std::vector<std::vector<uint32_t>> InvalidateClusterGroups(state::Scene &, std::span<const ClusterGroupSeeds>);
// Rebuilds the DAG above each entity's stale groups and every group they feed, with the pools of every owner simplifying together.
// Each primitive's kept members pool by level and re-partition as a full build of them does, and the stale groups retire with their coarse clusters.
// The chain records each owner's traversal refit, and its next submit retires the stale groups, after which the owners have none.
void RepairDirtyClusterGroups(state::Scene &, mtl::ComputeChain &, std::span<const state::Entity>);
