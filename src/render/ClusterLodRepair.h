#pragma once
#include "state/Entity.h"
#include <cstdint>
#include <span>
#include <vector>

struct GpuBuffers;
struct MeshBuffers;

// Returns the seed groups and every group their proxies feed, transitively, in ascending order.
std::vector<uint32_t> ClusterGroupClosure(const GpuBuffers &, std::span<const uint32_t> seeds);
// Rewrites the member and proxy runs of every group a removed or added cluster names.
// Removed records still carry the group fields they were linked under.
void ReplaceGroupClusters(GpuBuffers &, std::span<const uint32_t> removed, std::span<const uint32_t> added);
// Marks the closure of the seed groups stale on the entity's render owner until the next repair.
// A stale group's error is infinite, so the cut refines through it to current geometry.
// Returns the members of the stale groups, whose traversal leaves carry those errors.
std::vector<uint32_t> InvalidateClusterGroups(state::Scene &, state::Entity, std::span<const uint32_t> seeds);
// Rebuilds the DAG above the owner's stale groups and every group they feed.
// Each primitive's kept members pool by level and re-partition as a full build of them does, and the stale groups retire with their coarse clusters.
// The owner has no stale groups afterward.
void RepairDirtyClusterGroups(state::Scene &, MeshBuffers &);
