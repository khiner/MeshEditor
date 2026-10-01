#pragma once

#include "gpu/MeshTopologyJob.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"

// Pure reservation calculations, shared by production and limit checks that
// must not allocate the geometry represented by these counts.
MeshStore::TopologyCounts TopologyOutputBounds(const MeshTopologyTask &, MeshStore::TopologyCounts source);
uint32_t LayoutTopologyScratch(MeshTopologyJob &, MeshStore::TopologyCounts source, MeshStore::TopologyCounts bounds, uint32_t base);
// The words of a job's open-addressing table: a merge by distance's vertex cells, and the output lines of a line core whose operator joins lines.
uint32_t TopologyTableWords(MeshTopologyOp, MeshStore::TopologyCounts source);
