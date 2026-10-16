#pragma once

#include "gpu/ElementWork.h"
#include "gpu/MeshRecord.h"

// A tile can fill eight 48-triangle meshlets without forcing a short final
// cluster. Its 1,152 corner keys fit the shared-memory construction workspace.
GPU_CONSTANT uint32_t MeshletBuildTileElements = 384u;
GPU_CONSTANT uint32_t MeshletBuildTileRecords = (MeshletBuildTileElements + 20u) / 21u;
// One SIMD group chooses each next triangle without a cross-group reduction.
GPU_CONSTANT uint32_t MeshletBuildClusterThreads = 32u;

struct MeshletBuildJob {
    MeshRecord Mesh DEFAULT();
    SlotOffset AuxIndices DEFAULT();
    uint32_t Topology DEFAULT();
    uint32_t ElementCount DEFAULT();
    uint32_t PrimitiveCount DEFAULT();
    uint32_t BlockCount DEFAULT();
    uint32_t TileBound DEFAULT();
    uint32_t RadixPassCount DEFAULT();
    uint32_t StatsOffset DEFAULT();
    uint32_t KeysOffset DEFAULT();
    uint32_t OrderOffset DEFAULT();
    uint32_t TempOrderOffset DEFAULT();
    uint32_t HistogramOffset DEFAULT();
    uint32_t DigitTotalsOffset DEFAULT();
    uint32_t PrimitiveScratchOffset DEFAULT();
    uint32_t TileScratchOffset DEFAULT();
    uint32_t TileCountsOffset DEFAULT();
    uint32_t RecordScratchOffset DEFAULT();
    uint32_t VertexScratchOffset DEFAULT();
    // The pool positions the job's payload runs fill, and their offsets in the payload ranges its records name them by.
    uint32_t TriangleAddress DEFAULT(), TriangleOffset DEFAULT();
    uint32_t LocalTriangleAddress DEFAULT(), LocalTriangleOffset DEFAULT();
    uint32_t VertexAddress DEFAULT(), VertexOffset DEFAULT();
    // The arena addresses and record-relative handles of the job's first cluster, primitive and traversal node.
    uint32_t MeshletAddress DEFAULT(), MeshletHandle DEFAULT();
    uint32_t PrimitiveAddress DEFAULT(), PrimitiveHandle DEFAULT();
    uint32_t NodeAddress DEFAULT(), NodeHandle DEFAULT();
    uint32_t PrimitiveRoutes DEFAULT(); // The destination's first route, indexed by source material * 3 + topology.
    // Canonical membership to gather; absent when Elements is supplied by a local edit.
    SlotOffset ElementBlockIds DEFAULT();
    uint32_t ElementBlockCount DEFAULT(), ElementBlocksSlot DEFAULT(InvalidSlot), ElementOwner DEFAULT(InvalidOffset);
    ElementWork Elements DEFAULT(), Materials DEFAULT();
    uint32_t ExistingPrimitive DEFAULT(InvalidOffset), ExistingGroup DEFAULT(InvalidOffset);
};
