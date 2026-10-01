#pragma once

#include "gpu/ElementWork.h"
#include "gpu/MeshRecord.h"
#include "gpu/SlotOffset.h"

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
    uint32_t TriangleOffset DEFAULT();
    uint32_t LocalTriangleOffset DEFAULT();
    uint32_t VertexOffset DEFAULT();
    uint32_t MeshletOffset DEFAULT();
    uint32_t PrimitiveOffset DEFAULT();
    uint32_t NodeOffset DEFAULT();
    uint32_t PrimitiveRoutes DEFAULT(); // The destination's first route, indexed by source primitive.
    ElementWork Elements DEFAULT(), Materials DEFAULT();
    uint32_t ExistingPrimitive DEFAULT(InvalidOffset), ExistingGroup DEFAULT(InvalidOffset);
};
static_assert(sizeof(MeshletBuildJob) == 324);
