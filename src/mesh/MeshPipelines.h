#pragma once

#include "metal/Shader.h"

#include "state/Entity.h"

#include <array>

// Every mesh build pass, grouped by driver in dispatch order.
enum class MeshPass : uint8_t {
    ConnectivityZero,
    ConnectivityCount,
    ConnectivityBlockSum,
    ConnectivityBlockPrefix,
    ConnectivityOffsets,
    ConnectivityScatter,
    ConnectivityPair,
    ConnectivityBits,
    ConnectivityWordBlockSum,
    ConnectivityWordBlockPrefix,
    ConnectivityRanks,
    ConnectivitySamples,
    AdjacencyZero,
    AdjacencyCount,
    AdjacencyBlockSum,
    AdjacencyBlockPrefix,
    AdjacencyOffsets,
    AdjacencyScatter,
    AdjacencySort,
    WeldTableInit,
    WeldInsert,
    WeldMarkReps,
    WeldBlockSum,
    WeldBlockPrefix,
    WeldScan,
    WeldEmit,
    WeldRemapCorners,
    WeldCompact,
    WeldWriteBack,
    Count,
};

struct MeshPipelines {
    explicit MeshPipelines(mtl::LibraryCache &);
    const mtl::ComputePipeline &operator[](MeshPass pass) const { return Pipelines[size_t(pass)]; }

    std::array<mtl::ComputePipeline, size_t(MeshPass::Count)> Pipelines;
};

// Returns the mesh build pipelines, compiling them on first use.
MeshPipelines &GetMeshPipelines(state::Scene &);
