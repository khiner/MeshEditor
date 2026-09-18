#pragma once

#include "metal/Shader.h"

#include "state/Entity.h"

#include <array>

// Every mesh build pass, grouped by driver in dispatch order.
enum class MeshPass : uint8_t {
    ConnectivityPrev,
    ConnectivityInit,
    ConnectivityInsert,
    ConnectivityResolve,
    ConnectivityLink,
    ConnectivityWordBlockSum,
    ConnectivityWordBlockPrefix,
    ConnectivityRanks,
    ConnectivityEdgeTables,
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
    EdgeEndpointsWrite,
    TriangleIndicesWrite,
    TopologyFaceIndex,
    TopologyZero,
    TopologyMarkHalfedges,
    TopologyMarkFaces,
    TopologyLink,
    TopologyJump,
    TopologyConverge,
    TopologyDissolveRegions,
    TopologyDissolveWalk,
    TopologyDissolveRevert,
    TopologyJoinBest,
    TopologyJoinMatch,
    TopologyZeroVertices,
    TopologyMergeTable,
    TopologyMergeInsert,
    TopologyMergeQuery,
    TopologyDissolveLimitVertices,
    TopologyListFill,
    TopologyCountVertices,
    TopologyCountHalfedges,
    TopologyCountFaces,
    TopologyScanBlockSum,
    TopologyScanBlockPrefix,
    TopologyScanOffsets,
    TopologyZeroOutput,
    TopologyScatterVertices,
    TopologyScatterHalfedges,
    TopologyScatterFaces,
    TopologyFaceTables,
    TopologyGatherVertices,
    TopologyGatherCorners,
    TopologyCustomPopcount,
    TopologyCustomPack,
    TopologyEdgeAttributes,
    Count,
};

struct MeshPipelines {
    explicit MeshPipelines(mtl::LibraryCache &);
    const mtl::ComputePipeline &operator[](MeshPass pass) const { return Pipelines[size_t(pass)]; }

    std::array<mtl::ComputePipeline, size_t(MeshPass::Count)> Pipelines;
};

// Returns the mesh build pipelines, compiling them on first use.
MeshPipelines &GetMeshPipelines(state::Scene &);
