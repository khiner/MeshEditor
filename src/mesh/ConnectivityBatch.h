#pragma once

#include "gpu/MeshConnectivityJob.h"
#include "mesh/MeshPipelines.h"
#include "mesh/TiledJobBatch.h"

// Scratch follows compact work ordinals.
// All connectivity values use canonical handles.
// Face ranges and corner values must be readable for the supplied halfedges.
// Halfedge work includes complete incidence for each rebuilt edge, and all incoming and outgoing corners of the writable vertices.
// Boundary vertices outside that vertex set retain their incidence, so a local driver supplies the closure.
// New edges take the first of the job's edge handles, and the count word at StateOffset reports how many.
using ConnectivityBatch = TiledJobBatch<MeshConnectivityJob, 6>;
inline constexpr std::array ConnectivityPasses{
    TiledPass{MeshPass::ConnectivityFaces, 4},
    TiledPass{MeshPass::ConnectivityInit, 0},
    TiledPass{MeshPass::ConnectivityInsert, 1},
    TiledPass{MeshPass::ConnectivityMatchEdges, 5},
    TiledPass{MeshPass::ConnectivityResolve, 1},
    TiledPass{MeshPass::ConnectivityLink, 1},
    TiledPass{MeshPass::ConnectivityWordBlockSum, 2},
    TiledPass{MeshPass::ConnectivityWordBlockPrefix, PerJob},
    TiledPass{MeshPass::ConnectivityRanks, 2},
    TiledPass{MeshPass::ConnectivityCounts, PerJob},
    TiledPass{MeshPass::ConnectivityRetiredEdges, 5},
    TiledPass{MeshPass::ConnectivityEdgeTables, 1},
};

uint32_t LayoutConnectivityScratch(MeshConnectivityJob &, uint32_t first_word = 0u);
void AddConnectivityJob(ConnectivityBatch &, MeshConnectivityJob);
