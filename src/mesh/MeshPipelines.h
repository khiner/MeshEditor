#pragma once
#include "mesh/MeshConnectivityPipelines.h"
#include "mesh/VertexAdjacencyPipelines.h"
#include "mesh/VertexWeldPipelines.h"

#include "state/Entity.h"

struct MeshPipelines {
    explicit MeshPipelines(mtl::LibraryCache &);
    VertexAdjacencyPipelines VertexAdjacency;
    VertexWeldPipelines VertexWeld;
    MeshConnectivityPipelines MeshConnectivity;
};

// Returns the mesh build pipelines, compiling them on first use.
MeshPipelines &GetMeshPipelines(state::Scene &);
