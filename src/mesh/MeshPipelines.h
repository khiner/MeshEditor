#pragma once
#include "mesh/MeshConnectivityPipelines.h"
#include "mesh/VertexAdjacencyPipelines.h"
#include "mesh/VertexWeldPipelines.h"

struct MeshPipelines {
    explicit MeshPipelines(mtl::LibraryCache &);
    void CompileShaders(mtl::LibraryCache &);
    VertexAdjacencyPipelines VertexAdjacency;
    VertexWeldPipelines VertexWeld;
    MeshConnectivityPipelines MeshConnectivity;
};
