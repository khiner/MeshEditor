#pragma once

#include "metal/Shader.h"

// The vertex-adjacency CSR build's passes, in dispatch order.
struct VertexAdjacencyPipelines {
    mtl::ComputePipeline Zero, Count, BlockSum, BlockPrefix, Offsets, Scatter, Sort;

    explicit VertexAdjacencyPipelines(mtl::LibraryCache &libraries)
        : Zero{libraries, {"VertexAdjacency.metal", "VertexAdjacencyZero"}},
          Count{libraries, {"VertexAdjacency.metal", "VertexAdjacencyCount"}},
          BlockSum{libraries, {"VertexAdjacency.metal", "VertexAdjacencyBlockSum"}},
          BlockPrefix{libraries, {"VertexAdjacency.metal", "VertexAdjacencyBlockPrefix"}},
          Offsets{libraries, {"VertexAdjacency.metal", "VertexAdjacencyOffsets"}},
          Scatter{libraries, {"VertexAdjacency.metal", "VertexAdjacencyScatter"}},
          Sort{libraries, {"VertexAdjacency.metal", "VertexAdjacencySort"}} {}
};
