#pragma once

#include "metal/Shader.h"

// The half-edge connectivity build's passes, in dispatch order.
struct MeshConnectivityPipelines {
    mtl::ComputePipeline Zero, Count, BlockSum, BlockPrefix, Offsets, Scatter, Pair, Bits, WordBlockSum, WordBlockPrefix, Ranks, Samples;

    explicit MeshConnectivityPipelines(mtl::LibraryCache &libraries)
        : Zero{libraries, {"MeshConnectivity.metal", "MeshConnectivityZero"}},
          Count{libraries, {"MeshConnectivity.metal", "MeshConnectivityCount"}},
          BlockSum{libraries, {"MeshConnectivity.metal", "MeshConnectivityBlockSum"}},
          BlockPrefix{libraries, {"MeshConnectivity.metal", "MeshConnectivityBlockPrefix"}},
          Offsets{libraries, {"MeshConnectivity.metal", "MeshConnectivityOffsets"}},
          Scatter{libraries, {"MeshConnectivity.metal", "MeshConnectivityScatter"}},
          Pair{libraries, {"MeshConnectivity.metal", "MeshConnectivityPair"}},
          Bits{libraries, {"MeshConnectivity.metal", "MeshConnectivityBits"}},
          WordBlockSum{libraries, {"MeshConnectivity.metal", "MeshConnectivityWordBlockSum"}},
          WordBlockPrefix{libraries, {"MeshConnectivity.metal", "MeshConnectivityWordBlockPrefix"}},
          Ranks{libraries, {"MeshConnectivity.metal", "MeshConnectivityRanks"}},
          Samples{libraries, {"MeshConnectivity.metal", "MeshConnectivitySamples"}} {}
};
