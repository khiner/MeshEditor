#pragma once

#include "metal/Shader.h"

// The vertex-weld passes, in dispatch order.
struct VertexWeldPipelines {
    mtl::ComputePipeline TableInit, Insert, MarkReps, BlockSum, BlockPrefix, Scan, Emit, RemapCorners, Compact, WriteBack;

    explicit VertexWeldPipelines(mtl::LibraryCache &libraries)
        : TableInit{libraries, {"VertexWeld.metal", "VertexWeldTableInit"}},
          Insert{libraries, {"VertexWeld.metal", "VertexWeldInsert"}},
          MarkReps{libraries, {"VertexWeld.metal", "VertexWeldMarkReps"}},
          BlockSum{libraries, {"VertexWeld.metal", "VertexWeldBlockSum"}},
          BlockPrefix{libraries, {"VertexWeld.metal", "VertexWeldBlockPrefix"}},
          Scan{libraries, {"VertexWeld.metal", "VertexWeldScan"}},
          Emit{libraries, {"VertexWeld.metal", "VertexWeldEmit"}},
          RemapCorners{libraries, {"VertexWeld.metal", "VertexWeldRemapCorners"}},
          Compact{libraries, {"VertexWeld.metal", "VertexWeldCompact"}},
          WriteBack{libraries, {"VertexWeld.metal", "VertexWeldWriteBack"}} {}
};
