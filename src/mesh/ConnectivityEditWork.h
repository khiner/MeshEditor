#pragma once

#include "gpu/ElementHandleRange.h"
#include "gpu/ElementWork.h"
#include "state/Entity.h"

namespace mtl { struct ComputeChain; }
struct MeshClosure;

// Assembles the incidence for local repair: the old neighborhood minus the replaced core halfedges and faces, plus the emitted handles.
// Old writable vertices remain in the work even when deleted, so repair clears their incidence before retirement.
// Before includes every existing vertex whose incidence emission changes, and emitted vertices can also name newly reserved slots.
// Emission and this pass precede handle retirement.
// Counts are valid once the chain has submitted.
struct ConnectivityEditWork {
    ConnectivityEditWork(state::Scene &, mtl::ComputeChain &, const MeshClosure &before, const std::array<ElementWork,2> &replaced,
                         const std::array<ElementHandleRange,3> &emitted);
    void Finish(const mtl::ComputeChain &);

    std::array<ElementWork,3> Elements{}; // Writable vertices, halfedges, faces.
    std::array<uint32_t,3> Counts{};
};
