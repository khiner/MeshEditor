#pragma once
#include "gpu/Types.h"

// Pointer-jumping rounds over doubled runs of `bound` words, which read the half at `parity` and write the other half.
inline void LabelRound(device uint *labels, device uint *jumps, uint bound, uint parity, uint i) {
    const uint in = parity * bound, out = bound - in;
    const uint j = jumps[in + i];
    labels[out + i] = j == InvalidOffset ? labels[in + i] : min(labels[in + i], labels[in + j]);
    jumps[out + i] = j == InvalidOffset ? j : jumps[in + j];
}
inline void RankRound(device uint *distances, device uint *successors, uint bound, uint parity, uint i) {
    const uint in = parity * bound, out = bound - in;
    const uint j = successors[in + i];
    distances[out + i] = j == InvalidOffset ? distances[in + i] : distances[in + i] + distances[in + j];
    successors[out + i] = j == InvalidOffset ? j : successors[in + j];
}
