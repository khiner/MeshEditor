#pragma once

#include "gpu/Types.h"

// The argument table index of the pattern pass's work list, and the word of its first pixel.
// The list holds the count of the pixels of materials whose node graphs compute inputs, its threadgroup dispatch arguments, then each pixel as y times the width plus x.
GPU_CONSTANT uint32_t BufferIndex_PatternWork = 6u;
GPU_CONSTANT uint32_t SurfacePatternPixels = 4u;
// The argument table indices of the ray samples a hit kernel lists for the hit driver, of the ray queue the hit kernel keeps their surfaces in, and of the hit driver's program table.
// The list holds the count appended, the dispatch arguments of the listed entries in threadgroups of 64, the count listed, then two words per entry.
// Its arguments step moves the appended count to the listed count and clears it for the next hit kernel.
GPU_CONSTANT uint32_t BufferIndex_PatternHits = 12u, BufferIndex_PatternQueue = 13u, BufferIndex_PatternPrograms = 14u;
GPU_CONSTANT uint32_t PatternHitsListed = 4u, PatternHitEntries = 8u;
