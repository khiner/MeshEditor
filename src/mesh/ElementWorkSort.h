#pragma once
#include "gpu/ElementWork.h"
#include "state/Entity.h"
#include <span>

namespace mtl {
struct ComputeChain;
}

// Records sorting occupied block keys and publishing compact prefixes, with temporaries in the chain's scratch.
void EncodeSortElementWork(state::Scene &, mtl::ComputeChain &, std::span<const ElementWork>);
// The chain scratch words sorting one work domain of `capacity` slots takes.
uint32_t SortElementWorkWords(uint32_t capacity);
