#pragma once
#include "gpu/Element.h"
#include <cstdint>
#include <span>
namespace state {
struct Scene;
}
enum class EditVisibilityOperation { HideSelected,
                                     HideUnselected,
                                     Reveal,
                                     RevealSelected };
bool EditVisibility(state::Scene &, std::span<const uint32_t> meshes, Element, EditVisibilityOperation);
