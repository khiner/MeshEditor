#pragma once
#include "gpu/Element.h"
#include <span>
namespace state {
struct Registry;
} // namespace state
enum class EditVisibilityOperation { HideSelected,
                                     HideUnselected,
                                     Reveal,
                                     RevealSelected };
bool EditVisibility(state::Registry &, std::span<const uint32_t> meshes, Element, EditVisibilityOperation);
