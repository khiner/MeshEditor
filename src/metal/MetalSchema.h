#pragma once

#include "state/Slot.h"

namespace mtl {
struct BindlessSet;
struct Context;
struct LibraryCache;

using MetalSchema = state::Schema<state::MetalBase, state::MetalCount, BindlessSet, Context, LibraryCache>;
} // namespace mtl
template<typename T>
    requires mtl::MetalSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = mtl::MetalSchema::Of<T>();
