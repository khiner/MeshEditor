#pragma once

#include "state/Slot.h"

namespace mtl {
struct BindlessSet;
struct Context;
struct LibraryCache;
struct ProgramRegistry;

using MetalSchema = state::Schema<state::MetalBase, state::MetalCount, BindlessSet, Context, LibraryCache, ProgramRegistry>;
} // namespace mtl
template<typename T>
    requires mtl::MetalSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = mtl::MetalSchema::Of<T>();
