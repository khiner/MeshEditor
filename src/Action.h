#pragma once

#include "action/Audio.h"
#include "action/Bone.h"
#include "action/Core.h"
#include "action/Io.h"
#include "action/Object.h"
#include "action/Physics.h"
#include "action/Selection.h"
#include "action/Timeline.h"
#include "action/View.h"

namespace action {
// Each domain owns its own action surface (its explicit actions plus any domain-specific
// Update/Replace alternatives) via its `Action` typedef; the top-level variant is their union
// plus Core. Apply(Action) routes each leaf to the owning domain's Apply.
using Action = MergedVariantT<
    Core,
    selection::Action, object::Action, view::Action,
    physics::Action, audio::Action, bone::Action, timeline::Action, io::Action>;

using FallibleAction = io::FallibleActions;

// First action per frame applies - subsequent actions are dropped.
using Emit = void (*)(Action);

// Promote a sub-domain Action variant into the top-level Action.
template<typename... Ts>
void Assign(Emit emit, std::variant<Ts...> v) {
    std::visit([&](auto &&x) { emit(std::forward<decltype(x)>(x)); }, std::move(v));
}
} // namespace action
