#pragma once

#include "Variant.h"
#include "action/Audio.h"
#include "action/Bone.h"
#include "action/Core.h"
#include "action/Io.h"
#include "action/Object.h"
#include "action/Physics.h"
#include "action/Selection.h"
#include "action/Timeline.h"
#include "action/View.h"
#include "viewport/ViewportInteractionState.h"

namespace action {
// One alternative per domain, each the domain's own action variant.
using Action = std::variant<
    Core,
    selection::Action, object::Action, view::Action,
    physics::Action, audio::Action, bone::Action, timeline::Action, io::Action>;

// Index of the domain variant holding leaf action type `L`.
template<typename L>
inline constexpr size_t DomainIndex = []<size_t... Is>(std::index_sequence<Is...>) {
    size_t index = std::variant_size_v<Action>;
    (void)((is_variant_member_v<L, std::variant_alternative_t<Is, Action>> && (index = Is, true)) || ...);
    return index;
}(std::make_index_sequence<std::variant_size_v<Action>>{});

// Wrap a leaf action in its domain's variant, merged into an Action.
template<typename L> Action MakeAction(L &&leaf) {
    using Leaf = std::decay_t<L>;
    constexpr auto D = DomainIndex<Leaf>;
    static_assert(D < std::variant_size_v<Action>, "Type is not a leaf action of any domain.");
    return Action{std::in_place_index<D>, std::variant_alternative_t<D, Action>{std::in_place_type<Leaf>, std::forward<L>(leaf)}};
}

// Invoke `f` templated over each domain variant and collect the results in a tuple.
template<typename F> auto MapDomains(F f) {
    return [&]<size_t... Ds>(std::index_sequence<Ds...>) {
        return std::tuple{f.template operator()<std::variant_alternative_t<Ds, Action>>()...};
    }(std::make_index_sequence<std::variant_size_v<Action>>{});
}

// Actions are logged for replay unless they only produce an external artifact.
// E.g. replaying a save would clobber a file.
template<typename T> inline constexpr bool Recordable = true;
template<> inline constexpr bool Recordable<io::SaveGltf> = false;
template<> inline constexpr bool Recordable<io::SaveState> = false;
// Latch state is live-only: the recorded DragGizmo already encodes the resolved transform.
template<> inline constexpr bool Recordable<view::LatchScreenTransform> = false;
template<> inline constexpr bool Recordable<view::ClearScreenTransformLatch> = false;
// View-camera navigation is not recorded. For selection replay correctness, selection actions hold the ViewProj.
// The snapshot still stores the ViewCamera.
template<> inline constexpr bool Recordable<view::OrbitViewCamera> = false;
template<> inline constexpr bool Recordable<view::ZoomViewCamera> = false;
template<> inline constexpr bool Recordable<view::ResetViewCamera> = false;
template<> inline constexpr bool Recordable<view::SetViewCameraTarget> = false;
template<> inline constexpr bool Recordable<view::SetViewCameraLens> = false;
template<> inline constexpr bool Recordable<view::SetViewCameraTargetDirection> = false;

inline bool IsRecordable(const Action &a) {
    return std::visit([](const auto &dv) { return std::visit([]<typename L>(const L &) { return Recordable<L>; }, dv); }, a);
}
} // namespace action
