#pragma once

#include "Variant.h"
#include "action/Animation.h"
#include "action/Audio.h"
#include "action/Bone.h"
#include "action/Emit.h"
#include "action/Io.h"
#include "action/Mesh.h"
#include "action/Nodes.h"
#include "action/Object.h"
#include "action/Physics.h"
#include "action/Scene.h"
#include "action/Selection.h"
#include "action/Timeline.h"
#include "action/View.h"

namespace action {
using Action = std::variant<
    Core,
    selection::Action, object::Action, view::Action,
    physics::Action, audio::Action, bone::Action, timeline::Action, io::Action, animation::Action, mesh::Action, nodes::Action, scene::Action>;

static_assert(sizeof(Action) <= 64, "Keep actions at or below 64 bytes");

template<typename L>
inline constexpr size_t DomainIndex = []<size_t... Is>(std::index_sequence<Is...>) {
    size_t index = std::variant_size_v<Action>;
    (void)((is_variant_member_v<L, std::variant_alternative_t<Is, Action>> && (index = Is, true)) || ...);
    return index;
}(std::make_index_sequence<std::variant_size_v<Action>>{});

template<typename L> Action MakeAction(L &&leaf) {
    using Leaf = std::decay_t<L>;
    constexpr auto D = DomainIndex<Leaf>;
    static_assert(D < std::variant_size_v<Action>, "Type is not a leaf action of any domain.");
    return Action{std::in_place_index<D>, std::variant_alternative_t<D, Action>{std::in_place_type<Leaf>, std::forward<L>(leaf)}};
}

auto MapDomains(auto f) {
    return [&]<size_t... Ds>(std::index_sequence<Ds...>) {
        return std::tuple{f.template operator()<std::variant_alternative_t<Ds, Action>>()...};
    }(std::make_index_sequence<std::variant_size_v<Action>>{});
}

// Exclude actions with external side effects from replay.
template<typename T> inline constexpr bool Recordable = true;
template<> inline constexpr bool Recordable<io::SaveGltf> = false;
// Latch state is live-only: the recorded TransformSelection already encodes the resolved transform.
template<> inline constexpr bool Recordable<view::LatchTransform> = false;
// An action naming a viewport changes its view values, which belong to the workspace.
template<typename T>
    requires requires(const T &a) { a.Viewport; }
inline constexpr bool Recordable<T> = false;
// A viewport's existence and scene belong to the document.
template<> inline constexpr bool Recordable<view::SetViewportScene> = true;
template<> inline constexpr bool Recordable<view::RemoveViewport> = true;
// A presentation sets up the active viewport's view and doesn't change document state.
template<> inline constexpr bool Recordable<timeline::EnterPresentation> = false;
// The output device is a user preference.
template<> inline constexpr bool Recordable<audio::SetOutputDevice> = false;

void Emit(Action, Phase = Phase::Record);

template<typename A, typename F> decltype(auto) VisitLeaf(A &&a, F &&fn) {
    return std::visit([&](auto &&domain) { return std::visit(fn, domain); }, std::forward<A>(a));
}

inline bool IsRecordable(const Action &a) {
    if (const auto *core = std::get_if<Core>(&a)) return IsRecordable(*core);
    return VisitLeaf(a, []<typename L>(const L &) { return Recordable<L>; });
}

// Each later step of a preview's stage starts over from the stage base.
// Placement-drag operators manage their own previews, and an element transform previews each update.
template<typename T> inline constexpr bool Previews = !std::same_as<T, mesh::Extrude> && !std::same_as<T, mesh::Duplicate> && !std::same_as<T, mesh::Rip>;

inline bool IsPreview(const Action &a) {
    if (const auto *v = std::get_if<view::Action>(&a)) return std::holds_alternative<view::TransformElements>(*v);
    const auto *m = std::get_if<mesh::Action>(&a);
    return m && std::visit([]<typename L>(const L &) { return Previews<L>; }, *m);
}
// Whether a later preview step of a stage edits the nodes the stage's first step appended instead of appending again from the stage base.
// An identity transform appends nothing, so its step restores the stage base.
inline bool PatchesPreview(const Action &a) {
    if (const auto *v = std::get_if<view::Action>(&a)) {
        const auto *transform = std::get_if<view::TransformElements>(v);
        return transform && transform->Delta != Transform{};
    }
    const auto *m = std::get_if<mesh::Action>(&a);
    return m && mesh::Patches(*m);
}
} // namespace action
