#include "Action.h"

#include "action/Audio.h"
#include "action/Bone.h"
#include "action/Io.h"
#include "action/Object.h"
#include "action/Physics.h"
#include "action/Selection.h"
#include "action/Timeline.h"
#include "action/View.h"

using namespace action;

namespace {
using Action = MergedVariantT<
    Core,
    selection::Action, object::Action, view::Action,
    action::physics::Action, audio::Action, bone::Action, timeline::Action, io::Action>;
std::optional<Action> Staged;
} // namespace

namespace action {
template<typename ActionType> void Emit(ActionType a) {
    if (!Staged) Staged.emplace(std::move(a)); // First action emitted in the frame wins.
}

void ApplyEmitted(entt::registry &r, entt::entity viewport) {
    if (!Staged) return;

    std::visit(
        [&]<typename T>(T &&a) {
            using L = std::decay_t<T>;
            if constexpr (is_variant_member_v<L, selection::Action>) selection::Apply(r, viewport, selection::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, object::Action>) object::Apply(r, viewport, object::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, view::Action>) view::Apply(r, viewport, view::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, action::physics::Action>) action::physics::Apply(r, viewport, action::physics::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, audio::Action>) audio::Apply(r, viewport, audio::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, bone::Action>) bone::Apply(r, viewport, bone::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, timeline::Action>) timeline::Apply(r, viewport, timeline::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, io::Action>) io::Apply(r, viewport, io::Action{std::forward<T>(a)});
            else Apply(r, viewport, Core{std::forward<T>(a)});
        },
        std::move(*Staged)
    );
    Staged.reset();
}

std::size_t ActionSize() { return sizeof(Action); }
} // namespace action

namespace {
// Instantiate Emit for all action types
using EmitPtr = void (*)();
template<std::size_t... I>
std::array<EmitPtr, sizeof...(I)> AllEmits(std::index_sequence<I...>) {
    return {reinterpret_cast<EmitPtr>(static_cast<void (*)(std::variant_alternative_t<I, Action>)>(&Emit))...};
}
const auto _ = AllEmits(std::make_index_sequence<std::variant_size_v<Action>>{});
} // namespace
