#include "Apply.h"

// Routes each action leaf to the owning domain's action::{domain}::Apply, which lives in
// src/action/{Domain}.cpp alongside that domain's dependencies. Anything not in a domain variant
// is a Core action (trivial Update/Replace/Tag/DestroyEntity) handled by action::Apply(Core).
void Apply(entt::registry &r, entt::entity viewport, action::Action action) {
    std::visit(
        [&]<typename T>(T &&a) {
            using L = std::decay_t<T>;
            if constexpr (is_variant_member_v<L, action::selection::Action>) action::selection::Apply(r, viewport, action::selection::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, action::object::Action>) action::object::Apply(r, viewport, action::object::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, action::view::Action>) action::view::Apply(r, viewport, action::view::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, action::physics::Action>) action::physics::Apply(r, viewport, action::physics::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, action::audio::Action>) action::audio::Apply(r, viewport, action::audio::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, action::bone::Action>) action::bone::Apply(r, viewport, action::bone::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, action::timeline::Action>) action::timeline::Apply(r, viewport, action::timeline::Action{std::forward<T>(a)});
            else if constexpr (is_variant_member_v<L, action::io::Action>) action::io::Apply(r, viewport, action::io::Action{std::forward<T>(a)});
            else action::Apply(r, viewport, action::Core{std::forward<T>(a)});
        },
        std::move(action)
    );
}

std::expected<void, std::string> Apply(entt::registry &r, entt::entity viewport, const action::FallibleAction &action) {
    return action::io::Apply(r, viewport, action);
}
