#pragma once

#include "Action.h"

#include <entt/entity/fwd.hpp>

#include <expected>
#include <string>

// Routes each action leaf to the owning domain's action::{domain}::Apply (in src/action/{Domain}.cpp).
// Registry-only: GPU side-effects are deferred via PendingX components that ProcessComponentEvents observes.
// Taken by value: the merged variant holds move-only alternatives, so the dispatcher moves each leaf into its domain variant.
void Apply(entt::registry &, entt::entity viewport, action::Action);
// File IO (LoadGltf/SaveGltf/LoadRealImpact) runs GPU work synchronously and reports failure inline.
std::expected<void, std::string> Apply(entt::registry &, entt::entity viewport, const action::FallibleAction &);
