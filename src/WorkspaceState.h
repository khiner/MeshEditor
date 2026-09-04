#pragma once

#include "Window.h"
#include "numeric/vec2.h"
#include "viewport/ViewCameraOps.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace workspace {
inline constexpr std::string_view FileName{"workspace.state"};

struct State {
    ViewCameraState ViewCamera;
    uvec2 ViewportExtent{};
    WindowVisibility Windows;
    std::string ImGuiIni;
    std::vector<TabSelection> Tabs;
};

State Capture(const entt::registry &, entt::entity viewport, const WindowsState &);
void Apply(entt::registry &, entt::entity viewport, WindowsState &, const State &);
void ApplyPendingTabs(WindowsState &);

std::vector<std::byte> Serialize(const State &);
std::optional<State> Deserialize(std::span<const std::byte>);
std::optional<State> Load(const std::filesystem::path &);
bool Save(const std::filesystem::path &, std::span<const std::byte>);
} // namespace workspace
