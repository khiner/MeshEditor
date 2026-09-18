#pragma once

#include "numeric/uvec2.h"

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
using numeric::uvec2;

inline constexpr std::string_view FileName{"workspace.state"};

struct State {
    ViewCameraState ViewCamera;
    uvec2 ViewportExtent{};
    WindowVisibility Windows;
    std::string ImGuiIni;
    std::vector<TabSelection> Tabs;
    std::vector<WindowState> WindowStates;
};

State Capture(const state::Scene &, state::Entity viewport, const WindowsState &);
void Apply(state::Scene &, state::Entity viewport, WindowsState &, const State &);
// Apply pending layout and widget state after ImGui::Render().
void ApplyPending(WindowsState &);

std::vector<std::byte> Serialize(const State &);
std::optional<State> Deserialize(std::span<const std::byte>);
std::optional<State> Load(const std::filesystem::path &);
bool Save(const std::filesystem::path &, std::span<const std::byte>);
} // namespace workspace
