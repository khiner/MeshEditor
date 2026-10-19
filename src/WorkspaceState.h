#pragma once

#include "Window.h"
#include "audio/AudioTypes.h"
#include "viewport/Viewport.h"

#include <filesystem>

namespace workspace {
using numeric::uvec2;

inline constexpr std::string_view FileName{"workspace.state"};

// A viewport's identity, visibility, and view values.
struct ViewportEntry {
    uint32_t Id;
    bool Visible;
    ViewportState State;
};

struct State {
    std::vector<ViewportEntry> Viewports; // In service order.
    AudioOutputConfig AudioOutput;
    WindowVisibility Windows;
    NodeCanvas NodesCanvas;
    std::string ImGuiIni;
    std::vector<TabSelection> Tabs;
    std::vector<WindowState> WindowStates;
};

State Capture(const state::Registry &, const WindowsState &);
// Updates the viewports matching a stored Id in place, creates the rest, and removes the ones the state omits, keeping at least one.
void Apply(state::Registry &, WindowsState &, const State &);
// Apply pending layout and widget state after ImGui::Render().
void ApplyPending(WindowsState &);

std::vector<std::byte> Serialize(const State &);
std::optional<State> Deserialize(std::span<const std::byte>);
std::optional<State> Load(const std::filesystem::path &);
bool Save(const std::filesystem::path &, std::span<const std::byte>);
} // namespace workspace
