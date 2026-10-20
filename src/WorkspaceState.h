#pragma once

#include "Panel.h"
#include "audio/AudioTypes.h"
#include "viewport/Viewport.h"

#include <filesystem>

namespace workspace {
using numeric::uvec2;

inline constexpr std::string_view FileName{"workspace.state"};

struct ViewportEntry {
    state::Entity Viewport;
    ViewportState State;
};

struct State {
    std::vector<ViewportEntry> Viewports; // Every viewport entity the session has held, in entity index order.
    state::Entity ActiveViewport{state::Null};
    AudioOutputConfig AudioOutput;
    PanelVisibility Panels;
    NodeCanvas NodesCanvas;
    std::string ImGuiIni;
    std::vector<TabSelection> Tabs;
    std::vector<PanelState> PanelStates;
};

State Capture(const state::Registry &, const PanelsState &);
// Replaces the stored viewports' values, and activates the stored active viewport when the document holds it.
void Apply(state::Registry &, PanelsState &, const State &);
// Apply pending layout and widget state after ImGui::Render().
void ApplyPending(PanelsState &);

std::vector<std::byte> Serialize(const State &);
std::optional<State> Deserialize(std::span<const std::byte>);
std::optional<State> Load(const std::filesystem::path &);
bool Save(const std::filesystem::path &, std::span<const std::byte>);
} // namespace workspace
