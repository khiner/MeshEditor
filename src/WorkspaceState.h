#pragma once

#include "Panel.h"

#include <filesystem>

namespace workspace {
inline constexpr std::string_view FileName{"workspace.state"};

struct State {
    std::vector<std::byte> Components; // The Workspace components of live and dead identities, encoded by snapshot::SerializeWorkspace.
    PanelVisibility Panels;
    NodeCanvas NodesCanvas;
    std::string ImGuiIni;
    std::vector<TabSelection> Tabs;
    std::vector<PanelState> PanelStates;
};

State Capture(const state::Registry &, const PanelsState &);
// Places the stored Workspace components on their live identities, or into retention for identities that are dead.
void Apply(state::Registry &, PanelsState &, const State &);
// Apply pending layout and widget state after ImGui::Render().
void ApplyPending(PanelsState &);

std::vector<std::byte> Serialize(const State &);
std::optional<State> Deserialize(std::span<const std::byte>);
std::optional<State> Load(const std::filesystem::path &);
bool Save(const std::filesystem::path &, std::span<const std::byte>);
} // namespace workspace
