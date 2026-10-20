#include "WorkspaceState.h"

#include "File.h"
#include "render/GpuBuffers.h"
#include "viewport/ViewCameraSerialize.h"
#include "viewport/ViewportConsumerFence.h"

#include "state/Registry.h"
#include <Metal/MTLCommandBuffer.hpp>
#include <imgui_internal.h>
#include <zpp_bits.h>

namespace workspace {
namespace {
void ApplyPendingIni(PanelsState &panels) {
    if (!ImGui::GetCurrentContext() || GImGui->WithinFrameScope || panels.PendingIni.empty()) return;
    // Previous focus would override the restored dock-tab selection.
    ImGui::SetWindowFocus(nullptr);
    ImGui::LoadIniSettingsFromMemory(panels.PendingIni.data(), panels.PendingIni.size());
    panels.PendingIni.clear();
}

void MergePending(auto &values, auto &&pending, auto key) {
    for (const auto &value : pending) {
        const auto existing = std::ranges::find(values, value.*key, key);
        if (existing == values.end()) values.push_back(value);
        else *existing = value;
    }
    std::ranges::sort(values, {}, key);
}

std::vector<TabSelection> CaptureTabs(const PanelsState &panels) {
    std::vector<TabSelection> tabs;
    if (ImGui::GetCurrentContext()) {
        auto &pool = GImGui->TabBars;
        tabs.reserve(size_t(pool.GetAliveCount()) + panels.PendingTabs.size());
        for (int i = 0; i < pool.GetMapSize(); ++i) {
            const auto *bar = pool.TryGetMapData(i);
            if (bar && !(bar->Flags & ImGuiTabBarFlags_DockNode) && bar->SelectedTabId) {
                tabs.push_back({bar->ID, bar->SelectedTabId});
            }
        }
    }
    MergePending(tabs, panels.PendingTabs, &TabSelection::Bar);
    return tabs;
}

std::vector<PanelState> CapturePanels(const PanelsState &panels) {
    std::vector<PanelState> states;
    if (ImGui::GetCurrentContext()) {
        for (const auto *window : GImGui->Windows) {
            if (window->IsFallbackWindow) continue;
            if (window->Flags & (ImGuiWindowFlags_Popup | ImGuiWindowFlags_Tooltip)) continue;
            if ((window->Flags & ImGuiWindowFlags_NoSavedSettings) && !(window->Flags & ImGuiWindowFlags_ChildWindow)) continue;
            auto &state = states.emplace_back(PanelState{window->ID, window->Scroll.x, window->Scroll.y, {}});
            for (const auto &entry : window->StateStorage.Data) {
                // App window storage contains only 32-bit scalars (int, bool, float), never pointers.
                uint32_t value;
                std::memcpy(&value, &entry.val_i, sizeof(value));
                state.Storage.push_back({entry.key, value});
            }
        }
    }
    MergePending(states, panels.PendingPanels | std::views::transform(&PendingPanelState::Value), &PanelState::Window);
    return states;
}
} // namespace

State Capture(const state::Registry &r, const PanelsState &panels) {
    size_t ini_size = 0;
    const char *ini = ImGui::GetCurrentContext() && panels.PendingIni.empty() ? ImGui::SaveIniSettingsToMemory(&ini_size) : nullptr;
    const auto &viewports = r.Resources.get<const ::Viewports>().All;
    return {
        .Viewports = viewports | std::views::transform([](const Viewport &v) { return ViewportEntry{v.Id, v.Visible, v.State}; }) | std::ranges::to<std::vector>(),
        .AudioOutput = r.get<const AudioOutputConfig>(state::Document),
        .Panels = GetPanelVisibility(panels),
        .NodesCanvas = panels.Nodes.Canvas,
        .ImGuiIni = ini ? std::string{ini, ini_size} : panels.PendingIni,
        .Tabs = CaptureTabs(panels),
        .PanelStates = CapturePanels(panels),
    };
}

void Apply(state::Registry &r, PanelsState &panels, const State &state) {
    auto &viewports = r.Resources.get<::Viewports>();
    // The live UI frame stops sampling the images of the viewports the state omits before they go.
    const bool keeps_all = std::ranges::all_of(viewports.All, [&](const Viewport &v) { return std::ranges::contains(state.Viewports, v.Id, &ViewportEntry::Id); });
    if (auto *consumer = r.Resources.get<const ViewportConsumerFence>().Value; consumer && !keeps_all) consumer->waitUntilCompleted();
    auto &buffers = r.Resources.get<GpuBuffers>().Ctx;
    auto &slots = r.Resources.get<mtl::BindlessSet>();
    std::vector<Viewport> applied;
    for (const auto &entry : state.Viewports) {
        auto *existing = viewports.Find(entry.Id);
        auto &viewport = existing ? applied.emplace_back(std::move(*existing)) : applied.emplace_back(entry.Id, buffers, slots);
        viewport.Visible = entry.Visible;
        viewport.State = entry.State;
    }
    if (applied.empty()) applied.push_back(std::move(viewports.All.front()));
    viewports.All = std::move(applied);
    viewports.ActiveIndex = 0;
    // The audio settle reconciles the device with the published config.
    r.replace<AudioOutputConfig>(state::Document, state.AudioOutput);
    SetPanelVisibility(panels, state.Panels);
    panels.Nodes.Canvas = state.NodesCanvas;
    panels.PendingIni = state.ImGuiIni;
    if (!panels.PendingIni.empty()) panels.LayoutLoaded = true;
    ApplyPendingIni(panels);
    panels.PendingTabs = state.Tabs;
    panels.PendingPanels.clear();
    for (const auto &panel : state.PanelStates) panels.PendingPanels.push_back({panel});
}

void ApplyPending(PanelsState &panels) {
    if (!ImGui::GetCurrentContext()) return;
    ApplyPendingIni(panels);
    std::erase_if(panels.PendingTabs, [](const TabSelection &selection) {
        auto *bar = ImGui::TabBarFindByID(selection.Bar);
        if (!bar) return false;
        if (!ImGui::TabBarFindTabByID(bar, selection.Tab) || bar->SelectedTabId == selection.Tab) return true;
        bar->NextSelectedTabId = selection.Tab;
        return false;
    });
    std::erase_if(panels.PendingPanels, [](PendingPanelState &pending) {
        auto *window = ImGui::FindWindowByID(pending.Value.Window);
        if (!window) return false;
        if (!pending.StorageApplied) {
            window->StateStorage.Clear();
            for (const auto &entry : pending.Value.Storage) {
                window->StateStorage.SetInt(entry.Key, std::bit_cast<int32_t>(entry.Value));
            }
            pending.StorageApplied = true;
            // Rebuild content from restored widget state before setting scroll targets.
            return false;
        }
        if (!window->Active) return false;
        if (pending.ScrollApplied) return true;
        if (window->Hidden || window->Appearing) return false;
        // Begin() clamps scroll targets using the preceding frame's content size.
        ImGui::SetScrollX(window, pending.Value.X);
        ImGui::SetScrollY(window, pending.Value.Y);
        pending.ScrollApplied = true;
        return false;
    });
}

std::vector<std::byte> Serialize(const State &state) {
    std::vector<std::byte> bytes;
    zpp::bits::out archive{bytes};
    if (zpp::bits::failure(archive(state.Viewports, state.AudioOutput, state.Panels, state.NodesCanvas, state.ImGuiIni, state.Tabs, state.PanelStates))) return {};
    bytes.resize(archive.position());
    return bytes;
}

std::optional<State> Deserialize(std::span<const std::byte> bytes) {
    zpp::bits::in archive{bytes};
    State state;
    if (zpp::bits::failure(archive(state.Viewports, state.AudioOutput, state.Panels, state.NodesCanvas, state.ImGuiIni, state.Tabs, state.PanelStates))) return std::nullopt;
    return state;
}

std::optional<State> Load(const std::filesystem::path &path) {
    const auto bytes = File::Read(path);
    return bytes ? Deserialize(*bytes) : std::nullopt;
}

bool Save(const std::filesystem::path &path, std::span<const std::byte> bytes) {
    if (bytes.empty()) return false;

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    return !ec && bool(File::WriteAtomic(path, bytes));
}
} // namespace workspace
