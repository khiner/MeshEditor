#include "WorkspaceState.h"

#include "File.h"
#include "action/SerializeNumeric.h"
#include "viewport/ViewCameraSerialize.h"
#include "viewport/ViewportDisplay.h"

#include <entt/entity/registry.hpp>
#include <imgui_internal.h>
#include <zpp_bits.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <ranges>

namespace workspace {
namespace {
constexpr uint32_t Version = 4;

constexpr auto SerializeWindowVisibility(auto &archive, auto &visibility) {
    return archive(
        visibility.SceneControls,
        visibility.Viewport,
        visibility.Animation,
        visibility.ImGuiDemo,
        visibility.ImSpinnerDemo,
        visibility.ImPlotDemo,
        visibility.Debug
    );
}

void MergePending(auto &values, auto &&pending, auto key) {
    for (const auto &value : pending) {
        const auto existing = std::ranges::find(values, value.*key, key);
        if (existing == values.end()) values.push_back(value);
        else *existing = value;
    }
    std::ranges::sort(values, {}, key);
}

std::vector<TabSelection> CaptureTabs(const WindowsState &windows) {
    std::vector<TabSelection> tabs;
    if (ImGui::GetCurrentContext()) {
        auto &pool = GImGui->TabBars;
        tabs.reserve(size_t(pool.GetAliveCount()) + windows.PendingTabs.size());
        for (int i = 0; i < pool.GetMapSize(); ++i) {
            const auto *bar = pool.TryGetMapData(i);
            if (bar && !(bar->Flags & ImGuiTabBarFlags_DockNode) && bar->SelectedTabId) {
                tabs.push_back({bar->ID, bar->SelectedTabId});
            }
        }
    }
    MergePending(tabs, windows.PendingTabs, &TabSelection::Bar);
    return tabs;
}

std::vector<WindowState> CaptureWindows(const WindowsState &windows) {
    std::vector<WindowState> states;
    if (ImGui::GetCurrentContext()) {
        for (const auto *window : GImGui->Windows) {
            if (window->IsFallbackWindow) continue;
            if (window->Flags & (ImGuiWindowFlags_Popup | ImGuiWindowFlags_Tooltip)) continue;
            if ((window->Flags & ImGuiWindowFlags_NoSavedSettings) && !(window->Flags & ImGuiWindowFlags_ChildWindow)) continue;
            auto &state = states.emplace_back(WindowState{window->ID, window->Scroll.x, window->Scroll.y, {}});
            for (const auto &entry : window->StateStorage.Data) {
                // App window storage contains only 32-bit scalars (int, bool, float), never pointers.
                uint32_t value;
                std::memcpy(&value, &entry.val_i, sizeof(value));
                state.Storage.push_back({entry.key, value});
            }
        }
    }
    MergePending(states, windows.PendingWindows | std::views::transform(&PendingWindowState::Value), &WindowState::Window);
    return states;
}
} // namespace

State Capture(const entt::registry &r, entt::entity viewport, const WindowsState &windows) {
    size_t ini_size = 0;
    const char *ini = ImGui::GetCurrentContext() ? ImGui::SaveIniSettingsToMemory(&ini_size) : nullptr;
    return {
        .ViewCamera = GetViewCameraState(r, viewport),
        .ViewportExtent = r.ctx().get<const ViewportExtent>().Value,
        .Windows = GetWindowVisibility(windows),
        .ImGuiIni = ini ? std::string{ini, ini_size} : std::string{},
        .Tabs = CaptureTabs(windows),
        .WindowStates = CaptureWindows(windows),
    };
}

void Apply(entt::registry &r, entt::entity viewport, WindowsState &windows, const State &state) {
    r.ctx().get<ViewportExtent>().Value = state.ViewportExtent;
    SetViewCameraState(r, viewport, state.ViewCamera);
    SetWindowVisibility(windows, state.Windows);
    if (ImGui::GetCurrentContext() && !state.ImGuiIni.empty()) {
        ImGui::LoadIniSettingsFromMemory(state.ImGuiIni.data(), state.ImGuiIni.size());
        windows.LayoutLoaded = true;
    }
    windows.PendingTabs = state.Tabs;
    windows.PendingWindows.clear();
    for (const auto &window : state.WindowStates) windows.PendingWindows.push_back({window});
}

void ApplyPending(WindowsState &windows) {
    if (!ImGui::GetCurrentContext()) return;
    std::erase_if(windows.PendingTabs, [](const TabSelection &selection) {
        auto *bar = ImGui::TabBarFindByID(selection.Bar);
        if (!bar) return false;
        if (!ImGui::TabBarFindTabByID(bar, selection.Tab) || bar->SelectedTabId == selection.Tab) return true;
        bar->NextSelectedTabId = selection.Tab;
        return false;
    });
    std::erase_if(windows.PendingWindows, [](PendingWindowState &pending) {
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
    const bool has_saved_view = state.ViewCamera.LookThroughSaved.has_value();
    if (zpp::bits::failure(archive(Version, state.ViewCamera.Active, has_saved_view)) ||
        (has_saved_view && zpp::bits::failure(archive(*state.ViewCamera.LookThroughSaved))) ||
        zpp::bits::failure(archive(state.ViewportExtent)) ||
        zpp::bits::failure(SerializeWindowVisibility(archive, state.Windows)) ||
        zpp::bits::failure(archive(state.ImGuiIni, state.Tabs, state.WindowStates))) {
        return {};
    }
    bytes.resize(archive.position());
    return bytes;
}

std::optional<State> Deserialize(std::span<const std::byte> bytes) {
    zpp::bits::in archive{bytes};
    uint32_t version{};
    State state{
        .ViewCamera = {ViewCamera{vec3{0, 0, 1}, vec3{0}, Camera{}}, std::nullopt},
        .ViewportExtent = {},
        .Windows = {},
        .ImGuiIni = {},
        .Tabs = {},
        .WindowStates = {},
    };
    bool has_saved_view{};
    if (zpp::bits::failure(archive(version, state.ViewCamera.Active, has_saved_view)) || version == 0 || version > Version) {
        return std::nullopt;
    }
    if (has_saved_view) {
        state.ViewCamera.LookThroughSaved.emplace(vec3{0, 0, 1}, vec3{0}, Camera{});
        if (zpp::bits::failure(archive(*state.ViewCamera.LookThroughSaved))) return std::nullopt;
    }
    if (zpp::bits::failure(archive(state.ViewportExtent)) ||
        zpp::bits::failure(SerializeWindowVisibility(archive, state.Windows)) ||
        zpp::bits::failure(archive(state.ImGuiIni))) {
        return std::nullopt;
    }
    if (version >= 2 && zpp::bits::failure(archive(state.Tabs))) return std::nullopt;
    if (version == 3) {
        struct WindowScroll {
            uint32_t Window{};
            float X{}, Y{};
        };
        std::vector<WindowScroll> scroll;
        if (zpp::bits::failure(archive(scroll))) return std::nullopt;
        for (const auto &value : scroll) state.WindowStates.push_back({value.Window, value.X, value.Y, {}});
    }
    if (version >= 4 && zpp::bits::failure(archive(state.WindowStates))) return std::nullopt;
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
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream out{temporary, std::ios::binary | std::ios::trunc};
        out.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
        if (!out) return false;
    }
    std::filesystem::rename(temporary, path, ec);
    if (!ec) return true;
    std::filesystem::remove(temporary, ec);
    return false;
}
} // namespace workspace
