#pragma once

#include <cstdint>
#include <vector>

struct TabSelection {
    uint32_t Bar{}, Tab{};
    bool operator==(const TabSelection &) const = default;
};

struct Window {
    const char *Name{""};
    bool Visible{true};
};

struct WindowsState {
    Window
        SceneControls{"Scene controls"},
        Viewport{"Viewport"},
        Animation{"Animation"},
        ImGuiDemo{"Dear ImGui Demo", false},
        ImSpinnerDemo{"ImSpinner Demo", false},
        ImPlotDemo{"ImPlot Demo", false},
        Debug{"Debug", false};
    std::vector<TabSelection> PendingTabs;
    bool LayoutLoaded{false};
};

struct WindowVisibility {
    bool SceneControls{true};
    bool Viewport{true};
    bool Animation{true};
    bool ImGuiDemo{false};
    bool ImSpinnerDemo{false};
    bool ImPlotDemo{false};
    bool Debug{false};
};

inline WindowVisibility GetWindowVisibility(const WindowsState &windows) {
    return {
        windows.SceneControls.Visible,
        windows.Viewport.Visible,
        windows.Animation.Visible,
        windows.ImGuiDemo.Visible,
        windows.ImSpinnerDemo.Visible,
        windows.ImPlotDemo.Visible,
        windows.Debug.Visible,
    };
}

inline void SetWindowVisibility(WindowsState &windows, const WindowVisibility &visibility) {
    windows.SceneControls.Visible = visibility.SceneControls;
    windows.Viewport.Visible = visibility.Viewport;
    windows.Animation.Visible = visibility.Animation;
    windows.ImGuiDemo.Visible = visibility.ImGuiDemo;
    windows.ImSpinnerDemo.Visible = visibility.ImSpinnerDemo;
    windows.ImPlotDemo.Visible = visibility.ImPlotDemo;
    windows.Debug.Visible = visibility.Debug;
}
