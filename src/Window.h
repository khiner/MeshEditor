#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct TabSelection {
    uint32_t Bar{}, Tab{};
    bool operator==(const TabSelection &) const = default;
};

struct WindowState {
    struct StorageValue {
        uint32_t Key{}, Value{};
        bool operator==(const StorageValue &) const = default;
    };
    uint32_t Window{};
    float X{}, Y{};
    std::vector<StorageValue> Storage;
    bool operator==(const WindowState &) const = default;
};

struct PendingWindowState {
    WindowState Value;
    bool StorageApplied{false}, ScrollApplied{false};
};

struct Window {
    const char *Name{""};
    bool Visible{true};
};

struct HistoryRow {
    int Node;
    bool Editable; // Whether any of the node's actions has parameters.
};
struct HistoryWindow : Window {
    bool ValidateRequested{false};
    uint64_t TreeRevision{UINT64_MAX};
    std::vector<HistoryRow> Rows{};
    int EditorNode{-1}; // The node whose editor the open state belongs to.
    bool EditorOpen{true};
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
    HistoryWindow History{{"History"}};
    std::string PendingIni;
    std::vector<TabSelection> PendingTabs;
    std::vector<PendingWindowState> PendingWindows;
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
    bool History{true};
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
        windows.History.Visible,
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
    windows.History.Visible = visibility.History;
}
