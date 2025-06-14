#pragma once

struct Window {
    const char *Name{""};
    bool Visible{true};
};

struct WindowsState {
    Window
        SceneControls{"Scene controls"},
        Scene{"Scene"},
        ImGuiDemo{"Dear ImGui Demo", false},
        ImPlotDemo{"ImPlot Demo", false},
        Debug{"Debug", false};
};
