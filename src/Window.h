#pragma once

struct Window {
    const char *Name{""};
    bool Visible{true};
};

struct WindowsState {
    Window SceneControls{"Scene controls"};
    Window Scene{"Scene"};
    Window ImGuiDemo{"Dear ImGui Demo", true};
};
