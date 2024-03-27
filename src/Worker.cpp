#include "Worker.h"

#include "imgui.h"
#include "imspinner.h"

using namespace ImGui;

void Worker::Launch(const std::function<void()> &work) {
    OpenPopup(WorkingMessage.c_str());
    if (Thread.joinable()) Thread.join();
    Thread = std::thread([&]() {
        Working = true;
        work();
        Working = false;
    });
}

void Worker::Launch() {
    Launch(Work);
}

void Worker::RenderLauncher(const std::function<void()> &work) {
    if (Button(LaunchLabel.c_str())) {
        Launch(work);
    }
}

bool Worker::Render() {
    bool completed = false;
    SetNextWindowPos(GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
    SetNextWindowSize(GetMainViewport()->Size / 4);
    if (BeginPopupModal(WorkingMessage.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        const auto &ws = GetWindowSize();
        const float spinner_size = std::min(ws.x, ws.y) / 2;
        SetCursorPos((ws - ImVec2{spinner_size, spinner_size}) / 2 + ImVec2(0, GetTextLineHeight()));
        ImSpinner::SpinnerMultiFadeDots(WorkingMessage.c_str(), spinner_size / 2, 3);
        if (!Working) {
            Thread.join();
            CloseCurrentPopup();
            completed = true;
        }
        EndPopup();
    }
    return completed;
}
