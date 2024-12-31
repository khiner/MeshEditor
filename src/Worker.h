#pragma once

#include <functional>
#include <future>
#include <string>
#include <string_view>

#include "imgui.h"
#include "imspinner.h"

template<typename Result>
struct Worker {
    Worker(std::string_view working_message, std::function<Result()> work = {})
        : WorkingMessage(working_message) {
        ImGui::OpenPopup(WorkingMessage.c_str());
        ResultFuture = std::async(std::launch::async, [work]() -> Result { return work(); });
    }

    ~Worker() = default;

    std::optional<Result> Render() {
        std::optional<Result> result = {};
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
        ImGui::SetNextWindowSize(ImGui::GetMainViewport()->Size / 4);
        if (ImGui::BeginPopupModal(WorkingMessage.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            const auto &ws = ImGui::GetWindowSize();
            const float spinner_size = std::min(ws.x, ws.y) / 2;
            ImGui::SetCursorPos((ws - ImVec2{spinner_size, spinner_size}) / 2 + ImVec2(0, ImGui::GetTextLineHeight()));
            ImSpinner::SpinnerMultiFadeDots(WorkingMessage.c_str(), spinner_size / 2, 3);
            if (ResultFuture.valid() && ResultFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                result = std::make_optional(std::move(ResultFuture.get()));
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        return result;
    }

    std::string WorkingMessage;
    std::future<Result> ResultFuture;
};
