#pragma once

#include "imgui.h"
#include "imspinner.h"

#include <chrono>
#include <future>
#include <string_view>

using namespace std::chrono_literals;

template<typename Result>
struct Worker {
    Worker(std::string_view title, auto &&work)
        : Title(title), ResultFuture(std::async(std::launch::async, std::forward<decltype(work)>(work))) {
        ImGui::OpenPopup(Title.data());
    }
    ~Worker() = default;

    std::optional<Result> Render() {
        using namespace ImGui;

        SetNextWindowPos(GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
        SetNextWindowSize(GetMainViewport()->Size / 4);
        if (BeginPopupModal(Title.data(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            const auto &ws = GetWindowSize();
            const float spinner_size = std::min(ws.x, ws.y) / 2;
            const ImVec2 spinner_pos = (ws - ImVec2{spinner_size, spinner_size}) / 2;
            SetCursorPos({(ws.x - CalcTextSize(Message.data()).x) / 2, spinner_pos.y - GetTextLineHeight()});
            TextUnformatted(Message.data());
            Spacing();
            Spacing();
            SetCursorPosX(spinner_pos.x);
            ImSpinner::SpinnerMultiFadeDots(Title.data(), spinner_size / 2, 3);
            std::optional<Result> result;
            if (ResultFuture.wait_for(0s) == std::future_status::ready) {
                result = std::move(ResultFuture.get());
                CloseCurrentPopup();
            }
            EndPopup();
            return result;
        }
        return {};
    }

    void SetMessage(std::string_view message) { Message = message; }

private:
    std::string Title;
    std::string Message{""};
    std::future<Result> ResultFuture;
};
