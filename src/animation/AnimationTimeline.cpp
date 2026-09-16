#include "animation/AnimationTimeline.h"
#include "animation/TimelineUi.h"
#include "render/SvgResource.h"
#include "viewport/ViewportIcons.h"

#include "imgui.h"

#include "state/Scene.h"

#include <algorithm>
#include <format>

using namespace ImGui;

namespace {
constexpr float HeaderHeight{20}, MinPixelsPerFrame{1}, MaxPixelsPerFrame{400};

bool IconButton(const char *id, const SvgResource *icon, ImDrawFlags corners = ImDrawFlags_RoundCornersAll, float width_scale = 1.f) {
    const float h = GetFrameHeight();
    const ImVec2 size{h * width_scale, h};
    const float icon_dim = h * 0.8f;
    static constexpr ImVec2 padding{0.5f, 0.5f};

    PushID(id);
    const bool clicked = InvisibleButton("##btn", size);
    const bool hovered = IsItemHovered();
    auto *dl = GetWindowDrawList();
    dl->AddRectFilled(GetItemRectMin() + padding, GetItemRectMax() - padding, hovered ? GetColorU32(ImGuiCol_ButtonHovered) : GetColorU32(ImGuiCol_Button), 6.0f, corners);
    if (icon) {
        const auto saved = GetCursorScreenPos();
        SetCursorScreenPos({GetItemRectMin().x + (size.x - icon_dim) * 0.5f, GetItemRectMin().y + (h - icon_dim) * 0.5f});
        icon->DrawIcon({icon_dim, icon_dim});
        SetCursorScreenPos(saved);
    }
    PopID();
    return clicked;
}

// Select a major step that prevents label overlap.
int ComputeMajorStep(float pixels_per_frame) {
    static constexpr float MinLabelSpacingPx{80};
    static constexpr int steps[]{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
    for (const int s : steps) {
        if (s * pixels_per_frame >= MinLabelSpacingPx) return s;
    }
    return steps[std::size(steps) - 1];
}
} // namespace

std::optional<action::timeline::Action> HandleTimelineShortcuts(const TimelinePlayback &playback) {
    const auto &io = GetIO();
    if (io.WantTextInput || io.NavVisible) return {};
    if (playback.Playing && Shortcut(ImGuiKey_Escape, ImGuiInputFlags_RouteGlobal)) return action::timeline::CancelPlay{};
    constexpr auto flags = ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_Repeat;
    if (Shortcut(ImGuiKey_LeftArrow, flags)) return action::timeline::OffsetFrame{-1};
    if (Shortcut(ImGuiKey_RightArrow, flags)) return action::timeline::OffsetFrame{1};
    if (Shortcut(ImGuiMod_Shift | ImGuiKey_LeftArrow, flags)) return action::timeline::JumpToStart{};
    if (Shortcut(ImGuiMod_Shift | ImGuiKey_RightArrow, flags)) return action::timeline::JumpToEnd{};
    if (Shortcut(ImGuiMod_Ctrl | ImGuiKey_LeftArrow, flags)) return action::timeline::JumpTime{.Backward = true};
    if (Shortcut(ImGuiMod_Ctrl | ImGuiKey_RightArrow, flags)) return action::timeline::JumpTime{.Backward = false};
    if (Shortcut(ImGuiKey_UpArrow, flags)) return action::timeline::JumpKeyframe{.Next = false};
    if (Shortcut(ImGuiKey_DownArrow, flags)) return action::timeline::JumpKeyframe{.Next = true};
    return {};
}

std::optional<action::timeline::Action> RenderAnimationTimeline(const TimelineRange &range, const TimelinePlayback &playback, const AnimationTimelineView &view, const TimelineNavigation &navigation, std::span<const float> keyframes, const AnimationIcons &icons, bool &scrubbing) {
    std::optional<action::timeline::Action> action;
    scrubbing = false;

    const float w = GetContentRegionAvail().x;
    const float h = GetFrameHeight();
    const auto spacing = GetStyle().ItemSpacing;

    const auto bar_origin = GetCursorScreenPos();
    {
        // The transport group holds the range jumps and play controls.
        // The delta group to its right holds the Jump Time by Delta buttons and the playback options.
        // Pause spans the reverse and forward play slots while playing.
        constexpr float TransportSlots{6}, DeltaSlots{3};
        const auto transport_p = bar_origin + ImVec2{(w - h * (TransportSlots + DeltaSlots) - spacing.x) * 0.5f, 0.f};
        SetCursorScreenPos(transport_p);
        if (IconButton("jump_start", icons.JumpStart.get(), ImDrawFlags_RoundCornersLeft)) action = action::timeline::JumpToStart{};
        SetCursorScreenPos({transport_p.x + h, transport_p.y});
        if (IconButton("prev_keyframe", icons.PrevKeyframe.get(), ImDrawFlags_RoundCornersNone)) action = action::timeline::JumpKeyframe{.Next = false};
        SetCursorScreenPos({transport_p.x + h * 2, transport_p.y});
        if (playback.Playing) {
            if (IconButton("pause", icons.Pause.get(), ImDrawFlags_RoundCornersNone, 2.f)) action = action::timeline::TogglePlay{playback.CurrentFrame};
        } else {
            if (IconButton("play_reverse", icons.PlayReverse.get(), ImDrawFlags_RoundCornersNone)) action = action::timeline::TogglePlay{playback.CurrentFrame, /*Reverse=*/true};
            SetCursorScreenPos({transport_p.x + h * 3, transport_p.y});
            if (IconButton("play", icons.Play.get(), ImDrawFlags_RoundCornersNone)) action = action::timeline::TogglePlay{playback.CurrentFrame};
        }
        SetCursorScreenPos({transport_p.x + h * 4, transport_p.y});
        if (IconButton("next_keyframe", icons.NextKeyframe.get(), ImDrawFlags_RoundCornersNone)) action = action::timeline::JumpKeyframe{.Next = true};
        SetCursorScreenPos({transport_p.x + h * 5, transport_p.y});
        if (IconButton("jump_end", icons.JumpEnd.get(), ImDrawFlags_RoundCornersRight)) action = action::timeline::JumpToEnd{};

        const auto delta_p = ImVec2{transport_p.x + h * TransportSlots + spacing.x, transport_p.y};
        SetCursorScreenPos(delta_p);
        if (IconButton("jump_back", icons.FramePrev.get(), ImDrawFlags_RoundCornersLeft)) action = action::timeline::JumpTime{.Backward = true};
        SetCursorScreenPos({delta_p.x + h, delta_p.y});
        if (IconButton("jump_forward", icons.FrameNext.get(), ImDrawFlags_RoundCornersNone)) action = action::timeline::JumpTime{.Backward = false};
        SetCursorScreenPos({delta_p.x + h * 2, delta_p.y});
        if (IconButton("playback_options", nullptr, ImDrawFlags_RoundCornersRight)) OpenPopup("##PlaybackOptions");
        {
            const auto center = (GetItemRectMin() + GetItemRectMax()) * 0.5f;
            constexpr float arrow_half = 3.5f;
            GetWindowDrawList()->AddTriangleFilled(
                center - ImVec2{arrow_half, arrow_half * 0.5f},
                center + ImVec2{arrow_half, -arrow_half * 0.5f},
                center + ImVec2{0.f, arrow_half * 0.5f},
                GetColorU32(ImGuiCol_Text)
            );
        }
        if (BeginPopup("##PlaybackOptions")) {
            auto nav = navigation;
            bool changed = Checkbox("Wrap Timeline Navigation", &nav.Wrap);
            Separator();
            TextUnformatted("Jump Unit");
            if (RadioButton("Frame", !nav.JumpInSeconds)) {
                nav.JumpInSeconds = false;
                changed = true;
            }
            SameLine();
            if (RadioButton("Second", nav.JumpInSeconds)) {
                nav.JumpInSeconds = true;
                changed = true;
            }
            SetNextItemWidth(GetFontSize() * 5);
            InputFloat("Delta", &nav.JumpDelta, 0.f, 0.f, "%.2f");
            if (IsItemDeactivatedAfterEdit()) {
                nav.JumpDelta = std::max(nav.JumpDelta, 0.1f);
                changed = true;
            }
            if (changed) action = action::timeline::SetNavigation{nav};
            EndPopup();
        }
    }

    SetCursorScreenPos(bar_origin);
    Dummy({h, h});
    static constexpr float input_width{50};
    SameLine(w - CalcTextSize("Frame").x - CalcTextSize("Start").x - CalcTextSize("End").x - spacing.x * 6 - input_width * 3);
    PushItemWidth(input_width);
    {
        int frame = playback.CurrentFrame;
        InputInt("Frame", &frame, 0, 0);
        if (IsItemDeactivatedAfterEdit()) action = action::timeline::SetFrame{frame};
    }
    SameLine(0, spacing.x * 2);
    {
        int start = range.StartFrame;
        InputInt("Start", &start, 0, 0);
        if (IsItemDeactivatedAfterEdit()) action = action::timeline::SetStartFrame{start};
    }
    SameLine(0, spacing.x);
    {
        int end = range.EndFrame;
        InputInt("End", &end, 0, 0);
        if (IsItemDeactivatedAfterEdit()) action = action::timeline::SetEndFrame{end};
    }
    PopItemWidth();
    SetCursorScreenPos({bar_origin.x, bar_origin.y + h});
    Dummy({0, 0});

    const auto area = GetContentRegionAvail();
    if (area.x <= 0 || area.y <= 0) return action;

    const auto p0 = GetCursorScreenPos(), p1 = p0 + area;
    InvisibleButton("##timeline", area);

    auto *dl = GetWindowDrawList();
    dl->AddRectFilled(p0, p1, IM_COL32(30, 30, 30, 255));

    const auto frame_to_x = [&](float frame) -> float { return p0.x + area.x * 0.5f + (frame - view.ViewCenterFrame) * view.PixelsPerFrame; };
    const auto x_to_frame = [&](float x) -> float { return (x - p0.x - area.x * 0.5f) / view.PixelsPerFrame + view.ViewCenterFrame; };

    if (const float sx = std::max(frame_to_x(range.StartFrame), p0.x), ex = std::min(frame_to_x(range.EndFrame), p1.x); ex > sx) {
        dl->AddRectFilled({sx, p0.y}, {ex, p1.y}, IM_COL32(50, 50, 55, 255));
    }

    const auto major_step = ComputeMajorStep(view.PixelsPerFrame);
    const float major_px = major_step * view.PixelsPerFrame;
    const bool show_minor = major_step >= 2 && major_px >= 32.0f;

    const float half_width_frames = (area.x * 0.5f) / view.PixelsPerFrame;
    const int vis_start = int(std::floor(view.ViewCenterFrame - half_width_frames));
    const int vis_end = int(std::ceil(view.ViewCenterFrame + half_width_frames));
    const int first_major = (vis_start / major_step - 1) * major_step;
    for (int f = first_major; f <= vis_end + major_step; f += major_step) {
        if (const float fx = frame_to_x(f); fx >= p0.x && fx <= p1.x) {
            dl->AddLine({fx, p0.y + HeaderHeight}, {fx, p1.y}, IM_COL32(80, 80, 80, 255));
            const auto label = std::format("{}", f);
            const auto text_size = CalcTextSize(label.c_str());
            dl->AddText({fx - text_size.x * 0.5f, p0.y + (HeaderHeight - text_size.y) * 0.5f}, IM_COL32(180, 180, 180, 255), label.c_str());
        }
        if (show_minor) {
            if (const float mfx = frame_to_x(f + major_step * 0.5f); mfx >= p0.x && mfx <= p1.x) {
                dl->AddLine({mfx, p0.y + HeaderHeight}, {mfx, p1.y}, IM_COL32(55, 55, 55, 255));
            }
        }
    }

    dl->AddLine({p0.x, p0.y + HeaderHeight}, {p1.x, p0.y + HeaderHeight}, IM_COL32(60, 60, 60, 255));

    // Summary row of keyframe diamonds under the ruler.
    if (!keyframes.empty()) {
        constexpr float KeyHalf{5.f};
        const float ky = p0.y + HeaderHeight + KeyHalf + 4.f;
        const auto first = std::ranges::lower_bound(keyframes, x_to_frame(p0.x - KeyHalf));
        for (auto it = first; it != keyframes.end(); ++it) {
            const float kx = frame_to_x(*it);
            if (kx > p1.x + KeyHalf) break;
            const ImVec2 top{kx, ky - KeyHalf}, right{kx + KeyHalf, ky}, bottom{kx, ky + KeyHalf}, left{kx - KeyHalf, ky};
            dl->AddQuadFilled(top, right, bottom, left, IM_COL32(230, 230, 230, 255));
            dl->AddQuad(top, right, bottom, left, IM_COL32(20, 20, 20, 255));
        }
    }

    if (const float cfx = frame_to_x(float(playback.CurrentFrame)); cfx >= p0.x && cfx <= p1.x) {
        dl->AddLine({cfx, p0.y}, {cfx, p1.y}, IM_COL32(100, 160, 255, 200), 2.0f);
        const auto label = std::format("{}", playback.CurrentFrame);
        const auto text_size = CalcTextSize(label.c_str());
        const float lx = cfx - text_size.x * 0.5f;
        const float ly = p0.y + (HeaderHeight - text_size.y) * 0.5f;
        dl->AddRectFilled({lx - 3, ly - 1}, {lx + text_size.x + 3, ly + text_size.y + 1}, IM_COL32(80, 130, 200, 200), 3.0f);
        dl->AddText({lx, ly}, IM_COL32(255, 255, 255, 255), label.c_str());
    }

    if (const bool timeline_hovered = IsItemHovered(), timeline_active = IsItemActive(); timeline_hovered || timeline_active) {
        const auto &io = GetIO();
        const bool in_header = io.MousePos.y >= p0.y && io.MousePos.y < p0.y + HeaderHeight;
        // Preserve scrubbing until pointer release.
        scrubbing = timeline_active && io.MouseClickedPos[0].y < p0.y + HeaderHeight;
        if ((in_header && IsMouseClicked(ImGuiMouseButton_Left)) || (scrubbing && IsMouseDragging(ImGuiMouseButton_Left))) {
            if (const int frame = int(std::round(x_to_frame(io.MousePos.x))); frame != playback.CurrentFrame) {
                action = action::timeline::SetFrame{frame};
            }
        }
        // Vertical scrolling zooms and horizontal scrolling pans.
        if (io.MouseWheel != 0.0f) {
            const float mouse_frame = x_to_frame(io.MousePos.x);
            const float mouse_frac = (io.MousePos.x - p0.x) / area.x - 0.5f;
            const float new_ppf = std::clamp(view.PixelsPerFrame * std::pow(1.1f, io.MouseWheel), MinPixelsPerFrame, MaxPixelsPerFrame);
            action = action::timeline::SetView{new_ppf, mouse_frame - mouse_frac * area.x / new_ppf};
        }
        if (io.MouseWheelH != 0.0f) {
            action = action::timeline::SetView{view.PixelsPerFrame, view.ViewCenterFrame - io.MouseWheelH * 20.0f / view.PixelsPerFrame};
        }
    }

    return action;
}
