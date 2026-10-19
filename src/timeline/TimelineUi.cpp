#include "timeline/TimelineUi.h"
#include "action/Build.h"
#include "action/View.h"
#include "timeline/Timeline.h"
#include "viewport/ViewportIcons.h"

#include "timeline/TimelineTheme.h"
#include "ui/CtrlShortcut.h"
#include "ui/FieldEdit.h"
#include "ui/InterfaceTheme.h"
#include "ui/InterfaceUi.h"

#include <format>

using namespace ImGui;

namespace {
constexpr float MinPixelsPerFrame{1}, MaxPixelsPerFrame{400};

int ComputeMajorStep(float pixels_per_frame, float min_label_spacing) {
    static constexpr int steps[]{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
    for (const int s : steps) {
        if (s * pixels_per_frame >= min_label_spacing) return s;
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
    if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_LeftArrow, flags)) return action::timeline::JumpTime{.Backward = true};
    if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_RightArrow, flags)) return action::timeline::JumpTime{.Backward = false};
    if (Shortcut(ImGuiKey_UpArrow, flags)) return action::timeline::JumpKeyframe{.Next = false};
    if (Shortcut(ImGuiKey_DownArrow, flags)) return action::timeline::JumpKeyframe{.Next = true};
    return {};
}

std::optional<action::Action> RenderTimeline(const TimelineRange &range, const TimelinePlayback &playback, const TimelineView &view, state::Registry &r, const Animations &animations, std::span<const float> keyframes, const TimelineIcons &icons, const InterfaceTheme &interface_theme, const TimelineTheme &theme, bool &scrubbing) {
    const float rounding = interface_theme.IconButtonRounding;
    const auto &colors = theme.Colors;
    const float ruler_h = theme.Sizes.RulerHeight;
    std::optional<action::Action> result;
    const auto emit = [&](auto leaf) { result = action::MakeAction(std::move(leaf)); };
    scrubbing = false;

    const float w = GetContentRegionAvail().x;
    const float h = GetFrameHeight();
    const auto spacing = GetStyle().ItemSpacing;

    const auto bar_origin = GetCursorScreenPos();
    {
        // Pause spans the reverse and forward play slots while playing.
        constexpr float RecordSlots{1}, TransportSlots{6}, DeltaSlots{3};
        const float bar_w = h * (RecordSlots + TransportSlots + DeltaSlots) + ui::IconButtonSeam * (TransportSlots - 1 + DeltaSlots - 1) + spacing.x * 2;
        const auto record_p = bar_origin + ImVec2{(w - bar_w) * 0.5f, 0.f};
        SetCursorScreenPos(record_p);
        ui::BeginIconButtonGroup(ImGuiAxis_X, {h, h}, 1, rounding);
        if (ui::IconButton("##record", nullptr)) emit(action::UpdateOf<&Animations::Record>(state::Document, !animations.Record));
        ui::Describe("Auto keying", "Insert keyframes for animated properties on edit");
        GetWindowDrawList()->AddCircleFilled((GetItemRectMin() + GetItemRectMax()) * 0.5f, h * 0.22f, animations.Record ? IM_COL32(230, 60, 50, 255) : IM_COL32(120, 120, 120, 255));
        ui::EndIconButtonGroup();

        const auto transport_p = ImVec2{record_p.x + h * RecordSlots + spacing.x, record_p.y};
        SetCursorScreenPos(transport_p);
        ui::BeginIconButtonGroup(ImGuiAxis_X, {h, h}, playback.Playing ? 5 : 6, rounding);
        if (ui::IconButton("##jump_start", icons.JumpStart.get())) emit(action::timeline::JumpToStart{});
        if (ui::IconButton("##prev_keyframe", icons.PrevKeyframe.get())) emit(action::timeline::JumpKeyframe{.Next = false});
        if (playback.Playing) {
            if (ui::IconButton("##pause", icons.Pause.get(), false, true, h * 2 + ui::IconButtonSeam)) emit(action::timeline::TogglePlay{playback.CurrentFrame});
        } else {
            if (ui::IconButton("##play_reverse", icons.PlayReverse.get())) emit(action::timeline::TogglePlay{playback.CurrentFrame, /*Reverse=*/true});
            if (ui::IconButton("##play", icons.Play.get())) emit(action::timeline::TogglePlay{playback.CurrentFrame});
        }
        if (ui::IconButton("##next_keyframe", icons.NextKeyframe.get())) emit(action::timeline::JumpKeyframe{.Next = true});
        if (ui::IconButton("##jump_end", icons.JumpEnd.get())) emit(action::timeline::JumpToEnd{});
        ui::EndIconButtonGroup();

        SetCursorScreenPos({GetItemRectMax().x + spacing.x, transport_p.y});
        ui::BeginIconButtonGroup(ImGuiAxis_X, {h, h}, 3, rounding);
        if (ui::IconButton("##jump_back", icons.FramePrev.get())) emit(action::timeline::JumpTime{.Backward = true});
        if (ui::IconButton("##jump_forward", icons.FrameNext.get())) emit(action::timeline::JumpTime{.Backward = false});
        if (ui::IconButton("##playback_options", nullptr)) OpenPopup("##PlaybackOptions");
        ui::DropdownArrow();
        ui::EndIconButtonGroup();
        if (BeginPopup("##PlaybackOptions")) {
            ui::DrawFields<TimelineNavigation>(ui::Edit{r, state::Document});
            if (animations.Active < animations.Names.size()) {
                Separator();
                TextUnformatted("Animation name");
                static char name[128];
                if (IsWindowAppearing()) std::snprintf(name, sizeof name, "%s", animations.Names[animations.Active].c_str());
                SetNextItemWidth(GetFontSize() * 10);
                InputText("##animation_name", name, sizeof name);
                if (IsItemDeactivatedAfterEdit()) emit(action::animation::RenameAnimation{animations.Active, name});
            }
            EndPopup();
        }
    }

    if (!animations.Names.empty()) {
        SetCursorScreenPos(bar_origin);
        SetNextItemWidth(GetFontSize() * 9);
        if (BeginCombo("##animation", animations.Names[animations.Active].c_str())) {
            for (uint32_t i = 0; i < animations.Names.size(); ++i) {
                PushID(int(i));
                if (Selectable(animations.Names[i].empty() ? "(unnamed)" : animations.Names[i].c_str(), i == animations.Active)) emit(action::animation::SelectAnimation{i});
                PopID();
            }
            EndCombo();
        }
        SameLine(0, spacing.x);
        ui::BeginIconButtonGroup(ImGuiAxis_X, {h, h}, 1, rounding);
        if (ui::IconButton("##add_animation", nullptr)) emit(action::animation::AddAnimation{std::format("Animation {}", animations.Names.size() + 1)});
        ui::Describe("Add animation", "Create a new animation");
        const auto plus_center = (GetItemRectMin() + GetItemRectMax()) * 0.5f;
        constexpr float arm = 4.f;
        auto *draw = GetWindowDrawList();
        draw->AddLine({plus_center.x - arm, plus_center.y}, {plus_center.x + arm, plus_center.y}, GetColorU32(ImGuiCol_Text), 1.5f);
        draw->AddLine({plus_center.x, plus_center.y - arm}, {plus_center.x, plus_center.y + arm}, GetColorU32(ImGuiCol_Text), 1.5f);
        ui::EndIconButtonGroup();
    }
    SetCursorScreenPos(bar_origin);
    Dummy({h, h});
    const float frame_width = GetFontSize() * 4, range_width = GetFontSize() * 11;
    SameLine(w - frame_width - range_width - spacing.x * 3);
    SetNextItemWidth(frame_width);
    int frame = playback.CurrentFrame;
    ui::Gesture(DragInt("##frame", &frame), [&] { return action::timeline::SetFrame{frame}; });
    SameLine(0, spacing.x);
    SetNextItemWidth(range_width);
    int start = range.StartFrame, end = range.EndFrame;
    ui::Gesture(DragIntRange2("##range", &start, &end, 1.f, 0, 0, "Start %d", "End %d"), [&] {
        return start != range.StartFrame ? action::MakeAction(action::timeline::SetStartFrame{start}) : action::MakeAction(action::timeline::SetEndFrame{end});
    });
    SetCursorScreenPos({bar_origin.x, bar_origin.y + h});
    Dummy({0, 0});

    const auto area = GetContentRegionAvail();
    if (area.x <= 0 || area.y <= 0) return result;

    const auto p0 = GetCursorScreenPos(), p1 = p0 + area;
    InvisibleButton("##timeline", area);

    auto *dl = GetWindowDrawList();
    dl->AddRectFilled(p0, p1, ui::U32(colors.Background));

    const auto frame_to_x = [&](float frame) -> float { return p0.x + area.x * 0.5f + (frame - view.ViewCenterFrame) * view.PixelsPerFrame; };
    const auto x_to_frame = [&](float x) -> float { return (x - p0.x - area.x * 0.5f) / view.PixelsPerFrame + view.ViewCenterFrame; };

    if (const float sx = std::max(frame_to_x(range.StartFrame), p0.x), ex = std::min(frame_to_x(range.EndFrame), p1.x); ex > sx) {
        dl->AddRectFilled({sx, p0.y}, {ex, p1.y}, ui::U32(colors.FrameRange));
    }

    const auto major_step = ComputeMajorStep(view.PixelsPerFrame, theme.Sizes.TickLabelSpacing);
    const float major_px = major_step * view.PixelsPerFrame;
    const bool show_minor = major_step >= 2 && major_px >= theme.Sizes.MinorTickSpacing;

    const float half_width_frames = (area.x * 0.5f) / view.PixelsPerFrame;
    const auto vis_start = int(std::floor(view.ViewCenterFrame - half_width_frames));
    const auto vis_end = int(std::ceil(view.ViewCenterFrame + half_width_frames));
    const int first_major = (vis_start / major_step - 1) * major_step;
    for (int f = first_major; f <= vis_end + major_step; f += major_step) {
        if (const float fx = frame_to_x(f); fx >= p0.x && fx <= p1.x) {
            dl->AddLine({fx, p0.y + ruler_h}, {fx, p1.y}, ui::U32(colors.MajorTick));
            const auto label = std::format("{}", f);
            const auto text_size = CalcTextSize(label.c_str());
            dl->AddText({fx - text_size.x * 0.5f, p0.y + (ruler_h - text_size.y) * 0.5f}, ui::U32(colors.TickLabel), label.c_str());
        }
        if (show_minor) {
            if (const float mfx = frame_to_x(f + major_step * 0.5f); mfx >= p0.x && mfx <= p1.x) {
                dl->AddLine({mfx, p0.y + ruler_h}, {mfx, p1.y}, ui::U32(colors.MinorTick));
            }
        }
    }

    dl->AddLineH(p0.x, p1.x, p0.y + ruler_h, ui::U32(colors.RulerLine));

    if (!keyframes.empty()) {
        const float kr = theme.Sizes.KeyframeRadius;
        const float ky = p0.y + ruler_h + kr + theme.Sizes.KeyframePadding;
        const auto first = std::ranges::lower_bound(keyframes, x_to_frame(p0.x - kr));
        for (auto it = first; it != keyframes.end(); ++it) {
            const float kx = frame_to_x(*it);
            if (kx > p1.x + kr) break;
            const ImVec2 top{kx, ky - kr}, right{kx + kr, ky}, bottom{kx, ky + kr}, left{kx - kr, ky};
            dl->AddQuadFilled(top, right, bottom, left, ui::U32(colors.Keyframe));
            dl->AddQuad(top, right, bottom, left, ui::U32(colors.KeyframeOutline));
        }
    }

    if (const float cfx = frame_to_x(float(playback.CurrentFrame)); cfx >= p0.x && cfx <= p1.x) {
        dl->AddLine({cfx, p0.y}, {cfx, p1.y}, ui::U32(colors.Playhead), theme.Sizes.PlayheadWidth);
        const auto label = std::format("{}", playback.CurrentFrame);
        const auto text_size = CalcTextSize(label.c_str());
        const float lx = cfx - text_size.x * 0.5f;
        const float ly = p0.y + (ruler_h - text_size.y) * 0.5f;
        const auto pad = std::bit_cast<ImVec2>(theme.Sizes.PlayheadLabelPadding);
        dl->AddRectFilled(ImVec2{lx, ly} - pad, ImVec2{lx + text_size.x, ly + text_size.y} + pad, ui::U32(colors.PlayheadLabelBackground), theme.Sizes.PlayheadLabelRounding);
        dl->AddText({lx, ly}, ui::U32(colors.PlayheadLabel), label.c_str());
    }

    if (const bool timeline_hovered = IsItemHovered(), timeline_active = IsItemActive(); timeline_hovered || timeline_active) {
        const auto &io = GetIO();
        const bool in_header = io.MousePos.y >= p0.y && io.MousePos.y < p0.y + ruler_h;
        // Preserve scrubbing until pointer release.
        scrubbing = timeline_active && io.MouseClickedPos[0].y < p0.y + ruler_h;
        if ((in_header && IsMouseClicked(ImGuiMouseButton_Left)) || (scrubbing && IsMouseDragging(ImGuiMouseButton_Left))) {
            if (const auto frame = int(std::round(x_to_frame(io.MousePos.x))); frame != playback.CurrentFrame) {
                emit(action::timeline::SetFrame{frame});
            }
        }
        // Vertical scrolling zooms and horizontal scrolling pans.
        if (io.MouseWheel != 0.0f) {
            const float mouse_frame = x_to_frame(io.MousePos.x);
            const float mouse_frac = (io.MousePos.x - p0.x) / area.x - 0.5f;
            const float new_ppf = std::clamp(view.PixelsPerFrame * std::pow(1.1f, io.MouseWheel), MinPixelsPerFrame, MaxPixelsPerFrame);
            emit(action::timeline::SetView{new_ppf, mouse_frame - mouse_frac * area.x / new_ppf});
        }
        if (io.MouseWheelH != 0.0f) {
            emit(action::timeline::SetView{view.PixelsPerFrame, view.ViewCenterFrame - io.MouseWheelH * 20.0f / view.PixelsPerFrame});
        }
    }

    return result;
}

void DrawTimelineStyle(state::Registry &r) {
    if (Button("Reset")) action::Emit(action::view::ResetTimelineTheme{});
    ui::DrawFields<TimelineTheme>(ui::Edit{r, state::Document});
}
