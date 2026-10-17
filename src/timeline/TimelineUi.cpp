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

bool IconButton(const char *id, const SvgResource *icon, float rounding, ImDrawFlags corners = ImDrawFlags_RoundCornersAll, float width_scale = 1.f) {
    const float h = GetFrameHeight();
    PushID(id);
    const bool clicked = InvisibleButton("##btn", {h * width_scale, h});
    ui::DrawIconButton(GetItemRectMin(), GetItemRectMax(), GetColorU32(IsItemHovered() ? ImGuiCol_ButtonHovered : ImGuiCol_Button), rounding, corners, icon, h * 0.8f);
    PopID();
    return clicked;
}

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

std::optional<action::Action> RenderTimeline(const TimelineRange &range, const TimelinePlayback &playback, const TimelineView &view, const TimelineNavigation &navigation, const Animations &animations, std::span<const float> keyframes, const TimelineIcons &icons, const InterfaceTheme &interface_theme, const TimelineTheme &theme, bool &scrubbing) {
    const float rounding = interface_theme.IconButtonRounding;
    const auto &colors = theme.Colors;
    const float ruler_h = theme.RulerHeight;
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
        const auto record_p = bar_origin + ImVec2{(w - h * (RecordSlots + TransportSlots + DeltaSlots) - spacing.x * 2) * 0.5f, 0.f};
        SetCursorScreenPos(record_p);
        if (IconButton("record", nullptr, rounding, ImDrawFlags_RoundCornersAll)) emit(action::UpdateOf<&Animations::Record>(action::OnViewport{}, !animations.Record));
        SetItemTooltip("Auto keying: insert keyframes for animated properties on edit");
        GetWindowDrawList()->AddCircleFilled((GetItemRectMin() + GetItemRectMax()) * 0.5f, h * 0.22f, animations.Record ? IM_COL32(230, 60, 50, 255) : IM_COL32(120, 120, 120, 255));

        const auto transport_p = ImVec2{record_p.x + h * RecordSlots + spacing.x, record_p.y};
        SetCursorScreenPos(transport_p);
        if (IconButton("jump_start", icons.JumpStart.get(), rounding, ImDrawFlags_RoundCornersLeft)) emit(action::timeline::JumpToStart{});
        SetCursorScreenPos({transport_p.x + h, transport_p.y});
        if (IconButton("prev_keyframe", icons.PrevKeyframe.get(), rounding, ImDrawFlags_RoundCornersNone)) emit(action::timeline::JumpKeyframe{.Next = false});
        SetCursorScreenPos({transport_p.x + h * 2, transport_p.y});
        if (playback.Playing) {
            if (IconButton("pause", icons.Pause.get(), rounding, ImDrawFlags_RoundCornersNone, 2.f)) emit(action::timeline::TogglePlay{playback.CurrentFrame});
        } else {
            if (IconButton("play_reverse", icons.PlayReverse.get(), rounding, ImDrawFlags_RoundCornersNone)) emit(action::timeline::TogglePlay{playback.CurrentFrame, /*Reverse=*/true});
            SetCursorScreenPos({transport_p.x + h * 3, transport_p.y});
            if (IconButton("play", icons.Play.get(), rounding, ImDrawFlags_RoundCornersNone)) emit(action::timeline::TogglePlay{playback.CurrentFrame});
        }
        SetCursorScreenPos({transport_p.x + h * 4, transport_p.y});
        if (IconButton("next_keyframe", icons.NextKeyframe.get(), rounding, ImDrawFlags_RoundCornersNone)) emit(action::timeline::JumpKeyframe{.Next = true});
        SetCursorScreenPos({transport_p.x + h * 5, transport_p.y});
        if (IconButton("jump_end", icons.JumpEnd.get(), rounding, ImDrawFlags_RoundCornersRight)) emit(action::timeline::JumpToEnd{});

        const auto delta_p = ImVec2{transport_p.x + h * TransportSlots + spacing.x, transport_p.y};
        SetCursorScreenPos(delta_p);
        if (IconButton("jump_back", icons.FramePrev.get(), rounding, ImDrawFlags_RoundCornersLeft)) emit(action::timeline::JumpTime{.Backward = true});
        SetCursorScreenPos({delta_p.x + h, delta_p.y});
        if (IconButton("jump_forward", icons.FrameNext.get(), rounding, ImDrawFlags_RoundCornersNone)) emit(action::timeline::JumpTime{.Backward = false});
        SetCursorScreenPos({delta_p.x + h * 2, delta_p.y});
        if (IconButton("playback_options", nullptr, rounding, ImDrawFlags_RoundCornersRight)) OpenPopup("##PlaybackOptions");
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
            if (changed) emit(action::timeline::SetNavigation{nav});
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
        if (IconButton("add_animation", nullptr, rounding, ImDrawFlags_RoundCornersAll)) emit(action::animation::AddAnimation{std::format("Animation {}", animations.Names.size() + 1)});
        SetItemTooltip("Create a new animation");
        const auto plus_center = (GetItemRectMin() + GetItemRectMax()) * 0.5f;
        constexpr float arm = 4.f;
        auto *draw = GetWindowDrawList();
        draw->AddLine({plus_center.x - arm, plus_center.y}, {plus_center.x + arm, plus_center.y}, GetColorU32(ImGuiCol_Text), 1.5f);
        draw->AddLine({plus_center.x, plus_center.y - arm}, {plus_center.x, plus_center.y + arm}, GetColorU32(ImGuiCol_Text), 1.5f);
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
        dl->AddRectFilled({sx, p0.y}, {ex, p1.y}, ui::U32(colors.Range));
    }

    const auto major_step = ComputeMajorStep(view.PixelsPerFrame, theme.TickLabelSpacing);
    const float major_px = major_step * view.PixelsPerFrame;
    const bool show_minor = major_step >= 2 && major_px >= theme.MinorTickSpacing;

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

    dl->AddLine({p0.x, p0.y + ruler_h}, {p1.x, p0.y + ruler_h}, ui::U32(colors.RulerLine));

    if (!keyframes.empty()) {
        const float kr = theme.KeyframeRadius;
        const float ky = p0.y + ruler_h + kr + theme.KeyframePadding;
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
        dl->AddLine({cfx, p0.y}, {cfx, p1.y}, ui::U32(colors.Playhead), theme.PlayheadWidth);
        const auto label = std::format("{}", playback.CurrentFrame);
        const auto text_size = CalcTextSize(label.c_str());
        const float lx = cfx - text_size.x * 0.5f;
        const float ly = p0.y + (ruler_h - text_size.y) * 0.5f;
        const ImVec2 pad{theme.PlayheadLabelPaddingX, theme.PlayheadLabelPaddingY};
        dl->AddRectFilled(ImVec2{lx, ly} - pad, ImVec2{lx + text_size.x, ly + text_size.y} + pad, ui::U32(colors.PlayheadLabelBackground), theme.PlayheadLabelRounding);
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

void DrawTimelineStyle(state::Scene &r, state::Entity viewport) {
    using C = TimelineThemeColors;
    ui::Edit f{r, viewport};
    if (Button("Reset")) action::Emit(action::view::ResetTimelineTheme{});
    SeparatorText("Colors");
    auto c = f.Sub<&TimelineTheme::Colors>();
    c.Color<&C::Background>();
    c.Color<&C::Range>("Frame range");
    c.Color<&C::RulerLine>();
    c.Color<&C::MajorTick>();
    c.Color<&C::MinorTick>();
    c.Color<&C::TickLabel>();
    c.Color<&C::Keyframe>();
    c.Color<&C::KeyframeOutline>();
    c.Color<&C::Playhead>();
    c.Color<&C::PlayheadLabel>();
    c.Color<&C::PlayheadLabelBackground>();
    SeparatorText("Sizes");
    f.Slider<&TimelineTheme::RulerHeight>(nullptr, 12.f, 48.f, "%.0f");
    f.Slider<&TimelineTheme::TickLabelSpacing>(nullptr, 20.f, 200.f, "%.0f");
    f.Slider<&TimelineTheme::MinorTickSpacing>(nullptr, 8.f, 128.f, "%.0f");
    f.Slider<&TimelineTheme::KeyframeRadius>(nullptr, 2.f, 12.f, "%.1f");
    f.Slider<&TimelineTheme::KeyframePadding>(nullptr, 0.f, 24.f, "%.0f");
    f.Slider<&TimelineTheme::PlayheadWidth>(nullptr, .5f, 8.f, "%.1f");
    f.Slider<&TimelineTheme::PlayheadLabelRounding>(nullptr, 0.f, 12.f, "%.1f");
    f.Slider<&TimelineTheme::PlayheadLabelPaddingX>("Playhead label padding X", 0.f, 12.f, "%.0f");
    f.Slider<&TimelineTheme::PlayheadLabelPaddingY>("Playhead label padding Y", 0.f, 12.f, "%.0f");
}
