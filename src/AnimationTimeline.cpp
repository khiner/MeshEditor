#include "AnimationTimeline.h"
#include "SvgResource.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <format>

using namespace ImGui;

namespace {
constexpr float HeaderHeight{20};
constexpr float MinPixelsPerFrame{1};
constexpr float MaxPixelsPerFrame{400};

bool IconButton(const char *id, const SvgResource *icon, ImDrawFlags corners = ImDrawFlags_RoundCornersAll) {
    const float h = GetFrameHeight();
    const ImVec2 size{h, h};
    const float icon_dim = h * 0.7f;
    static constexpr ImVec2 padding{0.5f, 0.5f};

    PushID(id);
    const bool clicked = InvisibleButton("##btn", size);
    const bool hovered = IsItemHovered();
    auto *dl = GetWindowDrawList();
    dl->AddRectFilled(GetItemRectMin() + padding, GetItemRectMax() - padding, hovered ? GetColorU32(ImGuiCol_ButtonHovered) : GetColorU32(ImGuiCol_Button), 6.0f, corners);
    if (icon) {
        const auto saved = GetCursorScreenPos();
        SetCursorScreenPos({GetItemRectMin().x + (h - icon_dim) * 0.5f, GetItemRectMin().y + (h - icon_dim) * 0.5f});
        icon->DrawIcon({icon_dim, icon_dim});
        SetCursorScreenPos(saved);
    }
    PopID();
    return clicked;
}

// Find the major step size so labels don't overlap
int ComputeMajorStep(float pixels_per_frame) {
    static constexpr float MinLabelSpacingPx{80};
    static constexpr int steps[]{1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
    for (int s : steps) {
        if (s * pixels_per_frame >= MinLabelSpacingPx) return s;
    }
    return steps[std::size(steps) - 1];
}
} // namespace

std::optional<AnimationTimelineAction> RenderAnimationTimeline(const AnimationTimeline &tl, AnimationTimelineView &view, const AnimationIcons &icons) {
    std::optional<AnimationTimelineAction> action;

    // --- Transport bar ---
    const float w = GetContentRegionAvail().x;
    const float h = GetFrameHeight();
    const auto spacing = GetStyle().ItemSpacing;

    // Center transport buttons vertically in their bar
    const auto bar_origin = GetCursorScreenPos();
    {
        const auto transport_p = bar_origin + ImVec2{(w - h * 3) * 0.5f, spacing.y};
        SetCursorScreenPos(transport_p);
        if (IconButton("jump_start", icons.JumpStart.get(), ImDrawFlags_RoundCornersLeft)) action = timeline_action::JumpToStart{};
        SetCursorScreenPos({transport_p.x + h, transport_p.y});
        if (IconButton("play_pause", tl.Playing ? icons.Pause.get() : icons.Play.get(), ImDrawFlags_RoundCornersNone)) action = timeline_action::TogglePlay{};
        SetCursorScreenPos({transport_p.x + h * 2, transport_p.y});
        if (IconButton("jump_end", icons.JumpEnd.get(), ImDrawFlags_RoundCornersRight)) action = timeline_action::JumpToEnd{};
    }

    // Right-aligned frame inputs
    SetCursorScreenPos({bar_origin.x, bar_origin.y + spacing.y});
    Dummy({h, h});
    static constexpr float input_width{50};
    SameLine(w - CalcTextSize("Frame").x - CalcTextSize("Start").x - CalcTextSize("End").x - spacing.x * 6 - input_width * 3);
    PushItemWidth(input_width);
    {
        int frame = tl.CurrentFrame;
        InputInt("Frame", &frame, 0, 0);
        if (IsItemDeactivatedAfterEdit()) action = timeline_action::SetFrame{frame};
    }
    SameLine(0, spacing.x * 2);
    {
        int start = tl.StartFrame;
        InputInt("Start", &start, 0, 0);
        if (IsItemDeactivatedAfterEdit()) action = timeline_action::SetStartFrame{start};
    }
    SameLine(0, spacing.x);
    {
        int end = tl.EndFrame;
        InputInt("End", &end, 0, 0);
        if (IsItemDeactivatedAfterEdit()) action = timeline_action::SetEndFrame{end};
    }
    PopItemWidth();
    SetCursorScreenPos({bar_origin.x, bar_origin.y + h + spacing.y * 2 - spacing.y});
    Dummy({0, 0});

    // --- Timeline area ---
    const auto area = GetContentRegionAvail();
    if (area.x <= 0 || area.y <= 0) return action;

    const auto p0 = GetCursorScreenPos(), p1 = p0 + area;
    InvisibleButton("##timeline", area);

    // Background
    auto *dl = GetWindowDrawList();
    dl->AddRectFilled(p0, p1, IM_COL32(30, 30, 30, 255));

    const auto frame_to_x = [&](float frame) -> float { return p0.x + area.x * 0.5f + (frame - view.ViewCenterFrame) * view.PixelsPerFrame; };
    const auto x_to_frame = [&](float x) -> float { return (x - p0.x - area.x * 0.5f) / view.PixelsPerFrame + view.ViewCenterFrame; };

    // [StartFrame, EndFrame] range highlight
    if (const float sx = std::max(frame_to_x(tl.StartFrame), p0.x), ex = std::min(frame_to_x(tl.EndFrame), p1.x); ex > sx) {
        dl->AddRectFilled({sx, p0.y}, {ex, p1.y}, IM_COL32(50, 50, 55, 255));
    }

    // Adaptive frame lines
    const auto major_step = ComputeMajorStep(view.PixelsPerFrame);
    const float major_px = major_step * view.PixelsPerFrame;
    const bool show_minor = major_step >= 2 && major_px >= 32.0f; // Half-step lines when spacing allows

    // Compute visible frame range
    const float half_width_frames = (area.x * 0.5f) / view.PixelsPerFrame;
    const int vis_start = int(std::floor(view.ViewCenterFrame - half_width_frames));
    const int vis_end = int(std::ceil(view.ViewCenterFrame + half_width_frames));
    const int first_major = (vis_start / major_step - 1) * major_step;
    for (int f = first_major; f <= vis_end + major_step; f += major_step) {
        if (const float fx = frame_to_x(f); fx >= p0.x && fx <= p1.x) { // Major line
            dl->AddLine({fx, p0.y + HeaderHeight}, {fx, p1.y}, IM_COL32(80, 80, 80, 255));
            // Frame number label in header
            const auto label = std::format("{}", f);
            const auto text_size = CalcTextSize(label.c_str());
            dl->AddText({fx - text_size.x * 0.5f, p0.y + (HeaderHeight - text_size.y) * 0.5f}, IM_COL32(180, 180, 180, 255), label.c_str());
        }
        if (show_minor) { // Minor (half-step) line
            if (const float mfx = frame_to_x(f + major_step * 0.5f); mfx >= p0.x && mfx <= p1.x) {
                dl->AddLine({mfx, p0.y + HeaderHeight}, {mfx, p1.y}, IM_COL32(55, 55, 55, 255));
            }
        }
    }

    // Header separator
    dl->AddLine({p0.x, p0.y + HeaderHeight}, {p1.x, p0.y + HeaderHeight}, IM_COL32(60, 60, 60, 255));

    // Current frame line (light blue)
    if (const float cfx = frame_to_x(float(tl.CurrentFrame)); cfx >= p0.x && cfx <= p1.x) {
        dl->AddLine({cfx, p0.y}, {cfx, p1.y}, IM_COL32(100, 160, 255, 200), 2.0f);
        // Current frame label with background
        const auto label = std::format("{}", tl.CurrentFrame);
        const auto text_size = CalcTextSize(label.c_str());
        const float lx = cfx - text_size.x * 0.5f;
        const float ly = p0.y + (HeaderHeight - text_size.y) * 0.5f;
        dl->AddRectFilled({lx - 3, ly - 1}, {lx + text_size.x + 3, ly + text_size.y + 1}, IM_COL32(80, 130, 200, 200), 3.0f);
        dl->AddText({lx, ly}, IM_COL32(255, 255, 255, 255), label.c_str());
    }

    // Mouse interaction
    if (const bool timeline_hovered = IsItemHovered(), timeline_active = IsItemActive(); timeline_hovered || timeline_active) {
        const auto &io = GetIO();
        // Click/drag in header to scrub frame
        const bool in_header = io.MousePos.y >= p0.y && io.MousePos.y < p0.y + HeaderHeight;
        if ((in_header && IsMouseClicked(ImGuiMouseButton_Left)) || (timeline_active && IsMouseDragging(ImGuiMouseButton_Left) && io.MouseClickedPos[0].y < p0.y + HeaderHeight)) {
            action = timeline_action::SetFrame{int(std::round(x_to_frame(io.MousePos.x)))};
        }
        // Vertical scroll (or pinch) → zoom, horizontal scroll → pan
        if (io.MouseWheel != 0.0f) {
            const float mouse_frame = x_to_frame(io.MousePos.x);
            const float mouse_frac = (io.MousePos.x - p0.x) / area.x - 0.5f;
            view.PixelsPerFrame = std::clamp(view.PixelsPerFrame * std::pow(1.1f, io.MouseWheel), MinPixelsPerFrame, MaxPixelsPerFrame);
            view.ViewCenterFrame = mouse_frame - mouse_frac * area.x / view.PixelsPerFrame;
        }
        if (io.MouseWheelH != 0.0f) {
            view.ViewCenterFrame -= io.MouseWheelH * 20.0f / view.PixelsPerFrame;
        }
    }

    return action;
}
