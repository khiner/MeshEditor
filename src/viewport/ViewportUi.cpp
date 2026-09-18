#include "numeric/uvec2.h"
#include "numeric/vec2.h"

#include "viewport/ViewportUi.h"

#include "Camera.h"
#include "Profile.h"
#include "Variant.h"
#include "action/Animation.h"
#include "action/Audio.h"
#include "action/Bone.h"
#include "action/Mesh.h"
#include "action/Object.h"
#include "action/Selection.h"
#include "action/Timeline.h"
#include "action/View.h"
#include "animation/AnimationTimeline.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "gizmo/TransformGizmo.h"
#include "gltf/SourceAssets.h"
#include "numeric/Angles.h"
#include "numeric/MatrixMath.h"
#include "project/Project.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/TextureRefs.h"
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "ui/CtrlShortcut.h"
#include "ui/FieldEdit.h"
#include "viewport/FrameState.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/ScreenSpace.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportIcons.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"
#include <imgui_internal.h>

#include "state/Scene.h"

#include <algorithm>
#include <ranges>

#include "gizmo/OrientationGizmo.h"

using numeric::Min;

using std::ranges::any_of, std::ranges::fold_left;
using std::views::transform;

using namespace ImGui;

namespace {
constexpr vec2 ToVec2(ImVec2 v) { return std::bit_cast<vec2>(v); }
constexpr float WheelOrbitRadPerUnit{0.05f}, WheelZoomStep{1.04f};

std::optional<std::pair<uvec2, uvec2>> ComputeBoxSelectPixels(vec2 start, vec2 end, vec2 window_pos, uvec2 logical_extent, uvec2 render_extent) {
    static constexpr float DragThresholdSq{2 * 2};
    if (Distance2(start, end) <= DragThresholdSq) return {};

    const vec2 logical_size{float(logical_extent.x), float(logical_extent.y)};
    const vec2 render_scale{
        logical_extent.x > 0u ? float(render_extent.x) / float(logical_extent.x) : 1.f,
        logical_extent.y > 0u ? float(render_extent.y) / float(logical_extent.y) : 1.f
    };
    // Intersect the drag with the viewport. A drag that ends up wholly outside it selects nothing.
    const auto local_min = Max(Min(start, end) - window_pos, vec2{0});
    const auto local_max = Min(Max(start, end) - window_pos, logical_size);
    if (local_min.x > local_max.x || local_min.y > local_max.y) return {};

    // The box names pixels, so its maximum is the last one, not one past it.
    const auto last_px = Max(render_extent, uvec2{1}) - uvec2{1};
    const auto render_min = local_min * render_scale;
    const auto render_max = local_max * render_scale;
    const auto box_min_px = Min(uvec2{std::floor(render_min.x), std::floor(render_min.y)}, last_px);
    const auto box_max_px = Min(uvec2{std::ceil(render_max.x), std::ceil(render_max.y)}, last_px);
    return std::pair{box_min_px, box_max_px};
}

void WrapMousePos(const ImRect &wrap_rect, vec2 &accumulated_wrap_mouse_delta) {
    const auto &g = *GImGui;
    // After wrapping, require 2 non-boundary frames before re-wrapping (guards against failed OS cursor warps).
    static uint32_t wrap_guard[2]{};
    ImVec2 mouse_delta{0, 0};
    for (uint32_t axis = 0; axis < 2; ++axis) {
        if (g.IO.MousePos[axis] >= wrap_rect.Max[axis] || g.IO.MousePos[axis] <= wrap_rect.Min[axis]) {
            if (wrap_guard[axis]) continue;
            if (g.IO.MousePos[axis] >= wrap_rect.Max[axis]) mouse_delta[axis] = -wrap_rect.GetSize()[axis] + 1;
            else mouse_delta[axis] = wrap_rect.GetSize()[axis] - 1;
            wrap_guard[axis] = 2;
        } else if (wrap_guard[axis]) {
            wrap_guard[axis]--;
        }
    }
    if (mouse_delta != ImVec2{0, 0}) {
        accumulated_wrap_mouse_delta -= ToVec2(mouse_delta);
        TeleportMousePos(g.IO.MousePos + mouse_delta);
    }
}

bool IsSingleClicked(ImGuiMouseButton button) {
    static bool EscapePressed = false;
    if (IsMouseClicked(button)) EscapePressed = false;
    if (IsKeyPressed(ImGuiKey_Escape, false)) EscapePressed = true;
    if (IsMouseReleased(button)) {
        const bool was_escape_pressed = EscapePressed;
        EscapePressed = false;
        if (was_escape_pressed) return false;
    }
    return IsMouseReleased(button) && !IsMouseDragPastThreshold(button);
}

// Navigation actions are not recorded. During look-through, the first navigation input emits the recorded exit action.
void EmitViewNav(const state::Scene &r, auto &&nav) {
    if (LookThroughCameraEntity(r) != state::Null) action::Emit(action::view::ExitLookThroughCamera{});
    else action::Emit(std::forward<decltype(nav)>(nav));
}

struct OverlayIconButtonInfo {
    const SvgResource *Icon;
    ImVec2 Offset;
    ImDrawFlags Corners;
    bool Enabled{true};
    bool Active{false};
    const char *Tooltip{nullptr};
    // Dims a button whose setting has no effect in the current mode while keeping it clickable.
    bool Muted{false};
};

struct OverlayIconButtonStyle {
    ImVec2 ButtonSize{36, 30};
    ImVec2 Padding{0.5f, 0.5f};
    float IconScale{0.75f};
    float CornerRounding{8.f};
};

template<size_t N>
std::optional<size_t> DrawOverlayIconButtonGroup(
    const char *id,
    ImVec2 start_pos,
    const OverlayIconButtonInfo (&buttons)[N],
    bool interactions_enabled,
    bool *any_hovered = nullptr,
    OverlayIconButtonStyle style = {}
) {
    const auto saved_cursor_pos = GetCursorScreenPos();
    const float icon_dim = style.ButtonSize.y * style.IconScale;
    const ImVec2 icon_size{icon_dim, icon_dim};
    auto &dl = *GetWindowDrawList();
    std::optional<size_t> clicked_index;

    PushID(id);
    for (size_t i = 0; i < N; ++i) {
        const auto &button = buttons[i];
        const ImVec2 button_min = start_pos + button.Offset;
        const ImVec2 button_max = button_min + style.ButtonSize;

        bool hovered = false;
        if (interactions_enabled) {
            SetCursorScreenPos(button_min);
            if (!button.Enabled) BeginDisabled();
            PushID(int(i));
            if (InvisibleButton("##icon", style.ButtonSize) && button.Enabled) clicked_index = i;
            hovered = IsItemHovered();
            PopID();
            if (!button.Enabled) EndDisabled();
        }
        if (any_hovered && hovered) *any_hovered = true;

        if (button.Muted) PushStyleVar(ImGuiStyleVar_Alpha, GetStyle().Alpha * 0.5f);
        const auto bg_color = GetColorU32(
            !button.Enabled   ? ImGuiCol_FrameBg :
                button.Active ? ImGuiCol_ButtonActive :
                hovered       ? ImGuiCol_ButtonHovered :
                                ImGuiCol_Button
        );
        dl.AddRectFilled(button_min + style.Padding, button_max - style.Padding, bg_color, style.CornerRounding, button.Corners);
        if (button.Icon) {
            SetCursorScreenPos(button_min + (style.ButtonSize - icon_size) * 0.5f);
            button.Icon->DrawIcon(std::bit_cast<vec2>(icon_size));
        }
        if (button.Muted) PopStyleVar();
    }
    PopID();
    SetCursorScreenPos(saved_cursor_pos);
    return clicked_index;
}

// Opens `popup_id` with ImGui's PressedOnClick boundary.
void DrawOverlayDropdownArrow(ImVec2 pos, ImVec2 size, const OverlayIconButtonStyle &style, const char *id, const char *popup_id, bool &any_hovered) {
    const auto saved_cursor = GetCursorScreenPos();
    SetCursorScreenPos(pos);
    PushID(id);
    InvisibleButton("##btn", size);
    const bool arrow_hovered = IsItemHovered();
    PopID();
    SetCursorScreenPos(saved_cursor);

    if (arrow_hovered) any_hovered = true;
    const auto arrow_min = pos + style.Padding;
    const auto arrow_max = pos + size - style.Padding;
    const bool popup_open = IsPopupOpen(popup_id);
    const auto bg_color = GetColorU32(popup_open ? ImGuiCol_ButtonActive : arrow_hovered ? ImGuiCol_ButtonHovered :
                                                                                           ImGuiCol_Button);
    auto &dl = *GetWindowDrawList();
    dl.AddRectFilled(arrow_min, arrow_max, bg_color, style.CornerRounding, ImDrawFlags_RoundCornersRight);

    const auto center = (arrow_min + arrow_max) * 0.5f;
    constexpr float arrow_half = 3.5f;
    dl.AddTriangleFilled(
        center - ImVec2{arrow_half, arrow_half * 0.5f},
        center + ImVec2{arrow_half, -arrow_half * 0.5f},
        center + ImVec2{0.f, arrow_half * 0.5f},
        GetColorU32(ImGuiCol_Text)
    );
    if (IsMouseClicked(0) && arrow_hovered && !popup_open) OpenPopup(popup_id);
}

// The world-space center of the selected vertices across the edit meshes' primary instances.
vec3 EditSelectionCenter(const state::Scene &r, Element edit_mode) {
    vec3 center{};
    uint32_t vertex_count = 0;
    for (const auto &[mesh_entity, instance_entity] : selection::ComputePrimaryEditInstances(r, false)) {
        const auto *stats = GetElementSelectionSummary(r, mesh_entity, edit_mode);
        if (!stats || stats->SelectedVertexCount == 0) continue;
        const auto &world = r.get<const WorldTransform>(instance_entity);
        center += float(stats->SelectedVertexCount) * world.P + Rotate(world.R, world.S * stats->PositionSum);
        vertex_count += stats->SelectedVertexCount;
    }
    return vertex_count > 0 ? center / float(vertex_count) : center;
}

// A world point's screen position in logical pixels.
vec2 ScreenPx(const mat4 &vp, const rect &viewport_rect, vec3 p) {
    const auto cs = vp * vec4{p, 1.f};
    return viewport_rect.pos + NdcToUv(vec2{cs.x, cs.y} / cs.w) * viewport_rect.size;
}

// A screen position in pixels of the render target.
vec2 ToRenderPx(const state::Scene &r, vec2 screen_px) {
    const auto logical = r.ctx().get<const ViewportExtent>().Value;
    const auto render_extent = RenderExtentPx(r);
    const vec2 scale{logical.x > 0u ? float(render_extent.x) / float(logical.x) : 1.f, logical.y > 0u ? float(render_extent.y) / float(logical.y) : 1.f};
    return (screen_px - ToVec2(GetCursorScreenPos())) * scale;
}

void BeginMeshDrag(state::Scene &r, state::Entity viewport, FrameState &frame, MeshOperatorDrag::Op op) {
    const rect viewport_rect{ToVec2(GetWindowPos()), ToVec2(GetContentRegionAvail())};
    const auto &camera = r.get<const ViewCamera>(viewport);
    const auto vp = camera.Projection(viewport_rect.size.x / viewport_rect.size.y) * camera.View();
    const auto center = EditSelectionCenter(r, r.get<const EditMode>(viewport).Value);
    const auto center_px = ScreenPx(vp, viewport_rect, center);
    // World units per logical pixel at the center, measured along camera right.
    const float pixels = Length(ScreenPx(vp, viewport_rect, center + camera.Basis()[0]) - center_px);
    frame.MeshDrag = MeshOperatorDrag{.Value = op, .StartPx = ToVec2(GetMousePos()), .CenterPx = center_px, .WorldPerPx = pixels > 0.f ? 1.f / pixels : 0.f};
}

// Sizes the active mesh drag from the mouse, commits on release and cancels on Escape.
void UpdateMeshDrag(state::Scene &r, FrameState &frame) {
    using Op = MeshOperatorDrag::Op;
    auto &drag = *frame.MeshDrag;
    const auto mouse = ToVec2(GetMousePos());
    const bool knife = drag.Value == Op::Knife;
    drag.WheelAccum += knife ? 0.f : std::exchange(frame.PreciseWheelDelta, vec2{0}).y;
    const int steps = int(drag.WheelAccum) + IsKeyPressed(ImGuiKey_Equal, true) - IsKeyPressed(ImGuiKey_Minus, true);
    drag.WheelAccum -= float(int(drag.WheelAccum));
    const auto segments = uint32_t(std::clamp(int(drag.Segments) + steps, 1, 16));
    bool changed = std::exchange(drag.Segments, segments) != segments;
    if (drag.Value == Op::Inset && IsKeyPressed(ImGuiKey_I, false)) {
        drag.Individual = !drag.Individual;
        changed = true;
    }
    if (IsKeyPressed(ImGuiKey_Escape, false) || IsMouseClicked(ImGuiMouseButton_Right)) {
        if (drag.Staged) action::Cancel();
        frame.MeshDrag.reset();
        return;
    }
    // Every viewport modal ends on the button release, so the pick never sees an edge a modal consumed.
    if (IsMouseReleased(ImGuiMouseButton_Left)) {
        if (drag.Staged) action::Commit();
        frame.MeshDrag.reset();
        return;
    }
    if (knife) {
        // The knife restages on every mouse move, keyed by the segment's length.
        const auto end = ToRenderPx(r, mouse), start = ToRenderPx(r, drag.StartPx);
        const float length = Length(end - start);
        if (length <= 0.f || drag.Staged == length) return;
        action::Emit(action::mesh::Knife{.Start = start, .End = end, .View = std::make_unique<RenderView>(r.ctx().get<const GpuBuffers>().FrameView)}, action::Phase::Stage);
        drag.Staged = length;
        return;
    }
    // An inset closes toward the center and reopens past it, and a bevel widens away from the center.
    const float travel = Length(mouse - drag.CenterPx) - Length(drag.StartPx - drag.CenterPx);
    const float value = std::max((drag.Value == Op::Inset ? -travel : travel) * drag.WorldPerPx, 0.f);
    if ((value <= 0.f && !drag.Staged) || (drag.Staged == value && !changed)) return;
    if (drag.Value == Op::Inset) action::Emit(action::mesh::Inset{.Thickness = value, .Depth = 0.f, .Individual = drag.Individual, .Even = true}, action::Phase::Stage);
    else action::Emit(action::mesh::Bevel{.Width = value, .Segments = drag.Segments, .Vertices = drag.Value == Op::BevelVertices}, action::Phase::Stage);
    drag.Staged = value;
}

constexpr const char *DeleteNames[]{"Vertices", "Edges", "Faces", "Only Edges & Faces", "Only Faces", "Loose"};
constexpr const char *MergeNames[]{"At Center", "At First", "At Last", "Collapse", "By Distance"};

// The edit-mode operator popups, opened by keys and the right mouse button.
void DrawMeshOperatorMenus(state::Scene &r, state::Entity viewport) {
    using namespace action::mesh;
    const auto op = [](const char *label, auto a, action::Phase phase = action::Phase::Record) {
        if (MenuItem(label)) action::Emit(std::move(a), phase);
    };
    const auto submenu = [](const char *label, auto items) {
        if (!BeginMenu(label)) return;
        items();
        EndMenu();
    };
    const auto merge_items = [&] {
        for (size_t i = 0; i < std::size(MergeNames); ++i) op(MergeNames[i], Merge{Merge::Mode(i)});
    };
    const auto delete_items = [&] {
        for (size_t i = 0; i < 5; ++i) op(DeleteNames[i], action::mesh::Delete{MeshTopologyOp(i)});
        Separator();
        op("Dissolve Vertices", Dissolve{Dissolve::Mode::Vertices});
        op("Dissolve Edges", Dissolve{Dissolve::Mode::Edges});
        op("Dissolve Faces", Dissolve{Dissolve::Mode::Faces});
        op("Limited Dissolve", Dissolve{Dissolve::Mode::Limited});
        Separator();
        op("Edge Collapse", Merge{Merge::Mode::Collapse});
        op("Edge Loops", Dissolve{Dissolve::Mode::Edges});
    };
    const auto vertex_items = [&] {
        op("Bevel Vertices", Bevel{.Vertices = true});
        op("New Edge/Face from Vertices", Fill{});
        op("Connect Vertex Path", ConnectVertices{});
        op("Rip Vertices", Rip{}, action::Phase::Stage);
        Separator();
        submenu("Merge Vertices", merge_items);
        op("Separate", Separate{});
        op("Dissolve Vertices", Dissolve{Dissolve::Mode::Vertices});
    };
    const auto edge_items = [&] {
        op("Extrude Edges", Extrude{Extrude::Mode::Edges}, action::Phase::Stage);
        op("Bevel Edges", Bevel{});
        op("Bridge Edge Loops", BridgeEdgeLoops{});
        op("Loop Cut", LoopCut{});
        op("Subdivide", Subdivide{});
        Separator();
        op("Rotate Edge", EdgeRotate{});
        op("Edge Split", EdgeSplit{});
        op("Rip", Rip{}, action::Phase::Stage);
        op("Dissolve Edges", Dissolve{Dissolve::Mode::Edges});
    };
    const auto face_items = [&] {
        op("Extrude Faces", Extrude{Extrude::Mode::Region}, action::Phase::Stage);
        op("Extrude Individual Faces", Extrude{Extrude::Mode::FacesIndividual}, action::Phase::Stage);
        op("Inset Faces", Inset{});
        op("Poke Faces", Poke{});
        op("Triangulate Faces", Triangulate{});
        op("Tris to Quads", TrisToQuads{});
        op("Solidify Faces", Solidify{});
        Separator();
        op("Fill", Fill{});
        op("Grid Fill", GridFill{});
        op("Fill Holes", FillHoles{});
        Separator();
        op("Flip Normals", FlipNormals{});
        op("Dissolve Faces", Dissolve{Dissolve::Mode::Faces});
    };
    const auto cleanup_items = [&] {
        op("Delete Loose", action::mesh::Delete{MeshTopologyOp::DeleteLoose});
        op("Degenerate Dissolve", Dissolve{Dissolve::Mode::Degenerate});
        op("Limited Dissolve", Dissolve{Dissolve::Mode::Limited});
        op("Merge by Distance", Merge{Merge::Mode::ByDistance});
        op("Fill Holes", FillHoles{});
    };
    const auto mesh_items = [&] {
        // The bisect plane and spin axis follow the view through the selection center.
        const auto &camera = r.get<const ViewCamera>(viewport);
        const auto center = EditSelectionCenter(r, r.get<const EditMode>(viewport).Value);
        op("Duplicate", action::mesh::Duplicate{}, action::Phase::Stage);
        op("Extrude Region", Extrude{}, action::Phase::Stage);
        op("Split", Split{});
        op("Separate", Separate{});
        Separator();
        op("Bisect", Bisect{.Point = center, .Normal = camera.Basis()[0]});
        op("Symmetrize", Symmetrize{});
        op("Spin", Spin{.Axis = camera.Forward(), .Center = center});
        op("Extrude Repeat", ExtrudeRepeat{});
        op("Convex Hull", ConvexHull{});
        Separator();
        submenu("Vertex", vertex_items);
        submenu("Edge", edge_items);
        submenu("Face", face_items);
        submenu("Merge", merge_items);
        submenu("Clean Up", cleanup_items);
        submenu("Delete", delete_items);
    };
    PushStyleVar(ImGuiStyleVar_WindowPadding, {8, 8});
    const auto popup = [&](const char *id, const char *title, auto items) {
        if (!BeginPopup(id)) return;
        TextDisabled("%s", title);
        Separator();
        items();
        EndPopup();
    };
    popup("##MeshDelete", "Delete", delete_items);
    popup("##MeshMerge", "Merge", merge_items);
    popup("##MeshVertex", "Vertex", vertex_items);
    popup("##MeshEdge", "Edge", edge_items);
    popup("##MeshFace", "Face", face_items);
    popup("##MeshContext", "Mesh", mesh_items);
    PopStyleVar();
}

std::string SpacedName(std::string_view name) {
    std::string out;
    for (const char c : name) {
        if (!out.empty() && c >= 'A' && c <= 'Z') out += ' ';
        out += c;
    }
    return out;
}

// The panel that reruns the last mesh operator with edited parameters on the state it was applied to.
void DrawLastOperationPanel(state::Scene &r, state::Entity viewport, const rect &viewport_rect) {
    using namespace action::mesh;
    const auto *last = r.try_get<const LastOperation>(viewport);
    auto &session = project::Session(r);
    const auto &history = session.History;
    if (!last || (history.Present != last->Node && session.Editing != last->Node)) return;
    if (std::visit([]<typename L>(const L &) { return std::is_empty_v<L>; }, last->Value)) return;

    constexpr float Pad{12.f};
    SetNextWindowPos(std::bit_cast<ImVec2>(viewport_rect.pos) + ImVec2{Pad, viewport_rect.size.y - Pad}, ImGuiCond_Always, {0.f, 1.f});
    SetNextWindowBgAlpha(0.85f);
    PushStyleVar(ImGuiStyleVar_WindowPadding, {10.f, 6.f});
    PushStyleVar(ImGuiStyleVar_WindowRounding, 6.f);
    constexpr ImGuiWindowFlags PanelFlags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove;
    if (Begin("##LastMeshOperation", nullptr, PanelFlags)) {
        BringWindowToDisplayFront(GetCurrentWindow());
        const auto &node = history.Nodes[last->Node];
        if (TreeNodeEx(SpacedName(node.Label).c_str(), ImGuiTreeNodeFlags_CollapsingHeader)) {
            // The widgets edit the record in place, every change restages the operator on the node's parent, and a release commits the gesture in the node's place.
            auto &op = r.edit<LastOperation>(viewport);
            bool changed = false, finished = false;
            const auto released = [&] { finished |= IsItemDeactivatedAfterEdit(); };
            const auto drag_float = [&](const char *label, float &v, float speed, float min, float max, const char *format = "%.3f") {
                changed |= DragFloat(label, &v, speed, min, max, format, ImGuiSliderFlags_AlwaysClamp);
                released();
            };
            const auto drag_int = [&](const char *label, uint32_t &v, int min, int max) {
                int value = int(v);
                if (DragInt(label, &value, 0.1f, min, max, "%d", ImGuiSliderFlags_AlwaysClamp)) {
                    v = uint32_t(value);
                    changed = true;
                }
                released();
            };
            const auto drag_vec3 = [&](const char *label, vec3 &v) {
                changed |= DragFloat3(label, &v.x, 0.01f);
                released();
            };
            const auto degrees = [&](const char *label, float &radians, float min, float max) {
                float value = numeric::Degrees(radians);
                if (DragFloat(label, &value, 0.5f, min, max, "%.1f deg", ImGuiSliderFlags_AlwaysClamp)) {
                    radians = numeric::Radians(value);
                    changed = true;
                }
                released();
            };
            const auto check = [&](const char *label, bool &v) {
                if (Checkbox(label, &v)) changed = finished = true;
            };
            const auto combo = [&](const char *label, auto &mode, std::span<const char *const> names) {
                int value = int(mode);
                if (Combo(label, &value, names.data(), int(names.size()))) {
                    mode = std::remove_cvref_t<decltype(mode)>(value);
                    changed = finished = true;
                }
            };
            PushItemWidth(160.f);
            std::visit(
                overloaded{
                    [&](action::mesh::Delete &a) { combo("Type", a.Op, DeleteNames); },
                    [&](Merge &a) {
                        combo("Mode", a.Value, MergeNames);
                        if (a.Value == Merge::Mode::ByDistance) drag_float("Distance", a.Distance, 0.0001f, 0.f, 10.f, "%.4f");
                    },
                    [&](Extrude &a) {
                        static constexpr const char *Names[]{"Region", "Edges", "Individual Faces"};
                        combo("Mode", a.Value, Names);
                    },
                    [&](Dissolve &a) {
                        static constexpr const char *Names[]{"Vertices", "Edges", "Faces", "Limited", "Degenerate"};
                        combo("Mode", a.Value, Names);
                        if (a.Value == Dissolve::Mode::Limited) degrees("Max Angle", a.Angle, 0.f, 180.f);
                        if (a.Value == Dissolve::Mode::Degenerate) drag_float("Distance", a.Distance, 0.0001f, 0.f, 10.f, "%.4f");
                    },
                    [&](Subdivide &a) { drag_int("Cuts", a.Cuts, 1, 32); },
                    [&](Poke &a) { drag_float("Offset", a.Offset, 0.01f, -100.f, 100.f); },
                    [&](Inset &a) {
                        drag_float("Thickness", a.Thickness, 0.01f, 0.f, 100.f);
                        drag_float("Depth", a.Depth, 0.01f, -100.f, 100.f);
                        check("Individual", a.Individual);
                        check("Even", a.Even);
                    },
                    [&](LoopCut &a) { drag_int("Cuts", a.Cuts, 1, 32); },
                    [&](Spin &a) {
                        drag_int("Steps", a.Steps, 1, 256);
                        degrees("Angle", a.Angle, -360.f, 360.f);
                        drag_vec3("Axis", a.Axis);
                        drag_vec3("Center", a.Center);
                        drag_float("Offset", a.Offset, 0.01f, -100.f, 100.f);
                    },
                    [&](ExtrudeRepeat &a) {
                        drag_int("Steps", a.Steps, 1, 256);
                        drag_vec3("Offset", a.Offset);
                    },
                    [&](Bisect &a) {
                        drag_vec3("Point", a.Point);
                        drag_vec3("Normal", a.Normal);
                        check("Clear Inner", a.ClearInner);
                        check("Clear Outer", a.ClearOuter);
                    },
                    [&](Symmetrize &a) {
                        static constexpr const char *Names[]{"X", "Y", "Z"};
                        combo("Axis", a.Axis, Names);
                        check("Negative", a.Negative);
                    },
                    [&](Solidify &a) { drag_float("Thickness", a.Thickness, 0.01f, -100.f, 100.f); },
                    [&](GridFill &a) { drag_int("Span", a.Span, 0, 256); },
                    [&](FillHoles &a) { drag_int("Sides", a.Sides, 0, 1000); },
                    [&](Bevel &a) {
                        drag_float("Width", a.Width, 0.01f, 0.f, 100.f);
                        drag_int("Segments", a.Segments, 1, 16);
                        check("Vertices", a.Vertices);
                    },
                    [](auto &) {},
                },
                op.Value
            );
            PopItemWidth();
            if (changed) {
                if (session.Editing != last->Node) session.EditNode(last->Node);
                std::visit([&]<typename L>(const L &leaf) {
                    if constexpr (std::copyable<L>) action::Emit(L{leaf}, action::Phase::Stage);
                },
                           op.Value);
            }
            if (finished) action::Commit();
        }
    }
    End();
    PopStyleVar(2);
}
} // namespace

void Interact(state::Scene &r, state::Entity viewport, FrameState &frame) {
    // Any open popup (e.g. Viewport shading dropdown) blocks viewport mouse/keyboard input.
    // Without this, wheel/click events still patch the camera while the popup overlays the viewport_rect.
    if (IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
        frame.PreciseWheelDelta = {0, 0};
        return;
    }

    static ImVec2 PrevClickPos{-FLT_MAX, -FLT_MAX}, CurrentClickPos{-FLT_MAX, -FLT_MAX};
    if (GetIO().MouseClicked[0]) {
        PrevClickPos = CurrentClickPos;
        CurrentClickPos = GetIO().MouseClickedPos[0];
    }

    const auto logical_extent = r.ctx().get<ViewportExtent>().Value;
    if (logical_extent.x == 0 || logical_extent.y == 0) return;

    if (frame.MeshDrag) {
        if (frame.MeshDrag->Staged && !project::Session(r).HasStaged()) frame.MeshDrag.reset();
        else return UpdateMeshDrag(r, frame);
    }

    const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
    const auto active_entity = FindActiveEntity(r);
    const bool has_frozen_selected = r.view<Selected, ScaleLocked>().begin() != r.view<Selected, ScaleLocked>().end();
    const bool edit_transform_locked = interaction_mode == InteractionMode::Edit &&
        any_of(selection::GetSelectedMeshEntities(r), [&](state::Entity mesh_entity) { return selection::HasScaleLockedInstance(r, mesh_entity); });
    const bool transform_shortcuts_enabled = !edit_transform_locked;
    const bool scale_shortcut_enabled = transform_shortcuts_enabled && !has_frozen_selected;
    // Route shortcuts globally while preserving ImGui ownership for active widgets, navigation, and text input.
    constexpr auto VKey = ImGuiInputFlags_RouteGlobal;
    if (r.get<const GizmoInteraction>(viewport).IsUsing()) {
        // During an active transform, only allow transform switching shortcuts.
        if (Shortcut(ImGuiKey_G, VKey) && transform_shortcuts_enabled) action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Translate}, action::Phase::Cancel);
        else if (Shortcut(ImGuiKey_R, VKey) && transform_shortcuts_enabled) action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Rotate}, action::Phase::Cancel);
        else if (Shortcut(ImGuiKey_S, VKey) && scale_shortcut_enabled) action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Scale}, action::Phase::Cancel);
    } else {
        if (interaction_mode != InteractionMode::Edit) {
            if (Shortcut(ImGuiKey_I, VKey)) action::Emit(action::animation::InsertKey{{.Scope = action::Scope::Selected}});
            else if (Shortcut(ImGuiMod_Alt | ImGuiKey_I, VKey)) action::Emit(action::animation::DeleteKey{{.Scope = action::Scope::Selected}});
        }
        if (Shortcut(ImGuiKey_Space, VKey)) action::Emit(action::timeline::TogglePlay{r.get<const TimelinePlayback>(viewport).CurrentFrame});
        else if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Space, VKey)) action::Emit(action::timeline::TogglePlay{r.get<const TimelinePlayback>(viewport).CurrentFrame, /*Reverse=*/true});
        else if (Shortcut(ImGuiKey_Z, VKey)) {
            const auto current = r.get<const ViewportDisplay>(viewport).ViewportShading;
            const auto next = current == ViewportShadingMode::Solid ? ViewportShadingMode::MaterialPreview :
                current == ViewportShadingMode::MaterialPreview     ? ViewportShadingMode::Rendered :
                                                                      ViewportShadingMode::Solid;
            action::Emit(action::view::SetViewportShading{.Mode = next});
        } else if (Shortcut(ImGuiMod_Shift | ImGuiKey_Z, VKey)) {
            const auto &settings = r.get<const ViewportDisplay>(viewport);
            action::Emit(action::view::SetViewportShading{.Mode = settings.ViewportShading == ViewportShadingMode::Wireframe ? settings.FillMode : ViewportShadingMode::Wireframe});
        } else if (Shortcut(ImGuiMod_Alt | ImGuiKey_Z, VKey)) {
            action::Emit(action::view::ToggleXRay{});
        }
        // Tab uses default RouteFocused (not VKey/RouteGlobal) so widget tabbing in panels keeps working.
        const bool tab_no_mods = Shortcut(ImGuiKey_Tab);
        const bool tab_ctrl = CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_Tab);
        if (tab_no_mods || tab_ctrl) {
            const bool is_armature = FindArmatureObject(r, active_entity) != state::Null;
            if (is_armature && tab_ctrl) {
                action::Emit(action::view::SetInteractionMode{.Mode = interaction_mode == InteractionMode::Pose ? InteractionMode::Object : InteractionMode::Pose});
            } else if (is_armature) {
                action::Emit(action::view::SetInteractionMode{.Mode = interaction_mode == InteractionMode::Edit ? InteractionMode::Object : InteractionMode::Edit});
            } else if (tab_no_mods) {
                action::Emit(action::view::CycleInteractionMode{});
            }
        }
        if (interaction_mode == InteractionMode::Edit) {
            if (Shortcut(ImGuiKey_1, VKey)) action::Emit(action::view::SetEditMode{.Mode = Element::Vertex});
            else if (Shortcut(ImGuiKey_2, VKey)) action::Emit(action::view::SetEditMode{.Mode = Element::Edge});
            else if (Shortcut(ImGuiKey_3, VKey)) action::Emit(action::view::SetEditMode{.Mode = Element::Face});
        }
        if (Shortcut(ImGuiKey_A, VKey)) action::Emit(action::selection::SelectAll{});
        if (Shortcut(ImGuiMod_Alt | ImGuiKey_A, VKey)) action::Emit(action::selection::DeselectAll{});
        const bool bone_edit = interaction_mode == InteractionMode::Edit && FindArmatureObject(r, active_entity) != state::Null;
        const bool mesh_edit = interaction_mode == InteractionMode::Edit && !bone_edit;
        if (mesh_edit) {
            namespace mesh = action::mesh;
            using Op = MeshOperatorDrag::Op;
            using DissolveMode = mesh::Dissolve::Mode;
            const auto element = r.get<const EditMode>(viewport).Value;
            // Placement drags join the gesture, so an extrude, duplicate, or rip and its move commit as one node.
            if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_X, VKey)) action::Emit(mesh::Dissolve{element == Element::Face ? DissolveMode::Faces : element == Element::Edge ? DissolveMode::Edges :
                                                                                                                                                                        DissolveMode::Vertices});
            else if (Shortcut(ImGuiKey_X, VKey) || Shortcut(ImGuiKey_Delete, VKey) || Shortcut(ImGuiKey_Backspace, VKey)) OpenPopup("##MeshDelete");
            else if (Shortcut(ImGuiKey_E, VKey)) action::Emit(mesh::Extrude{element == Element::Edge ? mesh::Extrude::Mode::Edges : mesh::Extrude::Mode::Region}, action::Phase::Stage);
            else if (Shortcut(ImGuiMod_Shift | ImGuiKey_D, VKey)) action::Emit(mesh::Duplicate{}, action::Phase::Stage);
            else if (Shortcut(ImGuiKey_Y, VKey)) action::Emit(mesh::Split{});
            else if (Shortcut(ImGuiKey_P, VKey)) action::Emit(mesh::Separate{});
            else if (Shortcut(ImGuiKey_M, VKey)) OpenPopup("##MeshMerge");
            else if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_T, VKey)) action::Emit(mesh::Triangulate{});
            else if (Shortcut(ImGuiMod_Alt | ImGuiKey_J, VKey)) action::Emit(mesh::TrisToQuads{});
            else if (Shortcut(ImGuiKey_F, VKey)) action::Emit(mesh::Fill{});
            else if (Shortcut(ImGuiKey_V, VKey)) action::Emit(mesh::Rip{}, action::Phase::Stage);
            else if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_V, VKey)) OpenPopup("##MeshVertex");
            else if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_E, VKey)) OpenPopup("##MeshEdge");
            else if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_F, VKey)) OpenPopup("##MeshFace");
            else if (!IsWindowHovered()) {
            } else if (Shortcut(ImGuiKey_I, VKey)) BeginMeshDrag(r, viewport, frame, Op::Inset);
            else if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_B, VKey)) BeginMeshDrag(r, viewport, frame, Op::BevelVertices);
            else if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_B, VKey)) BeginMeshDrag(r, viewport, frame, Op::BevelEdges);
            else if (Shortcut(ImGuiKey_K, VKey)) BeginMeshDrag(r, viewport, frame, Op::Knife);
        }
        if (bone_edit) {
            if (Shortcut(ImGuiMod_Shift | ImGuiKey_A, VKey)) {
                action::Emit(action::bone::Add{});
            } else if (Shortcut(ImGuiKey_E, VKey)) {
                action::Emit(action::bone::Extrude{}, action::Phase::Stage);
            } else if (Shortcut(ImGuiKey_X, VKey) || Shortcut(ImGuiKey_Delete, VKey) || Shortcut(ImGuiKey_Backspace, VKey)) {
                Delete(r, viewport);
            } else if (Shortcut(ImGuiMod_Shift | ImGuiKey_D, VKey)) {
                Duplicate(r, viewport);
            }
        }
        if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_E, VKey)) {
            action::Emit(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
        } else if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_A, VKey)) {
            action::Emit(action::object::AddArmature{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
        } else if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_C, VKey)) {
            action::Emit(action::object::AddCamera{.Info = std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
        } else if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_L, VKey)) {
            action::Emit(action::object::AddLight{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
        }
        if (!r.view<const Selected>().empty()) {
            if (!bone_edit && !mesh_edit && Shortcut(ImGuiMod_Shift | ImGuiKey_D, VKey)) Duplicate(r, viewport);
            else if (!bone_edit && !mesh_edit && Shortcut(ImGuiMod_Alt | ImGuiKey_D, VKey)) action::Emit(action::object::DuplicateLinked{}, action::Phase::Stage);
            else if (!bone_edit && !mesh_edit && CanDelete(r, viewport) && (Shortcut(ImGuiKey_Delete, VKey) || Shortcut(ImGuiKey_Backspace, VKey))) Delete(r, viewport);
            else if (interaction_mode == InteractionMode::Pose && Shortcut(ImGuiMod_Alt | ImGuiKey_G, VKey)) action::Emit(action::bone::ClearSelectedTransforms{.Position = true});
            else if (interaction_mode == InteractionMode::Pose && Shortcut(ImGuiMod_Alt | ImGuiKey_R, VKey)) action::Emit(action::bone::ClearSelectedTransforms{.Rotation = true});
            else if (interaction_mode == InteractionMode::Pose && Shortcut(ImGuiMod_Alt | ImGuiKey_S, VKey)) action::Emit(action::bone::ClearSelectedTransforms{.Scale = true});
            else if (Shortcut(ImGuiKey_G, VKey) && transform_shortcuts_enabled) {
                // Start transform gizmo in both Object and Edit modes.
                // In Edit mode, shader applies transform to selected vertices.
                // In Object mode, shader applies transform to selected instances.
                action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Translate}, action::Phase::Cancel);
            } else if (Shortcut(ImGuiKey_R, VKey) && transform_shortcuts_enabled) action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Rotate}, action::Phase::Cancel);
            else if (Shortcut(ImGuiKey_S, VKey) && scale_shortcut_enabled) action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Scale}, action::Phase::Cancel);
            else if (Shortcut(ImGuiKey_H, VKey)) action::Emit(action::object::ToggleHidden{});
            else if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_P, VKey)) action::Emit(action::object::ParentToActive{});
            else if (Shortcut(ImGuiMod_Alt | ImGuiKey_P, VKey)) action::Emit(action::object::ClearParent{});
        }
    }

    const bool active_transform = r.get<const GizmoInteraction>(viewport).IsUsing();
    if (active_transform) {
        // TransformGizmo overrides this mouse cursor during some actions, so this is a default.
        SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        WrapMousePos(GetCurrentWindowRead()->InnerClipRect, frame.AccumulatedWrapMouseDelta);
    } else {
        frame.AccumulatedWrapMouseDelta = {0, 0};
    }
    if (active_transform) return; // Only transform gizmo should consume viewport mouse input while active.

    if (!IsWindowHovered() && !frame.BoxSelectStart) return;

    // Mouse wheel for camera rotation, Cmd+wheel to zoom.
    const auto &io = GetIO();
    if (const vec2 wheel = std::exchange(frame.PreciseWheelDelta, vec2{0}); wheel != vec2{0, 0}) {
        if (io.KeyCtrl || io.KeySuper) EmitViewNav(r, action::view::ZoomViewCamera{.Factor = std::pow(WheelZoomStep, -wheel.y)});
        else EmitViewNav(r, action::view::OrbitViewCamera{.DeltaRad = wheel * WheelOrbitRadPerUnit});
    }
    if (OrientationGizmo::IsActive() || frame.OverlayControlsHovered) return;

    const auto render_extent = RenderExtentPx(r);
    // Pick against the preceding rendered view; the live camera may already have advanced its animation.
    const auto selection_view = r.ctx().get<const GpuBuffers>().FrameView;
    const auto edit_mode = r.get<const EditMode>(viewport).Value;
    const auto arm_obj_entity = FindArmatureObject(r, active_entity);
    const bool active_is_armature = arm_obj_entity != state::Null;
    const bool bone_mode = interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && active_is_armature);
    if (r.get<const BoxSelectState>(viewport).Gesture == SelectionGesture::Box && interaction_mode != InteractionMode::Excite) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            frame.BoxSelectStart = frame.BoxSelectEnd = ToVec2(GetMousePos());
            frame.BoxSelectStaged = false;
            if (IsKeyDown(ImGuiMod_Shift)) action::Emit(action::selection::SnapshotBoxSelectBaseline{});
        } else if (IsMouseDown(ImGuiMouseButton_Left) && frame.BoxSelectStart) {
            frame.BoxSelectEnd = ToVec2(GetMousePos());
            if (const auto box_px = ComputeBoxSelectPixels(*frame.BoxSelectStart, *frame.BoxSelectEnd, ToVec2(GetCursorScreenPos()), logical_extent, render_extent); box_px) {
                const bool is_additive = r.all_of<AdditiveBoxSelectBaseline>(viewport);
                frame.BoxSelectStaged = true;
                // The hit set (object/bone instances or edit-mode elements) is resolved later.
                action::Emit(action::selection::ApplyBoxSelect{.BoxPx = *box_px, .Additive = is_additive, .View = std::make_unique<RenderView>(selection_view)}, action::Phase::Stage);
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left) && frame.BoxSelectStart) {
            frame.BoxSelectStart.reset();
            frame.BoxSelectEnd.reset();
            // Gated on the same condition that staged, so a staged box-select always commits.
            if (frame.BoxSelectStaged) action::Emit(action::selection::ClearBoxSelectBaseline{});
            frame.BoxSelectStaged = false;
        }
        if (frame.BoxSelectStart) return;
    }

    const auto mouse_pos_render = ToRenderPx(r, ToVec2(GetMousePos()));
    const float max_x = float(std::max(render_extent.x, 1u) - 1u);
    const float max_y = float(std::max(render_extent.y, 1u) - 1u);
    // ImGui's origin and the picking pass's pixel rows both start at the top left.
    const uvec2 mouse_px{Clamp(mouse_pos_render.x, 0.0f, max_x), Clamp(mouse_pos_render.y, 0.0f, max_y)};

    if (interaction_mode == InteractionMode::Excite) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            if (const auto hit_entities = RunObjectPick(r, mouse_px); !hit_entities.empty()) {
                if (const auto hit_entity = hit_entities.front(); r.all_of<SoundVertices>(hit_entity)) {
                    if (const auto vertex = RunSoundVerticesVertexPick(r, hit_entity, mouse_px)) {
                        action::Emit(action::audio::ApplyExciteImpact{.InstanceEntity = hit_entity, .VertexIndex = *vertex});
                    }
                }
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left)) {
            action::Emit(action::audio::ClearExciteImpacts{});
        }
        return;
    }
    if (interaction_mode == InteractionMode::Edit && !active_is_armature && CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_R, VKey)) {
        // The pick takes the edge under the cursor, and the cut follows once the pick resolves.
        if (edit_mode != Element::Edge) action::Emit(action::view::SetEditMode{.Mode = Element::Edge});
        action::EmitSystem(action::selection::ApplyEditElementClick{.MousePx = mouse_px, .Toggle = false, .View = std::make_unique<RenderView>(selection_view)});
        action::EmitSystem(action::mesh::LoopCut{});
        return;
    }
    if (interaction_mode == InteractionMode::Edit && !active_is_armature && IsMouseClicked(ImGuiMouseButton_Right)) OpenPopup("##MeshContext");
    if (!IsSingleClicked(ImGuiMouseButton_Left)) return;
    if (interaction_mode == InteractionMode::Edit && edit_mode == Element::None && !active_is_armature) return;

    if (interaction_mode == InteractionMode::Edit && !active_is_armature) {
        const bool toggle = IsKeyDown(ImGuiMod_Shift) || IsKeyDown(ImGuiMod_Ctrl) || IsKeyDown(ImGuiMod_Super);
        action::Emit(action::selection::ApplyEditElementClick{.MousePx = mouse_px, .Toggle = toggle, .View = std::make_unique<RenderView>(selection_view)});
    } else if (interaction_mode == InteractionMode::Object || bone_mode) {
        const bool shift = IsKeyDown(ImGuiMod_Shift);
        // Store only the pixel, the GPU pick and selection resolution run later.
        // A re-click at the same spot cycles to the next overlapping hit.
        if (ImLengthSqr(CurrentClickPos - PrevClickPos) > 16) action::Emit(action::selection::Pick{mouse_px, shift, std::make_unique<RenderView>(selection_view)});
        else action::Emit(action::selection::PickCycle{mouse_px, shift, std::make_unique<RenderView>(selection_view)});
    }
}

void InteractOverlay(state::Scene &r, state::Entity viewport, FrameState &frame) {
    const profile::CpuScope scope{"ViewportOverlayUi"};
    const auto &icons = r.ctx().get<const ViewportIcons>();
    const rect viewport_rect{ToVec2(GetWindowPos()), ToVec2(GetContentRegionAvail())};
    const bool active_transform = r.get<const GizmoInteraction>(viewport).IsUsing();
    static constexpr float OrientationGizmoSize{84};
    const OverlayIconButtonStyle overlay_button_style{};
    const float overlay_corner_gap = GetTextLineHeightWithSpacing() / 2.f;
    const OverlayIconButtonStyle shading_button_style{
        .ButtonSize = {overlay_button_style.ButtonSize.x * 0.75f, overlay_button_style.ButtonSize.y * 0.75f},
        .Padding = overlay_button_style.Padding,
        .IconScale = overlay_button_style.IconScale,
        .CornerRounding = overlay_button_style.CornerRounding * 0.75f,
    };
    // Preserve the guard through release because IsSingleClicked fires on release.
    if (!IsMouseDown(ImGuiMouseButton_Left)) frame.OverlayControlsHovered = false;
    const bool any_popup_open = IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);

    { // Transform mode pill buttons (top-left overlay)
        using enum TransformGizmo::Type;
        const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
        const bool has_frozen_selected = r.view<Selected, ScaleLocked>().begin() != r.view<Selected, ScaleLocked>().end();
        const bool edit_transform_locked = interaction_mode == InteractionMode::Edit &&
            any_of(selection::GetSelectedMeshEntities(r), [&](state::Entity mesh_entity) { return selection::HasScaleLockedInstance(r, mesh_entity); });
        const bool transform_enabled = !edit_transform_locked;
        const bool scale_enabled = transform_enabled && !has_frozen_selected;

        const ui::Edit gizmo_edit{r, viewport};
        const auto transform_type = r.get<const TransformGizmoState>(viewport).Config.Type;
        if (!transform_enabled && transform_type != None) {
            gizmo_edit.Set<&TransformGizmoState::Config, &TransformGizmo::Config::Type>(None);
        } else if (!scale_enabled && transform_type == Scale) {
            gizmo_edit.Set<&TransformGizmoState::Config, &TransformGizmo::Config::Type>(Translate);
        }

        const auto start_pos = std::bit_cast<ImVec2>(viewport_rect.pos) + GetWindowContentRegionMin() + ImVec2{overlay_corner_gap, overlay_corner_gap};
        static constexpr float gap{4}; // Gap between select buttons and transform buttons
        const float button_h = overlay_button_style.ButtonSize.y;
        const auto make_button = [](const SvgResource *icon, ImVec2 offset, ImDrawFlags corners, bool enabled, bool active, const char *tooltip = nullptr) {
            return OverlayIconButtonInfo{icon, offset, corners, enabled, active, tooltip};
        };
        const auto gesture = r.get<const BoxSelectState>(viewport).Gesture;
        const OverlayIconButtonInfo buttons[]{
            make_button(icons.Transform.SelectBox.get(), {0.f, 0.f}, ImDrawFlags_RoundCornersTop, true, transform_type == None && gesture == SelectionGesture::Box),
            make_button(icons.Transform.Select.get(), {0.f, button_h}, ImDrawFlags_RoundCornersBottom, true, transform_type == None && gesture == SelectionGesture::Click),
            make_button(icons.Transform.Move.get(), {0.f, button_h * 2.f + gap}, ImDrawFlags_RoundCornersTop, transform_enabled, transform_type == Translate),
            make_button(icons.Transform.Rotate.get(), {0.f, button_h * 3.f + gap}, ImDrawFlags_RoundCornersNone, transform_enabled, transform_type == Rotate),
            make_button(icons.Transform.Scale.get(), {0.f, button_h * 4.f + gap}, ImDrawFlags_RoundCornersNone, scale_enabled, transform_type == Scale),
            make_button(icons.Transform.Universal.get(), {0.f, button_h * 5.f + gap}, ImDrawFlags_RoundCornersBottom, transform_enabled, transform_type == Universal),
        };

        if (const auto clicked = DrawOverlayIconButtonGroup("TransformModes", start_pos, buttons, !active_transform, &frame.OverlayControlsHovered, overlay_button_style)) {
            using Tool = action::view::SetActiveTool::Tool;
            action::Emit(action::view::SetActiveTool{*clicked == 0 ? Tool::SelectBox : *clicked == 1 ? Tool::SelectClick :
                                                         *clicked == 2                               ? Tool::Translate :
                                                         *clicked == 3                               ? Tool::Rotate :
                                                         *clicked == 4                               ? Tool::Scale :
                                                                                                       Tool::Universal});
        }
    }

    auto &settings = r.get<ViewportDisplay>(viewport);

    const auto shading_arrow_w = shading_button_style.ButtonSize.y * 0.55f;
    const float shading_button_w = shading_button_style.ButtonSize.x;
    const float shading_group_width = shading_button_w * 4.f + shading_arrow_w;
    const auto shading_button_h = shading_button_style.ButtonSize.y;
    const auto buttons_gap = 6.f;
    const auto shading_start = std::bit_cast<ImVec2>(viewport_rect.pos + vec2{GetWindowContentRegionMax().x - shading_group_width, GetWindowContentRegionMin().y}) + ImVec2{-overlay_corner_gap, overlay_corner_gap};

    { // Viewport shading button group + dropdown (top-right overlay)
        const auto start_pos = shading_start;
        const auto make_shading_button = [&](const SvgResource *icon, float x, ImDrawFlags corners, ViewportShadingMode mode, const char *tooltip) {
            return OverlayIconButtonInfo{icon, {x, 0.f}, corners, true, settings.ViewportShading == mode, tooltip};
        };
        const OverlayIconButtonInfo buttons[]{
            make_shading_button(icons.Shading.Wireframe.get(), 0.f, ImDrawFlags_RoundCornersLeft, ViewportShadingMode::Wireframe, "Wireframe"),
            make_shading_button(icons.Shading.Solid.get(), shading_button_w, ImDrawFlags_RoundCornersNone, ViewportShadingMode::Solid, "Solid"),
            make_shading_button(icons.Shading.MaterialPreview.get(), shading_button_w * 2.f, ImDrawFlags_RoundCornersNone, ViewportShadingMode::MaterialPreview, "Material Preview"),
            make_shading_button(icons.Shading.Rendered.get(), shading_button_w * 3.f, ImDrawFlags_RoundCornersNone, ViewportShadingMode::Rendered, "Rendered"),
        };

        if (const auto clicked = DrawOverlayIconButtonGroup("ViewportShading", start_pos, buttons, !active_transform, &frame.OverlayControlsHovered, shading_button_style)) {
            action::Emit(action::view::SetViewportShading{
                .Mode = *clicked == 0 ? ViewportShadingMode::Wireframe : *clicked == 1 ? ViewportShadingMode::Solid :
                    *clicked == 2                                                      ? ViewportShadingMode::MaterialPreview :
                                                                                         ViewportShadingMode::Rendered,
            });
        }

        DrawOverlayDropdownArrow(start_pos + ImVec2{shading_button_w * 4.f, 0.f}, {shading_arrow_w, shading_button_h}, shading_button_style, "##ShadingArrow", "##ShadingDropdown", frame.OverlayControlsHovered);
        { // Dropdown popup
            SetNextWindowPos(start_pos + ImVec2{shading_group_width, shading_button_h + 2.f}, ImGuiCond_Always, {1.f, 0.f});
            PushStyleVar(ImGuiStyleVar_WindowPadding, {8, 8});
            if (BeginPopup("##ShadingDropdown")) {
                frame.OverlayControlsHovered = true;
                TextUnformatted("Viewport shading");
                Separator();
                const auto current_mode = settings.ViewportShading;

                const auto render_pbr_controls = [&]<typename T>(const char *id) {
                    PushID(id);
                    using L = PBRViewportLighting;
                    const auto &lighting = r.get<const T>(viewport).Value;
                    auto edit = ui::Edit{r, viewport}.template Sub<&T::Value>();
                    if (Button("Reset")) action::Emit(action::view::ResetPbrLighting{.Rendered = std::is_same_v<T, RenderedLighting>});
                    edit.template Check<&L::UseSceneLights>("Scene lights");
                    SameLine();
                    edit.template Check<&L::UseSceneWorld>("Scene world");
                    if (lighting.UseSceneWorld) {
                        if (r.all_of<ImageLight>(viewport)) ui::Edit{r, viewport}.Slider<&ImageLight::Intensity>("Intensity", 0.f, 2.f, "%.2f");
                    } else {
                        const auto hdris = GetHdriRefs(r);
                        if (BeginCombo("Environment", hdris.Names[hdris.ActiveIndex].c_str())) {
                            for (uint32_t i = 0; i < hdris.Names.size(); ++i) {
                                const bool selected = (i == hdris.ActiveIndex);
                                if (Selectable(hdris.Names[i].c_str(), selected)) action::Emit(action::view::SetStudioEnvironment{hdris.Names[i]});
                                if (selected) SetItemDefaultFocus();
                            }
                            EndCombo();
                        }
                        edit.template Run<&L::EnvIntensity>([](float &v) { return SliderFloat("Intensity", &v, 0.f, 2.f, "%.2f"); });
                        edit.template Run<&L::EnvRotationDegrees>([](float &v) { return SliderFloat("Rotation", &v, -180.f, 180.f, "%.1f deg"); });
                    }
                    edit.template Run<&L::BackgroundBlur>([](float &v) { return SliderFloat("Blur", &v, 0.f, 1.f, "%.2f"); });
                    edit.template Run<&L::WorldOpacity>([](float &v) { return SliderFloat("World opacity", &v, 0.f, 1.f, "%.2f"); });
                    edit.template Check<&L::RealTransmission>("Real transmission");
                    if (IsItemHovered()) SetTooltip("Sample transmission from a pre-rendered scene framebuffer instead of from the IBL.");
                    // AlwaysClamp: extreme typed EV values overflow exp2 into inf/NaN in the renderer.
                    edit.template Run<&L::ExposureEV>([](float &v) { return SliderFloat("Exposure", &v, -10.f, 10.f, "%.1f EV", ImGuiSliderFlags_AlwaysClamp); });
                    PopID();
                };

                if (WorkbenchShading(current_mode)) {
                    const bool wireframe = current_mode == ViewportShadingMode::Wireframe;
                    ui::Edit edit{r, viewport};
                    if (wireframe) edit.Check<&ViewportDisplay::XRayWireframe>("##XRay");
                    else edit.Check<&ViewportDisplay::XRaySolid>("##XRay");
                    SameLine();
                    PushStyleVar(ImGuiStyleVar_Alpha, GetStyle().Alpha * (XRayFlag(settings) ? 1.f : 0.5f));
                    const auto opacity_slider = [](float &v) { return SliderFloat("X-ray", &v, 0.f, 1.f, "%.2f"); };
                    if (wireframe) edit.Run<&ViewportDisplay::XRayAlphaWireframe>(opacity_slider);
                    else edit.Run<&ViewportDisplay::XRayAlpha>(opacity_slider);
                    PopStyleVar();
                }

                if (current_mode == ViewportShadingMode::MaterialPreview) {
                    SeparatorText("Material Preview lighting");
                    render_pbr_controls.template operator()<MaterialPreviewLighting>("MatPreviewLighting");
                } else if (current_mode == ViewportShadingMode::Rendered) {
                    SeparatorText("Rendered lighting");
                    render_pbr_controls.template operator()<RenderedLighting>("RenderedLighting");
                } else if (current_mode == ViewportShadingMode::Solid) {
                    SeparatorText("Solid lighting");
                    auto lights = r.get<const WorkspaceLights>(viewport);
                    bool changed = false;
                    if (Button("Reset##Lighting")) {
                        lights = Defaults::WorkspaceLights;
                        changed = true;
                    }
                    bool use_specular = lights.UseSpecular != 0;
                    if (Checkbox("Specular highlights", &use_specular)) {
                        lights.UseSpecular = use_specular ? 1 : 0;
                        changed = true;
                    }
                    // Light colors are stored in linear space. Display/edit as sRGB.
                    static const auto linear_color_edit = [](const char *label, vec3 &linear) -> bool {
                        if (auto srgb = Pow(linear, vec3{1.f / 2.2f}); ColorEdit3(label, &srgb[0])) {
                            linear = Pow(srgb, vec3{2.2f});
                            return true;
                        }
                        return false;
                    };
                    changed |= linear_color_edit("Ambient color", lights.AmbientColor);
                    static const char *const light_names[]{"Light 1", "Light 2", "Light 3", "Light 4"};
                    for (int i = 0; i < 4; i++) {
                        auto &light = lights.Lights[i];
                        if (CollapsingHeader(light_names[i])) {
                            PushID(i);
                            if (SliderFloat3("Direction", &light.Direction[0], -1, 1)) {
                                const float len = sqrtf(light.Direction[0] * light.Direction[0] + light.Direction[1] * light.Direction[1] + light.Direction[2] * light.Direction[2]);
                                if (len > 0.0001f) {
                                    light.Direction[0] /= len;
                                    light.Direction[1] /= len;
                                    light.Direction[2] /= len;
                                }
                                changed = true;
                            }
                            changed |= linear_color_edit("Diffuse color", light.DiffuseColor);
                            changed |= linear_color_edit("Specular color", light.SpecularColor);
                            changed |= SliderFloat("Wrap", &light.Wrap, 0, 1);
                            PopID();
                        }
                    }
                    if (changed) action::Emit(action::view::SetWorkspaceLights{std::make_unique<WorkspaceLights>(lights)});
                }

                if (current_mode == ViewportShadingMode::MaterialPreview || current_mode == ViewportShadingMode::Rendered) {
                    SeparatorText("Debug");
                    struct DebugChannelEntry {
                        DebugChannel Value;
                        const char *Label;
                    };
                    struct DebugChannelGroup {
                        const char *Label; // null = no header (used for the leading "None" entry)
                        std::initializer_list<DebugChannelEntry> Entries;
                    };
                    static const DebugChannelGroup groups[]{
                        {nullptr, {{DebugChannel::None, "None"}}},
                        {"Generic", {
                                        {DebugChannel::UvCoords0, "Texture Coordinates 0"},
                                        {DebugChannel::UvCoords1, "Texture Coordinates 1"},
                                        {DebugChannel::NormalTexture, "Normal Texture"},
                                        {DebugChannel::NormalGeometry, "Geometry Normal"},
                                        {DebugChannel::Tangent, "Geometry Tangent"},
                                        {DebugChannel::Bitangent, "Geometry Bitangent"},
                                        {DebugChannel::TangentW, "Geometry Tangent W"},
                                        {DebugChannel::NormalShading, "Shading Normal"},
                                        {DebugChannel::Alpha, "Alpha"},
                                        {DebugChannel::Occlusion, "Occlusion"},
                                        {DebugChannel::Emissive, "Emissive"},
                                    }},
                        {"Metallic-Roughness", {
                                                   {DebugChannel::BaseColor, "Base Color"},
                                                   {DebugChannel::Metallic, "Metallic"},
                                                   {DebugChannel::Roughness, "Roughness"},
                                               }},
                        {"Clearcoat", {
                                          {DebugChannel::ClearcoatFactor, "Clearcoat Strength"},
                                          {DebugChannel::ClearcoatRoughness, "Clearcoat Roughness"},
                                          {DebugChannel::ClearcoatNormal, "Clearcoat Normal"},
                                      }},
                        {"Sheen", {
                                      {DebugChannel::SheenColor, "Sheen Color"},
                                      {DebugChannel::SheenRoughness, "Sheen Roughness"},
                                  }},
                        {"Specular", {
                                         {DebugChannel::SpecularFactor, "Specular Strength"},
                                         {DebugChannel::SpecularColor, "Specular Color"},
                                     }},
                        {"Transmission", {
                                             {DebugChannel::TransmissionFactor, "Transmission Strength"},
                                             {DebugChannel::VolumeThickness, "Volume Thickness"},
                                         }},
                        {"Diffuse Transmission", {
                                                     {DebugChannel::DiffuseTransmissionFactor, "Diffuse Transmission Strength"},
                                                     {DebugChannel::DiffuseTransmissionColor, "Diffuse Transmission Color"},
                                                 }},
                        {"Iridescence", {
                                            {DebugChannel::IridescenceFactor, "Iridescence Strength"},
                                            {DebugChannel::IridescenceThickness, "Iridescence Thickness"},
                                        }},
                        {"Anisotropy", {
                                           {DebugChannel::AnisotropyStrength, "Anisotropic Strength"},
                                           {DebugChannel::AnisotropyDirection, "Anisotropic Direction"},
                                       }},
                    };
                    const auto label_for = [&](DebugChannel ch) {
                        for (const auto &group : groups) {
                            for (const auto &entry : group.Entries) {
                                if (entry.Value == ch) return entry.Label;
                            }
                        }
                        return "None";
                    };
                    if (BeginCombo("Channel", label_for(settings.DebugChannel))) {
                        for (const auto &group : groups) {
                            if (group.Label) SeparatorText(group.Label);
                            for (const auto &entry : group.Entries) {
                                const bool selected = entry.Value == settings.DebugChannel;
                                if (Selectable(entry.Label, selected) && !selected) {
                                    action::Emit(action::UpdateOn<&ViewportDisplay::DebugChannel>(viewport, entry.Value));
                                }
                                if (selected) SetItemDefaultFocus();
                            }
                        }
                        EndCombo();
                    }
                }

                EndPopup();
            }
            PopStyleVar();
        }
    }

    { // X-ray toggle, left of the shading group as in Blender
        const bool applies = WorkbenchShading(settings.ViewportShading) || r.get<const Interaction>(viewport).Mode == InteractionMode::Edit;
        const auto start_pos = shading_start - ImVec2{buttons_gap + shading_button_w, 0.f};
        const OverlayIconButtonInfo button[]{
            {icons.XRay.get(), {0.f, 0.f}, ImDrawFlags_RoundCornersAll, true, XRayFlag(settings), "Toggle X-ray", !applies},
        };
        if (DrawOverlayIconButtonGroup("ViewportXRay", start_pos, button, !active_transform, &frame.OverlayControlsHovered, shading_button_style)) {
            action::Emit(action::view::ToggleXRay{});
        }
    }

    { // Viewport overlays toggle + dropdown
        const auto arrow_w = shading_arrow_w;
        const auto icon_w = shading_button_w;
        const auto button_h = shading_button_h;
        const auto overlay_group_width = icon_w + arrow_w;
        const auto group_start = shading_start - ImVec2{2.f * buttons_gap + shading_button_w + overlay_group_width, 0.f};

        {
            const OverlayIconButtonInfo icon_button[]{
                {icons.Overlay.get(), {0.f, 0.f}, ImDrawFlags_RoundCornersLeft, true, settings.ShowOverlays, "Toggle overlays"},
            };
            if (const auto clicked = DrawOverlayIconButtonGroup("ViewportOverlays", group_start, icon_button, !active_transform, &frame.OverlayControlsHovered, shading_button_style)) {
                action::Emit(action::UpdateOn<&ViewportDisplay::ShowOverlays>(viewport, !settings.ShowOverlays));
            }
        }
        DrawOverlayDropdownArrow(group_start + ImVec2{icon_w, 0.f}, {arrow_w, button_h}, shading_button_style, "##OverlayArrow", "##OverlayDropdown", frame.OverlayControlsHovered);
        { // Dropdown popup
            SetNextWindowPos(group_start + ImVec2{0.f, button_h + 2.f});
            PushStyleVar(ImGuiStyleVar_WindowPadding, {8, 8});
            if (BeginPopup("##OverlayDropdown")) {
                frame.OverlayControlsHovered = true;
                TextUnformatted("Viewport overlays");
                Separator();
                ui::Edit f{r, viewport};
                f.Check<&ViewportDisplay::ShowGrid>("Grid");
                f.Check<&ViewportDisplay::ShowExtras>("Extras");
                f.Check<&ViewportDisplay::ShowBones>("Bones");
                f.Check<&ViewportDisplay::ShowOrigins>("Origins");
                f.Check<&ViewportDisplay::ShowOutlineSelected>("Outline selected");
                EndPopup();
            }
            PopStyleVar();
        }
    }

    const auto &camera = r.get<const ViewCamera>(viewport);
    { // Orientation gizmo
        const float shading_group_height = shading_button_style.ButtonSize.y;
        const auto pos = viewport_rect.pos + vec2{GetWindowContentRegionMax().x - OrientationGizmoSize, GetWindowContentRegionMin().y} + vec2{-overlay_corner_gap, overlay_corner_gap * 2 + shading_group_height};
        if (auto interaction = OrientationGizmo::Interact(pos, OrientationGizmoSize, camera, !active_transform && !any_popup_open)) {
            std::visit(
                overloaded{
                    [&](OrientationGizmo::RotateBy rot) { EmitViewNav(r, action::view::OrbitViewCamera{rot.Delta}); },
                    [&](OrientationGizmo::AlignTo a) { EmitViewNav(r, action::view::SetViewCameraTargetDirection{a.Direction}); },
                },
                *interaction
            );
        }
    }
    const auto selected_view = r.view<const Selected>();
    const auto bone_selected_view = r.view<const BoneSelection>();
    const auto edit_mode = r.get<const EditMode>(viewport).Value;
    const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
    const auto active_entity = FindActiveEntity(r);
    const auto arm_obj = FindArmatureObject(r, active_entity);
    const bool bone_edit_mode = interaction_mode == InteractionMode::Edit && arm_obj != state::Null;
    const bool bone_mode = bone_edit_mode || (interaction_mode == InteractionMode::Pose && arm_obj != state::Null);
    const bool mesh_edit_mode = interaction_mode == InteractionMode::Edit && !bone_edit_mode;

    const auto has_transform_target = [&]() {
        if (bone_mode) return !bone_selected_view.empty();
        if (selected_view.empty()) return false;
        if (!mesh_edit_mode) return true;
        for (const auto [e, instance] : r.view<const Instance, const Selected>(state::Exclude<ScaleLocked>).each()) {
            const auto *stats = GetElementSelectionSummary(r, instance.Entity, edit_mode);
            if (stats && stats->SelectedCount > 0) return true;
        }
        return false;
    }();
    if (has_transform_target) { // Transform gizmo
        // Transform root selections around their average position using the active entity's rotation and scale.
        const auto gizmo_active_entity = bone_mode ? FindActiveBone(r) : active_entity;
        const auto active_transform = [&]() -> Transform {
            if (gizmo_active_entity == state::Null) return {};
            const auto &wt = r.get<WorldTransform>(gizmo_active_entity);
            return wt;
        }();

        const auto root_selected = RootSelectedForTransform(r, viewport);
        const auto root_count = root_selected.size();
        vec3 pivot{};
        if (mesh_edit_mode) {
            pivot = EditSelectionCenter(r, edit_mode);
            // Apply pending transform to gizmo position (vertices aren't modified until commit).
            if (const auto *pending = r.try_get<const PendingTransform>(viewport)) {
                pivot += pending->Delta.P;
            }
        } else {
            if (bone_edit_mode) {
                // Bone pivot: contribute head position for selected Root, tail for selected Tip.
                // A fully-selected bone (Root+Tip+Body) contributes both, giving the midpoint.
                vec3 pivot_sum{};
                uint32_t pivot_count = 0;
                for (const auto e : root_selected) {
                    const auto &wt = r.get<WorldTransform>(e);
                    const auto *parts = r.try_get<const BoneSelection>(e);
                    if (!parts || parts->Root) {
                        pivot_sum += wt.P;
                        ++pivot_count;
                    }
                    if (parts && parts->Tip) {
                        const float bl = r.get<BoneDisplayScale>(e).Value;
                        pivot_sum += wt.P + Rotate(wt.R, vec3{0, bl, 0});
                        ++pivot_count;
                    }
                }
                pivot = pivot_count > 0 ? pivot_sum / float(pivot_count) : vec3{};
            } else {
                pivot = fold_left(root_selected | transform([&](auto e) { return r.get<WorldTransform>(e).P; }), vec3{}, [](vec3 sum, vec3 position) { return sum + position; }) / float(root_count);
            }
        }

        const auto start_transform_view = r.view<const StartTransform>();
        const auto &gizmo_state = r.get<const TransformGizmoState>(viewport);
        auto &gizmo = r.edit<GizmoInteraction>(viewport);
        const auto gizmo_transform = GizmoTransform{{.P = pivot, .R = active_transform.R, .S = active_transform.S}, gizmo_state.Mode};
        const auto *start_screen = r.try_get<const StartScreenTransform>(viewport);
        const bool was_using = gizmo.IsUsing();
        auto interact_result = TransformGizmo::Interact(
            gizmo,
            gizmo_transform,
            gizmo_state.Config, camera, viewport_rect, ToVec2(GetMousePos()) + frame.AccumulatedWrapMouseDelta,
            start_screen ? std::optional{start_screen->Value} : std::nullopt
        );
        if (gizmo.Cancelled) {
            // Escape discards the whole gesture, including an operator whose placement this drag was.
            gizmo.Cancelled = false;
            action::Cancel();
        } else if (interact_result) {
            const auto &[ts, td] = *interact_result;
            if (mesh_edit_mode) {
                // Mesh Edit mode: store pending transform for shader-based preview.
                // Actual vertex positions are only modified on commit.
                action::Emit(action::view::TransformElements{std::make_unique<PendingTransform>(ts.P, ts.R, td)}, action::Phase::Stage);
            } else {
                // Object/bone mode: store the gizmo pivot + delta. Apply recomputes per-entity transforms.
                action::Emit(action::view::TransformSelection{std::make_unique<PendingTransform>(ts.P, ts.R, td)}, action::Phase::Stage);
            }
        } else if (was_using || !start_transform_view.empty()) {
            action::Emit(action::view::EndTransform{});
        }

        gizmo.RenderTransform = gizmo_transform;
        if (interact_result) gizmo.RenderTransform->P = interact_result->Start.P + interact_result->Delta.P;
    }

    if (mesh_edit_mode) {
        DrawMeshOperatorMenus(r, viewport);
        DrawLastOperationPanel(r, viewport, viewport_rect);
    }

    // The latch is a handoff from the operator's Apply to this overlay, consumed here rather than through an action.
    r.remove<StartScreenTransform>(viewport);
}

void DrawOverlay(state::Scene &r, state::Entity viewport, FrameState &frame) {
    const rect viewport_rect{ToVec2(GetWindowPos()), ToVec2(GetContentRegionAvail())};
    const auto axes = colors::MakeAxes(r.get<const ViewportTheme>(viewport).AxisColors);
    const auto &camera = r.get<const ViewCamera>(viewport);

    OrientationGizmo::Render(axes);
    TransformGizmo::Render(r.edit<GizmoInteraction>(viewport), r.get<const TransformGizmoState>(viewport).Config.Type, camera, viewport_rect, axes);

    const auto &settings = r.get<const ViewportDisplay>(viewport);
    if (settings.ShowOverlays && settings.ShowOrigins && (!r.view<const Selected>().empty() || !r.view<const Active>().empty())) {
        const auto &theme = r.get<const ViewportTheme>(viewport);
        const auto vp = camera.Projection(viewport_rect.size.x / viewport_rect.size.y) * camera.View();
        auto draw_dot = [&](vec3 pos, bool is_active) {
            const auto p_cs = vp * vec4{pos, 1.f};
            if (p_cs.w <= 0) return; // Behind camera

            const auto p_ndc = vec3{p_cs} / p_cs.w;
            const auto p_uv = NdcToUv(vec2{p_ndc});
            const auto p_px = std::bit_cast<ImVec2>(viewport_rect.pos + p_uv * viewport_rect.size);
            auto &dl = *GetWindowDrawList();
            dl.AddCircleFilled(p_px, 3.5f, colors::RgbToU32(is_active ? theme.Colors.ObjectActive : theme.Colors.ObjectSelected), 10);
            dl.AddCircle(p_px, 3.5f, IM_COL32(0, 0, 0, 255), 10, 1.f);
        };
        const auto origins = SortedEntities(
            r.view<const WorldTransform>(state::Exclude<SubElementOf>) |
                std::views::filter([&](auto e) { return r.any_of<Active, Selected>(e); }),
            std::ranges::greater{}
        );
        for (const auto e : origins) draw_dot(r.get<const WorldTransform>(e).P, r.all_of<Active>(e));
    }

    if (frame.BoxSelectStart && frame.BoxSelectEnd) {
        auto &dl = *GetWindowDrawList();
        const auto box_min = Min(*frame.BoxSelectStart, *frame.BoxSelectEnd);
        const auto box_max = Max(*frame.BoxSelectStart, *frame.BoxSelectEnd);
        dl.AddRectFilled(std::bit_cast<ImVec2>(box_min), std::bit_cast<ImVec2>(box_max), IM_COL32(255, 255, 255, 30));

        // Dashed outline: dashes step from `a` toward `b` along their one differing axis.
        static constexpr auto outline_color{IM_COL32(255, 255, 255, 200)};
        static constexpr float dash_size{4}, gap_size{4};
        const auto dash_line = [&](vec2 a, vec2 b) {
            const uint32_t axis = a.x == b.x ? 1 : 0;
            for (float v = a[axis]; v < b[axis]; v += dash_size + gap_size) {
                auto d0 = a, d1 = b;
                d0[axis] = v;
                d1[axis] = Min(v + dash_size, b[axis]);
                dl.AddLine(std::bit_cast<ImVec2>(d0), std::bit_cast<ImVec2>(d1), outline_color, 1.f);
            }
        };
        dash_line({box_min.x, box_min.y}, {box_max.x, box_min.y});
        dash_line({box_min.x, box_max.y}, {box_max.x, box_max.y});
        dash_line({box_min.x, box_min.y}, {box_min.x, box_max.y});
        dash_line({box_max.x, box_min.y}, {box_max.x, box_max.y});
    }

    // Match the centered frame to the captured look-through region.
    if (const auto look_through_entity = LookThroughCameraEntity(r); look_through_entity != state::Null && !camera.IsAnimating()) {
        if (const auto cd = LensOf(r, look_through_entity)) {
            const float cam_aspect = AspectRatio(*cd);
            const auto frame_size = vec2{viewport_rect.size.y * cam_aspect, viewport_rect.size.y} * LookThroughFrameRatio(cam_aspect, viewport_rect.size.x / viewport_rect.size.y);
            const vec2 vp_center = viewport_rect.pos + viewport_rect.size * 0.5f;
            const vec2 fmin = vp_center - frame_size * 0.5f, fmax = vp_center + frame_size * 0.5f;
            const auto vmin = viewport_rect.pos, vmax = viewport_rect.pos + viewport_rect.size;

            // Dim the area outside the camera's view.
            auto &dl = *GetWindowDrawList();
            static constexpr auto dim = IM_COL32(0, 0, 0, 100);
            auto iv = [](vec2 v) { return std::bit_cast<ImVec2>(v); };
            dl.AddRectFilled(iv(vmin), iv({vmax.x, fmin.y}), dim);
            dl.AddRectFilled(iv({vmin.x, fmax.y}), iv(vmax), dim);
            dl.AddRectFilled(iv({vmin.x, fmin.y}), iv({fmin.x, fmax.y}), dim);
            dl.AddRectFilled(iv({fmax.x, fmin.y}), iv({vmax.x, fmax.y}), dim);
        }
    }
}
