#include "viewport/ViewportUi.h"

#include "Camera.h"
#include "Panel.h"
#include "Profile.h"
#include "TransformMath.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "gizmo/TransformGizmo.h"
#include "numeric/MatrixMath.h"
#include "procedure/Procedure.h"
#include "project/Project.h"
#include "render/Instance.h"
#include "render/TextureRefs.h"
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "ui/CtrlShortcut.h"
#include "ui/DashedLine.h"
#include "ui/FieldEdit.h"
#include "ui/InterfaceTheme.h"
#include "ui/InterfaceUi.h"
#include "viewport/FrameState.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/ScreenSpace.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportComponents.h"
#include "viewport/ViewportIcons.h"
#include "viewport/ViewportOps.h"
#include <imgui_internal.h>

#include "gizmo/OrientationGizmo.h"

using numeric::Min;

using namespace ImGui;

namespace {
constexpr vec2 ToVec2(ImVec2 v) { return std::bit_cast<vec2>(v); }
constexpr float WheelOrbitRadPerUnit{0.05f}, WheelZoomStep{1.04f};

// Return viewport-relative box fractions, or none for drags below the threshold or entirely outside the viewport.
std::optional<std::pair<vec2, vec2>> ComputeBoxSelect(vec2 start, vec2 end, vec2 window_pos, uvec2 logical_extent) {
    static constexpr float DragThresholdSq{2 * 2};
    if (Distance2(start, end) <= DragThresholdSq) return {};
    const vec2 logical_size{float(logical_extent.x), float(logical_extent.y)};
    const auto local_min = Max(Min(start, end) - window_pos, vec2{0});
    const auto local_max = Min(Max(start, end) - window_pos, logical_size);
    if (local_min.x > local_max.x || local_min.y > local_max.y) return {};
    return std::pair{local_min / logical_size, local_max / logical_size};
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

// During look-through, the first navigation input exits look-through instead.
void EmitViewNav(const state::Registry &r, state::Entity viewport, auto &&nav) {
    if (r.Resources.get<const FrameState>().BoxDrag) return;
    if (r.get<const ViewCameraState>(viewport).LookThrough != state::Null) action::Emit(action::view::ExitLookThroughCamera{viewport});
    else action::Emit(std::forward<decltype(nav)>(nav));
}

// A world point's screen position in logical pixels.
vec2 ScreenPx(const mat4 &vp, const rect &viewport_rect, vec3 p) {
    const auto cs = vp * vec4{p, 1.f};
    return viewport_rect.pos + NdcToUv(vec2{cs.x, cs.y} / cs.w) * viewport_rect.size;
}

vec2 ToViewFraction(const state::Registry &r, vec2 screen_px) {
    const auto logical = r.get<const ViewportPanel>(FindActiveViewport(r)).Extent;
    const auto local = screen_px - ToVec2(GetCursorScreenPos());
    return Clamp(vec2{local.x / float(std::max(logical.x, 1u)), local.y / float(std::max(logical.y, 1u))}, vec2{0}, vec2{1});
}
// A screen position in pixels of the render target.
vec2 ToRenderPx(const state::Registry &r, vec2 screen_px) {
    const auto viewport = FindActiveViewport(r);
    const auto logical = r.get<const ViewportPanel>(viewport).Extent;
    const auto render_extent = RenderExtentPx(r, viewport);
    const vec2 scale{logical.x > 0u ? float(render_extent.x) / float(logical.x) : 1.f, logical.y > 0u ? float(render_extent.y) / float(logical.y) : 1.f};
    return (screen_px - ToVec2(GetCursorScreenPos())) * scale;
}

void BeginMeshDrag(state::Registry &r, FrameState &frame, MeshOperatorDrag::Op op) {
    const rect viewport_rect{ToVec2(GetWindowPos()), ToVec2(GetContentRegionAvail())};
    const auto &camera = r.get<const ViewCameraState>(FindActiveViewport(r)).Active;
    const auto vp = camera.Projection(viewport_rect.size.x / viewport_rect.size.y) * camera.View();
    const auto center = EditSelectionCenter(r);
    const auto center_px = ScreenPx(vp, viewport_rect, center);
    // World units per logical pixel at the center, measured along camera right.
    const float pixels = Length(ScreenPx(vp, viewport_rect, center + camera.Basis()[0]) - center_px);
    frame.MeshDrag = MeshOperatorDrag{.Value = op, .StartPx = ToVec2(GetMousePos()), .CenterPx = center_px, .WorldPerPx = pixels > 0.f ? 1.f / pixels : 0.f};
}

void UpdateMeshDrag(state::Registry &r, FrameState &frame) {
    using Op = MeshOperatorDrag::Op;
    auto &drag = *frame.MeshDrag;
    const auto mouse = ToVec2(GetMousePos());
    const bool knife = drag.Value == Op::Knife;
    drag.WheelAccum += knife ? 0.f : std::exchange(frame.PreciseWheelDelta, vec2{0}).y;
    const int steps = int(drag.WheelAccum) + IsKeyPressed(ImGuiKey_Equal, true) - IsKeyPressed(ImGuiKey_Minus, true);
    drag.WheelAccum -= float(int(drag.WheelAccum));
    const auto segments = uint32_t(std::clamp(int64_t(drag.Segments) + steps, int64_t{1}, int64_t{UINT32_MAX}));
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
        action::Emit(action::mesh::Knife{.Start = start, .End = end}, action::Phase::Stage);
        drag.Staged = length;
        return;
    }
    // An inset closes toward the center and reopens past it, and a bevel widens away from the center.
    const float travel = Length(mouse - drag.CenterPx) - Length(drag.StartPx - drag.CenterPx);
    const float value = std::max((drag.Value == Op::Inset ? -travel : travel) * drag.WorldPerPx, 0.f);
    if ((value <= 0.f && !drag.Staged) || (drag.Staged == value && !changed)) return;
    // Each step previews the operator from the stage base.
    if (drag.Value == Op::Inset) {
        action::Emit(action::mesh::Inset{.Thickness = value, .Depth = 0.f, .Individual = drag.Individual, .Even = true}, action::Phase::Stage);
    } else {
        action::Emit(action::mesh::Bevel{.Width = value, .Segments = drag.Segments, .Vertices = drag.Value == Op::BevelVertices}, action::Phase::Stage);
    }
    drag.Staged = value;
}

void DrawMeshOperatorMenus(state::Registry &r) {
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
        for (size_t i = 0; i < field::EnumCount<MergeMode>; ++i) op(field::EnumLabels<MergeMode>[i], Merge { field::EnumValues<MergeMode>[i] });
    };
    const auto delete_items = [&] {
        for (size_t i = 0; i < 5; ++i) op(field::EnumLabels<DeleteMode>[i], action::mesh::Delete { field::EnumValues<DeleteMode>[i] });
        Separator();
        op("Dissolve Vertices", Dissolve{DissolveMode::Vertices});
        op("Dissolve Edges", Dissolve{DissolveMode::Edges});
        op("Dissolve Faces", Dissolve{DissolveMode::Faces});
        op("Limited Dissolve", Dissolve{DissolveMode::Limited});
        Separator();
        op("Edge Collapse", Merge{MergeMode::Collapse});
        op("Edge Loops", Dissolve{DissolveMode::Edges});
    };
    const auto vertex_items = [&] {
        op("Extrude Vertices", Extrude{ExtrudeMode::Vertices}, action::Phase::Stage);
        op("Smooth Vertices", SmoothVertices{});
        op("Curve Between Selected", CurveBetweenSelected{});
        op("To Circle", Circularize{});
        op("Flatten", Flatten{});
        op("Randomize", Randomize{});
        op("Vertex Slide", VertexSlide{.ViewSpace = true});
        op("To Sphere", ToSphere{});
        op("Push/Pull", PushPull{});
        op("Shear", Shear{});
        if (MenuItem("Warp")) action::Emit(Warp{});
        if (MenuItem("Bend")) action::Emit(Bend{});
        op("Shrink/Fatten", ShrinkFatten{});
        op("Bevel Vertices", Bevel{.Vertices = true});
        op("New Edge/Face from Vertices", Fill{});
        op("Connect Vertex Path", ConnectVertices{});
        op("Rip Vertices", Rip{}, action::Phase::Stage);
        Separator();
        submenu("Merge Vertices", merge_items);
        submenu("Separate", [&] {
            op("Selection", Separate{});
            op("By Loose Parts", Separate{SeparateMode::LooseParts});
            op("By Material", Separate{SeparateMode::Material});
        });
        op("Dissolve Vertices", Dissolve{DissolveMode::Vertices});
    };
    const auto edge_items = [&] {
        op("Edge Slide", EdgeSlide{.ViewSpace = true});
        op("Space Evenly", SpaceEvenly{});
        op("Relax Edge Loops", RelaxEdgeLoops{});
        op("Curve Between Selected", CurveBetweenSelected{});
        op("To Circle", Circularize{});
        op("Flatten", Flatten{});
        op("Extrude Edges", Extrude{ExtrudeMode::Edges}, action::Phase::Stage);
        op("Bevel Edges", Bevel{});
        op("Bridge Edge Loops", BridgeEdgeLoops{});
        op("Loop Cut", LoopCut{});
        op("Subdivide", Subdivide{});
        op("Un-Subdivide", Unsubdivide{});
        op("Decimate Geometry", Decimate{});
        Separator();
        op("Rotate Edge", EdgeRotate{});
        op("Edge Split", EdgeSplit{});
        op("Rip", Rip{}, action::Phase::Stage);
        op("Dissolve Edges", Dissolve{DissolveMode::Edges});
    };
    const auto face_items = [&] {
        op("Extrude Faces", Extrude{ExtrudeMode::Region}, action::Phase::Stage);
        op("Extrude Individual Faces", Extrude{ExtrudeMode::FacesIndividual}, action::Phase::Stage);
        op("Inset Faces", Inset{});
        op("Poke Faces", Poke{});
        op("Make Planar Faces", MakePlanarFaces{});
        op("To Circle", Circularize{});
        op("Flatten", Flatten{});
        op("Shrink/Fatten", ShrinkFatten{});
        op("Split Non-Planar Faces", SplitNonplanarFaces{});
        op("Split Concave Faces", SplitConcaveFaces{});
        op("Triangulate Faces", Triangulate{});
        op("Beautify Faces", BeautifyFaces{});
        op("Tris to Quads", TrisToQuads{});
        op("Solidify Faces", Solidify{});
        op("Wireframe", Wireframe{});
        Separator();
        op("Rotate UVs", RotateUVs{});
        op("Reverse UVs", ReverseUVs{});
        op("Rotate Colors", RotateColors{});
        op("Reverse Colors", ReverseColors{});
        Separator();
        op("Fill", Fill{});
        op("Grid Fill", GridFill{});
        op("Fill Holes", FillHoles{});
        Separator();
        op("Flip Normals", FlipNormals{});
        op("Recalculate Outside", RecalculateNormals{});
        op("Recalculate Inside", RecalculateNormals{true});
        op("Dissolve Faces", Dissolve{DissolveMode::Faces});
    };
    const auto cleanup_items = [&] {
        op("Delete Loose", action::mesh::Delete{DeleteMode::Loose});
        op("Degenerate Dissolve", Dissolve{DissolveMode::Degenerate});
        op("Limited Dissolve", Dissolve{DissolveMode::Limited});
        op("Merge by Distance", Merge{MergeMode::ByDistance});
        op("Fill Holes", FillHoles{});
    };
    const auto mesh_items = [&] {
        // The bisect plane and spin axis follow the view through the selection center.
        const auto center = EditSelectionCenter(r);
        op("Duplicate", action::mesh::Duplicate{}, action::Phase::Stage);
        op("Extrude Region", Extrude{}, action::Phase::Stage);
        op("Split", Split{});
        submenu("Separate", [&] {
            op("Selection", Separate{});
            op("By Loose Parts", Separate{SeparateMode::LooseParts});
            op("By Material", Separate{SeparateMode::Material});
        });
        Separator();
        op("Bisect", Bisect{.Point = center, .ViewSpace = true});
        op("Symmetrize", Symmetrize{});
        op("Snap to Symmetry", SnapSymmetry{});
        Separator();
        op("Hide Selected", action::mesh::Hide{});
        op("Hide Unselected", action::mesh::Hide{true});
        op("Reveal", action::mesh::Reveal{});
        op("Spin", Spin{.Center = center, .ViewSpace = true});
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

const procedure::OperationHandle *ActiveOperation(const state::Registry &r) {
    const auto *handle = r.try_get<const procedure::OperationHandle>(state::Document);
    if (!handle) return nullptr;
    const auto *node = r.try_get<const procedure::ProcedureNode>(handle->Node);
    const auto *object = r.try_get<const Instance>(handle->Object);
    return node && object && object->Entity == node->Procedure && r.all_of<procedure::NodeHandle>(handle->Node) ? handle : nullptr;
}

bool DrawOperationHandle(state::Registry &r, FrameState &frame, const ViewCamera &camera, rect viewport_rect) {
    using H = procedure::HandleParameter;
    const auto *handle = ActiveOperation(r);
    if (!handle) {
        frame.OperationStart.reset();
        return false;
    }
    const auto node = handle->Node;
    const auto &support = r.get<const procedure::NodeHandle>(node);
    const auto world = *WorldTransformOf(r, handle->Object);
    auto center = TransformPoint(world, support.Center);
    const auto saved_cursor = GetCursorScreenPos();
    PushID("OperationHandle");
    const bool transform = handle->Parameter == H::Transform || handle->Parameter == H::ExtrudeOffset;
    if (!transform) {
        const auto vp = camera.Projection(viewport_rect.size.x / viewport_rect.size.y) * camera.View();
        const auto at = ScreenPx(vp, viewport_rect, center);
        SetCursorScreenPos(ImVec2{at.x + 12.f, at.y + 12.f});
        SetNextItemWidth(130.f);
        const auto scalar = [&]<auto Member>(const char *label) {
            using C = field::Owner<Member>;
            auto value = r.get<const C>(node).*Member;
            ui::Gesture(DragFloat(label, &value, .01f), [&] { return action::UpdateOf<Member>(node, value); });
        };
        if (handle->Parameter == H::ExtrudeScale) scalar.operator()<&action::mesh::Extrude::OffsetScale>("Offset Scale");
        if (handle->Parameter == H::BevelWidth) scalar.operator()<&action::mesh::Bevel::Width>("Width");
        if (handle->Parameter == H::InsetThickness) scalar.operator()<&action::mesh::Inset::Thickness>("Thickness");
        if (handle->Parameter == H::InsetDepth) scalar.operator()<&action::mesh::Inset::Depth>("Depth");
        frame.OverlayControlsHovered |= IsItemHovered() || IsItemActive();
    } else {
        Transform value;
        Transform parameter_frame;
        if (handle->Parameter == H::Transform) {
            const auto &p = r.get<const procedure::TransformElements>(node);
            value = p.Delta;
            parameter_frame = p.Frame;
            const auto before = TransformPoint(p.Frame, support.Center);
            const auto after = p.Pivot + p.Delta.P + p.Delta.R * ((before - p.Pivot) * p.Delta.S);
            center = TransformPoint(world, InverseTransformPoint(p.Frame, after));
        } else {
            const auto &p = r.get<const action::mesh::Extrude>(node);
            value.P = (p.AlongNormals ? support.Normal : p.Offset) * p.OffsetScale;
            center = TransformPoint(world, support.Center + value.P);
        }
        auto &gizmo = r.edit<GizmoInteraction>(state::Document);
        const bool was_using = gizmo.IsUsing();
        if (!was_using || !frame.OperationStart || frame.OperationStart->Node != node) frame.OperationStart = FrameState::OperationDrag{node, value};
        const auto &settings = r.get<const TransformGizmoState>(state::Document);
        auto config = settings.Config;
        config.Type = handle->Parameter == H::Transform ? TransformGizmo::Type::Universal : TransformGizmo::Type::Translate;
        const GizmoTransform placement{{.P = center, .R = world.R * Conjugate(parameter_frame.R)}, settings.Mode};
        const auto interaction = TransformGizmo::Interact(gizmo, placement, config, camera, viewport_rect, ToVec2(GetMousePos()) + frame.AccumulatedWrapMouseDelta);
        if (gizmo.Cancelled) {
            gizmo.Cancelled = false;
            frame.OperationStart.reset();
            action::Cancel();
        } else if (interaction) {
            const auto &delta = interaction->Delta;
            auto changed = frame.OperationStart->Value;
            changed.P += parameter_frame.R * (((Conjugate(world.R) * delta.P) / world.S) * parameter_frame.S);
            const auto relative = parameter_frame.R * Conjugate(world.R);
            changed.R = Normalize(relative * delta.R * Conjugate(relative) * changed.R);
            changed.S *= delta.S;
            if (handle->Parameter == H::ExtrudeOffset) action::Emit(action::nodes::SetExtrudeOffset{node, changed.P}, action::Phase::Stage);
            else if (gizmo.Current->Type == TransformGizmo::TransformType::Translate) action::Emit(action::UpdateOf<&procedure::TransformElements::Delta, &Transform::P>(node, changed.P), action::Phase::Stage);
            else if (gizmo.Current->Type == TransformGizmo::TransformType::Rotate) action::Emit(action::UpdateOf<&procedure::TransformElements::Delta, &Transform::R>(node, changed.R), action::Phase::Stage);
            else action::Emit(action::UpdateOf<&procedure::TransformElements::Delta, &Transform::S>(node, changed.S), action::Phase::Stage);
        } else if (was_using) {
            frame.OperationStart.reset();
            action::Commit();
        }
        gizmo.RenderTransform = placement;
        if (interaction) gizmo.RenderTransform->P = interaction->Start.P + interaction->Delta.P;
    }
    PopID();
    SetCursorScreenPos(saved_cursor);
    return true;
}

TransformGizmo::Type EffectiveGizmoType(const state::Registry &r) {
    using enum TransformGizmo::Type;
    if (const auto *handle = ActiveOperation(r)) {
        return handle->Parameter == procedure::HandleParameter::Transform ? Universal : handle->Parameter == procedure::HandleParameter::ExtrudeOffset ? Translate :
                                                                                                                                                         None;
    }
    const auto type = r.get<const TransformGizmoState>(state::Document).Config.Type;
    const auto &flags = r.get<const SelectionFlags>(state::Document);
    if (!flags.Transformable) return None;
    return !flags.Scalable && type == Scale ? Translate : type;
}
} // namespace

void Interact(state::Registry &r, FrameState &frame) {
    // Any open popup (e.g. Viewport shading dropdown) blocks viewport mouse/keyboard input.
    // Without this, wheel/click events still patch the camera while the popup overlays the viewport_rect.
    if (IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
        frame.PreciseWheelDelta = {0, 0};
        frame.Magnification = 0.f;
        return;
    }

    static ImVec2 PrevClickPos{-FLT_MAX, -FLT_MAX}, CurrentClickPos{-FLT_MAX, -FLT_MAX};
    if (GetIO().MouseClicked[0]) {
        PrevClickPos = CurrentClickPos;
        CurrentClickPos = GetIO().MouseClickedPos[0];
    }

    const auto viewport = FindActiveViewport(r);
    const auto logical_extent = r.get<const ViewportPanel>(viewport).Extent;
    if (logical_extent.x == 0 || logical_extent.y == 0) return;

    if (frame.MeshDrag) {
        if (frame.MeshDrag->Staged && !project::Session(r).HasStaged()) frame.MeshDrag.reset();
        else return UpdateMeshDrag(r, frame);
    }

    const auto interaction_mode = r.get<const Interaction>(state::Document).Mode;
    const auto active_entity = FindActiveEntity(r);
    const auto &flags = r.get<const SelectionFlags>(state::Document);
    // Route shortcuts globally while preserving ImGui ownership for active widgets, navigation, and text input.
    constexpr auto VKey = ImGuiInputFlags_RouteGlobal;
    // The hovered node canvas takes the editing shortcuts instead.
    const bool nodes_hovered = r.Resources.get<const PanelsState>().Nodes.Hovered;
    const auto key = [&](ImGuiKeyChord chord, ImGuiInputFlags route = VKey) { return !nodes_hovered && Shortcut(chord, route); };
    const auto ctrl_key = [&](ImGuiKeyChord chord, ImGuiInputFlags route = VKey) { return !nodes_hovered && CtrlShortcut(chord, route); };
    if (r.get<const GizmoInteraction>(state::Document).IsUsing()) {
        // During an active transform, only allow transform switching shortcuts.
        if (key(ImGuiKey_G) && flags.Transformable) action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Translate}, action::Phase::Cancel);
        else if (key(ImGuiKey_R) && flags.Transformable) action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Rotate}, action::Phase::Cancel);
        else if (key(ImGuiKey_S) && flags.Scalable) action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Scale}, action::Phase::Cancel);
    } else {
        if (interaction_mode != InteractionMode::Edit) {
            if (key(ImGuiKey_I)) action::Emit(action::animation::InsertKey{{.Target = action::OnSelected{}}});
            else if (key(ImGuiMod_Alt | ImGuiKey_I)) action::Emit(action::animation::DeleteKey{{.Target = action::OnSelected{}}});
        }
        if (key(ImGuiKey_Space)) action::Emit(action::timeline::TogglePlay{r.get<const TimelinePlayback>(state::Document).CurrentFrame});
        else if (ctrl_key(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Space)) action::Emit(action::timeline::TogglePlay{r.get<const TimelinePlayback>(state::Document).CurrentFrame, /*Reverse=*/true});
        else if (key(ImGuiKey_Z)) {
            const auto current = r.get<const ViewportDisplay>(viewport).ViewportShading;
            const auto next = current == ViewportShadingMode::Solid ? ViewportShadingMode::MaterialPreview :
                current == ViewportShadingMode::MaterialPreview     ? ViewportShadingMode::Rendered :
                                                                      ViewportShadingMode::Solid;
            action::Emit(action::view::SetViewportShading{.Viewport = viewport, .Mode = next});
        } else if (key(ImGuiMod_Shift | ImGuiKey_Z)) {
            const auto &settings = r.get<const ViewportDisplay>(viewport);
            action::Emit(action::view::SetViewportShading{.Viewport = viewport, .Mode = settings.ViewportShading == ViewportShadingMode::Wireframe ? settings.FillMode : ViewportShadingMode::Wireframe});
        } else if (key(ImGuiMod_Alt | ImGuiKey_Z)) {
            action::Emit(action::view::ToggleXRay{viewport});
        }
        // Tab uses default RouteFocused (not VKey/RouteGlobal) so widget tabbing in panels keeps working.
        const bool tab_no_mods = key(ImGuiKey_Tab, ImGuiInputFlags_None);
        const bool tab_ctrl = ctrl_key(ImGuiMod_Ctrl | ImGuiKey_Tab, ImGuiInputFlags_None);
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
            if (key(ImGuiKey_1)) action::Emit(action::view::SetEditMode{.Mode = Element::Vertex});
            else if (key(ImGuiKey_2)) action::Emit(action::view::SetEditMode{.Mode = Element::Edge});
            else if (key(ImGuiKey_3)) action::Emit(action::view::SetEditMode{.Mode = Element::Face});
        }
        if (key(ImGuiKey_A)) action::Emit(action::selection::SelectAll{});
        if (key(ImGuiMod_Alt | ImGuiKey_A)) action::Emit(action::selection::DeselectAll{});
        const bool bone_edit = interaction_mode == InteractionMode::Edit && FindArmatureObject(r, active_entity) != state::Null;
        const bool mesh_edit = interaction_mode == InteractionMode::Edit && !bone_edit;
        if (mesh_edit) {
            namespace mesh = action::mesh;
            using Op = MeshOperatorDrag::Op;
            using mesh::DissolveMode;
            const auto element = r.get<const EditMode>(state::Document).Value;
            // Placement drags join the gesture, so an extrude, duplicate, or rip and its move commit as one node.
            if (ctrl_key(ImGuiMod_Ctrl | ImGuiKey_X)) {
                action::Emit(mesh::Dissolve{element == Element::Face ? DissolveMode::Faces : element == Element::Edge ? DissolveMode::Edges :
                                                                                                                        DissolveMode::Vertices});
            } else if (key(ImGuiKey_X) || key(ImGuiKey_Delete) || key(ImGuiKey_Backspace)) {
                OpenPopup("##MeshDelete");
            } else if (key(ImGuiKey_E)) {
                action::Emit(mesh::Extrude{element == Element::Edge ? mesh::ExtrudeMode::Edges : mesh::ExtrudeMode::Region}, action::Phase::Stage);
            } else if (key(ImGuiMod_Shift | ImGuiKey_D)) {
                action::Emit(mesh::Duplicate{}, action::Phase::Stage);
            } else if (key(ImGuiKey_H)) {
                action::Emit(mesh::Hide{});
            } else if (key(ImGuiMod_Shift | ImGuiKey_H)) {
                action::Emit(mesh::Hide{true});
            } else if (key(ImGuiMod_Alt | ImGuiKey_H)) {
                action::Emit(mesh::Reveal{});
            } else if (key(ImGuiKey_Y)) {
                action::Emit(mesh::Split{});
            } else if (key(ImGuiKey_P)) {
                action::Emit(mesh::Separate{});
            } else if (key(ImGuiKey_M)) {
                OpenPopup("##MeshMerge");
            } else if (ctrl_key(ImGuiMod_Ctrl | ImGuiKey_T)) {
                action::Emit(mesh::Triangulate{});
            } else if (key(ImGuiMod_Alt | ImGuiKey_J)) {
                action::Emit(mesh::TrisToQuads{});
            } else if (key(ImGuiKey_F)) {
                action::Emit(mesh::Fill{});
            } else if (key(ImGuiKey_V)) {
                action::Emit(mesh::Rip{}, action::Phase::Stage);
            } else if (ctrl_key(ImGuiMod_Ctrl | ImGuiKey_V)) {
                OpenPopup("##MeshVertex");
            } else if (ctrl_key(ImGuiMod_Ctrl | ImGuiKey_E)) {
                OpenPopup("##MeshEdge");
            } else if (ctrl_key(ImGuiMod_Ctrl | ImGuiKey_F)) {
                OpenPopup("##MeshFace");
            } else if (!IsWindowHovered()) {
            } else if (key(ImGuiKey_I)) {
                BeginMeshDrag(r, frame, Op::Inset);
            } else if (ctrl_key(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_B)) {
                BeginMeshDrag(r, frame, Op::BevelVertices);
            } else if (ctrl_key(ImGuiMod_Ctrl | ImGuiKey_B)) {
                BeginMeshDrag(r, frame, Op::BevelEdges);
            } else if (key(ImGuiKey_K)) {
                BeginMeshDrag(r, frame, Op::Knife);
            }
        }
        if (bone_edit) {
            if (key(ImGuiMod_Shift | ImGuiKey_A)) {
                action::Emit(action::bone::Add{});
            } else if (key(ImGuiKey_E)) {
                action::Emit(action::bone::Extrude{}, action::Phase::Stage);
            } else if (key(ImGuiKey_X) || key(ImGuiKey_Delete) || key(ImGuiKey_Backspace)) {
                Delete(r);
            } else if (key(ImGuiMod_Shift | ImGuiKey_D)) {
                Duplicate(r);
            }
        }
        if (ctrl_key(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_E)) {
            action::Emit(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
        } else if (ctrl_key(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_A)) {
            action::Emit(action::object::AddArmature{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
        } else if (ctrl_key(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_C)) {
            action::Emit(action::object::AddCamera{.Info = std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
        } else if (ctrl_key(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_L)) {
            action::Emit(action::object::AddLight{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
        }
        if (!SelectedIn(r).empty()) {
            if (!bone_edit && !mesh_edit && key(ImGuiMod_Shift | ImGuiKey_D)) Duplicate(r);
            else if (!bone_edit && !mesh_edit && key(ImGuiMod_Alt | ImGuiKey_D)) action::Emit(action::object::DuplicateLinked{}, action::Phase::Stage);
            else if (!bone_edit && !mesh_edit && CanDelete(r) && (key(ImGuiKey_Delete) || key(ImGuiKey_Backspace))) Delete(r);
            else if (interaction_mode == InteractionMode::Pose && key(ImGuiMod_Alt | ImGuiKey_G)) action::Emit(action::bone::ClearSelectedTransforms{.Position = true});
            else if (interaction_mode == InteractionMode::Pose && key(ImGuiMod_Alt | ImGuiKey_R)) action::Emit(action::bone::ClearSelectedTransforms{.Rotation = true});
            else if (interaction_mode == InteractionMode::Pose && key(ImGuiMod_Alt | ImGuiKey_S)) action::Emit(action::bone::ClearSelectedTransforms{.Scale = true});
            else if (key(ImGuiKey_G) && flags.Transformable) {
                // Start transform gizmo in both Object and Edit modes.
                // In Edit mode, shader applies transform to selected vertices.
                // In Object mode, shader applies transform to selected instances.
                action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Translate}, action::Phase::Cancel);
            } else if (key(ImGuiKey_R) && flags.Transformable) {
                action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Rotate}, action::Phase::Cancel);
            } else if (key(ImGuiKey_S) && flags.Scalable) {
                action::Emit(action::view::LatchTransform{TransformGizmo::TransformType::Scale}, action::Phase::Cancel);
            } else if (key(ImGuiKey_H)) {
                action::Emit(action::object::ToggleHidden{});
            } else if (ctrl_key(ImGuiMod_Ctrl | ImGuiKey_P)) {
                action::Emit(action::object::ParentToActive{});
            } else if (key(ImGuiMod_Alt | ImGuiKey_P)) {
                action::Emit(action::object::ClearParent{});
            }
        }
    }

    const bool active_transform = r.get<const GizmoInteraction>(state::Document).IsUsing();
    if (active_transform) {
        // TransformGizmo overrides this mouse cursor during some actions, so this is a default.
        SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        WrapMousePos(GetCurrentWindowRead()->InnerClipRect, frame.AccumulatedWrapMouseDelta);
    } else {
        frame.AccumulatedWrapMouseDelta = {0, 0};
    }
    if (active_transform) return; // Only transform gizmo should consume viewport mouse input while active.

    if (!IsWindowHovered() && !frame.BoxDrag) return;

    const auto &io = GetIO();
    if (const vec2 wheel = std::exchange(frame.PreciseWheelDelta, vec2{0}); wheel != vec2{0, 0}) {
        if (io.KeyCtrl || io.KeySuper) EmitViewNav(r, viewport, action::view::ZoomViewCamera{.Viewport = viewport, .Factor = std::pow(WheelZoomStep, -wheel.y)});
        else EmitViewNav(r, viewport, action::view::OrbitViewCamera{.Viewport = viewport, .DeltaRad = wheel * WheelOrbitRadPerUnit});
    }
    if (const float magnification = std::exchange(frame.Magnification, 0.f); magnification != 0.f) {
        EmitViewNav(r, viewport, action::view::ZoomViewCamera{.Viewport = viewport, .Factor = 1.f / (1.f + magnification)});
    }
    if (OrientationGizmo::IsActive() || frame.OverlayControlsHovered) return;

    const auto render_extent = RenderExtentPx(r, viewport);
    // Camera advancement follows input and action execution.
    const auto selection_view = r.get<const Viewport>(viewport).FrameView;
    const auto edit_mode = r.get<const EditMode>(state::Document).Value;
    const auto arm_obj_entity = FindArmatureObject(r, active_entity);
    const bool active_is_armature = arm_obj_entity != state::Null;
    const bool bone_mode = interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && active_is_armature);
    if (r.get<const BoxSelectState>(state::Document).Gesture == SelectionGesture::Box && interaction_mode != InteractionMode::Excite) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            frame.BoxDrag = BoxSelectDrag{ToVec2(GetMousePos()), ToVec2(GetMousePos()), IsKeyDown(ImGuiMod_Shift), {}};
        } else if (IsMouseDown(ImGuiMouseButton_Left) && frame.BoxDrag) {
            frame.BoxDrag->End = ToVec2(GetMousePos());
            // A box or view the drag already resolved selects nothing new.
            if (const auto box = ComputeBoxSelect(frame.BoxDrag->Start, frame.BoxDrag->End, ToVec2(GetCursorScreenPos()), logical_extent);
                box && frame.BoxDrag->Emitted != std::pair{*box, selection_view}) {
                frame.BoxDrag->Emitted = {*box, selection_view};
                // The hit set (object/bone instances or edit-mode elements) is resolved later.
                action::Emit(action::selection::ApplyBoxSelect{.Box = *box, .Additive = frame.BoxDrag->Additive}, action::Phase::Stage);
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left) && frame.BoxDrag) {
            frame.BoxDrag.reset();
            action::Commit();
        }
        if (frame.BoxDrag) return;
    }

    const auto mouse_pos_render = ToRenderPx(r, ToVec2(GetMousePos()));
    const float max_x = float(std::max(render_extent.x, 1u) - 1u);
    const float max_y = float(std::max(render_extent.y, 1u) - 1u);
    // ImGui's origin and the picking pass's pixel rows both start at the top left.
    const uvec2 mouse_px{Clamp(mouse_pos_render.x, 0.0f, max_x), Clamp(mouse_pos_render.y, 0.0f, max_y)};
    const auto mouse = ToViewFraction(r, ToVec2(GetMousePos()));
    const auto pick_px = SelectionPixel(selection_view.Extent, mouse);

    if (interaction_mode == InteractionMode::Excite) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            if (const auto hit_entities = RunObjectPick(r, viewport, mouse_px); !hit_entities.empty()) {
                if (const auto hit_entity = hit_entities.front(); r.all_of<SoundVertices>(hit_entity)) {
                    if (const auto vertex = RunSoundVerticesVertexPick(r, viewport, hit_entity, mouse_px)) {
                        action::Emit(action::audio::ApplyExciteImpact{.InstanceEntity = hit_entity, .VertexIndex = *vertex});
                    }
                }
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left)) {
            action::Emit(action::audio::ClearExciteImpacts{});
        }
        return;
    }
    if (interaction_mode == InteractionMode::Edit && !active_is_armature && ctrl_key(ImGuiMod_Ctrl | ImGuiKey_R)) {
        std::vector<action::Action> actions;
        if (edit_mode != Element::Edge) actions.push_back(action::MakeAction(action::view::SetEditMode{.Mode = Element::Edge}));
        actions.push_back(action::MakeAction(action::selection::ApplyEditElementClick{false, RunEditElementPick(r, viewport, Element::Edge, pick_px)}));
        actions.push_back(action::MakeAction(action::mesh::LoopCut{}));
        action::Emit(std::move(actions));
        return;
    }
    if (interaction_mode == InteractionMode::Edit && !active_is_armature && IsMouseClicked(ImGuiMouseButton_Right)) OpenPopup("##MeshContext");
    if (!IsSingleClicked(ImGuiMouseButton_Left)) return;
    if (interaction_mode == InteractionMode::Edit && edit_mode == Element::None && !active_is_armature) return;

    if (interaction_mode == InteractionMode::Edit && !active_is_armature) {
        const bool toggle = IsKeyDown(ImGuiMod_Shift) || IsKeyDown(ImGuiMod_Ctrl) || IsKeyDown(ImGuiMod_Super);
        action::Emit(action::selection::ApplyEditElementClick{toggle, RunEditElementPick(r, viewport, edit_mode, pick_px)});
    } else if (interaction_mode == InteractionMode::Object || bone_mode) {
        const bool shift = IsKeyDown(ImGuiMod_Shift);
        action::Emit(action::selection::Pick{shift, PickObject(r, viewport, pick_px, ImLengthSqr(CurrentClickPos - PrevClickPos) <= 16)});
    }
}

void InteractOverlay(state::Registry &r, FrameState &frame, state::Entity viewport, bool active) {
    const profile::CpuScope scope{"ViewportOverlayUi"};
    const auto &icons = r.Resources.get<const ViewportIcons>();
    const rect viewport_rect{ToVec2(GetWindowPos()), ToVec2(GetContentRegionAvail())};
    const bool active_transform = r.get<const GizmoInteraction>(state::Document).IsUsing();
    static constexpr float OrientationGizmoSize{84};
    const auto &interface_theme = r.get<const InterfaceTheme>(state::Document);
    const float rounding = interface_theme.IconButtonRounding;
    const auto tool_button_size = std::bit_cast<ImVec2>(interface_theme.ToolButtonSize), header_button_size = std::bit_cast<ImVec2>(interface_theme.HeaderButtonSize);
    const float overlay_corner_gap = GetTextLineHeightWithSpacing() / 2.f;
    // Preserve the guard through release because IsSingleClicked fires on release.
    // Only the active viewport's controls take input, so they alone set the guard.
    if (active && !IsMouseDown(ImGuiMouseButton_Left)) frame.OverlayControlsHovered = false;
    const bool any_popup_open = IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);

    // Overlay buttons keep the cursor where the viewport content left it, and take no input during a transform.
    const auto saved_cursor = GetCursorScreenPos();
    PushItemFlag(ImGuiItemFlags_Disabled, active_transform);
    { // Transform mode pill buttons (top-left overlay)
        using enum TransformGizmo::Type;
        using Tool = action::view::SetActiveTool::Tool;
        const auto &flags = r.get<const SelectionFlags>(state::Document);
        const auto transform_type = EffectiveGizmoType(r);
        const auto gesture = r.get<const BoxSelectState>(state::Document).Gesture;
        static constexpr float gap{4}; // Gap between select buttons and transform buttons

        SetCursorScreenPos(std::bit_cast<ImVec2>(viewport_rect.pos) + GetWindowContentRegionMin() + ImVec2{overlay_corner_gap, overlay_corner_gap});
        ui::BeginIconButtonGroup(ImGuiAxis_Y, tool_button_size, 2, rounding);
        if (ui::IconButton("##SelectBox", icons.Transform.SelectBox.get(), transform_type == None && gesture == SelectionGesture::Box)) action::Emit(action::view::SetActiveTool{Tool::SelectBox});
        if (ui::IconButton("##SelectClick", icons.Transform.Select.get(), transform_type == None && gesture == SelectionGesture::Click)) action::Emit(action::view::SetActiveTool{Tool::SelectClick});
        ui::EndIconButtonGroup();
        if (IsItemHovered()) frame.OverlayControlsHovered = true;

        SetCursorScreenPos({GetItemRectMin().x, GetItemRectMax().y + gap});
        ui::BeginIconButtonGroup(ImGuiAxis_Y, tool_button_size, 4, rounding);
        if (ui::IconButton("##Translate", icons.Transform.Move.get(), transform_type == Translate, flags.Transformable)) action::Emit(action::view::SetActiveTool{Tool::Translate});
        if (ui::IconButton("##Rotate", icons.Transform.Rotate.get(), transform_type == Rotate, flags.Transformable)) action::Emit(action::view::SetActiveTool{Tool::Rotate});
        if (ui::IconButton("##Scale", icons.Transform.Scale.get(), transform_type == Scale, flags.Scalable)) action::Emit(action::view::SetActiveTool{Tool::Scale});
        if (ui::IconButton("##Universal", icons.Transform.Universal.get(), transform_type == Universal, flags.Transformable)) action::Emit(action::view::SetActiveTool{Tool::Universal});
        ui::EndIconButtonGroup();
        if (IsItemHovered()) frame.OverlayControlsHovered = true;
    }

    const auto &settings = r.get<const ViewportDisplay>(viewport);

    const float shading_arrow_w = header_button_size.y * 0.55f;
    const float shading_group_width = header_button_size.x * 4.f + shading_arrow_w + ui::IconButtonSeam * 4.f;
    const auto buttons_gap = 6.f;
    const auto shading_start = std::bit_cast<ImVec2>(viewport_rect.pos + vec2{GetWindowContentRegionMax().x - shading_group_width, GetWindowContentRegionMin().y}) + ImVec2{-overlay_corner_gap, overlay_corner_gap};

    { // Viewport shading button group + dropdown (top-right overlay)
        const auto shading_button = [&](const char *id, const SvgResource *icon, ViewportShadingMode mode) {
            if (ui::IconButton(id, icon, settings.ViewportShading == mode)) action::Emit(action::view::SetViewportShading{.Viewport = viewport, .Mode = mode});
        };
        SetCursorScreenPos(shading_start);
        ui::BeginIconButtonGroup(ImGuiAxis_X, header_button_size, 5, rounding);
        shading_button("##Wireframe", icons.Shading.Wireframe.get(), ViewportShadingMode::Wireframe);
        shading_button("##Solid", icons.Shading.Solid.get(), ViewportShadingMode::Solid);
        shading_button("##MaterialPreview", icons.Shading.MaterialPreview.get(), ViewportShadingMode::MaterialPreview);
        shading_button("##Rendered", icons.Shading.Rendered.get(), ViewportShadingMode::Rendered);
        if (ui::IconButton("##ShadingArrow", nullptr, IsPopupOpen("##ShadingDropdown"), true, shading_arrow_w)) OpenPopup("##ShadingDropdown");
        ui::DropdownArrow();
        ui::EndIconButtonGroup();
        if (IsItemHovered()) frame.OverlayControlsHovered = true;
        { // Dropdown popup
            SetNextWindowPos(shading_start + ImVec2{shading_group_width, header_button_size.y + 2.f}, ImGuiCond_Always, {1.f, 0.f});
            PushStyleVar(ImGuiStyleVar_WindowPadding, {8, 8});
            if (BeginPopup("##ShadingDropdown")) {
                frame.OverlayControlsHovered = true;
                TextUnformatted("Viewport shading");
                Separator();
                const auto current_mode = settings.ViewportShading;

                const auto render_pbr_controls = [&]<typename T>(const char *id) {
                    PushID(id);
                    const auto &lighting = r.get<const T>(viewport).Value;
                    if (Button("Reset")) action::Emit(action::view::ResetPbrLighting{.Viewport = viewport, .Rendered = std::same_as<T, RenderedLighting>});
                    ui::DrawFields<PBRViewportLighting>(ui::Edit{r, viewport}.template Sub<&T::Value>());
                    if (lighting.UseSceneWorld) {
                        if (const auto scene = FrameScene(r); r.all_of<ImageLight>(scene)) ui::Edit{r, scene}.Draw<&ImageLight::Intensity>();
                    } else {
                        const auto hdris = GetHdriRefs(r, r.get<const StudioEnvironment>(viewport).Name);
                        if (BeginCombo("Environment", hdris.Names[hdris.ActiveIndex].c_str())) {
                            for (uint32_t i = 0; i < hdris.Names.size(); ++i) {
                                const bool selected = (i == hdris.ActiveIndex);
                                if (Selectable(hdris.Names[i].c_str(), selected)) action::Emit(action::view::SetStudioEnvironment{viewport, hdris.Names[i]});
                                if (selected) SetItemDefaultFocus();
                            }
                            EndCombo();
                        }
                    }
                    PopID();
                };

                if (WorkbenchShading(current_mode)) {
                    const bool wireframe = current_mode == ViewportShadingMode::Wireframe;
                    ui::Edit edit{r, viewport};
                    if (!wireframe) edit.Draw<&ViewportDisplay::BackfaceCulling>();
                    if (wireframe) edit.Draw<&ViewportDisplay::XRayWireframe>("##XRay");
                    else edit.Draw<&ViewportDisplay::XRaySolid>("##XRay");
                    SameLine();
                    PushStyleVar(ImGuiStyleVar_Alpha, GetStyle().Alpha * (XRayFlag(settings) ? 1.f : 0.5f));
                    if (wireframe) edit.Draw<&ViewportDisplay::XRayAlphaWireframe>("X-ray");
                    else edit.Draw<&ViewportDisplay::XRayAlpha>("X-ray");
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
                    // Light colors are stored in linear space.
                    // Display/edit as sRGB.
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
                    if (changed) action::Emit(action::view::SetWorkspaceLights{viewport, std::make_unique<WorkspaceLights>(lights)});
                }

                if (!WorkbenchShading(current_mode)) {
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
                                    action::Emit(action::UpdateOf<&ViewportDisplay::DebugChannel>(viewport, entry.Value));
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
        const bool applies = WorkbenchShading(settings.ViewportShading) || r.get<const Interaction>(state::Document).Mode == InteractionMode::Edit;
        SetCursorScreenPos(shading_start - ImVec2{buttons_gap + header_button_size.x, 0.f});
        ui::BeginIconButtonGroup(ImGuiAxis_X, header_button_size, 1, rounding);
        if (!applies) PushStyleVar(ImGuiStyleVar_Alpha, GetStyle().Alpha * 0.5f);
        if (ui::IconButton("##XRay", icons.XRay.get(), XRayFlag(settings))) action::Emit(action::view::ToggleXRay{viewport});
        if (!applies) PopStyleVar();
        ui::EndIconButtonGroup();
        if (IsItemHovered()) frame.OverlayControlsHovered = true;
    }

    { // Viewport overlays toggle + dropdown
        const float overlay_group_width = header_button_size.x + ui::IconButtonSeam + shading_arrow_w;
        const auto group_start = shading_start - ImVec2{2.f * buttons_gap + header_button_size.x + overlay_group_width, 0.f};
        SetCursorScreenPos(group_start);
        ui::BeginIconButtonGroup(ImGuiAxis_X, header_button_size, 2, rounding);
        if (ui::IconButton("##Overlays", icons.Overlay.get(), settings.ShowOverlays)) action::Emit(action::UpdateOf<&ViewportDisplay::ShowOverlays>(viewport, !settings.ShowOverlays));
        if (ui::IconButton("##OverlayArrow", nullptr, IsPopupOpen("##OverlayDropdown"), true, shading_arrow_w)) OpenPopup("##OverlayDropdown");
        ui::DropdownArrow();
        ui::EndIconButtonGroup();
        if (IsItemHovered()) frame.OverlayControlsHovered = true;
        { // Dropdown popup
            SetNextWindowPos(group_start + ImVec2{0.f, header_button_size.y + 2.f});
            PushStyleVar(ImGuiStyleVar_WindowPadding, {8, 8});
            if (BeginPopup("##OverlayDropdown")) {
                frame.OverlayControlsHovered = true;
                TextUnformatted("Viewport overlays");
                Separator();
                ui::DrawFields<ViewportOverlays>(ui::Edit{r, viewport}.Sub<&ViewportDisplay::Overlays>());
                EndPopup();
            }
            PopStyleVar();
        }
    }

    PopItemFlag();
    SetCursorScreenPos(saved_cursor);

    const auto &camera = r.get<const ViewCameraState>(viewport).Active;
    { // Orientation gizmo
        const auto pos = viewport_rect.pos + vec2{GetWindowContentRegionMax().x - OrientationGizmoSize, GetWindowContentRegionMin().y} + vec2{-overlay_corner_gap, overlay_corner_gap * 2 + header_button_size.y};
        const auto axes = colors::MakeAxes(r.get<const ViewportTheme>(state::Document).AxisColors);
        if (active) {
            const auto interaction = OrientationGizmo::Interact(pos, OrientationGizmoSize, camera, !active_transform && !any_popup_open);
            const auto &ctx = OrientationGizmo::Ctx;
            OrientationGizmo::Render(ctx.View, axes, ctx.Hovered || ctx.DragEndPos, ctx.HoveredAxis);
            if (interaction) {
                std::visit(
                    overloaded{
                        [&](OrientationGizmo::RotateBy rot) { EmitViewNav(r, viewport, action::view::OrbitViewCamera{viewport, rot.Delta}); },
                        [&](OrientationGizmo::AlignTo a) { EmitViewNav(r, viewport, action::view::SetViewCameraTargetDirection{viewport, a.Direction}); },
                    },
                    *interaction
                );
            }
        } else {
            OrientationGizmo::Render(OrientationGizmo::LayoutOf(pos, OrientationGizmoSize, camera), axes);
        }
    }
    // The transform gizmo's input and the mesh operator menus belong to the gesture in the active viewport.
    if (!active) return;
    const auto mode_scope = ScopeOf(r);

    r.edit<GizmoInteraction>(state::Document).RenderTransform.reset();
    if (!DrawOperationHandle(r, frame, camera, viewport_rect) && r.get<const SelectionFlags>(state::Document).HasTransformTarget) { // Transform gizmo
        // Transform root selections around their average position using the active entity's rotation and scale.
        const auto gizmo_active_entity = mode_scope.Bone ? FindActiveBone(r) : FindActiveEntity(r);
        const auto active_scale = gizmo_active_entity == state::Null ? vec3{1} : WorldTransformOf(r, gizmo_active_entity)->S;
        // The edit-mode pivot is the drag's start pivot moved by the pending delta.
        const auto *start_pivot = r.try_get<const StartPivot>(state::Document);
        auto pivot = start_pivot ? *start_pivot : TransformPivot(r);
        if (mode_scope.MeshEdit) {
            if (const auto *pending = r.try_get<const PendingTransform>(state::Document)) pivot.P += pending->Delta.P;
        }

        const auto start_transform_view = r.view<const StartTransform>();
        const auto &gizmo_state = r.get<const TransformGizmoState>(state::Document);
        auto gizmo_config = gizmo_state.Config;
        gizmo_config.Type = EffectiveGizmoType(r);
        auto &gizmo = r.edit<GizmoInteraction>(state::Document);
        const auto gizmo_transform = GizmoTransform{{.P = pivot.P, .R = pivot.R, .S = active_scale}, gizmo_state.Mode};
        const auto *start_screen = r.try_get<const StartScreenTransform>(state::Document);
        const bool was_using = gizmo.IsUsing();
        auto interact_result = TransformGizmo::Interact(
            gizmo,
            gizmo_transform,
            gizmo_config, camera, viewport_rect, ToVec2(GetMousePos()) + frame.AccumulatedWrapMouseDelta,
            start_screen ? std::optional{start_screen->Value} : std::nullopt
        );
        if (gizmo.Cancelled) {
            // Escape discards the whole gesture, including an operator whose placement this drag was.
            gizmo.Cancelled = false;
            action::Cancel();
        } else if (interact_result) {
            const auto &td = interact_result->Delta;
            if (mode_scope.MeshEdit) {
                // Each update emits the delta, which a transform node on each transformable mesh applies.
                action::Emit(action::view::TransformElements{td}, action::Phase::Stage);
            } else {
                // Store the gizmo pivot and delta for object and bone transforms.
                // Apply recomputes each entity transform.
                action::Emit(action::view::TransformSelection{td}, action::Phase::Stage);
            }
        } else if (was_using || !start_transform_view.empty()) {
            action::Commit();
        }

        gizmo.RenderTransform = gizmo_transform;
        if (interact_result) gizmo.RenderTransform->P = interact_result->Start.P + interact_result->Delta.P;
    }

    if (mode_scope.MeshEdit) {
        DrawMeshOperatorMenus(r);
    }

    // The latch is a handoff from the operator's Apply to this overlay, consumed here rather than through an action.
    r.remove<StartScreenTransform>(state::Document);
}

void DrawOverlay(state::Registry &r, FrameState &frame, state::Entity viewport, bool active) {
    const rect viewport_rect{ToVec2(GetWindowPos()), ToVec2(GetContentRegionAvail())};
    const auto &camera = r.get<const ViewCameraState>(viewport).Active;

    { // The transform gizmo, highlighted and with its guides in the active viewport whose mouse it follows.
        const auto axes = colors::MakeAxes(r.get<const ViewportTheme>(state::Document).AxisColors);
        const auto &gizmo = r.get<const GizmoInteraction>(state::Document);
        const auto type = EffectiveGizmoType(r);
        if (active) {
            TransformGizmo::Render(gizmo, type, camera, viewport_rect, axes);
        } else {
            auto idle = gizmo;
            idle.Current.reset();
            idle.Start.reset();
            TransformGizmo::Render(idle, type, camera, viewport_rect, axes);
        }
    }
    if (active) { // The guides of the gesture in progress
        // Inset and bevel size by the mouse's distance from the selection center, so they show the same guide as scale and rotate.
        if (frame.MeshDrag && frame.MeshDrag->Value != MeshOperatorDrag::Op::Knife) DrawDashedGuide(*GetWindowDrawList(), std::bit_cast<ImVec2>(frame.MeshDrag->CenterPx), GetMousePos());

        if (frame.BoxDrag) {
            auto &dl = *GetWindowDrawList();
            const auto box_min = Min(frame.BoxDrag->Start, frame.BoxDrag->End);
            const auto box_max = Max(frame.BoxDrag->Start, frame.BoxDrag->End);
            dl.AddRectFilled(std::bit_cast<ImVec2>(box_min), std::bit_cast<ImVec2>(box_max), IM_COL32(255, 255, 255, 30));

            static constexpr auto outline_color{IM_COL32(255, 255, 255, 200)};
            DrawDashedLine(dl, {box_min.x, box_min.y}, {box_max.x, box_min.y}, outline_color);
            DrawDashedLine(dl, {box_min.x, box_max.y}, {box_max.x, box_max.y}, outline_color);
            DrawDashedLine(dl, {box_min.x, box_min.y}, {box_min.x, box_max.y}, outline_color);
            DrawDashedLine(dl, {box_max.x, box_min.y}, {box_max.x, box_max.y}, outline_color);
        }
    }

    // Match the centered frame to the captured look-through region.
    if (const auto look_through_entity = LookThroughCamera(r, viewport); look_through_entity != state::Null && !camera.IsAnimating()) {
        if (const auto cd = LensOf(r, look_through_entity)) {
            const float cam_aspect = AspectRatio(*cd);
            const auto frame_size = vec2{viewport_rect.size.y * cam_aspect, viewport_rect.size.y} * LookThroughFrameRatio(cam_aspect, viewport_rect.size.x / viewport_rect.size.y);
            const vec2 vp_center = viewport_rect.pos + viewport_rect.size * 0.5f;
            const vec2 fmin = vp_center - frame_size * 0.5f, fmax = vp_center + frame_size * 0.5f;
            const auto vmin = viewport_rect.pos, vmax = viewport_rect.pos + viewport_rect.size;

            auto &dl = *GetWindowDrawList();
            static constexpr auto dim = IM_COL32(0, 0, 0, 100);
            const auto iv = [](vec2 v) { return std::bit_cast<ImVec2>(v); };
            dl.AddRectFilled(iv(vmin), iv({vmax.x, fmin.y}), dim);
            dl.AddRectFilled(iv({vmin.x, fmax.y}), iv(vmax), dim);
            dl.AddRectFilled(iv({vmin.x, fmin.y}), iv({fmin.x, fmax.y}), dim);
            dl.AddRectFilled(iv({fmax.x, fmin.y}), iv({vmax.x, fmax.y}), dim);
        }
    }
}
