#include "viewport/ViewportUi.h"
#include "Camera.h"
#include "Timer.h"
#include "action/Audio.h"
#include "action/Bone.h"
#include "action/Object.h"
#include "action/Selection.h"
#include "action/Timeline.h"
#include "action/View.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "gizmo/TransformGizmo.h"
#include "gltf/SourceAssets.h"
#include "mesh/MeshStore.h"
#include "render/Instance.h"
#include "render/TextureRefs.h"
#include "scene/Defaults.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionQueries.h"
#include "ui/FieldEdit.h"
#include "viewport/FrameState.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewportIcons.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"

#include <entt/entity/registry.hpp>
#include <imgui_internal.h>

#include "gizmo/OrientationGizmo.h"

using std::ranges::any_of, std::ranges::fold_left;
using std::views::transform;

using namespace ImGui;

namespace {
constexpr vec2 ToGlm(ImVec2 v) { return std::bit_cast<vec2>(v); }
constexpr float WheelOrbitRadPerUnit{0.05f}, WheelZoomStep{1.04f};

std::optional<std::pair<uvec2, uvec2>> ComputeBoxSelectPixels(vec2 start, vec2 end, vec2 window_pos, uvec2 logical_extent, uvec2 render_extent) {
    static constexpr float DragThresholdSq{2 * 2};
    if (glm::distance2(start, end) <= DragThresholdSq) return {};

    const vec2 logical_size{float(logical_extent.x), float(logical_extent.y)};
    const vec2 render_scale{
        logical_extent.x > 0u ? float(render_extent.x) / float(logical_extent.x) : 1.f,
        logical_extent.y > 0u ? float(render_extent.y) / float(logical_extent.y) : 1.f
    };
    const auto box_min = glm::min(start, end) - window_pos;
    const auto box_max = glm::max(start, end) - window_pos;
    const auto local_min = glm::clamp(glm::min(box_min, box_max), vec2{0}, logical_size);
    const auto local_max = glm::clamp(glm::max(box_min, box_max), vec2{0}, logical_size);
    const auto render_min = local_min * render_scale;
    const auto render_max = local_max * render_scale;
    const uvec2 box_min_px{glm::floor(render_min.x), glm::floor(float(render_extent.y) - render_max.y)};
    const uvec2 box_max_px{glm::ceil(render_max.x), glm::ceil(float(render_extent.y) - render_min.y)};
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
        accumulated_wrap_mouse_delta -= ToGlm(mouse_delta);
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

struct OverlayIconButtonInfo {
    const SvgResource *Icon;
    ImVec2 Offset;
    ImDrawFlags Corners;
    bool Enabled{true};
    bool Active{false};
    const char *Tooltip{nullptr};
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
            // todo - better tooltips and add them to all viewport buttons.
            //  - anchor, don't follow mouse
            //  - padding, border
            // if (hovered && button.Tooltip) SetTooltip("%s", button.Tooltip);
            PopID();
            if (!button.Enabled) EndDisabled();
        }
        if (any_hovered && hovered) *any_hovered = true;

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
    }
    PopID();
    SetCursorScreenPos(saved_cursor_pos);
    return clicked_index;
}

} // namespace

void Interact(entt::registry &r, entt::entity viewport, FrameState &frame) {
    // Any open popup (e.g. Viewport shading dropdown) blocks viewport mouse/keyboard input.
    // Without this, wheel/click events still patch the camera while the popup overlays the viewport_rect.
    if (IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
        frame.PreciseWheelDelta = {0, 0};
        return;
    }

    // Track the previous click position for pick-cycle gating.
    static ImVec2 PrevClickPos{-FLT_MAX, -FLT_MAX}, CurrentClickPos{-FLT_MAX, -FLT_MAX};
    if (GetIO().MouseClicked[0]) {
        PrevClickPos = CurrentClickPos;
        CurrentClickPos = GetIO().MouseClickedPos[0];
    }

    const auto logical_extent = r.get<const ViewportExtent>(viewport).Value;
    const auto render_extent = RenderExtentPx(logical_extent);
    if (logical_extent.x == 0 || logical_extent.y == 0 || render_extent.x == 0 || render_extent.y == 0) return;

    const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
    const auto active_entity = FindActiveEntity(r);
    const bool has_frozen_selected = r.view<Selected, ScaleLocked>().begin() != r.view<Selected, ScaleLocked>().end();
    const bool edit_transform_locked = interaction_mode == InteractionMode::Edit &&
        any_of(selection::GetSelectedMeshEntities(r), [&](entt::entity mesh_entity) { return selection::HasScaleLockedInstance(r, mesh_entity); });
    const bool transform_shortcuts_enabled = !edit_transform_locked;
    const bool scale_shortcut_enabled = transform_shortcuts_enabled && !has_frozen_selected;
    // Keyboard shortcuts use ImGui's Shortcut() routing system with RouteGlobal so they fire from any
    // focused window in the dockspace. RouteGlobal yields to active items (sliders mid-drag, focused
    // InputText, etc.) and ImGui's Nav (Tab/arrows) via key-ownership, so widget editing and tree/list
    // navigation in panels keep working. char-input keys (G/A/etc.) are auto-filtered while WantTextInput.
    constexpr auto VKey = ImGuiInputFlags_RouteGlobal;
    if (TransformGizmo::IsUsing(r, viewport)) {
        // During an active transform, only allow transform switching shortcuts.
        if (Shortcut(ImGuiKey_G, VKey) && transform_shortcuts_enabled) action::Emit(action::view::SetStartScreenTransform{TransformGizmo::TransformType::Translate});
        else if (Shortcut(ImGuiKey_R, VKey) && transform_shortcuts_enabled) action::Emit(action::view::SetStartScreenTransform{TransformGizmo::TransformType::Rotate});
        else if (Shortcut(ImGuiKey_S, VKey) && scale_shortcut_enabled) action::Emit(action::view::SetStartScreenTransform{TransformGizmo::TransformType::Scale});
    } else {
        if (Shortcut(ImGuiKey_Space, VKey)) action::Emit(action::timeline::TogglePlay{});
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
            r.replace<SelectionXRay>(viewport, !r.get<const SelectionXRay>(viewport).Value);
        }
        // Tab uses default RouteFocused (not VKey/RouteGlobal) so widget tabbing in panels keeps working.
        const bool tab_no_mods = Shortcut(ImGuiKey_Tab);
        const bool tab_ctrl = Shortcut(ImGuiMod_Ctrl | ImGuiKey_Tab);
        if (tab_no_mods || tab_ctrl) {
            const bool is_armature = FindArmatureObject(r, active_entity) != entt::null;
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
        const bool bone_edit = interaction_mode == InteractionMode::Edit && FindArmatureObject(r, active_entity) != entt::null;
        if (bone_edit) {
            if (Shortcut(ImGuiMod_Shift | ImGuiKey_A, VKey)) {
                action::Emit(action::bone::Add{});
            } else if (Shortcut(ImGuiKey_E, VKey)) {
                action::Emit(action::bone::Extrude{});
            } else if (Shortcut(ImGuiKey_X, VKey) || Shortcut(ImGuiKey_Delete, VKey) || Shortcut(ImGuiKey_Backspace, VKey)) {
                Delete(r, viewport);
            } else if (Shortcut(ImGuiMod_Shift | ImGuiKey_D, VKey)) {
                Duplicate(r, viewport);
            }
        }
        if (Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_E, VKey)) {
            action::Emit(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
        } else if (Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_A, VKey)) {
            action::Emit(action::object::AddArmature{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
        } else if (Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_C, VKey)) {
            action::Emit(action::object::AddCamera{.Info = std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive}), .Props = {}});
        } else if (Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_L, VKey)) {
            action::Emit(action::object::AddLight{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
        }
        if (!r.storage<Selected>().empty()) {
            if (!bone_edit && Shortcut(ImGuiMod_Shift | ImGuiKey_D, VKey)) Duplicate(r, viewport);
            else if (!bone_edit && Shortcut(ImGuiMod_Alt | ImGuiKey_D, VKey)) action::Emit(action::object::DuplicateLinked{});
            else if (!bone_edit && CanDelete(r, viewport) && (Shortcut(ImGuiKey_Delete, VKey) || Shortcut(ImGuiKey_Backspace, VKey))) Delete(r, viewport);
            else if (interaction_mode == InteractionMode::Pose && Shortcut(ImGuiMod_Alt | ImGuiKey_G, VKey)) action::Emit(action::bone::ClearSelectedTransforms{.Position = true});
            else if (interaction_mode == InteractionMode::Pose && Shortcut(ImGuiMod_Alt | ImGuiKey_R, VKey)) action::Emit(action::bone::ClearSelectedTransforms{.Rotation = true});
            else if (interaction_mode == InteractionMode::Pose && Shortcut(ImGuiMod_Alt | ImGuiKey_S, VKey)) action::Emit(action::bone::ClearSelectedTransforms{.Scale = true});
            else if (Shortcut(ImGuiKey_G, VKey) && transform_shortcuts_enabled) {
                // Start transform gizmo in both Object and Edit modes.
                // In Edit mode, shader applies transform to selected vertices.
                // In Object mode, shader applies transform to selected instances.
                action::Emit(action::view::SetStartScreenTransform{TransformGizmo::TransformType::Translate});
            } else if (Shortcut(ImGuiKey_R, VKey) && transform_shortcuts_enabled) action::Emit(action::view::SetStartScreenTransform{TransformGizmo::TransformType::Rotate});
            else if (Shortcut(ImGuiKey_S, VKey) && scale_shortcut_enabled) action::Emit(action::view::SetStartScreenTransform{TransformGizmo::TransformType::Scale});
            else if (Shortcut(ImGuiKey_H, VKey)) action::Emit(action::object::ToggleHidden{});
            else if (Shortcut(ImGuiMod_Ctrl | ImGuiKey_P, VKey)) action::Emit(action::object::ParentToActive{});
            else if (Shortcut(ImGuiMod_Alt | ImGuiKey_P, VKey)) action::Emit(action::object::ClearParent{});
        }
    }

    // Handle mouse input.
    const bool active_transform = TransformGizmo::IsUsing(r, viewport);
    if (active_transform) {
        // TransformGizmo overrides this mouse cursor during some actions - this is a default.
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
        if (io.KeyCtrl || io.KeySuper) action::Emit(action::view::ZoomViewCamera{.Factor = std::pow(WheelZoomStep, -wheel.y)});
        else action::Emit(action::view::OrbitViewCamera{.DeltaRad = wheel * WheelOrbitRadPerUnit});
    }
    if (OrientationGizmo::IsActive() || frame.OverlayControlsHovered) return;

    const auto edit_mode = r.get<const EditMode>(viewport).Value;
    const auto arm_obj_entity = FindArmatureObject(r, active_entity);
    const bool active_is_armature = arm_obj_entity != entt::null;
    const bool bone_mode = interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && active_is_armature);
    if (r.get<const BoxSelectState>(viewport).Gesture == SelectionGesture::Box && interaction_mode != InteractionMode::Excite) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            frame.BoxSelectStart = frame.BoxSelectEnd = ToGlm(GetMousePos());
            if (IsKeyDown(ImGuiMod_Shift)) action::Emit(action::selection::SnapshotBoxSelectBaseline{});
        } else if (IsMouseDown(ImGuiMouseButton_Left) && frame.BoxSelectStart) {
            frame.BoxSelectEnd = ToGlm(GetMousePos());
            if (const auto box_px = ComputeBoxSelectPixels(*frame.BoxSelectStart, *frame.BoxSelectEnd, ToGlm(GetCursorScreenPos()), logical_extent, render_extent); box_px) {
                const bool is_additive = r.all_of<AdditiveBoxSelectBaseline>(viewport);
                if (interaction_mode == InteractionMode::Edit && !active_is_armature) {
                    Timer timer{"BoxSelectElements (all)"};
                    RunBoxSelectElements(r, viewport, GetBitsetRangesForSelected(r), edit_mode, *box_px, is_additive);
                } else {
                    // Object/bone box-select: the hit set is resolved against current scene state when applied.
                    action::Emit(action::selection::ApplyBoxSelect{.BoxPx = *box_px, .Additive = is_additive});
                }
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left) && frame.BoxSelectStart) {
            const bool was_drag = IsMouseDragPastThreshold(ImGuiMouseButton_Left);
            frame.BoxSelectStart.reset();
            frame.BoxSelectEnd.reset();
            if (was_drag) action::Emit(action::selection::ClearBoxSelectBaseline{});
        }
        if (frame.BoxSelectStart) return;
    }

    const vec2 render_scale{
        logical_extent.x > 0u ? float(render_extent.x) / float(logical_extent.x) : 1.0f,
        logical_extent.y > 0u ? float(render_extent.y) / float(logical_extent.y) : 1.0f
    };
    const auto mouse_pos_rel = GetMousePos() - GetCursorScreenPos();
    const auto mouse_pos_render = ToGlm(mouse_pos_rel) * render_scale;
    const float max_x = float(std::max(render_extent.x, 1u) - 1u);
    const float max_y = float(std::max(render_extent.y, 1u) - 1u);
    // Flip y-coordinate: ImGui uses top-left origin, but Vulkan gl_FragCoord uses bottom-left origin
    const uvec2 mouse_px{glm::clamp(mouse_pos_render.x, 0.0f, max_x), glm::clamp(float(render_extent.y) - mouse_pos_render.y, 0.0f, max_y)};

    if (interaction_mode == InteractionMode::Excite) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            if (const auto hit_entities = RunObjectPick(r, viewport, frame.ObjectPickEpochTag, mouse_px); !hit_entities.empty()) {
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
    if (!IsSingleClicked(ImGuiMouseButton_Left)) return;
    if (interaction_mode == InteractionMode::Edit && edit_mode == Element::None && !active_is_armature) return;

    if (interaction_mode == InteractionMode::Edit && !active_is_armature) {
        const bool toggle = IsKeyDown(ImGuiMod_Shift) || IsKeyDown(ImGuiMod_Ctrl) || IsKeyDown(ImGuiMod_Super);
        action::Emit(action::selection::ApplyEditElementClick{.MousePx = mouse_px, .Toggle = toggle});
    } else if (interaction_mode == InteractionMode::Object || bone_mode) {
        const bool shift = IsKeyDown(ImGuiMod_Shift);
        // Store only the pixel; the GPU pick + selection resolution run in ProcessComponentEvents.
        // A re-click at the same spot cycles to the next overlapping hit.
        if (ImLengthSqr(CurrentClickPos - PrevClickPos) > 16) action::Emit(action::selection::Pick{mouse_px, shift});
        else action::Emit(action::selection::PickCycle{mouse_px, shift});
    }
}

void InteractOverlay(entt::registry &r, entt::entity viewport, FrameState &frame) {
    auto &meshes = r.ctx().get<MeshStore>();
    const auto &icons = r.ctx().get<const ViewportIcons>();
    const rect viewport_rect{ToGlm(GetWindowPos()), ToGlm(GetContentRegionAvail())};
    const bool active_transform = TransformGizmo::IsUsing(r, viewport);
    static constexpr float OrientationGizmoSize{84};
    const OverlayIconButtonStyle overlay_button_style{};
    const float overlay_corner_gap = GetTextLineHeightWithSpacing() / 2.f;
    const OverlayIconButtonStyle shading_button_style{
        .ButtonSize = {overlay_button_style.ButtonSize.x * 0.75f, overlay_button_style.ButtonSize.y * 0.75f},
        .Padding = overlay_button_style.Padding,
        .IconScale = overlay_button_style.IconScale,
        .CornerRounding = overlay_button_style.CornerRounding * 0.75f,
    };
    // Hold through the press-release cycle so IsSingleClicked (which fires on release) is still guarded.
    if (!IsMouseDown(ImGuiMouseButton_Left)) frame.OverlayControlsHovered = false;
    const bool any_popup_open = IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);

    { // Transform mode pill buttons (top-left overlay)
        using enum TransformGizmo::Type;
        const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
        const bool has_frozen_selected = r.view<Selected, ScaleLocked>().begin() != r.view<Selected, ScaleLocked>().end();
        const bool edit_transform_locked = interaction_mode == InteractionMode::Edit &&
            any_of(selection::GetSelectedMeshEntities(r), [&](entt::entity mesh_entity) { return selection::HasScaleLockedInstance(r, mesh_entity); });
        const bool transform_enabled = !edit_transform_locked;
        const bool scale_enabled = transform_enabled && !has_frozen_selected;

        ui::Edit gizmo_edit{r, viewport};
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

    { // Viewport shading button group + dropdown (top-right overlay)
        const auto start_pos = std::bit_cast<ImVec2>(viewport_rect.pos + vec2{GetWindowContentRegionMax().x - shading_group_width, GetWindowContentRegionMin().y}) + ImVec2{-overlay_corner_gap, overlay_corner_gap};
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

        { // Dropdown arrow button
            auto &dl = *GetWindowDrawList();
            const auto saved_cursor = GetCursorScreenPos();
            SetCursorScreenPos(start_pos + ImVec2{shading_button_w * 4.f, 0.f});
            PushID("##ShadingArrow");
            InvisibleButton("##btn", {shading_arrow_w, shading_button_h});
            const bool arrow_hovered = IsItemHovered();
            PopID();
            SetCursorScreenPos(saved_cursor);

            if (arrow_hovered) frame.OverlayControlsHovered = true;
            const auto arrow_min = start_pos + ImVec2{shading_button_w * 4.f + shading_button_style.Padding.x, shading_button_style.Padding.y};
            const auto arrow_max = start_pos + ImVec2{shading_button_w * 4.f + shading_arrow_w - shading_button_style.Padding.x, shading_button_h - shading_button_style.Padding.y};
            const bool popup_open = IsPopupOpen("##ShadingDropdown");
            const auto bg_color = GetColorU32(popup_open ? ImGuiCol_ButtonActive : arrow_hovered ? ImGuiCol_ButtonHovered :
                                                                                                   ImGuiCol_Button);
            dl.AddRectFilled(arrow_min, arrow_max, bg_color, shading_button_style.CornerRounding, ImDrawFlags_RoundCornersRight);

            const auto center = (arrow_min + arrow_max) * 0.5f;
            const auto arrow_half = 3.5f;
            dl.AddTriangleFilled(
                center - ImVec2{arrow_half, arrow_half * 0.5f},
                center + ImVec2{arrow_half, -arrow_half * 0.5f},
                center + ImVec2{0.f, arrow_half * 0.5f},
                GetColorU32(ImGuiCol_Text)
            );
            if (IsMouseClicked(0) && arrow_hovered && !popup_open) OpenPopup("##ShadingDropdown");
        }
        { // Dropdown popup
            SetNextWindowPos(start_pos + ImVec2{shading_group_width, shading_button_h + 2.f}, ImGuiCond_Always, {1.f, 0.f});
            PushStyleVar(ImGuiStyleVar_WindowPadding, {8, 8});
            if (BeginPopup("##ShadingDropdown")) {
                frame.OverlayControlsHovered = true;
                TextUnformatted("Viewport shading");
                Separator();
                const auto current_mode = settings.ViewportShading;

                const auto render_pbr_controls = [&]<typename T>(const T &lighting, const char *id) {
                    PushID(id);
                    const auto apply_update = [&]<typename Field>(Field PBRViewportLighting::*member, Field v) {
                        action::Emit(action::UpdateOf<T>(viewport, static_cast<Field T::*>(member), v));
                    };
                    if (Button("Reset")) action::Emit(action::view::ResetPbrLighting{.Rendered = std::is_same_v<T, RenderedLighting>});
                    if (bool v = lighting.UseSceneLights; Checkbox("Scene lights", &v)) apply_update(&PBRViewportLighting::UseSceneLights, v);
                    SameLine();
                    if (bool v = lighting.UseSceneWorld; Checkbox("Scene world", &v)) apply_update(&PBRViewportLighting::UseSceneWorld, v);
                    const auto *source_assets = r.try_get<const gltf::SourceAssets>(viewport);
                    const auto *source_ibl = source_assets && source_assets->ImageBasedLight.has_value() ? &*source_assets->ImageBasedLight : nullptr;
                    if (lighting.UseSceneWorld) {
                        // Spec-defined intensity edits the source IBL directly so save round-trips.
                        if (source_ibl) {
                            if (float v = source_ibl->Intensity; SliderFloat("Intensity", &v, 0.f, 2.f, "%.2f"))
                                action::Emit(action::view::SetSourceIblIntensity{v});
                        }
                    } else {
                        const auto hdris = GetHdriRefs(r);
                        if (BeginCombo("Environment", hdris.Names[hdris.ActiveIndex].c_str())) {
                            for (uint32_t i = 0; i < hdris.Names.size(); ++i) {
                                const bool selected = (i == hdris.ActiveIndex);
                                if (Selectable(hdris.Names[i].c_str(), selected)) action::Emit(action::view::SetStudioEnvironment{i});
                                if (selected) SetItemDefaultFocus();
                            }
                            EndCombo();
                        }
                        if (float v = lighting.EnvIntensity; SliderFloat("Intensity", &v, 0.f, 2.f, "%.2f"))
                            apply_update(&PBRViewportLighting::EnvIntensity, v);
                        if (float v = lighting.EnvRotationDegrees; SliderFloat("Rotation", &v, -180.f, 180.f, "%.1f deg"))
                            apply_update(&PBRViewportLighting::EnvRotationDegrees, v);
                    }
                    if (float v = lighting.BackgroundBlur; SliderFloat("Blur", &v, 0.f, 1.f, "%.2f"))
                        apply_update(&PBRViewportLighting::BackgroundBlur, v);
                    if (float v = lighting.WorldOpacity; SliderFloat("World opacity", &v, 0.f, 1.f, "%.2f"))
                        apply_update(&PBRViewportLighting::WorldOpacity, v);
                    if (bool v = lighting.RealTransmission; Checkbox("Real transmission", &v))
                        apply_update(&PBRViewportLighting::RealTransmission, v);
                    if (IsItemHovered()) SetTooltip("Sample transmission from a pre-rendered scene framebuffer instead of from the IBL.");
                    PopID();
                };

                if (current_mode == ViewportShadingMode::MaterialPreview) {
                    SeparatorText("Material Preview lighting");
                    render_pbr_controls(r.get<const MaterialPreviewLighting>(viewport), "MatPreviewLighting");
                } else if (current_mode == ViewportShadingMode::Rendered) {
                    SeparatorText("Rendered lighting");
                    render_pbr_controls(r.get<const RenderedLighting>(viewport), "RenderedLighting");
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
                        if (auto srgb = glm::pow(linear, vec3{1.f / 2.2f}); ColorEdit3(label, &srgb[0])) {
                            linear = glm::pow(srgb, vec3{2.2f});
                            return true;
                        }
                        return false;
                    };
                    changed |= linear_color_edit("Ambient color", lights.AmbientColor);
                    static const char *light_names[]{"Light 1", "Light 2", "Light 3", "Light 4"};
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
                    if (changed) action::Emit(action::Replace<WorkspaceLights>{viewport, std::make_unique<WorkspaceLights>(lights)});
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

    { // Viewport overlays toggle + dropdown
        const auto buttons_gap = 6.f;
        const auto arrow_w = shading_arrow_w;
        const auto icon_w = shading_button_w;
        const auto button_h = shading_button_h;
        const auto overlay_group_width = icon_w + arrow_w;
        const auto group_start = std::bit_cast<ImVec2>(viewport_rect.pos + vec2{GetWindowContentRegionMax().x - shading_group_width - buttons_gap - overlay_group_width, GetWindowContentRegionMin().y}) + ImVec2{-overlay_corner_gap, overlay_corner_gap};

        {
            const OverlayIconButtonInfo icon_button[]{
                {icons.Overlay.get(), {0.f, 0.f}, ImDrawFlags_RoundCornersLeft, true, settings.ShowOverlays, "Toggle overlays"},
            };
            if (const auto clicked = DrawOverlayIconButtonGroup("ViewportOverlays", group_start, icon_button, !active_transform, &frame.OverlayControlsHovered, shading_button_style)) {
                action::Emit(action::UpdateOf<&ViewportDisplay::ShowOverlays>(viewport, !settings.ShowOverlays));
            }
        }
        { // Dropdown arrow button
            auto &dl = *GetWindowDrawList();

            const auto saved_cursor = GetCursorScreenPos();
            SetCursorScreenPos(group_start + ImVec2{icon_w, 0.f});
            PushID("##OverlayArrow");
            InvisibleButton("##btn", {arrow_w, button_h});
            const bool arrow_hovered = IsItemHovered();
            PopID();
            SetCursorScreenPos(saved_cursor);

            if (arrow_hovered) frame.OverlayControlsHovered = true;
            const auto arrow_min = group_start + ImVec2{icon_w + shading_button_style.Padding.x, shading_button_style.Padding.y};
            const auto arrow_max = group_start + ImVec2{icon_w + arrow_w - shading_button_style.Padding.x, button_h - shading_button_style.Padding.y};
            const bool popup_open = IsPopupOpen("##OverlayDropdown");
            const auto bg_color = GetColorU32(popup_open ? ImGuiCol_ButtonActive : arrow_hovered ? ImGuiCol_ButtonHovered :
                                                                                                   ImGuiCol_Button);
            dl.AddRectFilled(arrow_min, arrow_max, bg_color, shading_button_style.CornerRounding, ImDrawFlags_RoundCornersRight);

            // Triangle arrow
            const auto center = (arrow_min + arrow_max) * 0.5f;
            const auto arrow_half = 3.5f;
            dl.AddTriangleFilled(
                center - ImVec2{arrow_half, arrow_half * 0.5f},
                center + ImVec2{arrow_half, -arrow_half * 0.5f},
                center + ImVec2{0.f, arrow_half * 0.5f},
                GetColorU32(ImGuiCol_Text)
            );
            // Open on press (not release) so the popup is still in the stack and !popup_open gates correctly,
            // matching how ImGui's own BeginCombo works (PressedOnClick).
            if (IsMouseClicked(0) && arrow_hovered && !popup_open) OpenPopup("##OverlayDropdown");
        }
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
    { // Orientation gizmo (interacted before tick so camera animations it initiates begin this frame)
        const float shading_group_height = shading_button_style.ButtonSize.y;
        const auto pos = viewport_rect.pos + vec2{GetWindowContentRegionMax().x - OrientationGizmoSize, GetWindowContentRegionMin().y} + vec2{-overlay_corner_gap, overlay_corner_gap * 2 + shading_group_height};
        if (auto interaction = OrientationGizmo::Interact(pos, OrientationGizmoSize, camera, !active_transform && !any_popup_open)) {
            std::visit(
                overloaded{
                    [&](OrientationGizmo::RotateBy r) { action::Emit(action::view::OrbitViewCamera{r.Delta}); },
                    [&](OrientationGizmo::AlignTo a) { action::Emit(action::view::SetViewCameraTargetDirection{a.Direction}); },
                },
                *interaction
            );
        }
    }
    // Intentionally mutating registry outside of Apply. TODO should all non-saved state be outside the registry?
    if (r.get<ViewCamera>(viewport).Tick()) r.patch<ViewCamera>(viewport, [](auto &) {});

    const auto selected_view = r.view<const Selected>();
    const auto bone_selected_view = r.view<const BoneSelection>();
    const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
    const auto active_entity = FindActiveEntity(r);
    const auto arm_obj = FindArmatureObject(r, active_entity);
    const bool bone_edit_mode = interaction_mode == InteractionMode::Edit && arm_obj != entt::null;
    const bool bone_mode = bone_edit_mode || (interaction_mode == InteractionMode::Pose && arm_obj != entt::null);
    const bool mesh_edit_mode = interaction_mode == InteractionMode::Edit && !bone_edit_mode;

    const auto has_transform_target = [&]() {
        if (bone_mode) return !bone_selected_view.empty();
        if (selected_view.empty()) return false;
        if (!mesh_edit_mode) return true;
        const auto *bits = r.ctx().get<const SelectionBitsetRef>().Value.data();
        for (const auto [e, instance] : r.view<const Instance, const Selected>(entt::exclude<ScaleLocked>).each()) {
            if (const auto *br = r.try_get<const MeshSelectionBitsetRange>(instance.Entity)) {
                if (selection::CountSelected(bits, br->Offset, br->Count) > 0) return true;
            }
        }
        return false;
    }();
    if (has_transform_target) { // Transform gizmo
        // Transform all root selected entities (whose parent is not also selected) around their average
        // position, using the active entity's rotation/scale.
        const auto gizmo_active_entity = bone_mode ? FindActiveBone(r) : active_entity;
        const auto active_transform = [&]() -> Transform {
            if (gizmo_active_entity == entt::null) return {};
            const auto &wt = r.get<WorldTransform>(gizmo_active_entity);
            return wt;
        }();

        const auto root_selected = RootSelectedForTransform(r, viewport);
        const auto root_count = root_selected.size();
        const auto edit_transform_instances = mesh_edit_mode ?
            selection::ComputePrimaryEditInstances(r, false) :
            std::unordered_map<entt::entity, entt::entity>{};

        vec3 pivot{};
        if (mesh_edit_mode) {
            // Compute world-space centroid of selected vertices once per selected mesh
            // (using a representative selected instance for world transform).
            uint32_t vertex_count = 0;
            for (const auto &[mesh_entity, instance_entity] : edit_transform_instances) {
                const auto &mesh = r.get<const Mesh>(mesh_entity);
                const auto vertex_states = meshes.GetVertexStates(mesh.GetStoreId());
                const auto vertices = mesh.GetVerticesSpan();
                const auto &wt = r.get<const WorldTransform>(instance_entity);
                for (uint32_t vi = 0; vi < vertex_states.size(); ++vi) {
                    if ((vertex_states[vi] & ElementStateSelected) == 0u) continue;
                    pivot += wt.P + glm::rotate(wt.R, wt.S * vertices[vi].Position);
                    ++vertex_count;
                }
            }
            if (vertex_count > 0) pivot /= float(vertex_count);
            // Apply pending transform to gizmo position (vertices aren't modified until commit).
            if (const auto *pending = r.try_get<const PendingTransform>(viewport)) {
                pivot += pending->Delta.P;
            }
        } else {
            if (bone_edit_mode) {
                // Bone pivot: contribute head position for selected Root, tail for selected Tip.
                // A fully-selected bone (Root+Tip+Body) contributes both → midpoint.
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
                        pivot_sum += wt.P + glm::rotate(wt.R, vec3{0, bl, 0});
                        ++pivot_count;
                    }
                }
                pivot = pivot_count > 0 ? pivot_sum / float(pivot_count) : vec3{};
            } else {
                pivot = fold_left(root_selected | transform([&](auto e) { return r.get<WorldTransform>(e).P; }), vec3{}, std::plus{}) / float(root_count);
            }
        }

        const auto start_transform_view = r.view<const StartTransform>();
        const auto &gizmo_state = r.get<const TransformGizmoState>(viewport);
        auto &gizmo = r.get<GizmoInteraction>(viewport);
        const auto gizmo_transform = GizmoTransform{{.P = pivot, .R = active_transform.R, .S = active_transform.S}, gizmo_state.Mode};
        const auto *start_screen = r.try_get<const StartScreenTransform>(viewport);
        auto interact_result = TransformGizmo::Interact(
            gizmo,
            gizmo_transform,
            gizmo_state.Config, camera, viewport_rect, ToGlm(GetMousePos()) + frame.AccumulatedWrapMouseDelta,
            start_screen ? std::optional{start_screen->Value} : std::nullopt
        );
        if (interact_result) {
            const auto &[ts, td] = *interact_result;
            if (mesh_edit_mode) {
                // Mesh Edit mode: store pending transform for shader-based preview.
                // Actual vertex positions are only modified on commit.
                action::Emit(action::view::DragGizmoMeshEdit{
                    .Value = std::make_unique<PendingTransform>(PendingTransform{ts.P, ts.R, td}),
                });
            } else {
                // Object/bone mode: store the gizmo pivot + delta. Apply recomputes per-entity transforms.
                action::Emit(action::view::DragGizmo{.Value = std::make_unique<PendingTransform>(PendingTransform{ts.P, ts.R, td})});
            }
        } else if (!start_transform_view.empty()) {
            action::Emit(action::view::EndGizmoDrag{});
        }

        // Store gizmo render transform for DrawOverlay.
        gizmo.RenderTransform = gizmo_transform;
        if (interact_result) gizmo.RenderTransform->P = interact_result->Start.P + interact_result->Delta.P;
    }

    if (r.all_of<StartScreenTransform>(viewport)) action::Emit(action::view::SetStartScreenTransform{});
}

void DrawOverlay(entt::registry &r, entt::entity viewport, FrameState &frame) {
    const rect viewport_rect{ToGlm(GetWindowPos()), ToGlm(GetContentRegionAvail())};
    const auto axes = colors::MakeAxes(r.get<const ViewportTheme>(viewport).AxisColors);
    const auto &camera = r.get<const ViewCamera>(viewport);

    OrientationGizmo::Render(axes);
    TransformGizmo::Render(r.get<GizmoInteraction>(viewport), r.get<const TransformGizmoState>(viewport).Config.Type, camera, viewport_rect, axes);

    const auto &settings = r.get<const ViewportDisplay>(viewport);
    // Draw origin dots for active/selected entities
    if (settings.ShowOverlays && settings.ShowOrigins && (!r.storage<Selected>().empty() || !r.storage<Active>().empty())) {
        const auto &theme = r.get<const ViewportTheme>(viewport);
        const auto vp = camera.Projection(viewport_rect.size.x / viewport_rect.size.y) * camera.View();
        auto draw_dot = [&](vec3 pos, bool is_active) {
            const auto p_cs = vp * vec4{pos, 1.f};
            if (p_cs.w <= 0) return; // Behind camera

            const auto p_ndc = vec3{p_cs} / p_cs.w;
            const auto p_uv = vec2{p_ndc.x + 1, 1 - p_ndc.y} * 0.5f;
            const auto p_px = std::bit_cast<ImVec2>(viewport_rect.pos + p_uv * viewport_rect.size);
            auto &dl = *GetWindowDrawList();
            dl.AddCircleFilled(p_px, 3.5f, colors::RgbToU32(is_active ? theme.Colors.ObjectActive : theme.Colors.ObjectSelected), 10);
            dl.AddCircle(p_px, 3.5f, IM_COL32(0, 0, 0, 255), 10, 1.f);
        };
        // Top-level objects: draw dot at own position.
        for (const auto [e, wt] : r.view<const WorldTransform>(entt::exclude<SubElementOf>).each()) {
            if (!r.any_of<Active, Selected>(e)) continue;
            draw_dot(wt.P, r.all_of<Active>(e));
        }
    }

    if (frame.BoxSelectStart && frame.BoxSelectEnd) {
        auto &dl = *GetWindowDrawList();
        const auto box_min = glm::min(*frame.BoxSelectStart, *frame.BoxSelectEnd);
        const auto box_max = glm::max(*frame.BoxSelectStart, *frame.BoxSelectEnd);
        dl.AddRectFilled(std::bit_cast<ImVec2>(box_min), std::bit_cast<ImVec2>(box_max), IM_COL32(255, 255, 255, 30));

        // Dashed outline
        static constexpr auto outline_color{IM_COL32(255, 255, 255, 200)};
        static constexpr float dash_size{4}, gap_size{4};
        // Top
        for (float x = box_min.x; x < box_max.x; x += dash_size + gap_size) {
            dl.AddLine({x, box_min.y}, {glm::min(x + dash_size, box_max.x), box_min.y}, outline_color, 1.f);
        }
        // Bottom
        for (float x = box_min.x; x < box_max.x; x += dash_size + gap_size) {
            dl.AddLine({x, box_max.y}, {glm::min(x + dash_size, box_max.x), box_max.y}, outline_color, 1.f);
        }
        // Left
        for (float y = box_min.y; y < box_max.y; y += dash_size + gap_size) {
            dl.AddLine({box_min.x, y}, {box_min.x, glm::min(y + dash_size, box_max.y)}, outline_color, 1.f);
        }
        // Right
        for (float y = box_min.y; y < box_max.y; y += dash_size + gap_size) {
            dl.AddLine({box_max.x, y}, {box_max.x, glm::min(y + dash_size, box_max.y)}, outline_color, 1.f);
        }
    }

    // Camera look-through frame overlay: show the looked-through camera's view as a centered frame.
    // The ViewCamera's FOV is widened so the camera's view fits inside with padding.
    // The frame marks exactly what the camera captures.
    if (const auto look_through_entity = LookThroughCameraEntity(r); look_through_entity != entt::null && !camera.IsAnimating()) {
        if (const auto *cd = r.try_get<Camera>(look_through_entity)) {
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
