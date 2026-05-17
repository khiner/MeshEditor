#include "PbrFeature.h"
#include "Scene.h"
#include "SceneDefaults.h"
#include "SceneOps.h"
#include "SceneTextures.h"
#include "SceneTree.h"
#include "TransformMath.h"
#include "Widgets.h" // imgui

#include "Armature.h"
#include "Instance.h"
#include "MeshComponents.h"
#include "NodeTransformAnimation.h"
#include "OrientationGizmo.h"
#include "Path.h"
#include "SceneSelection.h"
#include "SoundVertices.h"
#include "SvgResource.h"
#include "Tets.h"
#include "Timer.h"
#include "Variant.h"
#include "audio/AudioSystem.h"
#include "audio/RealImpact.h"
#include "audio/RealImpactComponents.h"
#include "gltf/GltfScene.h"
#include "gpu/Transform.h"
#include "gpu/WorkspaceLights.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "numeric/mat3.h"
#include "numeric/rect.h"
#include "physics/PhysicsUi.h"
#include "physics/PhysicsWorld.h"
#include "ui/FieldEdit.h"

#include <algorithm>
#include <cmath>
#include <entt/entity/registry.hpp>
#include <imgui_internal.h>

#include "scene_impl/SceneBuffers.h"
#include "scene_impl/SceneComponents.h"
#include "scene_impl/SceneInternalTypes.h"
#include "scene_impl/SceneTransformUtils.h"

using std::ranges::any_of, std::ranges::contains, std::ranges::distance, std::ranges::find, std::ranges::find_if, std::ranges::fold_left, std::ranges::to;
using std::views::transform;
using namespace ImGui;

namespace {
constexpr vec2 ToGlm(ImVec2 v) { return std::bit_cast<vec2>(v); }
constexpr float WheelOrbitRadPerUnit{0.05f};
constexpr float WheelZoomStep{1.04f};

struct SelectionHit {
    entt::entity Entity;
    std::optional<BoneSel> Part{};
    bool operator==(const SelectionHit &) const = default;
};

// Map raw GPU pick/box-select instances to logical selection targets.
// In bone mode, body + joint spheres collapse to one entry per bone.
// merge_parts: true merges multiple parts to nullopt (= all parts); false keeps the first (closest) part.
// In object mode, bones fall through to SubElementOf like any other sub-element, collapsing to the armature.
std::vector<SelectionHit> ResolveHits(entt::registry &r, const std::vector<entt::entity> &raw, bool bone_mode, bool merge_parts = false) {
    std::vector<SelectionHit> hits;
    for (const auto e : raw) {
        if (bone_mode && r.all_of<BoneIndex>(e)) {
            if (auto it = find(hits, e, &SelectionHit::Entity); it == hits.end()) hits.emplace_back(e, BoneSel::Body);
            else if (merge_parts) it->Part = {};
        } else if (bone_mode && r.all_of<BoneSubPartOf>(e)) {
            const auto &sub = r.get<BoneSubPartOf>(e);
            if (auto it = find(hits, sub.BoneEntity, &SelectionHit::Entity); it == hits.end()) hits.emplace_back(sub.BoneEntity, sub.IsTip ? BoneSel::Tip : BoneSel::Root);
            else if (merge_parts) it->Part = {};
        } else if (!bone_mode) {
            if (const auto target = r.all_of<SubElementOf>(e) ? r.get<SubElementOf>(e).Parent : e; !contains(hits, target, &SelectionHit::Entity)) hits.emplace_back(target);
        }
    }
    return hits;
}

std::optional<std::pair<uvec2, uvec2>> ComputeBoxSelectPixels(vec2 start, vec2 end, vec2 window_pos, vk::Extent2D logical_extent, vk::Extent2D render_extent) {
    static constexpr float DragThresholdSq{2 * 2};
    if (glm::distance2(start, end) <= DragThresholdSq) return {};

    const vec2 logical_size{float(logical_extent.width), float(logical_extent.height)};
    const vec2 render_scale{
        logical_extent.width > 0u ? float(render_extent.width) / float(logical_extent.width) : 1.0f,
        logical_extent.height > 0u ? float(render_extent.height) / float(logical_extent.height) : 1.0f
    };
    const auto box_min = glm::min(start, end) - window_pos;
    const auto box_max = glm::max(start, end) - window_pos;
    const auto local_min = glm::clamp(glm::min(box_min, box_max), vec2{0}, logical_size);
    const auto local_max = glm::clamp(glm::max(box_min, box_max), vec2{0}, logical_size);
    const auto render_min = local_min * render_scale;
    const auto render_max = local_max * render_scale;
    const uvec2 box_min_px{glm::floor(render_min.x), glm::floor(float(render_extent.height) - render_max.y)};
    const uvec2 box_max_px{glm::ceil(render_max.x), glm::ceil(float(render_extent.height) - render_min.y)};
    return std::pair{box_min_px, box_max_px};
}

constexpr std::string Capitalize(std::string_view str) {
    if (str.empty()) return {};

    std::string result{str};
    char &c = result[0];
    if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
    return result;
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

std::optional<MeshData> PrimitiveEditor(PrimitiveShape &shape) {
    static constexpr float MinSize = 0.01f, MaxSize = 100.f, SizeSpeed = 0.01f;
    return std::visit([](auto &s) -> std::optional<MeshData> {
        using T = std::decay_t<decltype(s)>;
        bool changed = false;
        if constexpr (std::is_same_v<T, primitive::Plane>) {
            vec2 size = s.HalfExtents * 2.f;
            changed = DragFloat2("Size", &size.x, SizeSpeed, MinSize, MaxSize);
            if (changed) s.HalfExtents = size / 2.f;
        } else if constexpr (std::is_same_v<T, primitive::Circle>) {
            changed |= DragFloat("Radius", &s.Radius, SizeSpeed, MinSize, MaxSize);
            int segments = int(s.Segments);
            changed |= SliderInt("Segments", &segments, 3, 128);
            if (changed) s.Segments = uint(segments);
        } else if constexpr (std::is_same_v<T, primitive::Cuboid>) {
            vec3 size = s.HalfExtents * 2.f;
            changed = DragFloat3("Size", &size.x, SizeSpeed, MinSize, MaxSize);
            if (changed) s.HalfExtents = size / 2.f;
        } else if constexpr (std::is_same_v<T, primitive::IcoSphere>) {
            changed |= DragFloat("Radius", &s.Radius, SizeSpeed, MinSize, MaxSize);
            int subdivisions = int(s.Subdivisions);
            changed |= SliderInt("Subdivisions", &subdivisions, 1, 6);
            if (changed) s.Subdivisions = uint(subdivisions);
        } else if constexpr (std::is_same_v<T, primitive::UVSphere>) {
            changed |= DragFloat("Radius", &s.Radius, SizeSpeed, MinSize, MaxSize);
            int slices = int(s.Slices), stacks = int(s.Stacks);
            changed |= SliderInt("Slices", &slices, 3, 128);
            changed |= SliderInt("Stacks", &stacks, 2, 64);
            if (changed) {
                s.Slices = uint(slices);
                s.Stacks = uint(stacks);
            }
        } else if constexpr (std::is_same_v<T, primitive::Torus>) {
            changed |= DragFloat("Major radius", &s.MajorRadius, SizeSpeed, MinSize, MaxSize);
            changed |= DragFloat("Minor radius", &s.MinorRadius, SizeSpeed, MinSize, s.MajorRadius);
            int major_seg = int(s.MajorSegments), minor_seg = int(s.MinorSegments);
            changed |= SliderInt("Major segments", &major_seg, 3, 256);
            changed |= SliderInt("Minor segments", &minor_seg, 3, 256);
            if (changed) {
                s.MajorSegments = uint(major_seg);
                s.MinorSegments = uint(minor_seg);
            }
        } else if constexpr (std::is_same_v<T, primitive::Cylinder> || std::is_same_v<T, primitive::Cone>) {
            changed |= DragFloat("Radius", &s.Radius, SizeSpeed, MinSize, MaxSize);
            changed |= DragFloat("Height", &s.Height, SizeSpeed, MinSize, MaxSize);
            int slices = int(s.Slices);
            changed |= SliderInt("Slices", &slices, 3, 128);
            if (changed) s.Slices = uint(slices);
        }
        if (changed) return primitive::CreateMesh(s);
        return {};
    },
                      shape);
}

std::string to_string(InteractionMode mode) {
    switch (mode) {
        case InteractionMode::Object: return "Object";
        case InteractionMode::Edit: return "Edit";
        case InteractionMode::Excite: return "Excite";
        case InteractionMode::Pose: return "Pose";
    }
}

using namespace he;

float AngleFromCos(float cos_theta) { return std::acos(std::clamp(cos_theta, -1.f, 1.f)); }

bool AllSelectedAreMeshes(const entt::registry &r) {
    for (const auto [e, ok] : r.view<const Selected, const ObjectKind>().each()) {
        if (ok.Value != ObjectType::Mesh) return false;
    }
    return true;
}

// `viewport_aspect` is set when the camera is bound to a viewport that determines its aspect.
bool RenderCameraLensEditor(Camera &camera, float distance, std::optional<float> viewport_aspect = {}) {
    bool lens_changed = false;

    int proj_i = std::holds_alternative<Orthographic>(camera) ? 1 : 0;
    const char *proj_names[]{"Perspective", "Orthographic"};
    if (Combo("Projection", &proj_i, proj_names, IM_ARRAYSIZE(proj_names))) {
        if (proj_i == 0 && !std::holds_alternative<Perspective>(camera)) {
            camera = PerspectiveFromOrthographic(std::get<Orthographic>(camera), distance);
            lens_changed = true;
        } else if (proj_i == 1 && !std::holds_alternative<Orthographic>(camera)) {
            camera = OrthographicFromPerspective(std::get<Perspective>(camera), distance, viewport_aspect);
            lens_changed = true;
        }
    }

    if (auto *perspective = std::get_if<Perspective>(&camera)) {
        float fov_deg = glm::degrees(perspective->FieldOfViewRad);
        const float far_max = std::max(perspective->NearClip + MinNearFarDelta, MaxFarClip);
        if (SliderFloat("Field of view (deg)", &fov_deg, 1.f, 179.f)) {
            perspective->FieldOfViewRad = glm::radians(fov_deg);
            lens_changed = true;
        }
        const float near_max = perspective->FarClip ? std::max(*perspective->FarClip - MinNearFarDelta, MinNearClip) : far_max;
        lens_changed |= SliderFloat("Near clip", &perspective->NearClip, MinNearClip, near_max);
        bool infinite_far = !perspective->FarClip.has_value();
        if (Checkbox("Infinite far clip", &infinite_far)) {
            if (infinite_far) perspective->FarClip.reset();
            else perspective->FarClip = far_max;
            lens_changed = true;
        }
        if (perspective->FarClip) {
            lens_changed |= SliderFloat("Far clip", &*perspective->FarClip, perspective->NearClip + MinNearFarDelta, far_max);
        }
        if (!viewport_aspect) {
            float aspect = perspective->AspectRatio.value_or(DefaultAspectRatio);
            if (SliderFloat("Aspect ratio", &aspect, 0.1f, 5.f)) {
                perspective->AspectRatio = aspect;
                lens_changed = true;
            }
        }
    } else if (auto *orthographic = std::get_if<Orthographic>(&camera)) {
        const float far_max = std::max(orthographic->NearClip + MinNearFarDelta, MaxFarClip);
        lens_changed |= SliderFloat("X Mag", &orthographic->Mag.x, 0.01f, 100.f);
        lens_changed |= SliderFloat("Y Mag", &orthographic->Mag.y, 0.01f, 100.f);
        lens_changed |= SliderFloat("Near clip", &orthographic->NearClip, MinNearClip, orthographic->FarClip - MinNearFarDelta);
        lens_changed |= SliderFloat("Far clip", &orthographic->FarClip, orthographic->NearClip + MinNearFarDelta, far_max);
    }
    return lens_changed;
}

std::string NamedOr(const std::string &name, std::string_view fallback, uint32_t i) {
    return name.empty() ? std::format("{}{}", fallback, i) : name;
}

constexpr std::string_view MimeTypeName(gltf::MimeType m) {
    using gltf::MimeType;
    switch (m) {
        case MimeType::None: return "—";
        case MimeType::JPEG: return "image/jpeg";
        case MimeType::PNG: return "image/png";
        case MimeType::KTX2: return "image/ktx2";
        case MimeType::DDS: return "image/vnd-ms.dds";
        case MimeType::GltfBuffer: return "model/gltf-buffer";
        case MimeType::OctetStream: return "application/octet-stream";
        case MimeType::WEBP: return "image/webp";
    }
    return "?";
}

std::string AttributeFlagsString(uint32_t flags) {
    std::string s;
    const auto add = [&](std::string_view tag) {
        if (!s.empty()) s += '|';
        s += tag;
    };
    add("POS"); // Always present.
    if (flags & MeshAttributeBit_Normal) add("NRM");
    if (flags & MeshAttributeBit_Tangent) add("TAN");
    if (flags & MeshAttributeBit_Color0) add("COL0");
    if (flags & MeshAttributeBit_TexCoord0) add("UV0");
    if (flags & MeshAttributeBit_TexCoord1) add("UV1");
    if (flags & MeshAttributeBit_TexCoord2) add("UV2");
    if (flags & MeshAttributeBit_TexCoord3) add("UV3");
    return s;
}

void RenderJsonBlock(const char *label, std::string_view json) {
    SeparatorText(label);
    PushID(label);
    const float row_h = GetTextLineHeightWithSpacing();
    InputTextMultiline(
        "##json", const_cast<char *>(json.data()), json.size() + 1,
        ImVec2{-FLT_MIN, row_h * 6.f}, ImGuiInputTextFlags_ReadOnly
    );
    if (SmallButton("Copy")) SetClipboardText(std::string{json}.c_str());
    PopID();
}

constexpr ImGuiTableFlags MetadataTableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;

} // namespace

entt::entity Scene::LookThroughCameraEntity() const {
    auto view = R.view<LookingThrough>();
    return view.empty() ? entt::null : *view.begin();
}

void Scene::SetLookThrough(entt::entity target) {
    const auto previous = LookThroughCameraEntity();
    if (previous == target) return;
    // Preserve the saved view across camera switches; only capture fresh on first entry.
    auto saved = previous != entt::null ? R.get<LookingThrough>(previous).SavedViewCamera : R.get<ViewCamera>(SceneEntity);
    if (previous != entt::null) R.remove<LookingThrough>(previous);
    R.emplace<LookingThrough>(target, std::move(saved));
}

void Scene::Interact() {
    // Any open popup (e.g. Viewport shading dropdown) blocks viewport mouse/keyboard input.
    // Without this, wheel/click events still patch the camera while the popup overlays the viewport.
    if (IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
        PreciseWheelDelta = {0, 0};
        return;
    }

    // Track the previous click position for pick-cycle gating.
    static ImVec2 PrevClickPos{-FLT_MAX, -FLT_MAX}, CurrentClickPos{-FLT_MAX, -FLT_MAX};
    if (GetIO().MouseClicked[0]) {
        PrevClickPos = CurrentClickPos;
        CurrentClickPos = GetIO().MouseClickedPos[0];
    }

    const auto logical_extent = R.get<const ViewportExtent>(SceneEntity).Value;
    const auto render_extent = ComputeRenderExtentPx(logical_extent, std::bit_cast<vec2>(GetIO().DisplayFramebufferScale));
    if (logical_extent.width == 0 || logical_extent.height == 0 || render_extent.width == 0 || render_extent.height == 0) return;

    const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    const auto active_entity = FindActiveEntity(R);
    const bool has_frozen_selected = R.view<Selected, ScaleLocked>().begin() != R.view<Selected, ScaleLocked>().end();
    const bool edit_transform_locked = interaction_mode == InteractionMode::Edit &&
        any_of(scene_selection::GetSelectedMeshEntities(R), [&](entt::entity mesh_entity) { return scene_selection::HasScaleLockedInstance(R, mesh_entity); });
    const bool transform_shortcuts_enabled = !edit_transform_locked;
    const bool scale_shortcut_enabled = transform_shortcuts_enabled && !has_frozen_selected;
    // Keyboard shortcuts use ImGui's Shortcut() routing system with RouteGlobal so they fire from any
    // focused window in the dockspace. RouteGlobal yields to active items (sliders mid-drag, focused
    // InputText, etc.) and ImGui's Nav (Tab/arrows) via key-ownership, so widget editing and tree/list
    // navigation in panels keep working. char-input keys (G/A/etc.) are auto-filtered while WantTextInput.
    constexpr auto VKey = ImGuiInputFlags_RouteGlobal;
    if (TransformGizmo::IsUsing()) {
        // During an active transform, only allow transform switching shortcuts.
        if (Shortcut(ImGuiKey_G, VKey) && transform_shortcuts_enabled) {
            StartScreenTransform = TransformGizmo::TransformType::Translate;
        } else if (Shortcut(ImGuiKey_R, VKey) && transform_shortcuts_enabled) {
            StartScreenTransform = TransformGizmo::TransformType::Rotate;
        } else if (Shortcut(ImGuiKey_S, VKey) && scale_shortcut_enabled) {
            StartScreenTransform = TransformGizmo::TransformType::Scale;
        }
    } else {
        if (Shortcut(ImGuiKey_Space, VKey)) Apply(action::timeline::TogglePlay{});
        if (Shortcut(ImGuiKey_Z, VKey)) {
            const auto current = R.get<const SceneSettings>(SceneEntity).ViewportShading;
            const auto next = current == ViewportShadingMode::Solid ? ViewportShadingMode::MaterialPreview :
                current == ViewportShadingMode::MaterialPreview     ? ViewportShadingMode::Rendered :
                                                                      ViewportShadingMode::Solid;
            Apply(action::scene::SetViewportShading{.Mode = next});
        } else if (Shortcut(ImGuiMod_Shift | ImGuiKey_Z, VKey)) {
            const auto &settings = R.get<const SceneSettings>(SceneEntity);
            Apply(action::scene::SetViewportShading{.Mode = settings.ViewportShading == ViewportShadingMode::Wireframe ? settings.FillMode : ViewportShadingMode::Wireframe});
        } else if (Shortcut(ImGuiMod_Alt | ImGuiKey_Z, VKey)) {
            SelectionXRay = !SelectionXRay;
        }
        // Tab uses default RouteFocused (not VKey/RouteGlobal) so widget tabbing in panels keeps working.
        const bool tab_no_mods = Shortcut(ImGuiKey_Tab);
        const bool tab_ctrl = Shortcut(ImGuiMod_Ctrl | ImGuiKey_Tab);
        if (tab_no_mods || tab_ctrl) {
            const bool is_armature = FindArmatureObject(R, active_entity) != entt::null;
            if (is_armature && tab_ctrl) {
                Apply(action::scene::SetInteractionMode{.Mode = interaction_mode == InteractionMode::Pose ? InteractionMode::Object : InteractionMode::Pose});
            } else if (is_armature) {
                Apply(action::scene::SetInteractionMode{.Mode = interaction_mode == InteractionMode::Edit ? InteractionMode::Object : InteractionMode::Edit});
            } else if (tab_no_mods) {
                Apply(action::scene::CycleInteractionMode{});
            }
        }
        if (interaction_mode == InteractionMode::Edit) {
            if (Shortcut(ImGuiKey_1, VKey)) Apply(action::scene::SetEditMode{.Mode = Element::Vertex});
            else if (Shortcut(ImGuiKey_2, VKey)) Apply(action::scene::SetEditMode{.Mode = Element::Edge});
            else if (Shortcut(ImGuiKey_3, VKey)) Apply(action::scene::SetEditMode{.Mode = Element::Face});
        }
        if (Shortcut(ImGuiKey_A, VKey)) Apply(action::scene::SelectAll{});
        const bool bone_edit = interaction_mode == InteractionMode::Edit && FindArmatureObject(R, active_entity) != entt::null;
        if (bone_edit) {
            if (Shortcut(ImGuiMod_Shift | ImGuiKey_A, VKey)) {
                AddBone();
            } else if (Shortcut(ImGuiKey_E, VKey)) {
                ExtrudeBone();
                StartScreenTransform = TransformGizmo::TransformType::Translate;
            } else if (Shortcut(ImGuiKey_X, VKey) || Shortcut(ImGuiKey_Delete, VKey) || Shortcut(ImGuiKey_Backspace, VKey)) {
                Delete();
            } else if (Shortcut(ImGuiMod_Shift | ImGuiKey_D, VKey)) {
                Duplicate();
            }
        }
        if (Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_E, VKey)) {
            Apply(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
            StartScreenTransform = TransformGizmo::TransformType::Translate;
        } else if (Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_A, VKey)) {
            Apply(action::object::AddArmature{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
            StartScreenTransform = TransformGizmo::TransformType::Translate;
        } else if (Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_C, VKey)) {
            Apply(action::object::AddCamera{.Info = std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive}), .Props = {}});
            StartScreenTransform = TransformGizmo::TransformType::Translate;
        } else if (Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_L, VKey)) {
            Apply(action::object::AddLight{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
            StartScreenTransform = TransformGizmo::TransformType::Translate;
        }
        if (!R.storage<Selected>().empty()) {
            if (!bone_edit && Shortcut(ImGuiMod_Shift | ImGuiKey_D, VKey)) Duplicate();
            else if (!bone_edit && Shortcut(ImGuiMod_Alt | ImGuiKey_D, VKey)) DuplicateLinked();
            else if (!bone_edit && CanDelete() && (Shortcut(ImGuiKey_Delete, VKey) || Shortcut(ImGuiKey_Backspace, VKey))) Delete();
            else if (interaction_mode == InteractionMode::Pose && Shortcut(ImGuiMod_Alt | ImGuiKey_G, VKey)) Apply(action::bone::ClearSelectedTransforms{.Position = true});
            else if (interaction_mode == InteractionMode::Pose && Shortcut(ImGuiMod_Alt | ImGuiKey_R, VKey)) Apply(action::bone::ClearSelectedTransforms{.Rotation = true});
            else if (interaction_mode == InteractionMode::Pose && Shortcut(ImGuiMod_Alt | ImGuiKey_S, VKey)) Apply(action::bone::ClearSelectedTransforms{.Scale = true});
            else if (Shortcut(ImGuiKey_G, VKey) && transform_shortcuts_enabled) {
                // Start transform gizmo in both Object and Edit modes.
                // In Edit mode, shader applies transform to selected vertices.
                // In Object mode, shader applies transform to selected instances.
                StartScreenTransform = TransformGizmo::TransformType::Translate;
            } else if (Shortcut(ImGuiKey_R, VKey) && transform_shortcuts_enabled) StartScreenTransform = TransformGizmo::TransformType::Rotate;
            else if (Shortcut(ImGuiKey_S, VKey) && scale_shortcut_enabled) StartScreenTransform = TransformGizmo::TransformType::Scale;
            else if (Shortcut(ImGuiKey_H, VKey)) Apply(action::object::ToggleHidden{});
            else if (Shortcut(ImGuiMod_Ctrl | ImGuiKey_P, VKey)) Apply(action::object::ParentToActive{});
            else if (Shortcut(ImGuiMod_Alt | ImGuiKey_P, VKey)) Apply(action::object::ClearParent{});
        }
    }

    // Handle mouse input.
    if (!IsMouseDown(ImGuiMouseButton_Left)) Apply(action::scene::ClearExciteImpacts{});

    const bool active_transform = TransformGizmo::IsUsing();
    if (active_transform) {
        // TransformGizmo overrides this mouse cursor during some actions - this is a default.
        SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        WrapMousePos(GetCurrentWindowRead()->InnerClipRect, AccumulatedWrapMouseDelta);
    } else {
        AccumulatedWrapMouseDelta = {0, 0};
    }
    if (active_transform) return; // Only transform gizmo should consume viewport mouse input while active.

    if (!IsWindowHovered() && !BoxSelectStart) return;

    // Mouse wheel for camera rotation, Cmd+wheel to zoom.
    const auto &io = GetIO();
    if (const vec2 wheel = std::exchange(PreciseWheelDelta, vec2{0}); wheel != vec2{0, 0}) {
        // Exit "look through" camera view on any orbit/zoom interaction.
        Apply(action::scene::ExitLookThroughCamera{});
        if (io.KeyCtrl || io.KeySuper) Apply(action::scene::ZoomViewCamera{.Factor = std::pow(WheelZoomStep, -wheel.y)});
        else Apply(action::scene::OrbitViewCamera{.DeltaRad = wheel * WheelOrbitRadPerUnit});
    }
    if (OrientationGizmo::IsActive() || OverlayControlsHovered) return;

    const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    const auto arm_obj_entity = FindArmatureObject(R, active_entity);
    const bool active_is_armature = arm_obj_entity != entt::null;
    const bool bone_mode = interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && active_is_armature);
    const auto deselect_all = [&] { Apply(action::selection::DeselectAll{}); };
    auto set_bone_sel = [&](entt::entity entity, std::optional<BoneSel> part, bool additive) {
        Apply(action::selection::SetBoneSelectionPart{.Entity = entity, .Part = part, .Additive = additive});
    };
    if (SelectionMode == SelectionMode::Box && interaction_mode != InteractionMode::Excite) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            BoxSelectStart = BoxSelectEnd = ToGlm(GetMousePos());
            if (IsKeyDown(ImGuiMod_Shift)) Apply(action::selection::SnapshotBoxSelectBaseline{});
        } else if (IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart) {
            BoxSelectEnd = ToGlm(GetMousePos());
            if (const auto box_px = ComputeBoxSelectPixels(*BoxSelectStart, *BoxSelectEnd, ToGlm(GetCursorScreenPos()), logical_extent, render_extent); box_px) {
                const bool is_additive = R.all_of<AdditiveBoxSelectBaseline>(SceneEntity);
                if (interaction_mode == InteractionMode::Edit && !active_is_armature) {
                    Timer timer{"BoxSelectElements (all)"};
                    RunBoxSelectElements(GetBitsetRangesForSelected(), edit_mode, *box_px, is_additive);
                } else if (bone_mode) {
                    const auto hits = ResolveHits(R, RunBoxSelect(*box_px), bone_mode, true);
                    std::vector<action::selection::BoneHit> bone_hits;
                    bone_hits.reserve(hits.size());
                    for (const auto &[entity, part] : hits) bone_hits.emplace_back(entity, part);
                    Apply(action::selection::ApplyBoxSelectBoneHits{.Hits = std::move(bone_hits), .Additive = is_additive});
                } else if (interaction_mode == InteractionMode::Object) {
                    const auto hits = ResolveHits(R, RunBoxSelect(*box_px), bone_mode, true);
                    std::vector<entt::entity> entities;
                    entities.reserve(hits.size());
                    for (const auto &[entity, _] : hits) entities.push_back(entity);
                    Apply(action::selection::ApplyBoxSelectObjectHits{.Hits = std::move(entities), .Additive = is_additive});
                }
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart) {
            BoxSelectStart.reset();
            BoxSelectEnd.reset();
            Apply(action::selection::ClearBoxSelectBaseline{});
        }
        if (BoxSelectStart) return;
    }

    const vec2 render_scale{
        logical_extent.width > 0u ? float(render_extent.width) / float(logical_extent.width) : 1.0f,
        logical_extent.height > 0u ? float(render_extent.height) / float(logical_extent.height) : 1.0f
    };
    const auto mouse_pos_rel = GetMousePos() - GetCursorScreenPos();
    const auto mouse_pos_render = ToGlm(mouse_pos_rel) * render_scale;
    const float max_x = float(std::max(render_extent.width, 1u) - 1u);
    const float max_y = float(std::max(render_extent.height, 1u) - 1u);
    // Flip y-coordinate: ImGui uses top-left origin, but Vulkan gl_FragCoord uses bottom-left origin
    const uvec2 mouse_px{glm::clamp(mouse_pos_render.x, 0.0f, max_x), glm::clamp(float(render_extent.height) - mouse_pos_render.y, 0.0f, max_y)};

    if (interaction_mode == InteractionMode::Excite) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            if (const auto hit_entities = RunObjectPick(mouse_px); !hit_entities.empty()) {
                if (const auto hit_entity = hit_entities.front(); R.all_of<SoundVertices>(hit_entity)) {
                    if (const auto vertex = RunSoundVerticesVertexPick(hit_entity, mouse_px)) {
                        Apply(action::scene::ApplyExciteImpact{.InstanceEntity = hit_entity, .VertexIndex = *vertex});
                    }
                }
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left)) {
            Apply(action::scene::ClearExciteImpacts{});
        }
        return;
    }
    if (!IsSingleClicked(ImGuiMouseButton_Left)) return;
    if (interaction_mode == InteractionMode::Edit && edit_mode == Element::None && !active_is_armature) return;

    if (interaction_mode == InteractionMode::Edit && !active_is_armature) {
        const bool toggle = IsKeyDown(ImGuiMod_Shift) || IsKeyDown(ImGuiMod_Ctrl) || IsKeyDown(ImGuiMod_Super);
        Apply(action::selection::ApplyEditElementClick{.MousePx = mouse_px, .Toggle = toggle});
    } else if (interaction_mode == InteractionMode::Object || bone_mode) {
        const auto scaled_pick_radius = std::max(1u, uint32_t(float(ObjectSelectRadiusPx) * std::max(render_scale.x, render_scale.y) + 0.5f));
        const auto active = bone_mode ? FindActiveBone(R) : active_entity;
        const auto hits = ResolveHits(R, RunObjectPick(mouse_px, scaled_pick_radius), bone_mode);
        const auto pick = hits.empty() ? std::optional<SelectionHit>{} : [&] {
            if (ImLengthSqr(CurrentClickPos - PrevClickPos) > 16) return hits.front();
            const auto *bs = R.try_get<BoneSelection>(active);
            auto it = find_if(hits, [&](const SelectionHit &h) { return h.Entity == active && (!h.Part || (bs && bs->Has(*h.Part))); });
            return it != hits.end() && ++it != hits.end() ? *it : hits.front();
        }();
        const bool shift = IsKeyDown(ImGuiMod_Shift);
        if (pick && shift) {
            if (active == pick->Entity && !bone_mode) Apply(action::selection::ToggleSelected{pick->Entity});
            else if (bone_mode) Apply(action::selection::ExtendBoneActive{pick->Entity});
            else Apply(action::selection::ExtendActive{pick->Entity});
        } else if (pick || !shift) {
            if (pick && bone_mode) Apply(action::selection::SelectBone{pick->Entity});
            else if (pick) Apply(action::selection::Select{pick->Entity});
            else deselect_all();
        }
        // Bone sub-part tracking: merge on shift (intentional part accumulation), replace otherwise.
        if (bone_mode && pick && pick->Part) set_bone_sel(pick->Entity, pick->Part, shift);
    }
}

void Scene::InteractOverlay() {
    const rect viewport{ToGlm(GetWindowPos()), ToGlm(GetContentRegionAvail())};
    const bool active_transform = TransformGizmo::IsUsing();
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
    if (!IsMouseDown(ImGuiMouseButton_Left)) OverlayControlsHovered = false;
    const bool any_popup_open = IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);

    { // Transform mode pill buttons (top-left overlay)
        using enum TransformGizmo::Type;
        const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
        const bool has_frozen_selected = R.view<Selected, ScaleLocked>().begin() != R.view<Selected, ScaleLocked>().end();
        const bool edit_transform_locked = interaction_mode == InteractionMode::Edit &&
            any_of(scene_selection::GetSelectedMeshEntities(R), [&](entt::entity mesh_entity) { return scene_selection::HasScaleLockedInstance(R, mesh_entity); });
        const bool transform_enabled = !edit_transform_locked;
        const bool scale_enabled = transform_enabled && !has_frozen_selected;

        auto &transform_type = MGizmo.Config.Type;
        if (!transform_enabled) transform_type = None;
        else if (!scale_enabled && transform_type == Scale) transform_type = Translate;

        const auto start_pos = std::bit_cast<ImVec2>(viewport.pos) + GetWindowContentRegionMin() + ImVec2{overlay_corner_gap, overlay_corner_gap};
        static constexpr float gap{4}; // Gap between select buttons and transform buttons
        const float button_h = overlay_button_style.ButtonSize.y;
        const auto make_button = [](const SvgResource *icon, ImVec2 offset, ImDrawFlags corners, bool enabled, bool active, const char *tooltip = nullptr) {
            return OverlayIconButtonInfo{icon, offset, corners, enabled, active, tooltip};
        };
        const OverlayIconButtonInfo buttons[]{
            make_button(Icons.SelectBox.get(), {0.f, 0.f}, ImDrawFlags_RoundCornersTop, true, transform_type == None && SelectionMode == SelectionMode::Box),
            make_button(Icons.Select.get(), {0.f, button_h}, ImDrawFlags_RoundCornersBottom, true, transform_type == None && SelectionMode == SelectionMode::Click),
            make_button(Icons.Move.get(), {0.f, button_h * 2.f + gap}, ImDrawFlags_RoundCornersTop, transform_enabled, transform_type == Translate),
            make_button(Icons.Rotate.get(), {0.f, button_h * 3.f + gap}, ImDrawFlags_RoundCornersNone, transform_enabled, transform_type == Rotate),
            make_button(Icons.Scale.get(), {0.f, button_h * 4.f + gap}, ImDrawFlags_RoundCornersNone, scale_enabled, transform_type == Scale),
            make_button(Icons.Universal.get(), {0.f, button_h * 5.f + gap}, ImDrawFlags_RoundCornersBottom, transform_enabled, transform_type == Universal),
        };

        if (const auto clicked = DrawOverlayIconButtonGroup("TransformModes", start_pos, buttons, !active_transform, &OverlayControlsHovered, overlay_button_style)) {
            if (*clicked == 0) {
                SelectionMode = SelectionMode::Box;
                transform_type = None;
            } else if (*clicked == 1) {
                SelectionMode = SelectionMode::Click;
                transform_type = None;
            } else {
                transform_type = *clicked == 2 ? Translate :
                    *clicked == 3              ? Rotate :
                    *clicked == 4              ? Scale :
                                                 Universal;
            }
        }
    }

    auto &settings = R.get<SceneSettings>(SceneEntity);

    const auto shading_arrow_w = shading_button_style.ButtonSize.y * 0.55f;
    const float shading_button_w = shading_button_style.ButtonSize.x;
    const float shading_group_width = shading_button_w * 4.f + shading_arrow_w;
    const auto shading_button_h = shading_button_style.ButtonSize.y;

    { // Viewport shading button group + dropdown (top-right overlay)
        const auto start_pos = std::bit_cast<ImVec2>(viewport.pos + vec2{GetWindowContentRegionMax().x - shading_group_width, GetWindowContentRegionMin().y}) + ImVec2{-overlay_corner_gap, overlay_corner_gap};
        const auto make_shading_button = [&](const SvgResource *icon, float x, ImDrawFlags corners, ViewportShadingMode mode, const char *tooltip) {
            return OverlayIconButtonInfo{icon, {x, 0.f}, corners, true, settings.ViewportShading == mode, tooltip};
        };
        const OverlayIconButtonInfo buttons[]{
            make_shading_button(ShadingIcons.Wireframe.get(), 0.f, ImDrawFlags_RoundCornersLeft, ViewportShadingMode::Wireframe, "Wireframe"),
            make_shading_button(ShadingIcons.Solid.get(), shading_button_w, ImDrawFlags_RoundCornersNone, ViewportShadingMode::Solid, "Solid"),
            make_shading_button(ShadingIcons.MaterialPreview.get(), shading_button_w * 2.f, ImDrawFlags_RoundCornersNone, ViewportShadingMode::MaterialPreview, "Material Preview"),
            make_shading_button(ShadingIcons.Rendered.get(), shading_button_w * 3.f, ImDrawFlags_RoundCornersNone, ViewportShadingMode::Rendered, "Rendered"),
        };

        if (const auto clicked = DrawOverlayIconButtonGroup("ViewportShading", start_pos, buttons, !active_transform, &OverlayControlsHovered, shading_button_style)) {
            Apply(action::scene::SetViewportShading{
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

            if (arrow_hovered) OverlayControlsHovered = true;
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
                OverlayControlsHovered = true;
                TextUnformatted("Viewport shading");
                Separator();
                const auto current_mode = settings.ViewportShading;

                const auto render_pbr_controls = [&]<typename T>(const T &lighting, const char *id) {
                    PushID(id);
                    const auto apply_update = [&]<typename Field>(Field PBRViewportLighting::*member, Field v) {
                        Apply(action::UpdateOf<T>(SceneEntity, static_cast<Field T::*>(member), v));
                    };
                    if (Button("Reset")) Apply(action::scene::ResetPbrLighting{.Rendered = std::is_same_v<T, RenderedLighting>});
                    if (bool v = lighting.UseSceneLights; Checkbox("Scene lights", &v)) apply_update(&PBRViewportLighting::UseSceneLights, v);
                    SameLine();
                    if (bool v = lighting.UseSceneWorld; Checkbox("Scene world", &v)) apply_update(&PBRViewportLighting::UseSceneWorld, v);
                    const auto *source_assets = R.try_get<const gltf::SourceAssets>(SceneEntity);
                    const auto *source_ibl = source_assets && source_assets->ImageBasedLight.has_value() ? &*source_assets->ImageBasedLight : nullptr;
                    if (lighting.UseSceneWorld) {
                        // Spec-defined intensity edits the source IBL directly so save round-trips.
                        if (source_ibl) {
                            if (float v = source_ibl->Intensity; SliderFloat("Intensity", &v, 0.f, 2.f, "%.2f"))
                                Apply(action::scene::SetSourceIblIntensity{v});
                        }
                    } else {
                        auto &environments = *Stores.Environments;
                        const auto &current_name = environments.Hdris[environments.ActiveHdriIndex].Name;
                        if (BeginCombo("Environment", current_name.c_str())) {
                            for (uint32_t i = 0; i < environments.Hdris.size(); ++i) {
                                const bool selected = (i == environments.ActiveHdriIndex);
                                if (Selectable(environments.Hdris[i].Name.c_str(), selected)) Apply(action::scene::SetStudioEnvironment{i});
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
                    render_pbr_controls(R.get<const MaterialPreviewLighting>(SceneEntity), "MatPreviewLighting");
                } else if (current_mode == ViewportShadingMode::Rendered) {
                    SeparatorText("Rendered lighting");
                    render_pbr_controls(R.get<const RenderedLighting>(SceneEntity), "RenderedLighting");
                } else if (current_mode == ViewportShadingMode::Solid) {
                    SeparatorText("Solid lighting");
                    auto &lights = Stores.Buffers->GetWorkspaceLights();
                    bool changed = false;
                    if (Button("Reset##Lighting")) {
                        lights = SceneDefaults::WorkspaceLights;
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
                    if (changed) Apply(action::SetTagOf<SubmitDirty>(SceneEntity, true));
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
                                    Apply(action::UpdateOf<&SceneSettings::DebugChannel>(SceneEntity, entry.Value));
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
        const auto group_start = std::bit_cast<ImVec2>(viewport.pos + vec2{GetWindowContentRegionMax().x - shading_group_width - buttons_gap - overlay_group_width, GetWindowContentRegionMin().y}) + ImVec2{-overlay_corner_gap, overlay_corner_gap};

        {
            const OverlayIconButtonInfo icon_button[]{
                {OverlayIcon.get(), {0.f, 0.f}, ImDrawFlags_RoundCornersLeft, true, settings.ShowOverlays, "Toggle overlays"},
            };
            if (const auto clicked = DrawOverlayIconButtonGroup("ViewportOverlays", group_start, icon_button, !active_transform, &OverlayControlsHovered, shading_button_style)) {
                Apply(action::UpdateOf<&SceneSettings::ShowOverlays>(SceneEntity, !settings.ShowOverlays));
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

            if (arrow_hovered) OverlayControlsHovered = true;
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
                OverlayControlsHovered = true;
                TextUnformatted("Viewport overlays");
                Separator();
                ui::Edit f{R, ui::Applier{this}, SceneEntity};
                f.Check<&SceneSettings::ShowGrid>("Grid");
                f.Check<&SceneSettings::ShowExtras>("Extras");
                f.Check<&SceneSettings::ShowBones>("Bones");
                f.Check<&SceneSettings::ShowOrigins>("Origins");
                f.Check<&SceneSettings::ShowOutlineSelected>("Outline selected");
                EndPopup();
            }
            PopStyleVar();
        }
    }

    // Exit "look through" camera view if the user interacts with the orientation gizmo.
    if (!active_transform && OrientationGizmo::IsUsing()) Apply(action::scene::ExitLookThroughCamera{});
    auto &camera = R.get<ViewCamera>(SceneEntity);
    { // Orientation gizmo (interacted before tick so camera animations it initiates begin this frame)
        const float shading_group_height = shading_button_style.ButtonSize.y;
        const auto pos = viewport.pos + vec2{GetWindowContentRegionMax().x - OrientationGizmoSize, GetWindowContentRegionMin().y} + vec2{-overlay_corner_gap, overlay_corner_gap * 2 + shading_group_height};
        OrientationGizmo::Interact(pos, OrientationGizmoSize, camera, !active_transform && !any_popup_open);
    }
    Apply(action::scene::TickViewCamera{});

    const auto selected_view = R.view<const Selected>();
    const auto bone_selected_view = R.view<const BoneSelection>();
    const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    const auto active_entity = FindActiveEntity(R);
    const auto arm_obj = FindArmatureObject(R, active_entity);
    const bool bone_edit_mode = interaction_mode == InteractionMode::Edit && arm_obj != entt::null;
    const bool bone_mode = bone_edit_mode || (interaction_mode == InteractionMode::Pose && arm_obj != entt::null);
    const bool mesh_edit_mode = interaction_mode == InteractionMode::Edit && !bone_edit_mode;

    const auto has_transform_target = [&]() {
        if (bone_mode) return !bone_selected_view.empty();
        if (selected_view.empty()) return false;
        if (!mesh_edit_mode) return true;
        const auto *bits = Stores.Buffers->SelectionBitset.Data();
        for (const auto [e, instance] : R.view<const Instance, const Selected>(entt::exclude<ScaleLocked>).each()) {
            if (const auto *br = R.try_get<const MeshSelectionBitsetRange>(instance.Entity)) {
                if (scene_selection::CountSelected(bits, br->Offset, br->Count) > 0) return true;
            }
        }
        return false;
    }();
    if (has_transform_target) { // Transform gizmo
        // Transform all root selected entities (whose parent is not also selected) around their average position,
        // using the active entity's rotation/scale.
        // Non-root selected entities already follow their parent's transform.
        const auto is_parent_selected = [&](entt::entity e) {
            if (const auto *node = R.try_get<SceneNode>(e)) {
                if (node->Parent == entt::null) return false;
                return bone_mode ? R.all_of<BoneSelection>(node->Parent) : R.all_of<Selected>(node->Parent);
            }
            return false;
        };

        const auto gizmo_active_entity = bone_mode ? FindActiveBone(R) : active_entity;
        const auto active_transform = [&]() -> Transform {
            if (gizmo_active_entity == entt::null) return {};
            const auto &wt = R.get<WorldTransform>(gizmo_active_entity);
            return wt;
        }();

        std::vector<entt::entity> root_selected;
        if (bone_edit_mode) {
            // Edit mode: all selected bones are roots (rest-pose edits don't propagate during drag).
            for (const auto e : bone_selected_view) root_selected.emplace_back(e);
        } else if (bone_mode) {
            // Pose mode: filter out children whose parent is also selected (FK propagates).
            for (const auto e : bone_selected_view) {
                if (!is_parent_selected(e)) root_selected.emplace_back(e);
            }
        } else {
            for (const auto e : selected_view) {
                if (!is_parent_selected(e)) root_selected.emplace_back(e);
            }
        }
        const auto root_count = root_selected.size();
        const auto edit_transform_instances = mesh_edit_mode ?
            scene_selection::ComputePrimaryEditInstances(R, false) :
            std::unordered_map<entt::entity, entt::entity>{};

        vec3 pivot{};
        if (mesh_edit_mode) {
            // Compute world-space centroid of selected vertices once per selected mesh
            // (using a representative selected instance for world transform).
            uint32_t vertex_count = 0;
            for (const auto &[mesh_entity, instance_entity] : edit_transform_instances) {
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                const auto vertex_states = Stores.Meshes->GetVertexStates(mesh.GetStoreId());
                const auto vertices = mesh.GetVerticesSpan();
                const auto &wt = R.get<const WorldTransform>(instance_entity);
                for (uint32_t vi = 0; vi < vertex_states.size(); ++vi) {
                    if ((vertex_states[vi] & ElementStateSelected) == 0u) continue;
                    pivot += wt.P + glm::rotate(wt.R, wt.S * vertices[vi].Position);
                    ++vertex_count;
                }
            }
            if (vertex_count > 0) pivot /= float(vertex_count);
            // Apply pending transform to gizmo position (vertices aren't modified until commit).
            if (const auto *pending = R.try_get<const PendingTransform>(SceneEntity)) {
                pivot += pending->Delta.P;
            }
        } else {
            if (bone_edit_mode) {
                // Bone pivot: contribute head position for selected Root, tail for selected Tip.
                // A fully-selected bone (Root+Tip+Body) contributes both → midpoint.
                vec3 pivot_sum{};
                uint32_t pivot_count = 0;
                for (const auto e : root_selected) {
                    const auto &wt = R.get<WorldTransform>(e);
                    const auto *parts = R.try_get<const BoneSelection>(e);
                    if (!parts || parts->Root) {
                        pivot_sum += wt.P;
                        ++pivot_count;
                    }
                    if (parts && parts->Tip) {
                        const float bl = R.get<BoneDisplayScale>(e).Value;
                        pivot_sum += wt.P + glm::rotate(wt.R, vec3{0, bl, 0});
                        ++pivot_count;
                    }
                }
                pivot = pivot_count > 0 ? pivot_sum / float(pivot_count) : vec3{};
            } else {
                pivot = fold_left(root_selected | transform([&](auto e) { return R.get<WorldTransform>(e).P; }), vec3{}, std::plus{}) / float(root_count);
            }
        }

        const auto start_transform_view = R.view<const StartTransform>();
        const auto gizmo_transform = GizmoTransform{{.P = pivot, .R = active_transform.R, .S = active_transform.S}, MGizmo.Mode};
        auto interact_result = TransformGizmo::Interact(
            gizmo_transform,
            MGizmo.Config, camera, viewport, ToGlm(GetMousePos()) + AccumulatedWrapMouseDelta,
            StartScreenTransform
        );
        if (interact_result) {
            const auto &[ts, td] = *interact_result;
            if (start_transform_view.empty()) {
                action::scene::BeginGizmoDrag begin;
                if (mesh_edit_mode) {
                    for (const auto &[_, instance_entity] : edit_transform_instances) {
                        begin.Starts.emplace_back(instance_entity, StartTransform{.T = R.get<const Transform>(instance_entity), .ParentDelta = {}});
                    }
                } else {
                    // Capture parent transform at drag start so world->local conversion uses a stable reference frame throughout the interaction.
                    for (const auto e : root_selected) {
                        const auto &wt = R.get<WorldTransform>(e);
                        begin.Starts.emplace_back(e, StartTransform{wt, ToTransform(GetParentDelta(R, e))});
                        if (bone_edit_mode) {
                            if (const auto *ds = R.try_get<BoneDisplayScale>(e)) begin.StartBoneLengths.emplace_back(e, ds->Value);
                        }
                    }
                }
                Apply(std::move(begin));
            }
            if (mesh_edit_mode) {
                // Mesh Edit mode: store pending transform for shader-based preview.
                // Actual vertex positions are only modified on commit.
                Apply(action::scene::UpdateGizmoMeshEditPending{std::make_unique<PendingTransform>(PendingTransform{ts.P, ts.R, td})});
            } else {
                // Object mode: apply transform to entity components immediately during drag.
                // Compute new world result, then convert to local for parented entities.
                action::scene::UpdateGizmoDragLocals update;
                const auto make_local = [&](entt::entity e, const Transform &world, const Transform &pd) {
                    Transform local;
                    local.P = glm::conjugate(pd.R) * ((world.P - pd.P) / pd.S);
                    local.R = glm::conjugate(pd.R) * world.R;
                    local.S = R.all_of<ScaleLocked>(e) ? R.get<const Transform>(e).S : world.S / pd.S;
                    update.Locals.emplace_back(e, local);
                };

                const auto r = ts.R, rT = glm::conjugate(r);
                for (const auto &[e, ts_e_comp] : start_transform_view.each()) {
                    const auto &ts_e = ts_e_comp.T; // world transform at start

                    // Head/tail-only bone transform: stretch/rotate bone instead of moving it rigidly.
                    if (bone_edit_mode) {
                        // Use current parent WT for world→local (parent may have been moved earlier in this loop).
                        const auto pd = ToTransform(GetParentDelta(R, e));
                        const auto *sbl = R.try_get<StartBoneLength>(e);
                        const auto *parts = R.try_get<BoneSelection>(e);
                        if (sbl && parts) {
                            const bool tip_only = parts->Tip && !parts->Root && !parts->Body;
                            const bool root_only = parts->Root && !parts->Tip && !parts->Body;
                            if (tip_only || root_only) {
                                const auto transform_point = [&](vec3 p) { return td.P + ts.P + glm::rotate(td.R, r * (rT * (p - ts.P) * td.S)); };

                                const float bone_length = sbl->Value;
                                const auto start_head = ts_e.P;
                                const auto start_tail = start_head + glm::rotate(ts_e.R, vec3{0, bone_length, 0});
                                const auto new_head = tip_only ? start_head : transform_point(start_head);
                                const auto new_tail = root_only ? start_tail : transform_point(start_tail);
                                const auto dir = new_tail - new_head;
                                const auto new_length = glm::length(dir);
                                constexpr float eps = 1e-6f;
                                const auto new_world_rot = new_length > eps ? glm::rotation(glm::normalize(glm::rotate(ts_e.R, vec3{0, 1, 0})), dir / new_length) * ts_e.R : ts_e.R;
                                update.BoneDisplayScales.emplace_back(e, std::max(new_length, eps));
                                make_local(e, {new_head, new_world_rot, ts_e.S}, pd);
                                continue;
                            }
                        }

                        // Full bone transform in bone edit mode.
                        const auto offset = ts_e.P - ts.P;
                        make_local(e, {td.P + ts.P + glm::rotate(td.R, r * (rT * offset * td.S)), glm::normalize(td.R * ts_e.R), ts_e.S}, pd);
                        continue;
                    }

                    // Object mode / non-bone transform.
                    const auto &pd = ts_e_comp.ParentDelta;
                    const bool frozen = R.all_of<ScaleLocked>(e);
                    const auto offset = ts_e.P - ts.P;
                    make_local(e, {td.P + ts.P + glm::rotate(td.R, frozen ? offset : r * (rT * offset * td.S)), glm::normalize(td.R * ts_e.R), frozen ? ts_e.S : td.S * ts_e.S}, pd);
                }
                Apply(std::move(update));
            }
        } else if (!start_transform_view.empty()) {
            Apply(action::scene::EndGizmoDrag{});
        }

        // Store gizmo render transform for DrawOverlay.
        GizmoRenderTransform = gizmo_transform;
        if (interact_result) GizmoRenderTransform->P = interact_result->Start.P + interact_result->Delta.P;
    }

    StartScreenTransform = {};
}

void Scene::DrawOverlay() {
    const rect viewport{ToGlm(GetWindowPos()), ToGlm(GetContentRegionAvail())};
    const auto &axes = R.get<const colors::AxesArray>(SceneEntity);
    const auto &camera = R.get<const ViewCamera>(SceneEntity);

    OrientationGizmo::Render(axes);
    if (GizmoRenderTransform) {
        TransformGizmo::Render(*GizmoRenderTransform, MGizmo.Config.Type, camera, viewport, axes);
        GizmoRenderTransform.reset();
    }

    const auto &settings = R.get<const SceneSettings>(SceneEntity);
    // Draw origin dots for active/selected entities
    if (settings.ShowOverlays && settings.ShowOrigins && (!R.storage<Selected>().empty() || !R.storage<Active>().empty())) {
        const auto &theme = R.get<const ViewportTheme>(SceneEntity);
        const auto vp = camera.Projection(viewport.size.x / viewport.size.y) * camera.View();
        auto draw_dot = [&](vec3 pos, bool is_active) {
            const auto p_cs = vp * vec4{pos, 1.f};
            if (p_cs.w <= 0) return; // Behind camera

            const auto p_ndc = vec3{p_cs} / p_cs.w;
            const auto p_uv = vec2{p_ndc.x + 1, 1 - p_ndc.y} * 0.5f;
            const auto p_px = std::bit_cast<ImVec2>(viewport.pos + p_uv * viewport.size);
            auto &dl = *GetWindowDrawList();
            dl.AddCircleFilled(p_px, 3.5f, colors::RgbToU32(is_active ? theme.Colors.ObjectActive : theme.Colors.ObjectSelected), 10);
            dl.AddCircle(p_px, 3.5f, IM_COL32(0, 0, 0, 255), 10, 1.f);
        };
        // Top-level objects: draw dot at own position.
        for (const auto [e, wt] : R.view<const WorldTransform>(entt::exclude<SubElementOf>).each()) {
            if (!R.any_of<Active, Selected>(e)) continue;
            draw_dot(wt.P, R.all_of<Active>(e));
        }
    }

    if (BoxSelectStart.has_value() && BoxSelectEnd.has_value()) {
        auto &dl = *GetWindowDrawList();
        const auto box_min = glm::min(*BoxSelectStart, *BoxSelectEnd);
        const auto box_max = glm::max(*BoxSelectStart, *BoxSelectEnd);
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
    if (const auto look_through_entity = LookThroughCameraEntity(); look_through_entity != entt::null && !camera.IsAnimating()) {
        if (const auto *cd = R.try_get<Camera>(look_through_entity)) {
            const float cam_aspect = AspectRatio(*cd);
            const auto frame_size = vec2{viewport.size.y * cam_aspect, viewport.size.y} * LookThroughFrameRatio(cam_aspect, viewport.size.x / viewport.size.y);
            const vec2 vp_center = viewport.pos + viewport.size * 0.5f;
            const vec2 fmin = vp_center - frame_size * 0.5f, fmax = vp_center + frame_size * 0.5f;
            const auto vmin = viewport.pos, vmax = viewport.pos + viewport.size;

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

void Scene::RenderEntityControls(entt::entity active_entity) {
    if (active_entity == entt::null) {
        TextUnformatted("Active object: None");
        return;
    }

    PushID("EntityControls");
    Text("Active entity: %s", GetName(R, active_entity).c_str());
    Indent();

    if (const auto *node = R.try_get<SceneNode>(active_entity)) {
        if (auto parent_entity = node->Parent; parent_entity != entt::null) {
            AlignTextToFramePadding();
            Text("Parent: %s", GetName(R, parent_entity).c_str());
        }
    }

    if (const auto *instance = R.try_get<Instance>(active_entity)) {
        Text("Instance of: %s", GetName(R, instance->Entity).c_str());
    }
    if (const auto *armature_modifier = R.try_get<ArmatureModifier>(active_entity)) {
        Text("Armature data: %s", GetName(R, armature_modifier->ArmatureEntity).c_str());
        if (armature_modifier->ArmatureObjectEntity != entt::null) {
            Text("Armature object: %s", GetName(R, armature_modifier->ArmatureObjectEntity).c_str());
        }
    }
    if (const auto *bone_attachment = R.try_get<BoneAttachment>(active_entity)) {
        Text("Attached bone ID: %u", bone_attachment->Bone);
    }
    const auto object_type = R.all_of<ObjectKind>(active_entity) ? R.get<const ObjectKind>(active_entity).Value : ObjectType::Empty;
    Text("Object type: %s", ObjectTypeName(object_type).data());
    const auto active_bone_entity = FindActiveBone(R);
    const auto *active_instance = R.try_get<Instance>(active_entity);
    const bool is_mesh_instance = active_instance && R.all_of<Mesh>(active_instance->Entity);
    if (is_mesh_instance) {
        const auto active_mesh_entity = active_instance->Entity;
        const auto &active_mesh = R.get<const Mesh>(active_mesh_entity);
        TextUnformatted(
            std::format("Vertices | Edges | Faces: {:L} | {:L} | {:L}", active_mesh.VertexCount(), active_mesh.EdgeCount(), active_mesh.FaceCount()).c_str()
        );
    } else if (const auto *armature_object = R.try_get<ArmatureObject>(active_entity)) {
        const auto &armature = R.get<const Armature>(armature_object->Entity);
        Text("Bones: %zu", armature.Bones.size());
    }
    Unindent();
    const bool is_bone_edit = R.get<const SceneInteraction>(SceneEntity).Mode == InteractionMode::Edit && active_bone_entity != entt::null && R.all_of<BoneDisplayScale>(active_bone_entity);
    if (CollapsingHeader("Transform")) {
        if (is_bone_edit) {
            const auto &wt = R.get<WorldTransform>(active_bone_entity);
            const float bone_length = R.get<BoneDisplayScale>(active_bone_entity).Value;

            vec3 head = wt.P;
            vec3 tail = head + glm::rotate(wt.R, vec3{0, bone_length, 0});
            vec3 dir;
            float roll;
            BoneMat3ToVecRoll(glm::mat3_cast(wt.R), dir, roll);
            float roll_deg = glm::degrees(roll);
            float length = bone_length;

            bool changed = DragFloat3("Head", &head[0], 0.01f);
            changed |= DragFloat3("Tail", &tail[0], 0.01f);
            if (DragFloat("Roll", &roll_deg, 1.f)) {
                roll = glm::radians(roll_deg);
                changed = true;
            }
            if (DragFloat("Length", &length, 0.01f, 0.001f, 0.f)) {
                tail = head + glm::normalize(tail - head) * std::max(length, 1e-4f);
                changed = true;
            }

            if (changed) {
                const auto new_dir = tail - head;
                if (const auto new_length = glm::length(new_dir); new_length > 1e-6f) {
                    const auto new_rot = glm::quat_cast(BoneVecRollToMat3(new_dir, roll));
                    const auto pd = ToTransform(GetParentDelta(R, active_bone_entity));
                    Apply(action::bone::SetEditHeadTailRoll{
                        .LocalP = glm::conjugate(pd.R) * ((head - pd.P) / pd.S),
                        .LocalR = glm::conjugate(pd.R) * new_rot,
                        .DisplayScale = new_length,
                    });
                }
            }
        } else {
            // Standard PRS transform editor (objects, pose mode bones).
            // In Pose mode, edit the active bone rather than the armature.
            const bool is_pose_bone = R.get<const SceneInteraction>(SceneEntity).Mode == InteractionMode::Pose && active_bone_entity != entt::null;
            const auto transform_entity = is_pose_bone ? active_bone_entity : active_entity;
            ui::Edit transform_edit{R, ui::Applier{this}, transform_entity};
            transform_edit.Drag<&Transform::P>("Position", 0.01f);
            // Rotation editor (RotationUiVariant is reactively created; may not exist yet on the first frame)
            if (const auto *rotation_ui_ptr = R.try_get<const RotationUiVariant>(transform_entity)) {
                int mode_i = rotation_ui_ptr->index();
                const char *modes[]{"Quat (WXYZ)", "XYZ Euler", "Axis Angle"};
                if (Combo("Rotation mode", &mode_i, modes, IM_ARRAYSIZE(modes)))
                    Apply(action::scene::SetRotationUiMode{mode_i});
                auto ui_local = *rotation_ui_ptr;
                std::visit(
                    overloaded{
                        [&](RotationQuat &v) {
                            if (DragFloat4("Rotation (quat WXYZ)", &v.Value[0], 0.01f))
                                Apply(action::scene::SetTransformRotationFromUi{glm::normalize(v.Value), ui_local});
                        },
                        [&](RotationEuler &v) {
                            if (DragFloat3("Rotation (XYZ Euler, deg)", &v.Value[0], 1.f)) {
                                const auto rads = glm::radians(v.Value);
                                Apply(action::scene::SetTransformRotationFromUi{
                                    glm::normalize(glm::quat_cast(glm::eulerAngleXYZ(rads.x, rads.y, rads.z))),
                                    ui_local,
                                });
                            }
                        },
                        [&](RotationAxisAngle &v) {
                            bool changed = DragFloat3("Rotation axis (XYZ)", &v.Value[0], 0.01f);
                            changed |= DragFloat("Angle (deg)", &v.Value.w, 1.f);
                            if (changed) {
                                const auto axis = glm::normalize(vec3{v.Value});
                                const auto angle = glm::radians(v.Value.w);
                                Apply(action::scene::SetTransformRotationFromUi{
                                    glm::normalize(quat{std::cos(angle / 2), axis * std::sin(angle / 2)}),
                                    ui_local,
                                });
                            }
                        },
                    },
                    ui_local
                );
            }

            const bool frozen = R.all_of<ScaleLocked>(transform_entity);
            if (frozen) BeginDisabled();
            transform_edit.Drag<&Transform::S>(std::format("Scale{}", frozen ? " (frozen)" : "").c_str(), 0.01f, 0.01f, 10);
            if (frozen) EndDisabled();
        }
        Spacing();
        {
            AlignTextToFramePadding();
            Text("Mode:");
            SameLine();
            using enum TransformGizmo::Mode;
            auto &mode = MGizmo.Mode;
            if (RadioButton("Local", mode == Local)) mode = Local;
            SameLine();
            if (RadioButton("World", mode == World)) mode = World;
            Spacing();
            Checkbox("Snap", &MGizmo.Config.Snap);
            if (MGizmo.Config.Snap) {
                SameLine();
                // todo link/unlink snap values
                DragFloat3("Snap", &MGizmo.Config.SnapValue.x, 1.f, 0.01f, 100.f);
            }
        }
        Spacing();
        if (TreeNode("Debug")) {
            if (const auto label = TransformGizmo::ToString(); label != "") {
                Text("%s op: %s", TransformGizmo::IsUsing() ? "Active" : "Hovered", label.data());
            } else {
                TextUnformatted("Not hovering");
            }
            TreePop();
        }
        if (TreeNode("World transform")) {
            const auto &wt = R.get<WorldTransform>(active_entity);
            Text("Position: %.3f, %.3f, %.3f", wt.P.x, wt.P.y, wt.P.z);
            Text("Rotation: %.3f, %.3f, %.3f, %.3f", wt.R.x, wt.R.y, wt.R.z, wt.R.w);
            Text("Scale: %.3f, %.3f, %.3f", wt.S.x, wt.S.y, wt.S.z);
            TreePop();
        }
    }
    if (active_bone_entity != entt::null && CollapsingHeader("Bone Constraints")) {
        PushID("BoneConstraints");
        const auto *constraints = R.try_get<const BoneConstraints>(active_bone_entity);
        const size_t stack_size = constraints ? constraints->Stack.size() : 0;
        std::optional<uint32_t> delete_index;
        for (size_t i = 0; i < stack_size; ++i) {
            PushID(int(i));
            const auto &c = constraints->Stack[i];
            const char *type_label = std::visit([]<typename T>(const T &) {
                if constexpr (std::is_same_v<T, CopyTransformsData>) return "Copy Transforms";
                else if constexpr (std::is_same_v<T, ChildOfData>) return "Child Of";
                else return "?";
            },
                                                c.Data);
            const bool expanded = TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanTextWidth, "%s", type_label);
            SameLine();
            if (SmallButton("X")) delete_index = uint32_t(i);
            if (expanded) {
                const auto *cur_name = c.TargetEntity != entt::null && R.valid(c.TargetEntity) ? R.try_get<const Name>(c.TargetEntity) : nullptr;
                const std::string preview = c.TargetEntity == entt::null ? "None" :
                    cur_name && !cur_name->Value.empty()                 ? cur_name->Value :
                                                                           IdString(c.TargetEntity);
                if (BeginCombo("Target", preview.c_str())) {
                    if (Selectable("None", c.TargetEntity == entt::null))
                        Apply(action::bone::SetConstraintTarget{uint32_t(i), entt::null});
                    for (auto [te, kind, name] : R.view<const ObjectKind, const Name>().each()) {
                        if (R.any_of<BoneIndex, BoneSubPartOf, BoneJoint, SubElementOf>(te)) continue;
                        const std::string label = name.Value.empty() ? IdString(te) : name.Value;
                        if (Selectable(label.c_str(), te == c.TargetEntity))
                            Apply(action::bone::SetConstraintTarget{uint32_t(i), te});
                    }
                    EndCombo();
                }
                if (std::holds_alternative<ChildOfData>(c.Data)) {
                    if (Button("Set Inverse") && c.TargetEntity != entt::null && R.valid(c.TargetEntity)) {
                        // Bake inverse(target_world) * bone_world so current relative pose becomes the new rest.
                        const auto *twt = R.try_get<const WorldTransform>(c.TargetEntity);
                        const auto *bwt = R.try_get<const WorldTransform>(active_bone_entity);
                        if (twt && bwt) {
                            const mat4 inv = glm::inverse(ToMatrix(*twt)) * ToMatrix(*bwt);
                            Apply(action::bone::SetConstraintChildOfInverse{uint32_t(i), std::make_unique<mat4>(inv)});
                        }
                    }
                    SameLine();
                    if (Button("Clear Inverse"))
                        Apply(action::bone::SetConstraintChildOfInverse{uint32_t(i), std::make_unique<mat4>(I4)});
                }
                if (float influence = c.Influence; SliderFloat("Influence", &influence, 0.f, 1.f))
                    Apply(action::bone::SetConstraintInfluence{uint32_t(i), influence});
                TreePop();
            }
            PopID();
        }
        if (delete_index) Apply(action::bone::DeleteConstraint{*delete_index});
        if (Button("Add Copy Transforms")) Apply(action::bone::AddConstraint{action::bone::BoneConstraintKind::CopyTransforms});
        SameLine();
        if (Button("Add Child Of")) Apply(action::bone::AddConstraint{action::bone::BoneConstraintKind::ChildOf});
        PopID();
    }
    if (is_mesh_instance) {
        const auto active_mesh_entity = active_instance->Entity;
        if (auto *prim_shape = R.try_get<PrimitiveShape>(active_mesh_entity)) {
            const bool frozen = scene_selection::HasScaleLockedInstance(R, active_mesh_entity);
            if (frozen) BeginDisabled();
            if (const auto update_label = std::format("Edit primitive{}", frozen ? " (frozen)" : "");
                CollapsingHeader(update_label.c_str()) && !frozen) {
                if (auto primitive_mesh = PrimitiveEditor(*prim_shape)) {
                    Apply(action::object::ReplaceMesh{std::make_unique<MeshData>(std::move(*primitive_mesh))});
                }
            }
            if (frozen) EndDisabled();
        }

        if (CollapsingHeader("Material")) {
            const auto &active_mesh = R.get<const Mesh>(active_mesh_entity);
            auto &material_store = R.get<MaterialStore>(SceneEntity);
            auto &texture_store = *Stores.Textures;
            std::span<const uint32_t> primitive_materials = Stores.Meshes->GetPrimitiveMaterialIndices(active_mesh.GetStoreId());
            const auto material_count = Stores.Buffers->Materials.Count();
            const auto material_name = [&](uint32_t index) {
                if (index < material_store.Names.size() && !material_store.Names[index].empty()) return std::string{material_store.Names[index]};
                return std::format("Material{}", index);
            };
            if (primitive_materials.empty()) {
                TextUnformatted("No material slots on this mesh.");
            } else if (material_count == 0) {
                TextUnformatted("No materials.");
            } else {
                const uint32_t max_primitive = primitive_materials.size() - 1;
                const auto *existing_slot = R.try_get<const MeshMaterialSlotSelection>(active_mesh_entity);
                uint32_t slot_primitive = existing_slot ? existing_slot->PrimitiveIndex : 0u;
                if (!existing_slot || slot_primitive > max_primitive) {
                    slot_primitive = std::min(slot_primitive, max_primitive);
                    Apply(action::Replace<MeshMaterialSlotSelection>{active_mesh_entity, {slot_primitive}});
                }

                BeginChild("MaterialSlots", ImVec2(0, 110), true);
                for (uint32_t primitive_index = 0; primitive_index < primitive_materials.size(); ++primitive_index) {
                    const uint32_t material_index = std::min(primitive_materials[primitive_index], material_count - 1);
                    if (const auto label = std::format("Slot {:L}: {}", primitive_index, material_name(material_index));
                        Selectable(label.c_str(), slot_primitive == primitive_index) && slot_primitive != primitive_index) {
                        Apply(action::Replace<MeshMaterialSlotSelection>{active_mesh_entity, {primitive_index}});
                        slot_primitive = primitive_index;
                    }
                }
                EndChild();

                const auto *pending_assignment = R.try_get<const MeshMaterialAssignment>(active_mesh_entity);
                uint32_t material_index = pending_assignment && pending_assignment->PrimitiveIndex == slot_primitive ?
                    pending_assignment->MaterialIndex :
                    primitive_materials[slot_primitive];
                material_index = std::min(material_index, material_count - 1);
                if (const auto assigned_material_name = material_name(material_index);
                    BeginCombo("Assigned material", assigned_material_name.c_str())) {
                    for (uint32_t i = 0; i < material_count; ++i) {
                        if (const auto option_name = material_name(i);
                            Selectable(option_name.c_str(), material_index == i)) {
                            Apply(action::Replace<MeshMaterialAssignment>{active_mesh_entity, {slot_primitive, i}});
                            material_index = i;
                        }
                    }
                    EndCombo();
                }

                auto &material = Stores.Buffers->Materials.Get(material_index);
                const auto edit_texture_slot = [&](const char *label, uint32_t &slot) {
                    std::string preview = "None";
                    bool has_match = false;
                    for (const auto &texture : texture_store.Textures) {
                        if (texture.SamplerSlot != slot) continue;
                        preview = texture.Name;
                        has_match = true;
                        break;
                    }
                    if (!has_match && slot != InvalidSlot) preview = std::format("Missing slot {}", slot);

                    bool changed = false;
                    if (BeginCombo(label, preview.c_str())) {
                        if (Selectable("None", slot == InvalidSlot)) {
                            slot = InvalidSlot;
                            changed = true;
                        }
                        for (const auto &texture : texture_store.Textures) {
                            if (Selectable(texture.Name.c_str(), slot == texture.SamplerSlot)) {
                                slot = texture.SamplerSlot;
                                changed = true;
                            }
                        }
                        EndCombo();
                    }
                    return changed;
                };
                const auto edit_uv_transform = [&](const char *offset_label, const char *scale_label, const char *rotation_label, vec2 &offset, vec2 &scale, float &rotation) {
                    bool changed = false;
                    changed |= DragFloat2(offset_label, &offset.x, 0.01f);
                    changed |= DragFloat2(scale_label, &scale.x, 0.01f);
                    changed |= DragFloat(rotation_label, &rotation, 0.01f);
                    return changed;
                };
                const auto edit_texture_info = [&](const char *label, TextureInfo &tex) {
                    bool changed = false;
                    changed |= edit_texture_slot(std::format("{} texture", label).c_str(), tex.Slot);
                    changed |= MeshEditor::SliderUInt(std::format("{} UV set", label).c_str(), &tex.TexCoord, 0u, 3u);
                    changed |= edit_uv_transform(
                        std::format("{} UV offset", label).c_str(),
                        std::format("{} UV scale", label).c_str(),
                        std::format("{} UV rotation", label).c_str(),
                        tex.UvOffset, tex.UvScale, tex.UvRotation
                    );
                    return changed;
                };
                bool material_changed = false;
                material_changed |= ColorEdit4("Base color", &material.BaseColorFactor.x);
                material_changed |= SliderFloat("Metallic", &material.MetallicFactor, 0.f, 1.f);
                material_changed |= SliderFloat("Roughness", &material.RoughnessFactor, 0.f, 1.f);
                material_changed |= edit_texture_info("Base color", material.BaseColorTexture);
                material_changed |= edit_texture_info("Metallic-roughness", material.MetallicRoughnessTexture);
                material_changed |= edit_texture_info("Normal", material.NormalTexture);
                material_changed |= SliderFloat("Normal scale", &material.NormalScale, -2.f, 2.f);
                material_changed |= edit_texture_info("Occlusion", material.OcclusionTexture);
                material_changed |= SliderFloat("Occlusion strength", &material.OcclusionStrength, 0.f, 1.f);
                material_changed |= ColorEdit3("Emissive", &material.EmissiveFactor.x);
                material_changed |= edit_texture_info("Emissive", material.EmissiveTexture);

                static constexpr std::array alpha_mode_labels{"Opaque", "Mask", "Blend"};
                int alpha_mode = std::clamp<int>(int(material.AlphaMode), 0, int(alpha_mode_labels.size() - 1));
                if (Combo("Alpha mode", &alpha_mode, alpha_mode_labels.data(), int(alpha_mode_labels.size()))) {
                    material.AlphaMode = MaterialAlphaMode(alpha_mode);
                    material_changed = true;
                }
                if (material.AlphaMode == MaterialAlphaMode::Mask) {
                    material_changed |= SliderFloat("Alpha cutoff", &material.AlphaCutoff, 0.f, 1.f);
                }
                bool double_sided = material.DoubleSided != 0u;
                if (Checkbox("Double sided", &double_sided)) {
                    material.DoubleSided = double_sided ? 1u : 0u;
                    material_changed = true;
                }

                // IOR — always visible; affects Fresnel reflectance even for non-transmissive dielectrics.
                material_changed |= SliderFloat("IOR", &material.Ior, 1.0f, 3.0f, "%.3f");

                auto pbr_features_mask = R.all_of<PbrMeshFeatures>(active_mesh_entity) ? R.get<const PbrMeshFeatures>(active_mesh_entity).Mask : 0u;
                bool pbr_features_changed = false;
                const auto feature_toggle = [&](const char *label, PbrFeature feature) {
                    bool enabled = HasFeature(pbr_features_mask, feature);
                    if (Checkbox(label, &enabled)) {
                        if (enabled) pbr_features_mask |= feature;
                        else pbr_features_mask &= ~uint32_t(feature);
                        pbr_features_changed = true;
                    }
                    return enabled;
                };

                // Transmission
                if (feature_toggle("Transmission", PbrFeature::Transmission)) {
                    SeparatorText("Transmission");
                    material_changed |= SliderFloat("Transmission factor", &material.Transmission.Factor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Transmission", material.Transmission.Texture);
                    material_changed |= SliderFloat("Dispersion", &material.Dispersion, 0.f, 1.f);
                    // Volume (only meaningful with transmission)
                    material_changed |= SliderFloat("Thickness", &material.Volume.ThicknessFactor, 0.f, 10.f);
                    material_changed |= edit_texture_info("Thickness", material.Volume.ThicknessTexture);
                    material_changed |= ColorEdit3("Attenuation color", &material.Volume.AttenuationColor.x);
                    material_changed |= DragFloat("Attenuation distance", &material.Volume.AttenuationDistance, 0.01f, 0.f, 0.f, material.Volume.AttenuationDistance <= 0.f ? "Infinite" : "%.3f m");
                }

                // Diffuse transmission
                if (feature_toggle("Diffuse transmission", PbrFeature::DiffuseTrans)) {
                    SeparatorText("Diffuse transmission");
                    material_changed |= SliderFloat("Diffuse transmission factor", &material.DiffuseTransmission.Factor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Diffuse transmission", material.DiffuseTransmission.Texture);
                    material_changed |= ColorEdit3("Diffuse transmission color", &material.DiffuseTransmission.ColorFactor.x);
                    material_changed |= edit_texture_info("Diffuse transmission color", material.DiffuseTransmission.ColorTexture);
                }

                // Clearcoat
                if (feature_toggle("Clearcoat", PbrFeature::Clearcoat)) {
                    SeparatorText("Clearcoat");
                    material_changed |= SliderFloat("Clearcoat factor", &material.Clearcoat.Factor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Clearcoat", material.Clearcoat.Texture);
                    material_changed |= SliderFloat("Clearcoat roughness", &material.Clearcoat.RoughnessFactor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Clearcoat roughness", material.Clearcoat.RoughnessTexture);
                    material_changed |= edit_texture_info("Clearcoat normal", material.Clearcoat.NormalTexture);
                    material_changed |= SliderFloat("Clearcoat normal scale", &material.Clearcoat.NormalScale, -2.f, 2.f);
                }

                // Anisotropy
                if (feature_toggle("Anisotropy", PbrFeature::Anisotropy)) {
                    SeparatorText("Anisotropy");
                    material_changed |= SliderFloat("Anisotropy strength", &material.Anisotropy.Strength, 0.f, 1.f);
                    material_changed |= SliderFloat("Anisotropy rotation", &material.Anisotropy.Rotation, 0.f, 6.2832f, "%.3f rad");
                    material_changed |= edit_texture_info("Anisotropy", material.Anisotropy.Texture);
                }

                // Sheen
                if (feature_toggle("Sheen", PbrFeature::Sheen)) {
                    SeparatorText("Sheen");
                    material_changed |= ColorEdit3("Sheen color", &material.Sheen.ColorFactor.x);
                    material_changed |= edit_texture_info("Sheen color", material.Sheen.ColorTexture);
                    material_changed |= SliderFloat("Sheen roughness", &material.Sheen.RoughnessFactor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Sheen roughness", material.Sheen.RoughnessTexture);
                }

                // Iridescence
                if (feature_toggle("Iridescence", PbrFeature::Iridescence)) {
                    SeparatorText("Iridescence");
                    material_changed |= SliderFloat("Iridescence factor", &material.Iridescence.Factor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Iridescence", material.Iridescence.Texture);
                    material_changed |= SliderFloat("Iridescence IOR", &material.Iridescence.Ior, 1.0f, 5.0f);
                    material_changed |= SliderFloat("Thickness min", &material.Iridescence.ThicknessMinimum, 0.f, 1000.f, "%.0f nm");
                    material_changed |= SliderFloat("Thickness max", &material.Iridescence.ThicknessMaximum, 1.f, 1000.f, "%.0f nm");
                    material_changed |= edit_texture_info("Iridescence thickness", material.Iridescence.ThicknessTexture);
                }

                if (pbr_features_changed) Apply(action::scene::SetPbrMeshFeaturesMask{pbr_features_mask});
                if (material_changed) Apply(action::Replace<MaterialDirty>{SceneEntity, {material_index}});
            }
        }
    }
    if (const auto *cd = R.try_get<const Camera>(active_entity)) {
        if (CollapsingHeader("Camera")) {
            // Use the camera's distance from world origin as the conversion distance.
            const float distance = std::max(glm::length(R.get<WorldTransform>(active_entity).P), 1.f);
            auto edited = *cd;
            if (RenderCameraLensEditor(edited, distance)) Apply(action::ReplaceActive<Camera>{edited});
            Separator();
            if (LookThroughCameraEntity() == active_entity) {
                if (Button("Exit camera view")) Apply(action::scene::ExitLookThroughCamera{});
            } else {
                if (Button("Look through")) {
                    Apply(action::scene::EnterLookThroughCamera{});
                    Apply(action::scene::AnimateToCamera{});
                }
            }
        }
    }
    if (R.all_of<LightIndex>(active_entity) &&
        CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto light = Stores.Buffers->Lights.Get(R.get<const LightIndex>(active_entity).Value);
        bool changed{false}, wireframe_changed{false};

        const char *type_names[]{"Directional", "Point", "Spot"};
        if (int type_i = int(light.Type); Combo("Type", &type_i, type_names, IM_ARRAYSIZE(type_names))) {
            auto next = SceneDefaults::MakePunctualLight(PunctualLightType(type_i));
            next.TransformSlotOffset = light.TransformSlotOffset;
            next.Color = light.Color;
            next.Intensity = light.Intensity;
            light = next;
            changed = wireframe_changed = true;
        }
        changed |= ColorEdit3("Color", &light.Color.x);
        changed |= SliderFloat("Intensity", &light.Intensity, 0.f, 1000.f, "%.2f");
        if (light.Type == PunctualLightType::Point || light.Type == PunctualLightType::Spot) {
            bool infinite_range = light.Range <= 0.f;
            if (Checkbox("Infinite range", &infinite_range)) {
                light.Range = infinite_range ? 0.f : 100.f;
                changed = wireframe_changed = true;
            }
            if (!infinite_range && SliderFloat("Range", &light.Range, 0.01f, 1000.f, "%.2f")) {
                changed = wireframe_changed = true;
            }
        }
        if (light.Type == PunctualLightType::Spot) {
            float outer_deg = std::clamp(glm::degrees(AngleFromCos(light.OuterConeCos)), 0.f, 90.f);
            const float inner_deg = std::clamp(glm::degrees(AngleFromCos(light.InnerConeCos)), 0.f, outer_deg);
            float blend = outer_deg > 1e-4f ? std::clamp(1.f - inner_deg / outer_deg, 0.f, 1.f) : 0.f;
            const bool size_changed = SliderFloat("Size", &outer_deg, 0.f, 90.f, "%.1f deg");
            const bool blend_changed = SliderFloat("Blend", &blend, 0.f, 1.f, "%.2f");
            if (size_changed || blend_changed) {
                const float outer_rad = glm::radians(std::clamp(outer_deg, 0.f, 90.f));
                light.OuterConeCos = std::cos(outer_rad);
                light.InnerConeCos = std::cos(outer_rad * (1.f - std::clamp(blend, 0.f, 1.f)));
                changed = wireframe_changed = true;
            }
        }
        if (changed) {
            Apply(action::ReplaceActive<PunctualLight>{light});
            Apply(action::SetTagOf<SubmitDirty>(true));
        }
        if (wireframe_changed) Apply(action::SetTagOf<LightWireframeDirty>(true));
    }
    // Audio controls (mesh instance = sound object or eligible to become one; microphone)
    if (const auto *instance = R.try_get<Instance>(active_entity); instance && R.all_of<Mesh>(instance->Entity)) {
        const bool has_sound = R.all_of<SoundVerticesModel>(active_entity);
        if (CollapsingHeader("Audio", has_sound ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
            DrawObjectAudioControls(R, SceneEntity, active_entity, GetMeshEntity(active_entity), Stores.Buffers->SelectionBitset.Data(), [this](action::audio::Action a) { Apply(std::move(a)); });
            if (const auto *active_mic = R.try_get<RealImpactActiveMicrophone>(active_entity)) {
                SeparatorText("Microphone");
                Text("Active: %s", GetName(R, active_mic->Entity).c_str());
                if (Button("Select microphone entity")) Select(active_mic->Entity);
            }
        }
    } else if (const auto *mic = R.try_get<const RealImpactMicrophone>(active_entity)) {
        if (CollapsingHeader("Audio", ImGuiTreeNodeFlags_DefaultOpen)) {
            Text("Microphone index: %u", mic->Index);
            // Target = sound object currently bound to this mic, else first sound object with a dataset Path.
            auto target = entt::entity{entt::null};
            for (const auto &[e, active] : R.view<const RealImpactActiveMicrophone>().each()) {
                if (active.Entity == active_entity) {
                    target = e;
                    break;
                }
            }
            if (target == entt::null) {
                for (auto [e, _, inst] : R.view<SoundVerticesModel, Instance>().each()) {
                    if (R.all_of<Path>(inst.Entity)) {
                        target = e;
                        break;
                    }
                }
            }
            if (target == entt::null) {
                TextUnformatted("No matching sound object found.");
            } else {
                const auto target_name = GetName(R, target);
                const auto *active = R.try_get<const RealImpactActiveMicrophone>(target);
                const bool is_active = active && active->Entity == active_entity;
                if (is_active) {
                    Text("Active for: %s", target_name.c_str());
                } else if (Button(std::format("Set as active for {}", target_name).c_str())) {
                    const auto dir = R.get<const Path>(R.get<const Instance>(target).Entity).Value.parent_path();
                    const auto &vertex_indices = R.get<const RealImpactVertices>(target).Vertices;
                    Apply(action::audio::SetVertexSamples{target, vertex_indices, RealImpact::LoadSamples(dir, mic->Index) | to<std::vector>()});
                    Apply(action::Replace<RealImpactActiveMicrophone>{target, {active_entity}});
                }
                if (Button("Select sound object")) Select(target);
            }
        }
    }
    physics_ui::RenderEntityProperties(R, active_entity, SceneEntity, *Physics, [this](action::physics::Action a) { Apply(std::move(a)); });

    // glTF metadata: round-trip-only source state on the active entity.
    // TODO: surface per-material source metadata here once material editing UI exists:
    //   - `extras` JSON via SourceAssets::ExtrasByEntity[(Category::Materials, source_index)]
    //   - `MaterialSourceMeta::ExtensionPresence` bits (which extension blocks the source had)
    if (const auto *sa = R.try_get<const gltf::SourceAssets>(SceneEntity)) {
        const auto mesh_entity = active_instance ? active_instance->Entity : entt::null;
        const auto extras = [&](const auto *src, gltf::ExtrasCategory cat) -> std::optional<std::string_view> {
            return src ? gltf::GetExtras(*sa, cat, src->Value) : std::nullopt;
        };
        const std::pair<const char *, std::optional<std::string_view>> sections[]{
            {"Extras (Node)", extras(R.try_get<const SourceNodeIndex>(active_entity), gltf::ExtrasCategory::Nodes)},
            {"Extras (Mesh)", extras(mesh_entity != entt::null ? R.try_get<const SourceMeshIndex>(mesh_entity) : nullptr, gltf::ExtrasCategory::Meshes)},
            {"Extras (Camera)", extras(R.try_get<const SourceCameraIndex>(active_entity), gltf::ExtrasCategory::Cameras)},
            {"Extras (Light)", extras(R.try_get<const SourceLightIndex>(active_entity), gltf::ExtrasCategory::Lights)},
        };
        const auto *mesh_layout = mesh_entity != entt::null ? R.try_get<const MeshSourceLayout>(mesh_entity) : nullptr;
        const bool any_extras = std::ranges::any_of(sections, [](const auto &s) { return s.second.has_value(); });
        if ((any_extras || mesh_layout) && CollapsingHeader("glTF metadata")) {
            for (const auto &[label, json] : sections) {
                if (json) RenderJsonBlock(label, *json);
            }
            if (mesh_layout && !mesh_layout->VertexCounts.empty()) {
                SeparatorText("Mesh source layout");
                Text("Primitives: %zu", mesh_layout->VertexCounts.size());
                for (size_t i = 0; i < mesh_layout->VertexCounts.size(); ++i) {
                    const uint32_t flags = i < mesh_layout->AttributeFlags.size() ? mesh_layout->AttributeFlags[i] : 0u;
                    const bool indexed = i < mesh_layout->HasSourceIndices.size() && mesh_layout->HasSourceIndices[i];
                    Text("[%zu] verts:%u%s attrs:%s", i, mesh_layout->VertexCounts[i], indexed ? "" : " (non-indexed)", AttributeFlagsString(flags).c_str());
                }
                if (!mesh_layout->MorphTangentDeltas.empty()) {
                    Text("Morph tangent deltas: %zu", mesh_layout->MorphTangentDeltas.size());
                }
            }
        }
    }

    PopID();
}

void Scene::RenderControls() {
    if (BeginTabBar("Scene controls")) {
        if (BeginTabItem("Object")) {
            {
                const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
                const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
                PushID("InteractionMode");
                AlignTextToFramePadding();
                TextUnformatted("Interaction mode:");
                auto interaction_mode_value = int(interaction_mode);
                bool interaction_mode_changed = false;
                const auto active_entity_rc = FindActiveEntity(R);
                const bool active_is_armature_rc = FindArmatureObject(R, active_entity_rc) != entt::null;
                const bool edit_allowed = AllSelectedAreMeshes(R) || active_is_armature_rc;
                const bool pose_allowed = active_is_armature_rc;
                for (const auto mode : InteractionModes) {
                    if (mode == InteractionMode::Edit && !edit_allowed) continue;
                    if (mode == InteractionMode::Pose && !pose_allowed) continue;
                    SameLine();
                    interaction_mode_changed |= RadioButton(to_string(mode).c_str(), &interaction_mode_value, int(mode));
                }
                if (interaction_mode_changed) Apply(action::scene::SetInteractionMode{.Mode = InteractionMode(interaction_mode_value)});
                if (interaction_mode == InteractionMode::Edit || interaction_mode == InteractionMode::Excite) {
                    Checkbox("Orbit to active", &OrbitToActive);
                }
                if (interaction_mode == InteractionMode::Edit) {
                    Checkbox("X-ray selection", &SelectionXRay);
                }
                if (interaction_mode == InteractionMode::Edit && !active_is_armature_rc) {
                    AlignTextToFramePadding();
                    TextUnformatted("Edit mode:");
                    auto type_interaction_mode = int(edit_mode);
                    for (const auto element : Elements) {
                        auto name = Capitalize(label(element));
                        SameLine();
                        if (RadioButton(name.c_str(), &type_interaction_mode, int(element))) {
                            Apply(action::scene::SetEditMode{.Mode = element});
                        }
                    }
                    if (const auto active_entity = FindActiveEntity(R); active_entity != entt::null) {
                        if (const auto *instance = R.try_get<Instance>(active_entity); instance && R.all_of<Mesh>(instance->Entity)) {
                            const auto *br = R.try_get<const MeshSelectionBitsetRange>(instance->Entity);
                            const uint32_t selected_count = br ?
                                scene_selection::CountSelected(Stores.Buffers->SelectionBitset.Data(), br->Offset, br->Count) :
                                0;
                            Text("Editing %s: %u selected", label(edit_mode).data(), selected_count);
                        }
                    }
                }
                PopID();
            }
            if (auto *sa = R.try_get<gltf::SourceAssets>(SceneEntity); sa && sa->Scenes.size() > 1) {
                const auto &active_name = sa->Scenes[sa->ActiveSceneIndex].Name;
                const auto preview = NamedOr(active_name, "Scene ", sa->ActiveSceneIndex);
                if (BeginCombo("Scene", preview.c_str())) {
                    for (uint32_t i = 0; i < sa->Scenes.size(); ++i) {
                        const bool selected = i == sa->ActiveSceneIndex;
                        if (Selectable(NamedOr(sa->Scenes[i].Name, "Scene ", i).c_str(), selected) && !selected) {
                            gltf::SwitchActiveScene(R, SceneEntity, i);
                        }
                        if (selected) SetItemDefaultFocus();
                    }
                    EndCombo();
                }
            }
            if (CollapsingHeader("Object tree", ImGuiTreeNodeFlags_DefaultOpen)) RenderObjectTree();
            SeparatorText("");
            if (CollapsingHeader("Add object")) {
                bool added{false};
                for (uint32_t i = 0; i < AllPrimitiveShapes.size(); ++i) {
                    if (i % 4 != 0) SameLine();
                    const auto &shape = AllPrimitiveShapes[i];
                    if (Button(ToString(shape).c_str())) {
                        Apply(action::object::AddMeshPrimitive{shape, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Name = ToString(shape)})});
                        added = true;
                    }
                }
                Spacing();
                if (Button("Empty")) {
                    Apply(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
                    added = true;
                }
                SameLine();
                if (Button("Armature")) {
                    Apply(action::object::AddArmature{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
                    added = true;
                }
                SameLine();
                if (Button("Camera")) {
                    Apply(action::object::AddCamera{.Info = std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive}), .Props = {}});
                    added = true;
                }
                SameLine();
                if (Button("Light")) {
                    Apply(action::object::AddLight{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
                    added = true;
                }
                if (added) StartScreenTransform = TransformGizmo::TransformType::Translate;
            }
            if (auto *mv = R.try_get<MaterialVariants>(SceneEntity); mv && !mv->Names.empty() && CollapsingHeader("Material variants")) {
                const auto active = mv->Active;
                const auto preview = active ? NamedOr(mv->Names[*active], "Variant ", *active) : std::string{"Default"};
                const auto set_variant = [&](std::optional<uint32_t> v) {
                    if (active != v) Apply(action::UpdateOf<&MaterialVariants::Active>(SceneEntity, v));
                };
                if (BeginCombo("Active variant", preview.c_str())) {
                    if (Selectable("Default", !active)) set_variant({});
                    for (uint32_t i = 0; i < mv->Names.size(); ++i) {
                        if (Selectable(NamedOr(mv->Names[i], "Variant ", i).c_str(), active == i)) set_variant(i);
                    }
                    EndCombo();
                }
            }
            if (!R.storage<Selected>().empty()) {
                SeparatorText("Selection actions");
                std::vector<entt::entity> selected_mesh_instances;
                for (const auto entity : R.view<const Selected, const Instance>()) {
                    if (!R.all_of<SubElementOf>(entity)) selected_mesh_instances.emplace_back(entity);
                }

                if (!selected_mesh_instances.empty()) {
                    const bool any_visible = any_of(selected_mesh_instances, [&](entt::entity e) { return R.all_of<RenderInstance>(e); });
                    const bool any_hidden = any_of(selected_mesh_instances, [&](entt::entity e) { return !R.all_of<RenderInstance>(e); });
                    const bool mixed_visible = any_visible && any_hidden;
                    if (mixed_visible) PushItemFlag(ImGuiItemFlags_MixedValue, true);
                    if (bool set_visible = any_visible && !any_hidden; Checkbox("Visible", &set_visible)) {
                        Apply(action::object::SetSelectedVisible{set_visible});
                    }
                    if (mixed_visible) PopItemFlag();

                    const auto face_mesh_entities = scene_selection::GetSelectedMeshEntities(R) |
                        std::views::filter([&](entt::entity me) { return R.get<const Mesh>(me).FaceCount() > 0; }) |
                        to<std::vector>();
                    if (!face_mesh_entities.empty()) {
                        const bool any_smooth = any_of(face_mesh_entities, [&](entt::entity me) { return R.all_of<SmoothShading>(me); });
                        const bool any_flat = any_of(face_mesh_entities, [&](entt::entity me) { return !R.all_of<SmoothShading>(me); });
                        const bool mixed_smooth = any_smooth && any_flat;
                        SameLine();
                        if (mixed_smooth) PushItemFlag(ImGuiItemFlags_MixedValue, true);
                        if (bool set_smooth = any_smooth && !any_flat; Checkbox("Smooth shading", &set_smooth)) {
                            for (const auto me : face_mesh_entities) Apply(action::SetTagOf<SmoothShading>(me, set_smooth));
                        }
                        if (mixed_smooth) PopItemFlag();
                    }
                }
                if (CanDuplicate() && Button("Duplicate")) Duplicate();
                if (CanDuplicateLinked()) {
                    SameLine();
                    if (Button("Duplicate linked")) DuplicateLinked();
                }
                if (CanDelete() && Button("Delete")) Delete();
                if (R.get<const SceneInteraction>(SceneEntity).Mode == InteractionMode::Pose && !R.view<const BoneSelection>().empty()) {
                    AlignTextToFramePadding();
                    TextUnformatted("Clear transform:");
                    SameLine();
                    if (Button("All")) Apply(action::bone::ClearSelectedTransforms{.Position = true, .Rotation = true, .Scale = true});
                    SameLine();
                    if (Button("Position")) Apply(action::bone::ClearSelectedTransforms{.Position = true});
                    SameLine();
                    if (Button("Rotation")) Apply(action::bone::ClearSelectedTransforms{.Rotation = true});
                    SameLine();
                    if (Button("Scale")) Apply(action::bone::ClearSelectedTransforms{.Scale = true});
                }
            }
            RenderEntityControls(FindActiveEntity(R));
            EndTabItem();
        }

        if (BeginTabItem("Render")) {
            ui::Edit f{R, ui::Applier{this}, SceneEntity};
            const auto &settings = R.get<const SceneSettings>(SceneEntity);
            {
                auto color = settings.ClearColor;
                if (ColorEdit3("Background color", color.float32)) {
                    color.float32[3] = 1.f;
                    f.Set<&SceneSettings::ClearColor>(color);
                }
            }
            if (Button("Recompile shaders")) ShaderRecompileRequested = true;

            if (!R.view<Selected>().empty()) {
                SeparatorText("Selection overlays");
                AlignTextToFramePadding();
                TextUnformatted("Normals");
                for (const auto element : NormalElements) {
                    SameLine();
                    bool show = ElementMaskContains(settings.NormalOverlays, element);
                    if (const auto type_name = Capitalize(label(element));
                        Checkbox(type_name.c_str(), &show)) {
                        auto next_mask = settings.NormalOverlays;
                        SetElementMask(next_mask, element, show);
                        f.Set<&SceneSettings::NormalOverlays>(next_mask);
                    }
                }
                f.Check<&SceneSettings::ShowBoundingBoxes>("Bounding boxes");
                if (!R.view<const TetMeshData>().empty()) f.Check<&SceneSettings::ShowTetWireframe>("Tet wireframe");
            }
            {
                using VC = ViewportThemeColors;
                using AC = AxisThemeColors;
                SeparatorText("Viewport theme");
                const auto &theme = R.get<const ViewportTheme>(SceneEntity);
                if (Button("Reset##ViewportTheme")) Apply(action::scene::ResetViewportTheme{});
                auto c = f.Sub<&ViewportTheme::Colors>();
                c.Color<&VC::Grid>("Grid");
                c.Color<&VC::Wire>("Wire");
                c.Color<&VC::WireEdit>("Wire edit");
                c.Color<&VC::ObjectActive>("Object active");
                c.Color<&VC::ObjectSelected>("Object selected");
                c.Color<&VC::Light>("Light");
                c.Color<&VC::Vertex>("Vertex");
                c.Color<&VC::VertexSelected>("Vertex selected");
                c.Color<&VC::EdgeSelectedIncidental>("Edge selected (incidental)");
                c.Color<&VC::EdgeSelected>("Edge selected");
                c.Color<&VC::FaceSelectedIncidental>("Face selected (incidental)");
                c.Color<&VC::FaceSelected>("Face selected");
                c.Color<&VC::ElementActive>("Element active");
                c.Color<&VC::ElementExcited>("Element excited");
                c.Color<&VC::FaceNormal>("Face normal");
                c.Color<&VC::VertexNormal>("Vertex normal");
                c.Color<&VC::BoneSolid>("Bone solid");
                c.Color<&VC::BonePose>("Bone pose");
                c.Color<&VC::BonePoseActive>("Bone pose active");
                c.Color<&VC::Transform>("Transform");
                SeparatorText("Axis colors");
                auto a = f.Sub<&ViewportTheme::AxisColors>();
                a.Color<&AC::X>("Axis X");
                a.Color<&AC::Y>("Axis Y");
                a.Color<&AC::Z>("Axis Z");
                // UI edits "full" width; storage is half-width.
                if (float full_width = theme.EdgeWidth * 2.f; SliderFloat("Edge width", &full_width, 0.5f, 4.f))
                    f.Set<&ViewportTheme::EdgeWidth>(full_width / 2.f);
                if (uint32_t v = theme.SilhouetteEdgeWidth; MeshEditor::SliderUInt("Silhouette edge width", &v, 1, 4))
                    f.Set<&ViewportTheme::SilhouetteEdgeWidth>(v);
            }
            EndTabItem();
        }

        if (BeginTabItem("Camera")) {
            const auto &camera = R.get<const ViewCamera>(SceneEntity);
            const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
            const float viewport_aspect = extent.width == 0 || extent.height == 0 ? 1.f : float(extent.width) / float(extent.height);
            if (Button("Reset##Camera")) Apply(action::scene::ResetViewCamera{});
            if (vec3 target = camera.Target; SliderFloat3("Target", &target.x, -10, 10))
                Apply(action::scene::SetViewCameraTarget{target});
            if (Camera lens = camera.Data; RenderCameraLensEditor(lens, camera.Distance, viewport_aspect))
                Apply(action::scene::SetViewCameraLens{lens});
            EndTabItem();
        }

        if (BeginTabItem("Physics")) {
            physics_ui::RenderTab(R, SceneEntity, *Physics, [this](action::physics::Action a) { Apply(std::move(a)); });
            EndTabItem();
        }

        if (const auto *sa = R.try_get<const gltf::SourceAssets>(SceneEntity)) {
            if (BeginTabItem("glTF metadata")) {
                const bool has_asset = !sa->Generator.empty() || !sa->Copyright.empty() || !sa->MinVersion.empty() || !sa->AssetExtras.empty() || !sa->AssetExtensions.empty();
                if (has_asset && CollapsingHeader("Asset", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (!sa->Generator.empty()) Text("Generator: %s", sa->Generator.c_str());
                    if (!sa->Copyright.empty()) Text("Copyright: %s", sa->Copyright.c_str());
                    if (!sa->MinVersion.empty()) Text("Min version: %s", sa->MinVersion.c_str());
                    if (!sa->AssetExtras.empty()) RenderJsonBlock("asset.extras", sa->AssetExtras);
                    if (!sa->AssetExtensions.empty()) RenderJsonBlock("asset.extensions", sa->AssetExtensions);
                }
                if (!sa->ExtensionsRequired.empty() && CollapsingHeader("Extensions required", ImGuiTreeNodeFlags_DefaultOpen)) {
                    for (const auto &e : sa->ExtensionsRequired) BulletText("%s", e.c_str());
                }
                if (CollapsingHeader("Counts", ImGuiTreeNodeFlags_DefaultOpen)) {
                    Text("Source nodes: %zu", R.view<const SourceNodeIndex>().size());
                    Text("Meshes: %zu", R.view<const MeshName>().size());
                    Text("Materials: %zu", sa->MaterialMetas.size());
                    Text("Textures: %zu", sa->Textures.size());
                    Text("Images: %zu", sa->Images.size());
                    Text("Samplers: %zu", sa->Samplers.size());
                    Text("Animations: %zu", sa->AnimationOrder.size());
                    Text("Skins: %zu", R.view<const SkinName>().size());
                    Text("Cameras: %zu", R.view<const CameraName>().size());
                    Text("Lights: %zu", R.view<const LightName>().size());
                    Text("Physics materials: %zu", R.view<const SourcePhysicsMaterialIndex>().size());
                    Text("Collision filters: %zu", R.view<const SourceCollisionFilterIndex>().size());
                    Text("Physics joints: %zu", R.view<const SourcePhysicsJointDefIndex>().size());
                }
                if (!sa->Images.empty() && CollapsingHeader("Image registry")) {
                    if (BeginTable("Images", 5, MetadataTableFlags)) {
                        TableSetupColumn("#");
                        TableSetupColumn("Name");
                        TableSetupColumn("Mime");
                        TableSetupColumn("Source");
                        TableSetupColumn("Bytes");
                        TableHeadersRow();
                        for (size_t i = 0; i < sa->Images.size(); ++i) {
                            const auto &img = sa->Images[i];
                            TableNextRow();
                            TableNextColumn();
                            Text("%zu", i);
                            TableNextColumn();
                            TextUnformatted(img.Name.c_str());
                            TableNextColumn();
                            TextUnformatted(MimeTypeName(img.MimeType).data());
                            TableNextColumn();
                            if (img.SourceDataUri) TextUnformatted("data URI");
                            else if (!img.SourceAbsPath.empty()) TextUnformatted(img.SourceAbsPath.c_str());
                            else TextUnformatted("embedded");
                            TableNextColumn();
                            if (img.Bytes.empty()) TextUnformatted(img.SourceAbsPath.empty() ? "—" : "external");
                            else Text("%zu", img.Bytes.size());
                            if (img.IsDirty) {
                                SameLine();
                                TextUnformatted("(dirty)");
                            }
                        }
                        EndTable();
                    }
                }
                if (!sa->Textures.empty() && CollapsingHeader("Texture registry")) {
                    if (BeginTable("Textures", 4, MetadataTableFlags)) {
                        TableSetupColumn("#");
                        TableSetupColumn("Name");
                        TableSetupColumn("Sampler");
                        TableSetupColumn("Image (default/WebP/Basisu/DDS)");
                        TableHeadersRow();
                        const auto idx_or_dash = [](const std::optional<uint32_t> &i) {
                            return i ? std::format("{}", *i) : std::string{"—"};
                        };
                        for (size_t i = 0; i < sa->Textures.size(); ++i) {
                            const auto &t = sa->Textures[i];
                            TableNextRow();
                            TableNextColumn();
                            Text("%zu", i);
                            TableNextColumn();
                            TextUnformatted(t.Name.c_str());
                            TableNextColumn();
                            TextUnformatted(idx_or_dash(t.SamplerIndex).c_str());
                            TableNextColumn();
                            Text("%s / %s / %s / %s", idx_or_dash(t.ImageIndex).c_str(), idx_or_dash(t.WebpImageIndex).c_str(), idx_or_dash(t.BasisuImageIndex).c_str(), idx_or_dash(t.DdsImageIndex).c_str());
                        }
                        EndTable();
                    }
                }
                if (!sa->Samplers.empty() && CollapsingHeader("Sampler registry")) {
                    if (BeginTable("Samplers", 5, MetadataTableFlags)) {
                        TableSetupColumn("#");
                        TableSetupColumn("Name");
                        TableSetupColumn("Mag");
                        TableSetupColumn("Min");
                        TableSetupColumn("Wrap S/T");
                        TableHeadersRow();
                        const auto filter_name = [](gltf::Filter f) -> std::string_view {
                            using F = gltf::Filter;
                            switch (f) {
                                case F::Nearest: return "Nearest";
                                case F::Linear: return "Linear";
                                case F::NearestMipMapNearest: return "Nearest/Nearest";
                                case F::LinearMipMapNearest: return "Linear/Nearest";
                                case F::NearestMipMapLinear: return "Nearest/Linear";
                                case F::LinearMipMapLinear: return "Linear/Linear";
                            }
                            return "?";
                        };
                        const auto wrap_name = [](gltf::Wrap w) -> std::string_view {
                            using W = gltf::Wrap;
                            switch (w) {
                                case W::ClampToEdge: return "Clamp";
                                case W::MirroredRepeat: return "Mirror";
                                case W::Repeat: return "Repeat";
                            }
                            return "?";
                        };
                        for (size_t i = 0; i < sa->Samplers.size(); ++i) {
                            const auto &s = sa->Samplers[i];
                            TableNextRow();
                            TableNextColumn();
                            Text("%zu", i);
                            TableNextColumn();
                            TextUnformatted(s.Name.c_str());
                            TableNextColumn();
                            TextUnformatted(s.MagFilter ? filter_name(*s.MagFilter).data() : "—");
                            TableNextColumn();
                            TextUnformatted(s.MinFilter ? filter_name(*s.MinFilter).data() : "—");
                            TableNextColumn();
                            Text("%s / %s", wrap_name(s.WrapS).data(), wrap_name(s.WrapT).data());
                        }
                        EndTable();
                    }
                }
                EndTabItem();
            }
        }
        EndTabBar();
    }
}

void Scene::RenderClipPickers() {
    static constexpr float ComboWidth = 200.f;
    // Names live on object entities, but ArmatureAnimation lives on the data entity.
    const auto display_name = [&]<typename Anim>(entt::entity entity) {
        if constexpr (std::is_same_v<Anim, ArmatureAnimation>) {
            for (const auto [obj_e, obj] : R.view<const ArmatureObject>().each()) {
                if (obj.Entity == entity) return GetName(R, obj_e);
            }
        }
        return GetName(R, entity);
    };
    const auto clip_picker = [&]<typename Anim>(std::string_view kind) {
        for (auto [entity, anim] : R.view<Anim>().each()) {
            if (anim.Clips.size() < 2) continue;
            const auto active_idx = anim.ActiveClipIndex;
            const auto label = std::format("{}: {}", kind, display_name.template operator()<Anim>(entity));
            PushID(label.c_str());
            SetNextItemWidth(ComboWidth);
            if (BeginCombo("##clip", NamedOr(anim.Clips[active_idx].Name, "Clip ", active_idx).c_str())) {
                for (uint32_t i = 0; i < anim.Clips.size(); ++i) {
                    if (Selectable(NamedOr(anim.Clips[i].Name, "Clip ", i).c_str(), active_idx == i) && active_idx != i) {
                        Apply(action::UpdateOf<&Anim::ActiveClipIndex>(entity, i));
                    }
                }
                EndCombo();
            }
            SameLine();
            TextUnformatted(label.c_str());
            PopID();
        }
    };
    clip_picker.template operator()<ArmatureAnimation>("Armature");
    clip_picker.template operator()<MorphWeightAnimation>("Morph");
    clip_picker.template operator()<NodeTransformAnimation>("Node");
}

void Scene::RenderObjectTree() {
    PushStyleVar(ImGuiStyleVar_ItemSpacing, {GetStyle().ItemSpacing.x, 0.f});

    const auto ToSelectionUserData = [](entt::entity e) -> ImGuiSelectionUserData { return ImGuiSelectionUserData(uint32_t(e)); };
    const auto FromSelectionUserData = [&](ImGuiSelectionUserData data) -> entt::entity {
        if (data == ImGuiSelectionUserData_Invalid) return entt::null;
        const auto e = entt::entity(uint32_t(data));
        return R.valid(e) ? e : entt::null;
    };

    const auto GetEntityTypeName = [&](entt::entity e) -> std::string_view {
        if (R.all_of<BoneIndex>(e)) return "Bone";
        if (R.all_of<ObjectKind>(e)) return ObjectTypeName(R.get<const ObjectKind>(e).Value);
        return ObjectTypeName(ObjectType::Empty);
    };
    std::vector<entt::entity> visible_entities;
    const auto ResolveSelectionRequests = [&](std::span<const ImGuiSelectionRequest> requests, ImGuiSelectionUserData nav_item) -> action::selection::ApplyTreeSelection {
        using Clear = action::selection::ApplyTreeSelection::ClearKind;
        action::selection::ApplyTreeSelection out;
        const auto add_target = [&](entt::entity e, bool selected) {
            if (e == entt::null) return;
            (selected ? out.ToSelect : out.ToDeselect).push_back(e);
        };
        for (const auto &request : requests) {
            if (request.Type == ImGuiSelectionRequestType_SetAll) {
                if (request.Selected) {
                    for (const auto e : visible_entities) add_target(e, true);
                } else {
                    const auto nav = FromSelectionUserData(nav_item);
                    out.Clear = nav != entt::null && R.all_of<BoneIndex>(nav) ? Clear::BonesOnly : Clear::All;
                }
                continue;
            }
            if (request.Type != ImGuiSelectionRequestType_SetRange) continue;

            const auto first = FromSelectionUserData(request.RangeFirstItem), last = FromSelectionUserData(request.RangeLastItem);
            const auto first_it = find(visible_entities, first), last_it = find(visible_entities, last);
            if (first_it == visible_entities.end() || last_it == visible_entities.end()) {
                add_target(first, request.Selected);
                add_target(last, request.Selected);
                continue;
            }
            const auto first_i = distance(visible_entities.begin(), first_it);
            const auto last_i = distance(visible_entities.begin(), last_it);
            const auto [i0, i1] = std::minmax(first_i, last_i);
            for (auto i = i0; i <= i1; ++i) add_target(visible_entities[i], request.Selected);
        }
        out.NavToActive = FromSelectionUserData(nav_item);
        return out;
    };

    const int total_selected = R.storage<Selected>().size() + R.storage<BoneSelection>().size();
    auto *ms_begin = BeginMultiSelect(ImGuiMultiSelectFlags_None, total_selected, -1);
    std::vector<ImGuiSelectionRequest> begin_requests;
    begin_requests.reserve(ms_begin->Requests.Size);
    for (const auto &request : ms_begin->Requests) begin_requests.emplace_back(request);
    const auto begin_nav_item = ms_begin->NavIdItem;

    // Build the set of ancestors of any selected entity (for secondary highlight).
    std::unordered_set<entt::entity> ancestor_of_selected;
    const auto mark_ancestors = [&](entt::entity selected_entity) {
        const auto *n = R.try_get<SceneNode>(selected_entity);
        auto parent = n ? n->Parent : entt::null;
        while (parent != entt::null) {
            if (!ancestor_of_selected.insert(parent).second) break; // already inserted — parents already covered
            const auto *pn = R.try_get<SceneNode>(parent);
            parent = pn ? pn->Parent : entt::null;
        }
    };
    for (const auto e : R.view<Selected>()) mark_ancestors(e);
    for (const auto e : R.view<BoneSelection>()) mark_ancestors(e);

    const auto render_entity = [&](const auto &self, entt::entity e) -> void {
        const auto *node = R.try_get<SceneNode>(e);
        const bool has_children = node && node->FirstChild != entt::null;
        const bool is_selected = R.any_of<Selected, BoneSelection>(e);
        const bool is_ancestor_selected = !is_selected && ancestor_of_selected.contains(e);

        auto flags =
            ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanFullWidth |
            ImGuiTreeNodeFlags_FramePadding |
            ImGuiTreeNodeFlags_NavLeftJumpsToParent;
        if (!has_children) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (is_selected || is_ancestor_selected) flags |= ImGuiTreeNodeFlags_Selected;

        if (is_ancestor_selected) {
            const auto col = GetStyleColorVec4(ImGuiCol_Header);
            PushStyleColor(ImGuiCol_Header, ImVec4{col.x, col.y, col.z, col.w * 0.4f});
            PushStyleColor(ImGuiCol_HeaderHovered, ImVec4{col.x, col.y, col.z, col.w * 0.6f});
        }

        SetNextItemSelectionUserData(ToSelectionUserData(e));
        const bool open = TreeNodeEx(reinterpret_cast<void *>(uintptr_t(uint32_t(e))), flags, "%s", GetName(R, e).c_str());
        SameLine();
        if (const auto type_suffix = GetEntityTypeName(e); R.any_of<Active, BoneActive>(e)) {
            const auto &theme = R.get<const ViewportTheme>(SceneEntity);
            const auto color = R.all_of<BoneActive>(e) ? theme.Colors.BoneActive : theme.Colors.ObjectActive;
            TextColored(ImVec4{color.x, color.y, color.z, 1.f}, "[%s]", type_suffix.data());
        } else {
            TextDisabled("[%s]", type_suffix.data());
        }
        if (is_ancestor_selected) PopStyleColor(2);
        visible_entities.emplace_back(e);
        if (open && has_children) {
            for (const auto child : Children{&R, e}) self(self, child);
            TreePop();
        }
    };

    bool has_root = false;
    for (const auto [entity, _] : R.view<const Name>().each()) {
        if (const auto *node = R.try_get<SceneNode>(entity); node && node->Parent != entt::null) continue;
        has_root = true;
        render_entity(render_entity, entity);
    }
    if (!has_root) TextDisabled("No objects");

    Apply(ResolveSelectionRequests(begin_requests, begin_nav_item));
    auto *ms_end = EndMultiSelect();
    Apply(ResolveSelectionRequests({ms_end->Requests.Data, size_t(ms_end->Requests.Size)}, ms_end->NavIdItem));

    PopStyleVar();
}
