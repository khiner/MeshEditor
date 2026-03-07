#include "PbrFeature.h"
#include "Scene.h"
#include "SceneDefaults.h"
#include "SceneMaterials.h"
#include "SceneTextures.h"
#include "SceneTree.h"
#include "Widgets.h" // imgui

#include "Armature.h"
#include "Excitable.h"
#include "MeshComponents.h"
#include "MeshInstance.h"
#include "OrientationGizmo.h"
#include "SceneSelection.h"
#include "SvgResource.h"
#include "Timer.h"
#include "Variant.h"
#include "gpu/WorkspaceLights.h"
#include "gpu/WorldTransform.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "numeric/mat3.h"
#include "numeric/rect.h"

#include <algorithm>
#include <cmath>
#include <entt/entity/registry.hpp>
#include <imgui_internal.h>

#include "scene_impl/SceneBuffers.h"
#include "scene_impl/SceneComponents.h"
#include "scene_impl/SceneInternalTypes.h"
#include "scene_impl/SceneTransformUtils.h"

using std::ranges::all_of, std::ranges::any_of, std::ranges::distance, std::ranges::find, std::ranges::fold_left, std::ranges::to;
using std::views::filter, std::views::transform;
using namespace ImGui;

namespace {
constexpr vec2 ToGlm(ImVec2 v) { return std::bit_cast<vec2>(v); }
constexpr float WheelZoomBaseSpeed{0.01f}; // Base per-wheel-unit zoom speed for isolated scroll ticks.
constexpr float WheelZoomMaxBurst{8.f}; // Cap on accumulated rapid-scroll acceleration.
constexpr double WheelZoomAccelerationWindow{0.25}; // Seconds between ticks that still count as one accelerated burst.

vk::Extent2D ComputeRenderExtentPx(vk::Extent2D logical_extent) {
    const auto scale = GetIO().DisplayFramebufferScale;
    const auto scaled_dim = [](uint32_t logical, float s) -> uint32_t {
        if (logical == 0u) return 0u;
        const float scale_value = s > 0.0f ? s : 1.0f;
        return std::max(1u, uint32_t(float(logical) * scale_value + 0.5f));
    };
    return {scaled_dim(logical_extent.width, scale.x), scaled_dim(logical_extent.height, scale.y)};
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
    ImVec2 mouse_delta{0, 0};
    for (int axis = 0; axis < 2; ++axis) {
        if (g.IO.MousePos[axis] >= wrap_rect.Max[axis]) mouse_delta[axis] = -wrap_rect.GetSize()[axis] + 1;
        else if (g.IO.MousePos[axis] <= wrap_rect.Min[axis]) mouse_delta[axis] = wrap_rect.GetSize()[axis] - 1;
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

std::optional<MeshData> PrimitiveEditor(PrimitiveType type, bool is_create = true) {
    const char *create_label = is_create ? "Add" : "Update";
    if (type == PrimitiveType::Rect) {
        static vec2 size{1, 1};
        InputFloat2("Size", &size.x);
        if (Button(create_label)) return primitive::Rect(size / 2.f);
    } else if (type == PrimitiveType::Circle) {
        static float r = 0.5;
        InputFloat("Radius", &r);
        if (Button(create_label)) return primitive::Circle(r);
    } else if (type == PrimitiveType::Cube) {
        static vec3 size{1.0, 1.0, 1.0};
        InputFloat3("Size", &size.x);
        if (Button(create_label)) return primitive::Cuboid(size / 2.f);
    } else if (type == PrimitiveType::IcoSphere) {
        static float r = 0.5;
        static int subdivisions = 3;
        InputFloat("Radius", &r);
        InputInt("Subdivisions", &subdivisions);
        if (Button(create_label)) return primitive::IcoSphere(r, uint(subdivisions));
    } else if (type == PrimitiveType::UVSphere) {
        static float r = 0.5;
        InputFloat("Radius", &r);
        if (Button(create_label)) return primitive::UVSphere(r);
    } else if (type == PrimitiveType::Torus) {
        static vec2 radii{0.5, 0.2};
        static glm::ivec2 n_segments = {32, 16};
        InputFloat2("Major/minor radius", &radii.x);
        InputInt2("Major/minor segments", &n_segments.x);
        if (Button(create_label)) return primitive::Torus(radii.x, radii.y, uint(n_segments.x), uint(n_segments.y));
    } else if (type == PrimitiveType::Cylinder) {
        static float r = 1, h = 1;
        InputFloat("Radius", &r);
        InputFloat("Height", &h);
        if (Button(create_label)) return primitive::Cylinder(r, h);
    } else if (type == PrimitiveType::Cone) {
        static float r = 1, h = 1;
        InputFloat("Radius", &r);
        InputFloat("Height", &h);
        if (Button(create_label)) return primitive::Cone(r, h);
    }

    return {};
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

struct ViewportContext {
    float Distance, AspectRatio;
};

bool RenderCameraLensEditor(Camera &camera, std::optional<ViewportContext> viewport = {}) {
    bool lens_changed = false;

    int proj_i = std::holds_alternative<Orthographic>(camera) ? 1 : 0;
    const char *proj_names[]{"Perspective", "Orthographic"};
    if (Combo("Projection", &proj_i, proj_names, IM_ARRAYSIZE(proj_names))) {
        if (proj_i == 0 && !std::holds_alternative<Perspective>(camera)) {
            camera = PerspectiveFromOrthographic(std::get<Orthographic>(camera), viewport ? std::optional<float>{viewport->Distance} : std::nullopt);
            lens_changed = true;
        } else if (proj_i == 1 && !std::holds_alternative<Orthographic>(camera)) {
            camera = OrthographicFromPerspective(
                std::get<Perspective>(camera),
                viewport ? std::optional<float>{viewport->Distance} : std::nullopt,
                viewport ? std::optional<float>{viewport->AspectRatio} : std::nullopt
            );
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
        if (!viewport) {
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

WorkspaceLights &GetWorkspaceLights(SceneBuffers &buffers) {
    return *reinterpret_cast<WorkspaceLights *>(buffers.WorkspaceLightsUBO.GetMappedData().data());
}

} // namespace

void Scene::SetInteractionMode(InteractionMode mode) {
    if (R.get<const SceneInteraction>(SceneEntity).Mode == mode) return;

    const auto active_entity = FindActiveEntity(R);
    // Active entity may be the armature object or a bone — resolve to arm_obj_entity.
    const auto active_arm = active_entity != entt::null ? (R.all_of<ArmatureObject>(active_entity) ? active_entity : R.all_of<BoneIndex>(active_entity) ? R.get<const MeshInstance>(active_entity).MeshEntity :
                                                                                                                                                          entt::null) :
                                                          entt::null;
    const bool active_is_armature = active_arm != entt::null;
    if (mode == InteractionMode::Edit && !AllSelectedAreMeshes(R) && !active_is_armature) return;
    if (mode == InteractionMode::Pose && !active_is_armature) return;

    const auto current_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    if (current_mode == InteractionMode::Edit) {
        // Leaving Edit mode: clear GPU element-state colors, but keep bitset ranges + bits
        // so element selections are remembered when toggling back into Edit mode.
        for (const auto [mesh_entity, br, mesh] : R.view<const MeshSelectionBitsetRange, const Mesh>().each()) {
            if (br.Count == 0) continue;
            Meshes->UpdateElementStates(mesh, Element::None, {}, {}, {}, {}, std::nullopt);
        }
        ElementStatesDirty = true;
    }
    if (mode == InteractionMode::Edit && !active_is_armature) {
        // Entering Edit mode: assign ranges only for selected meshes missing one.
        // Existing ranges preserve remembered selection.
        // ProcessComponentEvents handles the GPU state update via the InteractionMode reactive event.
        const auto edit_element = R.get<const SceneEditMode>(SceneEntity).Value;
        if (edit_element != Element::None) {
            uint32_t next_offset = 0;
            for (const auto [_, br] : R.view<const MeshSelectionBitsetRange>().each()) {
                next_offset = std::max(next_offset, (br.Offset + br.Count + 31) / 32 * 32);
            }
            auto *mapped = Buffers->SelectionBitsetBuffer.GetMappedData().data();
            for (const auto mesh_entity : scene_selection::GetSelectedMeshEntities(R)) {
                if (R.all_of<MeshSelectionBitsetRange>(mesh_entity)) continue;
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                const uint32_t count = scene_selection::GetElementCount(mesh, edit_element);
                if (count == 0) continue;
                const uint32_t word_count = (count + 31) / 32;
                auto *u32 = reinterpret_cast<uint32_t *>(mapped) + next_offset / 32;
                memset(u32, 0xFF, word_count * sizeof(uint32_t)); // Select all elements by default.
                if (const uint32_t rem = count & 31) u32[word_count - 1] = (1u << rem) - 1u; // Clear gap bits.
                R.emplace<MeshSelectionBitsetRange>(mesh_entity, next_offset, count);
                next_offset = (next_offset + count + 31) / 32 * 32;
            }
        }
    }
    R.patch<SceneInteraction>(SceneEntity, [mode](auto &s) { s.Mode = mode; });
    R.patch<ViewportTheme>(SceneEntity, [](auto &) {});
}

void Scene::SetEditMode(Element mode) {
    const auto current_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    if (current_mode == mode) return;

    auto *bits = reinterpret_cast<uint32_t *>(Buffers->SelectionBitsetBuffer.GetMappedData().data());

    // Phase 1: scan old selection handles and zero old bits for every mesh.
    // Stash the from-handles so we can write new bits in phase 3, after the GPU
    // has already cleared the old element state buffers (phase 2).
    struct PendingConvert {
        entt::entity MeshEntity;
        uint32_t NewCount;
        std::vector<uint32_t> FromHandles;
    };
    std::vector<PendingConvert> pending;
    std::vector<ElementRange> old_ranges;
    uint32_t old_max_end = 0;
    for (auto [mesh_entity, br, mesh] : R.view<MeshSelectionBitsetRange, const Mesh>().each()) {
        const uint32_t old_count = br.Count, new_count = scene_selection::GetElementCount(mesh, mode);
        auto from_handles = scene_selection::ScanBitsetRange(bits, br.Offset, old_count);
        if (old_count > 0) old_ranges.emplace_back(mesh_entity, br.Offset, old_count);
        old_max_end = std::max(old_max_end, br.Offset + old_count);
        R.remove<MeshActiveElement>(mesh_entity);
        pending.push_back({mesh_entity, new_count, std::move(from_handles)});
    }

    // Clear the superset of old and new packed bit ranges once, to avoid stale/overlap bits.
    uint32_t new_max_end = 0;
    for (const auto &p : pending) new_max_end += (p.NewCount + 31) / 32 * 32;
    const uint32_t clear_words = (std::max(old_max_end, new_max_end) + 31) / 32;
    if (clear_words > 0) memset(bits, 0, clear_words * sizeof(uint32_t));

    // Phase 2: clear old element state buffers while the old bits are all zero.
    if (!old_ranges.empty()) {
        DispatchUpdateSelectionStates(old_ranges, current_mode);
        // Face mode also derives edge states via CPU; clear them when exiting face-select.
        if (current_mode == Element::Face) {
            for (const auto &p : pending) Meshes->UpdateEdgeStatesFromFaces(R.get<const Mesh>(p.MeshEntity), {}, {});
        }
    }

    // Phase 3: repack ranges, write converted bits and update new element state buffers.
    std::vector<ElementRange> new_ranges;
    uint32_t next_offset = 0;
    for (const auto &p : pending) {
        auto &br = R.get<MeshSelectionBitsetRange>(p.MeshEntity);
        br.Offset = next_offset;
        br.Count = p.NewCount;
        const auto &mesh = R.get<const Mesh>(p.MeshEntity);
        for (const uint32_t h : scene_selection::ConvertSelectionElement(p.FromHandles, mesh, current_mode, mode)) {
            if (h >= p.NewCount) continue;
            const uint32_t gbit = next_offset + h;
            bits[gbit >> 5] |= 1u << (gbit & 31u);
        }
        if (p.NewCount > 0) new_ranges.emplace_back(p.MeshEntity, next_offset, p.NewCount);
        next_offset = (next_offset + p.NewCount + 31) / 32 * 32;
    }

    R.patch<SceneEditMode>(SceneEntity, [mode](auto &edit_mode) { edit_mode.Value = mode; });
    if (!new_ranges.empty()) {
        ApplySelectionStateUpdate(new_ranges, mode);
    } else if (!old_ranges.empty()) {
        ElementStatesDirty = true; // Re-render to show cleared states.
    }
}

void Scene::ExitLookThroughCamera() {
    if (!SavedViewCamera) return;
    R.replace<ViewCamera>(SceneEntity, *SavedViewCamera);
    SavedViewCamera.reset();
}

void Scene::AnimateToCamera(entt::entity camera_entity) {
    const auto &wt = R.get<WorldTransform>(camera_entity);
    const vec3 pos = wt.Position;
    const vec3 fwd = -glm::normalize(glm::rotate(Vec4ToQuat(wt.Rotation), vec3{0.f, 0.f, 1.f}));
    const vec3 away = -fwd; // Forward() points from target to position
    R.patch<ViewCamera>(SceneEntity, [&](auto &vc) {
        vc.AnimateTo(pos + fwd, {std::atan2(away.z, away.x), std::asin(away.y)}, 1.f);
    });
}

void Scene::Interact() {
    const auto logical_extent = R.get<const ViewportExtent>(SceneEntity).Value;
    const auto render_extent = ComputeRenderExtentPx(logical_extent);
    if (logical_extent.width == 0 || logical_extent.height == 0 || render_extent.width == 0 || render_extent.height == 0) return;

    const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    const auto active_entity = FindActiveEntity(R);
    const bool has_frozen_selected = R.view<Selected, Frozen>().begin() != R.view<Selected, Frozen>().end();
    const bool edit_transform_locked = interaction_mode == InteractionMode::Edit &&
        any_of(scene_selection::GetSelectedMeshEntities(R), [&](entt::entity mesh_entity) { return scene_selection::HasFrozenInstance(R, mesh_entity); });
    const bool transform_shortcuts_enabled = !edit_transform_locked;
    const bool scale_shortcut_enabled = transform_shortcuts_enabled && !has_frozen_selected;
    // Handle keyboard input.
    if (IsWindowFocused()) {
        if (TransformGizmo::IsUsing()) {
            // During an active transform, only allow transform switching shortcuts.
            if (IsKeyPressed(ImGuiKey_G, false) && transform_shortcuts_enabled) {
                StartScreenTransform = TransformGizmo::TransformType::Translate;
            } else if (IsKeyPressed(ImGuiKey_R, false) && transform_shortcuts_enabled) {
                StartScreenTransform = TransformGizmo::TransformType::Rotate;
            } else if (IsKeyPressed(ImGuiKey_S, false) && scale_shortcut_enabled) {
                StartScreenTransform = TransformGizmo::TransformType::Scale;
            }
        } else {
            if (IsKeyPressed(ImGuiKey_Space, false)) R.patch<AnimationTimeline>(SceneEntity, [](auto &tl) { tl.Playing = !tl.Playing; });
            if (IsKeyPressed(ImGuiKey_Z, false) && !GetIO().KeyCtrl && !GetIO().KeyShift && !GetIO().KeyAlt && !GetIO().KeySuper) {
                R.patch<SceneSettings>(SceneEntity, [](auto &s) {
                    const auto next = s.ViewportShading == ViewportShadingMode::Solid ? ViewportShadingMode::MaterialPreview : s.ViewportShading == ViewportShadingMode::MaterialPreview ? ViewportShadingMode::Rendered :
                                                                                                                                                                                           ViewportShadingMode::Solid;
                    s.ViewportShading = next;
                    s.FillMode = next;
                });
            } else if (IsKeyPressed(ImGuiKey_Z, false) && !GetIO().KeyCtrl && GetIO().KeyShift && !GetIO().KeyAlt && !GetIO().KeySuper) {
                R.patch<SceneSettings>(SceneEntity, [](auto &s) {
                    s.ViewportShading = s.ViewportShading == ViewportShadingMode::Wireframe ? s.FillMode : ViewportShadingMode::Wireframe;
                });
            } else if (IsKeyPressed(ImGuiKey_Z, false) && !GetIO().KeyCtrl && !GetIO().KeyShift && GetIO().KeyAlt && !GetIO().KeySuper) {
                SelectionXRay = !SelectionXRay;
            }
            if (IsKeyPressed(ImGuiKey_Tab)) {
                const bool is_armature = active_entity != entt::null &&
                    (R.all_of<ArmatureObject>(active_entity) || R.all_of<BoneIndex>(active_entity));
                if (is_armature && GetIO().KeyCtrl) {
                    // Ctrl+Tab: toggle Object ↔ Pose
                    SetInteractionMode(interaction_mode == InteractionMode::Pose ? InteractionMode::Object : InteractionMode::Pose);
                } else if (is_armature) {
                    // Tab: toggle Object ↔ Edit
                    SetInteractionMode(interaction_mode == InteractionMode::Edit ? InteractionMode::Object : InteractionMode::Edit);
                } else {
                    // Cycle to the next interaction mode, wrapping around to the first.
                    auto it = find(InteractionModes, interaction_mode);
                    SetInteractionMode(++it != InteractionModes.end() ? *it : *InteractionModes.begin());
                }
            }
            if (interaction_mode == InteractionMode::Edit) {
                if (IsKeyPressed(ImGuiKey_1, false)) SetEditMode(Element::Vertex);
                else if (IsKeyPressed(ImGuiKey_2, false)) SetEditMode(Element::Edge);
                else if (IsKeyPressed(ImGuiKey_3, false)) SetEditMode(Element::Face);
            }
            if (IsKeyPressed(ImGuiKey_E, false) && GetIO().KeyCtrl && GetIO().KeyShift) {
                AddEmpty({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                StartScreenTransform = TransformGizmo::TransformType::Translate;
            } else if (IsKeyPressed(ImGuiKey_A, false) && GetIO().KeyCtrl && GetIO().KeyShift) {
                AddArmature({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                StartScreenTransform = TransformGizmo::TransformType::Translate;
            } else if (IsKeyPressed(ImGuiKey_C, false) && GetIO().KeyCtrl && GetIO().KeyShift) {
                AddCamera({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                StartScreenTransform = TransformGizmo::TransformType::Translate;
            } else if (IsKeyPressed(ImGuiKey_L, false) && GetIO().KeyCtrl && GetIO().KeyShift) {
                AddLight({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                StartScreenTransform = TransformGizmo::TransformType::Translate;
            }
            if (!R.storage<Selected>().empty()) {
                if (IsKeyPressed(ImGuiKey_D, false) && GetIO().KeyShift) Duplicate();
                else if (IsKeyPressed(ImGuiKey_D, false) && GetIO().KeyAlt) DuplicateLinked();
                else if (IsKeyPressed(ImGuiKey_Delete, false) || IsKeyPressed(ImGuiKey_Backspace, false)) Delete();
                else if (IsKeyPressed(ImGuiKey_G, false) && transform_shortcuts_enabled) {
                    // Start transform gizmo in both Object and Edit modes.
                    // In Edit mode, shader applies transform to selected vertices.
                    // In Object mode, shader applies transform to selected instances.
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                } else if (IsKeyPressed(ImGuiKey_R, false) && transform_shortcuts_enabled) StartScreenTransform = TransformGizmo::TransformType::Rotate;
                else if (IsKeyPressed(ImGuiKey_S, false) && scale_shortcut_enabled) StartScreenTransform = TransformGizmo::TransformType::Scale;
                else if (IsKeyPressed(ImGuiKey_H, false)) {
                    for (const auto e : R.view<Selected>()) SetVisible(e, !R.all_of<RenderInstance>(e));
                } else if (IsKeyPressed(ImGuiKey_P, false) && GetIO().KeyCtrl) {
                    if (active_entity != entt::null) {
                        for (const auto e : R.view<Selected>()) {
                            if (e != active_entity) SetParent(R, e, active_entity);
                        }
                    }
                } else if (IsKeyPressed(ImGuiKey_P, false) && GetIO().KeyAlt) {
                    for (const auto e : R.view<Selected>()) ClearParent(R, e);
                }
            }
        }
    }

    // Handle mouse input.
    if (!IsMouseDown(ImGuiMouseButton_Left)) R.clear<ExcitedVertex>();

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
    if (const vec2 wheel{io.MouseWheelH, io.MouseWheel}; wheel != vec2{0, 0}) {
        // Exit "look through" camera view on any orbit/zoom interaction.
        ExitLookThroughCamera();
        if (io.KeyCtrl || io.KeySuper) {
            const double now = GetTime();
            const float zoom_direction = wheel.y > 0.f ? 1.f : -1.f;
            const bool accelerate = LastWheelZoomTime >= 0.0 &&
                now - LastWheelZoomTime <= WheelZoomAccelerationWindow &&
                WheelZoomBurst * zoom_direction > 0.f;
            WheelZoomBurst = accelerate ? glm::clamp(WheelZoomBurst + zoom_direction, -WheelZoomMaxBurst, WheelZoomMaxBurst) : zoom_direction;
            LastWheelZoomTime = now;

            const float zoom_speed = WheelZoomBaseSpeed * (1.f + std::max(std::abs(WheelZoomBurst), 0.f));
            const float burst_ratio = std::abs(WheelZoomBurst) / WheelZoomMaxBurst;
            const float signed_zoom = wheel.y * zoom_speed * (1.f + 0.15f * burst_ratio);
            R.patch<ViewCamera>(SceneEntity, [&](auto &camera) {
                camera.SetTargetDistance(std::max(camera.TargetDistance() * std::exp2(-signed_zoom), 0.01f));
            });
        } else {
            R.patch<ViewCamera>(SceneEntity, [&](auto &camera) { camera.SetTargetYawPitch(camera.YawPitch + wheel * 0.15f); });
        }
    }
    if (OrientationGizmo::IsActive() || OverlayControlsHovered) return;

    const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    const bool active_is_armature = active_entity != entt::null &&
        (R.all_of<ArmatureObject>(active_entity) || R.all_of<BoneIndex>(active_entity));
    const bool bone_mode = interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && active_is_armature);
    if (SelectionMode == SelectionMode::Box && interaction_mode != InteractionMode::Excite) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            BoxSelectStart = BoxSelectEnd = ToGlm(GetMousePos());
        } else if (IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart) {
            BoxSelectEnd = ToGlm(GetMousePos());
            if (const auto box_px = ComputeBoxSelectPixels(*BoxSelectStart, *BoxSelectEnd, ToGlm(GetCursorScreenPos()), logical_extent, render_extent); box_px) {
                const bool is_additive = IsKeyDown(ImGuiMod_Shift);
                if (interaction_mode == InteractionMode::Edit && !active_is_armature) {
                    Timer timer{"BoxSelectElements (all)"};
                    RunBoxSelectElements(GetBitsetRangesForSelected(), edit_mode, *box_px, is_additive);
                } else if (bone_mode) {
                    const auto selected_entities = RunBoxSelect(*box_px);
                    if (!is_additive) {
                        for (const auto e : R.view<Selected, BoneIndex>()) R.remove<Selected>(e);
                    }
                    for (const auto e : selected_entities) {
                        if (R.all_of<BoneIndex>(e)) R.emplace_or_replace<Selected>(e);
                    }
                } else if (interaction_mode == InteractionMode::Object) {
                    const auto selected_entities = RunBoxSelect(*box_px);
                    if (!is_additive) R.clear<Selected>();
                    for (const auto e : selected_entities) R.emplace_or_replace<Selected>(e);
                }
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart) {
            BoxSelectStart.reset();
            BoxSelectEnd.reset();
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
                if (const auto hit_entity = hit_entities.front(); R.all_of<Excitable>(hit_entity)) {
                    if (const auto vertex = RunExcitableVertexPick(hit_entity, mouse_px)) {
                        R.emplace_or_replace<ExcitedVertex>(hit_entity, *vertex, 1.f);
                    }
                }
            }
        } else if (!IsMouseDown(ImGuiMouseButton_Left)) {
            R.clear<ExcitedVertex>();
        }
        return;
    }
    if (!IsSingleClicked(ImGuiMouseButton_Left)) return;

    if (interaction_mode == InteractionMode::Edit && edit_mode == Element::None && !active_is_armature) return;

    if (interaction_mode == InteractionMode::Edit && !active_is_armature) {
        const bool toggle = IsKeyDown(ImGuiMod_Shift) || IsKeyDown(ImGuiMod_Ctrl) || IsKeyDown(ImGuiMod_Super);
        const auto ranges = GetBitsetRangesForSelected();
        auto *mapped = Buffers->SelectionBitsetBuffer.GetMappedData().data();
        auto *bits = reinterpret_cast<uint32_t *>(mapped);
        if (!toggle) {
            for (const auto &range : ranges) {
                const uint32_t first_word = range.Offset / 32;
                const uint32_t last_word = (range.Offset + range.Count + 31) / 32;
                memset(mapped + first_word * sizeof(uint32_t), 0, (last_word - first_word) * sizeof(uint32_t));
                R.remove<MeshActiveElement>(range.MeshEntity);
            }
        }
        const auto hit = RunElementPickFromRanges(ranges, edit_mode, mouse_px);
        if (hit) {
            const auto [mesh_entity, element_index] = *hit;
            const auto *current_active = R.try_get<MeshActiveElement>(mesh_entity);
            const bool is_active = current_active && current_active->Handle == element_index;
            if (const auto *br = R.try_get<const MeshSelectionBitsetRange>(mesh_entity)) {
                const uint32_t global_bit = br->Offset + element_index;
                const bool was_selected = (bits[global_bit >> 5] >> (global_bit & 31)) & 1;
                if (toggle && was_selected) {
                    bits[global_bit >> 5] &= ~(1u << (global_bit & 31));
                } else {
                    bits[global_bit >> 5] |= 1u << (global_bit & 31);
                }
            }
            if (toggle && is_active) {
                R.remove<MeshActiveElement>(mesh_entity);
            } else {
                R.emplace_or_replace<MeshActiveElement>(mesh_entity, element_index);
            }
        } else if (!toggle) {
            for (const auto &range : ranges) {
                R.remove<MeshActiveElement>(range.MeshEntity);
            }
        }
        if (!ranges.empty() && (!toggle || hit)) SelectionBitsDirty = true;
    } else if (interaction_mode == InteractionMode::Object) {
        const uint32_t scaled_pick_radius = std::max(1u, uint32_t(float(ObjectSelectRadiusPx) * std::max(render_scale.x, render_scale.y) + 0.5f));
        const auto hit_entities = RunObjectPick(mouse_px, scaled_pick_radius);

        entt::entity hit = entt::null;
        if (!hit_entities.empty()) {
            auto it = find(hit_entities, active_entity);
            if (it != hit_entities.end()) ++it;
            if (it == hit_entities.end()) it = hit_entities.begin();
            hit = *it;
        }
        // In Object mode, clicking a bone sphere selects its armature object
        if (hit != entt::null && R.all_of<BoneIndex>(hit)) {
            hit = R.get<MeshInstance>(hit).MeshEntity;
        }
        if (hit != entt::null && IsKeyDown(ImGuiMod_Shift)) {
            if (active_entity == hit) {
                ToggleSelected(hit);
            } else {
                R.clear<Active>();
                R.emplace<Active>(hit);
                R.emplace_or_replace<Selected>(hit);
            }
        } else if (hit != entt::null || !IsKeyDown(ImGuiMod_Shift)) {
            Select(hit);
        }
    } else if (interaction_mode == InteractionMode::Pose || active_is_armature) {
        // Bone pick: used in Pose mode and armature Edit mode
        const uint32_t scaled_pick_radius = std::max(1u, uint32_t(float(ObjectSelectRadiusPx) * std::max(render_scale.x, render_scale.y) + 0.5f));
        const auto hit_entities = RunObjectPick(mouse_px, scaled_pick_radius);
        entt::entity hit = entt::null;
        for (const auto e : hit_entities) {
            if (R.all_of<BoneIndex>(e)) {
                hit = e;
                break;
            }
        }
        if (hit != entt::null && IsKeyDown(ImGuiMod_Shift)) {
            ToggleSelected(hit);
        } else if (hit != entt::null || !IsKeyDown(ImGuiMod_Shift)) {
            Select(hit);
        }
    }
}

void Scene::RenderOverlay() {
    const rect viewport{ToGlm(GetWindowPos()), ToGlm(GetContentRegionAvail())};
    const bool active_transform = TransformGizmo::IsUsing();
    static constexpr float OrientationGizmoSize{90};
    const OverlayIconButtonStyle overlay_button_style{};
    const float overlay_corner_gap = GetTextLineHeightWithSpacing() / 2.f;
    const OverlayIconButtonStyle shading_button_style{
        .ButtonSize = {overlay_button_style.ButtonSize.x * 0.75f, overlay_button_style.ButtonSize.y * 0.75f},
        .Padding = overlay_button_style.Padding,
        .IconScale = overlay_button_style.IconScale,
        .CornerRounding = overlay_button_style.CornerRounding * 0.75f,
    };
    { // Transform mode pill buttons (top-left overlay)
        using enum TransformGizmo::Type;
        const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
        const bool has_frozen_selected = R.view<Selected, Frozen>().begin() != R.view<Selected, Frozen>().end();
        const bool edit_transform_locked = interaction_mode == InteractionMode::Edit &&
            any_of(scene_selection::GetSelectedMeshEntities(R), [&](entt::entity mesh_entity) { return scene_selection::HasFrozenInstance(R, mesh_entity); });
        const bool transform_enabled = !edit_transform_locked;
        const bool scale_enabled = transform_enabled && !has_frozen_selected;

        auto &transform_type = MGizmo.Config.Type;
        if (!transform_enabled) transform_type = None;
        else if (!scale_enabled && transform_type == Scale) transform_type = Translate;

        const auto start_pos = std::bit_cast<ImVec2>(viewport.pos) + GetWindowContentRegionMin() + ImVec2{overlay_corner_gap, overlay_corner_gap};
        OverlayControlsHovered = false;
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

    { // Viewport shading group (top-right overlay)
        const float group_width = shading_button_style.ButtonSize.x * 4.f;
        const ImVec2 start_pos = std::bit_cast<ImVec2>(viewport.pos + vec2{GetWindowContentRegionMax().x - group_width, GetWindowContentRegionMin().y}) + ImVec2{-overlay_corner_gap, overlay_corner_gap};
        auto &settings = R.get<SceneSettings>(SceneEntity);
        const float button_w = shading_button_style.ButtonSize.x;
        const auto make_shading_button = [&](const SvgResource *icon, float x, ImDrawFlags corners, ViewportShadingMode mode, const char *tooltip) {
            return OverlayIconButtonInfo{icon, {x, 0.f}, corners, true, settings.ViewportShading == mode, tooltip};
        };
        const OverlayIconButtonInfo buttons[]{
            make_shading_button(ShadingIcons.Wireframe.get(), 0.f, ImDrawFlags_RoundCornersLeft, ViewportShadingMode::Wireframe, "Wireframe"),
            make_shading_button(ShadingIcons.Solid.get(), button_w, ImDrawFlags_RoundCornersNone, ViewportShadingMode::Solid, "Solid"),
            make_shading_button(ShadingIcons.MaterialPreview.get(), button_w * 2.f, ImDrawFlags_RoundCornersNone, ViewportShadingMode::MaterialPreview, "Material Preview"),
            make_shading_button(ShadingIcons.Rendered.get(), button_w * 3.f, ImDrawFlags_RoundCornersRight, ViewportShadingMode::Rendered, "Rendered"),
        };

        if (const auto clicked = DrawOverlayIconButtonGroup("ViewportShading", start_pos, buttons, !active_transform, &OverlayControlsHovered, shading_button_style)) {
            const auto mode = *clicked == 0 ? ViewportShadingMode::Wireframe :
                *clicked == 1               ? ViewportShadingMode::Solid :
                *clicked == 2               ? ViewportShadingMode::MaterialPreview :
                                              ViewportShadingMode::Rendered;
            settings.ViewportShading = mode;
            if (mode != ViewportShadingMode::Wireframe) settings.FillMode = mode;
            R.patch<SceneSettings>(SceneEntity, [](auto &) {});
        }
    }

    // Exit "look through" camera view if the user interacts with the orientation gizmo.
    if (!active_transform && OrientationGizmo::IsActive()) ExitLookThroughCamera();
    auto &camera = R.get<ViewCamera>(SceneEntity);
    { // Orientation gizmo (drawn before tick so camera animations it initiates begin this frame)
        const float shading_group_height = shading_button_style.ButtonSize.y;
        const auto pos = viewport.pos + vec2{GetWindowContentRegionMax().x - OrientationGizmoSize, GetWindowContentRegionMin().y} + vec2{-overlay_corner_gap, overlay_corner_gap * 2.f + shading_group_height * 1.25f};
        OrientationGizmo::Draw(pos, OrientationGizmoSize, camera, !active_transform);
    }
    if (camera.Tick()) R.patch<ViewCamera>(SceneEntity, [](auto &) {});

    const auto selected_view = R.view<const Selected>();
    const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;

    // Check if there's anything to transform:
    // - Object mode: at least one object selected
    // - Edit mode: at least one element selected within selected meshes
    const auto has_transform_target = [&]() {
        if (selected_view.empty()) return false;
        if (interaction_mode != InteractionMode::Edit) return true;
        const auto *bits = reinterpret_cast<const uint32_t *>(Buffers->SelectionBitsetBuffer.GetMappedData().data());
        for (const auto [e, mi] : R.view<const MeshInstance, const Selected>(entt::exclude<Frozen>).each()) {
            if (const auto *br = R.try_get<const MeshSelectionBitsetRange>(mi.MeshEntity)) {
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
                return node->Parent != entt::null && R.all_of<Selected>(node->Parent);
            }
            return false;
        };

        auto root_selected = selected_view | filter([&](auto e) { return !is_parent_selected(e); });
        const auto root_count = distance(root_selected);

        const auto active_entity = FindActiveEntity(R);
        const auto active_transform = [&]() -> Transform {
            if (active_entity == entt::null) return {};
            const auto &wt = R.get<WorldTransform>(active_entity);
            return {wt.Position, Vec4ToQuat(wt.Rotation), wt.Scale};
        }();
        const auto edit_transform_instances = interaction_mode == InteractionMode::Edit ?
            scene_selection::ComputePrimaryEditInstances(R, false) :
            std::unordered_map<entt::entity, entt::entity>{};

        vec3 pivot{};
        if (interaction_mode == InteractionMode::Edit) {
            // Compute world-space centroid of selected vertices once per selected mesh
            // (using a representative selected instance for world transform).
            uint32_t vertex_count = 0;
            for (const auto &[mesh_entity, instance_entity] : edit_transform_instances) {
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                const auto vertex_states = Meshes->GetVertexStates(mesh.GetStoreId());
                const auto vertices = mesh.GetVerticesSpan();
                const auto &wt = R.get<const WorldTransform>(instance_entity);
                for (uint32_t vi = 0; vi < vertex_states.size(); ++vi) {
                    if ((vertex_states[vi] & ElementStateSelected) == 0u) continue;
                    pivot += wt.Position + glm::rotate(Vec4ToQuat(wt.Rotation), wt.Scale * vertices[vi].Position);
                    ++vertex_count;
                }
            }
            if (vertex_count > 0) pivot /= float(vertex_count);
            // Apply pending transform to gizmo position (vertices aren't modified until commit).
            if (const auto *pending = R.try_get<const PendingTransform>(SceneEntity)) {
                pivot += pending->P;
            }
        } else {
            pivot = fold_left(root_selected | transform([&](auto e) { return R.get<WorldTransform>(e).Position; }), vec3{}, std::plus{}) / float(root_count);
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
                if (interaction_mode == InteractionMode::Edit) {
                    for (const auto &[_, instance_entity] : edit_transform_instances) {
                        R.emplace<StartTransform>(instance_entity, GetTransform(R, instance_entity));
                    }
                } else {
                    for (const auto e : root_selected) {
                        const auto &wt = R.get<WorldTransform>(e);
                        R.emplace<StartTransform>(e, Transform{wt.Position, Vec4ToQuat(wt.Rotation), wt.Scale});
                    }
                }
            }
            if (interaction_mode == InteractionMode::Edit) {
                // Edit mode: store pending transform for shader-based preview.
                // Actual vertex positions are only modified on commit.
                R.emplace_or_replace<PendingTransform>(SceneEntity, ts.P, ts.R, td.P, td.R, td.S);
            } else {
                // Object mode: apply transform to entity components immediately during drag.
                // StartTransform stores world P/R/S; compute new world result, then convert to local for parented entities.
                const auto r = ts.R, rT = glm::conjugate(r);
                for (const auto &[e, ts_e_comp] : start_transform_view.each()) {
                    const auto &ts_e = ts_e_comp.T; // world P/R/S at start
                    const bool frozen = R.all_of<Frozen>(e);
                    const auto offset = ts_e.P - ts.P;
                    const vec3 new_world_p = td.P + ts.P + glm::rotate(td.R, frozen ? offset : r * (rT * offset * td.S));
                    const quat new_world_r = glm::normalize(td.R * ts_e.R);
                    const vec3 new_world_s = frozen ? ts_e.S : td.S * ts_e.S;

                    if (const auto *node = R.try_get<SceneNode>(e); node && node->Parent != entt::null) {
                        if (const auto *pi = R.try_get<ParentInverse>(e)) {
                            const auto &parent_wt = R.get<WorldTransform>(node->Parent);
                            const mat4 parent_delta = ToMatrix(parent_wt) * pi->M;
                            const vec3 new_local_p = vec3(glm::inverse(parent_delta) * vec4(new_world_p, 1.f));
                            const mat3 rs{parent_delta};
                            const vec3 parent_delta_s{glm::length(rs[0]), glm::length(rs[1]), glm::length(rs[2])};
                            const quat parent_delta_r = glm::normalize(glm::quat_cast(mat3{rs[0] / parent_delta_s.x, rs[1] / parent_delta_s.y, rs[2] / parent_delta_s.z}));
                            const quat new_local_r = glm::conjugate(parent_delta_r) * new_world_r;
                            const vec3 new_local_s = new_world_s / parent_delta_s;
                            SetTransform(R, e, {.P = new_local_p, .R = new_local_r, .S = new_local_s});
                            continue;
                        }
                    }
                    SetTransform(R, e, {.P = new_world_p, .R = new_world_r, .S = new_world_s});
                }
            }
        } else if (!start_transform_view.empty()) {
            R.clear<StartTransform>(); // Transform ended - triggers commit in ProcessComponentEvents.
        }

        // Render gizmo at the post-delta position so it matches the applied transform.
        auto render_transform = gizmo_transform;
        if (interact_result) render_transform.P = interact_result->Start.P + interact_result->Delta.P;
        TransformGizmo::Render(render_transform, MGizmo.Config.Type, camera, viewport);
    }

    if (!R.storage<Selected>().empty()) { // Draw center-dot for active/selected entities
        const auto &theme = R.get<const ViewportTheme>(SceneEntity);
        const auto vp = camera.Projection(viewport.size.x / viewport.size.y) * camera.View();
        for (const auto [e, wt] : R.view<const WorldTransform>().each()) {
            if (!R.any_of<Active, Selected>(e)) continue;

            const auto p_cs = vp * vec4{wt.Position, 1.f}; // World to clip space
            const auto p_ndc = fabsf(p_cs.w) > FLT_EPSILON ? vec3{p_cs} / p_cs.w : vec3{p_cs}; // Clip space to NDC
            const auto p_uv = vec2{p_ndc.x + 1, 1 - p_ndc.y} * 0.5f; // NDC to UV [0,1] (top-left origin)
            const auto p_px = std::bit_cast<ImVec2>(viewport.pos + p_uv * viewport.size); // UV to px
            auto &dl = *GetWindowDrawList();
            dl.AddCircleFilled(p_px, 3.5f, colors::RgbToU32(R.all_of<Active>(e) ? theme.Colors.ObjectActive : theme.Colors.ObjectSelected), 10);
            dl.AddCircle(p_px, 3.5f, IM_COL32(0, 0, 0, 255), 10, 1.f);
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
            dl.AddLine({x, box_min.y}, {glm::min(x + dash_size, box_max.x), box_min.y}, outline_color, 1.0f);
        }
        // Bottom
        for (float x = box_min.x; x < box_max.x; x += dash_size + gap_size) {
            dl.AddLine({x, box_max.y}, {glm::min(x + dash_size, box_max.x), box_max.y}, outline_color, 1.0f);
        }
        // Left
        for (float y = box_min.y; y < box_max.y; y += dash_size + gap_size) {
            dl.AddLine({box_min.x, y}, {box_min.x, glm::min(y + dash_size, box_max.y)}, outline_color, 1.0f);
        }
        // Right
        for (float y = box_min.y; y < box_max.y; y += dash_size + gap_size) {
            dl.AddLine({box_max.x, y}, {box_max.x, glm::min(y + dash_size, box_max.y)}, outline_color, 1.0f);
        }
    }

    // Camera look-through frame overlay: show the active camera's view as a centered frame.
    // The ViewCamera's FOV is widened so the active camera's view fits inside with padding.
    // The frame marks exactly what the active camera captures.
    if (SavedViewCamera && !camera.IsAnimating()) {
        const auto active_entity = FindActiveEntity(R);
        if (const auto *cd = active_entity != entt::null ? R.try_get<Camera>(active_entity) : nullptr) {
            const float cam_aspect = AspectRatio(*cd);
            const float frame_ratio = LookThroughFrameRatio(cam_aspect, viewport.size.x / viewport.size.y);
            const vec2 frame_size{viewport.size.y * frame_ratio * cam_aspect, viewport.size.y * frame_ratio};
            const vec2 vp_center = viewport.pos + viewport.size * 0.5f;
            const vec2 fmin = vp_center - frame_size * 0.5f, fmax = vp_center + frame_size * 0.5f;
            const auto vmin = viewport.pos, vmax = viewport.pos + viewport.size;

            auto &dl = *GetWindowDrawList();
            static constexpr auto dim = IM_COL32(0, 0, 0, 100);
            auto iv = [](vec2 v) { return std::bit_cast<ImVec2>(v); };
            dl.AddRectFilled(iv(vmin), iv({vmax.x, fmin.y}), dim);
            dl.AddRectFilled(iv({vmin.x, fmax.y}), iv(vmax), dim);
            dl.AddRectFilled(iv({vmin.x, fmin.y}), iv({fmin.x, fmax.y}), dim);
            dl.AddRectFilled(iv({fmax.x, fmin.y}), iv({vmax.x, fmax.y}), dim);
            dl.AddRect(iv(fmin), iv(fmax), IM_COL32(255, 255, 255, 140), 0.f, 0, 1.5f);
        }
    }

    { // Viewport info overlay
        const auto &settings = R.get<const SceneSettings>(SceneEntity);
        const auto *mode_name = settings.ViewportShading == ViewportShadingMode::Wireframe ? "Wireframe" : settings.ViewportShading == ViewportShadingMode::Solid ? "Solid" :
            settings.ViewportShading == ViewportShadingMode::MaterialPreview                                                                                      ? "Material Preview" :
                                                                                                                                                                    "Rendered";
        const auto text = std::format("Shading: {}", mode_name);
        const auto text_size = CalcTextSize(text.c_str());
        const auto text_pos = std::bit_cast<ImVec2>(viewport.pos + viewport.size - vec2{10.f, 10.f}) - text_size;
        auto &dl = *GetWindowDrawList();
        dl.AddRectFilled(
            {text_pos.x - 6.f, text_pos.y - 4.f},
            {text_pos.x + text_size.x + 6.f, text_pos.y + text_size.y + 4.f},
            IM_COL32(0, 0, 0, 110),
            4.f
        );
        dl.AddText(text_pos, IM_COL32(230, 230, 230, 255), text.c_str());
    }

    StartScreenTransform = {};
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

    if (const auto *mesh_instance = R.try_get<MeshInstance>(active_entity)) {
        Text("Mesh entity: %s", GetName(R, mesh_instance->MeshEntity).c_str());
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
    const auto *active_mesh_instance = R.try_get<MeshInstance>(active_entity);
    if (active_mesh_instance) {
        const auto active_mesh_entity = active_mesh_instance->MeshEntity;
        const auto &active_mesh = R.get<const Mesh>(active_mesh_entity);
        TextUnformatted(
            std::format("Vertices | Edges | Faces: {:L} | {:L} | {:L}", active_mesh.VertexCount(), active_mesh.EdgeCount(), active_mesh.FaceCount()).c_str()
        );
    } else if (const auto *armature_object = R.try_get<ArmatureObject>(active_entity)) {
        const auto &armature = R.get<const Armature>(armature_object->Entity);
        Text("Bones: %zu", armature.Bones.size());
        const auto cur_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
        const bool show_bones = cur_mode == InteractionMode::Pose || cur_mode == InteractionMode::Edit;
        if (show_bones && CollapsingHeader("Bones", ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto active_bone = FindActiveEntity(R);
            for (const auto [b, mi, bi] : R.view<MeshInstance, const BoneIndex>().each()) {
                if (mi.MeshEntity != active_entity) continue;
                const bool is_active_bone = (b == active_bone);
                if (is_active_bone) PushStyleColor(ImGuiCol_Text, ImVec4{1, 0.8f, 0.2f, 1});
                if (Selectable(armature.Bones[bi.Index].Name.c_str(), R.all_of<Selected>(b))) Select(b);
                if (is_active_bone) PopStyleColor();
            }
            if (Button("Reset Pose")) {
                for (const auto [b, mi, bi] : R.view<MeshInstance, const BoneIndex>().each()) {
                    if (mi.MeshEntity != active_entity) continue;
                    const auto &rest = armature.Bones[bi.Index].RestLocal;
                    R.replace<Position>(b, rest.P);
                    R.replace<Rotation>(b, rest.R);
                }
                UpdateWorldTransform(R, active_entity);
            }
        }
    }
    Unindent();
    if (CollapsingHeader("Transform")) {
        auto &position = R.get<Position>(active_entity).Value;
        bool model_changed = DragFloat3("Position", &position[0], 0.01f);
        if (model_changed) R.patch<Position>(active_entity, [](auto &) {});
        // Rotation editor
        {
            int mode_i = R.get<const RotationUiVariant>(active_entity).index();
            const char *modes[]{"Quat (WXYZ)", "XYZ Euler", "Axis Angle"};
            if (Combo("Rotation mode", &mode_i, modes, IM_ARRAYSIZE(modes))) {
                R.replace<RotationUiVariant>(active_entity, CreateVariantByIndex<RotationUiVariant>(mode_i));
                SetRotation(R, active_entity, R.get<const Rotation>(active_entity).Value);
            }
        }
        auto &rotation_ui = R.get<RotationUiVariant>(active_entity);
        const bool rotation_changed = std::visit(
            overloaded{
                [&](RotationQuat &v) {
                    if (DragFloat4("Rotation (quat WXYZ)", &v.Value[0], 0.01f)) {
                        R.replace<Rotation>(active_entity, glm::normalize(v.Value));
                        return true;
                    }
                    return false;
                },
                [&](RotationEuler &v) {
                    if (DragFloat3("Rotation (XYZ Euler, deg)", &v.Value[0], 1.f)) {
                        const auto rads = glm::radians(v.Value);
                        R.replace<Rotation>(active_entity, glm::normalize(glm::quat_cast(glm::eulerAngleXYZ(rads.x, rads.y, rads.z))));
                        return true;
                    }
                    return false;
                },
                [&](RotationAxisAngle &v) {
                    bool changed = DragFloat3("Rotation axis (XYZ)", &v.Value[0], 0.01f);
                    changed |= DragFloat("Angle (deg)", &v.Value.w, 1.f);
                    if (changed) {
                        const auto axis = glm::normalize(vec3{v.Value});
                        const auto angle = glm::radians(v.Value.w);
                        R.replace<Rotation>(active_entity, glm::normalize(quat{std::cos(angle / 2), axis * std::sin(angle / 2)}));
                        return true;
                    }
                    return false;
                },
            },
            rotation_ui
        );
        if (rotation_changed) {
            R.patch<RotationUiVariant>(active_entity, [](auto &) {});
        }
        model_changed |= rotation_changed;

        const bool frozen = R.all_of<Frozen>(active_entity);
        if (frozen) BeginDisabled();
        const auto label = std::format("Scale{}", frozen ? " (frozen)" : "");
        auto &scale = R.get<Scale>(active_entity).Value;
        const bool scale_changed = DragFloat3(label.c_str(), &scale[0], 0.01f, 0.01f, 10);
        if (scale_changed) R.patch<Scale>(active_entity, [](auto &) {});
        model_changed |= scale_changed;
        if (frozen) EndDisabled();
        if (model_changed) {
            UpdateWorldTransform(R, active_entity);
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
            Text("Position: %.3f, %.3f, %.3f", wt.Position.x, wt.Position.y, wt.Position.z);
            Text("Rotation: %.3f, %.3f, %.3f, %.3f", wt.Rotation.x, wt.Rotation.y, wt.Rotation.z, wt.Rotation.w);
            Text("Scale: %.3f, %.3f, %.3f", wt.Scale.x, wt.Scale.y, wt.Scale.z);
            TreePop();
        }
    }
    if (active_mesh_instance) {
        const auto active_mesh_entity = active_mesh_instance->MeshEntity;
        if (const auto *primitive_type = R.try_get<PrimitiveType>(active_mesh_entity)) {
            const bool frozen = scene_selection::HasFrozenInstance(R, active_mesh_entity);
            if (frozen) BeginDisabled();
            const auto update_label = std::format("Update primitive{}", frozen ? " (frozen)" : "");
            if (CollapsingHeader(update_label.c_str()) && !frozen) {
                if (auto primitive_mesh = PrimitiveEditor(*primitive_type, false)) {
                    SetMeshPositions(active_mesh_entity, std::move(primitive_mesh->Positions));
                }
            }
            if (frozen) EndDisabled();
        }

        if (CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
            auto &material_store = R.get<MaterialStore>(SceneEntity);
            auto &texture_store = *Textures;
            const auto &active_mesh = R.get<const Mesh>(active_mesh_entity);
            std::span<const uint32_t> primitive_materials = Meshes->GetPrimitiveMaterialIndices(active_mesh.GetStoreId());
            const auto material_count = GetMaterialCount(*Buffers);
            const auto material_name = [&](uint32_t index) {
                if (index < material_store.Names.size() && !material_store.Names[index].empty()) return std::string{material_store.Names[index]};
                return std::format("Material{}", index);
            };
            if (primitive_materials.empty()) {
                TextUnformatted("No material slots on this mesh.");
            } else if (material_count == 0) {
                TextUnformatted("No materials.");
            } else {
                auto &slot_selection = R.get_or_emplace<MeshMaterialSlotSelection>(active_mesh_entity);
                bool slot_selection_changed = false;
                if (const auto max_primitive = uint32_t(primitive_materials.size() - 1);
                    slot_selection.PrimitiveIndex > max_primitive) {
                    slot_selection.PrimitiveIndex = max_primitive;
                    slot_selection_changed = true;
                }

                BeginChild("MaterialSlots", ImVec2(0, 110), true);
                for (uint32_t primitive_index = 0; primitive_index < primitive_materials.size(); ++primitive_index) {
                    const uint32_t material_index = std::min(primitive_materials[primitive_index], material_count - 1);
                    const auto label = std::format("Slot {:L}: {}", primitive_index, material_name(material_index));
                    if (Selectable(label.c_str(), slot_selection.PrimitiveIndex == primitive_index)) {
                        slot_selection.PrimitiveIndex = primitive_index;
                        slot_selection_changed = true;
                    }
                }
                EndChild();
                if (slot_selection_changed) R.patch<MeshMaterialSlotSelection>(active_mesh_entity, [](auto &) {});

                const auto *pending_assignment = R.try_get<const MeshMaterialAssignment>(active_mesh_entity);
                uint32_t material_index = pending_assignment && pending_assignment->PrimitiveIndex == slot_selection.PrimitiveIndex ?
                    pending_assignment->MaterialIndex :
                    primitive_materials[slot_selection.PrimitiveIndex];
                material_index = std::min(material_index, material_count - 1);
                const auto assigned_material_name = material_name(material_index);
                if (BeginCombo("Assigned material", assigned_material_name.c_str())) {
                    for (uint32_t i = 0; i < material_count; ++i) {
                        const auto option_name = material_name(i);
                        if (Selectable(option_name.c_str(), material_index == i)) {
                            material_index = i;
                            R.emplace_or_replace<MeshMaterialAssignment>(active_mesh_entity, slot_selection.PrimitiveIndex, i);
                        }
                    }
                    EndCombo();
                }

                auto &material = GetMaterial(*Buffers, material_index);
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

                if (pbr_features_changed) {
                    if (pbr_features_mask != 0u) R.emplace_or_replace<PbrMeshFeatures>(active_mesh_entity, pbr_features_mask);
                    else R.remove<PbrMeshFeatures>(active_mesh_entity);
                }
                if (material_changed) R.emplace_or_replace<MaterialDirty>(SceneEntity, material_index);
            }
        }
    }
    if (auto *cd = R.try_get<Camera>(active_entity)) {
        if (CollapsingHeader("Camera")) {
            if (RenderCameraLensEditor(*cd)) R.patch<Camera>(active_entity, [](auto &) {});
            Separator();
            if (SavedViewCamera) {
                if (Button("Exit camera view")) ExitLookThroughCamera();
            } else {
                if (Button("Look through")) {
                    SavedViewCamera = R.get<ViewCamera>(SceneEntity);
                    AnimateToCamera(active_entity);
                }
            }
        }
    }
    if (R.all_of<LightIndex>(active_entity) &&
        CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        constexpr float MaxLightIntensity{1000.f}, MaxLightRange{1000.f};
        auto light = Buffers->GetLight(R.get<const LightIndex>(active_entity).Value);
        bool changed{false}, wireframe_changed{false};

        const char *type_names[]{"Directional", "Point", "Spot"};
        int type_i = std::clamp(static_cast<int>(light.Type), 0, 2);
        if (Combo("Type", &type_i, type_names, IM_ARRAYSIZE(type_names))) {
            auto next = SceneDefaults::MakePunctualLight(static_cast<PunctualLightType>(type_i));
            next.TransformSlotOffset = light.TransformSlotOffset;
            next.Color = light.Color;
            next.Intensity = light.Intensity;
            light = next;
            changed = true;
            wireframe_changed = true;
        }

        changed |= ColorEdit3("Color", &light.Color.x);
        changed |= SliderFloat("Intensity", &light.Intensity, 0.f, MaxLightIntensity, "%.2f");
        if (light.Type == PunctualLightType::Point || light.Type == PunctualLightType::Spot) {
            bool infinite_range = light.Range <= 0.f;
            if (Checkbox("Infinite range", &infinite_range)) {
                light.Range = infinite_range ? 0.f : std::max(light.Range, SceneDefaults::PointRange);
                changed = true;
                wireframe_changed = true;
            }
            if (!infinite_range) {
                if (SliderFloat("Range", &light.Range, 0.01f, MaxLightRange, "%.2f")) {
                    changed = true;
                    wireframe_changed = true;
                }
            }
        }
        if (light.Type == PunctualLightType::Spot) {
            float outer_deg = std::clamp(glm::degrees(AngleFromCos(light.OuterConeCos)), 0.f, 90.f);
            float inner_deg = std::clamp(glm::degrees(AngleFromCos(light.InnerConeCos)), 0.f, outer_deg);
            float blend = outer_deg > 1e-4f ? std::clamp(1.f - inner_deg / outer_deg, 0.f, 1.f) : 0.f;
            if (SliderFloat("Size", &outer_deg, 0.f, 90.f, "%.1f deg")) {
                outer_deg = std::clamp(outer_deg, 0.f, 90.f);
                const float outer_rad = glm::radians(outer_deg);
                const float inner_rad = outer_rad * (1.f - blend);
                light.OuterConeCos = std::cos(outer_rad);
                light.InnerConeCos = std::cos(inner_rad);
                changed = true;
                wireframe_changed = true;
            }
            if (SliderFloat("Blend", &blend, 0.f, 1.f, "%.2f")) {
                blend = std::clamp(blend, 0.f, 1.f);
                const float outer_rad = glm::radians(std::clamp(outer_deg, 0.f, 90.f));
                const float inner_rad = outer_rad * (1.f - blend);
                light.OuterConeCos = std::cos(outer_rad);
                light.InnerConeCos = std::cos(inner_rad);
                changed = true;
                wireframe_changed = true;
            }
        }
        if (changed) {
            Buffers->SetLight(R.get<const LightIndex>(active_entity).Value, light);
            R.emplace_or_replace<SubmitDirty>(active_entity);
        }
        if (wireframe_changed) R.emplace_or_replace<LightWireframeDirty>(active_entity);
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
                const bool active_is_armature_rc = active_entity_rc != entt::null &&
                    (R.all_of<ArmatureObject>(active_entity_rc) || R.all_of<BoneIndex>(active_entity_rc));
                const bool edit_allowed = AllSelectedAreMeshes(R) || active_is_armature_rc;
                const bool pose_allowed = active_is_armature_rc;
                for (const auto mode : InteractionModes) {
                    if (mode == InteractionMode::Edit && !edit_allowed) continue;
                    if (mode == InteractionMode::Pose && !pose_allowed) continue;
                    SameLine();
                    interaction_mode_changed |= RadioButton(to_string(mode).c_str(), &interaction_mode_value, int(mode));
                }
                if (interaction_mode_changed) SetInteractionMode(InteractionMode(interaction_mode_value));
                if (interaction_mode == InteractionMode::Edit || interaction_mode == InteractionMode::Excite) {
                    Checkbox("Orbit to active", &OrbitToActive);
                }
                if (interaction_mode == InteractionMode::Edit) {
                    Checkbox("X-ray selection", &SelectionXRay);
                }
                if (interaction_mode == InteractionMode::Edit) {
                    AlignTextToFramePadding();
                    TextUnformatted("Edit mode:");
                    auto type_interaction_mode = int(edit_mode);
                    for (const auto element : Elements) {
                        auto name = Capitalize(label(element));
                        SameLine();
                        if (RadioButton(name.c_str(), &type_interaction_mode, int(element))) {
                            SetEditMode(element);
                        }
                    }
                    const auto active_entity = FindActiveEntity(R);
                    if (active_entity != entt::null) {
                        if (const auto *mesh_instance = R.try_get<MeshInstance>(active_entity)) {
                            const auto *br = R.try_get<const MeshSelectionBitsetRange>(mesh_instance->MeshEntity);
                            const uint32_t selected_count = br ? scene_selection::CountSelected(
                                                                     reinterpret_cast<const uint32_t *>(Buffers->SelectionBitsetBuffer.GetMappedData().data()),
                                                                     br->Offset, br->Count
                                                                 ) :
                                                                 0;
                            Text("Editing %s: %u selected", label(edit_mode).data(), selected_count);
                        } else {
                            TextUnformatted("Edit mode requires an active mesh object.");
                        }
                    }
                }
                PopID();
            }
            if (CollapsingHeader("Object tree", ImGuiTreeNodeFlags_DefaultOpen)) {
                RenderObjectTree();
            }
            if (!R.storage<Selected>().empty()) {
                SeparatorText("Selection actions");
                std::vector<entt::entity> selected_mesh_instances;
                for (const auto entity : R.view<const Selected, const MeshInstance>()) selected_mesh_instances.emplace_back(entity);

                if (!selected_mesh_instances.empty()) {
                    const bool any_visible = any_of(selected_mesh_instances, [&](entt::entity e) { return R.all_of<RenderInstance>(e); });
                    const bool any_hidden = any_of(selected_mesh_instances, [&](entt::entity e) { return !R.all_of<RenderInstance>(e); });
                    const bool mixed_visible = any_visible && any_hidden;
                    if (mixed_visible) PushItemFlag(ImGuiItemFlags_MixedValue, true);
                    if (bool set_visible = any_visible && !any_hidden; Checkbox("Visible", &set_visible)) {
                        for (const auto e : selected_mesh_instances) SetVisible(e, set_visible);
                    }
                    if (mixed_visible) PopItemFlag();
                }
                if (Button("Duplicate")) Duplicate();
                SameLine();
                if (Button("Duplicate linked")) DuplicateLinked();
                if (Button("Delete")) Delete();
            }
            RenderEntityControls(FindActiveEntity(R));

            if (CollapsingHeader("Add object")) {
                TextDisabled("Shortcuts: Ctrl+Shift+E (Empty), Ctrl+Shift+A (Armature), Ctrl+Shift+C (Camera)");
                if (Button("Add Empty")) {
                    AddEmpty({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
                SameLine();
                if (Button("Add Armature")) {
                    AddArmature({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
                SameLine();
                if (Button("Add Camera")) {
                    AddCamera({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
                SameLine();
                if (Button("Add Light")) {
                    AddLight({.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive});
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
            }

            if (CollapsingHeader("Add primitive")) {
                PushID("AddPrimitive");
                static auto selected_type_i = int(PrimitiveType::Cube);
                for (uint i = 0; i < PrimitiveTypes.size(); ++i) {
                    if (i % 3 != 0) SameLine();
                    RadioButton(ToString(PrimitiveTypes[i]).c_str(), &selected_type_i, i);
                }
                const auto selected_type = PrimitiveType(selected_type_i);
                if (auto primitive_mesh = PrimitiveEditor(selected_type, true)) {
                    R.emplace<PrimitiveType>(AddMesh(std::move(*primitive_mesh), MeshInstanceCreateInfo{.Name = ToString(selected_type)}).first, selected_type);
                    StartScreenTransform = TransformGizmo::TransformType::Translate;
                }
                PopID();
            }
            EndTabItem();
        }

        if (BeginTabItem("Render")) {
            auto &settings = R.get<SceneSettings>(SceneEntity);
            auto &mat_preview_lighting = R.get<MaterialPreviewLighting>(SceneEntity);
            auto &rendered_lighting = R.get<RenderedLighting>(SceneEntity);
            bool settings_changed = false;
            bool mat_preview_changed = false, rendered_changed = false;
            if (ColorEdit3("Background color", settings.ClearColor.float32)) {
                settings.ClearColor.float32[3] = 1.f;
                settings_changed = true;
            }
            bool show_grid = settings.ShowGrid;
            if (Checkbox("Show grid", &show_grid)) {
                settings.ShowGrid = show_grid;
                settings_changed = true;
            }
            if (Button("Recompile shaders")) ShaderRecompileRequested = true;
            SeparatorText("Viewport shading");
            TextUnformatted("Use the viewport icons in the top-right.");
            const auto current_mode = settings.ViewportShading;

            bool smooth_shading_changed = false;
            if (current_mode != ViewportShadingMode::Wireframe) {
                bool smooth_shading = settings.SmoothShading;
                if (Checkbox("Smooth shading", &smooth_shading)) {
                    settings.SmoothShading = smooth_shading;
                    smooth_shading_changed = true;
                }
            }

            const auto render_pbr_controls = [&](PBRViewportLighting &lighting, bool &lighting_changed, const char *id) {
                PushID(id);
                if (Button("Reset")) {
                    lighting = {false, false, 1.f, 0.f, 0.5f, 0.f};
                    lighting_changed = true;
                }
                lighting_changed |= Checkbox("Scene lights", &lighting.UseSceneLights);
                SameLine();
                lighting_changed |= Checkbox("Scene world", &lighting.UseSceneWorld);
                if (!lighting.UseSceneWorld) {
                    auto &environments = *Environments;
                    const auto &current_name = environments.Hdris[environments.ActiveHdriIndex].Name;
                    if (BeginCombo("Environment", current_name.c_str())) {
                        for (uint32_t i = 0; i < uint32_t(environments.Hdris.size()); ++i) {
                            const bool selected = (i == environments.ActiveHdriIndex);
                            if (Selectable(environments.Hdris[i].Name.c_str(), selected)) {
                                SetStudioEnvironment(i);
                                lighting_changed = true;
                            }
                            if (selected) SetItemDefaultFocus();
                        }
                        EndCombo();
                    }
                    lighting_changed |= SliderFloat("Intensity", &lighting.EnvIntensity, 0.f, 2.f, "%.2f");
                    lighting_changed |= SliderFloat("Rotation", &lighting.EnvRotationDegrees, -180.f, 180.f, "%.1f deg");
                    lighting_changed |= SliderFloat("Blur", &lighting.BackgroundBlur, 0.f, 1.f, "%.2f");
                    lighting_changed |= SliderFloat("World opacity", &lighting.WorldOpacity, 0.f, 1.f, "%.2f");
                }
                PopID();
            };

            if (current_mode == ViewportShadingMode::MaterialPreview) {
                SeparatorText("Material Preview lighting");
                render_pbr_controls(mat_preview_lighting, mat_preview_changed, "MatPreviewLighting");
            }
            if (current_mode == ViewportShadingMode::Rendered) {
                SeparatorText("Rendered lighting");
                render_pbr_controls(rendered_lighting, rendered_changed, "RenderedLighting");
            }

            auto color_mode = int(settings.FaceColorMode);
            bool color_mode_changed = false;
            if (current_mode == ViewportShadingMode::Solid) {
                PushID("FaceColorMode");
                AlignTextToFramePadding();
                TextUnformatted("Fill color mode");
                SameLine();
                color_mode_changed |= RadioButton("Mesh", &color_mode, int(FaceColorMode::Mesh));
                SameLine();
                color_mode_changed |= RadioButton("Normals", &color_mode, int(FaceColorMode::Normals));
                PopID();
            }
            if (color_mode_changed) {
                settings.FaceColorMode = FaceColorMode(color_mode);
                settings_changed = true;
            }
            if (smooth_shading_changed) settings_changed = true;
            if (!R.view<Selected>().empty()) {
                SeparatorText("Selection overlays");
                AlignTextToFramePadding();
                TextUnformatted("Normals");
                for (const auto element : NormalElements) {
                    SameLine();
                    bool show = ElementMaskContains(settings.NormalOverlays, element);
                    const auto type_name = Capitalize(label(element));
                    if (Checkbox(type_name.c_str(), &show)) {
                        SetElementMask(settings.NormalOverlays, element, show);
                        settings_changed = true;
                    }
                }
                bool show_bboxes = settings.ShowBoundingBoxes;
                if (Checkbox("Bounding boxes", &show_bboxes)) {
                    settings.ShowBoundingBoxes = show_bboxes;
                    settings_changed = true;
                }
            }
            {
                SeparatorText("Viewport theme");
                auto &theme = R.get<ViewportTheme>(SceneEntity);
                bool changed{false};
                if (Button("Reset##ViewportTheme")) {
                    theme = SceneDefaults::ViewportTheme;
                    changed = true;
                }
                changed |= ColorEdit3("Wire", &theme.Colors.Wire.x);
                changed |= ColorEdit3("Wire edit", &theme.Colors.WireEdit.x);
                changed |= ColorEdit3("Face normal", &theme.Colors.FaceNormal.x);
                changed |= ColorEdit3("Vertex normal", &theme.Colors.VertexNormal.x);
                changed |= ColorEdit3("Vertex", &theme.Colors.Vertex.x);
                changed |= ColorEdit3("Vertex selected", &theme.Colors.VertexSelected.x);
                changed |= ColorEdit3("Edge selected (incidental)", &theme.Colors.EdgeSelectedIncidental.x);
                changed |= ColorEdit3("Edge selected", &theme.Colors.EdgeSelected.x);
                changed |= ColorEdit4("Face selected (incidental)", &theme.Colors.FaceSelectedIncidental.x);
                changed |= ColorEdit4("Face selected", &theme.Colors.FaceSelected.x);
                changed |= ColorEdit4("Element active", &theme.Colors.ElementActive.x);
                changed |= ColorEdit3("Object active", &theme.Colors.ObjectActive.x);
                changed |= ColorEdit3("Object selected", &theme.Colors.ObjectSelected.x);
                changed |= ColorEdit3("Transform", &theme.Colors.Transform.x);
                changed |= SliderFloat("Edge width", &theme.EdgeWidth, 0.5f, 2.0f);
                changed |= MeshEditor::SliderUInt("Silhouette edge width", &theme.SilhouetteEdgeWidth, 1, 4);
                if (changed) R.patch<ViewportTheme>(SceneEntity, [](auto &) {});
            }
            if (settings_changed) R.patch<SceneSettings>(SceneEntity, [](auto &) {});
            if (mat_preview_changed) {
                mat_preview_lighting.EnvIntensity = std::max(0.f, mat_preview_lighting.EnvIntensity);
                mat_preview_lighting.BackgroundBlur = std::clamp(mat_preview_lighting.BackgroundBlur, 0.f, 1.f);
                R.patch<MaterialPreviewLighting>(SceneEntity, [](auto &) {});
            }
            if (rendered_changed) {
                rendered_lighting.EnvIntensity = std::max(0.f, rendered_lighting.EnvIntensity);
                rendered_lighting.BackgroundBlur = std::clamp(rendered_lighting.BackgroundBlur, 0.f, 1.f);
                R.patch<RenderedLighting>(SceneEntity, [](auto &) {});
            }
            EndTabItem();
        }

        if (BeginTabItem("Camera")) {
            auto &camera = R.get<ViewCamera>(SceneEntity);
            bool changed = false;
            const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
            const float viewport_aspect = extent.width == 0 || extent.height == 0 ? 1.f : float(extent.width) / float(extent.height);
            if (Button("Reset##Camera")) {
                camera = SceneDefaults::ViewCamera;
                changed = true;
            }
            changed |= SliderFloat3("Target", &camera.Target.x, -10, 10);
            changed |= RenderCameraLensEditor(camera.Data, ViewportContext{.Distance = camera.Distance, .AspectRatio = viewport_aspect});
            if (changed) R.patch<ViewCamera>(SceneEntity, [](auto &camera) { camera.StopMoving(); });
            EndTabItem();
        }

        // Note: Rendered world/light toggles are in Render -> Viewport shading.
        if (BeginTabItem("Lighting")) {
            auto &lights = GetWorkspaceLights(*Buffers);
            bool changed = false;
            if (Button("Reset##Lighting")) {
                lights = SceneDefaults::WorkspaceLights;
                changed = true;
            }
            changed |= ColorEdit3("Color##View", &lights.ViewColor[0]);
            changed |= SliderFloat("Intensity##Ambient", &lights.AmbientIntensity, 0, 1);
            changed |= SliderFloat3("Direction##Directional", &lights.Direction[0], -1, 1);
            changed |= ColorEdit3("Color##Directional", &lights.DirectionalColor[0]);
            changed |= SliderFloat("Intensity##Directional", &lights.DirectionalIntensity, 0, 1);
            if (changed) R.emplace_or_replace<SubmitDirty>(SceneEntity);
            EndTabItem();
        }
        EndTabBar();
    }
}

void Scene::RenderObjectTree() {
    PushStyleVar(ImGuiStyleVar_ItemSpacing, {GetStyle().ItemSpacing.x, 0.f});

    const auto ToSelectionUserData = [](entt::entity e) -> ImGuiSelectionUserData { return ImGuiSelectionUserData(uint32_t(e)); };
    const auto FromSelectionUserData = [&](ImGuiSelectionUserData data) -> entt::entity {
        if (data == ImGuiSelectionUserData_Invalid) return entt::null;
        const auto e = entt::entity(uint32_t(data));
        return R.valid(e) ? e : entt::null;
    };

    const auto GetObjectType = [&](entt::entity e) {
        if (R.all_of<ObjectKind>(e)) return R.get<const ObjectKind>(e).Value;
        if (R.all_of<MeshInstance>(e)) return ObjectType::Mesh;
        return ObjectType::Empty;
    };
    const auto SetSelectedState = [&](entt::entity e, bool selected) {
        if (e == entt::null) return;
        if (selected) {
            if (!R.all_of<Selected>(e)) R.emplace<Selected>(e);
        } else {
            if (R.all_of<Selected>(e)) R.remove<Selected>(e);
            if (R.all_of<Active>(e)) R.remove<Active>(e);
        }
    };

    std::vector<entt::entity> visible_entities;
    const auto ApplySelectionRequests = [&](std::span<const ImGuiSelectionRequest> requests, ImGuiSelectionUserData nav_item) {
        for (const auto &request : requests) {
            if (request.Type == ImGuiSelectionRequestType_SetAll) {
                if (request.Selected) {
                    for (const auto e : visible_entities) SetSelectedState(e, true);
                } else {
                    R.clear<Selected>();
                    R.clear<Active>();
                }
                continue;
            }
            if (request.Type != ImGuiSelectionRequestType_SetRange) continue;

            const auto first = FromSelectionUserData(request.RangeFirstItem), last = FromSelectionUserData(request.RangeLastItem);
            const auto first_it = std::ranges::find(visible_entities, first), last_it = std::ranges::find(visible_entities, last);
            if (first_it == visible_entities.end() || last_it == visible_entities.end()) {
                SetSelectedState(first, request.Selected);
                SetSelectedState(last, request.Selected);
                continue;
            }
            const auto first_i = std::ranges::distance(visible_entities.begin(), first_it);
            const auto last_i = std::ranges::distance(visible_entities.begin(), last_it);
            const auto [i0, i1] = std::minmax(first_i, last_i);
            for (auto i = i0; i <= i1; ++i) SetSelectedState(visible_entities[i], request.Selected);
        }

        const auto active_entity = FindActiveEntity(R);
        const auto nav_entity = FromSelectionUserData(nav_item);
        if (nav_entity != entt::null && R.all_of<Selected>(nav_entity)) {
            if (active_entity != nav_entity) {
                if (active_entity != entt::null) R.remove<Active>(active_entity);
                R.emplace_or_replace<Active>(nav_entity);
            }
        } else if (R.storage<Selected>().empty()) {
            if (active_entity != entt::null) R.remove<Active>(active_entity);
        } else if (active_entity != entt::null && !R.all_of<Selected>(active_entity)) {
            R.remove<Active>(active_entity);
        }
    };

    auto *ms_begin = BeginMultiSelect(ImGuiMultiSelectFlags_None, int(R.storage<Selected>().size()), -1);
    std::vector<ImGuiSelectionRequest> begin_requests;
    begin_requests.reserve(ms_begin->Requests.Size);
    for (const auto &request : ms_begin->Requests) begin_requests.emplace_back(request);
    const auto begin_nav_item = ms_begin->NavIdItem;

    // Build the set of ancestors of any selected entity (for secondary highlight).
    std::unordered_set<entt::entity> ancestor_of_selected;
    for (const auto selected_entity : R.view<Selected>()) {
        const auto *n = R.try_get<SceneNode>(selected_entity);
        auto parent = n ? n->Parent : entt::null;
        while (parent != entt::null) {
            if (!ancestor_of_selected.insert(parent).second) break; // already inserted — parents already covered
            const auto *pn = R.try_get<SceneNode>(parent);
            parent = pn ? pn->Parent : entt::null;
        }
    }

    const auto render_entity = [&](const auto &self, entt::entity e) -> void {
        const auto *node = R.try_get<SceneNode>(e);
        const bool has_children = node && node->FirstChild != entt::null;

        const bool is_selected = R.all_of<Selected>(e);
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
        const auto label = std::format("{} [{}]", GetName(R, e), ObjectTypeName(GetObjectType(e)));
        const bool open = TreeNodeEx(reinterpret_cast<void *>(uintptr_t(uint32_t(e))), flags, "%s", label.c_str());
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

    ApplySelectionRequests(begin_requests, begin_nav_item);
    auto *ms_end = EndMultiSelect();
    ApplySelectionRequests({ms_end->Requests.Data, size_t(ms_end->Requests.Size)}, ms_end->NavIdItem);

    PopStyleVar();
}
