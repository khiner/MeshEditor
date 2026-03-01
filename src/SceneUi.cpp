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
#include "numeric/rect.h"

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

std::optional<std::pair<uvec2, uvec2>> ComputeBoxSelectPixels(vec2 start, vec2 end, vec2 window_pos, vk::Extent2D extent) {
    static constexpr float DragThresholdSq{2 * 2};
    if (glm::distance2(start, end) <= DragThresholdSq) return {};

    const vec2 extent_size{float(extent.width), float(extent.height)};
    const auto box_min = glm::min(start, end) - window_pos;
    const auto box_max = glm::max(start, end) - window_pos;
    const auto local_min = glm::clamp(glm::min(box_min, box_max), vec2{0}, extent_size);
    const auto local_max = glm::clamp(glm::max(box_min, box_max), vec2{0}, extent_size);
    const uvec2 box_min_px{uint32_t(glm::floor(local_min.x)), uint32_t(glm::floor(extent_size.y - local_max.y))};
    const uvec2 box_max_px{uint32_t(glm::ceil(local_max.x)), uint32_t(glm::ceil(extent_size.y - local_min.y))};
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
        if (SliderFloat("Field of view (deg)", &fov_deg, 1.f, 179.f)) {
            perspective->FieldOfViewRad = glm::radians(fov_deg);
            lens_changed = true;
        }
        const float near_max = perspective->FarClip ? std::max(*perspective->FarClip - MinNearFarDelta, MinNearFarDelta) : MaxFarClip;
        lens_changed |= SliderFloat("Near clip", &perspective->NearClip, 0.001f, near_max);
        bool infinite_far = !perspective->FarClip.has_value();
        if (Checkbox("Infinite far clip", &infinite_far)) {
            if (infinite_far) perspective->FarClip.reset();
            else perspective->FarClip = std::max(perspective->NearClip + MinNearFarDelta, MaxFarClip);
            lens_changed = true;
        }
        if (perspective->FarClip) {
            lens_changed |= SliderFloat("Far clip", &*perspective->FarClip, perspective->NearClip + MinNearFarDelta, MaxFarClip);
        }
        if (!viewport) {
            float aspect = perspective->AspectRatio.value_or(DefaultAspectRatio);
            if (SliderFloat("Aspect ratio", &aspect, 0.1f, 5.f)) {
                perspective->AspectRatio = aspect;
                lens_changed = true;
            }
        }
    } else if (auto *orthographic = std::get_if<Orthographic>(&camera)) {
        lens_changed |= SliderFloat("X Mag", &orthographic->Mag.x, 0.01f, 100.f);
        lens_changed |= SliderFloat("Y Mag", &orthographic->Mag.y, 0.01f, 100.f);
        lens_changed |= SliderFloat("Near clip", &orthographic->NearClip, 0.001f, orthographic->FarClip - MinNearFarDelta);
        lens_changed |= SliderFloat("Far clip", &orthographic->FarClip, orthographic->NearClip + MinNearFarDelta, MaxFarClip);
    }
    return lens_changed;
}

WorkspaceLights &GetWorkspaceLights(SceneBuffers &buffers) {
    return *reinterpret_cast<WorkspaceLights *>(buffers.WorkspaceLightsUBO.GetMappedData().data());
}

} // namespace

void Scene::SetInteractionMode(InteractionMode mode) {
    if (R.get<const SceneInteraction>(SceneEntity).Mode == mode) return;
    if (mode == InteractionMode::Edit && !AllSelectedAreMeshes(R)) return;

    R.patch<SceneInteraction>(SceneEntity, [mode](auto &s) { s.Mode = mode; });
    R.patch<ViewportTheme>(SceneEntity, [](auto &) {});
}

void Scene::SetEditMode(Element mode) {
    const auto current_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    if (current_mode == mode) return;

    for (const auto &[e, selection, mesh] : R.view<MeshSelection, Mesh>().each()) {
        R.replace<MeshSelection>(e, scene_selection::ConvertSelectionElement(selection, mesh, current_mode, mode));
        R.remove<MeshActiveElement>(e);
    }
    R.patch<SceneEditMode>(SceneEntity, [mode](auto &edit_mode) { edit_mode.Value = mode; });
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
    const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
    if (extent.width == 0 || extent.height == 0) return;

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
                // Cycle to the next interaction mode, wrapping around to the first.
                auto it = find(InteractionModes, interaction_mode);
                SetInteractionMode(++it != InteractionModes.end() ? *it : *InteractionModes.begin());
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
            R.patch<ViewCamera>(SceneEntity, [&](auto &camera) { camera.SetTargetDistance(std::max(camera.Distance * (1 - wheel.y / 16.f), 0.01f)); });
        } else {
            R.patch<ViewCamera>(SceneEntity, [&](auto &camera) { camera.SetTargetYawPitch(camera.YawPitch + wheel * 0.15f); });
        }
    }
    if (OrientationGizmo::IsActive() || TransformModePillsHovered) return;

    const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    if (SelectionMode == SelectionMode::Box && (interaction_mode == InteractionMode::Edit || interaction_mode == InteractionMode::Object)) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            BoxSelectStart = BoxSelectEnd = ToGlm(GetMousePos());
        } else if (IsMouseDown(ImGuiMouseButton_Left) && BoxSelectStart) {
            BoxSelectEnd = ToGlm(GetMousePos());
            if (const auto box_px = ComputeBoxSelectPixels(*BoxSelectStart, *BoxSelectEnd, ToGlm(GetCursorScreenPos()), extent); box_px) {
                const bool is_additive = IsKeyDown(ImGuiMod_Shift);
                if (interaction_mode == InteractionMode::Edit) {
                    Timer timer{"BoxSelectElements (all)"};

                    const auto selected_mesh_entities = scene_selection::GetSelectedMeshEntities(R);

                    std::vector<ElementRange> ranges;
                    uint32_t offset = 0;
                    for (const auto mesh_entity : selected_mesh_entities) {
                        if (!is_additive && !R.get<MeshSelection>(mesh_entity).Handles.empty()) {
                            R.replace<MeshSelection>(mesh_entity, MeshSelection{});
                        }
                        if (const uint32_t count = scene_selection::GetElementCount(R.get<Mesh>(mesh_entity), edit_mode); count > 0) {
                            ranges.emplace_back(mesh_entity, offset, count);
                            offset += count;
                        }
                    }

                    auto results = RunBoxSelectElements(ranges, edit_mode, *box_px);
                    for (size_t i = 0; i < results.size(); ++i) {
                        const auto e = ranges[i].MeshEntity;
                        R.patch<MeshSelection>(e, [&](auto &s) {
                            if (is_additive) s.Handles.insert(results[i].begin(), results[i].end());
                            else s.Handles = {results[i].begin(), results[i].end()};
                        });
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

    const auto mouse_pos_rel = GetMousePos() - GetCursorScreenPos();
    // Flip y-coordinate: ImGui uses top-left origin, but Vulkan gl_FragCoord uses bottom-left origin
    const uvec2 mouse_px{uint32_t(mouse_pos_rel.x), uint32_t(extent.height - mouse_pos_rel.y)};

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
    if (interaction_mode == InteractionMode::Edit && edit_mode == Element::None) return;

    if (interaction_mode == InteractionMode::Edit) {
        const bool toggle = IsKeyDown(ImGuiMod_Shift) || IsKeyDown(ImGuiMod_Ctrl) || IsKeyDown(ImGuiMod_Super);
        std::vector<ElementRange> ranges;
        uint32_t offset = 0;
        std::unordered_set<entt::entity> seen_meshes;
        for (const auto [_, mesh_instance] : R.view<const MeshInstance, const Selected>().each()) {
            if (!seen_meshes.emplace(mesh_instance.MeshEntity).second) continue;
            const uint32_t count = scene_selection::GetElementCount(R.get<Mesh>(mesh_instance.MeshEntity), edit_mode);
            if (count == 0) continue;
            ranges.emplace_back(ElementRange{mesh_instance.MeshEntity, offset, count});
            offset += count;
        }
        if (!toggle) {
            for (const auto mesh_entity : seen_meshes) {
                if (!R.get<MeshSelection>(mesh_entity).Handles.empty()) {
                    R.replace<MeshSelection>(mesh_entity, MeshSelection{});
                }
            }
        }
        if (const auto hit = RunElementPickFromRanges(ranges, edit_mode, mouse_px)) {
            const auto [mesh_entity, element_index] = *hit;
            const auto *current_active = R.try_get<MeshActiveElement>(mesh_entity);
            const bool is_active = current_active && current_active->Handle == element_index;
            R.patch<MeshSelection>(mesh_entity, [&](auto &selection) {
                if (!toggle) selection = {};
                if (toggle && selection.Handles.contains(element_index)) {
                    selection.Handles.erase(element_index);
                } else {
                    selection.Handles.emplace(element_index);
                }
            });
            if (toggle && is_active) {
                R.remove<MeshActiveElement>(mesh_entity);
            } else {
                R.emplace_or_replace<MeshActiveElement>(mesh_entity, element_index);
            }
        }
    } else if (interaction_mode == InteractionMode::Object) {
        const auto hit_entities = RunObjectPick(mouse_px, ObjectSelectRadiusPx);

        entt::entity hit = entt::null;
        if (!hit_entities.empty()) {
            auto it = find(hit_entities, active_entity);
            if (it != hit_entities.end()) ++it;
            if (it == hit_entities.end()) it = hit_entities.begin();
            hit = *it;
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
    }
}

void Scene::RenderOverlay() {
    const rect viewport{ToGlm(GetWindowPos()), ToGlm(GetContentRegionAvail())};
    const bool active_transform = TransformGizmo::IsUsing();
    { // Transform mode pill buttons (top-left overlay)
        struct ButtonInfo {
            const SvgResource &Icon;
            TransformGizmo::Type ButtonType;
            ImDrawFlags Corners;
            bool Enabled;
        };

        using enum TransformGizmo::Type;
        const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
        const bool has_frozen_selected = R.view<Selected, Frozen>().begin() != R.view<Selected, Frozen>().end();
        const bool edit_transform_locked = interaction_mode == InteractionMode::Edit &&
            any_of(scene_selection::GetSelectedMeshEntities(R), [&](entt::entity mesh_entity) { return scene_selection::HasFrozenInstance(R, mesh_entity); });
        const bool transform_enabled = !edit_transform_locked;
        const bool scale_enabled = transform_enabled && !has_frozen_selected;
        const ButtonInfo buttons[]{
            {*Icons.SelectBox, None, ImDrawFlags_RoundCornersTop, true},
            {*Icons.Select, None, ImDrawFlags_RoundCornersBottom, true},
            {*Icons.Move, Translate, ImDrawFlags_RoundCornersTop, transform_enabled},
            {*Icons.Rotate, Rotate, ImDrawFlags_RoundCornersNone, transform_enabled},
            {*Icons.Scale, Scale, ImDrawFlags_RoundCornersNone, scale_enabled},
            {*Icons.Universal, Universal, ImDrawFlags_RoundCornersBottom, transform_enabled},
        };

        auto &transform_type = MGizmo.Config.Type;
        if (!transform_enabled) transform_type = None;
        else if (!scale_enabled && transform_type == Scale) transform_type = Translate;

        const float padding = GetTextLineHeightWithSpacing() / 2.f;
        const auto start_pos = std::bit_cast<ImVec2>(viewport.pos) + GetWindowContentRegionMin() + ImVec2{padding, padding};
        const auto saved_cursor_pos = GetCursorScreenPos();

        auto &dl = *GetWindowDrawList();
        TransformModePillsHovered = false;
        static constexpr ImVec2 button_size{36, 30};
        static constexpr float gap{4}; // Gap between select buttons and transform buttons
        for (uint i = 0; i < 6; ++i) {
            const auto &[icon, button_type, corners, enabled] = buttons[i];
            static constexpr ImVec2 padding{0.5f, 0.5f};
            static constexpr float icon_dim{button_size.y * 0.75f};
            static constexpr ImVec2 icon_size{icon_dim, icon_dim};
            const float y_offset = i < 2 ? i * button_size.y : 2 * button_size.y + gap + (i - 2) * button_size.y;
            const ImVec2 button_min{start_pos.x, start_pos.y + y_offset};
            const ImVec2 button_max = button_min + button_size;

            bool clicked = false;
            bool hovered = false;
            if (!active_transform) {
                SetCursorScreenPos(button_min);
                if (!enabled) BeginDisabled();
                PushID(i);
                clicked = InvisibleButton("##icon", button_size);
                PopID();
                if (!enabled) EndDisabled();
                hovered = IsItemHovered();
            }
            if (hovered) TransformModePillsHovered = true;
            if (clicked) {
                if (i == 0) {
                    SelectionMode = SelectionMode::Box;
                    transform_type = None;
                } else if (i == 1) {
                    SelectionMode = SelectionMode::Click;
                    transform_type = None;
                } else { // Transform buttons
                    transform_type = button_type;
                }
            }

            const bool is_active = i < 2 ?
                (transform_type == None && ((i == 0 && SelectionMode == SelectionMode::Box) || (i == 1 && SelectionMode == SelectionMode::Click))) :
                transform_type == button_type;
            const auto bg_color = GetColorU32(
                !enabled      ? ImGuiCol_FrameBg :
                    is_active ? ImGuiCol_ButtonActive :
                    hovered   ? ImGuiCol_ButtonHovered :
                                ImGuiCol_Button
            );
            dl.AddRectFilled(button_min + padding, button_max - padding, bg_color, 8.f, corners);
            SetCursorScreenPos(button_min + (button_size - icon_size) * 0.5f);
            icon.DrawIcon(std::bit_cast<vec2>(icon_size));
        }
        SetCursorScreenPos(saved_cursor_pos);
    }

    // Exit "look through" camera view if the user interacts with the orientation gizmo.
    if (!active_transform && OrientationGizmo::IsActive()) ExitLookThroughCamera();
    auto &camera = R.get<ViewCamera>(SceneEntity);
    { // Orientation gizmo (drawn before tick so camera animations it initiates begin this frame)
        static constexpr float OGizmoSize{90};
        const float padding = GetTextLineHeightWithSpacing();
        const auto pos = viewport.pos + vec2{GetWindowContentRegionMax().x, GetWindowContentRegionMin().y} - vec2{OGizmoSize, 0} + vec2{-padding, padding};
        OrientationGizmo::Draw(pos, OGizmoSize, camera, !active_transform);
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
        for (const auto [e, mi] : R.view<const MeshInstance, const Selected>(entt::exclude<Frozen>).each()) {
            if (!R.get<const MeshSelection>(mi.MeshEntity).Handles.empty()) return true;
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
        const auto active_transform = active_entity != entt::null ? GetTransform(R, active_entity) : Transform{};
        const auto edit_transform_instances = interaction_mode == InteractionMode::Edit ?
            scene_selection::ComputePrimaryEditInstances(R, false) :
            std::unordered_map<entt::entity, entt::entity>{};

        vec3 pivot{};
        if (interaction_mode == InteractionMode::Edit) {
            // Compute world-space centroid of selected vertices once per selected mesh
            // (using a representative selected instance for world transform).
            uint32_t vertex_count = 0;
            const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
            for (const auto &[mesh_entity, instance_entity] : edit_transform_instances) {
                const auto &mesh = R.get<const Mesh>(mesh_entity);
                const auto vertices = mesh.GetVerticesSpan();
                const auto &selection = R.get<const MeshSelection>(mesh_entity);
                if (selection.Handles.empty()) continue;
                const auto &wt = R.get<const WorldTransform>(instance_entity);
                const auto vertex_handles = scene_selection::ConvertSelectionElement(selection, mesh, edit_mode, Element::Vertex);
                for (const auto vi : vertex_handles) {
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
            pivot = fold_left(root_selected | transform([&](auto e) { return R.get<Position>(e).Value; }), vec3{}, std::plus{}) / float(root_count);
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
                    for (const auto e : root_selected) R.emplace<StartTransform>(e, GetTransform(R, e));
                }
            }
            if (interaction_mode == InteractionMode::Edit) {
                // Edit mode: store pending transform for shader-based preview.
                // Actual vertex positions are only modified on commit.
                R.emplace_or_replace<PendingTransform>(SceneEntity, ts.P, ts.R, td.P, td.R, td.S);
            } else {
                // Object mode: apply transform to entity components immediately during drag.
                const auto r = ts.R, rT = glm::conjugate(r);
                for (const auto &[e, ts_e_comp] : start_transform_view.each()) {
                    const auto &ts_e = ts_e_comp.T;
                    const bool frozen = R.all_of<Frozen>(e);
                    const auto offset = ts_e.P - ts.P;
                    SetTransform(
                        R, e,
                        {
                            .P = td.P + ts.P + glm::rotate(td.R, frozen ? offset : r * (rT * offset * td.S)),
                            .R = glm::normalize(td.R * ts_e.R),
                            .S = frozen ? ts_e.S : td.S * ts_e.S,
                        }
                    );
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
            const auto material_name = [&](uint32_t index) -> std::string {
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
                const auto max_primitive = uint32_t(primitive_materials.size() - 1);
                if (slot_selection.PrimitiveIndex > max_primitive) {
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

                uint32_t assigned_material = std::min(primitive_materials[slot_selection.PrimitiveIndex], material_count - 1);
                if (const auto *pending = R.try_get<const MeshMaterialAssignment>(active_mesh_entity);
                    pending && pending->PrimitiveIndex == slot_selection.PrimitiveIndex) {
                    assigned_material = std::min(pending->MaterialIndex, material_count - 1);
                }
                int assigned_material_i = int(assigned_material);
                const auto assigned_material_name = material_name(assigned_material);
                if (BeginCombo("Assigned material", assigned_material_name.c_str())) {
                    for (int i = 0; i < int(material_count); ++i) {
                        const auto option_name = material_name(uint32_t(i));
                        if (Selectable(option_name.c_str(), assigned_material_i == i)) {
                            assigned_material_i = i;
                            assigned_material = uint32_t(i);
                            R.emplace_or_replace<MeshMaterialAssignment>(active_mesh_entity, slot_selection.PrimitiveIndex, uint32_t(i));
                        }
                    }
                    EndCombo();
                }

                auto material = GetMaterial(*Buffers, assigned_material);
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

                // Transmission
                material_changed |= SliderFloat("Transmission", &material.Transmission.Factor, 0.f, 1.f);
                if (material.Transmission.Factor > 0.f) {
                    material_changed |= edit_texture_info("Transmission", material.Transmission.Texture);

                    // Volume (only meaningful with transmission)
                    material_changed |= SliderFloat("Thickness", &material.Volume.ThicknessFactor, 0.f, 10.f);
                    material_changed |= edit_texture_info("Thickness", material.Volume.ThicknessTexture);
                    material_changed |= ColorEdit3("Attenuation color", &material.Volume.AttenuationColor.x);
                    // 0 = infinite/disabled; display "Infinite" when zero.
                    material_changed |= DragFloat("Attenuation distance", &material.Volume.AttenuationDistance, 0.01f, 0.f, 0.f, material.Volume.AttenuationDistance <= 0.f ? "Infinite" : "%.3f m");
                }

                // Clearcoat
                material_changed |= SliderFloat("Clearcoat", &material.Clearcoat.Factor, 0.f, 1.f);
                if (material.Clearcoat.Factor > 0.f) {
                    material_changed |= edit_texture_info("Clearcoat", material.Clearcoat.Texture);
                    material_changed |= SliderFloat("Clearcoat roughness", &material.Clearcoat.RoughnessFactor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Clearcoat roughness", material.Clearcoat.RoughnessTexture);
                    material_changed |= edit_texture_info("Clearcoat normal", material.Clearcoat.NormalTexture);
                    material_changed |= SliderFloat("Clearcoat normal scale", &material.Clearcoat.NormalScale, -2.f, 2.f);
                }

                // Anisotropy
                material_changed |= SliderFloat("Anisotropy", &material.Anisotropy.Strength, 0.f, 1.f);
                if (material.Anisotropy.Strength > 0.f) {
                    material_changed |= SliderFloat("Anisotropy rotation", &material.Anisotropy.Rotation, 0.f, 6.2832f, "%.3f rad");
                    material_changed |= edit_texture_info("Anisotropy", material.Anisotropy.Texture);
                }

                // Iridescence
                material_changed |= SliderFloat("Iridescence", &material.Iridescence.Factor, 0.f, 1.f);
                if (material.Iridescence.Factor > 0.f) {
                    material_changed |= edit_texture_info("Iridescence", material.Iridescence.Texture);
                    material_changed |= SliderFloat("Iridescence IOR", &material.Iridescence.Ior, 1.0f, 5.0f);
                    material_changed |= SliderFloat("Thickness min", &material.Iridescence.ThicknessMinimum, 0.f, 1000.f, "%.0f nm");
                    material_changed |= SliderFloat("Thickness max", &material.Iridescence.ThicknessMaximum, 1.f, 1000.f, "%.0f nm");
                    material_changed |= edit_texture_info("Iridescence thickness", material.Iridescence.ThicknessTexture);
                }

                if (material_changed) R.emplace_or_replace<MaterialEdit>(SceneEntity, MaterialEdit{.Index = assigned_material, .Value = material});
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
                const bool edit_allowed = AllSelectedAreMeshes(R);
                for (const auto mode : InteractionModes) {
                    if (mode == InteractionMode::Edit && !edit_allowed) continue;
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
                            const auto &selection = R.get<MeshSelection>(mesh_instance->MeshEntity);
                            Text("Editing %s: %zu selected", label(edit_mode).data(), selection.Handles.size());
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
            PushID("ViewportShading");
            auto viewport_shading = int(settings.ViewportShading);
            bool viewport_shading_changed = RadioButton("Wireframe", &viewport_shading, int(ViewportShadingMode::Wireframe));
            SameLine();
            viewport_shading_changed |= RadioButton("Solid", &viewport_shading, int(ViewportShadingMode::Solid));
            SameLine();
            viewport_shading_changed |= RadioButton("Material Preview", &viewport_shading, int(ViewportShadingMode::MaterialPreview));
            SameLine();
            viewport_shading_changed |= RadioButton("Rendered", &viewport_shading, int(ViewportShadingMode::Rendered));
            PopID();

            const auto current_mode = ViewportShadingMode(viewport_shading);

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
                    lighting = {false, false, 1.f, 0.f};
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
            if (viewport_shading_changed || color_mode_changed) {
                settings.ViewportShading = current_mode;
                if (current_mode != ViewportShadingMode::Wireframe) settings.FillMode = current_mode;
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
                changed |= MeshEditor::SliderUInt("Silhouette edge width", &theme.SilhouetteEdgeWidth, 1, 4);
                if (changed) R.patch<ViewportTheme>(SceneEntity, [](auto &) {});
            }
            if (settings_changed) R.patch<SceneSettings>(SceneEntity, [](auto &) {});
            if (mat_preview_changed) {
                mat_preview_lighting.EnvIntensity = std::max(0.f, mat_preview_lighting.EnvIntensity);
                R.patch<MaterialPreviewLighting>(SceneEntity, [](auto &) {});
            }
            if (rendered_changed) {
                rendered_lighting.EnvIntensity = std::max(0.f, rendered_lighting.EnvIntensity);
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

    const auto render_entity = [&](const auto &self, entt::entity e) -> void {
        const auto *node = R.try_get<SceneNode>(e);
        const bool has_children = node && node->FirstChild != entt::null;

        auto flags =
            ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanFullWidth |
            ImGuiTreeNodeFlags_FramePadding |
            ImGuiTreeNodeFlags_NavLeftJumpsToParent;
        if (!has_children) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (R.all_of<Selected>(e)) flags |= ImGuiTreeNodeFlags_Selected;

        SetNextItemSelectionUserData(ToSelectionUserData(e));
        const auto label = std::format("{} [{}]", GetName(R, e), ObjectTypeName(GetObjectType(e)));
        const bool open = TreeNodeEx(reinterpret_cast<void *>(uintptr_t(uint32_t(e))), flags, "%s", label.c_str());
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
