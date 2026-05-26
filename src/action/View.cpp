#include "action/View.h"
#include "TransformMath.h"
#include "action/Dispatch.h"
#include "armature/ArmatureComponents.h"
#include "gltf/SourceAssets.h"
#include "scene/Defaults.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionComponents.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"

using std::ranges::find;

namespace {
void ExitLookThrough(entt::registry &r, entt::entity viewport) {
    const auto camera = LookThroughCameraEntity(r);
    if (camera == entt::null) return;
    r.replace<ViewCamera>(viewport, r.get<LookingThrough>(camera).SavedViewCamera);
    r.remove<LookingThrough>(camera);
}

// Components targeted by the view enum/color Updates (gizmo type/mode, debug channel, clear color).
using UpdateComponents = action::TypeList<TransformGizmoState, ViewportDisplay, ViewportTheme>;
} // namespace

namespace action::view {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    auto patch_camera_stopped = [&](auto &&fn) {
        r.patch<ViewCamera>(viewport, [&](auto &c) { fn(c); c.StopMoving(); });
    };
    auto poke_active_lighting = [&] {
        const auto mode = r.get<const ViewportDisplay>(viewport).ViewportShading;
        if (mode == ViewportShadingMode::MaterialPreview) r.patch<MaterialPreviewLighting>(viewport, [](auto &) {});
        else if (mode == ViewportShadingMode::Rendered) r.patch<RenderedLighting>(viewport, [](auto &) {});
    };
    // Pose mode targets the active bone if there is one; otherwise the active object.
    auto active_rotation_target = [&] {
        const auto bone = FindActiveBone(r);
        return r.get<const Interaction>(viewport).Mode == InteractionMode::Pose && bone != entt::null ? bone : FindActiveEntity(r);
    };
    std::visit(
        overloaded{
            [&](const SetInteractionMode &a) { ::SetInteractionMode(r, viewport, a.Mode); },
            [&](CycleInteractionMode) {
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                const auto &enabled_modes = r.get<const EnabledInteractionModes>(viewport).Value;
                auto it = find(enabled_modes, interaction_mode);
                for (size_t i = 0; i < enabled_modes.size(); ++i) {
                    if (++it == enabled_modes.end()) it = enabled_modes.begin();
                    if (::SetInteractionMode(r, viewport, *it)) break;
                }
            },
            [&](const SetEditMode &a) { r.emplace_or_replace<PendingSetEditMode>(viewport, a.Mode); },
            [&](EnterLookThroughCamera) {
                const auto e = FindActiveEntity(r);
                if (e == entt::null) return;
                SetLookThrough(r, viewport, e);
                const auto &wt = r.get<WorldTransform>(e);
                const vec3 fwd = CameraForward(wt), away = -fwd;
                r.patch<ViewCamera>(viewport, [&](auto &vc) { vc.AnimateTo(wt.P + fwd, {std::atan2(away.z, away.x), std::asin(away.y)}, 1.f); });
            },
            [&](ExitLookThroughCamera) { ExitLookThrough(r, viewport); },
            [&](const OrbitViewCamera &a) {
                ExitLookThrough(r, viewport);
                r.patch<ViewCamera>(viewport, [&](auto &camera) { camera.RotateBy(a.DeltaRad); });
            },
            [&](const ZoomViewCamera &a) {
                ExitLookThrough(r, viewport);
                r.patch<ViewCamera>(viewport, [&](auto &camera) { camera.ZoomBy(a.Factor); });
            },
            [&](const SetStudioEnvironment &a) { r.emplace_or_replace<PendingSetStudioEnvironment>(viewport, a.Index); poke_active_lighting(); },
            [&](const SetSourceIblIntensity &a) {
                r.patch<gltf::SourceAssets>(viewport, [&](auto &sa) { if (sa.ImageBasedLight) sa.ImageBasedLight->Intensity = a.Intensity; });
                poke_active_lighting();
            },
            [&](ResetViewCamera) { patch_camera_stopped([](auto &c) { c = Defaults::ViewCamera; }); },
            [&](ResetViewportTheme) { r.emplace_or_replace<ViewportTheme>(viewport, Defaults::ViewportTheme); },
            [&](const ResetPbrLighting &a) {
                static constexpr PBRViewportLighting Defaults{false, false, 1.f, 0.f, 0.5f, 0.f, true};
                if (a.Rendered) r.emplace_or_replace<RenderedLighting>(viewport, RenderedLighting{Defaults});
                else r.emplace_or_replace<MaterialPreviewLighting>(viewport, MaterialPreviewLighting{Defaults});
            },
            [&](const SetViewCameraTarget &a) { patch_camera_stopped([&](auto &c) { c.Target = a.Target; }); },
            [&](const SetViewCameraLens &a) { patch_camera_stopped([&](auto &c) { c.Data = a.Data; }); },
            [&](const SetViewCameraTargetDirection &a) {
                ExitLookThrough(r, viewport);
                r.patch<ViewCamera>(viewport, [&](auto &c) { c.SetTargetDirection(a.Direction); });
            },
            [&](const SetRotationUiMode &a) {
                const auto e = active_rotation_target();
                r.replace<RotationUiVariant>(e, CreateVariantByIndex<RotationUiVariant>(a.Index));
                r.patch<Transform>(e, [](auto &) {});
            },
            [&](const SetTransformRotationFromUi &a) {
                const auto e = active_rotation_target();
                r.replace<RotationUiVariant>(e, a.UiVariant);
                r.emplace_or_replace<RotationUiDriving>(e);
                r.patch<Transform>(e, [&](auto &t) { t.R = a.R; });
            },
            [&](const DragGizmo &a) {
                for (const auto &[e, _] : a.Locals) {
                    if (!r.all_of<StartTransform>(e)) r.emplace<StartTransform>(e, r.get<WorldTransform>(e), ToTransform(GetParentDelta(r, e)));
                }
                for (const auto &[e, _] : a.BoneDisplayScales) {
                    if (!r.all_of<StartBoneLength>(e)) {
                        if (const auto *ds = r.try_get<BoneDisplayScale>(e)) r.emplace<StartBoneLength>(e, ds->Value);
                    }
                }
                for (const auto &[e, local] : a.Locals) r.patch<Transform>(e, [&](auto &t) { t = local; });
                for (const auto &[e, length] : a.BoneDisplayScales) r.get_or_emplace<BoneDisplayScale>(e).Value = length;
            },
            [&](const DragGizmoMeshEdit &a) {
                for (const auto &[_, instance_entity] : ::selection::ComputePrimaryEditInstances(r, false)) {
                    if (!r.all_of<StartTransform>(instance_entity)) {
                        r.emplace<StartTransform>(instance_entity, r.get<WorldTransform>(instance_entity), ToTransform(GetParentDelta(r, instance_entity)));
                    }
                }
                r.emplace_or_replace<PendingTransform>(viewport, *a.Value);
            },
            [&](EndGizmoDrag) {
                r.clear<StartTransform, StartBoneLength>();
                r.remove<StartScreenTransform>(viewport);
            },
            [&](const SetActiveTool &a) {
                using Tool = SetActiveTool::Tool;
                using TT = TransformGizmo::Type;
                const auto type = a.Value == Tool::SelectBox || a.Value == Tool::SelectClick ? TT::None :
                    a.Value == Tool::Translate                                               ? TT::Translate :
                    a.Value == Tool::Rotate                                                  ? TT::Rotate :
                    a.Value == Tool::Scale                                                   ? TT::Scale :
                                                                                               TT::Universal;
                r.patch<TransformGizmoState>(viewport, [&](auto &s) { s.Config.Type = type; });
                if (a.Value == Tool::SelectBox || a.Value == Tool::SelectClick) {
                    const auto g = a.Value == Tool::SelectBox ? SelectionGesture::Box : SelectionGesture::Click;
                    r.patch<BoxSelectState>(viewport, [&](auto &b) { b.Gesture = g; });
                }
            },
            [&](const SetStartScreenTransform &a) {
                if (!a.Value) {
                    r.remove<StartScreenTransform>(viewport);
                    return;
                }
                // Mid-drag switch is a cancel-restart: revert any in-progress drag to its start state.
                // StartTransform / StartBoneLength components stay so the next drag (under the new latched type) reuses them.
                r.remove<PendingTransform>(viewport);
                for (const auto [e, st] : r.view<const StartTransform>().each()) {
                    const auto &pd = st.ParentDelta;
                    r.patch<Transform>(e, [&](auto &t) {
                        t.P = glm::conjugate(pd.R) * ((st.T.P - pd.P) / pd.S);
                        t.R = glm::conjugate(pd.R) * st.T.R;
                        if (!r.all_of<ScaleLocked>(e)) t.S = st.T.S / pd.S;
                    });
                }
                for (const auto [e, sbl] : r.view<const StartBoneLength>().each()) {
                    r.get_or_emplace<BoneDisplayScale>(e).Value = sbl.Value;
                }
                r.emplace_or_replace<StartScreenTransform>(viewport, *a.Value);
            },
            [&](const SetViewportShading &a) {
                r.patch<ViewportDisplay>(viewport, [&](auto &s) {
                    s.ViewportShading = a.Mode;
                    if (a.Mode != ViewportShadingMode::Wireframe) s.FillMode = a.Mode;
                });
            },
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate<UpdateComponents>(r, viewport, a); },
            [&](const Replace<::Camera> &a) { r.emplace_or_replace<::Camera>(a.Entity, a.Value); },
            [&](const ReplaceActive<::Camera> &a) { r.emplace_or_replace<::Camera>(FindActiveEntity(r), a.Value); },
        },
        action
    );
}
} // namespace action::view
