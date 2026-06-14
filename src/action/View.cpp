#include "action/View.h"
#include "TransformMath.h"
#include "action/Dispatch.h"
#include "action/ScopeResolve.h"
#include "armature/ArmatureComponents.h"
#include "gltf/GltfScene.h"
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

#include <cstddef>
#include <cstring>

using std::ranges::find;

namespace {
void ExitLookThrough(entt::registry &r, entt::entity viewport) {
    const auto camera = LookThroughCameraEntity(r);
    if (camera == entt::null) return;
    r.replace<ViewCamera>(viewport, r.get<LookingThrough>(camera).SavedViewCamera);
    r.remove<LookingThrough>(camera);
}

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
    // Selected/SelectedDelta fan out to the selected bones in Pose mode, else the selected objects.
    // Any other scope resolves to the single active rotation target.
    auto rotation_targets = [&](Scope scope) {
        std::vector<entt::entity> targets;
        if (scope != Scope::Selected && scope != Scope::SelectedDelta) {
            if (const auto e = active_rotation_target(); e != entt::null) targets.push_back(e);
        } else if (r.get<const Interaction>(viewport).Mode == InteractionMode::Pose) {
            for (const auto e : r.view<BoneSelection>()) targets.push_back(e);
        } else {
            for (const auto e : r.view<Selected, RotationUiVariant>()) targets.push_back(e);
        }
        return targets;
    };
    // Gesture-start Transform.R, snapshotted into the shared DragFieldStart baseline on first apply.
    auto rotation_start = [&](entt::entity e) -> quat {
        static constexpr uint16_t r_off = offsetof(Transform, R);
        const auto comp = entt::type_hash<Transform>::value();
        if (const auto *s = r.try_get<DragFieldStart>(e); s && s->Comp == comp && s->Offset == r_off) {
            quat q;
            std::memcpy(&q, s->Bytes.data(), sizeof(quat));
            return q;
        }
        const quat cur = r.get<const Transform>(e).R;
        DragFieldStart s{comp, r_off, sizeof(quat), {}};
        std::memcpy(s.Bytes.data(), &cur, sizeof(quat));
        r.emplace_or_replace<DragFieldStart>(e, s);
        return cur;
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
            [&](const SetExtent &a) { r.ctx().get<ViewportExtent>().Value = a.Extent; },
            [&](const SetStudioEnvironment &a) { r.emplace_or_replace<PendingSetStudioEnvironment>(viewport, a.Index); poke_active_lighting(); },
            [&](const SetSourceIblIntensity &a) {
                r.patch<gltf::SourceAssets>(viewport, [&](auto &sa) { if (sa.ImageBasedLight) sa.ImageBasedLight->Intensity = a.Intensity; });
                poke_active_lighting();
            },
            [&](const SetActiveScene &a) { gltf::SwitchActiveScene(r, a.Scene); },
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
                for (const auto e : rotation_targets(a.Scope)) {
                    r.replace<RotationUiVariant>(e, CreateVariantByIndex<RotationUiVariant>(a.Index));
                    r.patch<Transform>(e, [](auto &) {});
                }
            },
            [&](const SetTransformRotationFromUi &a) {
                if (a.Scope == Scope::SelectedDelta) {
                    // Rotate each selected entity by the same relative rotation the active turned through.
                    const auto active = active_rotation_target();
                    if (active == entt::null) return;
                    const quat delta = a.R * glm::conjugate(rotation_start(active));
                    for (const auto e : rotation_targets(Scope::SelectedDelta)) {
                        const quat rotation = glm::normalize(delta * rotation_start(e));
                        r.patch<Transform>(e, [&](auto &t) { t.R = rotation; });
                        if (e == active) { // keep the editor's representation stable; others re-sync from R
                            r.replace<RotationUiVariant>(e, a.UiVariant);
                            r.emplace_or_replace<RotationUiDriving>(e);
                        }
                    }
                } else {
                    for (const auto e : rotation_targets(a.Scope)) {
                        r.replace<RotationUiVariant>(e, a.UiVariant);
                        r.emplace_or_replace<RotationUiDriving>(e);
                        r.patch<Transform>(e, [&](auto &t) { t.R = a.R; });
                    }
                }
            },
            [&](const DragGizmo &a) {
                const bool bone_edit_mode = IsBoneEditMode(r, viewport);
                const auto root_selected = RootSelectedForTransform(r, viewport);

                const Transform ts{a.Value->Pivot, a.Value->PivotR, vec3{1}}; // only P/R are used
                const auto &td = a.Value->Delta;

                std::vector<std::pair<entt::entity, Transform>> locals;
                std::vector<std::pair<entt::entity, float>> bone_scales;
                const auto make_local = [&](entt::entity e, const Transform &world, const Transform &pd) {
                    Transform local;
                    local.P = glm::conjugate(pd.R) * ((world.P - pd.P) / pd.S);
                    local.R = glm::conjugate(pd.R) * world.R;
                    local.S = r.all_of<ScaleLocked>(e) ? r.get<const Transform>(e).S : world.S / pd.S;
                    locals.emplace_back(e, local);
                };
                // On the first drag frame StartTransform isn't snapshotted yet, so current WorldTransform is the start.
                const auto get_start = [&](entt::entity e) -> std::pair<Transform, Transform> {
                    if (const auto *st = r.try_get<const StartTransform>(e)) return {st->T, st->ParentDelta};
                    return {r.get<const WorldTransform>(e), ToTransform(GetParentDelta(r, e))};
                };
                const auto get_start_bone_length = [&](entt::entity e) -> std::optional<float> {
                    if (const auto *sbl = r.try_get<const StartBoneLength>(e)) return sbl->Value;
                    if (const auto *ds = r.try_get<const BoneDisplayScale>(e)) return ds->Value;
                    return std::nullopt;
                };

                const auto rot = ts.R, rT = glm::conjugate(rot);
                for (const auto e : root_selected) {
                    const auto [ts_e, start_pd] = get_start(e);

                    // Head/tail-only bone transform: stretch/rotate bone instead of moving it rigidly.
                    if (bone_edit_mode) {
                        // Use current parent WT for world→local (parent may have been moved earlier in this loop).
                        const auto pd = ToTransform(GetParentDelta(r, e));
                        const auto sbl = get_start_bone_length(e);
                        const auto *parts = r.try_get<BoneSelection>(e);
                        if (sbl && parts) {
                            const bool tip_only = parts->Tip && !parts->Root && !parts->Body;
                            const bool root_only = parts->Root && !parts->Tip && !parts->Body;
                            if (tip_only || root_only) {
                                const auto transform_point = [&](vec3 p) { return td.P + ts.P + glm::rotate(td.R, rot * (rT * (p - ts.P) * td.S)); };

                                const float bone_length = *sbl;
                                const auto start_head = ts_e.P;
                                const auto start_tail = start_head + glm::rotate(ts_e.R, vec3{0, bone_length, 0});
                                const auto new_head = tip_only ? start_head : transform_point(start_head);
                                const auto new_tail = root_only ? start_tail : transform_point(start_tail);
                                const auto dir = new_tail - new_head;
                                const auto new_length = glm::length(dir);
                                constexpr float eps = 1e-6f;
                                const auto new_world_rot = new_length > eps ? glm::rotation(glm::normalize(glm::rotate(ts_e.R, vec3{0, 1, 0})), dir / new_length) * ts_e.R : ts_e.R;
                                bone_scales.emplace_back(e, std::max(new_length, eps));
                                make_local(e, {new_head, new_world_rot, ts_e.S}, pd);
                                continue;
                            }
                        }

                        // Full bone transform in bone edit mode.
                        const auto offset = ts_e.P - ts.P;
                        make_local(e, {td.P + ts.P + glm::rotate(td.R, rot * (rT * offset * td.S)), glm::normalize(td.R * ts_e.R), ts_e.S}, pd);
                        continue;
                    }

                    // Object mode / non-bone transform.
                    const bool frozen = r.all_of<ScaleLocked>(e);
                    const auto offset = ts_e.P - ts.P;
                    make_local(e, {td.P + ts.P + glm::rotate(td.R, frozen ? offset : rot * (rT * offset * td.S)), glm::normalize(td.R * ts_e.R), frozen ? ts_e.S : td.S * ts_e.S}, start_pd);
                }

                // Snapshot starts before patching so later patches don't perturb the snapshot, then apply.
                for (const auto &[e, _] : locals)
                    if (!r.all_of<StartTransform>(e)) r.emplace<StartTransform>(e, r.get<WorldTransform>(e), ToTransform(GetParentDelta(r, e)));
                for (const auto &[e, _] : bone_scales)
                    if (!r.all_of<StartBoneLength>(e))
                        if (const auto *ds = r.try_get<BoneDisplayScale>(e)) r.emplace<StartBoneLength>(e, ds->Value);
                for (const auto &[e, local] : locals) r.patch<Transform>(e, [&](auto &t) { t = local; });
                for (const auto &[e, length] : bone_scales) r.emplace_or_replace<BoneDisplayScale>(e, length);
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
            [&](const LatchScreenTransform &a) {
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
                    r.emplace_or_replace<BoneDisplayScale>(e, sbl.Value);
                }
                r.emplace_or_replace<StartScreenTransform>(viewport, a.Value);
            },
            [&](ClearScreenTransformLatch) { r.remove<StartScreenTransform>(viewport); },
            [&](const SetViewportShading &a) {
                r.patch<ViewportDisplay>(viewport, [&](auto &s) {
                    s.ViewportShading = a.Mode;
                    if (a.Mode != ViewportShadingMode::Wireframe) s.FillMode = a.Mode;
                });
            },
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate(r, viewport, a); },
            [&](const Replace<::Camera> &a) { ForEachReplaceTarget<::Camera>(r, a.Scope, a.Entity, [&](entt::entity e) { r.emplace_or_replace<::Camera>(e, a.Value); }); },
            [&](const Replace<WorkspaceLights> &a) { r.replace<WorkspaceLights>(a.Entity, *a.Value); },
        },
        action
    );
}
} // namespace action::view
