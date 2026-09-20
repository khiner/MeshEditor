#include "action/View.h"
#include "TransformMath.h"
#include "Variant.h"
#include "action/Dispatch.h"
#include "action/ScopeResolve.h"
#include "armature/ArmatureComponents.h"
#include "gltf/GltfScene.h"
#include "gltf/SourceAssets.h"
#include "numeric/VectorMath.h"
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionComponents.h"
#include "state/Scene.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"

using std::ranges::find;

namespace {
// The drag's pivot, recorded from the selection on its first update.
const StartPivot &StartPivotOf(state::Scene &r, state::Entity viewport) {
    if (const auto *pivot = r.try_get<const StartPivot>(viewport)) return *pivot;
    return r.emplace<StartPivot>(viewport, TransformPivot(r, viewport));
}
} // namespace

namespace action::view {
void Apply(state::Scene &r, state::Entity viewport, const Action &action) {
    auto patch_camera_stopped = [&](auto &&fn) {
        r.patch<ViewCamera>(viewport, [&](auto &c) { fn(c); c.StopMoving(); });
    };
    auto poke_active_lighting = [&] {
        const auto mode = r.get<const ViewportDisplay>(viewport).ViewportShading;
        if (mode == ViewportShadingMode::MaterialPreview) r.patch<MaterialPreviewLighting>(viewport, [](auto &) {});
        else if (mode == ViewportShadingMode::Rendered) r.patch<RenderedLighting>(viewport, [](auto &) {});
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
                if (e == state::Null || !HasLens(r, e)) return;
                SetLookThrough(r, viewport, e);
                const auto &wt = r.get<WorldTransform>(e);
                r.patch<ViewCamera>(viewport, [&](auto &vc) { vc.AnimateToLookThrough(wt.P, wt.R, 1.f); });
            },
            [&](ExitLookThroughCamera) { ClearLookThrough(r, viewport); },
            [&](const SetLookThroughCamera &a) {
                if (!HasLens(r, a.Entity) || !r.all_of<WorldTransform>(a.Entity)) return;
                SetLookThrough(r, viewport, a.Entity);
                const auto &wt = r.get<WorldTransform>(a.Entity);
                r.replace<ViewCamera>(viewport, ViewCamera{wt.P, wt.R, *LensOf(r, a.Entity)});
            },
            [&](const OrbitViewCamera &a) { r.patch<ViewCamera>(viewport, [&](auto &camera) { camera.RotateBy(a.DeltaRad); }); },
            [&](const ZoomViewCamera &a) { r.patch<ViewCamera>(viewport, [&](auto &camera) { camera.ZoomBy(a.Factor); }); },
            [&](const SetStudioEnvironment &a) { r.emplace_or_replace<StudioEnvironment>(viewport, a.Name); poke_active_lighting(); },
            [&](const SetActiveScene &a) { gltf::SwitchActiveScene(r, a.Scene); },
            [&](ResetViewCamera) { patch_camera_stopped([](auto &c) { c = Defaults::ViewCamera; }); },
            [&](ResetViewportTheme) { r.emplace_or_replace<ViewportTheme>(viewport, Defaults::ViewportTheme); },
            [&](const ResetPbrLighting &a) {
                static constexpr PBRViewportLighting Defaults{false, false, 1.f, 0.f, 0.5f, 0.f, true};
                if (a.Rendered) r.replace<RenderedLighting>(viewport, Defaults);
                else r.replace<MaterialPreviewLighting>(viewport, Defaults);
            },
            [&](const SetWorkspaceLights &a) { r.replace<WorkspaceLights>(viewport, *a.Value); },
            [&](const SetViewCameraTarget &a) { patch_camera_stopped([&](auto &c) { c.Target = a.Target; }); },
            [&](const SetViewCameraLens &a) { patch_camera_stopped([&](auto &c) { c.Data = a.Data; }); },
            [&](const SetViewCameraTargetDirection &a) { r.patch<ViewCamera>(viewport, [&](auto &c) { c.SetTargetDirection(a.Direction); }); },
            [&](const TransformSelection &a) {
                const bool bone_edit_mode = IsBoneEditMode(r, viewport);
                const auto root_selected = RootSelectedForTransform(r, viewport);

                const auto &pivot = StartPivotOf(r, viewport);
                const auto &td = a.Delta;
                const PendingTransform pending{pivot.P, pivot.R, td};

                std::vector<std::pair<state::Entity, Transform>> locals;
                std::vector<std::pair<state::Entity, float>> bone_scales;
                const auto make_local = [&](state::Entity e, const Transform &world, const Transform &pd) {
                    Transform local{.P = Conjugate(pd.R) * ((world.P - pd.P) / pd.S), .R = Conjugate(pd.R) * world.R, .S = r.all_of<ScaleLocked>(e) ? EditedLocal(r, e)->S : world.S / pd.S};
                    locals.emplace_back(e, local);
                };
                // On the first drag frame StartTransform isn't snapshotted yet, so current WorldTransform is the start.
                const auto get_start = [&](state::Entity e) -> std::pair<Transform, Transform> {
                    if (const auto *st = r.try_get<const StartTransform>(e)) return {st->T, st->ParentDelta};
                    return {r.get<const WorldTransform>(e), ToTransform(GetParentDelta(r, e))};
                };
                const auto get_start_bone_length = [&](state::Entity e) -> std::optional<float> {
                    if (const auto *sbl = r.try_get<const StartBoneLength>(e)) return sbl->Value;
                    if (const auto *ds = r.try_get<const BoneDisplayScale>(e)) return ds->Value;
                    return std::nullopt;
                };

                const auto rot = pivot.R, rT = Conjugate(rot);
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
                                const auto transform_point = [&](vec3 p) { return td.P + pivot.P + Rotate(td.R, rot * (rT * (p - pivot.P) * td.S)); };

                                const float bone_length = *sbl;
                                const auto start_head = ts_e.P;
                                const auto start_tail = start_head + Rotate(ts_e.R, vec3{0, bone_length, 0});
                                const auto new_head = tip_only ? start_head : transform_point(start_head);
                                const auto new_tail = root_only ? start_tail : transform_point(start_tail);
                                const auto dir = new_tail - new_head;
                                const auto new_length = Length(dir);
                                constexpr float eps = 1e-6f;
                                const auto new_world_rot = new_length > eps ? Rotation(Normalize(Rotate(ts_e.R, vec3{0, 1, 0})), dir / new_length) * ts_e.R : ts_e.R;
                                bone_scales.emplace_back(e, std::max(new_length, eps));
                                make_local(e, {new_head, new_world_rot, ts_e.S}, pd);
                                continue;
                            }
                        }

                        // Full bone transform in bone edit mode.
                        const auto offset = ts_e.P - pivot.P;
                        make_local(e, {td.P + pivot.P + Rotate(td.R, rot * (rT * offset * td.S)), Normalize(td.R * ts_e.R), ts_e.S}, pd);
                        continue;
                    }

                    // Object mode / non-bone transform.
                    const bool frozen = r.all_of<ScaleLocked>(e);
                    make_local(e, pending.ApplyTo(ts_e, frozen), start_pd);
                }

                // Snapshot starts before patching so later patches don't perturb the snapshot, then apply.
                for (const auto &[e, _] : locals)
                    if (!r.all_of<StartTransform>(e)) r.emplace<StartTransform>(e, r.get<WorldTransform>(e), ToTransform(GetParentDelta(r, e)));
                for (const auto &[e, _] : bone_scales)
                    if (!r.all_of<StartBoneLength>(e))
                        if (const auto *ds = r.try_get<BoneDisplayScale>(e)) r.emplace<StartBoneLength>(e, ds->Value);
                for (const auto &[e, local] : locals) PatchEditedLocal(r, e, [&](auto &t) { t = local; });
                for (const auto &[e, length] : bone_scales) r.emplace_or_replace<BoneDisplayScale>(e, length);
            },
            [&](const TransformElements &a) {
                for (const auto &[_, instance_entity] : ::selection::ComputePrimaryEditInstances(r, false)) {
                    if (!r.all_of<StartTransform>(instance_entity)) {
                        r.emplace<StartTransform>(instance_entity, r.get<WorldTransform>(instance_entity), ToTransform(GetParentDelta(r, instance_entity)));
                    }
                }
                const auto &pivot = StartPivotOf(r, viewport);
                r.emplace_or_replace<PendingTransform>(viewport, pivot.P, pivot.R, a.Delta);
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
            [&](const LatchTransform &a) {
                // Mid-drag switch is a cancel-restart: revert any in-progress drag to its start state.
                // StartTransform / StartBoneLength components stay so the next drag (under the new latched type) reuses them.
                r.remove<PendingTransform>(viewport);
                for (const auto [e, st] : r.view<const StartTransform>().each()) {
                    const auto &pd = st.ParentDelta;
                    PatchEditedLocal(r, e, [&](auto &t) {
                        t.P = Conjugate(pd.R) * ((st.T.P - pd.P) / pd.S);
                        t.R = Conjugate(pd.R) * st.T.R;
                        if (!r.all_of<ScaleLocked>(e)) t.S = st.T.S / pd.S;
                    });
                }
                for (const auto [e, sbl] : r.view<const StartBoneLength>().each()) {
                    r.emplace_or_replace<BoneDisplayScale>(e, sbl.Value);
                }
                r.emplace_or_replace<StartScreenTransform>(viewport, a.Value);
            },
            [&](const SetViewportShading &a) {
                r.patch<ViewportDisplay>(viewport, [&](auto &s) {
                    s.ViewportShading = a.Mode;
                    if (a.Mode != ViewportShadingMode::Wireframe) s.FillMode = a.Mode;
                });
            },
            [&](ToggleXRay) {
                r.patch<ViewportDisplay>(viewport, [](auto &s) {
                    if (s.ViewportShading == ViewportShadingMode::Wireframe) s.XRayWireframe = !s.XRayWireframe;
                    else s.XRaySolid = !s.XRaySolid;
                });
            },
        },
        action
    );
}
} // namespace action::view
