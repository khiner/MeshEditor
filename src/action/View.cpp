#include "action/View.h"
#include "TransformMath.h"
#include "Variant.h"
#include "action/Nodes.h"
#include "armature/ArmatureComponents.h"
#include "gltf/GltfScene.h"
#include "procedure/NodeTheme.h"
#include "procedure/Procedure.h"
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionState.h"
#include "timeline/TimelineTheme.h"
#include "ui/InterfaceTheme.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"

using std::ranges::find;

namespace {
// The drag's pivot, recorded from the selection on its first update.
const StartPivot &StartPivotOf(state::Registry &r) {
    if (const auto *pivot = r.try_get<const StartPivot>(state::Document)) return *pivot;
    return r.emplace<StartPivot>(state::Document, TransformPivot(r));
}
} // namespace

namespace action::view {
void Apply(state::Registry &r, const Action &action) {
    // Runs `fn` on the state of the viewport with `id`, if it exists.
    const auto edit_view = [&](uint32_t id, auto &&fn) {
        if (auto *viewport = r.Context.get<Viewports>().Find(id)) fn(viewport->State);
    };
    const auto edit_camera_stopped = [&](uint32_t id, auto &&fn) {
        edit_view(id, [&](ViewportState &s) { fn(s.Camera.Active); s.Camera.Active.StopMoving(); });
    };
    std::visit(
        overloaded{
            [&](const SetInteractionMode &a) { ::SetInteractionMode(r, a.Mode); },
            [&](CycleInteractionMode) {
                const auto interaction_mode = r.get<const Interaction>(state::Document).Mode;
                const auto &enabled_modes = r.get<const EnabledInteractionModes>(state::Document).Value;
                auto it = find(enabled_modes, interaction_mode);
                for (size_t i = 0; i < enabled_modes.size(); ++i) {
                    if (++it == enabled_modes.end()) it = enabled_modes.begin();
                    if (::SetInteractionMode(r, *it)) break;
                }
            },
            [&](const SetEditMode &a) { r.emplace_or_replace<PendingSetEditMode>(state::Document, a.Mode); },
            [&](const EnterLookThroughCamera &a) {
                const auto e = FindActiveEntity(r);
                if (e == state::Null || !HasLens(r, e)) return;
                const auto &wt = *WorldTransformOf(r, e);
                edit_view(a.ViewportId, [&](ViewportState &s) {
                    SetLookThrough(r, s, e);
                    s.Camera.Active.AnimateToLookThrough(wt.P, wt.R, 1.f);
                });
            },
            [&](const ExitLookThroughCamera &a) { edit_view(a.ViewportId, [](ViewportState &s) { ClearLookThrough(s); }); },
            [&](const SetLookThroughCamera &a) {
                if (!HasLens(r, a.Entity) || !WorldTransformOf(r, a.Entity)) return;
                const auto &wt = *WorldTransformOf(r, a.Entity);
                edit_view(a.ViewportId, [&](ViewportState &s) {
                    SetLookThrough(r, s, a.Entity);
                    s.Camera.Active = ViewCamera{wt.P, wt.R, *LensOf(r, a.Entity)};
                });
            },
            [&](const OrbitViewCamera &a) { edit_view(a.ViewportId, [&](ViewportState &s) { s.Camera.Active.RotateBy(a.DeltaRad); }); },
            [&](const ZoomViewCamera &a) { edit_view(a.ViewportId, [&](ViewportState &s) { s.Camera.Active.ZoomBy(a.Factor); }); },
            [&](const SetStudioEnvironment &a) { edit_view(a.ViewportId, [&](ViewportState &s) { s.Environment.Name = a.Name; }); },
            [&](const SetActiveScene &a) { gltf::SwitchActiveScene(r, a.Scene); },
            [&](AddViewport) { ::AddViewport(r); },
            [&](const RemoveViewport &a) { ::RemoveViewport(r, a.ViewportId); },
            [&](const ResetViewCamera &a) { edit_camera_stopped(a.ViewportId, [](ViewCamera &c) { c = Defaults::ViewCamera; }); },
            [&](ResetViewportTheme) { r.emplace_or_replace<ViewportTheme>(state::Document, Defaults::ViewportTheme); },
            [&](ResetNodeTheme) { r.replace<procedure::NodeTheme>(state::Document, procedure::NodeTheme{}); },
            [&](ResetInterfaceTheme) { r.replace<InterfaceTheme>(state::Document, InterfaceTheme{}); },
            [&](ResetTimelineTheme) { r.replace<TimelineTheme>(state::Document, TimelineTheme{}); },
            [&](const ResetPbrLighting &a) {
                static constexpr PBRViewportLighting Defaults{false, false, 1.f, 0.f, 0.5f, 0.f, true};
                edit_view(a.ViewportId, [&](ViewportState &s) {
                    if (a.Rendered) s.RenderLighting = {Defaults};
                    else s.PreviewLighting = {Defaults};
                });
            },
            [&](const SetWorkspaceLights &a) { edit_view(a.ViewportId, [&](ViewportState &s) { s.Lights = *a.Value; }); },
            [&](const SetViewCameraTarget &a) { edit_camera_stopped(a.ViewportId, [&](ViewCamera &c) { c.Target = a.Target; }); },
            [&](const SetViewCameraLens &a) { edit_camera_stopped(a.ViewportId, [&](ViewCamera &c) { c.Data = a.Data; }); },
            [&](const SetViewCameraTargetDirection &a) { edit_view(a.ViewportId, [&](ViewportState &s) { s.Camera.Active.SetTargetDirection(a.Direction); }); },
            [&](const TransformSelection &a) {
                const bool bone_edit_mode = IsBoneEditMode(r);
                const auto &root_selected = r.get<const TransformRoots>(state::Document).Roots;

                const auto &pivot = StartPivotOf(r);
                const auto &td = a.Delta;
                const PendingTransform pending{pivot.P, pivot.R, td};

                std::vector<std::pair<state::Entity, Transform>> locals;
                std::vector<std::pair<state::Entity, float>> bone_scales;
                const auto make_local = [&](state::Entity e, const Transform &world, const Transform &pd) {
                    Transform local{.P = Conjugate(pd.R) * ((world.P - pd.P) / pd.S), .R = Conjugate(pd.R) * world.R, .S = r.all_of<ScaleLocked>(e) ? ComposedLocal(r, e)->S : world.S / pd.S};
                    locals.emplace_back(e, local);
                };
                // On the first drag frame StartTransform isn't snapshotted yet, so the current world transform is the start.
                const auto get_start = [&](state::Entity e) -> std::pair<Transform, Transform> {
                    if (const auto *st = r.try_get<const StartTransform>(e)) return {st->T, st->ParentDelta};
                    return {*WorldTransformOf(r, e), ToTransform(GetParentDelta(r, e))};
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

                        const auto offset = ts_e.P - pivot.P;
                        make_local(e, {td.P + pivot.P + Rotate(td.R, rot * (rT * offset * td.S)), Normalize(td.R * ts_e.R), ts_e.S}, pd);
                        continue;
                    }

                    const bool frozen = r.all_of<ScaleLocked>(e);
                    make_local(e, pending.ApplyTo(ts_e, frozen), start_pd);
                }

                // Snapshot starts before patching so later patches don't perturb the snapshot, then apply.
                for (const auto &[e, _] : locals) {
                    if (!r.all_of<StartTransform>(e)) r.emplace<StartTransform>(e, *WorldTransformOf(r, e), ToTransform(GetParentDelta(r, e)));
                }
                for (const auto &[e, _] : bone_scales) {
                    if (!r.all_of<StartBoneLength>(e)) {
                        if (const auto *ds = r.try_get<BoneDisplayScale>(e)) r.emplace<StartBoneLength>(e, ds->Value);
                    }
                }
                for (const auto &[e, local] : locals) PatchEditedLocal(r, e, [&](auto &t) { t = local; });
                for (const auto &[e, length] : bone_scales) r.emplace_or_replace<BoneDisplayScale>(e, length);
            },
            [&](const TransformElements &a) {
                for (const auto &[_, instance_entity] : r.get<const EditPrimaries>(state::Document).Transformable) {
                    if (!r.all_of<StartTransform>(instance_entity)) {
                        r.emplace<StartTransform>(instance_entity, *WorldTransformOf(r, instance_entity), ToTransform(GetParentDelta(r, instance_entity)));
                    }
                }
                // A transformable procedure mesh appends a transform node at the drag's first update, and each later update moves its delta.
                const bool started = r.all_of<StartPivot>(state::Document);
                const auto &pivot = StartPivotOf(r);
                const auto &primaries = r.get<const EditPrimaries>(state::Document).Transformable;
                if (started) {
                    nodes::ForEachResultNode(r, [&](state::Entity mesh_entity, state::Entity node) {
                        if (primaries.contains(mesh_entity) && r.all_of<procedure::TransformElements>(node)) r.patch<procedure::TransformElements>(node, [&](auto &p) { p.Delta = a.Delta; });
                    });
                } else {
                    for (const auto mesh_entity : mesh::SelectedEditMeshes(r)) {
                        const auto primary = primaries.find(mesh_entity);
                        if (primary == primaries.end() || !r.all_of<procedure::GeometryProcedure>(mesh_entity)) continue;
                        const auto node = nodes::AppendNode(r, mesh_entity, procedure::KindOf<procedure::TransformElements>);
                        if (node == state::Null) return;
                        r.replace<procedure::TransformElements>(node, a.Delta, pivot.P, *WorldTransformOf(r, primary->second));
                    }
                }
                r.emplace_or_replace<PendingTransform>(state::Document, pivot.P, pivot.R, a.Delta);
            },
            [&](const SetActiveTool &a) {
                using Tool = SetActiveTool::Tool;
                using TT = TransformGizmo::Type;
                const auto type = a.Value == Tool::SelectBox || a.Value == Tool::SelectClick ? TT::None :
                    a.Value == Tool::Translate                                               ? TT::Translate :
                    a.Value == Tool::Rotate                                                  ? TT::Rotate :
                    a.Value == Tool::Scale                                                   ? TT::Scale :
                                                                                               TT::Universal;
                r.patch<TransformGizmoState>(state::Document, [&](auto &s) { s.Config.Type = type; });
                if (a.Value == Tool::SelectBox || a.Value == Tool::SelectClick) {
                    const auto g = a.Value == Tool::SelectBox ? SelectionGesture::Box : SelectionGesture::Click;
                    r.patch<BoxSelectState>(state::Document, [&](auto &b) { b.Gesture = g; });
                }
            },
            [&](const LatchTransform &a) {
                // Mid-drag switch is a cancel-restart: revert any in-progress drag to its start state.
                // StartTransform / StartBoneLength components stay so the next drag (under the new latched type) reuses them.
                r.remove<PendingTransform>(state::Document);
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
                r.emplace_or_replace<StartScreenTransform>(state::Document, a.Value);
            },
            [&](const SetViewportShading &a) {
                edit_view(a.ViewportId, [&](ViewportState &s) {
                    s.Display.ViewportShading = a.Mode;
                    if (a.Mode != ViewportShadingMode::Wireframe) s.Display.FillMode = a.Mode;
                });
            },
            [&](const ToggleXRay &a) {
                edit_view(a.ViewportId, [](ViewportState &s) {
                    auto &d = s.Display;
                    if (d.ViewportShading == ViewportShadingMode::Wireframe) d.XRayWireframe = !d.XRayWireframe;
                    else d.XRaySolid = !d.XRaySolid;
                });
            },
        },
        action
    );
}
} // namespace action::view
