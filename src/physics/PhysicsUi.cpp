
#include "PhysicsUi.h"
#include "PhysicsSystem.h"
#include "action/Physics.h"
#include "timeline/Timeline.h"
#include "ui/ChoiceCombo.h"
#include "ui/FieldEdit.h"
#include "ui/ItemList.h"

#include <format>

using namespace ImGui;

namespace {
// Returns `name` if non-empty, else a bracketed fallback formatted from `fmt`/args (e.g., "<3>").
// Brackets signal "no explicit name" and can't be confused with a user-typed value.
// Uses vformat since consteval-checking of `fmt` is lost through the forwarded pack.
std::string DisplayName(std::string_view name, std::string_view fmt, auto &&...args) {
    if (!name.empty()) return std::string{name};
    return std::format("<{}>", std::vformat(fmt, std::make_format_args(args...)));
}

// Editable list of joint-def vec items, each a tree node with a delete button, plus an Add button.
// `body(item, edit)` renders the fields, where edit(fn) copies the item, applies fn, and returns the Set action.
template<typename T>
void RenderJointVecList(state::Entity jd_entity, const std::vector<T> &items, const char *label, const char *add_label, auto &&body) {
    PushID(label);
    const auto deleted = ui::ItemList(
        items.size(), [&](uint32_t i) { return std::format("{} {}", label, i); },
        [&](uint32_t i) {
            const auto edit = [&](auto &&fn) {
                auto e = items[i];
                fn(e);
                return action::physics::SetJointVecItem<T>{jd_entity, i, std::make_unique<T>(std::move(e))};
            };
            body(items[i], edit);
        }
    );
    if (deleted) action::Emit(action::physics::DeleteJointVecItem<T>{jd_entity, *deleted});
    if (Button(add_label)) action::Emit(action::physics::AddJointVecItem<T>{jd_entity});
    PopID();
}

// A row of radio buttons over every enumerator of E, true when one changes `value`.
template<typename E> bool RadioRow(int &value) {
    bool changed = false;
    for (size_t i = 0; i < field::EnumCount<E>; ++i) {
        if (i) SameLine();
        changed |= RadioButton(field::EnumLabels<E>[i], &value, int(field::EnumValues<E>[i]));
    }
    return changed;
}

template<typename Target>
void TargetCombo(state::Registry &r, const char *label, state::Entity current, const char *empty_preview, auto &&pick) {
    std::vector<state::Entity> choices{state::Null};
    for (const auto e : r.view<const Target>()) choices.push_back(e);
    if (choices.size() == 1) {
        BeginDisabled();
        if (BeginCombo(label, empty_preview)) EndCombo();
        EndDisabled();
        return;
    }
    const auto name = [&](state::Entity e) {
        const auto *target = e != state::Null && r.valid(e) ? r.try_get<const Target>(e) : nullptr;
        return target ? DisplayName(target->Name, "{:x}", uint32_t(e)) : std::string{"None"};
    };
    ui::ChoiceCombo(label, current, choices, name, pick);
}

std::string SystemDisplayName(const state::Registry &r, state::Entity e) {
    return !r.all_of<const CollisionSystem>(e) ? "<invalid>" : DisplayName(r.get<const CollisionSystem>(e).Name, "{:x}", uint32_t(e));
}

void RenderSystemMultiSelect(const state::Registry &r, const char *label, std::span<const state::Entity> selection, auto on_toggle) {
    const auto view = r.view<const CollisionSystem>();
    if (view.begin() == view.end()) {
        BeginDisabled();
        if (BeginCombo(label, "No systems defined")) EndCombo();
        EndDisabled();
        return;
    }

    std::string preview;
    if (selection.empty()) preview = "None";
    else if (selection.size() == 1) preview = SystemDisplayName(r, selection[0]);
    else preview = std::format("{} selected", selection.size());

    if (BeginCombo(label, preview.c_str())) {
        for (const auto se : view) {
            bool member = std::ranges::contains(selection, se);
            if (Checkbox(SystemDisplayName(r, se).c_str(), &member)) action::Emit(on_toggle(se, member));
        }
        EndCombo();
    }
}

void RenderCollisionFilterBody(state::Registry &r, state::Entity filter_e) {
    const auto &filter = r.get<const CollisionFilter>(filter_e);
    RenderSystemMultiSelect(r, "Member of", filter.Systems, [&](state::Entity se, bool add) {
        return action::physics::ToggleFilterEntity{filter_e, action::physics::ToggleFilterEntity::List::Systems, se, add};
    });

    auto mode = int(filter.Mode);
    TextUnformatted("Collide with:");
    SameLine();
    const bool mode_changed = RadioRow<CollideMode>(mode);
    if (mode_changed) action::Emit(action::UpdateOf<&CollisionFilter::Mode>(filter_e, CollideMode(mode)));

    if (mode != int(CollideMode::All)) {
        Indent();
        RenderSystemMultiSelect(r, "##collide", filter.CollideSystems, [&](state::Entity se, bool add) {
            return action::physics::ToggleFilterEntity{filter_e, action::physics::ToggleFilterEntity::List::CollideSystems, se, add};
        });
        if (filter.CollideSystems.empty() && mode == int(CollideMode::Allowlist)) {
            TextColored(ImVec4{0.9f, 0.7f, 0.3f, 1}, "No systems selected — this filter collides with nothing.");
        }
        Unindent();
    }
}

// Split each cell along the top-right to bottom-left diagonal.
// The upper-left triangle shows row-to-column permission, and the lower-right shows column-to-row permission.
// Green indicates mutual permission, red indicates a one-way permission, and an unpainted triangle indicates denial.
void DrawMatrixCell(ImDrawList *dl, ImVec2 p_min, ImVec2 p_max, bool a_to_b, bool b_to_a) {
    const auto collide = IM_COL32(60, 200, 60, 220);
    const auto overridden = IM_COL32(200, 60, 60, 220);
    const auto fill = (a_to_b && b_to_a) ? collide : overridden;
    const auto TL = p_min, BR = p_max;
    const ImVec2 TR{p_max.x, p_min.y}, BL{p_min.x, p_max.y};
    if (a_to_b) dl->AddTriangleFilled(TL, TR, BL, fill); // row→col
    if (b_to_a) dl->AddTriangleFilled(TR, BR, BL, fill); // col→row
}

// List of named entities with use count, name edit, Delete/Add buttons, and per-entry body.
// Both the list and node inspector use RenderResourceProperties for editing.
template<typename T>
void DrawNamedEntityList(state::Registry &r, const char *id, const char *add_label, auto add) {
    PushID(id);
    std::vector<state::Entity> entities;
    for (const auto e : r.view<T>()) entities.push_back(e);
    const auto &uses = r.Resources.get<const PhysicsDefinitionUses>();
    const auto deleted = ui::ItemList(
        entities.size(),
        [&](uint32_t i) { return std::format("{} ({})", DisplayName(r.get<const T>(entities[i]).Name, "{:x}", uint32_t(entities[i])), uses.Count(entities[i])); },
        [&](uint32_t i) { physics_ui::RenderResourceProperties(r, entities[i]); }
    );
    if (deleted) action::Emit(action::DestroyEntity{entities[*deleted]});
    if (Button(add_label)) action::Emit(add);
    PopID();
}

// Returns the edited shape iff the user changed something.
// Auto-fit sets the box, sphere, capsule, and cylinder dimensions, and their fields are disabled while it is on.
std::optional<PhysicsShape> RenderShapeEditor(const PhysicsShape &in, bool auto_fit) {
    static const char *const shape_names[]{"Box", "Sphere", "Capsule", "Cylinder", "Plane", "Convex Hull", "Triangle Mesh"};
    auto out = in;
    bool changed = false;
    if (auto shape_type_i = int(out.index());
        Combo("Shape", &shape_type_i, shape_names, IM_ARRAYSIZE(shape_names))) {
        out = CreateVariantByIndex<PhysicsShape>(shape_type_i);
        changed = true;
    }
    std::visit(
        overloaded{
            [&](physics::Plane &s) { ui::DrawFields<physics::Plane>(ui::ValueEdit{s, changed}); },
            [](physics::ConvexHull &) {},
            [](physics::TriangleMesh &) {},
            [&]<typename S>(S &s) {
                BeginDisabled(auto_fit);
                ui::DrawFields<S>(ui::ValueEdit{s, changed});
                EndDisabled();
            },
        },
        out
    );
    return changed ? std::optional{std::move(out)} : std::nullopt;
}
} // namespace

namespace {
void RenderJointDefinition(state::Entity jd_entity, const PhysicsJointDef &jd) {
    static const char *const axis_names[]{"X", "Y", "Z"};
    RenderJointVecList<PhysicsJointLimit>(jd_entity, jd.Limits, "Limit", "Add limit", [&](const auto &limit, auto &&edit_limit) {
        const auto axes = [&](const char *label, auto member, bool angular) {
            TextUnformatted(label);
            SameLine();
            for (uint8_t a = 0; a < 3; ++a) {
                if (angular) PushID(a + 3);
                bool active = std::ranges::contains(limit.*member, a);
                if (Checkbox(axis_names[a], &active)) action::Emit(edit_limit([&](auto &e) {
                    if (active) (e.*member).push_back(a);
                    else std::erase(e.*member, a);
                }));
                if (a < 2) SameLine();
                if (angular) PopID();
            }
        };
        axes("Linear axes:", &PhysicsJointLimit::LinearAxes, false);
        axes("Angular axes:", &PhysicsJointLimit::AngularAxes, true);
        const auto bound = [&](const char *label, const char *id, auto member) {
            bool enabled = (limit.*member).has_value();
            float value = (limit.*member).value_or(0.f);
            if (Checkbox(label, &enabled)) action::Emit(edit_limit([&](auto &e) { e.*member = enabled ? std::optional{value} : std::nullopt; }));
            if (enabled) {
                SameLine();
                ui::Gesture(ui::DragFloat(id, &value, .01f), [&] { return edit_limit([&](auto &e) { e.*member = value; }); });
            }
        };
        bound("Min", "##min", &PhysicsJointLimit::Min);
        bound("Max", "##max", &PhysicsJointLimit::Max);

        bool soft = limit.Stiffness.has_value();
        if (Checkbox("Soft limit", &soft)) action::Emit(edit_limit([&](auto &e) { e.Stiffness = soft ? std::optional{1000.0f} : std::nullopt; }));
        if (limit.Stiffness) {
            float stiffness = *limit.Stiffness, damping = limit.Damping;
            ui::Gesture(ui::DragFloat("Stiffness", &stiffness, 1.0f, 0.0f, 1e6f), [&] { return edit_limit([&](auto &e) { e.Stiffness = stiffness; }); });
            ui::Gesture(ui::DragFloat("Damping", &damping, 0.1f, 0.0f, 1e4f), [&] { return edit_limit([&](auto &e) { e.Damping = damping; }); });
        }
    });

    Spacing();

    RenderJointVecList<PhysicsJointDrive>(jd_entity, jd.Drives, "Drive", "Add drive", [&](const auto &drive, auto &&edit_drive) {
        if (auto type = int(drive.Type); Combo("Type", &type, "Linear\0Angular\0")) action::Emit(edit_drive([&](auto &e) { e.Type = PhysicsDriveType(type); }));
        if (int axis = drive.Axis; Combo("Axis", &axis, "X\0Y\0Z\0")) action::Emit(edit_drive([&](auto &e) { e.Axis = uint8_t(axis); }));
        if (auto mode = int(drive.Mode); Combo("Mode", &mode, "Force\0Acceleration\0")) action::Emit(edit_drive([&](auto &e) { e.Mode = PhysicsDriveMode(mode); }));
        float max_force = drive.MaxForce, pos_target = drive.PositionTarget, vel_target = drive.VelocityTarget;
        float stiffness = drive.Stiffness, damping = drive.Damping;
        ui::Gesture(ui::DragFloat("Max force", &max_force, 1.0f, 0.0f, 1e6f), [&] { return edit_drive([&](auto &e) { e.MaxForce = max_force; }); });
        ui::Gesture(ui::DragFloat("Position target", &pos_target, 0.01f), [&] { return edit_drive([&](auto &e) { e.PositionTarget = pos_target; }); });
        ui::Gesture(ui::DragFloat("Velocity target", &vel_target, 0.01f), [&] { return edit_drive([&](auto &e) { e.VelocityTarget = vel_target; }); });
        ui::Gesture(ui::DragFloat("Stiffness", &stiffness, 1.0f, 0.0f, 1e6f), [&] { return edit_drive([&](auto &e) { e.Stiffness = stiffness; }); });
        ui::Gesture(ui::DragFloat("Damping", &damping, 0.1f, 0.0f, 1e4f), [&] { return edit_drive([&](auto &e) { e.Damping = damping; }); });
    });
}
} // namespace

bool physics_ui::RenderResourceProperties(state::Registry &r, state::Entity entity) {
    const auto rename = [&](const auto &value) {
        char name[256];
        snprintf(name, sizeof name, "%s", value.Name.c_str());
        if (InputText("Name", name, sizeof name)) action::Emit(action::physics::RenameResource{entity, name});
    };
    if (const auto *m = r.try_get<const PhysicsMaterial>(entity)) {
        rename(*m);
        ui::DrawFields<PhysicsMaterial, &PhysicsMaterial::Name>(ui::Edit{r, entity});
    } else if (const auto *system = r.try_get<const CollisionSystem>(entity)) {
        rename(*system);
    } else if (const auto *filter = r.try_get<const CollisionFilter>(entity)) {
        rename(*filter);
        RenderCollisionFilterBody(r, entity);
    } else if (const auto *joint = r.try_get<const PhysicsJointDef>(entity)) {
        rename(*joint);
        RenderJointDefinition(entity, *joint);
    } else return false;
    return true;
}

void physics_ui::RenderTab(state::Registry &r) {
    SeparatorText("Simulation");
    Text("Bodies: %u", physics::BodyCount(r));
    ui::DrawFields<PhysicsSimulationSettings>(ui::Edit{r, state::Document});

    if (CollapsingHeader("Physics Materials")) {
        DrawNamedEntityList<PhysicsMaterial>(r, "PhysMaterials", "Add material", action::physics::AddPhysicsMaterial{});
    }

    if (CollapsingHeader("Collision Systems")) {
        DrawNamedEntityList<CollisionSystem>(r, "CollisionSystems", "Add system", action::physics::AddCollisionSystem{});
    }

    if (CollapsingHeader("Collision Filters")) {
        DrawNamedEntityList<CollisionFilter>(r, "CollisionFilters", "Add filter", action::physics::AddCollisionFilter{});
        PushID("CollisionFilters");

        std::vector<state::Entity> filter_entities;
        for (auto e : r.view<CollisionFilter>()) filter_entities.emplace_back(e);
        if (filter_entities.size() >= 2) {
            Spacing();
            SeparatorText("Collision Matrix");

            const auto n = filter_entities.size();
            const auto filter_label = [&](size_t i) {
                const auto fe = filter_entities[i];
                return DisplayName(r.get<const CollisionFilter>(fe).Name, "{:x}", uint32_t(fe));
            };

            if (BeginTable("##CollisionMatrix", int(n) + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit)) {
                TableSetupColumn("", ImGuiTableColumnFlags_NoHeaderLabel);
                for (size_t i = 0; i < n; ++i) TableSetupColumn(filter_label(i).c_str(), ImGuiTableColumnFlags_AngledHeader);
                TableAngledHeadersRow();

                const float cell_sz = GetFrameHeight();
                for (size_t row = 0; row < n; ++row) {
                    TableNextRow();
                    TableSetColumnIndex(0);
                    TextUnformatted(filter_label(row).c_str());
                    for (size_t col = 0; col < n; ++col) {
                        TableSetColumnIndex(int(col) + 1);
                        PushID(int(row * n + col));
                        const auto a = filter_entities[row], b = filter_entities[col];
                        const bool ab = physics::DoesFilterAllow(r, a, b);
                        const bool ba = physics::DoesFilterAllow(r, b, a);
                        const ImVec2 p = GetCursorScreenPos();
                        DrawMatrixCell(GetWindowDrawList(), p, {p.x + cell_sz, p.y + cell_sz}, ab, ba);
                        Dummy({cell_sz, cell_sz});
                        if (IsItemHovered()) {
                            const auto rn = filter_label(row), cn = filter_label(col);
                            BeginTooltip();
                            Text("%s \xe2\x86\x92 %s: %s", rn.c_str(), cn.c_str(), ab ? "allows" : "vetoes");
                            Text("%s \xe2\x86\x92 %s: %s", cn.c_str(), rn.c_str(), ba ? "allows" : "vetoes");
                            Separator();
                            const bool both = ab && ba;
                            TextColored(both ? ImVec4{0.4f, 0.9f, 0.4f, 1} : ImVec4{0.9f, 0.4f, 0.4f, 1}, "Effective: %s", both ? "collides" : (ab != ba ? "blocked (one side vetoes)" : "blocked"));
                            EndTooltip();
                        }
                        PopID();
                    }
                }
                EndTable();
            }
        }
        PopID();
    }

    if (CollapsingHeader("Joint Definitions")) {
        DrawNamedEntityList<PhysicsJointDef>(r, "JointDefs", "Add joint definition", action::physics::AddJointDef{});
    }
}

void physics_ui::RenderEntityProperties(state::Registry &r, state::Entity entity, bool explicit_target) {
    if (!CollapsingHeader("Physics")) return;

    PushID("PhysicsEntity");
    const action::Target active_target = explicit_target ? action::Target{entity} : action::Target{action::OnActive{}};
    const auto scope = [&] { return explicit_target ? action::Target{entity} : ui::TargetFromAlt(); };
    ui::Edit fields{r, explicit_target ? std::optional{entity} : std::nullopt};

    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    const auto *collider = r.try_get<const ColliderShape>(entity);

    // No body, static collider, dynamic body, or kinematic body.
    // Projects losslessly to/from {ColliderShape?, PhysicsMotion?, PhysicsMotion::IsKinematic}.
    using MType = action::physics::SetMotionType::Type;
    auto motion_type = int(MType::None);
    if (collider && !motion) motion_type = int(MType::Static);
    else if (motion && motion->IsKinematic) motion_type = int(MType::Kinematic);
    else if (motion) motion_type = int(MType::Dynamic);

    AlignTextToFramePadding();
    TextUnformatted("Motion type:");
    SameLine();
    if (RadioRow<MType>(motion_type)) action::Emit(action::physics::SetMotionType{MType(motion_type), scope()});

    if (collider) { // Collider shape editing
        Spacing();
        SeparatorText("Collider");

        fields.Draw<&ColliderPolicy::AutoFitDims>("Auto-fit");
        auto s = RenderShapeEditor(collider->Shape, r.get<const ColliderPolicy>(entity).AutoFitDims);
        ui::Gesture(bool(s), [&, target = scope()] { return action::physics::SetColliderShape{*s, s->index() != collider->Shape.index(), target}; });

        const auto &material = r.get<const ColliderMaterial>(entity);
        TargetCombo<PhysicsMaterial>(r, "Physics material", material.PhysicsMaterialEntity, "No materials defined", [&](state::Entity e) {
            action::Emit(action::UpdateOf<&ColliderMaterial::PhysicsMaterialEntity>(active_target, e));
        });
        TargetCombo<CollisionFilter>(r, "Collision filter", material.CollisionFilterEntity, "No filters defined", [&](state::Entity e) {
            action::Emit(action::UpdateOf<&ColliderMaterial::CollisionFilterEntity>(active_target, e));
        });
    }

    if (motion) {
        Spacing();
        SeparatorText("Motion");

        // Initial velocity is editable at the authored start frame while playback is paused.
        const auto &range = r.get<const TimelineRange>(state::Document);
        const auto &playback = r.get<const TimelinePlayback>(state::Document);
        const bool velocity_locked = playback.Playing || physics::BakedThrough(r) > range.StartFrame;
        if (velocity_locked) BeginDisabled();
        if (r.try_get<const PhysicsVelocity>(entity)) ui::DrawFields<PhysicsVelocity>(fields);
        if (velocity_locked) EndDisabled();

        // Kinematic bodies have no mass properties, gravity, or damping.
        if (!motion->IsKinematic) ui::DrawFields<PhysicsMotion, &PhysicsMotion::IsKinematic>(fields);
    }

    if (const auto *joint = r.try_get<const PhysicsJoint>(entity)) {
        Spacing();
        SeparatorText("Joint");

        TargetCombo<PhysicsJointDef>(r, "Definition", joint->JointDefEntity, "No joint definitions", [&](state::Entity e) {
            action::Emit(action::UpdateOf<&PhysicsJoint::JointDefEntity>(active_target, e));
        });
        if (const auto *def = joint->JointDefEntity != state::Null && r.valid(joint->JointDefEntity) ? r.try_get<const PhysicsJointDef>(joint->JointDefEntity) : nullptr) {
            Text("Limits: %zu, Drives: %zu", def->Limits.size(), def->Drives.size());
        }

        fields.Draw<&PhysicsJoint::EnableCollision>();

        // ConnectedNode picker.
        // KHR joint.connectedNode is the second attachment frame.
        // Mirrors Blender's rigid_body_constraint object1/object2 fields.
        const auto cn = joint->ConnectedNode;
        ui::ChoiceCombo(
            "Connected node", cn != state::Null && r.valid(cn) ? cn : state::Null,
            [&] {
                std::vector<state::Entity> nodes{state::Null};
                for (const auto ne : r.view<const SceneChildren>()) {
                    if (ne != entity) nodes.push_back(ne);
                }
                return nodes;
            },
            [&](state::Entity e) { return e == state::Null ? std::string{"None"} : GetName(r, e); },
            [&](state::Entity e) { action::Emit(action::UpdateOf<&PhysicsJoint::ConnectedNode>(active_target, e)); }
        );
    }

    if (const auto *trigger_nodes = r.try_get<const TriggerNodes>(entity)) {
        Spacing();
        SeparatorText("Trigger (compound)");
        PushID("Trigger");
        Text("Compound trigger: %zu nodes", trigger_nodes->Nodes.size());
        TargetCombo<CollisionFilter>(r, "Collision filter", trigger_nodes->CollisionFilterEntity, "No filters defined", [&](state::Entity e) {
            action::Emit(action::UpdateOf<&TriggerNodes::CollisionFilterEntity>(active_target, e));
        });
        if (Button("Remove Trigger")) action::Emit(action::physics::RemoveTriggerNodes{active_target});
        PopID();
    } else {
        Spacing();
        const bool is_shape_trigger = collider && r.all_of<const TriggerTag>(entity);
        if (collider) {
            if (is_shape_trigger) {
                if (Button("Convert to Collider")) action::Emit(action::physics::SetTrigger{false, active_target});
            } else {
                if (Button("Convert to Trigger")) action::Emit(action::physics::SetTrigger{true, active_target});
            }
        } else if (Button("Add Trigger")) {
            action::Emit(action::physics::AddTrigger{active_target});
        }
    }

    PopID();
}
