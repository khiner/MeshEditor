// Physics UI: "Physics" tab + per-entity physics properties.

#include "PhysicsUi.h"
#include "PhysicsSystem.h"
#include "Variant.h"
#include "action/Physics.h"
#include "animation/AnimationTimeline.h"
#include "numeric/Angles.h"
#include "numeric/vec2.h"
#include "scene/SceneGraph.h"
#include "ui/ChoiceCombo.h"
#include "ui/FieldEdit.h"
#include "ui/ItemList.h"

#include "state/Scene.h"
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

// Picks an entity holding `Target` by name, or None. Disabled with `empty_preview` while no Target exists.
template<typename Target>
void TargetCombo(state::Scene &r, const char *label, state::Entity current, const char *empty_preview, auto &&pick) {
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

std::string SystemDisplayName(const state::Scene &r, state::Entity e) {
    return !r.all_of<const CollisionSystem>(e) ? "<invalid>" : DisplayName(r.get<const CollisionSystem>(e).Name, "{:x}", uint32_t(e));
}

// Renders a CollisionSystem multi-select combo and calls on_toggle for membership changes.
// Renders a disabled combo when no systems exist.
template<typename Fn>
void RenderSystemMultiSelect(const state::Scene &r, const char *label, const std::vector<state::Entity> &selection, Fn on_toggle) {
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
        for (auto se : view) {
            bool member = std::find(selection.begin(), selection.end(), se) != selection.end();
            if (Checkbox(SystemDisplayName(r, se).c_str(), &member)) action::Emit(on_toggle(se, member));
        }
        EndCombo();
    }
}

size_t CountFilterUses(const state::Scene &r, state::Entity filter) {
    size_t n = 0;
    for (auto [e, m] : r.view<const ColliderMaterial>().each()) {
        if (m.CollisionFilterEntity == filter) ++n;
    }
    for (auto [e, t] : r.view<const TriggerNodes>().each()) {
        if (t.CollisionFilterEntity == filter) ++n;
    }
    return n;
}

// Edits collision-filter membership, mode, and collision systems in the Physics tab.
void RenderCollisionFilterBody(state::Scene &r, state::Entity filter_e) {
    const auto &filter = r.get<const CollisionFilter>(filter_e);
    RenderSystemMultiSelect(r, "Member of", filter.Systems, [&](state::Entity se, bool add) {
        return action::physics::ToggleFilterEntity{filter_e, action::physics::ToggleFilterEntity::List::Systems, se, add};
    });

    int mode = int(filter.Mode);
    TextUnformatted("Collide with:");
    SameLine();
    bool mode_changed = RadioButton("All", &mode, int(CollideMode::All));
    SameLine();
    mode_changed |= RadioButton("Allowlist", &mode, int(CollideMode::Allowlist));
    SameLine();
    mode_changed |= RadioButton("Blocklist", &mode, int(CollideMode::Blocklist));
    if (mode_changed) action::Emit(action::UpdateOn<&CollisionFilter::Mode>(filter_e, CollideMode(mode)));

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
// `add` is emitted by the Add button, `rename(e, name)` returns the rename action, and `body(e, x)` renders the entry.
template<typename T>
void DrawNamedEntityList(state::Scene &r, const char *id, const char *add_label, auto add, auto &&rename, auto &&count, auto &&body) {
    PushID(id);
    std::vector<state::Entity> entities;
    for (const auto e : r.view<T>()) entities.push_back(e);
    const auto deleted = ui::ItemList(
        entities.size(),
        [&](uint32_t i) { return std::format("{} ({})", DisplayName(r.get<const T>(entities[i]).Name, "{:x}", uint32_t(entities[i])), count(entities[i])); },
        [&](uint32_t i) {
            const auto e = entities[i];
            const auto &x = r.get<const T>(e);
            char buf[128];
            snprintf(buf, sizeof(buf), "%s", x.Name.c_str());
            if (InputText("Name", buf, sizeof(buf))) action::Emit(rename(e, std::string{buf}));
            body(e, x);
        }
    );
    if (deleted) action::Emit(action::DestroyEntity{entities[*deleted]});
    if (Button(add_label)) action::Emit(add);
    PopID();
}

// Returns the edited shape iff the user changed something.
// When `auto_fit`, dim widgets for AABB-fittable kinds are hidden — the fitter owns them.
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
            [&](physics::Box &s) {
                if (!auto_fit) changed |= ui::DragFloat3("Size", &s.Size.x, 0.01f, 0.01f, 100.f);
            },
            [&](physics::Sphere &s) {
                if (!auto_fit) changed |= ui::DragFloat("Radius", &s.Radius, 0.01f, 0.001f, 100.f);
            },
            [&]<typename S>(S &s)
                requires(std::same_as<S, physics::Capsule> || std::same_as<S, physics::Cylinder>)
            {
                if (auto_fit) return;
                changed |= ui::DragFloat("Height", &s.Height, 0.01f, 0.001f, 100.f);
                changed |= ui::DragFloat("Radius top", &s.RadiusTop, 0.01f, 0.001f, 100.f);
                changed |= ui::DragFloat("Radius bottom", &s.RadiusBottom, 0.01f, 0.001f, 100.f);
            },
            [&](physics::Plane &s) {
                // Plane normal is +Y; sizeX/sizeZ = 0 means infinite extent along that axis.
                bool infinite = s.SizeX <= 0.f || s.SizeZ <= 0.f;
                if (Checkbox("Infinite", &infinite)) {
                    s.SizeX = s.SizeZ = infinite ? 0.f : 2.f;
                    changed = true;
                }
                if (!infinite) {
                    vec2 size{s.SizeX, s.SizeZ};
                    if (ui::DragFloat2("Size (X, Z)", &size.x, 0.01f, 0.01f, 1000.f)) {
                        s.SizeX = size.x;
                        s.SizeZ = size.y;
                        changed = true;
                    }
                }
                changed |= Checkbox("Double-sided", &s.DoubleSided);
            },
            [](physics::ConvexHull &) {},
            [](physics::TriangleMesh &) {},
            },
            out
    );
    return changed ? std::optional{std::move(out)} : std::nullopt;
}
} // namespace

void physics_ui::RenderTab(state::Scene &r, state::Entity viewport) {
    SeparatorText("Simulation");
    Text("Bodies: %u", physics::BodyCount(r));
    {
        ui::Edit f{r, viewport};
        f.Slider<&PhysicsSimulationSettings::SubstepsPerFrame>("Substeps per frame");
        f.Slider<&PhysicsSimulationSettings::SolverIterations>("Solver iterations");
        f.Slider<&PhysicsSimulationSettings::TimeScale>("Time scale", "%.2fx");
        f.Drag<&PhysicsSimulationSettings::Gravity>("Gravity", 0.1f);
    }

    if (CollapsingHeader("Physics Materials")) {
        DrawNamedEntityList<PhysicsMaterial>(
            r, "PhysMaterials", "Add material", action::physics::AddPhysicsMaterial{},
            [](state::Entity e, std::string name) { return action::physics::RenamePhysicsMaterial{e, std::move(name)}; },
            [&](state::Entity mat_entity) {
                size_t n = 0;
                for (auto [e, m] : r.view<const ColliderMaterial>().each()) {
                    if (m.PhysicsMaterialEntity == mat_entity) ++n;
                }
                return n;
            },
            [&](state::Entity mat_entity, const PhysicsMaterial &) {
                ui::Edit f{r, mat_entity};
                f.Slider<&PhysicsMaterial::StaticFriction>("Static friction");
                f.Slider<&PhysicsMaterial::DynamicFriction>("Dynamic friction");
                f.Slider<&PhysicsMaterial::Restitution>("Restitution");
                f.Enum<&PhysicsMaterial::FrictionCombine>("Friction combine", "Average\0Minimum\0Maximum\0Multiply\0");
                f.Enum<&PhysicsMaterial::RestitutionCombine>("Restitution combine", "Average\0Minimum\0Maximum\0Multiply\0");
            }
        );
    }

    if (CollapsingHeader("Collision Systems")) {
        DrawNamedEntityList<CollisionSystem>(
            r, "CollisionSystems", "Add system", action::physics::AddCollisionSystem{},
            [](state::Entity e, std::string name) { return action::physics::RenameCollisionSystem{e, std::move(name)}; },
            [&](state::Entity se) {
                size_t n = 0;
                for (auto [fe, f] : r.view<const CollisionFilter>().each()) {
                    if (std::find(f.Systems.begin(), f.Systems.end(), se) != f.Systems.end() ||
                        std::find(f.CollideSystems.begin(), f.CollideSystems.end(), se) != f.CollideSystems.end()) ++n;
                }
                return n;
            },
            [](state::Entity, const CollisionSystem &) {}
        );
    }

    if (CollapsingHeader("Collision Filters")) {
        DrawNamedEntityList<CollisionFilter>(
            r, "CollisionFilters", "Add filter", action::physics::AddCollisionFilter{},
            [](state::Entity e, std::string name) { return action::physics::RenameCollisionFilter{e, std::move(name)}; },
            [&](state::Entity fe) { return CountFilterUses(r, fe); },
            [&](state::Entity fe, const CollisionFilter &) { RenderCollisionFilterBody(r, fe); }
        );
        PushID("CollisionFilters");

        // Collision matrix
        std::vector<state::Entity> filter_entities;
        for (auto e : r.view<CollisionFilter>()) filter_entities.emplace_back(e);
        if (filter_entities.size() >= 2) {
            Spacing();
            SeparatorText("Collision Matrix");

            const auto n = filter_entities.size();
            auto filter_label = [&](size_t i) {
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
        DrawNamedEntityList<PhysicsJointDef>(
            r, "JointDefs", "Add joint definition", action::physics::AddJointDef{},
            [](state::Entity e, std::string name) { return action::physics::RenameJointDef{e, std::move(name)}; },
            [&](state::Entity jd_entity) {
                size_t n = 0;
                for (auto [e, j] : r.view<const PhysicsJoint>().each()) {
                    if (j.JointDefEntity == jd_entity) ++n;
                }
                return n;
            },
            [&](state::Entity jd_entity, const PhysicsJointDef &jd) {
                static const char *const axis_names[]{"X", "Y", "Z"};
                RenderJointVecList<PhysicsJointLimit>(jd_entity, jd.Limits, "Limit", "Add limit", [&](const auto &limit, auto &&edit_limit) {
                    TextUnformatted("Linear axes:");
                    SameLine();
                    for (uint8_t a = 0; a < 3; ++a) {
                        bool active = std::find(limit.LinearAxes.begin(), limit.LinearAxes.end(), a) != limit.LinearAxes.end();
                        if (Checkbox(axis_names[a], &active)) action::Emit(edit_limit([&](auto &e) {
                            if (active) e.LinearAxes.push_back(a);
                            else std::erase(e.LinearAxes, a);
                        }));
                        if (a < 2) SameLine();
                    }
                    TextUnformatted("Angular axes:");
                    SameLine();
                    for (uint8_t a = 0; a < 3; ++a) {
                        PushID(a + 3);
                        bool active = std::find(limit.AngularAxes.begin(), limit.AngularAxes.end(), a) != limit.AngularAxes.end();
                        if (Checkbox(axis_names[a], &active)) action::Emit(edit_limit([&](auto &e) {
                            if (active) e.AngularAxes.push_back(a);
                            else std::erase(e.AngularAxes, a);
                        }));
                        if (a < 2) SameLine();
                        PopID();
                    }

                    bool has_min = limit.Min.has_value(), has_max = limit.Max.has_value();
                    float min_val = limit.Min.value_or(0.0f), max_val = limit.Max.value_or(0.0f);
                    if (Checkbox("Min", &has_min)) action::Emit(edit_limit([&](auto &e) { e.Min = has_min ? std::optional{min_val} : std::nullopt; }));
                    if (has_min) {
                        SameLine();
                        ui::Gesture(ui::DragFloat("##min", &min_val, 0.01f), [&] { return edit_limit([&](auto &e) { e.Min = min_val; }); });
                    }
                    if (Checkbox("Max", &has_max)) action::Emit(edit_limit([&](auto &e) { e.Max = has_max ? std::optional{max_val} : std::nullopt; }));
                    if (has_max) {
                        SameLine();
                        ui::Gesture(ui::DragFloat("##max", &max_val, 0.01f), [&] { return edit_limit([&](auto &e) { e.Max = max_val; }); });
                    }

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
                    if (int type = int(drive.Type); Combo("Type", &type, "Linear\0Angular\0")) action::Emit(edit_drive([&](auto &e) { e.Type = PhysicsDriveType(type); }));
                    if (int axis = drive.Axis; Combo("Axis", &axis, "X\0Y\0Z\0")) action::Emit(edit_drive([&](auto &e) { e.Axis = uint8_t(axis); }));
                    if (int mode = int(drive.Mode); Combo("Mode", &mode, "Force\0Acceleration\0")) action::Emit(edit_drive([&](auto &e) { e.Mode = PhysicsDriveMode(mode); }));
                    float max_force = drive.MaxForce, pos_target = drive.PositionTarget, vel_target = drive.VelocityTarget;
                    float stiffness = drive.Stiffness, damping = drive.Damping;
                    ui::Gesture(ui::DragFloat("Max force", &max_force, 1.0f, 0.0f, 1e6f), [&] { return edit_drive([&](auto &e) { e.MaxForce = max_force; }); });
                    ui::Gesture(ui::DragFloat("Position target", &pos_target, 0.01f), [&] { return edit_drive([&](auto &e) { e.PositionTarget = pos_target; }); });
                    ui::Gesture(ui::DragFloat("Velocity target", &vel_target, 0.01f), [&] { return edit_drive([&](auto &e) { e.VelocityTarget = vel_target; }); });
                    ui::Gesture(ui::DragFloat("Stiffness", &stiffness, 1.0f, 0.0f, 1e6f), [&] { return edit_drive([&](auto &e) { e.Stiffness = stiffness; }); });
                    ui::Gesture(ui::DragFloat("Damping", &damping, 0.1f, 0.0f, 1e4f), [&] { return edit_drive([&](auto &e) { e.Damping = damping; }); });
                });
            }
        );
    }
}

void physics_ui::RenderEntityProperties(state::Scene &r, state::Entity entity, state::Entity viewport) {
    if (!CollapsingHeader("Physics")) return;

    PushID("PhysicsEntity");

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
    bool changed = RadioButton("None", &motion_type, int(MType::None));
    SameLine();
    changed |= RadioButton("Static", &motion_type, int(MType::Static));
    SameLine();
    changed |= RadioButton("Kinematic", &motion_type, int(MType::Kinematic));
    SameLine();
    changed |= RadioButton("Dynamic", &motion_type, int(MType::Dynamic));

    if (changed) action::Emit(action::physics::SetMotionType{MType(motion_type), ui::ScopeFromAlt()});

    if (collider) { // Collider shape editing
        Spacing();
        SeparatorText("Collider");

        ui::Edit{r}.Check<&ColliderPolicy::AutoFitDims>("Auto-fit");
        auto s = RenderShapeEditor(collider->Shape, r.get<const ColliderPolicy>(entity).AutoFitDims);
        ui::Gesture(bool(s), [&, scope = ui::ScopeFromAlt()] { return action::physics::SetColliderShape{*s, s->index() != collider->Shape.index(), scope}; });

        const auto &material = r.get<const ColliderMaterial>(entity);
        TargetCombo<PhysicsMaterial>(r, "Physics material", material.PhysicsMaterialEntity, "No materials defined", [](state::Entity e) {
            action::Emit(action::UpdateActive<&ColliderMaterial::PhysicsMaterialEntity>(e));
        });
        TargetCombo<CollisionFilter>(r, "Collision filter", material.CollisionFilterEntity, "No filters defined", [](state::Entity e) {
            action::Emit(action::UpdateActive<&ColliderMaterial::CollisionFilterEntity>(e));
        });
    }

    // Motion properties editing
    if (motion) {
        Spacing();
        SeparatorText("Motion");

        // Initial velocity is editable at the authored start frame while playback is paused.
        const auto &range = r.get<const TimelineRange>(viewport);
        const auto &playback = r.get<const TimelinePlayback>(viewport);
        const bool velocity_locked = playback.Playing || physics::BakedThrough(r) > range.StartFrame;
        if (velocity_locked) BeginDisabled();
        if (r.try_get<const PhysicsVelocity>(entity)) {
            ui::Edit f{r};
            f.Drag<&PhysicsVelocity::Linear>("Linear velocity", 0.1f);
            f.Drag<&PhysicsVelocity::Angular>("Angular velocity", 0.1f);
        }
        if (velocity_locked) EndDisabled();

        // Mass/inertia/damping/gravity are only meaningful for Dynamic bodies.
        // Kinematic bodies move purely by velocity assignment; these fields are hidden to avoid noise.
        if (!motion->IsKinematic) {
            ui::Edit f{r};
            f.Drag<&PhysicsMotion::GravityFactor>("Gravity factor", 0.01f);

            Spacing();
            SeparatorText("Mass properties");

            // Mass/inertia/CoM are std::optional with presence toggles, not flat fields Update can address.
            PhysicsMotion edit = *motion;
            bool motion_changed = false;
            float mass = edit.Mass.value_or(DefaultMass);
            if (ui::DragFloat("Mass", &mass, 0.1f, 0.001f, 1e6f, "%.3f kg")) {
                edit.Mass = mass;
                motion_changed = true;
            }

            bool has_inertia = edit.InertiaDiagonal.has_value();
            if (Checkbox("Override inertia tensor", &has_inertia)) {
                if (has_inertia) {
                    edit.InertiaDiagonal = vec3{1.0f};
                    edit.InertiaOrientation = quat{1, 0, 0, 0};
                } else {
                    edit.InertiaDiagonal.reset();
                    edit.InertiaOrientation.reset();
                }
                motion_changed = true;
            }
            if (edit.InertiaDiagonal) {
                motion_changed |= ui::DragFloat3("Inertia diagonal", &edit.InertiaDiagonal->x, 0.01f, 0.001f, 1e6f);
                vec3 euler_deg = numeric::Degrees(numeric::EulerAngles(edit.InertiaOrientation.value_or(quat{1, 0, 0, 0})));
                if (ui::DragFloat3("Inertia orientation", &euler_deg.x, 0.1f)) {
                    edit.InertiaOrientation = quat{numeric::Radians(euler_deg)};
                    motion_changed = true;
                }
            }

            if (bool has_com = edit.CenterOfMass.has_value(); Checkbox("Override center of mass", &has_com)) {
                edit.CenterOfMass = has_com ? std::optional{vec3{0.0f}} : std::nullopt;
                motion_changed = true;
            }
            if (edit.CenterOfMass) motion_changed |= ui::DragFloat3("Center of mass", &edit.CenterOfMass->x, 0.01f);
            ui::Gesture(motion_changed, [&, scope = ui::ScopeFromAlt()] { return action::physics::SetMotion{std::make_unique<PhysicsMotion>(edit), scope}; });

            Spacing();
            SeparatorText("Dynamics");

            f.Drag<&PhysicsMotion::LinearDamping>("Damping translation", 0.01f);
            f.Drag<&PhysicsMotion::AngularDamping>("Damping rotation", 0.01f);
        }
    }

    // Joint properties
    if (const auto *joint = r.try_get<const PhysicsJoint>(entity)) {
        Spacing();
        SeparatorText("Joint");

        TargetCombo<PhysicsJointDef>(r, "Definition", joint->JointDefEntity, "No joint definitions", [](state::Entity e) {
            action::Emit(action::UpdateActive<&PhysicsJoint::JointDefEntity>(e));
        });
        if (const auto *def = joint->JointDefEntity != state::Null && r.valid(joint->JointDefEntity) ? r.try_get<const PhysicsJointDef>(joint->JointDefEntity) : nullptr) {
            Text("Limits: %zu, Drives: %zu", def->Limits.size(), def->Drives.size());
        }

        ui::Edit{r}.Check<&PhysicsJoint::EnableCollision>("Enable collision");

        // ConnectedNode picker. KHR joint.connectedNode is the second attachment frame.
        // Mirrors Blender's rigid_body_constraint object1/object2 fields.
        std::vector<state::Entity> nodes{state::Null};
        for (const auto ne : r.view<const SceneNode>())
            if (ne != entity) nodes.push_back(ne);
        const auto cn = joint->ConnectedNode;
        ui::ChoiceCombo(
            "Connected node", cn != state::Null && r.valid(cn) ? cn : state::Null, nodes,
            [&](state::Entity e) { return e == state::Null ? std::string{"None"} : GetName(r, e); },
            [](state::Entity e) { action::Emit(action::UpdateActive<&PhysicsJoint::ConnectedNode>(e)); }
        );
    }

    if (const auto *trigger_nodes = r.try_get<const TriggerNodes>(entity)) {
        Spacing();
        SeparatorText("Trigger (compound)");
        PushID("Trigger");
        Text("Compound trigger: %zu nodes", trigger_nodes->Nodes.size());
        TargetCombo<CollisionFilter>(r, "Collision filter", trigger_nodes->CollisionFilterEntity, "No filters defined", [](state::Entity e) {
            action::Emit(action::UpdateActive<&TriggerNodes::CollisionFilterEntity>(e));
        });
        if (Button("Remove Trigger")) action::Emit(action::physics::RemoveTriggerNodes{});
        PopID();
    } else {
        Spacing();
        const bool is_shape_trigger = collider && r.all_of<const TriggerTag>(entity);
        if (collider) {
            if (is_shape_trigger) {
                if (Button("Convert to Collider")) action::Emit(action::physics::SetTrigger{false});
            } else {
                if (Button("Convert to Trigger")) action::Emit(action::physics::SetTrigger{true});
            }
        } else if (Button("Add Trigger")) action::Emit(action::physics::AddTrigger{});
    }

    PopID();
}
