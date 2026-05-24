// Physics UI: "Physics" tab + per-entity physics properties.

#include "PhysicsUi.h"
#include "AnimationTimeline.h"
#include "Emit.h"
#include "PhysicsWorld.h"
#include "SceneGraph.h"
#include "action/Core.h"
#include "action/Physics.h"
#include "numeric/vec2.h"
#include "ui/FieldEdit.h"

#include <entt/entity/registry.hpp>
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

template<class T>
void RenderNameEdit(entt::entity e, const std::string &current) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s", current.c_str());
    if (InputText("Name", buf, sizeof(buf))) action::Emit(action::physics::SetNameOf<T>(e, std::string{buf}));
}

// Deduce owner class from a data-member pointer (entt::entity Owner::*).
template<class M> struct ptr_class;
template<class C, class V> struct ptr_class<V C::*> {
    using type = C;
};

// Renders a combo that selects a referenced resource entity (Field) from the registry's view<Target>.
// Returns the currently-selected Target (or nullptr if none/invalid). Emits when the user picks a new entry.
// When `empty_preview` is non-null and the registry has no Target entities, renders a disabled combo with that preview.
// `entity` is the active entity (used for the read; the write targets active via the no-entity UpdateOf).
template<class Target, auto Field>
const Target *RenderEntityCombo(entt::registry &r, entt::entity entity, const char *label, const char *empty_preview = nullptr) {
    using Owner = typename ptr_class<decltype(Field)>::type;
    const auto view = r.view<const Target>();
    if (empty_preview && view.begin() == view.end()) {
        BeginDisabled();
        if (BeginCombo(label, empty_preview)) EndCombo();
        EndDisabled();
        return nullptr;
    }
    const auto cur_e = r.get<const Owner>(entity).*Field;
    const auto *cur = cur_e != null_entity && r.valid(cur_e) ? r.try_get<const Target>(cur_e) : nullptr;
    if (const auto preview = cur ? DisplayName(cur->Name, "{:x}", uint32_t(cur_e)) : std::string{"None"};
        BeginCombo(label, preview.c_str())) {
        if (Selectable("None", cur_e == null_entity)) action::Emit(action::UpdateOf<Field>(entt::entity{null_entity}));
        for (auto [te, t] : view.each()) {
            if (Selectable(DisplayName(t.Name, "{:x}", uint32_t(te)).c_str(), cur_e == te)) action::Emit(action::UpdateOf<Field>(te));
        }
        EndCombo();
    }
    return cur;
}

std::string SystemDisplayName(const entt::registry &r, entt::entity e) {
    return !r.all_of<const CollisionSystem>(e) ? "<invalid>" : DisplayName(r.get<const CollisionSystem>(e).Name, "{:x}", uint32_t(e));
}

template<class T>
void ToggleInVector(std::vector<T> &v, T e, bool add) {
    if (add) {
        if (std::find(v.begin(), v.end(), e) == v.end()) v.push_back(e);
    } else {
        std::erase(v, e);
    }
}

// Multi-select combo over all CollisionSystem entities. `on_toggle(system, now_member)` returns
// the action for that change. Renders a disabled combo with "No systems defined" when none exist.
template<class Fn>
void RenderSystemMultiSelect(const entt::registry &r, const char *label, const std::vector<entt::entity> &selection, Fn on_toggle) {
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

size_t CountFilterUses(const entt::registry &r, entt::entity filter) {
    size_t n = 0;
    for (auto [e, m] : r.view<const ColliderMaterial>().each()) {
        if (m.CollisionFilterEntity == filter) ++n;
    }
    for (auto [e, t] : r.view<const TriggerNodes>().each()) {
        if (t.CollisionFilterEntity == filter) ++n;
    }
    return n;
}

// Editor for a collision filter's body: membership, Mode, CollideSystems. Sole editing surface —
// per-entity panels only reference filters by combo; property edits happen here in the Physics tab.
void RenderCollisionFilterBody(entt::registry &r, entt::entity filter_e) {
    const auto &filter = r.get<const CollisionFilter>(filter_e);
    RenderSystemMultiSelect(r, "Member of", filter.Systems, [&](entt::entity se, bool add) {
        return action::physics::ToggleFilterEntity{filter_e, &CollisionFilter::Systems, se, add};
    });

    int mode = int(filter.Mode);
    TextUnformatted("Collide with:");
    SameLine();
    bool mode_changed = RadioButton("All", &mode, int(CollideMode::All));
    SameLine();
    mode_changed |= RadioButton("Allowlist", &mode, int(CollideMode::Allowlist));
    SameLine();
    mode_changed |= RadioButton("Blocklist", &mode, int(CollideMode::Blocklist));
    if (mode_changed) action::Emit(action::UpdateOf<&CollisionFilter::Mode>(filter_e, CollideMode(mode)));

    if (mode != int(CollideMode::All)) {
        Indent();
        RenderSystemMultiSelect(r, "##collide", filter.CollideSystems, [&](entt::entity se, bool add) {
            return action::physics::ToggleFilterEntity{filter_e, &CollisionFilter::CollideSystems, se, add};
        });
        if (filter.CollideSystems.empty() && mode == int(CollideMode::Allowlist)) {
            TextColored(ImVec4{0.9f, 0.7f, 0.3f, 1}, "No systems selected — this filter collides with nothing.");
        }
        Unindent();
    }
}

// Split a cell along the '/' diagonal (TR→BL). Upper-left triangle shows row→col direction;
// lower-right triangle shows col→row. Blocked directions are unpainted; allowed directions paint
// green when both directions agree (effective collide) and red when the other direction vetoes
// (effective blocked — the allow is overridden).
void DrawMatrixCell(ImDrawList *dl, ImVec2 p_min, ImVec2 p_max, bool a_to_b, bool b_to_a) {
    const auto collide = IM_COL32(60, 200, 60, 220);
    const auto overridden = IM_COL32(200, 60, 60, 220);
    const auto fill = (a_to_b && b_to_a) ? collide : overridden;
    const auto TL = p_min, BR = p_max;
    const ImVec2 TR{p_max.x, p_min.y}, BL{p_min.x, p_max.y};
    if (a_to_b) dl->AddTriangleFilled(TL, TR, BL, fill); // row→col
    if (b_to_a) dl->AddTriangleFilled(TR, BR, BL, fill); // col→row
}

// List view of named entities with use-count, Delete/Add buttons, and per-entry body.
// `body(e, x)` renders the row's editable body and emits any actions.
template<class T>
void DrawNamedEntityList(entt::registry &r, const char *id, const char *add_label, std::string_view prefix, auto &&count, auto &&body) {
    PushID(id);
    entt::entity delete_entity = entt::null;
    for (auto [e, x] : r.view<T>().each()) {
        PushID(uint32_t(e));
        const auto label = DisplayName(x.Name, "{:x}", uint32_t(e));
        const bool expanded = TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanTextWidth, "%s", label.c_str());
        SameLine();
        TextDisabled("(%zu)", count(e));
        SameLine();
        if (SmallButton("X")) delete_entity = e;
        if (expanded) {
            RenderNameEdit<T>(e, x.Name);
            body(e, x);
            TreePop();
        }
        PopID();
    }
    if (delete_entity != entt::null) action::Emit(action::DestroyEntity{delete_entity});
    if (Button(add_label)) action::Emit(action::physics::CreateNamedOf<T>(prefix));
    PopID();
}

// Returns the edited shape iff the user changed something.
// When `auto_fit`, dim widgets for BBox-fittable kinds are hidden — the fitter owns them.
std::optional<PhysicsShape> RenderShapeEditor(const PhysicsShape &in, bool auto_fit) {
    static const char *shape_names[]{"Box", "Sphere", "Capsule", "Cylinder", "Plane", "Convex Hull", "Triangle Mesh"};
    auto out = in; // TODO avoid copy
    bool changed = false;
    if (auto shape_type_i = int(out.index());
        Combo("Shape", &shape_type_i, shape_names, IM_ARRAYSIZE(shape_names))) {
        out = CreateVariantByIndex<PhysicsShape>(size_t(shape_type_i));
        changed = true;
    }
    std::visit(
        overloaded{
            [&](physics::Box &s) {
                if (!auto_fit) changed |= DragFloat3("Size", &s.Size.x, 0.01f, 0.01f, 100.f);
            },
            [&](physics::Sphere &s) {
                if (!auto_fit) changed |= DragFloat("Radius", &s.Radius, 0.01f, 0.001f, 100.f);
            },
            [&](physics::Capsule &s) {
                if (auto_fit) return;
                changed |= DragFloat("Height", &s.Height, 0.01f, 0.001f, 100.f);
                changed |= DragFloat("Radius top", &s.RadiusTop, 0.01f, 0.001f, 100.f);
                changed |= DragFloat("Radius bottom", &s.RadiusBottom, 0.01f, 0.001f, 100.f);
            },
            [&](physics::Cylinder &s) {
                if (auto_fit) return;
                changed |= DragFloat("Height", &s.Height, 0.01f, 0.001f, 100.f);
                changed |= DragFloat("Radius top", &s.RadiusTop, 0.01f, 0.001f, 100.f);
                changed |= DragFloat("Radius bottom", &s.RadiusBottom, 0.01f, 0.001f, 100.f);
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
                    if (DragFloat2("Size (X, Z)", &size.x, 0.01f, 0.01f, 1000.f)) {
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

void physics_ui::RenderTab(entt::registry &r, entt::entity viewport, PhysicsWorld &physics) {
    SeparatorText("Simulation");
    Text("Bodies: %u", physics.BodyCount());
    {
        ui::Edit f{r, viewport};
        f.Slider<&PhysicsSimulationSettings::SubstepsPerFrame>("Substeps per frame", 1u, 100u);
        f.Slider<&PhysicsSimulationSettings::SolverIterations>("Solver iterations", 2u, 50u);
        f.Slider<&PhysicsSimulationSettings::TimeScale>("Time scale", 0.f, 10.f, "%.2fx");
        f.Drag<&PhysicsSimulationSettings::Gravity>("Gravity", 0.1f);
    }

    if (CollapsingHeader("Physics Materials")) {
        DrawNamedEntityList<PhysicsMaterial>(
            r, "PhysMaterials", "Add material", "Material",
            [&](entt::entity mat_entity) {
                size_t n = 0;
                for (auto [e, m] : r.view<const ColliderMaterial>().each()) {
                    if (m.PhysicsMaterialEntity == mat_entity) ++n;
                }
                return n;
            },
            [&](entt::entity mat_entity, const PhysicsMaterial &) {
                ui::Edit f{r, mat_entity};
                f.Slider<&PhysicsMaterial::StaticFriction>("Static friction", 0.f, 2.f);
                f.Slider<&PhysicsMaterial::DynamicFriction>("Dynamic friction", 0.f, 2.f);
                f.Slider<&PhysicsMaterial::Restitution>("Restitution", 0.f, 1.f);
                f.Enum<&PhysicsMaterial::FrictionCombine>("Friction combine", "Average\0Minimum\0Maximum\0Multiply\0");
                f.Enum<&PhysicsMaterial::RestitutionCombine>("Restitution combine", "Average\0Minimum\0Maximum\0Multiply\0");
            }
        );
    }

    if (CollapsingHeader("Collision Systems")) {
        DrawNamedEntityList<CollisionSystem>(
            r, "CollisionSystems", "Add system", "System",
            [&](entt::entity se) {
                size_t n = 0;
                for (auto [fe, f] : r.view<const CollisionFilter>().each()) {
                    if (std::find(f.Systems.begin(), f.Systems.end(), se) != f.Systems.end() ||
                        std::find(f.CollideSystems.begin(), f.CollideSystems.end(), se) != f.CollideSystems.end()) ++n;
                }
                return n;
            },
            [](entt::entity, const CollisionSystem &) {}
        );
    }

    if (CollapsingHeader("Collision Filters")) {
        DrawNamedEntityList<CollisionFilter>(
            r, "CollisionFilters", "Add filter", "Filter",
            [&](entt::entity fe) { return CountFilterUses(r, fe); },
            [&](entt::entity fe, const CollisionFilter &) { RenderCollisionFilterBody(r, fe); }
        );
        PushID("CollisionFilters");

        // Collision matrix
        std::vector<entt::entity> filter_entities;
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
                        const bool ab = physics.DoesFilterAllow(a, b);
                        const bool ba = physics.DoesFilterAllow(b, a);
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
            r, "JointDefs", "Add joint definition", "Joint",
            [&](entt::entity jd_entity) {
                size_t n = 0;
                for (auto [e, j] : r.view<const PhysicsJoint>().each()) {
                    if (j.JointDefEntity == jd_entity) ++n;
                }
                return n;
            },
            [&](entt::entity jd_entity, const PhysicsJointDef &jd) {
                static const char *axis_names[]{"X", "Y", "Z"};
                std::optional<uint32_t> delete_limit;
                for (uint32_t li = 0; li < jd.Limits.size(); ++li) {
                    PushID(int(li));
                    const auto &limit = jd.Limits[li];
                    const bool limit_expanded = TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanTextWidth, "Limit %u", li);
                    SameLine();
                    if (SmallButton("X")) delete_limit = li;
                    if (limit_expanded) {
                        const auto edit_limit = [&](auto &&fn) {
                            auto edit = limit;
                            fn(edit);
                            return action::physics::SetJointVecItem<PhysicsJointLimit>{jd_entity, &PhysicsJointDef::Limits, li, std::make_unique<PhysicsJointLimit>(std::move(edit))};
                        };
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
                            if (DragFloat("##min", &min_val, 0.01f)) action::Emit(edit_limit([&](auto &e) { e.Min = min_val; }));
                        }
                        if (Checkbox("Max", &has_max)) action::Emit(edit_limit([&](auto &e) { e.Max = has_max ? std::optional{max_val} : std::nullopt; }));
                        if (has_max) {
                            SameLine();
                            if (DragFloat("##max", &max_val, 0.01f)) action::Emit(edit_limit([&](auto &e) { e.Max = max_val; }));
                        }

                        bool soft = limit.Stiffness.has_value();
                        if (Checkbox("Soft limit", &soft)) action::Emit(edit_limit([&](auto &e) { e.Stiffness = soft ? std::optional{1000.0f} : std::nullopt; }));
                        if (limit.Stiffness) {
                            float stiffness = *limit.Stiffness, damping = limit.Damping;
                            if (DragFloat("Stiffness", &stiffness, 1.0f, 0.0f, 1e6f)) action::Emit(edit_limit([&](auto &e) { e.Stiffness = stiffness; }));
                            if (DragFloat("Damping", &damping, 0.1f, 0.0f, 1e4f)) action::Emit(edit_limit([&](auto &e) { e.Damping = damping; }));
                        }
                        TreePop();
                    }
                    PopID();
                }
                if (delete_limit) action::Emit(action::physics::DeleteJointVecItem<PhysicsJointLimit>{jd_entity, &PhysicsJointDef::Limits, *delete_limit});
                if (Button("Add limit")) action::Emit(action::physics::AddJointVecItem<PhysicsJointLimit>{jd_entity, &PhysicsJointDef::Limits});

                Spacing();

                std::optional<uint32_t> delete_drive;
                for (uint32_t di = 0; di < jd.Drives.size(); ++di) {
                    PushID(int(di + 1000));
                    const auto &drive = jd.Drives[di];
                    const bool drive_expanded = TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanTextWidth, "Drive %u", di);
                    SameLine();
                    if (SmallButton("X")) delete_drive = di;
                    if (drive_expanded) {
                        const auto edit_drive = [&](auto &&fn) {
                            auto edit = drive;
                            fn(edit);
                            return action::physics::SetJointVecItem<PhysicsJointDrive>{jd_entity, &PhysicsJointDef::Drives, di, std::make_unique<PhysicsJointDrive>(std::move(edit))};
                        };
                        if (int type = int(drive.Type); Combo("Type", &type, "Linear\0Angular\0")) action::Emit(edit_drive([&](auto &e) { e.Type = PhysicsDriveType(type); }));
                        if (int axis = drive.Axis; Combo("Axis", &axis, "X\0Y\0Z\0")) action::Emit(edit_drive([&](auto &e) { e.Axis = uint8_t(axis); }));
                        if (int mode = int(drive.Mode); Combo("Mode", &mode, "Force\0Acceleration\0")) action::Emit(edit_drive([&](auto &e) { e.Mode = PhysicsDriveMode(mode); }));
                        float max_force = drive.MaxForce, pos_target = drive.PositionTarget, vel_target = drive.VelocityTarget;
                        float stiffness = drive.Stiffness, damping = drive.Damping;
                        if (DragFloat("Max force", &max_force, 1.0f, 0.0f, 1e6f)) action::Emit(edit_drive([&](auto &e) { e.MaxForce = max_force; }));
                        if (DragFloat("Position target", &pos_target, 0.01f)) action::Emit(edit_drive([&](auto &e) { e.PositionTarget = pos_target; }));
                        if (DragFloat("Velocity target", &vel_target, 0.01f)) action::Emit(edit_drive([&](auto &e) { e.VelocityTarget = vel_target; }));
                        if (DragFloat("Stiffness", &stiffness, 1.0f, 0.0f, 1e6f)) action::Emit(edit_drive([&](auto &e) { e.Stiffness = stiffness; }));
                        if (DragFloat("Damping", &damping, 0.1f, 0.0f, 1e4f)) action::Emit(edit_drive([&](auto &e) { e.Damping = damping; }));
                        TreePop();
                    }
                    PopID();
                }
                if (delete_drive) action::Emit(action::physics::DeleteJointVecItem<PhysicsJointDrive>{jd_entity, &PhysicsJointDef::Drives, *delete_drive});
                if (Button("Add drive")) action::Emit(action::physics::AddJointVecItem<PhysicsJointDrive>{jd_entity, &PhysicsJointDef::Drives});
            }
        );
    }
}

void physics_ui::RenderEntityProperties(entt::registry &r, entt::entity entity, entt::entity viewport, const PhysicsWorld &physics) {
    if (!CollapsingHeader("Physics")) return;

    PushID("PhysicsEntity");

    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    const auto *collider = r.try_get<const ColliderShape>(entity);

    // 4-way motion type matching Jolt's EMotionType + a "none" state.
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

    if (changed) action::Emit(action::physics::SetMotionType{MType(motion_type)});

    if (collider) { // Collider shape editing
        Spacing();
        SeparatorText("Collider");

        ui::Edit{r}.Check<&ColliderPolicy::AutoFitDims>("Auto-fit");
        if (auto s = RenderShapeEditor(collider->Shape, r.get<const ColliderPolicy>(entity).AutoFitDims))
            action::Emit(action::physics::SetColliderShape{*s, s->index() != collider->Shape.index()});

        RenderEntityCombo<PhysicsMaterial, &ColliderMaterial::PhysicsMaterialEntity>(r, entity, "Physics material", "No materials defined");
        RenderEntityCombo<CollisionFilter, &ColliderMaterial::CollisionFilterEntity>(r, entity, "Collision filter", "No filters defined");
    }

    // Motion properties editing
    if (motion) {
        Spacing();
        SeparatorText("Motion");

        // Velocity is an authored initial condition (KHR_physics_rigid_bodies). Locked once
        // sim has produced any baked frames; JumpToStart unlocks it.
        const auto &range = r.get<const TimelineRange>(viewport);
        const auto &playback = r.get<const TimelinePlayback>(viewport);
        const bool velocity_locked = playback.Playing || physics.BakedThrough() >= range.StartFrame;
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
            PhysicsMotion edit = *motion;
            bool motion_changed = DragFloat("Gravity factor", &edit.GravityFactor, 0.01f, -10.f, 10.f);

            Spacing();
            SeparatorText("Mass properties");

            float mass = edit.Mass.value_or(DefaultMass);
            if (DragFloat("Mass", &mass, 0.1f, 0.001f, 1e6f, "%.3f kg")) {
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
                motion_changed |= DragFloat3("Inertia diagonal", &edit.InertiaDiagonal->x, 0.01f, 0.001f, 1e6f);
                vec3 euler_deg = glm::degrees(glm::eulerAngles(edit.InertiaOrientation.value_or(quat{1, 0, 0, 0})));
                if (DragFloat3("Inertia orientation", &euler_deg.x, 0.1f)) {
                    edit.InertiaOrientation = quat{glm::radians(euler_deg)};
                    motion_changed = true;
                }
            }

            if (bool has_com = edit.CenterOfMass.has_value(); Checkbox("Override center of mass", &has_com)) {
                edit.CenterOfMass = has_com ? std::optional{vec3{0.0f}} : std::nullopt;
                motion_changed = true;
            }
            if (edit.CenterOfMass) motion_changed |= DragFloat3("Center of mass", &edit.CenterOfMass->x, 0.01f);

            Spacing();
            SeparatorText("Dynamics");

            motion_changed |= DragFloat("Damping translation", &edit.LinearDamping, 0.01f, 0.f, 1.f);
            motion_changed |= DragFloat("Damping rotation", &edit.AngularDamping, 0.01f, 0.f, 1.f);
            if (motion_changed) action::Emit(action::ReplaceActive<PhysicsMotion>{std::make_unique<PhysicsMotion>(edit)});
        }
    }

    // Joint properties
    if (const auto *joint = r.try_get<const PhysicsJoint>(entity)) {
        Spacing();
        SeparatorText("Joint");

        const auto *cur = RenderEntityCombo<PhysicsJointDef, &PhysicsJoint::JointDefEntity>(r, entity, "Definition", "No joint definitions");
        if (cur) Text("Limits: %zu, Drives: %zu", cur->Limits.size(), cur->Drives.size());

        ui::Edit{r}.Check<&PhysicsJoint::EnableCollision>("Enable collision");

        // ConnectedNode picker — KHR joint.connectedNode is the second attachment frame.
        // Mirrors Blender's rigid_body_constraint object1/object2 fields.
        const auto cn = joint->ConnectedNode;
        if (const auto cn_label = cn != null_entity && r.valid(cn) ? GetName(r, cn) : std::string{"None"};
            BeginCombo("Connected node", cn_label.c_str())) {
            if (Selectable("None", cn == null_entity)) action::Emit(action::UpdateOf<&PhysicsJoint::ConnectedNode>(entt::entity{null_entity}));
            for (auto ne : r.view<const SceneNode>()) {
                if (ne == entity) continue; // self-connection is meaningless
                if (Selectable(GetName(r, ne).c_str(), cn == ne)) action::Emit(action::UpdateOf<&PhysicsJoint::ConnectedNode>(ne));
            }
            EndCombo();
        }
    }

    if (const auto *trigger_nodes = r.try_get<const TriggerNodes>(entity)) {
        Spacing();
        SeparatorText("Trigger (compound)");
        PushID("Trigger");
        Text("Compound trigger: %zu nodes", trigger_nodes->Nodes.size());
        RenderEntityCombo<CollisionFilter, &TriggerNodes::CollisionFilterEntity>(r, entity, "Collision filter", "No filters defined");
        if (Button("Remove Trigger")) action::Emit(action::physics::RemoveTriggerNodes{});
        PopID();
    } else {
        Spacing();
        const bool is_shape_trigger = collider && r.all_of<const TriggerTag>(entity);
        if (collider) {
            if (is_shape_trigger) {
                if (Button("Convert to Collider")) action::Emit(action::SetTagOf<TriggerTag>(false));
            } else {
                if (Button("Convert to Trigger")) action::Emit(action::SetTagOf<TriggerTag>(true));
            }
        } else if (Button("Add Trigger")) action::Emit(action::physics::AddTrigger{});
    }

    PopID();
}
