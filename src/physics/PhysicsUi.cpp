// Physics UI: "Physics" tab + per-entity physics properties.
// Free functions — no Scene.h dependency.

#include "PhysicsUi.h"
#include "AnimationTimeline.h"
#include "Instance.h"
#include "PhysicsWorld.h"
#include "mesh/Mesh.h"
#include "mesh/PrimitiveType.h"

#include <entt/entity/registry.hpp>
#include <format>
#include <glm/gtc/quaternion.hpp>
#include <glm/trigonometric.hpp>
#include <imgui.h>

using namespace ImGui;

namespace {
// Find the mesh entity for an instance or mesh entity.
entt::entity FindMeshEntity(const entt::registry &r, entt::entity entity) {
    if (const auto *instance = r.try_get<const Instance>(entity)) return instance->Entity;
    return entity;
}

// Initialize a PhysicsShape from the PrimitiveShape component if present,
// otherwise fall back to AABB-derived box dimensions.
PhysicsShape InitColliderShape(const entt::registry &r, entt::entity entity) {
    const auto mesh_entity = FindMeshEntity(r, entity);
    if (const auto *prim = r.try_get<const PrimitiveShape>(mesh_entity)) {
        auto shape = std::visit([](const auto &s) -> PhysicsShape {
            using T = std::decay_t<decltype(s)>;
            PhysicsShape shape;
            if constexpr (std::is_same_v<T, primitive::Cuboid>) {
                shape.Type = PhysicsShapeType::Box;
                shape.Size = s.HalfExtents * 2.f;
            } else if constexpr (std::is_same_v<T, primitive::IcoSphere> || std::is_same_v<T, primitive::UVSphere>) {
                shape.Type = PhysicsShapeType::Sphere;
                shape.Radius = s.Radius;
            } else if constexpr (std::is_same_v<T, primitive::Cylinder>) {
                shape.Type = PhysicsShapeType::Cylinder;
                shape.RadiusTop = shape.RadiusBottom = s.Radius;
                shape.Height = s.Height;
            } else if constexpr (std::is_same_v<T, primitive::Cone>) {
                shape.Type = PhysicsShapeType::Cylinder;
                shape.RadiusTop = 0;
                shape.RadiusBottom = s.Radius;
                shape.Height = s.Height;
            } else {
                // Rect, Circle, Torus → ConvexHull from mesh geometry
                shape.Type = PhysicsShapeType::ConvexHull;
            }
            return shape;
        },
                                *prim);
        if (shape.Type == PhysicsShapeType::ConvexHull) shape.MeshEntity = mesh_entity;
        return shape;
    }

    // Fallback: derive shape from mesh AABB.
    const auto *mesh = r.try_get<const Mesh>(mesh_entity);
    if (!mesh || mesh->VertexCount() == 0) return {};

    const auto verts = mesh->GetVerticesSpan();
    vec3 lo = verts[0].Position, hi = lo;
    for (const auto &v : verts) {
        lo = glm::min(lo, v.Position);
        hi = glm::max(hi, v.Position);
    }
    const vec3 extents = hi - lo;
    PhysicsShape shape;
    // If any AABB dimension is degenerate, use ConvexHull instead of a zero-thickness box.
    if (extents.x < 1e-6f || extents.y < 1e-6f || extents.z < 1e-6f) {
        shape.Type = PhysicsShapeType::ConvexHull;
        shape.MeshEntity = mesh_entity;
    } else {
        shape.Size = extents;
    }
    return shape;
}

bool DeleteButton(bool in_use) {
    SameLine();
    if (in_use) {
        BeginDisabled();
        Button("X");
        EndDisabled();
        return false;
    }
    return Button("X");
}

template<class T>
void RenderNameEdit(entt::registry &r, entt::entity e, const std::string &current) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s", current.c_str());
    if (InputText("Name", buf, sizeof(buf))) r.patch<T>(e, [&](auto &x) { x.Name = buf; });
}

// Deduce owner class from a data-member pointer (entt::entity Owner::*).
template<class M> struct ptr_class;
template<class C, class V> struct ptr_class<V C::*> {
    using type = C;
};

// Renders a combo that selects a referenced resource entity (Field) from the registry's view<Target>.
// Returns the currently-selected Target (or nullptr if none/invalid).
template<class Target, auto Field>
const Target *RenderEntityCombo(entt::registry &r, entt::entity entity, const char *label, const char *prefix) {
    using Owner = typename ptr_class<decltype(Field)>::type;
    const auto cur_e = r.get<const Owner>(entity).*Field;
    const auto *cur = cur_e != null_entity && r.valid(cur_e) ? r.try_get<const Target>(cur_e) : nullptr;
    const auto preview = cur ? (cur->Name.empty() ? std::format("{} {:x}", prefix, uint32_t(cur_e)) : cur->Name) : std::string{"None"};
    if (BeginCombo(label, preview.c_str())) {
        if (Selectable("None", cur_e == null_entity)) {
            r.patch<Owner>(entity, [](Owner &o) { o.*Field = null_entity; });
        }
        size_t i = 0;
        for (auto [te, t] : r.view<const Target>().each()) {
            const auto item_label = t.Name.empty() ? std::format("{} {}", prefix, i) : t.Name;
            if (Selectable(item_label.c_str(), cur_e == te)) {
                r.patch<Owner>(entity, [te](Owner &o) { o.*Field = te; });
            }
            ++i;
        }
        EndCombo();
    }
    return cur;
}

bool RenderShapeEditor(PhysicsShape &shape) {
    static const char *shape_names[]{"Box", "Sphere", "Capsule", "Cylinder", "Convex Hull", "Triangle Mesh"};
    auto shape_type_i = int(shape.Type);
    bool changed = false;
    if (Combo("Shape", &shape_type_i, shape_names, IM_ARRAYSIZE(shape_names))) {
        shape.Type = PhysicsShapeType(shape_type_i);
        changed = true;
    }
    switch (shape.Type) {
        case PhysicsShapeType::Box:
            changed |= DragFloat3("Size", &shape.Size.x, 0.01f, 0.01f, 100.f);
            break;
        case PhysicsShapeType::Sphere:
            changed |= DragFloat("Radius", &shape.Radius, 0.01f, 0.001f, 100.f);
            break;
        case PhysicsShapeType::Capsule:
            changed |= DragFloat("Height", &shape.Height, 0.01f, 0.001f, 100.f);
            changed |= DragFloat("Radius top", &shape.RadiusTop, 0.01f, 0.001f, 100.f);
            changed |= DragFloat("Radius bottom", &shape.RadiusBottom, 0.01f, 0.001f, 100.f);
            break;
        case PhysicsShapeType::Cylinder:
            changed |= DragFloat("Height", &shape.Height, 0.01f, 0.001f, 100.f);
            changed |= DragFloat("Radius top", &shape.RadiusTop, 0.01f, 0.001f, 100.f);
            changed |= DragFloat("Radius bottom", &shape.RadiusBottom, 0.01f, 0.001f, 100.f);
            break;
        case PhysicsShapeType::ConvexHull:
        case PhysicsShapeType::TriangleMesh:
            TextDisabled("Mesh-based shape (from glTF import)");
            break;
    }
    return changed;
}
} // namespace

void physics_ui::RenderTab(entt::registry &r, PhysicsWorld &physics) {
    SeparatorText("Simulation");
    Text("Bodies: %u", physics.BodyCount());
    SliderInt("Substeps", &physics.SubSteps, 1, 8);
    DragFloat3("Gravity", &physics.Gravity.x, 0.1f);
    {
        float timestep_ms = physics.TimeStep * 1000.f;
        if (DragFloat("Time step (ms)", &timestep_ms, 0.1f, 0.1f, 100.f, "%.1f")) {
            physics.TimeStep = std::max(timestep_ms * 0.001f, 1e-4f);
        }
    }

    if (CollapsingHeader("Physics Materials")) {
        PushID("PhysMaterials");
        entt::entity delete_entity = entt::null;
        size_t mat_index = 0;
        for (auto [mat_entity, mat] : r.view<PhysicsMaterial>().each()) {
            PushID(uint32_t(mat_entity));
            const auto label = mat.Name.empty() ? std::format("Material {}", mat_index) : mat.Name;
            const bool expanded = TreeNode(label.c_str());
            bool mat_in_use = false;
            for (auto [e, m] : r.view<const ColliderMaterial>().each()) {
                if (m.PhysicsMaterialEntity == mat_entity) {
                    mat_in_use = true;
                    break;
                }
            }
            if (DeleteButton(mat_in_use)) delete_entity = mat_entity;
            if (expanded) {
                RenderNameEdit<PhysicsMaterial>(r, mat_entity, mat.Name);

                float sf = mat.StaticFriction, df = mat.DynamicFriction, rest = mat.Restitution;
                if (SliderFloat("Static friction", &sf, 0.0f, 2.0f)) r.patch<PhysicsMaterial>(mat_entity, [&](auto &m) { m.StaticFriction = sf; });
                if (SliderFloat("Dynamic friction", &df, 0.0f, 2.0f)) r.patch<PhysicsMaterial>(mat_entity, [&](auto &m) { m.DynamicFriction = df; });
                if (SliderFloat("Restitution", &rest, 0.0f, 1.0f)) r.patch<PhysicsMaterial>(mat_entity, [&](auto &m) { m.Restitution = rest; });

                auto fc = int(mat.FrictionCombine);
                if (Combo("Friction combine", &fc, "Average\0Minimum\0Maximum\0Multiply\0")) {
                    r.patch<PhysicsMaterial>(mat_entity, [&](auto &m) { m.FrictionCombine = PhysicsCombineMode(fc); });
                }
                auto rc = int(mat.RestitutionCombine);
                if (Combo("Restitution combine", &rc, "Average\0Minimum\0Maximum\0Multiply\0")) {
                    r.patch<PhysicsMaterial>(mat_entity, [&](auto &m) { m.RestitutionCombine = PhysicsCombineMode(rc); });
                }

                TreePop();
            }
            PopID();
            ++mat_index;
        }
        if (delete_entity != entt::null) r.destroy(delete_entity);
        if (Button("Add material")) {
            const auto e = r.create();
            r.emplace<PhysicsMaterial>(e, PhysicsMaterial{.Name = std::format("Material {}", mat_index)});
        }
        PopID();
    }

    if (CollapsingHeader("Collision Filters")) {
        PushID("CollisionFilters");

        // Snapshot filter entities so mutation/destroy during the loop doesn't invalidate iteration.
        std::vector<entt::entity> filter_entities;
        for (auto e : r.view<CollisionFilter>()) filter_entities.emplace_back(e);

        // Collect all unique system names across filters for checkboxes.
        std::vector<std::string> all_systems;
        auto collect = [&](const std::vector<std::string> &names) {
            for (const auto &s : names)
                if (std::find(all_systems.begin(), all_systems.end(), s) == all_systems.end()) all_systems.push_back(s);
        };
        for (auto fe : filter_entities) {
            const auto &f = r.get<const CollisionFilter>(fe);
            collect(f.CollisionSystems);
            collect(f.CollideWithSystems);
            collect(f.NotCollideWithSystems);
        }
        std::sort(all_systems.begin(), all_systems.end());

        entt::entity delete_entity = entt::null;
        size_t idx = 0;
        for (auto fe : filter_entities) {
            PushID(uint32_t(fe));
            const auto &filter = r.get<const CollisionFilter>(fe);
            const auto label = filter.Name.empty() ? std::format("Filter {}", idx) : filter.Name;
            const bool expanded = TreeNode(label.c_str());
            bool filter_in_use = false;
            for (auto [e, m] : r.view<const ColliderMaterial>().each()) {
                if (m.CollisionFilterEntity == fe) {
                    filter_in_use = true;
                    break;
                }
            }
            if (!filter_in_use) {
                for (auto [e, t] : r.view<const PhysicsTrigger>().each()) {
                    if (t.CollisionFilterEntity == fe) {
                        filter_in_use = true;
                        break;
                    }
                }
            }
            if (DeleteButton(filter_in_use)) delete_entity = fe;
            if (expanded) {
                RenderNameEdit<CollisionFilter>(r, fe, filter.Name);

                // Collision systems (membership)
                if (TreeNode("Collision systems")) {
                    for (const auto &sys : all_systems) {
                        bool member = std::find(filter.CollisionSystems.begin(), filter.CollisionSystems.end(), sys) != filter.CollisionSystems.end();
                        if (Checkbox(sys.c_str(), &member)) {
                            r.patch<CollisionFilter>(fe, [&](auto &f) {
                                if (member) f.CollisionSystems.push_back(sys);
                                else std::erase(f.CollisionSystems, sys);
                            });
                        }
                    }
                    TreePop();
                }

                // Collide-with mode: All / Allowlist / Blocklist
                int mode = 0;
                if (!filter.CollideWithSystems.empty()) mode = 1;
                else if (!filter.NotCollideWithSystems.empty()) mode = 2;
                TextUnformatted("Collide with:");
                SameLine();
                if (RadioButton("All", &mode, 0)) {
                    r.patch<CollisionFilter>(fe, [](auto &f) {
                        f.CollideWithSystems.clear();
                        f.NotCollideWithSystems.clear();
                    });
                }
                SameLine();
                if (RadioButton("Allowlist", &mode, 1)) {
                    r.patch<CollisionFilter>(fe, [&](auto &f) {
                        f.NotCollideWithSystems.clear();
                        if (f.CollideWithSystems.empty()) f.CollideWithSystems = all_systems;
                    });
                }
                SameLine();
                if (RadioButton("Blocklist", &mode, 2)) {
                    r.patch<CollisionFilter>(fe, [](auto &f) { f.CollideWithSystems.clear(); });
                }

                if (mode == 1) {
                    Indent();
                    for (const auto &sys : all_systems) {
                        bool checked = std::find(filter.CollideWithSystems.begin(), filter.CollideWithSystems.end(), sys) != filter.CollideWithSystems.end();
                        if (Checkbox(sys.c_str(), &checked)) {
                            r.patch<CollisionFilter>(fe, [&](auto &f) {
                                if (checked) f.CollideWithSystems.push_back(sys);
                                else std::erase(f.CollideWithSystems, sys);
                            });
                        }
                    }
                    Unindent();
                } else if (mode == 2) {
                    Indent();
                    for (const auto &sys : all_systems) {
                        bool checked = std::find(filter.NotCollideWithSystems.begin(), filter.NotCollideWithSystems.end(), sys) != filter.NotCollideWithSystems.end();
                        if (Checkbox(sys.c_str(), &checked)) {
                            r.patch<CollisionFilter>(fe, [&](auto &f) {
                                if (checked) f.NotCollideWithSystems.push_back(sys);
                                else std::erase(f.NotCollideWithSystems, sys);
                            });
                        }
                    }
                    Unindent();
                }

                TreePop();
            }
            PopID();
            ++idx;
        }
        if (delete_entity != entt::null) r.destroy(delete_entity);

        // Add system name popup
        if (Button("Add system")) OpenPopup("AddSystem");
        if (BeginPopup("AddSystem")) {
            static char sys_buf[64] = "";
            InputText("System name", sys_buf, sizeof(sys_buf));
            if (Button("OK") && sys_buf[0] != '\0') {
                std::string name = sys_buf;
                if (std::find(all_systems.begin(), all_systems.end(), name) == all_systems.end()) {
                    for (auto fe : filter_entities) {
                        r.patch<CollisionFilter>(fe, [&](auto &f) { f.CollisionSystems.push_back(name); });
                    }
                }
                sys_buf[0] = '\0';
                CloseCurrentPopup();
            }
            EndPopup();
        }

        SameLine();
        if (Button("Add filter")) {
            const auto e = r.create();
            r.emplace<CollisionFilter>(e, CollisionFilter{.Name = std::format("Filter {}", filter_entities.size())});
        }

        // Collision matrix visualization (angled-header table)
        if (filter_entities.size() >= 2) {
            Spacing();
            SeparatorText("Collision Matrix");

            const auto n = filter_entities.size();
            if (BeginTable("##CollisionMatrix", int(n) + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit)) {
                TableSetupColumn("", ImGuiTableColumnFlags_NoHeaderLabel);
                for (size_t i = 0; i < n; ++i) {
                    const auto &f = r.get<const CollisionFilter>(filter_entities[i]);
                    auto col_label = f.Name.empty() ? std::format("{}", i) : f.Name;
                    TableSetupColumn(col_label.c_str(), ImGuiTableColumnFlags_AngledHeader);
                }
                TableAngledHeadersRow();

                for (size_t row = 0; row < n; ++row) {
                    TableNextRow();
                    TableSetColumnIndex(0);
                    const auto &fr = r.get<const CollisionFilter>(filter_entities[row]);
                    auto row_label = fr.Name.empty() ? std::format("{}", row) : fr.Name;
                    TextUnformatted(row_label.c_str());
                    for (size_t col = 0; col < n; ++col) {
                        TableSetColumnIndex(int(col) + 1);
                        const bool collides = physics.DoFiltersCollide(filter_entities[row], filter_entities[col]);
                        if (collides) TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Y");
                        else TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "-");
                    }
                }
                EndTable();
            }
        }

        PopID();
    }

    if (CollapsingHeader("Joint Definitions")) {
        PushID("JointDefs");
        size_t jd_index = 0;
        for (auto [jd_entity, jd] : r.view<PhysicsJointDef>().each()) {
            PushID(uint32_t(jd_entity));
            const auto label = jd.Name.empty() ? std::format("Joint {}", jd_index) : jd.Name;
            if (TreeNode(label.c_str())) {
                static const char *axis_names[]{"X", "Y", "Z"};
                // Limits
                for (size_t li = 0; li < jd.Limits.size(); ++li) {
                    PushID(int(li));
                    const auto &limit = jd.Limits[li];
                    if (TreeNode("Limit", "Limit %zu", li)) {
                        TextUnformatted("Linear axes:");
                        SameLine();
                        for (uint8_t a = 0; a < 3; ++a) {
                            bool active = std::find(limit.LinearAxes.begin(), limit.LinearAxes.end(), a) != limit.LinearAxes.end();
                            if (Checkbox(axis_names[a], &active)) {
                                r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) {
                                    if (active) d.Limits[li].LinearAxes.push_back(a);
                                    else std::erase(d.Limits[li].LinearAxes, a);
                                });
                            }
                            if (a < 2) SameLine();
                        }
                        TextUnformatted("Angular axes:");
                        SameLine();
                        for (uint8_t a = 0; a < 3; ++a) {
                            PushID(a + 3);
                            bool active = std::find(limit.AngularAxes.begin(), limit.AngularAxes.end(), a) != limit.AngularAxes.end();
                            if (Checkbox(axis_names[a], &active)) {
                                r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) {
                                    if (active) d.Limits[li].AngularAxes.push_back(a);
                                    else std::erase(d.Limits[li].AngularAxes, a);
                                });
                            }
                            if (a < 2) SameLine();
                            PopID();
                        }

                        bool has_min = limit.Min.has_value(), has_max = limit.Max.has_value();
                        float min_val = limit.Min.value_or(0.0f), max_val = limit.Max.value_or(0.0f);
                        if (Checkbox("Min", &has_min)) {
                            r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) {
                                d.Limits[li].Min = has_min ? std::optional{min_val} : std::nullopt;
                            });
                        }
                        if (has_min) {
                            SameLine();
                            if (DragFloat("##min", &min_val, 0.01f)) {
                                r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Limits[li].Min = min_val; });
                            }
                        }
                        if (Checkbox("Max", &has_max)) {
                            r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) {
                                d.Limits[li].Max = has_max ? std::optional{max_val} : std::nullopt;
                            });
                        }
                        if (has_max) {
                            SameLine();
                            if (DragFloat("##max", &max_val, 0.01f)) {
                                r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Limits[li].Max = max_val; });
                            }
                        }

                        bool soft = limit.Stiffness.has_value();
                        if (Checkbox("Soft limit", &soft)) {
                            r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) {
                                d.Limits[li].Stiffness = soft ? std::optional{1000.0f} : std::nullopt;
                            });
                        }
                        if (limit.Stiffness) {
                            float stiffness = *limit.Stiffness, damping = limit.Damping;
                            if (DragFloat("Stiffness", &stiffness, 1.0f, 0.0f, 1e6f)) {
                                r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Limits[li].Stiffness = stiffness; });
                            }
                            if (DragFloat("Damping", &damping, 0.1f, 0.0f, 1e4f)) {
                                r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Limits[li].Damping = damping; });
                            }
                        }
                        TreePop();
                    }
                    PopID();
                }
                if (Button("Add limit")) r.patch<PhysicsJointDef>(jd_entity, [](auto &d) { d.Limits.push_back({}); });

                Spacing();

                // Drives
                for (size_t di = 0; di < jd.Drives.size(); ++di) {
                    PushID(int(di + 1000));
                    const auto &drive = jd.Drives[di];
                    if (TreeNode("Drive", "Drive %zu", di)) {
                        int type = int(drive.Type);
                        if (Combo("Type", &type, "Linear\0Angular\0")) {
                            r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Drives[di].Type = PhysicsDriveType(type); });
                        }
                        int axis = drive.Axis;
                        if (Combo("Axis", &axis, "X\0Y\0Z\0")) {
                            r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Drives[di].Axis = uint8_t(axis); });
                        }
                        int mode = int(drive.Mode);
                        if (Combo("Mode", &mode, "Force\0Acceleration\0")) {
                            r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Drives[di].Mode = PhysicsDriveMode(mode); });
                        }
                        float max_force = drive.MaxForce, pos_target = drive.PositionTarget, vel_target = drive.VelocityTarget;
                        float stiffness = drive.Stiffness, damping = drive.Damping;
                        if (DragFloat("Max force", &max_force, 1.0f, 0.0f, 1e6f)) r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Drives[di].MaxForce = max_force; });
                        if (DragFloat("Position target", &pos_target, 0.01f)) r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Drives[di].PositionTarget = pos_target; });
                        if (DragFloat("Velocity target", &vel_target, 0.01f)) r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Drives[di].VelocityTarget = vel_target; });
                        if (DragFloat("Stiffness", &stiffness, 1.0f, 0.0f, 1e6f)) r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Drives[di].Stiffness = stiffness; });
                        if (DragFloat("Damping", &damping, 0.1f, 0.0f, 1e4f)) r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Drives[di].Damping = damping; });
                        TreePop();
                    }
                    PopID();
                }
                if (Button("Add drive")) r.patch<PhysicsJointDef>(jd_entity, [](auto &d) { d.Drives.push_back({}); });

                TreePop();
            }
            PopID();
            ++jd_index;
        }
        if (Button("Add joint definition")) {
            const auto e = r.create();
            r.emplace<PhysicsJointDef>(e, PhysicsJointDef{.Name = std::format("Joint {}", jd_index)});
        }
        PopID();
    }
}

void physics_ui::RenderEntityProperties(entt::registry &r, entt::entity entity, entt::entity scene_entity) {
    if (!CollapsingHeader("Physics")) return;

    PushID("PhysicsEntity");

    const auto *motion = r.try_get<const PhysicsMotion>(entity);
    const auto *collider = r.try_get<const ColliderShape>(entity);

    // Kinematic is an Infinite-mass sub-toggle inside Dynamic, not a separate motion type.
    enum : int { MT_None = 0,
                 MT_Static,
                 MT_Dynamic };
    int motion_type = MT_None;
    if (collider && !motion) motion_type = MT_Static;
    else if (motion) motion_type = MT_Dynamic;

    AlignTextToFramePadding();
    TextUnformatted("Motion type:");
    SameLine();
    bool changed = RadioButton("None", &motion_type, MT_None);
    SameLine();
    changed |= RadioButton("Static", &motion_type, MT_Static);
    SameLine();
    changed |= RadioButton("Dynamic", &motion_type, MT_Dynamic);

    if (changed) {
        if (motion_type < MT_Dynamic) r.remove<PhysicsMotion>(entity);
        if (motion_type < MT_Static) r.remove<ColliderShape>(entity);
        if (motion_type >= MT_Static && !r.all_of<ColliderShape>(entity)) {
            r.emplace<ColliderShape>(entity, ColliderShape{InitColliderShape(r, entity)});
        }
        if (motion_type >= MT_Dynamic && !r.all_of<PhysicsMotion>(entity)) {
            r.emplace<PhysicsMotion>(entity);
        }
        motion = r.try_get<const PhysicsMotion>(entity);
        collider = r.try_get<const ColliderShape>(entity);
    }

    if (collider) { // Collider shape editing
        Spacing();
        SeparatorText("Collider");

        PhysicsShape shape_edit = collider->Shape;
        if (RenderShapeEditor(shape_edit)) {
            r.patch<ColliderShape>(entity, [&](ColliderShape &cs) { cs.Shape = shape_edit; });
        }

        if (r.all_of<ColliderMaterial>(entity)) {
            RenderEntityCombo<PhysicsMaterial, &ColliderMaterial::PhysicsMaterialEntity>(r, entity, "Physics material", "Material");
            RenderEntityCombo<CollisionFilter, &ColliderMaterial::CollisionFilterEntity>(r, entity, "Collision filter", "Filter");
        }
    }

    // Motion properties editing
    if (motion) {
        Spacing();
        SeparatorText("Motion");

        const bool is_simulating = r.get<const AnimationTimeline>(scene_entity).Playing;
        if (is_simulating) BeginDisabled();
        if (auto *velocity = r.try_get<PhysicsVelocity>(entity)) {
            DragFloat3("Linear velocity", &velocity->Linear.x, 0.1f);
            DragFloat3("Angular velocity", &velocity->Angular.x, 0.1f);
        }
        if (is_simulating) EndDisabled();

        // Edit a local copy, then apply via patch<> so the reactive handler dispatches the update.
        PhysicsMotion edit = *motion;
        bool motion_changed = DragFloat("Gravity factor", &edit.GravityFactor, 0.01f, -10.f, 10.f);

        Spacing();
        SeparatorText("Mass properties");

        motion_changed |= Checkbox("Infinite mass", &edit.IsKinematic);

        if (edit.IsKinematic) BeginDisabled();

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

        if (edit.IsKinematic) EndDisabled();

        bool has_com = edit.CenterOfMass.has_value();
        if (Checkbox("Override center of mass", &has_com)) {
            edit.CenterOfMass = has_com ? std::optional{vec3{0.0f}} : std::nullopt;
            motion_changed = true;
        }
        if (edit.CenterOfMass) motion_changed |= DragFloat3("Center of mass", &edit.CenterOfMass->x, 0.01f);

        Spacing();
        SeparatorText("Dynamics");

        motion_changed |= DragFloat("Damping translation", &edit.LinearDamping, 0.01f, 0.f, 1.f);
        motion_changed |= DragFloat("Damping rotation", &edit.AngularDamping, 0.01f, 0.f, 1.f);

        if (motion_changed) r.patch<PhysicsMotion>(entity, [&](PhysicsMotion &m) { m = edit; });
    }

    // Joint properties
    if (const auto *joint = r.try_get<const PhysicsJoint>(entity)) {
        Spacing();
        SeparatorText("Joint");

        const auto *cur = RenderEntityCombo<PhysicsJointDef, &PhysicsJoint::JointDefEntity>(r, entity, "Definition", "Joint");
        if (cur) Text("Limits: %zu, Drives: %zu", cur->Limits.size(), cur->Drives.size());

        bool enable_collision = joint->EnableCollision;
        if (Checkbox("Enable collision", &enable_collision)) {
            r.patch<PhysicsJoint>(entity, [enable_collision](PhysicsJoint &j) { j.EnableCollision = enable_collision; });
        }

        if (joint->ConnectedNode != entt::null) Text("Connected: entity %u", joint->ConnectedNode);
        else TextDisabled("Not connected");
    }

    // Trigger properties
    if (const auto *trigger = r.try_get<const PhysicsTrigger>(entity); !trigger) {
        Spacing();
        if (Button("Add Trigger")) r.emplace<PhysicsTrigger>(entity, PhysicsTrigger{.Shape = PhysicsShape{}});
    } else {
        Spacing();
        SeparatorText("Trigger");
        PushID("Trigger");

        if (trigger->Shape.has_value()) {
            PhysicsShape shape_edit = *trigger->Shape;
            if (RenderShapeEditor(shape_edit)) {
                r.patch<PhysicsTrigger>(entity, [&](PhysicsTrigger &t) { t.Shape = shape_edit; });
            }
        } else if (!trigger->Nodes.empty()) {
            Text("Compound trigger: %zu nodes", trigger->Nodes.size());
        }

        RenderEntityCombo<CollisionFilter, &PhysicsTrigger::CollisionFilterEntity>(r, entity, "Collision filter", "Filter");

        if (Button("Remove Trigger")) r.remove<PhysicsTrigger>(entity);

        PopID();
    }

    PopID();
}
