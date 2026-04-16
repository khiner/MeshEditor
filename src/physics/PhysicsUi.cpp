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

void physics_ui::RenderTab(entt::registry &r, entt::entity scene_entity, PhysicsWorld &physics) {
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
        int delete_idx = -1;
        for (size_t i = 0; i < physics.Materials.size(); ++i) {
            PushID(i);
            auto &mat = physics.Materials[i];
            const auto label = mat.Name.empty() ? std::format("Material {}", i) : mat.Name;
            if (TreeNode(label.c_str())) {
                char buf[128];
                snprintf(buf, sizeof(buf), "%s", mat.Name.c_str());
                if (InputText("Name", buf, sizeof(buf))) mat.Name = buf;

                SliderFloat("Static friction", &mat.StaticFriction, 0.0f, 2.0f);
                SliderFloat("Dynamic friction", &mat.DynamicFriction, 0.0f, 2.0f);
                SliderFloat("Restitution", &mat.Restitution, 0.0f, 1.0f);

                auto fc = int(mat.FrictionCombine);
                if (Combo("Friction combine", &fc, "Average\0Minimum\0Maximum\0Multiply\0")) {
                    mat.FrictionCombine = PhysicsCombineMode(fc);
                }
                auto rc = int(mat.RestitutionCombine);
                if (Combo("Restitution combine", &rc, "Average\0Minimum\0Maximum\0Multiply\0")) {
                    mat.RestitutionCombine = PhysicsCombineMode(rc);
                }

                TreePop();
            }
            bool mat_in_use = false;
            for (auto [e, c] : r.view<const PhysicsCollider>().each()) {
                if (c.PhysicsMaterialIndex == uint32_t(i)) {
                    mat_in_use = true;
                    break;
                }
            }
            if (DeleteButton(mat_in_use)) delete_idx = int(i);
            PopID();
        }
        if (delete_idx >= 0) {
            physics.Materials.erase(physics.Materials.begin() + delete_idx);
            r.emplace_or_replace<PhysicsResourceDeleted>(scene_entity, PhysicsResourceDeleted::Material, uint32_t(delete_idx));
        }
        if (Button("Add material")) {
            physics.Materials.push_back({.Name = std::format("Material {}", physics.Materials.size())});
        }
        PopID();
    }

    if (CollapsingHeader("Collision Filters")) {
        PushID("CollisionFilters");
        bool filters_changed = false;

        // Collect all unique system names across filters for checkboxes.
        std::vector<std::string> all_systems;
        for (const auto &f : physics.Filters) {
            for (const auto &s : f.CollisionSystems)
                if (std::find(all_systems.begin(), all_systems.end(), s) == all_systems.end()) all_systems.push_back(s);
            for (const auto &s : f.CollideWithSystems)
                if (std::find(all_systems.begin(), all_systems.end(), s) == all_systems.end()) all_systems.push_back(s);
            for (const auto &s : f.NotCollideWithSystems)
                if (std::find(all_systems.begin(), all_systems.end(), s) == all_systems.end()) all_systems.push_back(s);
        }
        std::sort(all_systems.begin(), all_systems.end());

        int delete_filter_idx = -1;
        for (size_t i = 0; i < physics.Filters.size(); ++i) {
            PushID(int(i));
            auto &filter = physics.Filters[i];
            const auto label = filter.Name.empty() ? std::format("Filter {}", i) : filter.Name;
            if (TreeNode(label.c_str())) {
                char buf[128];
                snprintf(buf, sizeof(buf), "%s", filter.Name.c_str());
                if (InputText("Name", buf, sizeof(buf))) filter.Name = buf;

                // Collision systems (membership)
                if (TreeNode("Collision systems")) {
                    for (const auto &sys : all_systems) {
                        bool member = std::find(filter.CollisionSystems.begin(), filter.CollisionSystems.end(), sys) != filter.CollisionSystems.end();
                        if (Checkbox(sys.c_str(), &member)) {
                            if (member) filter.CollisionSystems.push_back(sys);
                            else std::erase(filter.CollisionSystems, sys);
                            filters_changed = true;
                        }
                    }
                    TreePop();
                }

                // Collide-with mode: All / Allowlist / Blocklist
                int mode = 0; // all
                if (!filter.CollideWithSystems.empty()) mode = 1; // allowlist
                else if (!filter.NotCollideWithSystems.empty()) mode = 2; // blocklist
                TextUnformatted("Collide with:");
                SameLine();
                if (RadioButton("All", &mode, 0)) {
                    filter.CollideWithSystems.clear();
                    filter.NotCollideWithSystems.clear();
                    filters_changed = true;
                }
                SameLine();
                if (RadioButton("Allowlist", &mode, 1)) {
                    filter.NotCollideWithSystems.clear();
                    if (filter.CollideWithSystems.empty()) filter.CollideWithSystems = all_systems; // default: all checked
                    filters_changed = true;
                }
                SameLine();
                if (RadioButton("Blocklist", &mode, 2)) {
                    filter.CollideWithSystems.clear();
                    filters_changed = true;
                }

                if (mode == 1) {
                    Indent();
                    for (const auto &sys : all_systems) {
                        bool checked = std::find(filter.CollideWithSystems.begin(), filter.CollideWithSystems.end(), sys) != filter.CollideWithSystems.end();
                        if (Checkbox(sys.c_str(), &checked)) {
                            if (checked) filter.CollideWithSystems.push_back(sys);
                            else std::erase(filter.CollideWithSystems, sys);
                            filters_changed = true;
                        }
                    }
                    Unindent();
                } else if (mode == 2) {
                    Indent();
                    for (const auto &sys : all_systems) {
                        bool checked = std::find(filter.NotCollideWithSystems.begin(), filter.NotCollideWithSystems.end(), sys) != filter.NotCollideWithSystems.end();
                        if (Checkbox(sys.c_str(), &checked)) {
                            if (checked) filter.NotCollideWithSystems.push_back(sys);
                            else std::erase(filter.NotCollideWithSystems, sys);
                            filters_changed = true;
                        }
                    }
                    Unindent();
                }

                TreePop();
            }
            bool filter_in_use = false;
            for (auto [e, c] : r.view<const PhysicsCollider>().each()) {
                if (c.CollisionFilterIndex == uint32_t(i)) {
                    filter_in_use = true;
                    break;
                }
            }
            if (!filter_in_use) {
                for (auto [e, t] : r.view<const PhysicsTrigger>().each()) {
                    if (t.CollisionFilterIndex == uint32_t(i)) {
                        filter_in_use = true;
                        break;
                    }
                }
            }
            if (DeleteButton(filter_in_use)) delete_filter_idx = int(i);
            PopID();
        }
        if (delete_filter_idx >= 0) {
            physics.Filters.erase(physics.Filters.begin() + delete_filter_idx);
            r.emplace_or_replace<PhysicsResourceDeleted>(scene_entity, PhysicsResourceDeleted::Filter, uint32_t(delete_filter_idx));
            filters_changed = true;
        }

        // Add system name popup
        if (Button("Add system")) OpenPopup("AddSystem");
        if (BeginPopup("AddSystem")) {
            static char sys_buf[64] = "";
            InputText("System name", sys_buf, sizeof(sys_buf));
            if (Button("OK") && sys_buf[0] != '\0') {
                std::string name = sys_buf;
                if (std::find(all_systems.begin(), all_systems.end(), name) == all_systems.end()) {
                    // Add to all filters' collision systems by default
                    for (auto &f : physics.Filters) f.CollisionSystems.push_back(name);
                    filters_changed = true;
                }
                sys_buf[0] = '\0';
                CloseCurrentPopup();
            }
            EndPopup();
        }

        SameLine();
        if (Button("Add filter")) {
            physics.Filters.push_back({.Name = std::format("Filter {}", physics.Filters.size())});
            filters_changed = true;
        }

        if (filters_changed) r.emplace_or_replace<PhysicsFiltersDirty>(scene_entity);

        // Collision matrix visualization (angled-header table)
        if (physics.Filters.size() >= 2) {
            Spacing();
            SeparatorText("Collision Matrix");

            const auto n = physics.Filters.size();
            if (BeginTable("##CollisionMatrix", int(n) + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit)) {
                // Row header column
                TableSetupColumn("", ImGuiTableColumnFlags_NoHeaderLabel);
                for (size_t i = 0; i < n; ++i) {
                    auto col_label = physics.Filters[i].Name.empty() ? std::format("{}", i) : physics.Filters[i].Name;
                    TableSetupColumn(col_label.c_str(), ImGuiTableColumnFlags_AngledHeader);
                }
                TableAngledHeadersRow();

                for (size_t row = 0; row < n; ++row) {
                    TableNextRow();
                    TableSetColumnIndex(0);
                    auto row_label = physics.Filters[row].Name.empty() ? std::format("{}", row) : physics.Filters[row].Name;
                    TextUnformatted(row_label.c_str());
                    for (size_t col = 0; col < n; ++col) {
                        TableSetColumnIndex(int(col) + 1);
                        bool collides = physics.DoFiltersCollide(uint32_t(row), uint32_t(col));
                        if (collides) {
                            TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Y");
                        } else {
                            TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "-");
                        }
                    }
                }
                EndTable();
            }
        }

        PopID();
    }

    if (CollapsingHeader("Joint Definitions")) {
        PushID("JointDefs");
        for (size_t i = 0; i < physics.JointDefs.size(); ++i) {
            PushID(int(i));
            auto &jd = physics.JointDefs[i];
            const auto label = jd.Name.empty() ? std::format("Joint {}", i) : jd.Name;
            if (TreeNode(label.c_str())) {
                static const char *axis_names[]{"X", "Y", "Z"};
                // Limits
                for (size_t li = 0; li < jd.Limits.size(); ++li) {
                    PushID(int(li));
                    auto &limit = jd.Limits[li];
                    if (TreeNode("Limit", "Limit %zu", li)) {
                        // Linear axes checkboxes
                        TextUnformatted("Linear axes:");
                        SameLine();
                        for (uint8_t a = 0; a < 3; ++a) {
                            bool active = std::find(limit.LinearAxes.begin(), limit.LinearAxes.end(), a) != limit.LinearAxes.end();
                            if (Checkbox(axis_names[a], &active)) {
                                if (active) limit.LinearAxes.push_back(a);
                                else std::erase(limit.LinearAxes, a);
                            }
                            if (a < 2) SameLine();
                        }
                        // Angular axes checkboxes
                        TextUnformatted("Angular axes:");
                        SameLine();
                        for (uint8_t a = 0; a < 3; ++a) {
                            PushID(a + 3);
                            bool active = std::find(limit.AngularAxes.begin(), limit.AngularAxes.end(), a) != limit.AngularAxes.end();
                            if (Checkbox(axis_names[a], &active)) {
                                if (active) limit.AngularAxes.push_back(a);
                                else std::erase(limit.AngularAxes, a);
                            }
                            if (a < 2) SameLine();
                            PopID();
                        }

                        bool has_min = limit.Min.has_value(), has_max = limit.Max.has_value();
                        float min_val = limit.Min.value_or(0.0f), max_val = limit.Max.value_or(0.0f);
                        if (Checkbox("Min", &has_min)) limit.Min = has_min ? std::optional{min_val} : std::nullopt;
                        if (has_min) {
                            SameLine();
                            DragFloat("##min", &min_val, 0.01f);
                            limit.Min = min_val;
                        }
                        if (Checkbox("Max", &has_max)) limit.Max = has_max ? std::optional{max_val} : std::nullopt;
                        if (has_max) {
                            SameLine();
                            DragFloat("##max", &max_val, 0.01f);
                            limit.Max = max_val;
                        }

                        bool soft = limit.Stiffness.has_value();
                        if (Checkbox("Soft limit", &soft)) limit.Stiffness = soft ? std::optional{1000.0f} : std::nullopt;
                        if (limit.Stiffness) {
                            float stiffness = *limit.Stiffness;
                            if (DragFloat("Stiffness", &stiffness, 1.0f, 0.0f, 1e6f)) limit.Stiffness = stiffness;
                            DragFloat("Damping", &limit.Damping, 0.1f, 0.0f, 1e4f);
                        }
                        TreePop();
                    }
                    PopID();
                }
                if (Button("Add limit")) jd.Limits.push_back({});

                Spacing();

                // Drives
                for (size_t di = 0; di < jd.Drives.size(); ++di) {
                    PushID(int(di + 1000));
                    auto &drive = jd.Drives[di];
                    if (TreeNode("Drive", "Drive %zu", di)) {
                        int type = int(drive.Type);
                        if (Combo("Type", &type, "Linear\0Angular\0")) drive.Type = PhysicsDriveType(type);
                        int axis = drive.Axis;
                        if (Combo("Axis", &axis, "X\0Y\0Z\0")) drive.Axis = uint8_t(axis);
                        int mode = int(drive.Mode);
                        if (Combo("Mode", &mode, "Force\0Acceleration\0")) drive.Mode = PhysicsDriveMode(mode);
                        DragFloat("Max force", &drive.MaxForce, 1.0f, 0.0f, 1e6f);
                        DragFloat("Position target", &drive.PositionTarget, 0.01f);
                        DragFloat("Velocity target", &drive.VelocityTarget, 0.01f);
                        DragFloat("Stiffness", &drive.Stiffness, 1.0f, 0.0f, 1e6f);
                        DragFloat("Damping", &drive.Damping, 0.1f, 0.0f, 1e4f);
                        TreePop();
                    }
                    PopID();
                }
                if (Button("Add drive")) jd.Drives.push_back({});

                TreePop();
            }
            PopID();
        }
        if (Button("Add joint definition")) {
            physics.JointDefs.push_back({.Name = std::format("Joint {}", physics.JointDefs.size())});
        }
        PopID();
    }
}

void physics_ui::RenderEntityProperties(entt::registry &r, entt::entity entity, entt::entity scene_entity, PhysicsWorld &physics) {
    if (!CollapsingHeader("Physics")) return;

    PushID("PhysicsEntity");

    auto *motion = r.try_get<PhysicsMotion>(entity);
    auto *collider = r.try_get<PhysicsCollider>(entity);

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
        r.remove<PhysicsMotion>(entity);
        r.remove<PhysicsCollider>(entity);
        physics.RemoveBody(r, entity);

        if (motion_type >= MT_Static) {
            PhysicsCollider c;
            c.Shape = InitColliderShape(r, entity);
            r.emplace<PhysicsCollider>(entity, std::move(c));
        }
        if (motion_type >= MT_Dynamic) r.emplace<PhysicsMotion>(entity);

        // Re-add body if collider present, or if trigger-only with geometry
        const auto *trig = r.try_get<PhysicsTrigger>(entity);
        if (motion_type >= MT_Static || (trig && trig->Shape.has_value())) physics.AddBody(r, entity);
        motion = r.try_get<PhysicsMotion>(entity);
        collider = r.try_get<PhysicsCollider>(entity);
    }

    if (collider) { // Collider shape editing
        Spacing();
        SeparatorText("Collider");

        if (RenderShapeEditor(collider->Shape)) {
            physics.RemoveBody(r, entity);
            physics.AddBody(r, entity);
        }

        // Physics material assignment
        if (!physics.Materials.empty()) {
            const int mat_idx = collider->PhysicsMaterialIndex.has_value() ? int(*collider->PhysicsMaterialIndex) : -1;
            const auto preview = mat_idx >= 0 && mat_idx < int(physics.Materials.size()) ? (physics.Materials[mat_idx].Name.empty() ? std::format("Material {}", mat_idx) : physics.Materials[mat_idx].Name) : std::string{"None"};
            if (BeginCombo("Physics material", preview.c_str())) {
                if (Selectable("None", mat_idx < 0)) collider->PhysicsMaterialIndex.reset();
                for (size_t i = 0; i < physics.Materials.size(); ++i) {
                    const auto &m = physics.Materials[i];
                    const auto label = m.Name.empty() ? std::format("Material {}", i) : m.Name;
                    if (Selectable(label.c_str(), mat_idx == int(i))) collider->PhysicsMaterialIndex = i;
                }
                EndCombo();
            }
        }

        // Collision filter assignment
        if (!physics.Filters.empty()) {
            const int filter_idx = collider->CollisionFilterIndex.has_value() ? int(*collider->CollisionFilterIndex) : -1;
            const auto preview = filter_idx >= 0 && filter_idx < int(physics.Filters.size()) ? (physics.Filters[filter_idx].Name.empty() ? std::format("Filter {}", filter_idx) : physics.Filters[filter_idx].Name) : std::string{"None"};
            if (BeginCombo("Collision filter", preview.c_str())) {
                if (Selectable("None", filter_idx < 0)) {
                    collider->CollisionFilterIndex.reset();
                    physics.RemoveBody(r, entity);
                    physics.AddBody(r, entity);
                }
                for (size_t i = 0; i < physics.Filters.size(); ++i) {
                    const auto &f = physics.Filters[i];
                    const auto flabel = f.Name.empty() ? std::format("Filter {}", i) : f.Name;
                    if (Selectable(flabel.c_str(), filter_idx == int(i))) {
                        collider->CollisionFilterIndex = uint32_t(i);
                        physics.RemoveBody(r, entity);
                        physics.AddBody(r, entity);
                    }
                }
                EndCombo();
            }
        }
    }

    // Motion properties editing
    if (motion) {
        Spacing();
        SeparatorText("Motion");

        const bool is_simulating = r.get<const AnimationTimeline>(scene_entity).Playing;
        if (is_simulating) BeginDisabled();
        DragFloat3("Linear velocity", &motion->LinearVelocity.x, 0.1f);
        DragFloat3("Angular velocity", &motion->AngularVelocity.x, 0.1f);
        if (is_simulating) EndDisabled();

        DragFloat("Gravity factor", &motion->GravityFactor, 0.01f, -10.0f, 10.0f);

        Spacing();
        SeparatorText("Mass properties");

        // IsKinematic changes Jolt EMotionType, so the body must be rebuilt.
        if (Checkbox("Infinite mass", &motion->IsKinematic)) {
            physics.RemoveBody(r, entity);
            physics.AddBody(r, entity);
        }

        // Mass/inertia overrides are meaningless on a kinematic body (infinite mass).
        if (motion->IsKinematic) BeginDisabled();

        // Mass: always shown. Unset displays DefaultMass and round-trips as absent;
        // any user drag makes it explicit.
        float mass = motion->Mass.value_or(DefaultMass);
        if (DragFloat("Mass", &mass, 0.1f, 0.001f, 1e6f, "%.3f kg")) motion->Mass = mass;

        bool has_inertia = motion->InertiaDiagonal.has_value();
        if (Checkbox("Override inertia tensor", &has_inertia)) {
            if (has_inertia) {
                motion->InertiaDiagonal = vec3{1.0f};
                motion->InertiaOrientation = quat{1, 0, 0, 0};
            } else {
                motion->InertiaDiagonal.reset();
                motion->InertiaOrientation.reset();
            }
        }
        if (motion->InertiaDiagonal) {
            DragFloat3("Inertia diagonal", &motion->InertiaDiagonal->x, 0.01f, 0.001f, 1e6f);
            vec3 euler_deg = glm::degrees(glm::eulerAngles(motion->InertiaOrientation.value_or(quat{1, 0, 0, 0})));
            if (DragFloat3("Inertia orientation", &euler_deg.x, 0.1f)) {
                motion->InertiaOrientation = quat{glm::radians(euler_deg)};
            }
        }

        if (motion->IsKinematic) EndDisabled();

        // Center of mass stays editable even when kinematic — still affects rotation pivoting.
        bool has_com = motion->CenterOfMass.has_value();
        if (Checkbox("Override center of mass", &has_com)) {
            motion->CenterOfMass = has_com ? std::optional{vec3{0.0f}} : std::nullopt;
        }
        if (motion->CenterOfMass) DragFloat3("Center of mass", &motion->CenterOfMass->x, 0.01f);
    }

    // Joint properties
    if (auto *joint = r.try_get<PhysicsJoint>(entity)) {
        Spacing();
        SeparatorText("Joint");

        // Joint definition selector
        if (!physics.JointDefs.empty()) {
            const uint32_t clamped_idx = std::min(joint->JointDefIndex, uint32_t(physics.JointDefs.size() - 1));
            const auto &def = physics.JointDefs[clamped_idx];
            const auto preview = def.Name.empty() ? std::format("Joint {}", clamped_idx) : def.Name;
            if (BeginCombo("Definition", preview.c_str())) {
                for (size_t i = 0; i < physics.JointDefs.size(); ++i) {
                    const auto &jd = physics.JointDefs[i];
                    const auto jlabel = jd.Name.empty() ? std::format("Joint {}", i) : jd.Name;
                    if (Selectable(jlabel.c_str(), clamped_idx == uint32_t(i))) joint->JointDefIndex = uint32_t(i);
                }
                EndCombo();
            }
            Text("Limits: %zu, Drives: %zu", def.Limits.size(), def.Drives.size());
        }

        Checkbox("Enable collision", &joint->EnableCollision);

        if (joint->ConnectedNode != entt::null) Text("Connected: entity %u", joint->ConnectedNode);
        else TextDisabled("Not connected");
    }

    // Trigger properties
    if (auto *trigger = r.try_get<PhysicsTrigger>(entity); !trigger) {
        Spacing();
        if (Button("Add Trigger")) {
            r.emplace<PhysicsTrigger>(entity, PhysicsTrigger{.Shape = PhysicsShape{}});
            // Only create a sensor body if no collider body already exists (collider wins).
            if (!r.all_of<PhysicsBodyHandle>(entity)) physics.AddBody(r, entity);
        }
    } else {
        Spacing();
        SeparatorText("Trigger");
        PushID("Trigger");

        if (trigger->Shape.has_value()) {
            if (RenderShapeEditor(*trigger->Shape)) {
                physics.RemoveBody(r, entity);
                physics.AddBody(r, entity);
            }
        } else if (!trigger->Nodes.empty()) {
            Text("Compound trigger: %zu nodes", trigger->Nodes.size());
        }

        // Collision filter assignment
        if (!physics.Filters.empty()) {
            const int filter_idx = trigger->CollisionFilterIndex.has_value() ? int(*trigger->CollisionFilterIndex) : -1;
            const auto preview = filter_idx >= 0 && filter_idx < int(physics.Filters.size()) ? (physics.Filters[filter_idx].Name.empty() ? std::format("Filter {}", filter_idx) : physics.Filters[filter_idx].Name) : std::string{"None"};
            if (BeginCombo("Collision filter", preview.c_str())) {
                if (Selectable("None", filter_idx < 0)) {
                    trigger->CollisionFilterIndex.reset();
                    physics.RemoveBody(r, entity);
                    physics.AddBody(r, entity);
                }
                for (size_t i = 0; i < physics.Filters.size(); ++i) {
                    const auto &f = physics.Filters[i];
                    const auto flabel = f.Name.empty() ? std::format("Filter {}", i) : f.Name;
                    if (Selectable(flabel.c_str(), filter_idx == int(i))) {
                        trigger->CollisionFilterIndex = uint32_t(i);
                        physics.RemoveBody(r, entity);
                        physics.AddBody(r, entity);
                    }
                }
                EndCombo();
            }
        }

        if (Button("Remove Trigger")) {
            physics.RemoveBody(r, entity);
            r.remove<PhysicsTrigger>(entity);
        }

        PopID();
    }

    PopID();
}
