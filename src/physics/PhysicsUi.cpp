// Physics UI: "Physics" tab + per-entity physics properties.
// Free functions — no Scene.h dependency.

#include "PhysicsUi.h"

#include "AnimationTimeline.h"
#include "PhysicsWorld.h"

#include <entt/entity/registry.hpp>
#include <format>
#include <imgui.h>

using namespace ImGui;

void physics_ui::RenderTab(PhysicsWorld &physics) {
    SeparatorText("Simulation");
    Text("Bodies: %u", physics.BodyCount());
    SliderInt("Substeps", &physics.SubSteps, 1, 8);
    DragFloat3("Gravity", &physics.Gravity.x, 0.1f);
    {
        float timestep_ms = physics.TimeStep * 1000.0f;
        if (DragFloat("Time step (ms)", &timestep_ms, 0.1f, 0.1f, 100.0f, "%.1f")) {
            physics.TimeStep = std::max(timestep_ms * 0.001f, 1e-4f);
        }
    }

    if (CollapsingHeader("Physics Materials")) {
        PushID("PhysMaterials");
        for (size_t i = 0; i < physics.Materials.size(); ++i) {
            PushID(int(i));
            auto &mat = physics.Materials[i];
            auto label = mat.Name.empty() ? std::format("Material {}", i) : mat.Name;
            if (TreeNode(label.c_str())) {
                SliderFloat("Static friction", &mat.StaticFriction, 0.0f, 2.0f);
                SliderFloat("Dynamic friction", &mat.DynamicFriction, 0.0f, 2.0f);
                SliderFloat("Restitution", &mat.Restitution, 0.0f, 1.0f);
                TreePop();
            }
            PopID();
        }
        if (Button("Add material")) {
            physics.Materials.push_back({.Name = std::format("Material {}", physics.Materials.size())});
        }
        PopID();
    }

    if (CollapsingHeader("Collision Filters")) {
        PushID("CollisionFilters");
        for (size_t i = 0; i < physics.Filters.size(); ++i) {
            PushID(int(i));
            auto &filter = physics.Filters[i];
            Text("Filter %zu: %s", i, filter.Name.empty() ? "(unnamed)" : filter.Name.c_str());
            PopID();
        }
        if (Button("Add filter")) {
            physics.Filters.push_back({.Name = std::format("Filter {}", physics.Filters.size())});
        }
        PopID();
    }

    if (CollapsingHeader("Joint Definitions")) {
        PushID("JointDefs");
        for (size_t i = 0; i < physics.JointDefs.size(); ++i) {
            PushID(int(i));
            auto &jd = physics.JointDefs[i];
            auto label = jd.Name.empty() ? std::format("Joint {}", i) : jd.Name;
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

    // Motion type radio: None / Static / Kinematic / Dynamic
    int motion_type = 0; // None
    if (collider && !motion) motion_type = 1; // Static
    else if (motion && motion->IsKinematic) motion_type = 2; // Kinematic
    else if (motion) motion_type = 3; // Dynamic

    AlignTextToFramePadding();
    TextUnformatted("Motion type:");
    SameLine();
    bool changed = RadioButton("None", &motion_type, 0);
    SameLine();
    changed |= RadioButton("Static", &motion_type, 1);
    SameLine();
    changed |= RadioButton("Kinematic", &motion_type, 2);
    SameLine();
    changed |= RadioButton("Dynamic", &motion_type, 3);

    if (changed) {
        r.remove<PhysicsMotion>(entity);
        r.remove<PhysicsCollider>(entity);
        physics.RemoveBody(r, entity);

        if (motion_type >= 1) {
            r.emplace<PhysicsCollider>(entity);
        }
        if (motion_type >= 2) {
            r.emplace<PhysicsMotion>(entity, PhysicsMotion{.IsKinematic = (motion_type == 2)});
        }
        if (motion_type >= 1) {
            physics.AddBody(r, entity);
        }
        motion = r.try_get<PhysicsMotion>(entity);
        collider = r.try_get<PhysicsCollider>(entity);
    }

    // Collider shape editing
    if (collider) {
        Spacing();
        SeparatorText("Collider");
        auto &shape = collider->Shape;

        static const char *shape_names[]{"Box", "Sphere", "Capsule", "Cylinder", "Convex Hull", "Triangle Mesh"};
        int shape_type_i = int(shape.Type);
        bool shape_changed = false;
        if (Combo("Shape", &shape_type_i, shape_names, IM_ARRAYSIZE(shape_names))) {
            shape.Type = PhysicsShapeType(shape_type_i);
            shape_changed = true;
        }

        switch (shape.Type) {
            case PhysicsShapeType::Box:
                shape_changed |= DragFloat3("Size", &shape.Size.x, 0.01f, 0.01f, 100.0f);
                break;
            case PhysicsShapeType::Sphere:
                shape_changed |= DragFloat("Radius", &shape.Radius, 0.01f, 0.001f, 100.0f);
                break;
            case PhysicsShapeType::Capsule:
                shape_changed |= DragFloat("Height", &shape.Height, 0.01f, 0.001f, 100.0f);
                shape_changed |= DragFloat("Radius top", &shape.RadiusTop, 0.01f, 0.001f, 100.0f);
                shape_changed |= DragFloat("Radius bottom", &shape.RadiusBottom, 0.01f, 0.001f, 100.0f);
                break;
            case PhysicsShapeType::Cylinder:
                shape_changed |= DragFloat("Height", &shape.Height, 0.01f, 0.001f, 100.0f);
                shape_changed |= DragFloat("Radius top", &shape.RadiusTop, 0.01f, 0.001f, 100.0f);
                shape_changed |= DragFloat("Radius bottom", &shape.RadiusBottom, 0.01f, 0.001f, 100.0f);
                break;
            case PhysicsShapeType::ConvexHull:
            case PhysicsShapeType::TriangleMesh:
                TextDisabled("Mesh-based shape (from glTF import)");
                break;
        }

        if (shape_changed) {
            physics.RemoveBody(r, entity);
            physics.AddBody(r, entity);
        }

        // Physics material assignment
        if (!physics.Materials.empty()) {
            int mat_idx = collider->PhysicsMaterialIndex.has_value() ? int(*collider->PhysicsMaterialIndex) : -1;
            auto preview = mat_idx >= 0 && mat_idx < int(physics.Materials.size()) ? (physics.Materials[mat_idx].Name.empty() ? std::format("Material {}", mat_idx) : physics.Materials[mat_idx].Name) : std::string{"None"};
            if (BeginCombo("Physics material", preview.c_str())) {
                if (Selectable("None", mat_idx < 0)) {
                    collider->PhysicsMaterialIndex.reset();
                }
                for (size_t i = 0; i < physics.Materials.size(); ++i) {
                    const auto &m = physics.Materials[i];
                    auto label = m.Name.empty() ? std::format("Material {}", i) : m.Name;
                    if (Selectable(label.c_str(), mat_idx == int(i))) {
                        collider->PhysicsMaterialIndex = uint32_t(i);
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

        float mass = motion->Mass.value_or(0.0f);
        bool has_mass = motion->Mass.has_value();
        if (Checkbox("Override mass", &has_mass)) {
            motion->Mass = has_mass ? std::optional{mass} : std::nullopt;
        }
        if (has_mass) {
            SameLine();
            if (DragFloat("##mass", &mass, 0.1f, 0.001f, 1e6f, "%.3f kg")) {
                motion->Mass = std::max(mass, 0.001f);
            }
        }

        DragFloat("Gravity factor", &motion->GravityFactor, 0.01f, -10.0f, 10.0f);

        if (is_simulating) BeginDisabled();
        DragFloat3("Linear velocity", &motion->LinearVelocity.x, 0.1f);
        DragFloat3("Angular velocity", &motion->AngularVelocity.x, 0.1f);
        if (is_simulating) EndDisabled();

        if (TreeNode("Inertia")) {
            DragFloat3("Center of mass", &motion->CenterOfMass.x, 0.01f);
            bool has_inertia = motion->InertiaDiagonal.has_value();
            if (Checkbox("Override inertia", &has_inertia)) {
                if (has_inertia) {
                    motion->InertiaDiagonal = vec3{1.0f};
                    motion->InertiaOrientation = quat{1, 0, 0, 0};
                } else {
                    motion->InertiaDiagonal.reset();
                    motion->InertiaOrientation.reset();
                }
            }
            if (motion->InertiaDiagonal.has_value()) {
                DragFloat3("Inertia diagonal", &motion->InertiaDiagonal->x, 0.01f, 0.001f, 1e6f);
            }
            TreePop();
        }
    }

    // Joint properties
    if (auto *joint = r.try_get<PhysicsJoint>(entity)) {
        Spacing();
        SeparatorText("Joint");

        // Joint definition selector
        if (!physics.JointDefs.empty()) {
            uint32_t clamped_idx = std::min(joint->JointDefIndex, uint32_t(physics.JointDefs.size() - 1));
            auto &def = physics.JointDefs[clamped_idx];
            auto preview = def.Name.empty() ? std::format("Joint {}", clamped_idx) : def.Name;
            if (BeginCombo("Definition", preview.c_str())) {
                for (size_t i = 0; i < physics.JointDefs.size(); ++i) {
                    auto &jd = physics.JointDefs[i];
                    auto jlabel = jd.Name.empty() ? std::format("Joint {}", i) : jd.Name;
                    if (Selectable(jlabel.c_str(), clamped_idx == uint32_t(i))) joint->JointDefIndex = uint32_t(i);
                }
                EndCombo();
            }
            Text("Limits: %zu, Drives: %zu", def.Limits.size(), def.Drives.size());
        }

        Checkbox("Enable collision", &joint->EnableCollision);

        if (joint->ConnectedNode != entt::null) {
            Text("Connected: entity %u", uint32_t(joint->ConnectedNode));
        } else {
            TextDisabled("Not connected");
        }
    }

    PopID();
}
