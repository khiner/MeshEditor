// Physics UI: "Physics" tab + per-entity physics properties.
// Free functions — no Scene.h dependency.

#include "PhysicsUi.h"
#include "AnimationTimeline.h"
#include "Entity.h"
#include "Instance.h"
#include "PhysicsWorld.h"
#include "SceneTree.h"
#include "Variant.h"
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

// Initialize a ColliderShape from the PrimitiveShape component if present,
// otherwise fall back to AABB-derived box dimensions.
ColliderShape InitColliderShape(const entt::registry &r, entt::entity entity) {
    const auto mesh_entity = FindMeshEntity(r, entity);
    const auto shape = [&]() -> PhysicsShape {
        if (const auto *prim = r.try_get<const PrimitiveShape>(mesh_entity)) {
            return std::visit(
                overloaded{
                    [](const primitive::Cuboid &s) -> PhysicsShape { return physics::Box{s.HalfExtents * 2.f}; },
                    [](const primitive::Plane &s) -> PhysicsShape { return physics::Plane{s.HalfExtents.x * 2.f, s.HalfExtents.y * 2.f}; },
                    [](const primitive::IcoSphere &s) -> PhysicsShape { return physics::Sphere{s.Radius}; },
                    [](const primitive::UVSphere &s) -> PhysicsShape { return physics::Sphere{s.Radius}; },
                    [](const primitive::Cylinder &s) -> PhysicsShape { return physics::Cylinder{s.Height, s.Radius, s.Radius}; },
                    [](const primitive::Cone &s) -> PhysicsShape { return physics::Cylinder{s.Height, 0.f, s.Radius}; },
                    // Circle, Torus → ConvexHull from mesh geometry
                    [](const auto &) -> PhysicsShape { return physics::ConvexHull{}; },
                },
                *prim
            );
        }

        // Fallback: derive shape from mesh AABB.
        const auto *mesh = r.try_get<const Mesh>(mesh_entity);
        if (!mesh || mesh->VertexCount() == 0) return physics::Box{};

        const auto verts = mesh->GetVerticesSpan();
        vec3 lo = verts[0].Position, hi = lo;
        for (const auto &v : verts) {
            lo = glm::min(lo, v.Position);
            hi = glm::max(hi, v.Position);
        }
        const vec3 extents = hi - lo;
        // If any AABB dimension is degenerate, use ConvexHull instead of a zero-thickness box.
        if (extents.x < 1e-6f || extents.y < 1e-6f || extents.z < 1e-6f) return physics::ConvexHull{};
        return physics::Box{extents};
    }();
    return ColliderShape{.Shape = shape, .MeshEntity = IsMeshBackedShape(shape) ? mesh_entity : null_entity};
}

// Returns `name` if non-empty, else a bracketed fallback formatted from `fmt`/args (e.g., "<3>").
// Brackets signal "no explicit name" and can't be confused with a user-typed value.
// Uses vformat since consteval-checking of `fmt` is lost through the forwarded pack.
std::string DisplayName(std::string_view name, std::string_view fmt, auto &&...args) {
    if (!name.empty()) return std::string{name};
    return std::format("<{}>", std::vformat(fmt, std::make_format_args(args...)));
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
// Returns the currently-selected Target (or nullptr if none/invalid). When `empty_preview` is non-null
// and the registry has no Target entities, renders a disabled combo with that preview instead.
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
    const auto preview = cur ? DisplayName(cur->Name, "{:x}", uint32_t(cur_e)) : std::string{"None"};
    if (BeginCombo(label, preview.c_str())) {
        if (Selectable("None", cur_e == null_entity)) {
            r.patch<Owner>(entity, [](Owner &o) { o.*Field = null_entity; });
        }
        for (auto [te, t] : view.each()) {
            if (Selectable(DisplayName(t.Name, "{:x}", uint32_t(te)).c_str(), cur_e == te)) {
                r.patch<Owner>(entity, [te](Owner &o) { o.*Field = te; });
            }
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

// Multi-select combo over all CollisionSystem entities. `on_toggle(system, now_member)` fires per change.
// Renders a disabled combo with "No systems defined" when none exist.
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
            if (Checkbox(SystemDisplayName(r, se).c_str(), &member)) on_toggle(se, member);
        }
        EndCombo();
    }
}

size_t CountFilterUses(const entt::registry &r, entt::entity filter) {
    size_t n = 0;
    for (auto [e, m] : r.view<const ColliderMaterial>().each()) {
        if (m.CollisionFilterEntity == filter) ++n;
    }
    for (auto [e, t] : r.view<const PhysicsTrigger>().each()) {
        if (t.CollisionFilterEntity == filter) ++n;
    }
    return n;
}

// Editor for a collision filter's body: membership, Mode, CollideSystems. Sole editing surface —
// per-entity panels only reference filters by combo; property edits happen here in the Physics tab.
void RenderCollisionFilterBody(entt::registry &r, entt::entity filter_e) {
    const auto &filter = r.get<const CollisionFilter>(filter_e);
    RenderSystemMultiSelect(r, "Member of", filter.Systems, [&](entt::entity se, bool add) {
        r.patch<CollisionFilter>(filter_e, [&](CollisionFilter &f) { ToggleInVector(f.Systems, se, add); });
    });

    int mode = int(filter.Mode);
    TextUnformatted("Collide with:");
    SameLine();
    bool mode_changed = RadioButton("All", &mode, int(CollideMode::All));
    SameLine();
    mode_changed |= RadioButton("Allowlist", &mode, int(CollideMode::Allowlist));
    SameLine();
    mode_changed |= RadioButton("Blocklist", &mode, int(CollideMode::Blocklist));
    if (mode_changed) r.patch<CollisionFilter>(filter_e, [mode](CollisionFilter &f) { f.Mode = CollideMode(mode); });

    if (mode != int(CollideMode::All)) {
        Indent();
        RenderSystemMultiSelect(r, "##collide", filter.CollideSystems, [&](entt::entity se, bool add) {
            r.patch<CollisionFilter>(filter_e, [&](CollisionFilter &f) { ToggleInVector(f.CollideSystems, se, add); });
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
    const ImU32 collide = IM_COL32(60, 200, 60, 220);
    const ImU32 overridden = IM_COL32(200, 60, 60, 220);
    const ImU32 fill = (a_to_b && b_to_a) ? collide : overridden;
    const ImVec2 TL = p_min, TR{p_max.x, p_min.y}, BR = p_max, BL{p_min.x, p_max.y};
    if (a_to_b) dl->AddTriangleFilled(TL, TR, BL, fill); // row→col
    if (b_to_a) dl->AddTriangleFilled(TR, BR, BL, fill); // col→row
}

// Pure-function shape editor: returns the edited shape iff the user changed something.
// MeshEntity is stored on the ColliderShape / PhysicsTrigger wrapper, so it's preserved
// across type changes without needing to be passed through here.
std::optional<PhysicsShape> RenderShapeEditor(const PhysicsShape &in) {
    static const char *shape_names[]{"Box", "Sphere", "Capsule", "Cylinder", "Plane", "Convex Hull", "Triangle Mesh"};
    auto out = in;
    bool changed = false;
    auto shape_type_i = int(out.index());
    if (Combo("Shape", &shape_type_i, shape_names, IM_ARRAYSIZE(shape_names))) {
        out = CreateVariantByIndex<PhysicsShape>(size_t(shape_type_i));
        changed = true;
    }
    std::visit(
        overloaded{
            [&](physics::Box &s) { changed |= DragFloat3("Size", &s.Size.x, 0.01f, 0.01f, 100.f); },
            [&](physics::Sphere &s) { changed |= DragFloat("Radius", &s.Radius, 0.01f, 0.001f, 100.f); },
            [&](physics::Capsule &s) {
                changed |= DragFloat("Height", &s.Height, 0.01f, 0.001f, 100.f);
                changed |= DragFloat("Radius top", &s.RadiusTop, 0.01f, 0.001f, 100.f);
                changed |= DragFloat("Radius bottom", &s.RadiusBottom, 0.01f, 0.001f, 100.f);
            },
            [&](physics::Cylinder &s) {
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
            const auto label = DisplayName(mat.Name, "{:x}", uint32_t(mat_entity));
            const bool expanded = TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanTextWidth, "%s", label.c_str());
            size_t mat_uses = 0;
            for (auto [e, m] : r.view<const ColliderMaterial>().each()) {
                if (m.PhysicsMaterialEntity == mat_entity) ++mat_uses;
            }
            SameLine();
            TextDisabled("(%zu)", mat_uses);
            SameLine();
            if (SmallButton("X")) delete_entity = mat_entity;
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

    if (CollapsingHeader("Collision Systems")) {
        PushID("CollisionSystems");
        entt::entity delete_entity = entt::null;
        size_t sys_idx = 0;
        for (auto [se, s] : r.view<CollisionSystem>().each()) {
            PushID(uint32_t(se));
            const auto label = DisplayName(s.Name, "{:x}", uint32_t(se));

            size_t uses = 0;
            for (auto [fe, f] : r.view<const CollisionFilter>().each()) {
                if (std::find(f.Systems.begin(), f.Systems.end(), se) != f.Systems.end() ||
                    std::find(f.CollideSystems.begin(), f.CollideSystems.end(), se) != f.CollideSystems.end()) ++uses;
            }
            const bool expanded = TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanTextWidth, "%s", label.c_str());
            SameLine();
            TextDisabled("(%zu)", uses);
            SameLine();
            if (SmallButton("X")) delete_entity = se;
            if (expanded) {
                RenderNameEdit<CollisionSystem>(r, se, s.Name);
                TreePop();
            }
            PopID();
            ++sys_idx;
        }
        if (delete_entity != entt::null) r.destroy(delete_entity);
        if (Button("Add system")) {
            const auto e = r.create();
            r.emplace<CollisionSystem>(e, CollisionSystem{.Name = std::format("System {}", sys_idx)});
        }
        PopID();
    }

    if (CollapsingHeader("Collision Filters")) {
        PushID("CollisionFilters");

        entt::entity delete_entity = entt::null;
        size_t idx = 0;
        for (auto [fe, filter] : r.view<CollisionFilter>().each()) {
            PushID(uint32_t(fe));
            const auto label = DisplayName(filter.Name, "{:x}", uint32_t(fe));
            const bool expanded = TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanTextWidth, "%s", label.c_str());
            SameLine();
            TextDisabled("(%zu)", CountFilterUses(r, fe));
            SameLine();
            if (SmallButton("X")) delete_entity = fe;
            if (expanded) {
                RenderNameEdit<CollisionFilter>(r, fe, filter.Name);
                RenderCollisionFilterBody(r, fe);
                TreePop();
            }
            PopID();
            ++idx;
        }
        if (delete_entity != entt::null) r.destroy(delete_entity);

        if (Button("Add filter")) {
            const auto e = r.create();
            r.emplace<CollisionFilter>(e, CollisionFilter{.Name = std::format("Filter {}", idx)});
        }

        // Collision matrix (directional split cells + per-direction tooltip).
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
        PushID("JointDefs");
        entt::entity delete_jd_entity = entt::null;
        size_t jd_index = 0;
        for (auto [jd_entity, jd] : r.view<PhysicsJointDef>().each()) {
            PushID(uint32_t(jd_entity));
            const auto label = DisplayName(jd.Name, "{:x}", uint32_t(jd_entity));
            size_t jd_uses = 0;
            for (auto [e, j] : r.view<const PhysicsJoint>().each()) {
                if (j.JointDefEntity == jd_entity) ++jd_uses;
            }
            const bool expanded = TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanTextWidth, "%s", label.c_str());
            SameLine();
            TextDisabled("(%zu)", jd_uses);
            SameLine();
            if (SmallButton("X")) delete_jd_entity = jd_entity;
            if (expanded) {
                RenderNameEdit<PhysicsJointDef>(r, jd_entity, jd.Name);
                static const char *axis_names[]{"X", "Y", "Z"};
                // Limits
                std::optional<size_t> delete_limit;
                for (size_t li = 0; li < jd.Limits.size(); ++li) {
                    PushID(int(li));
                    const auto &limit = jd.Limits[li];
                    const bool limit_expanded = TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanTextWidth, "Limit %zu", li);
                    SameLine();
                    if (SmallButton("X")) delete_limit = li;
                    if (limit_expanded) {
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
                if (delete_limit) r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Limits.erase(d.Limits.begin() + *delete_limit); });
                if (Button("Add limit")) r.patch<PhysicsJointDef>(jd_entity, [](auto &d) { d.Limits.push_back({}); });

                Spacing();

                // Drives
                std::optional<size_t> delete_drive;
                for (size_t di = 0; di < jd.Drives.size(); ++di) {
                    PushID(int(di + 1000));
                    const auto &drive = jd.Drives[di];
                    const bool drive_expanded = TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanTextWidth, "Drive %zu", di);
                    SameLine();
                    if (SmallButton("X")) delete_drive = di;
                    if (drive_expanded) {
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
                if (delete_drive) r.patch<PhysicsJointDef>(jd_entity, [&](auto &d) { d.Drives.erase(d.Drives.begin() + *delete_drive); });
                if (Button("Add drive")) r.patch<PhysicsJointDef>(jd_entity, [](auto &d) { d.Drives.push_back({}); });

                TreePop();
            }
            PopID();
            ++jd_index;
        }
        if (delete_jd_entity != entt::null) r.destroy(delete_jd_entity);
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

    // 4-way motion type matching Jolt's EMotionType + a "none" state.
    // Projects losslessly to/from {ColliderShape?, PhysicsMotion?, PhysicsMotion::IsKinematic}.
    enum : int { MT_None = 0,
                 MT_Static,
                 MT_Kinematic,
                 MT_Dynamic };
    int motion_type = MT_None;
    if (collider && !motion) motion_type = MT_Static;
    else if (motion && motion->IsKinematic) motion_type = MT_Kinematic;
    else if (motion) motion_type = MT_Dynamic;

    AlignTextToFramePadding();
    TextUnformatted("Motion type:");
    SameLine();
    bool changed = RadioButton("None", &motion_type, MT_None);
    SameLine();
    changed |= RadioButton("Static", &motion_type, MT_Static);
    SameLine();
    changed |= RadioButton("Kinematic", &motion_type, MT_Kinematic);
    SameLine();
    changed |= RadioButton("Dynamic", &motion_type, MT_Dynamic);

    if (changed) {
        const bool want_motion = motion_type >= MT_Kinematic;
        const bool want_collider = motion_type >= MT_Static;
        if (!want_motion) r.remove<PhysicsMotion>(entity);
        if (!want_collider) r.remove<ColliderShape>(entity);
        if (want_collider && !r.all_of<ColliderShape>(entity)) {
            r.emplace<ColliderShape>(entity, InitColliderShape(r, entity));
        }
        if (want_motion) {
            const bool is_kinematic = motion_type == MT_Kinematic;
            if (!r.all_of<PhysicsMotion>(entity)) r.emplace<PhysicsMotion>(entity, PhysicsMotion{.IsKinematic = is_kinematic});
            else r.patch<PhysicsMotion>(entity, [is_kinematic](PhysicsMotion &m) { m.IsKinematic = is_kinematic; });
        }
        motion = r.try_get<const PhysicsMotion>(entity);
        collider = r.try_get<const ColliderShape>(entity);
    }

    if (collider) { // Collider shape editing
        Spacing();
        SeparatorText("Collider");

        if (auto s = RenderShapeEditor(collider->Shape)) {
            const auto owner_mesh = FindMeshEntity(r, entity);
            r.patch<ColliderShape>(entity, [&](ColliderShape &cs) {
                cs.Shape = *s;
                // Seed MeshEntity on first-ever flip into a mesh-backed variant (primitive-only
                // collider being changed to ConvexHull/TriangleMesh). Preserve it across all other
                // flips — including mesh → primitive → mesh round-trips — so glTF-loaded divergent
                // links survive accidental editor edits.
                if (IsMeshBackedShape(*s) && cs.MeshEntity == null_entity) cs.MeshEntity = owner_mesh;
            });
        }

        RenderEntityCombo<PhysicsMaterial, &ColliderMaterial::PhysicsMaterialEntity>(r, entity, "Physics material", "No materials defined");

        RenderEntityCombo<CollisionFilter, &ColliderMaterial::CollisionFilterEntity>(r, entity, "Collision filter", "No filters defined");
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
    }

    // Joint properties
    if (const auto *joint = r.try_get<const PhysicsJoint>(entity)) {
        Spacing();
        SeparatorText("Joint");

        const auto *cur = RenderEntityCombo<PhysicsJointDef, &PhysicsJoint::JointDefEntity>(r, entity, "Definition", "No joint definitions");
        if (cur) Text("Limits: %zu, Drives: %zu", cur->Limits.size(), cur->Drives.size());

        bool enable_collision = joint->EnableCollision;
        if (Checkbox("Enable collision", &enable_collision)) {
            r.patch<PhysicsJoint>(entity, [enable_collision](PhysicsJoint &j) { j.EnableCollision = enable_collision; });
        }

        // ConnectedNode picker — KHR joint.connectedNode is the second attachment frame.
        // Mirrors Blender's rigid_body_constraint object1/object2 fields.
        const auto cn = joint->ConnectedNode;
        const auto cn_label = cn != null_entity && r.valid(cn) ? GetName(r, cn) : std::string{"None"};
        if (BeginCombo("Connected node", cn_label.c_str())) {
            if (Selectable("None", cn == null_entity)) {
                r.patch<PhysicsJoint>(entity, [](PhysicsJoint &j) { j.ConnectedNode = null_entity; });
            }
            for (auto ne : r.view<const SceneNode>()) {
                if (ne == entity) continue; // self-connection is meaningless
                if (Selectable(GetName(r, ne).c_str(), cn == ne)) {
                    r.patch<PhysicsJoint>(entity, [ne](PhysicsJoint &j) { j.ConnectedNode = ne; });
                }
            }
            EndCombo();
        }
    }

    // Trigger properties
    if (const auto *trigger = r.try_get<const PhysicsTrigger>(entity); !trigger) {
        Spacing();
        if (Button("Add Trigger")) r.emplace<PhysicsTrigger>(entity, PhysicsTrigger{.Shape = physics::Box{}});
    } else {
        Spacing();
        SeparatorText("Trigger");
        PushID("Trigger");

        if (trigger->Shape.has_value()) {
            if (auto s = RenderShapeEditor(*trigger->Shape)) {
                const auto owner_mesh = FindMeshEntity(r, entity);
                r.patch<PhysicsTrigger>(entity, [&](PhysicsTrigger &t) {
                    t.Shape = *s;
                    if (IsMeshBackedShape(*s) && t.MeshEntity == null_entity) t.MeshEntity = owner_mesh;
                });
            }
        } else if (!trigger->Nodes.empty()) {
            Text("Compound trigger: %zu nodes", trigger->Nodes.size());
        }

        RenderEntityCombo<CollisionFilter, &PhysicsTrigger::CollisionFilterEntity>(r, entity, "Collision filter", "No filters defined");

        if (Button("Remove Trigger")) r.remove<PhysicsTrigger>(entity);

        PopID();
    }

    PopID();
}
