#include "action/Mesh.h"
#include "action/Errors.h"

#include "TransformMath.h"
#include "mesh/EditVisibility.h"
#include "metal/Dispatch.h"
#include "object/ObjectOps.h"
#include "procedure/Cook.h"
#include "procedure/Procedure.h"
#include "project/Project.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MeshletBuildGpu.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "state/Registry.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportRenderGpu.h"

#include <format>

namespace {
// Copy each mesh's separated geometry into new meshes, and append a Delete node of the separated elements to the mesh.
void SeparateSelected(state::Registry &r, const ::selection::PrimaryEditInstanceMap &primaries, action::mesh::SeparateMode mode) {
    using namespace procedure;
    const auto &meshes = r.Resources.get<const MeshStore>();
    std::vector<MeshTopologyTask> copies;
    std::vector<state::Entity> copied;
    std::vector<std::pair<state::Entity, MeshTopologyTask>> removals;
    for (const auto &[entity, instance] : primaries) {
        if (!HasMesh(r, entity)) continue;
        const auto id = *DrawnStoreId(r, entity);
        auto planned = SeparateGeometryTasks(meshes, Mesh{meshes, id}, EditorGeometrySelection(meshes, id), mode);
        if (planned.empty()) continue;
        removals.emplace_back(entity, std::move(planned.back()));
        planned.pop_back();
        for (auto &task : planned) {
            PublishStoreSelection(meshes, task);
            copies.push_back(std::move(task));
            copied.push_back(entity);
        }
    }
    if (removals.empty()) return;
    const auto results = ExecuteGeometryTopology(r, copies);
    std::vector<state::Entity> created;
    for (uint32_t i = 0u; i < results.size(); ++i) {
        if (!results[i].Changed) continue;
        const auto instance = primaries.at(copied[i]);
        const MeshInstanceCreateInfo create{
            .Name = std::format("{}.001", GetName(r, instance)),
            .Transform = *WorldTransformOf(r, instance),
            .Select = MeshInstanceCreateInfo::SelectBehavior::None,
        };
        created.push_back(::AddMesh(r, results[i].GeometryId, create).first);
    }
    for (const auto &[entity, removal] : removals) {
        const auto id = *DrawnStoreId(r, entity);
        auto removed = std::get<GeometrySelection>(removal.Selection);
        // A record without faces separates its selected vertices, and keeps a separated vertex an unselected edge still reaches.
        // Its Delete node removes the wires between separated vertices, then the separated vertices they leave without edges.
        const bool wires = removal.Op == MeshTopologyOp::DeleteEdges;
        if (wires) {
            const Mesh mesh{meshes, id};
            for (const auto edge : mesh.edges()) {
                const auto h = mesh.GetHalfedge(edge, 0u);
                if (std::ranges::binary_search(removed.Vertices, *mesh.GetFromVertex(h)) && std::ranges::binary_search(removed.Vertices, *mesh.GetToVertex(h))) removed.Edges.push_back(*edge);
            }
        }
        const auto node = action::nodes::AppendNode(r, entity, KindOf<action::mesh::Delete>);
        if (node == state::Null) return;
        r.replace<action::mesh::Delete>(node, wires ? action::mesh::DeleteMode::Loose : action::mesh::DeleteMode(uint32_t(removal.Op)));
        // The separated elements of the drawn record.
        action::nodes::Apply(r, action::nodes::SetInlineSelection{node, std::make_unique<InlineSelection>(SelectionOf(removed, removal.SelectionElement))});
        r.remove<MeshActiveElement>(entity);
    }
    if (created.empty()) return;
    RequestRender(r);
    for (const auto e : created) r.emplace_or_replace<ClusterLodDemand>(e);
    mtl::ComputeChain chain{meshes.BufferContext()};
    UpdateAuthoredMorphShading(r, chain, created);
    chain.Submit();
}

// The cells an Object Info node sits left of and below the Boolean node it feeds.
constexpr uvec2 InfoOffset{6u, 5u};

// Require exactly two selected mesh objects.
// Append a Boolean node to the active mesh that reads the other object through an Object Info node and keeps the result in the active object's space.
void CombineSelected(state::Registry &r, const action::mesh::Boolean &boolean) {
    const auto active = FindActiveEntity(r);
    std::vector<state::Entity> others;
    for (const auto e : SelectedIn(r)) {
        if (e != active) others.push_back(e);
    }
    if (active == state::Null || others.size() != 1u || !r.all_of<Instance>(active) || !r.all_of<Instance>(others.front())) {
        return action::Fail(r, "A boolean combines the active object with one other selected object");
    }
    const auto first = r.get<const Instance>(active).Entity, second = r.get<const Instance>(others.front()).Entity;
    if (!HasMesh(r, first) || !DrawsMesh(r, second) || first == second) return action::Fail(r, "A boolean combines two different mesh objects");
    const auto first_world = *WorldTransformOf(r, active), second_world = *WorldTransformOf(r, others.front());
    const auto to_first = Inverse(ToMatrix(first_world)) * ToMatrix(second_world);
    const auto rotation = ToMat3(to_first);
    if (!(Determinant(rotation) > 0.f)) return action::Fail(r, "A boolean cannot combine an object mirrored relative to the active one");
    if (!BooleanTask(GetMesh(r, first), GetMesh(r, second), boolean.Operation, rotation, vec3(to_first[3]))) return action::Fail(r, "A boolean combines closed surfaces");
    namespace nodes = action::nodes;
    const auto boolean_entity = nodes::AppendNode(r, first, procedure::KindOf<procedure::Boolean>);
    if (boolean_entity == state::Null) return;
    r.replace<procedure::Boolean>(boolean_entity, boolean.Operation, active);
    const auto position = r.get<const procedure::NodeLayout>(boolean_entity).Position;
    nodes::Apply(r, nodes::AddNode{first, procedure::KindOf<procedure::ObjectInfo>, {position.x - std::min(position.x, InfoOffset.x), position.y + InfoOffset.y}});
    const auto info = r.get<const procedure::GeometryProcedure>(first).Last;
    r.replace<procedure::ObjectInfo>(info, others.front());
    nodes::Apply(r, nodes::Link{first, info, 0u, boolean_entity, 1u});
    nodes::Apply(r, nodes::SelectNodes{first, {boolean_entity}, boolean_entity});
}

// The world transform of the mesh's transformable primary instance, which transformed position operators edit the mesh in.
std::optional<Transform> PrimaryFrame(const state::Registry &r, state::Entity mesh_entity) {
    const auto &primaries = r.get<const EditPrimaries>(state::Document).Transformable;
    const auto primary = primaries.find(mesh_entity);
    if (primary == primaries.end()) return {};
    return *WorldTransformOf(r, primary->second);
}

// The parameters of the node an operator appends to the mesh, resolved from the editor state the operator runs in.
// Return none for a mesh the operator leaves unedited.
template<typename A>
    requires(procedure::KindOf<A> < procedure::KindCount)
std::optional<A> NodeOf(const state::Registry &, state::Entity, const A &a) { return a; }
// A sphere blend across several meshes takes the mean distance of all their selected vertices from the center in each mesh's local space, and one mesh's node measures its own.
std::optional<procedure::ToSphere> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::ToSphere &a) {
    const auto center = EditSelectionCenter(r);
    const auto &primaries = r.get<const EditPrimaries>(state::Document).Transformable;
    const auto blended = action::mesh::SelectedEditMeshes(r) | std::views::filter([&](state::Entity e) { return primaries.contains(e); }) | std::ranges::to<std::vector>();
    float radius = 0.f;
    if (blended.size() > 1u) {
        const auto &meshes = r.Resources.get<const MeshStore>();
        double distance = 0.0;
        uint32_t count = 0u;
        for (const auto e : blended) {
            const auto id = *DrawnStoreId(r, e);
            const Mesh mesh{meshes, id};
            const auto &world = *WorldTransformOf(r, primaries.at(e));
            auto local = Rotate(Conjugate(world.R), center - world.P);
            for (uint32_t axis = 0u; axis < 3u; ++axis) local[axis] = world.S[axis] != 0.f ? local[axis] / world.S[axis] : 0.f;
            for (const auto v : EditorGeometrySelection(meshes, id).Vertices) {
                distance += Length(mesh.GetPosition(he::VH{v}) - local);
                ++count;
            }
        }
        radius = count ? float(distance / double(count)) : 0.f;
    }
    return PrimaryFrame(r, mesh_entity).transform([&](const Transform &frame) { return procedure::ToSphere{a.Factor, center, radius, frame}; });
}
std::optional<procedure::PushPull> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::PushPull &a) {
    return PrimaryFrame(r, mesh_entity).transform([&](const Transform &frame) { return procedure::PushPull{a.Distance, EditSelectionCenter(r), frame}; });
}
// Whether the mesh's drawn record has the corner color layer, or the UV set, that a face attribute edit permutes.
bool HasCornerLayer(const state::Registry &r, state::Entity mesh_entity, bool colors, uint32_t uv_set) {
    if (!colors && uv_set >= MeshStore::MaxUvSets) return false;
    return r.Resources.get<const MeshStore>().Get(*DrawnStoreId(r, mesh_entity)).CornerAttributes & (colors ? MeshAttributeBit_Color0 : MeshAttributeBit_TexCoord0 << uv_set);
}
std::optional<action::mesh::RotateUVs> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::RotateUVs &a) {
    return HasCornerLayer(r, mesh_entity, false, a.UVSet) ? std::optional{a} : std::nullopt;
}
std::optional<action::mesh::ReverseUVs> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::ReverseUVs &a) {
    return HasCornerLayer(r, mesh_entity, false, a.UVSet) ? std::optional{a} : std::nullopt;
}
std::optional<action::mesh::RotateColors> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::RotateColors &a) {
    return HasCornerLayer(r, mesh_entity, true, 0u) ? std::optional{a} : std::nullopt;
}
std::optional<action::mesh::ReverseColors> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::ReverseColors &a) {
    return HasCornerLayer(r, mesh_entity, true, 0u) ? std::optional{a} : std::nullopt;
}
// Resolve the shear axes in world space, from the active object's orientation for a local shear.
std::optional<procedure::Shear> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::Shear &a) {
    const auto axis = uint32_t(a.Axis), ortho = uint32_t(a.AxisOrtho);
    if (axis > 2u || ortho > 2u || axis == ortho) return {};
    vec3 normal{}, direction{};
    normal[axis] = 1.f;
    direction[ortho] = 1.f;
    if (a.Local) {
        const auto active = PrimaryFrame(r, GetActiveMeshEntity(r));
        if (!active) return {};
        normal = active->R * normal * (active->S[axis] < 0.f ? -1.f : 1.f);
        direction = active->R * direction * (active->S[ortho] < 0.f ? -1.f : 1.f);
    }
    return PrimaryFrame(r, mesh_entity).transform([&](const Transform &frame) { return procedure::Shear{a.Angle, EditSelectionCenter(r), direction, normal, frame}; });
}
std::optional<procedure::Warp> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::Warp &a) {
    if (!a.Orientation) return {};
    return PrimaryFrame(r, mesh_entity).transform([&](const Transform &frame) { return procedure::Warp{a.Angle, a.OffsetAngle, a.Center, a.AutoRange, a.Min, a.Max, *a.Orientation, frame}; });
}
std::optional<procedure::Bend> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::Bend &a) {
    if (!a.Orientation) return {};
    return PrimaryFrame(r, mesh_entity).transform([&](const Transform &frame) { return procedure::Bend{a.Angle, a.Radius, a.OffsetAngle, a.Center, a.Clamp, *a.Orientation, frame}; });
}
std::optional<procedure::VertexSlide> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::VertexSlide &a) {
    return PrimaryFrame(r, mesh_entity).transform([&](const Transform &frame) { return procedure::VertexSlide{a.Factor, a.Direction, a.Even, a.Flipped, a.Clamp, frame}; });
}
std::optional<procedure::EdgeSlide> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::EdgeSlide &a) {
    return PrimaryFrame(r, mesh_entity).transform([&](const Transform &frame) { return procedure::EdgeSlide{a.Factor, a.Direction, a.Even, a.Flipped, a.Clamp, frame}; });
}
// Only the View method reads the frame.
std::optional<procedure::Flatten> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::Flatten &a) {
    const auto frame = PrimaryFrame(r, mesh_entity);
    if (!frame && a.Method == action::mesh::FlattenMethod::View) return {};
    return procedure::Flatten{a.Method, a.Factor, a.ViewNormal, a.X, a.Y, a.Z, frame.value_or(Transform{})};
}
// Resolve the view's projection and the world transform of the mesh's primary instance, which the knife projects the mesh with.
std::optional<procedure::Knife> NodeOf(const state::Registry &r, state::Entity mesh_entity, const action::mesh::Knife &a) {
    const auto &primaries = r.get<const EditPrimaries>(state::Document).All;
    const auto primary = primaries.find(mesh_entity);
    if (!a.View || !(a.View->Extent.x > 0.f && a.View->Extent.y > 0.f) || primary == primaries.end()) return {};
    const auto &camera = a.View->Camera;
    return procedure::Knife{a.Start, a.End, a.View->Extent, camera.Projection(a.View->Extent.x / a.View->Extent.y) * camera.View(), *WorldTransformOf(r, primary->second)};
}
} // namespace

namespace action::mesh {
bool Patches(const Action &action) {
    return std::visit(
        []<typename A>(const A &) {
            if constexpr (requires(const state::Registry &r, const A &a) { NodeOf(r, state::Null, a); }) return std::same_as<typename decltype(NodeOf(std::declval<const state::Registry &>(), state::Null, std::declval<const A &>()))::value_type, A>;
            else return false;
        },
        action
    );
}

std::vector<state::Entity> SelectedEditMeshes(const state::Registry &r) {
    std::vector<state::Entity> result;
    const auto element = r.get<const EditMode>(state::Document).Value;
    const auto &meshes = r.Resources.get<const MeshStore>();
    for (const auto e : r.view<const MeshElementSelection>()) {
        if (!HasMesh(r, e)) continue;
        const auto &summary = meshes.GetSelectionSummary(*DrawnStoreId(r, e));
        if (summary.Mode == element && summary.SelectedCount > 0) result.push_back(e);
    }
    return result;
}

void Apply(state::Registry &r, const Action &action) {
    // Append an operator's node to each selected edit mesh.
    // A boolean appends its own nodes, Hide and Reveal act on the drawn geometry, and Separate moves geometry into new meshes.
    const auto visibility = [&](EditVisibilityOperation operation) {
        std::vector<uint32_t> ids;
        for (const auto &[entity, instance] : r.get<const EditPrimaries>(state::Document).All) {
            if (HasMesh(r, entity)) ids.push_back(*DrawnStoreId(r, entity));
        }
        if (EditVisibility(r, ids, r.get<const EditMode>(state::Document).Value, operation)) {
            r.Resources.get<GpuSceneState>().EditSelectionDirty = true;
            for (const auto &[entity, instance] : r.get<const EditPrimaries>(state::Document).All) {
                r.remove<MeshActiveElement>(entity);
                if (!HasMesh(r, entity)) continue;
                ReopenSourceState(r, entity);
                nodes::CaptureActiveSelection(r, entity);
            }
        }
    };
    std::visit(
        [&]<typename A>(const A &a) {
            if constexpr (requires { NodeOf(r, state::Null, a); }) {
                for (const auto e : SelectedEditMeshes(r)) {
                    auto parameters = NodeOf(r, e, a);
                    if (!parameters) continue;
                    using N = decltype(parameters)::value_type;
                    // A later step of a stage replaces the parameters of the node its first step appended, which only the action sets.
                    if constexpr (std::same_as<N, A>) {
                        if (const auto staged = nodes::StagedResultNode(r, e, procedure::KindOf<N>); staged != state::Null) {
                            r.replace<N>(staged, std::move(*parameters));
                            continue;
                        }
                    }
                    const auto node = nodes::AppendNode(r, e, procedure::KindOf<N>);
                    if (node == state::Null) break;
                    r.replace<N>(node, std::move(*parameters));
                    r.emplace<procedure::StagedNode>(node);
                    r.remove<MeshActiveElement>(e);
                }
                if constexpr (std::same_as<A, Extrude> || std::same_as<A, Duplicate> || std::same_as<A, Rip>) r.emplace_or_replace<StartScreenTransform>(state::Document, TransformGizmo::TransformType::Translate);
            } else if constexpr (std::same_as<A, Boolean>) {
                CombineSelected(r, a);
            } else if constexpr (std::same_as<A, Hide>) {
                visibility(a.Unselected ? EditVisibilityOperation::HideUnselected : EditVisibilityOperation::HideSelected);
            } else if constexpr (std::same_as<A, Reveal>) {
                visibility(a.Select ? EditVisibilityOperation::RevealSelected : EditVisibilityOperation::Reveal);
            } else {
                static_assert(std::same_as<A, Separate>);
                SeparateSelected(r, r.get<const EditPrimaries>(state::Document).All, a.Mode);
            }
        },
        action
    );
}
} // namespace action::mesh
