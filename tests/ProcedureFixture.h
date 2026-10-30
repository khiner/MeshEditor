#pragma once

#include "ProjectFixture.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClone.h"
#include "numeric/MatrixMath.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "selection/SelectionGpu.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"

// A project fixture for procedure scenarios, with node authoring, viewport clicks and kernel references on clones of a mesh's source.
struct ProcedureFixture : Fixture {
    // Does the action, which must fail.
    void Fails(auto a) {
        P->Do(action::MakeAction(std::move(a)));
        boost::ut::expect(!R.Resources.get<action::Errors>().Messages.empty());
        R.Resources.get<action::Errors>().Messages.clear();
    }
    // Adds a mesh whose procedure displays a generator of the primitive.
    state::Entity AddGenerated(PrimitiveParameters parameters) {
        Do(action::object::AddMeshPrimitive{std::move(parameters), std::make_unique<MeshInstanceCreateInfo>()});
        return R.get<const Instance>(FindActiveEntity(R)).Entity;
    }
    // Adds a mesh whose procedure displays a Source node holding the primitive's record.
    state::Entity AddMesh(PrimitiveParameters parameters) {
        const auto mesh = AddGenerated(std::move(parameters));
        Do(action::nodes::CollapseProcedure{mesh});
        return mesh;
    }
    // Adds an unlinked node without changing the displayed stage.
    state::Entity Append(state::Entity mesh_entity, uint16_t kind) {
        Do(action::nodes::AddNode{mesh_entity, kind, {}});
        return R.get<const procedure::GeometryProcedure>(mesh_entity).Last;
    }
    // Links each node's first output into the next node's first input and displays the last node.
    void Route(state::Entity mesh_entity, std::initializer_list<state::Entity> nodes) {
        for (const auto *from = nodes.begin(); from + 1 != nodes.end(); ++from) Do(action::nodes::Link{mesh_entity, *from, Out(*from), *(from + 1), 0u});
        Do(action::nodes::SetDisplay{mesh_entity, *(nodes.end() - 1)});
    }
    // The node at the index in the procedure's authoring order.
    state::Entity Node(state::Entity mesh_entity, uint32_t node) const { return R.get<const procedure::CookPlan>(mesh_entity).Nodes[node].Entity; }
    // The node's first output socket.
    uint16_t Out(state::Entity node) const { return procedure::InputCount[R.get<const procedure::ProcedureNode>(node).Kind]; }
    void Stage(auto a) {
        action::Emit(std::move(a), action::Phase::Stage);
        P->Frame(action::Drain());
        boost::ut::expect(R.Resources.get<action::Errors>().Messages.empty());
    }
    void CommitGesture() {
        action::Commit();
        P->Frame(action::Drain());
        boost::ut::expect(!P->HasStaged());
        Audit();
        if (Step) Step(true);
    }
    void EnterEdit(Element element) {
        Do(action::view::SetInteractionMode{InteractionMode::Edit});
        Do(action::view::SetEditMode{.Mode = element});
    }
    void Render() {
        SubmitViewports(R);
        WaitForRender(R);
    }
    const SceneViewUBO &View() const { return *reinterpret_cast<const SceneViewUBO *>(R.get<const Viewport>(FirstViewport(R)).SceneViewUBO.Contents().data()); }
    // Renders a frame and clicks the element nearest the world position, adding it to the selection with `toggle`.
    void Click(vec3 position, bool toggle) {
        Render();
        const auto clip = View().ViewProj * vec4{position.x, position.y, position.z, 1.f};
        Do(action::selection::ApplyEditElementClick{toggle, RunEditElementPick(R, FirstViewport(R), R.get<const EditMode>(state::Document).Value, SelectionPixel(View().ViewportSize, {.5f + .5f * clip.x / clip.w, .5f - .5f * clip.y / clip.w}))});
    }
    // The state of the mesh's first Source node.
    procedure::SourceState SourceOf(state::Entity mesh_entity) const {
        for (const auto &node : R.get<const procedure::CookPlan>(mesh_entity).Nodes) {
            if (node.Kind == procedure::KindOf<procedure::Source>) return R.get<const procedure::SourceState>(node.Entity);
        }
        return {};
    }
    // A new record holding the Source node's geometry, with selection state.
    uint32_t CloneSource(const procedure::SourceState &source) {
        auto &meshes = Store();
        const auto id = source.State == store::Hash128{} ? CloneRecord(R, source.StoreId) : meshes.CloneStored(source.State, source.StoreId).value();
        meshes.RebuildSelectionIndex(std::array{id});
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.EnsureSelectionState(R, chain, std::array{id});
        chain.Submit();
        return id;
    }
    // Calls `read` with a new record holding the geometry of the mesh's first Source node, which it then releases with a direct commit.
    auto WithSource(state::Entity mesh_entity, auto &&read) {
        const auto id = CloneSource(SourceOf(mesh_entity));
        auto result = read(id);
        Store().Release(id);
        Commit("Release source");
        return result;
    }
    Shape SourceShape(state::Entity mesh_entity) {
        return WithSource(mesh_entity, [&](uint32_t id) { return Describe(Store(), id); });
    }
    state::Entity Object(state::Entity mesh_entity) const {
        for (const auto [e, instance] : R.view<const Instance>().each())
            if (instance.Entity == mesh_entity) return e;
        return state::Null;
    }
    // The record the mesh draws.
    uint32_t Output(state::Entity mesh_entity) const { return *DrawnStoreId(R, mesh_entity); }
    Shape Shaped(state::Entity mesh_entity) { return Describe(Store(), Output(mesh_entity)); }
    uint32_t Clone(uint32_t id) {
        auto &meshes = Store();
        meshes.PlanClone(Mesh{meshes, id});
        meshes.CommitReserves();
        CloneCopies copies;
        const auto clone = meshes.CloneMeshes(copies, std::array{id}, true).front();
        mtl::ComputeChain chain{meshes.BufferContext()};
        copies.Encode(chain, GetMeshPipelines(R));
        chain.Submit();
        meshes.RebuildSelectionIndex(std::array{clone});
        Audit();
        return clone;
    }
    void Run(uint32_t id, std::vector<MeshTopologyTask> tasks) {
        for (auto &task : tasks) task.SourceId = id;
        boost::ut::expect(std::ranges::all_of(ExecuteGeometryTopologyStages(R, tasks), &GeometryTopologyResult::Changed));
    }
    void CheckStep();
};

// The record and the selected elements a kernel edit leaves on a clone of `id`, where `edit` runs the kernel on the clone and returns the elements it selects.
struct Kernel {
    Shape Output, Selected;
};
Kernel Reference(ProcedureFixture &f, uint32_t id, auto &&edit) {
    auto &meshes = f.Store();
    const auto clone = f.Clone(id);
    const GeometrySelection selected = edit(clone);
    Kernel kernel{Describe(meshes, clone), Describe(meshes, clone, selected)};
    meshes.Release(clone);
    f.Commit("Release reference");
    return kernel;
}
// The kernel edit on a record holding the geometry of the mesh's Source node.
Kernel SourceReference(ProcedureFixture &f, state::Entity mesh_entity, auto &&edit) {
    return f.WithSource(mesh_entity, [&](uint32_t id) {
        const GeometrySelection selected = edit(id);
        return Kernel{Describe(f.Store(), id), Describe(f.Store(), id, selected)};
    });
}

inline GeometrySelection PublishedKernel(ProcedureFixture &f, uint32_t clone, MeshTopologyTask task) {
    task.SourceId = clone;
    PublishStoreSelection(f.Store(), task);
    f.Run(clone, {task});
    return EditorGeometrySelection(f.Store(), clone);
}


// Runs the viewport gesture rows, the recorded operator rows and the mesh edit scenarios.
void TestOperators(ProcedureFixture &f, Fixture &other);
