#include "procedure/Procedure.h"
#include "ProjectFixture.h"
#include "TestMeshOrdinals.h"
#include "TopologyCases.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClone.h"
#include "mesh/Primitives.h"
#include "numeric/MatrixMath.h"
#include "object/ObjectOps.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionGpu.h"
#include "viewport/ViewportEvents.h"

using boost::ut::expect, boost::ut::fatal;

namespace {
using namespace procedure;
using action::UpdateOf;
using action::nodes::AddNode, action::nodes::AddProcedure, action::nodes::ApplyProcedure, action::nodes::DeleteNodes, action::nodes::DuplicateNodes, action::nodes::InsertNode, action::nodes::Link, action::nodes::SelectNodes, action::nodes::SetAttributeName, action::nodes::SetInlineSelection;
using ::test::Op, ::test::Patchwork, ::test::PlanCase, ::test::Region, ::test::TopologyCases;

struct ProcedureFixture : Fixture {
    // Does the action, which must fail.
    void Fails(auto a) {
        P->Do(action::MakeAction(std::move(a)));
        expect(!R.Resources.get<action::Errors>().Messages.empty());
        R.Resources.get<action::Errors>().Messages.clear();
    }
    state::Entity AddMesh(PrimitiveShape shape) {
        Do(action::object::AddMeshPrimitive{std::move(shape), std::make_unique<MeshInstanceCreateInfo>()});
        return R.get<const Instance>(FindActiveEntity(R)).Entity;
    }
    std::pair<uint32_t, state::Entity> Append(state::Entity mesh_entity, uint16_t kind) {
        Do(AddNode{mesh_entity, kind, {}});
        const auto &nodes = R.get<const GeometryProcedure>(mesh_entity).Nodes;
        return {uint32_t(nodes.size() - 1u), nodes.back().Entity};
    }
    // Links each node's first output into the next node's first input.
    void Route(state::Entity mesh_entity, std::initializer_list<uint32_t> nodes) {
        for (auto from = nodes.begin(); from + 1 != nodes.end(); ++from)
            Do(Link{mesh_entity, *from, InputCount[R.get<const GeometryProcedure>(mesh_entity).Nodes[*from].Kind], *(from + 1), 0u});
    }
    state::Entity Node(state::Entity mesh_entity, uint32_t node) const { return R.get<const GeometryProcedure>(mesh_entity).Nodes[node].Entity; }
    uint16_t Out(state::Entity mesh_entity, uint32_t node) const { return InputCount[R.get<const GeometryProcedure>(mesh_entity).Nodes[node].Kind]; }
    void Stage(auto a) {
        action::Emit(std::move(a), action::Phase::Stage);
        P->Frame(action::Drain());
        expect(R.Resources.get<action::Errors>().Messages.empty());
    }
    void CommitGesture() {
        action::Commit();
        P->Frame(action::Drain());
        expect(!P->HasStaged());
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
        Do(action::selection::ApplyEditElementClick{{.5f + .5f * clip.x / clip.w, .5f - .5f * clip.y / clip.w}, toggle, std::make_unique<RenderView>(R.get<const Viewport>(FirstViewport(R)).FrameView)});
    }
    uint32_t Source(state::Entity mesh_entity) const { return R.get<const MeshHandle>(mesh_entity).StoreId; }
    state::Entity Object(state::Entity mesh_entity) const {
        for (const auto [e, instance] : R.view<const Instance>().each())
            if (instance.Entity == mesh_entity) return e;
        return state::Null;
    }
    // Return the evaluated procedure record when present, otherwise the source record.
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
        expect(std::ranges::all_of(ExecuteGeometryTopologyStages(R, tasks), &GeometryTopologyResult::Changed));
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
// Each vertex's position, sorted.
std::vector<Corner> Points(const MeshStore &meshes, uint32_t id, auto &&transform) {
    const Mesh mesh{meshes, id};
    std::vector<Corner> points;
    for (const auto v : mesh.vertices()) {
        const auto p = transform(*v, mesh.GetPosition(v));
        points.push_back({p.x, p.y, p.z});
    }
    std::ranges::sort(points);
    return points;
}
std::vector<Corner> Positions(const MeshStore &meshes, uint32_t id) {
    return Points(meshes, id, [](uint32_t, vec3 p) { return p; });
}
bool Near(const std::vector<Corner> &a, const std::vector<Corner> &b) {
    return a.size() == b.size() && std::ranges::equal(a, b, [](const Corner &p, const Corner &q) {
               return std::abs(p[0] - q[0]) < 1e-5f && std::abs(p[1] - q[1]) < 1e-5f && std::abs(p[2] - q[2]) < 1e-5f;
           });
}
// The primitive's vertex positions, sorted.
std::vector<Corner> ShapePositions(const auto &shape) {
    std::vector<Corner> points;
    for (const auto p : primitive::CreateMesh(shape).Positions) points.push_back({p.x, p.y, p.z});
    std::ranges::sort(points);
    return points;
}
// Runs the tasks on a clone of `id`, each selecting every element, and returns the record they leave.
Shape SelectAllReference(ProcedureFixture &f, uint32_t id, std::vector<MeshTopologyTask> tasks) {
    for (auto &task : tasks) task.Flags |= TopologyFlagSelectAll;
    return Reference(f, id, [&](uint32_t clone) {
               f.Run(clone, tasks);
               return GeometrySelection{};
           })
        .Output;
}
// The vertices `inside` accepts, and the edges and faces whose vertices it all accepts.
GeometrySelection SelectionWhere(const Mesh &mesh, auto &&inside) {
    GeometrySelection selection;
    for (const auto v : mesh.vertices())
        if (inside(v)) selection.Vertices.push_back(*v);
    for (const auto e : mesh.edges()) {
        const auto h = mesh.GetHalfedge(e, 0u);
        if (inside(mesh.GetFromVertex(h)) && inside(mesh.GetToVertex(h))) selection.Edges.push_back(*e);
    }
    for (const auto face : mesh.faces())
        if (std::ranges::all_of(mesh.fv_range(face), inside)) selection.Faces.push_back(*face);
    return selection;
}

struct OperatorNode {
    uint16_t Kind;
    void (*Set)(ProcedureFixture &, state::Entity node, const MeshTopologyTask &){};
};
// The node whose cook plans the task, with parameters from the task's op, Param0, Param1, flags, steps and copy translation.
// A merge at a target merges onto the selection's first vertex at its center, as the Center mode does.
// The planned decimation carries no ratio, and its row's ratio is the node's default.
// A face replacement is the area beautify that plans it.
std::optional<OperatorNode> NodeOf(const MeshTopologyTask &task) {
    using enum MeshTopologyOp;
    using namespace action::mesh;
    switch (task.Op) {
        case DeleteVertices:
        case DeleteEdges:
        case DeleteFaces:
        case DeleteOnlyEdgesFaces:
        case DeleteOnlyFaces:
        case DeleteLoose: return OperatorNode{KindOf<Delete>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&Delete::Mode>(n, DeleteMode(t.Op))); }};
        case DissolveVertices:
        case DissolveEdges:
        case DissolveFaces:
        case DissolveLimited:
        case DissolveDegenerate:
            return OperatorNode{KindOf<Dissolve>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    const auto mode = t.Op == DissolveVertices ? DissolveMode::Vertices : t.Op == DissolveEdges ? DissolveMode::Edges :
                                        t.Op == DissolveFaces                                                                   ? DissolveMode::Faces :
                                        t.Op == DissolveLimited                                                                 ? DissolveMode::Limited :
                                                                                                                                  DissolveMode::Degenerate;
                                    f.Do(UpdateOf<&Dissolve::Mode>(n, mode));
                                    f.Do(UpdateOf<&Dissolve::KeepVertices>(n, (t.Flags & TopologyFlagKeepVertices) != 0u));
                                    if (mode == DissolveMode::Limited) f.Do(UpdateOf<&Dissolve::Angle>(n, t.Param0));
                                    if (mode == DissolveMode::Degenerate) f.Do(UpdateOf<&Dissolve::Distance>(n, t.Param0));
                                }};
        case MergeAtTarget:
        case MergeByDistance:
        case MergeCollapse:
            return OperatorNode{KindOf<Merge>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&Merge::Mode>(n, t.Op == MergeByDistance ? MergeMode::ByDistance : t.Op == MergeCollapse ? MergeMode::Collapse :
                                                                                                                                             MergeMode::Center));
                                    if (t.Op == MergeByDistance) f.Do(UpdateOf<&Merge::Distance>(n, t.Param0));
                                }};
        case Decimate: return OperatorNode{KindOf<action::mesh::Decimate>};
        case AddPrimitives: return OperatorNode{KindOf<ConvexHull>};
        case Wireframe:
            return OperatorNode{KindOf<action::mesh::Wireframe>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    using W = action::mesh::Wireframe;
                                    f.Do(UpdateOf<&W::Thickness>(n, t.Param0));
                                    f.Do(UpdateOf<&W::Offset>(n, t.Param1));
                                    f.Do(UpdateOf<&W::Even>(n, (t.Flags & TopologyFlagEvenOffset) != 0u));
                                    f.Do(UpdateOf<&W::Boundary>(n, (t.Flags & TopologyFlagWireBoundary) != 0u));
                                    f.Do(UpdateOf<&W::Relative>(n, (t.Flags & TopologyFlagWireRelative) != 0u));
                                    f.Do(UpdateOf<&W::Replace>(n, (t.Flags & TopologyFlagWireReplace) != 0u));
                                }};
        case BevelEdges:
        case BevelVertices:
            return OperatorNode{KindOf<Bevel>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&Bevel::Width>(n, t.Param0));
                                    f.Do(UpdateOf<&Bevel::Segments>(n, t.Steps));
                                    f.Do(UpdateOf<&Bevel::Vertices>(n, t.Op == BevelVertices));
                                }};
        case TrisToQuads: return OperatorNode{KindOf<action::mesh::TrisToQuads>};
        case Triangulate: return OperatorNode{KindOf<action::mesh::Triangulate>};
        case FlipNormals: return OperatorNode{KindOf<action::mesh::FlipNormals>};
        case SplitGeometry: return OperatorNode{KindOf<Split>};
        case DuplicateGeometry: return OperatorNode{KindOf<Duplicate>};
        case ExtrudeVertices:
        case ExtrudeEdges:
        case ExtrudeFacesIndividual:
            return OperatorNode{KindOf<Extrude>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&Extrude::Mode>(n, t.Op == ExtrudeVertices ? ExtrudeMode::Vertices : t.Op == ExtrudeEdges ? ExtrudeMode::Edges :
                                                                                                                                              ExtrudeMode::FacesIndividual));
                                }};
        case ExtrudeRegion:
            if (!(task.Flags & TopologyFlagTransformCopies)) return OperatorNode{KindOf<Extrude>};
            return OperatorNode{KindOf<ExtrudeRepeat>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&ExtrudeRepeat::Steps>(n, t.Steps));
                                    f.Do(UpdateOf<&ExtrudeRepeat::Offset>(n, t.CopyTranslation));
                                }};
        case Subdivide: return OperatorNode{KindOf<action::mesh::Subdivide>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(n, uint32_t(t.Param0))); }};
        case EdgeSplit: return OperatorNode{task.Flags & TopologyFlagRipSelectCopies ? KindOf<Rip> : KindOf<action::mesh::EdgeSplit>};
        case Poke: return OperatorNode{KindOf<action::mesh::Poke>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&action::mesh::Poke::Offset>(n, t.Param0)); }};
        case InsetRegion:
        case InsetIndividual:
            return OperatorNode{KindOf<Inset>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&Inset::Thickness>(n, t.Param0));
                                    f.Do(UpdateOf<&Inset::Depth>(n, t.Param1));
                                    f.Do(UpdateOf<&Inset::Individual>(n, t.Op == InsetIndividual));
                                    f.Do(UpdateOf<&Inset::Even>(n, (t.Flags & TopologyFlagEvenOffset) != 0u));
                                }};
        case Solidify: return OperatorNode{KindOf<action::mesh::Solidify>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&action::mesh::Solidify::Thickness>(n, t.Param0)); }};
        case ConnectVertices: return OperatorNode{KindOf<action::mesh::ConnectVertices>};
        case SplitNonplanarFaces: return OperatorNode{KindOf<action::mesh::SplitNonplanarFaces>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&action::mesh::SplitNonplanarFaces::Angle>(n, t.Param0)); }};
        case SplitConcaveFaces: return OperatorNode{KindOf<action::mesh::SplitConcaveFaces>};
        case ReplaceFaces: return OperatorNode{KindOf<BeautifyFaces>};
        default: return std::nullopt;
    }
}
// Whether a node of the kind runs one topology task, as the kinds NodeOf returns do.
bool RunsTask(uint16_t kind) {
    using namespace action::mesh;
    static constexpr std::array Kinds{KindOf<Delete>, KindOf<Dissolve>, KindOf<Merge>, KindOf<action::mesh::Decimate>, KindOf<ConvexHull>, KindOf<action::mesh::Wireframe>, KindOf<Bevel>, KindOf<action::mesh::TrisToQuads>, KindOf<action::mesh::Triangulate>, KindOf<action::mesh::FlipNormals>, KindOf<Split>, KindOf<Duplicate>, KindOf<Extrude>, KindOf<ExtrudeRepeat>, KindOf<action::mesh::Subdivide>, KindOf<action::mesh::EdgeSplit>, KindOf<Rip>, KindOf<action::mesh::Poke>, KindOf<Inset>, KindOf<action::mesh::Solidify>, KindOf<action::mesh::ConnectVertices>, KindOf<action::mesh::SplitNonplanarFaces>, KindOf<action::mesh::SplitConcaveFaces>, KindOf<BeautifyFaces>};
    return std::ranges::contains(Kinds, kind);
}
// The live elements of the record that the inline selection selects, ascending, where each domain's masks follow the record's blocks in order.
GeometrySelection LiveHandles(const MeshStore &meshes, uint32_t id, const InlineSelection &selection) {
    GeometrySelection handles;
    for (const auto [element, masks, listed] : {std::tuple{Element::Vertex, &selection.Vertices, &handles.Vertices}, std::tuple{Element::Edge, &selection.Edges, &handles.Edges}, std::tuple{Element::Face, &selection.Faces, &handles.Faces}}) {
        for (uint32_t local = 0u; local < masks->size(); ++local) {
            for (uint32_t slot = 0u; slot < MeshElementBlockSize; ++slot) {
                const auto handle = local * MeshElementBlockSize + slot;
                if ((*masks)[local][slot / 32u] >> (slot % 32u) & 1u && meshes.IsLiveElement(id, element, ElementHandle{handle})) listed->push_back(handle);
            }
        }
    }
    return handles;
}
// The task a node of a RunsTask kind runs on record `id`, planned from its parameters and inline selection as its cook plans it, or none where the cook faults the node.
std::optional<MeshTopologyTask> TaskOf(const state::Registry &r, state::Entity node, uint16_t kind, uint32_t id) {
    using enum MeshTopologyOp;
    using namespace action::mesh;
    const auto &meshes = r.Resources.get<const MeshStore>();
    const Mesh mesh{meshes, id};
    const auto *inline_selection = r.try_get<const InlineSelection>(node);
    const bool masked = inline_selection && !(inline_selection->Vertices.empty() && inline_selection->Edges.empty() && inline_selection->Faces.empty());
    const auto domain = inline_selection ? inline_selection->Domain : Element::Vertex;
    const auto selection = masked ? LiveHandles(meshes, id, *inline_selection) : AllHandles(mesh);
    const auto task = [&](MeshTopologyOp op, float param0 = 0.f, float param1 = 0.f, uint32_t flags = 0u) {
        MeshTopologyTask planned{.SourceId = id, .Op = op, .Param0 = param0, .Param1 = param1, .Flags = flags};
        if (masked) planned.Selection = selection;
        else planned.Flags |= TopologyFlagSelectAll;
        return planned;
    };
    std::optional<MeshTopologyTask> planned;
    VisitKind(kind, [&]<typename K> {
        if constexpr (std::same_as<K, Delete>) {
            planned = task(MeshTopologyOp(uint32_t(r.get<const K>(node).Mode)));
        } else if constexpr (std::same_as<K, Dissolve>) {
            const auto &p = r.get<const K>(node);
            if (p.Mode == DissolveMode::Vertices) planned = task(DissolveVertices);
            else if (p.Mode == DissolveMode::Edges) planned = task(DissolveEdges, 0.f, 0.f, p.KeepVertices ? TopologyFlagKeepVertices : 0u);
            else if (p.Mode == DissolveMode::Faces) planned = task(DissolveFaces);
            else if (p.Mode == DissolveMode::Degenerate) planned = task(DissolveDegenerate, std::max(p.Distance, 0.f));
            else {
                const auto flags = (domain == Element::Face ? TopologyFlagFaceSelection : 0u) | (p.AllBoundaries ? TopologyFlagAllBoundaries : 0u) | (p.DelimitMaterials ? TopologyFlagDelimitMaterial : 0u) | (p.DelimitSharpEdges ? TopologyFlagDelimitSharp : 0u) | (p.DelimitUVs ? TopologyFlagDelimitUV : 0u);
                planned = task(DissolveLimited, std::clamp(p.Angle, 0.f, .5f * std::numbers::pi_v<float>), 0.f, flags);
            }
        } else if constexpr (std::same_as<K, Merge>) {
            const auto &p = r.get<const K>(node);
            const auto &vertices = selection.Vertices;
            const auto reference = vertices.empty() ? InvalidOffset : p.Mode == MergeMode::Last ? vertices.back() :
                                                                                                  vertices.front();
            planned = MergeTask(meshes, mesh, selection, p.Mode, std::max(p.Distance, 0.f), reference);
        } else if constexpr (std::same_as<K, action::mesh::Decimate>) {
            planned = DecimateTask(meshes, mesh, selection, r.get<const K>(node).Ratio);
        } else if constexpr (std::same_as<K, ConvexHull>) {
            planned = ConvexHullTask(meshes, mesh, selection);
            if (planned && planned->Op == AddPrimitives) planned->Selection = selection;
        } else if constexpr (std::same_as<K, action::mesh::Wireframe>) {
            const auto &p = r.get<const K>(node);
            const auto flags = (p.Even ? TopologyFlagEvenOffset : 0u) | (p.Boundary ? TopologyFlagWireBoundary : 0u) | (p.Relative ? TopologyFlagWireRelative : 0u) | (p.Replace ? TopologyFlagWireReplace : 0u);
            planned = task(MeshTopologyOp::Wireframe, std::max(p.Thickness, 0.f), std::clamp(p.Offset, -1.f, 1.f), flags);
        } else if constexpr (std::same_as<K, Bevel>) {
            const auto &p = r.get<const K>(node);
            planned = task(p.Vertices ? BevelVertices : BevelEdges, std::max(p.Width, 0.f));
            planned->Steps = std::max(p.Segments, 1u);
        } else if constexpr (std::same_as<K, action::mesh::TrisToQuads>) {
            planned = task(MeshTopologyOp::TrisToQuads);
        } else if constexpr (std::same_as<K, action::mesh::Triangulate>) {
            planned = task(MeshTopologyOp::Triangulate);
        } else if constexpr (std::same_as<K, action::mesh::FlipNormals>) {
            planned = task(MeshTopologyOp::FlipNormals);
        } else if constexpr (std::same_as<K, Split>) {
            planned = task(SplitGeometry);
        } else if constexpr (std::same_as<K, Duplicate>) {
            planned = task(DuplicateGeometry);
        } else if constexpr (std::same_as<K, Extrude>) {
            const auto mode = r.get<const K>(node).Mode;
            planned = task(mode == ExtrudeMode::Vertices ? ExtrudeVertices : mode == ExtrudeMode::Edges ? ExtrudeEdges :
                               mode == ExtrudeMode::FacesIndividual                                     ? ExtrudeFacesIndividual :
                                                                                                          ExtrudeRegion);
        } else if constexpr (std::same_as<K, ExtrudeRepeat>) {
            const auto &p = r.get<const K>(node);
            planned = ExtrudeStepsTask(mesh, selection, Element::None, p.Steps, mat3{1.f}, p.Offset, vec3{0.f});
        } else if constexpr (std::same_as<K, action::mesh::Subdivide>) {
            planned = task(MeshTopologyOp::Subdivide, float(std::max(r.get<const K>(node).Cuts, 1u)));
        } else if constexpr (std::same_as<K, action::mesh::EdgeSplit>) {
            planned = task(MeshTopologyOp::EdgeSplit);
        } else if constexpr (std::same_as<K, Rip>) {
            planned = task(MeshTopologyOp::EdgeSplit, 0.f, 0.f, TopologyFlagRipSelectCopies);
        } else if constexpr (std::same_as<K, action::mesh::Poke>) {
            planned = task(MeshTopologyOp::Poke, r.get<const K>(node).Offset);
        } else if constexpr (std::same_as<K, Inset>) {
            const auto &p = r.get<const K>(node);
            planned = task(p.Individual ? InsetIndividual : InsetRegion, std::max(p.Thickness, 0.f), p.Depth, p.Even ? TopologyFlagEvenOffset : 0u);
        } else if constexpr (std::same_as<K, action::mesh::Solidify>) {
            planned = task(MeshTopologyOp::Solidify, r.get<const K>(node).Thickness);
        } else if constexpr (std::same_as<K, action::mesh::ConnectVertices>) {
            planned = task(MeshTopologyOp::ConnectVertices);
        } else if constexpr (std::same_as<K, action::mesh::SplitNonplanarFaces>) {
            planned = task(MeshTopologyOp::SplitNonplanarFaces, std::clamp(r.get<const K>(node).Angle, 0.f, std::numbers::pi_v<float>));
        } else if constexpr (std::same_as<K, action::mesh::SplitConcaveFaces>) {
            planned = task(MeshTopologyOp::SplitConcaveFaces);
        } else if constexpr (std::same_as<K, BeautifyFaces>) {
            planned = BeautifyFaceTask(mesh, selection, r.get<const K>(node).Method == BeautifyMethod::Angle);
        }
    });
    return planned;
}

// Each procedure mesh whose Output reads a path that runs only topology tasks on its Source node draws what those tasks make of a clone of its source record, where a faulted node passes its input through.
void ProcedureFixture::CheckStep() {
    auto &meshes = Store();
    bool cloned = false;
    for (const auto [mesh, procedure] : R.view<const GeometryProcedure>().each()) {
        const auto *plan = R.try_get<const CookPlan>(mesh);
        if (!plan || procedure.OutputNode == NoNode) continue;
        const auto shown = procedure.Inputs[plan->InputFirst[procedure.OutputNode]].FromNode;
        if (shown == NoNode) continue;
        const auto upstream = Upstream(procedure, plan->InputFirst, shown, [](uint32_t) { return true; });
        bool sourced = false, runs = true;
        std::vector<NodeRecord> path;
        for (const auto n : plan->Order) {
            if (!upstream[n]) continue;
            const auto kind = procedure.Nodes[n].Kind;
            if (kind == KindOf<procedure::Source> && !sourced) sourced = true;
            else if (RunsTask(kind)) path.push_back(procedure.Nodes[n]);
            else runs = false;
        }
        if (!runs || !sourced || path.empty()) continue;
        const auto id = Clone(Source(mesh));
        cloned = true;
        for (const auto [node, kind] : path) {
            if (R.all_of<NodeError>(node)) continue;
            const auto task = TaskOf(R, node, kind, id);
            expect(task.has_value());
            if (task) ExecuteGeometryTopology(R, std::span{&*task, 1u});
        }
        expect(Describe(meshes, id) == Shaped(mesh));
        meshes.Release(id);
    }
    // The present node holds mesh storage as it stood before the clones.
    if (cloned) P->History.Revert();
}

// The output must be the kernel's, from a record of its own, with the source record unchanged.
void ExpectCooksLike(ProcedureFixture &f, state::Entity mesh, uint32_t source_id, const Shape &source, const Shape &reference) {
    expect(f.Shaped(mesh) == reference);
    expect(f.Output(mesh) != source_id && f.Source(mesh) == source_id);
    expect(Describe(f.Store(), source_id) == source);
}

// Every operator kind cooks like its kernel on the patchwork, each generator kind like its primitive, and a fan-out cooks the read branch alone.
void OperatorKinds(ProcedureFixture &f, Fixture &other) {
    constexpr std::array<std::string_view, 2> WithoutNode{"RotateEdges", "KeepSelectedFaces"};
    for (const auto &c : TopologyCases) {
        if (std::ranges::contains(WithoutNode, c.Name)) continue;
        Scenario(f, other, std::string{c.Name}, [&c](ProcedureFixture &f) {
            auto &r = f.R;
            const auto &meshes = f.Store();
            const auto mesh = AddMesh(r, CreateMesh(r, {.Data = Patchwork()}).StoreId, MeshInstanceCreateInfo{}).first;
            f.Commit("Add patchwork");
            const auto source_id = f.Source(mesh);
            const auto source = Describe(meshes, source_id);
            auto task = PlanCase(c, meshes, Mesh{meshes, source_id});
            expect(fatal(task.has_value()));
            const auto node = NodeOf(*task);
            expect(fatal(node.has_value()));
            f.Do(AddProcedure{mesh});
            const auto [index, entity] = f.Append(mesh, node->Kind);
            if (node->Set) node->Set(f, entity, *task);
            // The region as one mask block per block the source owns in each domain, as CaptureSelection captures a drawn selection.
            const auto region = Region(Mesh{meshes, source_id});
            auto selection = std::make_unique<InlineSelection>();
            using D = MeshStore::ElementDomain;
            for (const auto [domain, handles, blocks] : {std::tuple{D::Vertex, &region.Vertices, &selection->Vertices}, std::tuple{D::Edge, &region.Edges, &selection->Edges}, std::tuple{D::Face, &region.Faces, &selection->Faces}}) {
                for (const auto block : meshes.GetBlockList(source_id, domain).Blocks) {
                    auto &words = blocks->emplace_back();
                    for (const auto h : *handles)
                        if (h / MeshElementBlockSize == block) words[(h % MeshElementBlockSize) / 32u] |= 1u << (h % 32u);
                }
            }
            f.Do(SetInlineSelection{entity, std::move(selection)});
            f.Route(mesh, {0u, index, 1u});
            expect(f.Output(mesh) != source_id && f.Source(mesh) == source_id && Describe(meshes, source_id) == source);
            if (task->Param0 == 0.f) return;
            task->Param0 *= 2.f;
            node->Set(f, entity, *task);
            expect(f.Output(mesh) != source_id && f.Source(mesh) == source_id && Describe(meshes, source_id) == source);
        });
    }
    // Generator kinds follow PrimitiveShape's alternatives.
    for (const PrimitiveShape shape : {PrimitiveShape{primitive::Plane{}}, PrimitiveShape{primitive::Circle{}}, PrimitiveShape{primitive::Cuboid{}}, PrimitiveShape{primitive::IcoSphere{}}, PrimitiveShape{primitive::UVSphere{}}, PrimitiveShape{primitive::Torus{}}, PrimitiveShape{primitive::Cylinder{}}, PrimitiveShape{primitive::Cone{}}}) {
        Scenario(f, other, ToString(shape), [shape](ProcedureFixture &f) {
            auto &meshes = f.Store();
            const auto mesh = f.AddMesh(primitive::Plane{});
            const auto source_id = f.Source(mesh);
            const auto source = Describe(meshes, source_id);
            f.Do(AddProcedure{mesh});
            const auto generator = f.Append(mesh, uint16_t(KindOf<primitive::Plane> + shape.index())).first;
            const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>).first;
            f.Route(mesh, {generator, subdivide, 1u});
            const auto generated = CreateMesh(f.R, {.Data = primitive::CreateMesh(shape), .FlatShaded = true}).StoreId;
            const auto reference = SelectAllReference(f, generated, {Op(MeshTopologyOp::Subdivide, 1.f)});
            meshes.Release(generated);
            f.Commit("Release generated");
            ExpectCooksLike(f, mesh, source_id, source, reference);
        });
    }
    Scenario(f, other, "FanOut", [](ProcedureFixture &f) {
        const auto mesh = f.AddMesh(primitive::Plane{});
        const auto source_id = f.Source(mesh);
        const auto source = Describe(f.Store(), source_id);
        f.Do(AddProcedure{mesh});
        const auto [first, first_entity] = f.Append(mesh, KindOf<action::mesh::Extrude>);
        const auto [second, second_entity] = f.Append(mesh, KindOf<action::mesh::Extrude>);
        f.Do(Link{mesh, 0u, 0u, first, 0u});
        f.Do(Link{mesh, 0u, 0u, second, 0u});
        for (const auto read : {first, second}) {
            f.Do(Link{mesh, read, 2u, 1u, 0u});
            expect(!f.R.any_of<NodeError>(first_entity) && !f.R.any_of<NodeError>(second_entity));
            expect(f.Output(mesh) != source_id && f.Source(mesh) == source_id && Describe(f.Store(), source_id) == source);
        }
    });
}

// A graph on `Mesh`, an oracle for its output beyond the kernels each step checks, and an edit that changes a copy's output.
struct Graph {
    state::Entity Mesh;
    std::function<void(ProcedureFixture &)> Oracle{};
    std::function<void(ProcedureFixture &, state::Entity)> Edit{};
};
struct GraphRow {
    std::string_view Name;
    Graph (*Build)(ProcedureFixture &);
};

// Each graph's output matches its oracle, a duplicate cooks the same and edits apart, and deleting and applying the procedure leave its output.
void Survives(ProcedureFixture &f, Fixture &other, std::span<const GraphRow> rows) {
    for (const auto &row : rows) {
        Scenario(f, other, std::string{row.Name}, [&row](ProcedureFixture &f) {
            auto &r = f.R;
            const auto graph = row.Build(f);
            const auto mesh = graph.Mesh;
            if (graph.Oracle) graph.Oracle(f);
            if (!graph.Edit) return;
            const auto output = f.Shaped(mesh);
            f.Do(action::selection::Select{f.Object(mesh)});
            f.Do(action::object::Duplicate{});
            const auto copy = r.get<const Instance>(FindActiveEntity(r)).Entity;
            expect(copy != mesh && f.Output(copy) != f.Output(mesh) && f.Shaped(copy) == output);
            graph.Edit(f, copy);
            expect(f.Shaped(copy) != output && f.Shaped(mesh) == output);
            f.Undo();
            f.Undo();
            f.Do(action::object::Delete{});
            expect(!r.valid(mesh) && r.view<const GeometryProcedure>().empty());
            f.Undo();
            expect(f.Shaped(mesh) == output);
            f.Do(ApplyProcedure{mesh});
            expect(!r.any_of<GeometryProcedure, EvaluatedMesh, ResyncState, CookPlan>(mesh));
            expect(Describe(f.Store(), f.Source(mesh)) == output);
        });
    }
}

Graph ExtrudeSubdivide(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Plane{});
    f.Do(AddProcedure{mesh});
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>).first;
    const auto [subdivide, subdivide_entity] = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Route(mesh, {0u, extrude, subdivide, 1u});
    const auto once = f.Shaped(mesh);
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide_entity, 2u));
    expect(f.Shaped(mesh) != once);
    return {
        mesh,
        [mesh](ProcedureFixture &f) { expect(f.Shaped(mesh) != Describe(f.Store(), f.Source(mesh))); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 3u), 3u)); },
    };
}

// A sphere's cap above y = 0.9 insets through Position, Compare and Inset, and follows a moved source vertex.
Graph CapResync(ProcedureFixture &f) {
    auto &r = f.R;
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::UVSphere{.Slices = 128u, .Stacks = 64u});
    f.Do(AddProcedure{mesh});
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [compare, compare_entity] = f.Append(mesh, KindOf<Compare>);
    const auto [inset, inset_entity] = f.Append(mesh, KindOf<action::mesh::Inset>);
    f.Do(UpdateOf<&Compare::Component>(compare_entity, VectorComponent::Y));
    f.Do(UpdateOf<&Compare::B>(compare_entity, .9f));
    f.Do(Link{mesh, position, 0u, compare, 0u});
    f.Do(Link{mesh, compare, f.Out(mesh, compare), inset, 1u});
    f.Route(mesh, {0u, inset, 1u});
    const auto first = f.Shaped(mesh);
    const auto set = [&](float thickness) {
        f.Do(UpdateOf<&action::mesh::Inset::Thickness>(inset_entity, thickness));
        return f.Shaped(mesh);
    };
    set(.05f);
    set(.2f);
    const auto thin = set(.01f);
    expect(set(action::mesh::Inset{}.Thickness) == first);
    f.Undo();
    expect(f.Shaped(mesh) == thin);
    set(.03f);
    set(.06f);
    const auto source = f.Source(mesh);
    const PositionOperationTarget target{.StoreId = source, .Selection = GeometrySelection{.Vertices = {**Mesh{meshes, source}.vertices().begin()}}};
    for (const auto offset : {vec3{0.f, -.5f, 0.f}, vec3{.3f, 0.f, .1f}}) {
        ExecuteGeometryPositions(r, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = offset}}});
        r.emplace_or_replace<MeshPositionsChanged>(mesh);
        f.Commit("Move source vertex");
    }
    return {
        mesh,
        // Compare output against the kernel insetting the moved source cap on a clone.
        [mesh](ProcedureFixture &f) {
            const auto kernel = Reference(f, f.Source(mesh), [&](uint32_t clone) {
                const Mesh sphere{f.Store(), clone};
                const auto cap = SelectionWhere(sphere, [&](he::VH v) { return sphere.GetPosition(v).y > .9f; });
                f.Run(clone, {{.Op = MeshTopologyOp::InsetRegion, .Param0 = .06f, .Flags = TopologyFlagEvenOffset, .Selection = cap}});
                return GeometrySelection{};
            });
            expect(f.Shaped(mesh) == kernel.Output);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Inset::Thickness>(f.Node(mesh, 4u), .2f)); },
    };
}

// Vector Math feeding Set Position doubles vertex positions.
Graph ScalePositions(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [math, math_entity] = f.Append(mesh, KindOf<VectorMath>);
    const auto [set, set_entity] = f.Append(mesh, KindOf<SetPosition>);
    f.Do(UpdateOf<&VectorMath::Operation>(math_entity, VectorMathOp::Scale));
    f.Do(UpdateOf<&VectorMath::Scale>(math_entity, 2.f));
    f.Do(UpdateOf<&SetPosition::Offset>(set_entity, false));
    f.Do(Link{mesh, position, 0u, math, 0u});
    f.Do(Link{mesh, math, f.Out(mesh, math), set, 1u});
    f.Route(mesh, {0u, set, 1u});
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, f.Source(mesh), [](uint32_t, vec3 p) { return 2.f * p; })));
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&VectorMath::Scale>(f.Node(mesh, 3u), 3.f)); },
    };
}

// Position.z compared greater than zero selects the cube's upper face for Extrude.
Graph CompareSelectsExtrusion(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [compare, compare_entity] = f.Append(mesh, KindOf<Compare>);
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>).first;
    f.Do(UpdateOf<&Compare::Component>(compare_entity, VectorComponent::Z));
    f.Do(Link{mesh, position, 0u, compare, 0u});
    f.Do(Link{mesh, compare, f.Out(mesh, compare), extrude, 1u});
    f.Route(mesh, {0u, extrude, 1u});
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            const auto kernel = Reference(f, f.Source(mesh), [&](uint32_t clone) {
                const Mesh cube{meshes, clone};
                const auto upper = SelectionWhere(cube, [&](he::VH v) { return cube.GetPosition(v).z > 0.f; });
                expect(upper.Faces.size() == 1u);
                f.Run(clone, {{.Op = MeshTopologyOp::ExtrudeRegion, .Selection = upper}});
                return GeometrySelection{};
            });
            expect(f.Shaped(mesh) == kernel.Output);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&Compare::Component>(f.Node(mesh, 3u), VectorComponent::X)); },
    };
}

vec3 OffsetOf(uint32_t v) { return {float(v % 7u), 1.f, -float(v % 3u)}; }
// A named vertex attribute offsets each vertex through Set Position.
Graph NamedAttributeReads(ProcedureFixture &f) {
    auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto source = f.Source(mesh);
    const auto offset = meshes.AddAttribute(source, AttributeDomain::Vertex, AttributeType::Vec3, "offset").value();
    for (const auto v : Mesh{meshes, source}.vertices()) {
        const auto value = OffsetOf(*v);
        meshes.SetAttribute<vec3>(offset, *v, std::span{&value, 1u});
    }
    f.Commit("Add offset");
    f.Do(AddProcedure{mesh});
    const auto [named, named_entity] = f.Append(mesh, KindOf<procedure::NamedAttribute>);
    const auto set = f.Append(mesh, KindOf<SetPosition>).first;
    f.Do(SetAttributeName{named_entity, "offset"});
    f.Do(UpdateOf<&procedure::NamedAttribute::Type>(named_entity, AttributeType::Vec3));
    f.Do(Link{mesh, named, 0u, set, 1u});
    f.Route(mesh, {0u, set, 1u});
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, f.Source(mesh), [](uint32_t v, vec3 p) { return p + OffsetOf(v); })));
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&SetPosition::Offset>(f.Node(mesh, 3u), false)); },
    };
}

// Whether the output's "rest" attribute holds each vertex's position.
bool RestMatches(ProcedureFixture &f, state::Entity mesh) {
    const auto &meshes = f.Store();
    const auto id = f.Output(mesh);
    const auto rest = meshes.FindAttribute(id, "rest");
    if (rest == InvalidOffset) return false;
    const auto view = meshes.AttributeView<vec3>(rest);
    const Mesh output{meshes, id};
    for (const auto v : output.vertices()) {
        const auto d = view[*v] - output.GetPosition(v);
        if (std::abs(d.x) >= 1e-5f || std::abs(d.y) >= 1e-5f || std::abs(d.z) >= 1e-5f) return false;
    }
    return true;
}
// Store Named Attribute keeps Position under its name through a downstream subdivision and a re-cook with more cuts.
Graph StorePromotes(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Plane{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [store, store_entity] = f.Append(mesh, KindOf<StoreNamedAttribute>);
    const auto [subdivide, subdivide_entity] = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Do(SetAttributeName{store_entity, "rest"});
    f.Do(Link{mesh, position, 0u, store, 1u});
    f.Route(mesh, {0u, store, subdivide, 1u});
    expect(RestMatches(f, mesh));
    const auto faces = f.Shaped(mesh).Faces.size();
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide_entity, 2u));
    expect(f.Shaped(mesh).Faces.size() > faces);
    return {
        mesh,
        [mesh](ProcedureFixture &f) { expect(RestMatches(f, mesh)); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 4u), 3u)); },
    };
}

// A Store Selection of the cube's upper face keeps it through a subdivision inserted after the store, so a Delete reading it through Named Attribute removes the four faces it became.
Graph StoreSelectionDelete(ProcedureFixture &f) {
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto [store, store_entity] = f.Append(mesh, KindOf<StoreSelection>);
    const auto [named, named_entity] = f.Append(mesh, KindOf<procedure::NamedAttribute>);
    const auto [remove, remove_entity] = f.Append(mesh, KindOf<action::mesh::Delete>);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>).first;
    f.Do(SetAttributeName{store_entity, "upper"});
    f.Do(SetAttributeName{named_entity, "upper"});
    f.Do(UpdateOf<&procedure::NamedAttribute::Type>(named_entity, AttributeType::Bool));
    f.Do(UpdateOf<&action::mesh::Delete::Mode>(remove_entity, action::mesh::DeleteMode::OnlyFaces));
    // The upper face as a mask over the one face block the source owns.
    const auto source = f.Source(mesh);
    const Mesh cube{meshes, source};
    expect(fatal(meshes.GetBlockList(source, MeshStore::ElementDomain::Face).Blocks.size() == 1u));
    auto selection = std::make_unique<InlineSelection>(InlineSelection{.Domain = Element::Face});
    auto &words = selection->Faces.emplace_back();
    for (const auto face : cube.faces())
        if (cube.CalcFaceCentroid(face).y > .5f) words[*face / 32u] |= 1u << (*face % 32u);
    f.Do(SetInlineSelection{store_entity, std::move(selection)});
    f.Do(Link{mesh, named, 0u, remove, 1u});
    f.Route(mesh, {0u, store, remove, 1u});
    expect(Mesh{meshes, f.Output(mesh)}.FaceCount() == 5u);
    f.Do(InsertNode{mesh, subdivide, remove, 0u, 0u});
    return {
        mesh,
        [mesh, named_entity, remove_entity](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            expect(!f.R.any_of<NodeError>(named_entity) && !f.R.any_of<NodeError>(remove_entity));
            const auto kernel = Reference(f, f.Source(mesh), [&](uint32_t clone) {
                f.Run(clone, {Op(MeshTopologyOp::Subdivide, 1.f, TopologyFlagSelectAll)});
                const Mesh subdivided{meshes, clone};
                GeometrySelection upper;
                for (const auto face : subdivided.faces())
                    if (subdivided.CalcFaceCentroid(face).y > .99f) upper.Faces.push_back(*face);
                expect(upper.Faces.size() == 4u);
                f.Run(clone, {{.Op = MeshTopologyOp(uint32_t(action::mesh::DeleteMode::OnlyFaces)), .Selection = upper}});
                return GeometrySelection{};
            });
            expect(f.Shaped(mesh) == kernel.Output);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 5u), 2u)); },
    };
}

// Vector Math on a vertex and a face attribute errs and forwards A, so Set Position offsets each vertex by its position.
Graph DomainMismatch(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [normal, normal_entity] = f.Append(mesh, KindOf<Normal>);
    const auto [math, math_entity] = f.Append(mesh, KindOf<VectorMath>);
    const auto [set, set_entity] = f.Append(mesh, KindOf<SetPosition>);
    f.Do(UpdateOf<&Normal::Domain>(normal_entity, AttributeDomain::Face));
    f.Do(Link{mesh, position, 0u, math, 0u});
    f.Do(Link{mesh, normal, 0u, math, 1u});
    f.Do(Link{mesh, math, f.Out(mesh, math), set, 1u});
    f.Route(mesh, {0u, set, 1u});
    return {
        mesh,
        [mesh, math_entity](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            expect(f.R.any_of<NodeError>(math_entity));
            expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, f.Source(mesh), [](uint32_t, vec3 p) { return 2.f * p; })));
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&Normal::Domain>(f.Node(mesh, 3u), AttributeDomain::Vertex)); },
    };
}

// A join of the source and a generated cube doubles the elements, and the generated vertices take the source attribute's default.
Graph JoinGenerated(ProcedureFixture &f) {
    auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    const auto source = f.Source(mesh);
    const auto weight = meshes.AddAttribute(source, AttributeDomain::Vertex, AttributeType::Float, "weight", AttributeValue{std::bit_cast<uint32_t>(-1.f), 0u, 0u, 0u}).value();
    for (const auto v : Mesh{meshes, source}.vertices()) {
        const float value = 5.f;
        meshes.SetAttribute<float>(weight, *v, std::span{&value, 1u});
    }
    f.Commit("Add weight");
    f.Do(AddProcedure{mesh});
    const auto [cube, cube_entity] = f.Append(mesh, KindOf<primitive::Cuboid>);
    const auto join = f.Append(mesh, KindOf<Join>).first;
    f.Do(Link{mesh, 0u, 0u, join, 0u});
    f.Do(Link{mesh, cube, f.Out(mesh, cube), join, 1u});
    f.Do(Link{mesh, join, 2u, 1u, 0u});
    const auto joined = f.Shaped(mesh);
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(cube_entity, vec3{2.f, .5f, 1.f}));
    expect(f.Shaped(mesh) != joined);
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            const auto id = f.Output(mesh), source = f.Source(mesh);
            const Mesh joined{meshes, id}, single{meshes, source};
            expect(joined.VertexCount() == 2u * single.VertexCount() && joined.FaceCount() == 2u * single.FaceCount());
            const auto view = meshes.AttributeView<float>(meshes.FindAttribute(id, "weight"));
            std::array<uint32_t, 2> weights{};
            for (const auto v : joined.vertices()) {
                if (view[*v] == 5.f) ++weights[0];
                else if (view[*v] == -1.f) ++weights[1];
            }
            expect(weights[0] == single.VertexCount() && weights[1] == single.VertexCount());
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(f.Node(mesh, 2u), vec3{.5f, .5f, .5f})); },
    };
}

std::map<uint32_t, float> Heat(const MeshStore &meshes, uint32_t id) {
    std::map<uint32_t, float> heat;
    const auto view = meshes.AttributeView<float>(meshes.FindAttribute(id, "heat"));
    meshes.Arenas().Vertices.ForEach(meshes.Get(id).Vertices, [&](ElementHandle v, ElementAddress) { heat.emplace(v.Value, view[v.Value]); });
    return heat;
}
// A named attribute's values travel through an edit-mode subdivision of a plain cube.
// Retained vertices keep their heat, and an edge midpoint takes the mean of two corners and a face center the mean of four.
Graph HeatSubdivide(ProcedureFixture &f) {
    auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    const auto id = f.Source(mesh);
    const auto heat = meshes.AddAttribute(id, AttributeDomain::Vertex, AttributeType::Float, "heat").value();
    const auto vertices = meshes.Arenas().Vertices.Dense(meshes.Get(id).Vertices);
    std::vector<float> values(vertices.Count);
    for (uint32_t i = 0u; i < vertices.Count; ++i) values[i] = float(i * i);
    meshes.SetAttribute<float>(heat, 0u, values);
    f.Commit("Heat cube");
    const auto original = Heat(meshes, id);
    f.EnterEdit(Element::Vertex);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{1u});
    return {mesh, [id, original](ProcedureFixture &f) {
                const auto split = Heat(f.Store(), id);
                expect(split.size() > original.size());
                for (const auto [v, value] : original) expect(split.at(v) == value);
                std::vector<float> corners;
                for (const auto [v, value] : original) corners.push_back(value);
                std::set<float> means;
                for (size_t a = 0u; a < corners.size(); ++a)
                    for (size_t b = a + 1u; b < corners.size(); ++b) {
                        means.insert(.5f * (corners[a] + corners[b]));
                        for (size_t c = b + 1u; c < corners.size(); ++c)
                            for (size_t d = c + 1u; d < corners.size(); ++d) means.insert(.25f * (corners[a] + corners[b] + corners[c] + corners[d]));
                    }
                for (const auto [v, value] : split)
                    if (!original.contains(v)) expect(means.contains(value)) << "vertex" << v << "heat" << value;
            }};
}

// A boolean intersection of the cube with a generated bar keeps the bar's middle.
Graph BooleanIntersection(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto [box, box_entity] = f.Append(mesh, KindOf<primitive::Cuboid>);
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(box_entity, vec3{2.f, .5f, .5f}));
    const auto [boolean, boolean_entity] = f.Append(mesh, KindOf<procedure::Boolean>);
    f.Do(UpdateOf<&procedure::Boolean::Operation>(boolean_entity, BooleanOperation::Intersection));
    f.Route(mesh, {0u, boolean, 1u});
    // An unlinked operand passes the cube through.
    expect(f.Shaped(mesh) == Describe(f.Store(), f.Source(mesh)));
    f.Do(Link{mesh, box, f.Out(mesh, box), boolean, 1u});
    expect(std::abs(test::EnclosedVolume(Mesh{f.Store(), f.Output(mesh)}).value_or(-1.0) - 2.0) < 1e-4);
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(box_entity, vec3{2.f, .25f, .75f}));
    return {
        mesh,
        [mesh](ProcedureFixture &f) { expect(std::abs(test::EnclosedVolume(Mesh{f.Store(), f.Output(mesh)}).value_or(-1.0) - 1.5) < 1e-4); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&procedure::Boolean::Operation>(f.Node(mesh, 3u), BooleanOperation::Union)); },
    };
}

// The boolean action makes the active object's mesh the active cube minus the other selected cube, in the active object's space.
Graph BooleanAction(ProcedureFixture &f) {
    auto &r = f.R;
    const auto target = f.AddMesh(primitive::Cuboid{});
    const auto active = FindActiveEntity(r);
    f.Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = {.P = {1.f, 1.f, 1.f}}, .Select = MeshInstanceCreateInfo::SelectBehavior::Additive})});
    expect(FindActiveEntity(r) == active);
    f.Do(action::mesh::Boolean{BooleanOperation::Difference});
    return {target, [target, active](ProcedureFixture &f) {
                expect(std::abs(test::EnclosedVolume(Mesh{f.Store(), f.Source(target)}).value_or(-1.0) - 7.0) < 1e-4);
                for (const auto e : f.R.view<const Instance>())
                    if (e != active) expect(Mesh{f.Store(), f.Source(f.R.get<const Instance>(e).Entity)}.VertexCount() == 8u);
            }};
}

// Primitive shape edits regenerate the record.
Graph PrimitiveShapeEdit(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(action::UpdateOfShape(mesh, &primitive::Cuboid::HalfExtents, vec3{2.f, .5f, 1.f}));
    f.Do(action::UpdateOfShape(mesh, &primitive::Cuboid::HalfExtents, vec3{1.f, 1.f, 3.f}));
    return {mesh, [mesh](ProcedureFixture &f) { expect(Near(Positions(f.Store(), f.Output(mesh)), ShapePositions(primitive::Cuboid{{1.f, 1.f, 3.f}}))); }};
}

// An Object Info node reading its own procedure's mesh errs and forwards the source.
// One reading a mesh with a procedure of its own cooks after that procedure, so a resize there reaches the join.
Graph ObjectInfo(ProcedureFixture &f) {
    auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Plane{});
    f.Do(AddProcedure{mesh});
    const auto [info, info_entity] = f.Append(mesh, KindOf<procedure::ObjectInfo>);
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(info_entity, f.Object(mesh)));
    const auto [join, join_entity] = f.Append(mesh, KindOf<Join>);
    f.Do(Link{mesh, 0u, 0u, join, 0u});
    f.Do(Link{mesh, info, 0u, join, 1u});
    f.Do(Link{mesh, join, 2u, 1u, 0u});
    expect(r.all_of<NodeError>(info_entity) && !r.any_of<NodeError>(join_entity));
    expect(f.Shaped(mesh) == Describe(f.Store(), f.Source(mesh)));
    const auto cube = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{cube});
    const auto [box, box_entity] = f.Append(cube, KindOf<primitive::Cuboid>);
    f.Do(Link{cube, box, f.Out(cube, box), 1u, 0u});
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(info_entity, f.Object(cube)));
    expect(!r.any_of<NodeError>(info_entity));
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(box_entity, vec3{.5f, .25f, .5f}));
    return {mesh, [mesh](ProcedureFixture &f) {
                auto points = ShapePositions(primitive::Plane{});
                points.append_range(ShapePositions(primitive::Cuboid{{.5f, .25f, .5f}}));
                std::ranges::sort(points);
                expect(Mesh{f.Store(), f.Output(mesh)}.FaceCount() == 7u && Near(Positions(f.Store(), f.Output(mesh)), points));
            }};
}

// Instances on the plane's points publish as entities or realize as geometry, following scale, plane and cube edits.
template<bool Borrowed, bool Realize> Graph Instances(ProcedureFixture &f) {
    auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto object = FindActiveEntity(r);
    const auto cube_mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto to_points = f.Append(mesh, KindOf<MeshToPoints>).first;
    const auto [instancer, instancer_entity] = f.Append(mesh, KindOf<InstanceOnPoints>);
    const auto [instance, instance_entity] = f.Append(mesh, Borrowed ? KindOf<procedure::ObjectInfo> : KindOf<primitive::Cuboid>);
    if constexpr (Borrowed) f.Do(UpdateOf<&procedure::ObjectInfo::Object>(instance_entity, f.Object(cube_mesh)));
    f.Route(mesh, {0u, to_points, instancer});
    f.Do(Link{mesh, instance, f.Out(mesh, instance), instancer, 1u});
    if constexpr (Realize) {
        const auto realizer = f.Append(mesh, KindOf<RealizeInstances>).first;
        f.Route(mesh, {instancer, realizer, 1u});
    } else {
        f.Do(Link{mesh, instancer, f.Out(mesh, instancer), 1u, 0u});
    }
    f.Do(UpdateOf<&InstanceOnPoints::Scale>(instancer_entity, vec3{.5f, .5f, .5f}));
    f.Do(action::UpdateOfShape(mesh, &primitive::Plane::HalfExtents, vec2{2.f, 3.f}));
    if constexpr (Borrowed) f.Do(action::UpdateOfShape(cube_mesh, &primitive::Cuboid::HalfExtents, vec3{.5f, .25f, .5f}));
    else f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(instance_entity, vec3{.5f, .25f, .5f}));
    Graph graph{mesh, [=](ProcedureFixture &f) {
                    const auto &r = f.R;
                    const auto &meshes = f.Store();
                    const auto plane = ShapePositions(primitive::Plane{{2.f, 3.f}}), cube = ShapePositions(primitive::Cuboid{{.5f, .25f, .5f}});
                    if constexpr (Realize) {
                        std::vector<Corner> realized;
                        for (const auto &p : plane)
                            for (const auto &c : cube) realized.push_back({p[0] + .5f * c[0], p[1] + .5f * c[1], p[2] + .5f * c[2]});
                        std::ranges::sort(realized);
                        expect(Near(Positions(meshes, f.Output(mesh)), realized));
                    } else {
                        const auto &entities = r.get<const ProcedureInstances>(mesh).Entities;
                        expect(entities.size() == 4u);
                        std::vector<Corner> points;
                        for (const auto e : entities) {
                            expect(ParentOrNull(r, e) == object);
                            const auto &t = r.get<const Transform>(e);
                            expect(t.S == vec3{.5f, .5f, .5f});
                            points.push_back({t.P.x, t.P.y, t.P.z});
                            expect(Near(Positions(meshes, f.Source(r.get<const Instance>(e).Entity)), cube));
                        }
                        std::ranges::sort(points);
                        expect(Near(points, plane));
                    }
                }};
    if constexpr (Realize) graph.Edit = [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&InstanceOnPoints::Scale>(f.Node(mesh, 3u), vec3{.25f, .25f, .25f})); };
    return graph;
}

// Selecting, duplicating, inserting and deleting nodes leaves the output the chain they form.
Graph NodeEditing(ProcedureFixture &f) {
    auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto [subdivide, subdivide_entity] = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    const auto [triangulate, triangulate_entity] = f.Append(mesh, KindOf<action::mesh::Triangulate>);
    const auto position = f.Append(mesh, KindOf<Position>).first;
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide_entity, 3u));
    const auto source = Describe(f.Store(), f.Source(mesh));
    // Duplicates keep their parameters and only the links among them.
    f.Do(SelectNodes{mesh, {subdivide, triangulate}, subdivide});
    f.Do(DuplicateNodes{mesh, {subdivide, triangulate}});
    const auto &procedure = r.get<const GeometryProcedure>(mesh);
    expect(procedure.Nodes.size() == 7u && r.get<const action::mesh::Subdivide>(procedure.Nodes[5].Entity).Cuts == 3u);
    expect(f.Shaped(mesh) == source);
    // Inserting a node into the output link cooks through it, and a node without a geometry input inserts nothing.
    f.Fails(InsertNode{mesh, position, 1u, 0u, 0u});
    f.Do(SelectNodes{mesh, {subdivide}, subdivide});
    f.Stage(InsertNode{mesh, subdivide, 1u, 0u, 40u});
    f.CommitGesture();
    f.Do(InsertNode{mesh, triangulate, 1u, 0u, 0u});
    // Deleting a node heals the chain through it.
    f.Do(DeleteNodes{mesh, {subdivide}});
    f.Undo();
    f.Do(DeleteNodes{mesh, {subdivide, triangulate}});
    expect(f.Shaped(mesh) == source);
    f.Undo();
    return {mesh};
}

bool Facing(const Mesh &mesh, he::FH face, vec3 eye) { return face && Dot(mesh.GetNormal(face), eye - mesh.CalcFaceCentroid(face)) > 0.f; }
// The midpoints of the edges between two faces looking toward the eye.
std::vector<vec3> FacingEdgeCenters(const Mesh &mesh, vec3 eye) {
    std::vector<vec3> centers;
    for (const auto e : mesh.edges()) {
        const auto h = mesh.GetHalfedge(e, 0u);
        if (Facing(mesh, mesh.GetFace(h), eye) && Facing(mesh, mesh.GetFace(mesh.GetOppositeHalfedge(h)), eye)) centers.push_back(.5f * (mesh.GetPosition(mesh.GetFromVertex(h)) + mesh.GetPosition(mesh.GetToVertex(h))));
    }
    return centers;
}

struct ViewportRow {
    std::string_view Name;
    Element Domain;
    uint32_t Clicks;
    uint16_t Kind;
    std::vector<float> Values;
    void (*Gesture)(ProcedureFixture &, float value, bool first);
    // Run the operator on the clone at the clicked handles and return its selected elements.
    GeometrySelection (*Kernel)(ProcedureFixture &, uint32_t clone, const GeometrySelection &clicked, float value);
};
// Run the topology task on the clone, publish its emitted selection, and return that selection.
GeometrySelection PublishedKernel(ProcedureFixture &f, uint32_t clone, MeshTopologyTask task) {
    task.SourceId = clone;
    PublishStoreSelection(f.Store(), task);
    f.Run(clone, {task});
    return EditorGeometrySelection(f.Store(), clone);
}

// Each viewport edit appends one node cooking like the kernel at the clicked elements, and a drag commits as one history node.
void ViewportOperators(ProcedureFixture &f, Fixture &other) {
    using D = MeshStore::ElementDomain;
    static const std::array<ViewportRow, 3> rows{{
        {"Extrude", Element::Face, 2u, KindOf<action::mesh::Extrude>, {0.f}, [](ProcedureFixture &f, float, bool) { f.Do(action::mesh::Extrude{}); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::ExtrudeRegion, .Selection = clicked}); }},
        {"BevelDrag", Element::Edge, 1u, KindOf<action::mesh::Bevel>, {.05f, .1f, .2f, .3f}, [](ProcedureFixture &f, float width, bool first) {
             if (first) f.Stage(action::mesh::Bevel{.Width = width});
             else f.Stage(UpdateOf<&action::mesh::Bevel::Width>(action::OnSelected{}, width)); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float width) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::BevelEdges, .Param0 = width, .Steps = 1u, .Selection = clicked}); }},
        {"TransformDrag", Element::Vertex, 1u, KindOf<TransformElements>, {.5f, 1.f, 1.5f}, [](ProcedureFixture &f, float z, bool) { f.Stage(action::view::TransformElements{{.P = {0.f, 0.f, z}}}); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float z) {
             const PositionOperationTarget target{.StoreId = clone, .Selection = clicked};
             expect(!ExecuteGeometryPositions(f.R, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {0.f, 0.f, z}}}}).empty());
             return clicked; }},
    }};
    for (const auto &row : rows) {
        Scenario(f, other, std::string{row.Name}, [&row](ProcedureFixture &f) {
            auto &r = f.R;
            const auto &meshes = f.Store();
            const auto domains = [](GeometrySelection &selection) {
                return std::array{std::tuple{D::Vertex, Element::Vertex, &selection.Vertices}, std::tuple{D::Edge, Element::Edge, &selection.Edges}, std::tuple{D::Face, Element::Face, &selection.Faces}};
            };
            const auto mesh = f.AddMesh(primitive::Cuboid{});
            f.Do(AddProcedure{mesh});
            f.EnterEdit(row.Domain);
            f.Render();
            {
                // The row clicks the centers of the first elements of its domain whose faces all look toward the eye.
                const Mesh cube{meshes, f.Output(mesh)};
                const auto eye = f.View().CameraPosition;
                std::vector<vec3> centers;
                if (row.Domain == Element::Vertex) {
                    for (const auto v : cube.vertices())
                        if (std::ranges::all_of(cube.voh_range(v), [&](he::HH h) { return Facing(cube, cube.GetFace(h), eye); })) centers.push_back(cube.GetPosition(v));
                } else if (row.Domain == Element::Edge) {
                    centers = FacingEdgeCenters(cube, eye);
                } else {
                    for (const auto face : cube.faces())
                        if (Facing(cube, face, eye)) centers.push_back(cube.CalcFaceCentroid(face));
                }
                expect(fatal(centers.size() >= row.Clicks));
                for (uint32_t i = 0u; i < row.Clicks; ++i) f.Click(centers[i], i > 0u);
            }
            GeometrySelection ordinals;
            for (const auto [domain, element, list] : domains(ordinals))
                meshes.GetSelectedElements(f.Output(mesh), element).ForEach([&](uint32_t handle) { list->push_back(meshes.LiveElementOrdinal(f.Output(mesh), domain, ElementHandle{handle})); });
            expect(ordinals.Get(row.Domain).size() == row.Clicks);
            const auto source = Describe(meshes, f.Source(mesh));
            const auto start = f.P->History.Present.value();
            const auto node_count = uint32_t(r.get<const GeometryProcedure>(mesh).Nodes.size());
            auto previous = f.Shaped(mesh);
            for (uint32_t i = 0u; i < row.Values.size(); ++i) {
                row.Gesture(f, row.Values[i], i == 0u);
                expect(r.get<const GeometryProcedure>(mesh).Nodes.size() == node_count + 1u);
                const auto shape = f.Shaped(mesh);
                expect(shape != previous);
                previous = shape;
            }
            if (f.P->HasStaged()) f.CommitGesture();
            const auto &procedure = r.get<const GeometryProcedure>(mesh);
            const auto [node, kind] = procedure.Nodes.back();
            expect(procedure.Nodes.size() == node_count + 1u && kind == row.Kind);
            expect(r.all_of<PublishSelection>(node) && !r.any_of<NodeError>(node));
            expect(f.P->History.Nodes[f.P->History.Present.value()].Parent == start);
            const auto selection = EditorGeometrySelection(meshes, f.Output(mesh));
            const auto selected = Describe(meshes, f.Output(mesh), selection);
            expect(meshes.GetSelectionSummary(f.Output(mesh)).SelectedCount == selection.Get(row.Domain).size());
            f.Do(action::view::SetInteractionMode{InteractionMode::Object});
            expect(f.Shaped(mesh) == previous);
            expect(Describe(meshes, f.Source(mesh)) == source);
            const auto kernel = Reference(f, f.Source(mesh), [&](uint32_t clone) {
                GeometrySelection clicked;
                for (const auto [domain, element, list] : domains(clicked)) {
                    for (const auto ordinal : ordinals.Get(element)) list->push_back(meshes.LiveElementAt(clone, domain, ordinal).Value);
                    std::ranges::sort(*list);
                }
                return row.Kernel(f, clone, clicked, row.Values.back());
            });
            expect(kernel.Output == previous);
            expect(kernel.Selected == selected);
        });
    }
}

// A primitive or sourced object in World, a preparation, the edit domain and the elements selected in it, an operator gesture, and the node kind the gesture appends to a procedure mesh.
// A row without Targets selects every element, and a row with Targets leaves World at the identity, where local positions are world positions.
struct InPlaceRow {
    std::string_view Name;
    std::variant<PrimitiveShape, MeshSource (*)()> Shape;
    Element Domain;
    uint16_t Kind;
    void (*Gesture)(ProcedureFixture &);
    std::vector<vec3> (*Targets)(const Mesh &, vec3 eye){};
    Transform World{};
    void (*Prepare)(ProcedureFixture &){};
};
// A two by two grid of quads three units wide in the XZ plane facing +Y, with each vertex's X and Z as its UV and a color of its own.
MeshSource AttributedGrid() {
    MeshSource source;
    auto &uvs = source.Attrs.TexCoords0.emplace();
    auto &colors = source.Attrs.Colors0.emplace();
    source.Attrs.Colors0ComponentCount = 4u;
    for (uint32_t z = 0u; z < 3u; ++z)
        for (uint32_t x = 0u; x < 3u; ++x) {
            source.Data.Positions.push_back({3.f * (float(x) - 1.f), 0.f, 3.f * (float(z) - 1.f)});
            uvs.push_back({float(x), float(z)});
            colors.push_back({float(x) / 2.f, float(z) / 2.f, .25f, 1.f});
        }
    for (uint32_t z = 0u; z < 2u; ++z)
        for (uint32_t x = 0u; x < 2u; ++x) {
            const auto v = z * 3u + x;
            source.Data.AddFace(std::array{v, v + 3u, v + 4u, v + 1u});
        }
    return source;
}
// The sharp edges and flat faces by their corner positions, and each face's corners in loop order from its least position, as position, first UV and color, each sorted.
struct Marks {
    std::vector<std::array<Corner, 2>> SharpEdges;
    std::vector<std::vector<Corner>> FlatFaces;
    std::vector<std::vector<std::array<float, 9>>> Corners;
    bool operator==(const Marks &) const = default;
};
Marks MarksOf(const MeshStore &meshes, uint32_t id) {
    const Mesh mesh{meshes, id};
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const auto point = [&](he::VH v) {
        const auto p = mesh.GetPosition(v);
        return Corner{p.x, p.y, p.z};
    };
    Marks marks;
    for (const auto e : mesh.edges()) {
        if (!a.EdgeSharpness.Get({test::Address(mesh, e), 1u})[0]) continue;
        const auto h = mesh.GetHalfedge(e, 0u);
        const auto from = point(mesh.GetFromVertex(h)), to = point(mesh.GetToVertex(h));
        marks.SharpEdges.push_back({std::min(from, to), std::max(from, to)});
    }
    const bool uvs = record.CornerAttributes & MeshAttributeBit_TexCoord0, colors = record.CornerAttributes & MeshAttributeBit_Color0;
    for (const auto face : mesh.faces()) {
        auto &corners = marks.Corners.emplace_back();
        for (const auto h : mesh.fh_range(face)) {
            const auto p = mesh.GetPosition(mesh.GetToVertex(h));
            const auto uv = uvs ? a.CornerUvs[0].Get(record.CornerUvs[0], *h) : vec2{};
            const auto color = colors ? a.CornerColors.Get(record.CornerColors, *h) : vec4{};
            corners.push_back({p.x, p.y, p.z, uv.x, uv.y, color.x, color.y, color.z, color.w});
        }
        std::ranges::rotate(corners, std::ranges::min_element(corners));
        if (!a.FaceSharpness.Get({test::Address(mesh, face), 1u})[0]) continue;
        auto &flat = marks.FlatFaces.emplace_back();
        for (const auto v : mesh.fv_range(face)) flat.push_back(point(v));
        std::ranges::sort(flat);
    }
    std::ranges::sort(marks.SharpEdges);
    std::ranges::sort(marks.FlatFaces);
    std::ranges::sort(marks.Corners);
    return marks;
}
// The centroids of the first face and a face sharing an edge with it.
std::vector<vec3> AdjacentFaceCenters(const Mesh &mesh, vec3) {
    const auto first = *mesh.faces().begin();
    for (const auto h : mesh.fh_range(first)) {
        if (const auto opposite = mesh.GetOppositeHalfedge(h); opposite && mesh.GetFace(opposite)) return {mesh.CalcFaceCentroid(first), mesh.CalcFaceCentroid(mesh.GetFace(opposite))};
    }
    return {};
}
// The centroid of the first face looking toward the eye.
std::vector<vec3> FacingFaceCenter(const Mesh &mesh, vec3 eye) {
    for (const auto face : mesh.faces()) {
        if (Facing(mesh, face, eye)) return {mesh.CalcFaceCentroid(face)};
    }
    return {};
}
// The midpoint of the first edge between two faces looking toward the eye.
std::vector<vec3> FacingEdgeCenter(const Mesh &mesh, vec3 eye) { return FacingEdgeCenters(mesh, eye) | std::views::take(1) | std::ranges::to<std::vector>(); }
// Two edges meeting at a cuboid's corner nearest the eye, which form a chain of three vertices.
std::vector<vec3> CornerEdgeCenters(const Mesh &mesh, vec3 eye) { return FacingEdgeCenters(mesh, eye) | std::views::take(2) | std::ranges::to<std::vector>(); }
// The first and third equator vertices of a sphere about the origin facing the eye, in order around the equator, with one vertex between them.
std::vector<vec3> EquatorVertices(const Mesh &mesh, vec3 eye) {
    std::vector<vec3> equator;
    for (const auto v : mesh.vertices()) {
        const auto p = mesh.GetPosition(v);
        if (std::abs(p.y) < 1e-4f && Dot(p, eye - p) > 0.f) equator.push_back(p);
    }
    if (equator.size() < 3u) return {};
    std::ranges::sort(equator, {}, [](vec3 p) { return std::atan2(p.z, p.x); });
    return {equator[0], equator[2]};
}

// Two adjacent corners of the first face looking toward the eye, the lower handle first.
std::vector<vec3> FacingSide(const Mesh &mesh, vec3 eye) {
    for (const auto face : mesh.faces()) {
        if (!Facing(mesh, face, eye)) continue;
        auto corners = mesh.fv_range(face) | std::views::take(2) | std::ranges::to<std::vector>();
        std::ranges::sort(corners);
        return {mesh.GetPosition(corners[0]), mesh.GetPosition(corners[1])};
    }
    return {};
}
// Two opposite corners of the first quad looking toward the eye.
std::vector<vec3> FacingDiagonal(const Mesh &mesh, vec3 eye) {
    for (const auto face : mesh.faces()) {
        if (!Facing(mesh, face, eye) || mesh.GetValence(face) != 4u) continue;
        const auto corners = mesh.fv_range(face) | std::ranges::to<std::vector>();
        return {mesh.GetPosition(corners[0]), mesh.GetPosition(corners[2])};
    }
    return {};
}
// The midpoints of the edges with one face.
std::vector<vec3> BoundaryEdgeCenters(const Mesh &mesh, vec3) {
    std::vector<vec3> centers;
    for (const auto e : mesh.edges()) {
        const auto h = mesh.GetHalfedge(e, 0u), opposite = mesh.GetOppositeHalfedge(h);
        if (bool(mesh.GetFace(h)) != (opposite && mesh.GetFace(opposite))) centers.push_back(.5f * (mesh.GetPosition(mesh.GetFromVertex(h)) + mesh.GetPosition(mesh.GetToVertex(h))));
    }
    return centers;
}
// The midpoints of the first edge and of its reflection through the origin.
std::vector<vec3> OppositeEdgeCenters(const Mesh &mesh, vec3) {
    const auto h = mesh.GetHalfedge(*mesh.edges().begin(), 0u);
    const auto center = .5f * (mesh.GetPosition(mesh.GetFromVertex(h)) + mesh.GetPosition(mesh.GetToVertex(h)));
    return {center, -center};
}

// Shades every face smooth.
void ShadedSmooth(ProcedureFixture &f) { f.Do(action::object::SetSelectedSmoothShading{true}); }
// Randomly offsets every vertex.
template<float Amount> void Randomized(ProcedureFixture &f) {
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Randomize{.Amount = Amount, .Seed = 3u});
}
// Subdivides every edge three times.
void Subdivided(ProcedureFixture &f) {
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{.Cuts = 3u});
}
// Deletes every face, leaving the edges.
void Wires(ProcedureFixture &f) {
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Delete{.Mode = action::mesh::DeleteMode::OnlyFaces});
}
// Dissolves five consecutive triangles of an eight-segment disc into one polygon, concave at the disc's center.
void Notched(ProcedureFixture &f) {
    f.Render();
    const Mesh mesh{f.Store(), f.Output(f.R.get<const Instance>(FindActiveEntity(f.R)).Entity)};
    std::vector<vec3> centers;
    for (const auto face : mesh.faces()) centers.push_back(mesh.CalcFaceCentroid(face));
    std::ranges::sort(centers, {}, [](vec3 p) { return std::atan2(p.z, p.x); });
    for (uint32_t i = 0u; i < 5u; ++i) f.Click(centers[i], i > 0u);
    f.Do(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Faces});
}
// Triangulates every face and randomly offsets every vertex.
void TriangulatedRandomized(ProcedureFixture &f) {
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Triangulate{});
    f.Do(action::mesh::Randomize{.Amount = .3f, .Seed = 3u});
}
// Deletes the first edge between two faces looking toward the eye with its faces, which opens a boundary loop.
void Opened(ProcedureFixture &f) {
    f.Render();
    const auto mesh = f.R.get<const Instance>(FindActiveEntity(f.R)).Entity;
    f.Click(FacingEdgeCenter(Mesh{f.Store(), f.Output(mesh)}, f.View().CameraPosition).front(), false);
    f.Do(action::mesh::Delete{.Mode = action::mesh::DeleteMode::Edges});
}

// The node each operator appends to a procedure mesh cooks what the operator makes of the plain mesh in place at the same selection, with the same sharp marks, UVs and colors.
// Operators that edit in their mesh's world frame place the object at a rotated and unevenly scaled transform.
void InPlaceOperators(ProcedureFixture &f, Fixture &other) {
    using namespace action::mesh;
    static const Transform Placed{.P = {.5f, .25f, -.5f}, .R = numeric::AngleAxis(.4f, Normalize(vec3{1.f, 2.f, 3.f})), .S = {1.f, 1.5f, .75f}};
    static const std::array<InPlaceRow, 39> rows{{
        {"LimitedDissolveFaces", primitive::Circle{.Radius = 5.f, .Segments = 8u}, Element::Face, KindOf<Dissolve>, [](ProcedureFixture &f) { f.Do(Dissolve{.Mode = DissolveMode::Limited}); }, AdjacentFaceCenters},
        {"ShrinkFattenFace", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Face, KindOf<ShrinkFatten>, [](ProcedureFixture &f) { f.Do(ShrinkFatten{.Distance = .5f}); }, FacingFaceCenter},
        {"SpaceEvenly", primitive::Cuboid{{1.f, 2.f, 3.f}}, Element::Edge, KindOf<SpaceEvenly>, [](ProcedureFixture &f) { f.Do(SpaceEvenly{}); }, CornerEdgeCenters},
        {"RelaxEdgeLoops", primitive::Cuboid{{1.f, 2.f, 3.f}}, Element::Edge, KindOf<RelaxEdgeLoops>, [](ProcedureFixture &f) { f.Do(RelaxEdgeLoops{}); }, CornerEdgeCenters},
        {"CurveBetweenSelected", primitive::UVSphere{.Radius = 5.f, .Slices = 8u, .Stacks = 4u}, Element::Vertex, KindOf<CurveBetweenSelected>, [](ProcedureFixture &f) { f.Do(CurveBetweenSelected{}); }, EquatorVertices},
        {"Circularize", primitive::Cuboid{{1.f, 2.f, 3.f}}, Element::Face, KindOf<Circularize>, [](ProcedureFixture &f) { f.Do(Circularize{}); }, FacingFaceCenter},
        {"MakePlanarFaces", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Face, KindOf<MakePlanarFaces>, [](ProcedureFixture &f) { f.Do(MakePlanarFaces{}); }, {}, {}, Randomized<.3f>},
        {"FlattenView", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::Flatten>, [](ProcedureFixture &f) { f.Do(action::mesh::Flatten{.Method = FlattenMethod::View, .ViewNormal = Normalize(vec3{1.f, 2.f, 2.f})}); }, {}, Placed},
        {"ToSphere", primitive::Cone{.Radius = 2.f, .Height = 4.f, .Slices = 8u}, Element::Vertex, KindOf<procedure::ToSphere>, [](ProcedureFixture &f) { f.Do(action::mesh::ToSphere{}); }, {}, Placed},
        {"PushPull", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::PushPull>, [](ProcedureFixture &f) { f.Do(action::mesh::PushPull{.Distance = .5f}); }, {}, Placed},
        {"ShearLocal", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::Shear>, [](ProcedureFixture &f) { f.Do(action::mesh::Shear{.Local = true}); }, {}, Placed},
        {"Warp", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::Warp>, [](ProcedureFixture &f) { f.Do(action::mesh::Warp{.Angle = 1.f, .Center = {.5f, 0.f, 0.f}, .Orientation = std::make_unique<quat>(numeric::AngleAxis(.3f, Normalize(vec3{1.f, 1.f, 0.f})))}); }, {}, Placed},
        {"Bend", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::Bend>, [](ProcedureFixture &f) { f.Do(action::mesh::Bend{.Angle = .8f, .Radius = 2.f, .Clamp = false, .Orientation = std::make_unique<quat>(numeric::AngleAxis(.3f, Normalize(vec3{0.f, 1.f, 1.f})))}); }, {}, Placed},
        {"VertexSlide", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::VertexSlide>, [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Factor = .3f, .Direction = Normalize(vec3{1.f, .2f, .1f})}); }, {}, Placed},
        {"EdgeSlide", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Edge, KindOf<procedure::EdgeSlide>, [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .5f, .Direction = {0.f, 1.f, 0.f}}); }, FacingEdgeCenter},
        {"SnapSymmetry", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<SnapSymmetry>, [](ProcedureFixture &f) { f.Do(SnapSymmetry{.Threshold = .5f}); }, {}, {}, Randomized<.05f>},
        {"VertexSlideEvenActive", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::VertexSlide>, [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Factor = .3f, .Direction = Normalize(vec3{1.f, .2f, .1f}), .Even = true}); }, FacingSide, {}, Randomized<.3f>},
        {"EdgeSlideEvenActive", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::EdgeSlide>, [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .5f, .Direction = {0.f, 1.f, 0.f}, .Even = true}); }, FacingSide, {}, Randomized<.3f>},
        {"LoopCutActive", primitive::Cuboid{{1.f, 2.f, 3.f}}, Element::Edge, KindOf<LoopCut>, [](ProcedureFixture &f) { f.Do(LoopCut{.Cuts = 2u}); }, CornerEdgeCenters},
        {"SplitNonplanarCuboid", primitive::Cuboid{}, Element::Face, KindOf<SplitNonplanarFaces>, [](ProcedureFixture &f) { f.Do(SplitNonplanarFaces{}); }, {}, {}, Randomized<.3f>},
        {"SplitConcaveDisc", primitive::Circle{.Radius = 5.f, .Segments = 8u}, Element::Face, KindOf<SplitConcaveFaces>, [](ProcedureFixture &f) { f.Do(SplitConcaveFaces{}); }, {}, {}, Notched},
        {"ConnectDiagonal", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<ConnectVertices>, [](ProcedureFixture &f) { f.Do(ConnectVertices{}); }, FacingDiagonal},
        {"Fill", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<Fill>, [](ProcedureFixture &f) { f.Do(Fill{}); }, FacingDiagonal},
        {"BridgeEdgeLoops", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Edge, KindOf<BridgeEdgeLoops>, [](ProcedureFixture &f) { f.Do(BridgeEdgeLoops{}); }, OppositeEdgeCenters, {}, Wires},
        {"GridFill", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Edge, KindOf<GridFill>, [](ProcedureFixture &f) { f.Do(GridFill{}); }, BoundaryEdgeCenters, {}, Opened},
        {"EdgeRotate", primitive::IcoSphere{.Radius = 3.f, .Subdivisions = 1u}, Element::Edge, KindOf<EdgeRotate>, [](ProcedureFixture &f) { f.Do(EdgeRotate{}); }, FacingEdgeCenter},
        {"BeautifyFaces", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Face, KindOf<BeautifyFaces>, [](ProcedureFixture &f) { f.Do(BeautifyFaces{}); }, {}, {}, TriangulatedRandomized},
        {"Unsubdivide", primitive::Plane{}, Element::Vertex, KindOf<Unsubdivide>, [](ProcedureFixture &f) { f.Do(Unsubdivide{}); }, {}, {}, Subdivided},
        {"RotateUVs", AttributedGrid, Element::Face, KindOf<RotateUVs>, [](ProcedureFixture &f) { f.Do(RotateUVs{}); }, AdjacentFaceCenters},
        {"ReverseUVs", AttributedGrid, Element::Face, KindOf<ReverseUVs>, [](ProcedureFixture &f) { f.Do(ReverseUVs{}); }, AdjacentFaceCenters},
        {"RotateColors", AttributedGrid, Element::Face, KindOf<RotateColors>, [](ProcedureFixture &f) { f.Do(RotateColors{.CounterClockwise = true}); }, AdjacentFaceCenters},
        {"ReverseColors", AttributedGrid, Element::Face, KindOf<ReverseColors>, [](ProcedureFixture &f) { f.Do(ReverseColors{}); }, AdjacentFaceCenters},
        {"MarkSharpEdges", primitive::Cuboid{{1.f, 2.f, 3.f}}, Element::Edge, KindOf<procedure::MarkSharp>, [](ProcedureFixture &f) { f.Do(action::object::SetSelectedSharp{.Element = Element::Edge, .Sharp = true}); }, CornerEdgeCenters},
        {"MarkSharpVertices", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::MarkSharp>, [](ProcedureFixture &f) { f.Do(action::object::SetSelectedSharp{.Element = Element::Vertex, .Sharp = true}); }, FacingSide},
        {"SmoothFace", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::MarkSharp>, [](ProcedureFixture &f) { f.Do(action::object::SetSelectedSharp{.Element = Element::Face, .Sharp = false}); }, FacingFaceCenter},
        {"Knife", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::Knife>, [](ProcedureFixture &f) {
             f.Render();
             const auto &view = f.View();
             const auto pixel = [&](vec3 p) {
                 const auto clip = view.ViewProj * vec4{p.x, p.y, p.z, 1.f};
                 return vec2{(clip.x / clip.w + 1.f) * .5f * view.ViewportSize.x, (1.f - clip.y / clip.w) * .5f * view.ViewportSize.y};
             };
             f.Do(action::mesh::Knife{pixel(Placed.P + vec3{-4.f, .3f, 0.f}), pixel(Placed.P + vec3{4.f, -.2f, 0.f}), std::make_unique<RenderView>(f.R.get<const Viewport>(FirstViewport(f.R)).FrameView)}); }, {}, Placed},
        {"ShadeSmooth", primitive::Cuboid{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::ShadeSmooth>, [](ProcedureFixture &f) { f.Do(action::object::SetSelectedSmoothShading{true}); }},
        {"ShadeFlat", primitive::UVSphere{.Radius = 2.f, .Slices = 8u, .Stacks = 4u}, Element::Face, KindOf<procedure::ShadeSmooth>, [](ProcedureFixture &f) { f.Do(action::object::SetSelectedSmoothShading{false}); }, {}, {}, ShadedSmooth},
        {"ShadeSmoothByAngle", primitive::UVSphere{.Radius = 2.f, .Slices = 8u, .Stacks = 4u}, Element::Face, KindOf<procedure::ShadeSmoothByAngle>, [](ProcedureFixture &f) { f.Do(action::object::ShadeSelectedSmoothByAngle{.6f}); }},
    }};
    // A boolean with another object appends a Boolean node reading it through an Object Info node, in the space the in-place boolean maps it into.
    Scenario(f, other, "BooleanObject", [](ProcedureFixture &f) {
        auto &r = f.R;
        f.Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = Placed})});
        const auto active = FindActiveEntity(r);
        const auto mesh = r.get<const Instance>(active).Entity;
        const Transform world{.P = {1.f, .5f, -.25f}, .R = numeric::AngleAxis(.6f, Normalize(vec3{0.f, 1.f, 1.f})), .S = {.75f, 1.f, 1.25f}};
        f.Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = world, .Select = MeshInstanceCreateInfo::SelectBehavior::Additive})});
        expect(fatal(FindActiveEntity(r) == active));
        state::Entity tool = state::Null;
        for (const auto e : r.view<const Instance>())
            if (e != active) tool = e;
        const auto described = [&] { return std::pair{f.Shaped(mesh), MarksOf(f.Store(), f.Output(mesh))}; };
        const auto before = described();
        f.Do(action::mesh::Boolean{BooleanOperation::Difference});
        const auto in_place = described();
        expect(in_place != before);
        f.Undo();
        f.Do(AddProcedure{mesh});
        const auto node_count = uint32_t(r.get<const GeometryProcedure>(mesh).Nodes.size());
        f.Do(action::mesh::Boolean{BooleanOperation::Difference});
        const auto &procedure = r.get<const GeometryProcedure>(mesh);
        expect(fatal(procedure.Nodes.size() == node_count + 2u));
        const auto [boolean, boolean_kind] = procedure.Nodes[node_count];
        const auto [info, info_kind] = procedure.Nodes[node_count + 1u];
        expect(boolean_kind == KindOf<procedure::Boolean> && info_kind == KindOf<procedure::ObjectInfo> && r.get<const procedure::ObjectInfo>(info).Object == tool);
        expect(procedure.Inputs[InputFirst(procedure, node_count) + 1u].FromNode == node_count + 1u && procedure.ActiveNode == node_count);
        expect(!r.any_of<NodeError>(boolean) && !r.any_of<NodeError>(info));
        expect(described() == in_place);
    });
    for (const auto &row : rows) {
        Scenario(f, other, std::string{row.Name}, [&row](ProcedureFixture &f) {
            auto &r = f.R;
            if (const auto *shape = std::get_if<PrimitiveShape>(&row.Shape)) {
                f.Do(action::object::AddMeshPrimitive{*shape, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = row.World})});
            } else {
                const auto instance = AddMesh(r, CreateMesh(r, std::get<MeshSource (*)()>(row.Shape)()).StoreId, MeshInstanceCreateInfo{.Transform = row.World}).second;
                f.Commit("Add mesh");
                f.Do(action::selection::Select{instance});
            }
            const auto mesh = r.get<const Instance>(FindActiveEntity(r)).Entity;
            const auto described = [&] { return std::pair{f.Shaped(mesh), MarksOf(f.Store(), f.Output(mesh))}; };
            f.EnterEdit(row.Domain);
            if (row.Prepare) row.Prepare(f);
            const auto select = [&] {
                if (!row.Targets) {
                    f.Do(action::selection::SelectAll{});
                    return;
                }
                f.Render();
                const auto targets = row.Targets(Mesh{f.Store(), f.Output(mesh)}, f.View().CameraPosition);
                expect(fatal(!targets.empty()));
                for (uint32_t i = 0u; i < targets.size(); ++i) f.Click(targets[i], i > 0u);
            };
            select();
            const auto before = described();
            row.Gesture(f);
            const auto in_place = described();
            expect(in_place != before);
            f.Undo();
            f.Do(AddProcedure{mesh});
            select();
            const auto node_count = r.get<const GeometryProcedure>(mesh).Nodes.size();
            row.Gesture(f);
            const auto &procedure = r.get<const GeometryProcedure>(mesh);
            expect(fatal(procedure.Nodes.size() == node_count + 1u));
            const auto [node, kind] = procedure.Nodes.back();
            expect(kind == row.Kind && !r.any_of<NodeError>(node));
            expect(described() == in_place);
        });
    }
}

template<auto M, auto... Ms> uint16_t SocketOf() {
    const auto sockets = KindSockets(KindOf<field::Owner<M>>);
    constexpr std::array<uint8_t, sizeof...(Ms)> nested{uint8_t(field::IndexOf<Ms>)...};
    return uint16_t(std::ranges::find_if(sockets, [&](Socket socket) { return socket.Carries(uint8_t(field::IndexOf<M>), nested.empty() ? NoField : nested[0]); }) - sockets.begin());
}
template<AttributeDomain D> NamedAttributeRef FloatAttribute(MeshStore &meshes, uint32_t id, auto &&value) {
    const auto attribute = meshes.AddAttribute(id, D, AttributeType::Float, "parameter").value();
    const Mesh mesh{meshes, id};
    const auto set = [&](auto handle) {
        const float v = value(mesh, handle);
        meshes.SetAttribute<float>(attribute, *handle, std::span{&v, 1u});
    };
    if constexpr (D == AttributeDomain::Vertex) {
        for (const auto v : mesh.vertices()) set(v);
    } else {
        for (const auto face : mesh.faces()) set(face);
    }
    return meshes.AttributeRef(attribute);
}
float WidthAt(const Mesh &mesh, he::VH v) {
    const auto p = mesh.GetPosition(v);
    return .05f * (p.x + p.y + p.z) + .2f;
}

// Linked values override parameter constants, linked attributes set each element's parameter, and an attribute from another domain is rejected.
void ParameterSockets(ProcedureFixture &f) {
    auto &r = f.R;
    auto &meshes = f.Store();
    using action::mesh::Bevel, action::mesh::ShrinkFatten, action::mesh::Poke;
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto source_id = f.Source(mesh);
    const auto source = Describe(meshes, source_id);
    const auto output = [&] { return f.Shaped(mesh); };
    const auto [real, real_entity] = f.Append(mesh, KindOf<FloatValue>);
    {
        const auto [bevel, bevel_entity] = f.Append(mesh, KindOf<Bevel>);
        f.Route(mesh, {0u, bevel, 1u});
        const auto one = output();
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 3u));
        const auto three = output();
        expect(three != one);
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 1u));
        const auto [integer, integer_entity] = f.Append(mesh, KindOf<IntegerValue>);
        f.Do(UpdateOf<&IntegerValue::Value>(integer_entity, 3u));
        const auto segments = SocketOf<&Bevel::Segments>(), width = SocketOf<&Bevel::Width>();
        f.Do(Link{mesh, integer, 0u, bevel, segments});
        expect(output() == three);
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 5u));
        expect(output() == three);
        f.Do(UpdateOf<&IntegerValue::Value>(integer_entity, 1u));
        expect(output() == one);
        f.Do(action::nodes::Unlink{mesh, bevel, segments});
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 3u));
        expect(output() == three);
        const auto position = f.Append(mesh, KindOf<Position>).first;
        f.Fails(Link{mesh, real, 0u, bevel, segments});
        f.Fails(Link{mesh, position, 0u, bevel, segments});
        f.Do(UpdateOf<&FloatValue::Value>(real_entity, .2f));
        f.Do(Link{mesh, real, 0u, bevel, width});
        f.Do(UpdateOf<&Bevel::Width>(bevel_entity, .3f));
        expect(output() == SelectAllReference(f, source_id, {Op(MeshTopologyOp::BevelEdges, .2f, 0u, 3u)}));
        f.Do(action::nodes::Unlink{mesh, bevel, width});
        f.Do(UpdateOf<&Bevel::Width>(bevel_entity, .1f));
        const auto [flag, flag_entity] = f.Append(mesh, KindOf<BooleanValue>);
        f.Do(Link{mesh, flag, 0u, bevel, 1u});
        expect(output() == source);
        f.Do(UpdateOf<&BooleanValue::Value>(flag_entity, true));
        expect(output() == three);
        f.Do(DeleteNodes{mesh, {bevel, position}});
    }
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [dot, dot_entity] = f.Append(mesh, KindOf<VectorMath>);
    const auto [add, add_entity] = f.Append(mesh, KindOf<FloatMath>);
    f.Do(UpdateOf<&VectorMath::Operation>(dot_entity, VectorMathOp::Dot));
    f.Do(UpdateOf<&VectorMath::B>(dot_entity, vec3{.05f, .05f, .05f}));
    f.Do(UpdateOf<&FloatMath::B>(add_entity, .2f));
    f.Do(Link{mesh, position, 0u, dot, 0u});
    f.Do(Link{mesh, dot, f.Out(mesh, dot), add, 0u});
    // Run the operator on a source clone with all elements selected and Param0 read from the added parameter attribute.
    const auto reference = [&](MeshTopologyOp op, auto &&parameter) {
        return Reference(f, source_id, [&](uint32_t clone) {
                   auto task = Op(op, 0.f, TopologyFlagSelectAll);
                   task.Param0Attribute = parameter(clone);
                   f.Run(clone, {task});
                   return GeometrySelection{};
               })
            .Output;
    };
    const auto widths = [&](uint32_t clone) { return FloatAttribute<AttributeDomain::Vertex>(meshes, clone, WidthAt); };
    const auto add_operator = [&](uint16_t kind, uint16_t socket) {
        const auto [node, entity] = f.Append(mesh, kind);
        f.Do(Link{mesh, add, f.Out(mesh, add), node, socket});
        f.Route(mesh, {0u, node, 1u});
        return std::pair{node, entity};
    };
    {
        const auto [bevel, bevel_entity] = add_operator(KindOf<Bevel>, SocketOf<&Bevel::Width>());
        f.Do(UpdateOf<&Bevel::Vertices>(bevel_entity, true));
        expect(output() == reference(MeshTopologyOp::BevelVertices, widths));
        f.Do(DeleteNodes{mesh, {bevel}});
    }
    {
        const auto [shrink, shrink_entity] = add_operator(KindOf<ShrinkFatten>, SocketOf<&ShrinkFatten::Distance>());
        // The kernel reads the factor at each vertex from the attribute, past the constant factor's zero check.
        const auto kernel = Reference(f, source_id, [&](uint32_t clone) {
            const PositionOperationTarget target{.StoreId = clone, .Selection = AllHandles(Mesh{meshes, clone})};
            ExecuteGeometryPositions(r, std::span{&target, 1u}, PositionEditOp::ShrinkFatten, 1.f, 1u, {.Factor = widths(clone)});
            return GeometrySelection{};
        });
        expect(kernel.Output != source && output() == kernel.Output);
        f.Do(DeleteNodes{mesh, {shrink}});
    }
    {
        // A vertex attribute does not fit the poke's per-face offset, and a face normal's z does.
        const auto [poke, poke_entity] = add_operator(KindOf<Poke>, SocketOf<&Poke::Offset>());
        expect(output() == source);
        const auto [normal, normal_entity] = f.Append(mesh, KindOf<Normal>);
        f.Do(UpdateOf<&Normal::Domain>(normal_entity, AttributeDomain::Face));
        f.Do(UpdateOf<&VectorMath::B>(dot_entity, vec3{0.f, 0.f, .3f}));
        f.Do(Link{mesh, normal, 0u, dot, 0u});
        f.Do(Link{mesh, dot, f.Out(mesh, dot), poke, SocketOf<&Poke::Offset>()});
        const auto offsets = [&](uint32_t clone) { return FloatAttribute<AttributeDomain::Face>(meshes, clone, [](const Mesh &m, he::FH face) { return .3f * m.GetNormal(face).z; }); };
        expect(output() == reference(MeshTopologyOp::Poke, offsets));
        f.Do(DeleteNodes{mesh, {poke}});
    }
    {
        // Link vector and rotation values into nested transform members while editing scale directly.
        const auto [transform, transform_entity] = f.Append(mesh, KindOf<TransformGeometry>);
        const auto [vector, vector_entity] = f.Append(mesh, KindOf<VectorValue>);
        const auto [rotation, rotation_entity] = f.Append(mesh, KindOf<RotationValue>);
        // The rotation leaves no two moved corners a GPU rounding apart in sort order.
        const Transform delta{{1.f, 2.f, 3.f}, numeric::AngleAxis(.5f, Normalize(vec3{1.f, 2.f, 3.f})), {2.f, 1.f, .5f}};
        f.Do(UpdateOf<&VectorValue::Value>(vector_entity, delta.P));
        f.Do(UpdateOf<&RotationValue::Value>(rotation_entity, delta.R));
        f.Do(UpdateOf<&TransformGeometry::Delta, &Transform::S>(transform_entity, delta.S));
        f.Do(Link{mesh, vector, 0u, transform, SocketOf<&TransformGeometry::Delta, &Transform::P>()});
        f.Do(Link{mesh, rotation, 0u, transform, SocketOf<&TransformGeometry::Delta, &Transform::R>()});
        f.Route(mesh, {0u, transform, 1u});
        expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, source_id, [&](uint32_t, vec3 p) { return delta.R * (delta.S * p) + delta.P; })));
        f.Do(DeleteNodes{mesh, {transform, vector, rotation}});
    }
    {
        const auto [store, store_entity] = f.Append(mesh, KindOf<StoreNamedAttribute>);
        f.Do(UpdateOf<&FloatValue::Value>(real_entity, 2.5f));
        f.Do(UpdateOf<&StoreNamedAttribute::Domain>(store_entity, AttributeDomain::Face));
        f.Do(SetAttributeName{store_entity, "heat"});
        f.Do(Link{mesh, real, 0u, store, 1u});
        f.Route(mesh, {0u, store, 1u});
        const auto heat = meshes.FindAttribute(f.Output(mesh), "heat");
        expect(fatal(heat != InvalidOffset));
        expect(meshes.GetAttribute(heat).Domain == AttributeDomain::Face && meshes.GetAttribute(heat).Type == AttributeType::Float);
        const auto values = meshes.AttributeView<float>(heat);
        for (const auto face : Mesh{meshes, f.Output(mesh)}.faces()) expect(values[*face] == 2.5f);
    }
}

// A graph whose display path edits nothing draws the source record itself, and one editing the source draws a copy until the edit leaves the path.
// An unlinked Output draws the source, and applying an aliased procedure keeps the source record.
void SourceAlias(ProcedureFixture &f) {
    auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    const auto source = f.Source(mesh);
    const auto shape = Describe(f.Store(), source);
    const auto aliased = [&] { return f.Output(mesh) == source && !r.all_of<ResyncState>(mesh) && f.Shaped(mesh) == shape; };
    const auto copied = [&] { return f.Output(mesh) != source && r.all_of<ResyncState>(mesh) && Describe(f.Store(), source) == shape; };
    f.Do(AddProcedure{mesh});
    expect(aliased());
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>).first;
    f.Route(mesh, {0u, extrude, 1u});
    expect(copied());
    const auto extruded = f.Shaped(mesh);
    expect(extruded != shape);
    f.Do(DeleteNodes{mesh, {extrude}});
    expect(aliased());
    f.Undo();
    expect(copied() && f.Shaped(mesh) == extruded);
    f.Redo();
    expect(aliased());
    // A Boolean without a second mesh reads its first and passes it through.
    const auto boolean = f.Append(mesh, KindOf<procedure::Boolean>).first;
    f.Route(mesh, {0u, boolean, 1u});
    expect(aliased() && !r.any_of<NodeError>(f.Node(mesh, boolean)));
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>).first;
    f.Route(mesh, {boolean, subdivide, 1u});
    expect(copied() && Mesh{f.Store(), f.Output(mesh)}.FaceCount() == 24u);
    f.Do(action::nodes::Unlink{mesh, 1u, 0u});
    expect(aliased());
    f.Undo();
    expect(copied());
    f.Undo();
    expect(aliased());
    f.Do(ApplyProcedure{mesh});
    expect(!r.any_of<GeometryProcedure, EvaluatedMesh, ResyncState, CookPlan>(mesh));
    expect(f.Source(mesh) == source && Describe(f.Store(), source) == shape);
}

// Instances of the source on the source's own points place a copy of the source.
Graph SourceInstances(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto source = f.Source(mesh);
    const auto plane = Positions(f.Store(), source);
    f.Do(AddProcedure{mesh});
    const auto to_points = f.Append(mesh, KindOf<MeshToPoints>).first;
    const auto instancer = f.Append(mesh, KindOf<InstanceOnPoints>).first;
    f.Route(mesh, {0u, to_points, instancer});
    f.Do(Link{mesh, 0u, 0u, instancer, 1u});
    f.Route(mesh, {instancer, 1u});
    return {
        mesh,
        [mesh, source, plane](ProcedureFixture &f) {
            const auto &r = f.R;
            const auto &entities = r.get<const ProcedureInstances>(mesh).Entities;
            expect(fatal(entities.size() == plane.size()));
            const auto instanced = r.get<const Instance>(entities.front()).Entity;
            expect(instanced != mesh && f.Source(instanced) != source && Near(Positions(f.Store(), f.Source(instanced)), plane));
            expect(f.Output(mesh) != source && Positions(f.Store(), source) == plane);
        },
    };
}

constexpr std::array<GraphRow, 18> Graphs{{
    {"ExtrudeSubdivide", ExtrudeSubdivide},
    {"CapResync", CapResync},
    {"ScalePositions", ScalePositions},
    {"CompareSelectsExtrusion", CompareSelectsExtrusion},
    {"NamedAttributeReads", NamedAttributeReads},
    {"StorePromotes", StorePromotes},
    {"StoreSelectionDelete", StoreSelectionDelete},
    {"DomainMismatch", DomainMismatch},
    {"JoinGenerated", JoinGenerated},
    {"HeatSubdivide", HeatSubdivide},
    {"BooleanIntersection", BooleanIntersection},
    {"BooleanAction", BooleanAction},
    {"PrimitiveShapeEdit", PrimitiveShapeEdit},
    {"ObjectInfo", ObjectInfo},
    {"InstancesBorrowedPublish", Instances<true, false>},
    {"InstancesGeneratedRealize", Instances<false, true>},
    {"NodeEditing", NodeEditing},
    {"SourceInstances", SourceInstances},
}};
} // namespace

void TestProcedures() {
    ProcedureFixture f;
    Fixture other;
    OperatorKinds(f, other);
    Survives(f, other, Graphs);
    ViewportOperators(f, other);
    InPlaceOperators(f, other);
    Scenario(f, other, "ParameterSockets", ParameterSockets);
    Scenario(f, other, "SourceAlias", SourceAlias);
}
