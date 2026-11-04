#include "MeshProperties.h"
#include "ProcedureFixture.h"
#include "TopologyCases.h"
#include "animation/Fields.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/ModalModes.h"
#include "mesh/Unsubdivide.h"
#include "procedure/Macros.h"
#include "procedure/NodeEditor.h"
#include "project/HistoryUi.h"
#include "scene/HierarchyInstance.h"
#include "scene/SceneGraph.h"
#include <imgui_internal.h>

using boost::ut::expect, boost::ut::fatal;

namespace {
using namespace procedure;
using action::UpdateOf;
using action::nodes::AddNode, action::nodes::CollapseProcedure, action::nodes::DeleteNodes, action::nodes::DuplicateNodes, action::nodes::InsertNode, action::nodes::Link, action::nodes::SelectItems, action::nodes::SetAttributeName, action::nodes::SetDisplay, action::nodes::SetInlineSelection;
using ::test::Op, ::test::Patchwork, ::test::Region, ::test::SelectionWhere, ::test::TopologyCases;

std::vector<Corner> Points(const MeshStore &meshes, uint32_t id, auto &&transform) {
    const Mesh mesh{meshes, id};
    std::vector<Corner> points;
    for (const auto v : mesh.vertices()) {
        const auto p = transform(*v, test::Position(mesh, v));
        points.push_back({p.x, p.y, p.z});
    }
    std::ranges::sort(points);
    return points;
}
std::vector<Corner> SourcePoints(ProcedureFixture &f, state::Entity mesh_entity, auto &&transform) {
    return f.WithSource(mesh_entity, [&](uint32_t id) { return Points(f.Store(), id, transform); });
}
bool Near(const std::vector<Corner> &a, const std::vector<Corner> &b) {
    return a.size() == b.size() && std::ranges::equal(a, b, [](const Corner &p, const Corner &q) {
               return std::abs(p[0] - q[0]) < 1e-5f && std::abs(p[1] - q[1]) < 1e-5f && std::abs(p[2] - q[2]) < 1e-5f;
           });
}
std::vector<Corner> ShapePositions(const auto &shape) {
    std::vector<Corner> points;
    for (const auto p : primitive::CreateMesh(shape).Positions) points.push_back({p.x, p.y, p.z});
    std::ranges::sort(points);
    return points;
}

// Steps each scalar parameter of the node's kind up and down through the node panel's field update, clamped by the field's spec.
template<typename K> void SweepParameters(ProcedureFixture &f, state::Entity node) {
    const auto before = f.R.get<const K>(node);
    field::ForEach(before, [&]<size_t I>(const auto &member, std::integral_constant<size_t, I>) {
        using T = std::remove_cvref_t<decltype(member)>;
        const auto offset = uint16_t(reinterpret_cast<const std::byte *>(&member) - reinterpret_cast<const std::byte *>(&before));
        const auto update = [&](auto value) { f.Do(action::Update<decltype(value)>{node, state::Key<K>(), offset, value}); };
        if constexpr (std::same_as<T, bool>) {
            update(!member);
        } else if constexpr (std::is_enum_v<T>) {
            if constexpr (field::EnumCount<T> > 0u) {
                const auto &values = field::EnumValues<T>;
                const auto next = values[size_t(std::ranges::find(values, member) - values.begin() + 1u) % values.size()];
                update(std::underlying_type_t<T>(next));
            }
        } else if constexpr (std::is_arithmetic_v<T>) {
            const auto step = std::is_floating_point_v<T> ? T(Specs<K>.ByIndex[I].Speed * 25.f) : T(1);
            update(T(member + step));
            update(T(member - step));
        }
    });
}
void SweepParameters(ProcedureFixture &f, state::Entity node, uint16_t kind) {
    VisitKind(kind, [&]<typename K> {
        if constexpr (!std::is_empty_v<K>) SweepParameters<K>(f, node);
    });
}
GeometrySelection LiveHandles(const MeshStore &meshes, uint32_t id, const InlineSelection &selection) {
    GeometrySelection handles;
    for (const auto [element, selected, listed] : {std::tuple{Element::Vertex, &selection.Vertices, &handles.Vertices}, std::tuple{Element::Edge, &selection.Edges, &handles.Edges}, std::tuple{Element::Face, &selection.Faces, &handles.Faces}}) {
        ForEachSelectedBlock(*selected, [&](uint32_t local, const SelectionBlock &words) {
            for (uint32_t slot = 0u; slot < MeshElementBlockSize; ++slot) {
                const auto handle = local * MeshElementBlockSize + slot;
                if (words[slot / 32u] >> (slot % 32u) & 1u && meshes.IsLiveElement(id, element, ElementHandle{handle})) listed->push_back(handle);
            }
        });
    }
    return handles;
}
} // namespace

// A display path of only topology tasks on one Source node draws what those tasks make of a clone of the Source, and a faulted node passes its input through.
void ProcedureFixture::CheckStep() {
    auto &meshes = Store();
    bool cloned = false;
    for (const auto [mesh, plan] : R.view<const CookPlan>().each()) {
        if (plan.ResultNode == NoNode) continue;
        const auto upstream = Upstream(plan, plan.ResultNode, [](uint32_t) { return true; });
        std::optional<SourceState> source;
        std::vector<NodeRecord> path;
        bool runs = true;
        for (const auto n : plan.Order) {
            if (!upstream[n]) continue;
            if (std::ranges::any_of(std::span{plan.Inputs}.subspan(plan.InputFirst[n], plan.InputFirst[n + 1] - plan.InputFirst[n]), [](const auto &input) { return input.Constant.has_value(); })) runs = false;
            const auto [node, kind] = plan.Nodes[n];
            if (kind == KindOf<procedure::Extrude> && R.get<const procedure::Extrude>(node).OffsetScale != 0.f) runs = false;
            if (kind == KindOf<procedure::Source> && !source) source = R.get<const SourceState>(node);
            else if (std::ranges::contains(TopologyCases, std::optional{kind}, &::test::TopologyCase::Kind)) path.push_back(plan.Nodes[n]);
            else runs = false;
        }
        if (!runs || !source || path.empty()) continue;
        const auto id = CloneSource(*source);
        cloned = true;
        for (const auto [node, kind] : path) {
            if (R.all_of<NodeError>(node)) continue;
            const auto *inline_selection = R.try_get<const InlineSelection>(node);
            const TopologyInput input{
                .In = inline_selection && !inline_selection->All ? std::optional<SelectionSource>{LiveHandles(meshes, id, *inline_selection)} : std::nullopt,
                .Domain = inline_selection ? inline_selection->Domain : Element::Vertex,
            };
            std::vector<MeshTopologyTask> tasks;
            VisitKind(kind, [&]<typename K> {
                if constexpr (requires(const K &p) { PlanTopology(R, p, id, input); }) {
                    auto planned = PlanTopology(R, R.get<const K>(node), id, input);
                    if constexpr (std::same_as<decltype(planned), std::vector<MeshTopologyTask>>) tasks = std::move(planned);
                    else if constexpr (std::same_as<decltype(planned), MeshTopologyTask>) tasks.push_back(std::move(planned));
                    else if (planned) tasks.push_back(std::move(*planned));
                }
            });
            expect(!tasks.empty());
            ExecuteGeometryTopologyStages(R, tasks);
        }
        expect(Describe(meshes, id) == Shaped(mesh));
        meshes.Release(id);
    }
    // The present node holds mesh storage as it stood before the clones.
    if (cloned) P->History.Revert();
}

namespace {
// Every operator kind cooks like its kernel on the patchwork at every parameter value, each generator kind like its primitive, and a fan-out cooks the read branch alone.
void OperatorKinds(ProcedureFixture &f, Fixture &other) {
    // Rows sharing a kind run its scenario once, under the first row's name.
    for (const auto &c : TopologyCases) {
        const auto kind = c.Kind;
        if (!kind || &c != &*std::ranges::find(TopologyCases, kind, &::test::TopologyCase::Kind)) continue;
        Scenario(f, other, std::string{c.Name}, [kind](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            const auto mesh = f.AddSourced({.Data = Patchwork()}).first;
            const auto source = f.SourceOf(mesh);
            const auto entity = f.Append(mesh, *kind);
            f.Do(SetInlineSelection{entity, std::make_unique<InlineSelection>(SelectionOf(Region(Mesh{meshes, f.Output(mesh)}), Element::Vertex))});
            f.Route(mesh, {f.Node(mesh, 0u), entity});
            SweepParameters(f, entity, *kind);
            expect(f.SourceOf(mesh) == source);
        });
    }
    // Generator kinds follow PrimitiveParameters' alternatives.
    for (const PrimitiveParameters shape : {PrimitiveParameters{primitive::Plane{}}, PrimitiveParameters{primitive::Circle{}}, PrimitiveParameters{primitive::Cube{}}, PrimitiveParameters{primitive::IcoSphere{}}, PrimitiveParameters{primitive::UVSphere{}}, PrimitiveParameters{primitive::QuadSphere{}}, PrimitiveParameters{primitive::Torus{}}, PrimitiveParameters{primitive::Cylinder{}}}) {
        Scenario(f, other, std::visit([]<typename K>(const K &) { return std::string{Title<K>.c_str()}; }, shape), [shape](ProcedureFixture &f) {
            auto &meshes = f.Store();
            const auto mesh = f.AddMesh(primitive::Plane{});
            const auto source = f.SourceOf(mesh);
            const auto generator = f.Append(mesh, uint16_t(KindOf<primitive::Plane> + shape.index()));
            const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
            f.Route(mesh, {generator, subdivide});
            const auto generated = CreateMesh(f.R, {.Data = primitive::CreateMesh(shape), .FlatShaded = true}).StoreId;
            const auto reference = Reference(f, generated, [&](uint32_t clone) {
                                       f.Run(clone, {Op(MeshTopologyOp::Subdivide, 1.f, TopologyFlagSelectAll)});
                                       return GeometrySelection{};
                                   }).Output;
            meshes.Release(generated);
            f.Commit("Release generated");
            expect(f.Shaped(mesh) == reference);
            expect(f.SourceOf(mesh) == source);
        });
    }
    Scenario(f, other, "FanOut", [](ProcedureFixture &f) {
        const auto mesh = f.AddMesh(primitive::Plane{});
        const auto source = f.SourceOf(mesh);
        const auto first = f.Append(mesh, KindOf<procedure::Extrude>);
        const auto second = f.Append(mesh, KindOf<procedure::Extrude>);
        f.Do(Link{mesh, f.Node(mesh, 0u), 0u, first, 0u});
        f.Do(Link{mesh, f.Node(mesh, 0u), 0u, second, 0u});
        for (const auto read : {first, second}) {
            f.Do(SetDisplay{mesh, read});
            expect(!f.R.any_of<NodeError>(first) && !f.R.any_of<NodeError>(second));
            expect(f.SourceOf(mesh) == source);
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
struct GraphData {
    std::string_view Name;
    GraphDescription Description;
    void (*Oracle)(ProcedureFixture &, const BuiltGraph &);
    void (*Edit)(ProcedureFixture &, state::Entity);
};

// The graph's output matches its oracle, a duplicate cooks the same and edits apart, and deleting and collapsing the procedure leave its output.
void Survives(ProcedureFixture &f, Fixture &other, std::string_view name, auto &&build) {
    Scenario(f, other, std::string{name}, [&build](ProcedureFixture &f) {
        auto &r = f.R;
        const auto graph = build(f);
        const auto mesh = graph.Mesh;
        if (graph.Oracle) graph.Oracle(f);
        const auto output = f.Shaped(mesh);
        const auto display = r.get<const GeometryProcedure>(mesh);
        f.Do(SetDisplay{mesh});
        expect(f.Shaped(mesh) == Shape{});
        const auto *instances = r.try_get<const ProcedureInstances>(mesh);
        expect(!instances || instances->Entities.empty());
        f.Do(SetDisplay{mesh, display.DisplayNode, display.DisplayPort});
        expect(f.Shaped(mesh) == output);
        const auto unused = f.Append(mesh, KindOf<action::mesh::Bevel>);
        expect(r.get<const GeometryProcedure>(mesh).DisplayNode == display.DisplayNode);
        expect(f.Shaped(mesh) == output);
        const auto cooks = Cooks(r);
        f.Do(UpdateOf<&action::mesh::Bevel::Width>(unused, .2f));
        std::vector<state::Entity> groups;
        auto member = r.get<const GeometryProcedure>(mesh).DisplayNode;
        for (int depth = 0; depth < 2; ++depth) {
            f.Do(action::nodes::GroupNodes{mesh, {member}});
            member = r.get<const GeometryProcedure>(mesh).Last;
            groups.push_back(member);
            expect(f.Shaped(mesh) == output && Cooks(r) == cooks);
        }
        if (!graph.Edit) {
            for (const auto group : groups | std::views::reverse) {
                f.Do(action::nodes::UngroupNodes{group});
                expect(f.Shaped(mesh) == output && Cooks(r) == cooks);
            }
            return;
        }
        f.Do(action::selection::Select{f.Object(mesh)});
        f.Do(action::object::Duplicate{});
        const auto copy = r.get<const Instance>(FindActiveEntity(r)).Entity;
        expect(copy != mesh && f.Output(copy) != f.Output(mesh) && f.Shaped(copy) == output);
        graph.Edit(f, copy);
        expect(f.Shaped(copy) != output && f.Shaped(mesh) == output);
        f.Undo();
        f.Undo();
        const auto procedures = r.view<const GeometryProcedure>().size();
        f.Do(action::object::Delete{});
        expect(!r.valid(mesh) && r.view<const GeometryProcedure>().size() + 1u == procedures);
        f.Undo();
        expect(f.Shaped(mesh) == output);
        f.Do(CollapseProcedure{mesh});
        const auto &collapsed = r.get<const CookPlan>(mesh);
        expect(collapsed.Nodes.size() == 1u && collapsed.Nodes[0].Kind == KindOf<procedure::Source>);
        expect(f.SourceOf(mesh).StoreId == f.Output(mesh) && f.Shaped(mesh) == output);
    });
}

void ExtrudeDisplacement(ProcedureFixture &f) {
    using E = procedure::Extrude;
    using M = action::mesh::ExtrudeMode;
    for (const auto mode : {M::Region, M::Edges, M::FacesIndividual, M::Vertices}) {
        const auto mesh = f.AddMesh(primitive::Plane{});
        const Mesh plane{f.Store(), f.Output(mesh)};
        const auto normal = test::Normal(plane, *plane.faces().begin());
        const auto extrude = f.Append(mesh, KindOf<E>);
        f.Do(UpdateOf<&E::Mode>(extrude, mode));
        f.Route(mesh, {f.Node(mesh, 0u), extrude});
        const auto topology = f.R.get<const NodeExtrusion>(extrude).State;
        const auto zero = f.Shaped(mesh);
        for (const float distance : {.25f, .75f, -.5f, 0.f}) {
            f.Do(UpdateOf<&E::OffsetScale>(extrude, distance));
            expect(f.R.get<const NodeExtrusion>(extrude).State == topology);
            const Mesh result{f.Store(), f.Output(mesh)};
            uint32_t displaced = 0u;
            for (const auto vertex : result.vertices()) {
                const float height = Dot(test::Position(result, vertex), normal);
                expect(std::abs(height) < .0001f || std::abs(height - distance) < .0001f);
                if (std::abs(height - distance) < .0001f) ++displaced;
            }
            expect(displaced >= 4u);
            if (distance == 0.f) expect(f.Shaped(mesh) == zero);
        }
        f.Do(UpdateOf<&E::AlongNormals>(extrude, false));
        f.Do(UpdateOf<&E::Offset>(extrude, vec3{.3f, -.2f, .7f}));
        f.Do(UpdateOf<&E::OffsetScale>(extrude, 2.f));
        expect(f.Shaped(mesh) != zero);
        const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
        f.Route(mesh, {f.Node(mesh, 0u), extrude, subdivide});
        const auto downstream = f.Shaped(mesh);
        f.Do(action::nodes::SetOperationHandle{extrude, f.Object(mesh), HandleParameter::ExtrudeScale});
        f.Do(UpdateOf<&E::OffsetScale>(extrude, 1.f));
        f.Do(action::nodes::SetOperationHandle{});
        expect(f.R.get<const GeometryProcedure>(mesh).DisplayNode == subdivide);
        expect(f.Shaped(mesh) != downstream);
        expect(f.R.get<const NodeExtrusion>(extrude).State == topology);
    }
}

vec3 OffsetOf(uint32_t v) { return {float(v % 7u), 1.f, -float(v % 3u)}; }
bool RestMatches(ProcedureFixture &f, state::Entity mesh) {
    const auto &meshes = f.Store();
    const auto id = f.Output(mesh);
    const auto rest = meshes.FindAttribute(id, "rest");
    if (rest == InvalidOffset) return false;
    const auto view = meshes.AttributeView<vec3>(rest);
    const Mesh output{meshes, id};
    for (const auto v : output.vertices()) {
        const auto d = view[*v] - test::Position(output, v);
        if (std::abs(d.x) >= 1e-5f || std::abs(d.y) >= 1e-5f || std::abs(d.z) >= 1e-5f) return false;
    }
    return true;
}

// Expects a plane of eight unit cells a side, centered at the origin, to keep exactly its vertex columns at x up to `last` and the cells between them.
void ExpectColumnsThrough(ProcedureFixture &f, state::Entity mesh, int last) {
    const Mesh output{f.Store(), f.Output(mesh)};
    const auto columns = uint32_t(last + 5);
    expect(output.VertexCount() == 9u * columns && output.FaceCount() == 8u * (columns - 1u)) << "vertices" << output.VertexCount() << "faces" << output.FaceCount();
    for (const auto v : output.vertices()) expect(test::Position(output, v).x <= float(last)) << "vertex x" << test::Position(output, v).x;
}

std::vector<uint16_t> PortIds(const state::Registry &r, state::Entity subnet) {
    std::vector<uint16_t> ids;
    for (const auto &port : r.get<const procedure::SubnetInterface>(subnet).Ports) ids.push_back(port.Id);
    return ids;
}
// Feeds the cube's Source node and a generated bar through it into the macro's first two ports, displays its third, and returns the mesh and the port ids.
std::pair<state::Entity, std::vector<uint16_t>> CubeAndBarInto(ProcedureFixture &f, MacroIndex macro) {
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto source = f.Node(mesh, 0u);
    const auto box = f.Append(mesh, KindOf<primitive::Cube>);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(box, vec3{2.f, .5f, .5f}));
    f.Do(action::nodes::AddMacro{mesh, macro, {}});
    const auto node = ActiveNodeOf(f.R, mesh);
    const auto ports = PortIds(f.R, node);
    f.Do(Link{mesh, source, f.Out(source), node, ports[0]});
    f.Do(Link{mesh, box, f.Out(box), node, ports[1]});
    f.Do(SetDisplay{mesh, node, ports[2]});
    return {mesh, ports};
}

template<AttributeScalar T, AttributeDomain D = AttributeDomain::Vertex>
uint32_t AddValues(MeshStore &meshes, uint32_t id, std::string_view name, auto &&value, AttributeValue fallback = {}) {
    const auto attribute = meshes.AddAttribute(id, D, AttributeTypeOf<T>, name, fallback).value();
    const Mesh mesh{meshes, id};
    const auto set = [&](auto handle) {
        const T v = value(mesh, handle);
        meshes.SetAttribute<T>(attribute, *handle, std::span{&v, 1u});
    };
    if constexpr (D == AttributeDomain::Vertex) {
        for (const auto v : mesh.vertices()) set(v);
    } else {
        for (const auto face : mesh.faces()) set(face);
    }
    return attribute;
}

void AddOffsets(ProcedureFixture &f, uint32_t source) {
    AddValues<vec3>(f.Store(), source, "offset", [](const Mesh &, he::VH v) { return OffsetOf(*v); });
}
// Gives each source vertex a "weight" of 5 over the attribute's default of -1.
void AddWeights(ProcedureFixture &f, uint32_t source) {
    AddValues<float>(f.Store(), source, "weight", [](const Mesh &, he::VH) { return 5.f; }, AttributeValue{std::bit_cast<uint32_t>(-1.f), 0u, 0u, 0u});
}
void AddHeightHeat(ProcedureFixture &f, uint32_t source) {
    AddValues<float>(f.Store(), source, "heat", [](const Mesh &m, he::VH v) { return test::Position(m, v).z; });
}

void MoveFirstVertex(ProcedureFixture &f, uint32_t source) {
    const PositionOperationTarget target{.StoreId = source, .Selection = GeometrySelection{.Vertices = {**Mesh{f.Store(), source}.vertices().begin()}}};
    ExecuteGeometryPositions(f.R, target, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {.3f, -.5f, .1f}}}});
}

void ExpectUpperExtruded(ProcedureFixture &f, const BuiltGraph &g) {
    const auto &meshes = f.Store();
    const auto kernel = SourceReference(f, g.Mesh, [&](uint32_t clone) {
        const Mesh cube{meshes, clone};
        const auto upper = SelectionWhere(cube, [&](he::VH v) { return test::Position(cube, v).z > 0.f; });
        expect(upper.Faces.size() == 1u);
        f.Run(clone, {{.Op = MeshTopologyOp::ExtrudeRegion, .Selection = upper}});
        return GeometrySelection{};
    });
    expect(f.Shaped(g.Mesh) == kernel.Output);
}
// The output vertices beyond and within the unit major radius around the Y axis.
std::pair<uint32_t, uint32_t> OuterInner(ProcedureFixture &f, state::Entity mesh) {
    uint32_t outer = 0u, inner = 0u;
    for (const auto p : f.Shaped(mesh).Vertices) {
        const auto radius = std::sqrt(p[0] * p[0] + p[2] * p[2]);
        if (radius > 1.001f) ++outer;
        if (radius < .999f) ++inner;
    }
    return {outer, inner};
}
void ExpectJoinedDefaults(ProcedureFixture &f, const BuiltGraph &g) {
    const auto &meshes = f.Store();
    const auto [vertices, faces] = f.WithSource(g.Mesh, [&](uint32_t source) {
        const Mesh single{meshes, source};
        return std::pair{single.VertexCount(), single.FaceCount()};
    });
    const auto id = f.Output(g.Mesh);
    const Mesh joined{meshes, id};
    expect(joined.VertexCount() == 2u * vertices && joined.FaceCount() == 2u * faces);
    const auto view = meshes.AttributeView<float>(meshes.FindAttribute(id, "weight"));
    std::array<uint32_t, 2> weights{};
    for (const auto v : joined.vertices()) {
        if (view[*v] == 5.f) ++weights[0];
        else if (view[*v] == -1.f) ++weights[1];
    }
    expect(weights[0] == vertices && weights[1] == vertices);
}
void ExpectSmoothedTowardSphere(ProcedureFixture &f, const BuiltGraph &g) {
    const auto coarse = f.WithSource(g.Mesh, [&](uint32_t id) { return test::EnclosedVolume(Mesh{f.Store(), id}).value_or(-1.0); });
    const Mesh output{f.Store(), f.Output(g.Mesh)};
    const auto volume = test::EnclosedVolume(output).value_or(-1.0), sphere = 4.0 / 3.0 * std::numbers::pi;
    expect(output.HasClosedSurface() && volume > coarse && std::abs(volume - sphere) < .5 * std::abs(coarse - sphere)) << "coarse" << coarse << "smoothed" << volume;
}
void ExpectSphere(ProcedureFixture &f, state::Entity mesh, double radius) {
    const Mesh output{f.Store(), f.Output(mesh)};
    const auto volume = test::EnclosedVolume(output).value_or(-1.0), expected = 4.0 / 3.0 * std::numbers::pi * radius * radius * radius;
    expect(output.HasClosedSurface() && std::abs(volume - expected) < .03 * expected) << "volume" << volume << "vs" << expected;
}
// One Catmull-Clark level turns the cube into 24 quads whose corners sit at five ninths, with the heat of each corner blending into the new points.
void ExpectCatmullClarkCube(ProcedureFixture &f, const BuiltGraph &g) {
    const Mesh output{f.Store(), f.Output(g.Mesh)};
    expect(output.VertexCount() == 26u && output.FaceCount() == 24u) << "vertices" << output.VertexCount() << "faces" << output.FaceCount();
    const auto heat = f.Store().FindAttribute(f.Output(g.Mesh), "heat");
    expect(heat != InvalidOffset);
    uint32_t corners = 0u, edge_points = 0u, face_points = 0u;
    for (const auto v : output.vertices()) {
        const auto p = test::Position(output, v);
        const auto magnitudes = std::array{std::abs(p.x), std::abs(p.y), std::abs(p.z)};
        const auto zeros = std::ranges::count_if(magnitudes, [](float m) { return m < 1e-6f; });
        if (zeros == 0u) corners += std::ranges::all_of(magnitudes, [](float m) { return std::abs(m - 5.f / 9.f) < 1e-5f; });
        if (zeros == 1u) edge_points += std::ranges::count_if(magnitudes, [](float m) { return std::abs(m - .75f) < 1e-5f; }) == 2;
        if (zeros == 2u) face_points += std::ranges::any_of(magnitudes, [](float m) { return std::abs(m - 1.f) < 1e-5f; });
        if (heat != InvalidOffset) expect(std::abs(f.Store().AttributeView<float>(heat)[*v] - p.z) < 1e-5f) << "heat at" << p.z;
    }
    expect(corners == 8u && edge_points == 12u && face_points == 6u) << corners << edge_points << face_points;
}
void ExpectOneFaceOfArea(ProcedureFixture &f, state::Entity mesh, double expected) {
    const Mesh output{f.Store(), f.Output(mesh)};
    double area = 0.0;
    for (const auto face : output.faces()) area += Length(test::AreaVector(output, face));
    expect(output.FaceCount() == 1u && std::abs(area - expected) < 1e-3 * expected) << "faces" << output.FaceCount() << "area" << area << "vs" << expected;
}
void ExpectSourceInstances(ProcedureFixture &f, const BuiltGraph &g) {
    const auto &r = f.R;
    const auto plane = ShapePositions(primitive::Plane{});
    const auto &entities = r.get<const ProcedureInstances>(g.Mesh).Entities;
    expect(!r.any_of<NodeError>(g.Nodes[2]));
    expect(fatal(entities.size() == plane.size()));
    const auto instanced = r.get<const Instance>(entities.front()).Entity;
    const auto placed = r.get<const MeshHandle>(instanced).StoreId;
    expect(instanced != g.Mesh && placed != f.Output(g.Mesh) && Near(Describe(f.Store(), placed).Vertices, plane));
    expect(Near(SourcePoints(f, g.Mesh, [](uint32_t, vec3 p) { return p; }), plane) && f.Shaped(g.Mesh).Vertices != plane);
}
// An instancer without an Instance input faults and passes its points through.
void UnlinkInstance(ProcedureFixture &f, state::Entity mesh) {
    const auto instancer = f.Node(mesh, 2u);
    f.Do(action::nodes::Unlink{mesh, instancer, 1u});
    expect(f.R.all_of<NodeError>(instancer) && f.R.get<const NodeError>(instancer).Fault == NodeFault::NoMesh);
    const Mesh points{f.Store(), f.Output(mesh)};
    const auto *instances = f.R.try_get<const ProcedureInstances>(mesh);
    expect(points.VertexCount() == 4u && points.FaceCount() == 0u && (!instances || instances->Entities.empty()));
}
// The joined vertex blocks lie in descending order, and the Delete still removes every vertex at or above y = 1.5.
void ExpectJoinedSelection(ProcedureFixture &f, const BuiltGraph &g) {
    const auto &meshes = f.Store();
    expect(!f.R.any_of<NodeError>(g.Nodes[4]) && !f.R.any_of<NodeError>(g.Nodes[5]));
    const auto id = f.Output(g.Mesh);
    const auto blocks = meshes.GetBlockList(id, MeshStore::ElementDomain::Vertex).Blocks;
    expect(fatal(blocks.size() == 2u));
    expect(blocks.back() < blocks.front());
    const Mesh output{meshes, id};
    expect(output.VertexCount() == 12u);
    for (const auto v : output.vertices()) expect(test::Position(output, v).y < 1.5f);
}
void ExpectFaces(ProcedureFixture &f, state::Entity mesh, uint32_t faces) {
    const auto count = Mesh{f.Store(), f.Output(mesh)}.FaceCount();
    expect(count == faces) << "faces" << count;
}

// A sphere's cap above y = 0.9, with a vertex moved before its Source node's state is kept, insets through Position, Compare and Inset.
Graph CapInset(ProcedureFixture &f) {
    const auto [mesh, nodes] = Build(f, {primitive::UVSphere{.Slices = 128u, .Stacks = 64u}, {{KindOf<Position>}, {KindOf<Compare>, {Set<&Compare::Component>(VectorComponent::Y), Set<&Compare::B>(.9f)}}, {KindOf<action::mesh::Inset>}}, {{1, 2, 0}, {2, 3, 1}, {0, 3, 0}}, 3, MoveFirstVertex});
    const auto inset = nodes[3];
    const auto first = f.Shaped(mesh);
    const auto set = [&](float thickness) {
        f.Do(UpdateOf<&action::mesh::Inset::Thickness>(inset, thickness));
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
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto kernel = SourceReference(f, mesh, [&](uint32_t clone) {
                const Mesh sphere{f.Store(), clone};
                const auto cap = SelectionWhere(sphere, [&](he::VH v) { return test::Position(sphere, v).y > .9f; });
                f.Run(clone, {{.Op = MeshTopologyOp::InsetRegion, .Param0 = .06f, .Flags = TopologyFlagEvenOffset, .Selection = cap}});
                return GeometrySelection{};
            });
            expect(f.Shaped(mesh) == kernel.Output);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Inset::Thickness>(f.Node(mesh, 3u), .2f)); },
    };
}

// A Store Selection of the cube's upper face survives a later subdivision, so a Delete reading it through Named Attribute removes the four faces it became.
Graph StoreSelectionDelete(ProcedureFixture &f) {
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto store = f.Append(mesh, KindOf<StoreSelection>);
    const auto named = f.Append(mesh, KindOf<procedure::NamedAttribute>);
    const auto remove = f.Append(mesh, KindOf<action::mesh::Delete>);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Do(SetAttributeName{store, "upper"});
    f.Do(SetAttributeName{named, "upper"});
    f.Do(UpdateOf<&procedure::NamedAttribute::Type>(named, AttributeType::Bool));
    f.Do(UpdateOf<&action::mesh::Delete::Mode>(remove, action::mesh::DeleteMode::OnlyFaces));
    const Mesh cube{meshes, f.Output(mesh)};
    GeometrySelection upper;
    for (const auto face : cube.faces())
        if (test::Centroid(cube, face).y > .5f) upper.Faces.push_back(*face);
    f.Do(SetInlineSelection{store, std::make_unique<InlineSelection>(SelectionOf(upper, Element::Face))});
    f.Do(Link{mesh, named, 0u, remove, 1u});
    f.Route(mesh, {f.Node(mesh, 0u), store, remove});
    expect(Mesh{meshes, f.Output(mesh)}.FaceCount() == 5u);
    f.Do(InsertNode{mesh, subdivide, remove, 0u, 0u});
    return {
        mesh,
        [mesh, named, remove](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            expect(!f.R.any_of<NodeError>(named) && !f.R.any_of<NodeError>(remove));
            const auto kernel = SourceReference(f, mesh, [&](uint32_t clone) {
                f.Run(clone, {Op(MeshTopologyOp::Subdivide, 1.f, TopologyFlagSelectAll)});
                const Mesh subdivided{meshes, clone};
                GeometrySelection upper;
                for (const auto face : subdivided.faces())
                    if (test::Centroid(subdivided, face).y > .99f) upper.Faces.push_back(*face);
                expect(upper.Faces.size() == 4u);
                f.Run(clone, {{.Op = MeshTopologyOp(uint32_t(action::mesh::DeleteMode::OnlyFaces)), .Selection = upper}});
                return GeometrySelection{};
            });
            expect(f.Shaped(mesh) == kernel.Output);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 4u), 2u)); },
    };
}

std::map<uint32_t, float> Heat(const MeshStore &meshes, uint32_t id) {
    std::map<uint32_t, float> heat;
    const auto view = meshes.AttributeView<float>(meshes.FindAttribute(id, "heat"));
    meshes.Arenas().Vertices.ForEach(meshes.Get(id).Vertices, [&](ElementHandle v, ElementAddress) { heat.emplace(v.Value, view[v.Value]); });
    return heat;
}
// Retained vertices keep their heat, and an edge midpoint takes the mean of two corners and a face center the mean of four.
Graph HeatSubdivide(ProcedureFixture &f) {
    auto &meshes = f.Store();
    const auto mesh = f.AddPrepared(primitive::Cube{}, [&](ProcedureFixture &, uint32_t id) {
        const auto heat = meshes.AddAttribute(id, AttributeDomain::Vertex, AttributeType::Float, "heat").value();
        const auto vertices = meshes.Arenas().Vertices.Dense(meshes.Get(id).Vertices);
        std::vector<float> values(vertices.Count);
        for (uint32_t i = 0u; i < vertices.Count; ++i) values[i] = float(i * i);
        meshes.SetAttribute<float>(heat, 0u, values);
    });
    f.Do(action::selection::Select{f.Object(mesh)});
    const auto original = Heat(meshes, f.Output(mesh));
    f.EnterEdit(Element::Vertex);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{1u});
    return {mesh, [mesh, original](ProcedureFixture &f) {
                const auto split = Heat(f.Store(), f.Output(mesh));
                expect(split.size() > original.size());
                for (const auto [v, value] : original) expect(split.at(v) == value);
                std::vector<float> corners;
                corners.reserve(original.size());
                for (const auto [v, value] : original) corners.push_back(value);
                std::set<float> means;
                for (size_t a = 0u; a < corners.size(); ++a) {
                    for (size_t b = a + 1u; b < corners.size(); ++b) {
                        means.insert(.5f * (corners[a] + corners[b]));
                        for (size_t c = b + 1u; c < corners.size(); ++c)
                            for (size_t d = c + 1u; d < corners.size(); ++d) means.insert(.25f * (corners[a] + corners[b] + corners[c] + corners[d]));
                    }
                }
                for (const auto [v, value] : split)
                    if (!original.contains(v)) expect(means.contains(value)) << "vertex" << v << "heat" << value;
            }};
}

// The Cross Section macro across a cone's Y axis leaves a disc whose radius shrinks with height, and the section moves with the plane.
Graph CrossSectionOfCone(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cylinder{.TopRadius = 0.f, .Slices = 32});
    const auto source = f.Node(mesh, 0u);
    f.Do(action::nodes::AddMacro{mesh, procedure::CrossSectionMacro, {}});
    const auto section = procedure::ActiveNodeOf(f.R, mesh);
    const auto ports = PortIds(f.R, section);
    expect(fatal(ports.size() == 4u));
    f.Do(Link{mesh, source, f.Out(source), section, ports[0]});
    f.Do(action::nodes::SetPortValue{section, ports[2], {0u, std::bit_cast<uint32_t>(1.f), 0u, 0u}});
    f.Do(SetDisplay{mesh, section, ports[3]});
    // A section of radius r is the 32-gon of area 16 r^2 sin(pi / 16).
    const auto disc = [](ProcedureFixture &f, state::Entity mesh, double radius) { ExpectOneFaceOfArea(f, mesh, 16.0 * radius * radius * std::sin(std::numbers::pi / 16.0)); };
    return {
        mesh,
        [mesh, disc](ProcedureFixture &f) { disc(f, mesh, .5f); },
        [disc, point = ports[1]](ProcedureFixture &f, state::Entity mesh) {
            f.Do(action::nodes::SetPortValue{f.R.get<const GeometryProcedure>(mesh).DisplayNode, point, {0u, std::bit_cast<uint32_t>(-.5f), 0u, 0u}});
            disc(f, mesh, .75f);
        },
    };
}

// The split macro's outside and inside of the cube cut by a bar hold the cube's volume between them.
Graph SplitBySolid(ProcedureFixture &f) {
    const auto [mesh, ports] = CubeAndBarInto(f, SplitBySolidMacro);
    expect(ports.size() == 4u);
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto volume = OutputVolume(f, mesh);
            expect(std::abs(volume - 6.0) < 1e-4) << "outside volume" << volume;
        },
        [inside = ports[3]](ProcedureFixture &f, state::Entity mesh) {
            f.Do(SetDisplay{mesh, f.Node(mesh, 2u), inside});
            const auto volume = OutputVolume(f, mesh);
            expect(std::abs(volume - 2.0) < 1e-4) << "inside volume" << volume;
        },
    };
}
// The hull macro over the cube and a bar through it adds a frustum of 7/3 at each end.
Graph HullOfObjects(ProcedureFixture &f) {
    const auto [mesh, ports] = CubeAndBarInto(f, HullOfObjectsMacro);
    expect(ports.size() == 3u);
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const Mesh hull{f.Store(), f.Output(mesh)};
            const auto volume = test::EnclosedVolume(hull).value_or(-1.0);
            expect(std::abs(volume - (8.0 + 14.0 / 3.0)) < 1e-3) << "hull volume" << volume;
            expect(hull.VertexCount() == 16u && hull.FaceCount() == 28u) << "hull elements" << hull.VertexCount() << hull.FaceCount();
            expect(hull.HasClosedSurface() && test::ClosedComponentsOutward(hull));
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 1u), vec3{3.f, .5f, .5f})); },
    };
}

// The boolean action makes the active object's mesh the active cube minus the other selected cube, in the active object's space.
Graph BooleanAction(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto target = f.AddMesh(primitive::Cube{});
    const auto active = FindActiveEntity(r);
    f.Do(action::object::AddMeshPrimitive{primitive::Cube{}, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = {.P = {1.f, 1.f, 1.f}}, .Select = MeshInstanceCreateInfo::SelectBehavior::Additive})});
    expect(FindActiveEntity(r) == active);
    f.Do(action::mesh::Boolean{BooleanOperation::Difference});
    return {target, [target, active](ProcedureFixture &f) {
                expect(std::abs(OutputVolume(f, target) - 7.0) < 1e-4);
                for (const auto e : f.R.view<const Instance>())
                    if (e != active) expect(Mesh{f.Store(), f.Output(f.R.get<const Instance>(e).Entity)}.VertexCount() == 8u);
            }};
}

Graph GeneratorEdit(ProcedureFixture &f) {
    const auto mesh = f.AddGenerated(primitive::Cube{});
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 0u), vec3{2.f, .5f, 1.f}));
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 0u), vec3{1.f, 1.f, 3.f}));
    return {mesh, [mesh](ProcedureFixture &f) { expect(Near(f.Shaped(mesh).Vertices, ShapePositions(primitive::Cube{{1.f, 1.f, 3.f}}))); }};
}

// An Object Info node reading its own procedure's mesh errs and forwards the source.
// One reading a mesh with a procedure of its own cooks after that procedure, so a resize there reaches the join.
Graph ObjectInfo(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto plane = f.Shaped(mesh);
    const auto info = f.Append(mesh, KindOf<procedure::ObjectInfo>);
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(info, f.Object(mesh)));
    const auto join = f.Append(mesh, KindOf<Join>);
    f.Do(Link{mesh, f.Node(mesh, 0u), 0u, join, 0u});
    f.Do(Link{mesh, info, 0u, join, 1u});
    f.Do(SetDisplay{mesh, join});
    expect(r.all_of<NodeError>(info) && !r.any_of<NodeError>(join));
    expect(f.Shaped(mesh) == plane);
    const auto cube = f.AddMesh(primitive::Cube{});
    const auto box = f.Append(cube, KindOf<primitive::Cube>);
    f.Do(SetDisplay{cube, box});
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(info, f.Object(cube)));
    expect(!r.any_of<NodeError>(info));
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(box, vec3{.5f, .25f, .5f}));
    return {mesh, [mesh](ProcedureFixture &f) {
                auto points = ShapePositions(primitive::Plane{});
                points.append_range(ShapePositions(primitive::Cube{{.5f, .25f, .5f}}));
                std::ranges::sort(points);
                expect(Mesh{f.Store(), f.Output(mesh)}.FaceCount() == 7u && Near(f.Shaped(mesh).Vertices, points));
            }};
}

// Instances on the plane's points publish as entities or realize as geometry, following scale, plane and cube edits.
template<bool Borrowed, bool Realize> Graph Instances(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddGenerated(primitive::Plane{});
    const auto object = FindActiveEntity(r);
    const auto cube_mesh = f.AddGenerated(primitive::Cube{});
    const auto to_points = f.Append(mesh, KindOf<MeshToPoints>);
    const auto instancer = f.Append(mesh, KindOf<InstanceOnPoints>);
    const auto instance = f.Append(mesh, Borrowed ? KindOf<procedure::ObjectInfo> : KindOf<primitive::Cube>);
    if constexpr (Borrowed) f.Do(UpdateOf<&procedure::ObjectInfo::Object>(instance, f.Object(cube_mesh)));
    f.Route(mesh, {f.Node(mesh, 0u), to_points, instancer});
    f.Do(Link{mesh, instance, f.Out(instance), instancer, 1u});
    if constexpr (Realize) {
        const auto realizer = f.Append(mesh, KindOf<RealizeInstances>);
        f.Route(mesh, {instancer, realizer});
    }
    f.Do(UpdateOf<&InstanceOnPoints::Scale>(instancer, vec3{.5f, .5f, .5f}));
    f.Do(UpdateOf<&primitive::Plane::HalfExtents>(f.Node(mesh, 0u), vec2{2.f, 3.f}));
    if constexpr (Borrowed) f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(cube_mesh, 0u), vec3{.5f, .25f, .5f}));
    else f.Do(UpdateOf<&primitive::Cube::HalfExtents>(instance, vec3{.5f, .25f, .5f}));
    Graph graph{mesh, [=](ProcedureFixture &f) {
                    const auto &r = f.R;
                    const auto plane = ShapePositions(primitive::Plane{{2.f, 3.f}}), cube = ShapePositions(primitive::Cube{{.5f, .25f, .5f}});
                    if constexpr (Realize) {
                        std::vector<Corner> realized;
                        for (const auto &p : plane)
                            for (const auto &c : cube) realized.push_back({p[0] + .5f * c[0], p[1] + .5f * c[1], p[2] + .5f * c[2]});
                        std::ranges::sort(realized);
                        expect(Near(f.Shaped(mesh).Vertices, realized));
                    } else {
                        const auto &entities = r.get<const ProcedureInstances>(mesh).Entities;
                        expect(entities.size() == 4u);
                        std::vector<Corner> points;
                        for (const auto e : entities) {
                            expect(ParentOrNull(r, e) == object);
                            const auto &t = r.get<const Transform>(e);
                            expect(t.S == vec3{.5f, .5f, .5f});
                            points.push_back({t.P.x, t.P.y, t.P.z});
                            expect(Near(f.Shaped(r.get<const Instance>(e).Entity).Vertices, cube));
                        }
                        std::ranges::sort(points);
                        expect(Near(points, plane));
                    }
                }};
    if constexpr (Realize) graph.Edit = [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&InstanceOnPoints::Scale>(f.Node(mesh, 2u), vec3{.25f, .25f, .25f})); };
    return graph;
}

// Selecting, duplicating, inserting and deleting nodes leaves the output the chain they form.
Graph NodeEditing(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto source = f.Shaped(mesh);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    const auto triangulate = f.Append(mesh, KindOf<action::mesh::SplitFaces>);
    const auto position = f.Append(mesh, KindOf<Position>);
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide, 3u));
    const auto cooks = Cooks(r);
    const auto shown = r.get<const GeometryProcedure>(mesh).DisplayNode;
    f.Do(SelectItems{mesh, {subdivide}, {{subdivide}, {triangulate}}});
    f.Do(UpdateOf<&NodeLayout::Position>(subdivide, uvec2{12u, 4u}));
    expect(Cooks(r) == cooks);
    expect(r.get<const GeometryProcedure>(mesh).DisplayNode == shown && f.Shaped(mesh) == source);
    f.Do(DuplicateNodes{mesh, {subdivide, triangulate}});
    const auto &procedure = r.get<const CookPlan>(mesh);
    expect(procedure.Nodes.size() == 6u && r.get<const action::mesh::Subdivide>(procedure.Nodes[4].Entity).Cuts == 3u);
    expect(f.Shaped(mesh) == source);
    f.Route(mesh, {f.Node(mesh, 0u), triangulate});
    f.Fails(InsertNode{mesh, position, triangulate, 0u, 0u});
    f.Do(SelectItems{mesh, {subdivide}, {{subdivide}}});
    f.Stage(InsertNode{mesh, subdivide, triangulate, 0u, 40u});
    f.CommitGesture();
    const auto refined = f.Shaped(mesh);
    f.Do(action::nodes::SetBypass{subdivide, true});
    expect(f.Shaped(mesh) != refined);
    expect(r.get<const CookPlan>(mesh).Wires[r.get<const CookPlan>(mesh).InputFirst[2u]].FromNode == 1u);
    f.Do(action::nodes::SetBypass{subdivide, false});
    expect(f.Shaped(mesh) == refined);
    f.Do(DeleteNodes{mesh, {subdivide}});
    f.Undo();
    f.Do(DeleteNodes{mesh, {subdivide, triangulate}});
    expect(f.Shaped(mesh) == source && r.get<const CookPlan>(mesh).DisplayNode == 0u);
    f.Undo();
    return {mesh};
}

float WidthAt(const Mesh &mesh, he::VH v) {
    const auto p = test::Position(mesh, v);
    return .05f * (p.x + p.y + p.z) + .2f;
}

// Linked values override parameter constants, linked attributes set each element's parameter, and an attribute from another domain is rejected.
void ParameterSockets(ProcedureFixture &f) {
    auto &r = f.R;
    auto &meshes = f.Store();
    using action::mesh::Bevel, action::mesh::ShrinkFatten, action::mesh::Poke;
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto source = f.Shaped(mesh);
    const auto output = [&] { return f.Shaped(mesh); };
    const auto real = f.Append(mesh, KindOf<FloatValue>);
    {
        const auto bevel = f.Append(mesh, KindOf<Bevel>);
        f.Route(mesh, {f.Node(mesh, 0u), bevel});
        const auto one = output();
        f.Do(UpdateOf<&Bevel::Segments>(bevel, 3u));
        const auto three = output();
        expect(three != one);
        f.Do(UpdateOf<&Bevel::Segments>(bevel, 1u));
        const auto integer = f.Append(mesh, KindOf<IntegerValue>);
        f.Do(UpdateOf<&IntegerValue::Value>(integer, 3u));
        const auto segments = ParameterSocket<&Bevel::Segments>, width = ParameterSocket<&Bevel::Width>;
        f.Do(Link{mesh, integer, 0u, bevel, segments});
        expect(output() == three);
        f.Do(UpdateOf<&Bevel::Segments>(bevel, 5u));
        expect(output() == three);
        f.Do(UpdateOf<&IntegerValue::Value>(integer, 1u));
        expect(output() == one);
        f.Do(action::nodes::Unlink{mesh, bevel, segments});
        f.Do(UpdateOf<&Bevel::Segments>(bevel, 3u));
        expect(output() == three);
        const auto position = f.Append(mesh, KindOf<Position>);
        f.Fails(Link{mesh, real, 0u, bevel, segments});
        f.Fails(Link{mesh, position, 0u, bevel, segments});
        f.Do(UpdateOf<&FloatValue::Value>(real, .2f));
        f.Do(Link{mesh, real, 0u, bevel, width});
        f.Do(UpdateOf<&Bevel::Width>(bevel, .3f));
        expect(output() == SourceReference(f, mesh, [&](uint32_t id) {
                               f.Run(id, {Op(MeshTopologyOp::BevelEdges, .2f, TopologyFlagSelectAll, 3u)});
                               return GeometrySelection{};
                           }).Output);
        f.Do(action::nodes::Unlink{mesh, bevel, width});
        f.Do(UpdateOf<&Bevel::Width>(bevel, .1f));
        const auto flag = f.Append(mesh, KindOf<BooleanValue>);
        f.Do(Link{mesh, flag, 0u, bevel, 1u});
        expect(output() == source);
        f.Do(UpdateOf<&BooleanValue::Value>(flag, true));
        expect(output() == three);
        f.Do(DeleteNodes{mesh, {bevel, position}});
    }
    const auto position = f.Append(mesh, KindOf<Position>);
    const auto dot = f.Append(mesh, KindOf<VectorMath>);
    const auto add = f.Append(mesh, KindOf<FloatMath>);
    f.Do(UpdateOf<&VectorMath::Operation>(dot, VectorMathOp::Dot));
    f.Do(UpdateOf<&VectorMath::B>(dot, vec3{.05f, .05f, .05f}));
    f.Do(UpdateOf<&FloatMath::B>(add, .2f));
    f.Do(Link{mesh, position, 0u, dot, 0u});
    f.Do(Link{mesh, dot, f.Out(dot), add, 0u});
    const auto reference = [&](MeshTopologyOp op, auto &&parameter) {
        return SourceReference(f, mesh, [&](uint32_t clone) {
                   auto task = Op(op, 0.f, TopologyFlagSelectAll);
                   task.Param0Attribute = parameter(clone);
                   f.Run(clone, {task});
                   return GeometrySelection{};
               })
            .Output;
    };
    const auto widths = [&](uint32_t clone) { return meshes.AttributeRef(AddValues<float>(meshes, clone, "parameter", WidthAt)); };
    const auto add_operator = [&](uint16_t kind, uint16_t socket) {
        const auto node = f.Append(mesh, kind);
        f.Do(Link{mesh, add, f.Out(add), node, socket});
        f.Route(mesh, {f.Node(mesh, 0u), node});
        return node;
    };
    {
        const auto bevel = add_operator(KindOf<Bevel>, ParameterSocket<&Bevel::Width>);
        f.Do(UpdateOf<&Bevel::Vertices>(bevel, true));
        expect(output() == reference(MeshTopologyOp::BevelVertices, widths));
        f.Do(DeleteNodes{mesh, {bevel}});
    }
    {
        const auto shrink = add_operator(KindOf<ShrinkFatten>, ParameterSocket<&ShrinkFatten::Distance>);
        // The kernel reads the factor at each vertex from the attribute, past the constant factor's zero check.
        const auto kernel = SourceReference(f, mesh, [&](uint32_t clone) {
            const PositionOperationTarget target{.StoreId = clone, .Selection = AllHandles(Mesh{meshes, clone})};
            ExecuteGeometryPositions(r, target, PositionEditOp::ShrinkFatten, 1.f, 1u, {.Factor = widths(clone)});
            return GeometrySelection{};
        });
        expect(kernel.Output != source && output() == kernel.Output);
        f.Do(DeleteNodes{mesh, {shrink}});
    }
    {
        // A vertex attribute does not fit the poke's per-face offset, and a face normal's z does.
        const auto poke = add_operator(KindOf<Poke>, ParameterSocket<&Poke::Offset>);
        expect(output() == source);
        const auto normal = f.Append(mesh, KindOf<Normal>);
        f.Do(UpdateOf<&Normal::Domain>(normal, AttributeDomain::Face));
        f.Do(UpdateOf<&VectorMath::B>(dot, vec3{0.f, 0.f, .3f}));
        f.Do(Link{mesh, normal, 0u, dot, 0u});
        f.Do(Link{mesh, dot, f.Out(dot), poke, ParameterSocket<&Poke::Offset>});
        const auto offsets = [&](uint32_t clone) { return meshes.AttributeRef(AddValues<float, AttributeDomain::Face>(meshes, clone, "parameter", [](const Mesh &m, he::FH face) { return .3f * test::Normal(m, face).z; })); };
        expect(output() == reference(MeshTopologyOp::Poke, offsets));
        f.Do(DeleteNodes{mesh, {poke}});
    }
    {
        const auto transform = f.Append(mesh, KindOf<TransformElements>);
        const auto vector = f.Append(mesh, KindOf<VectorValue>);
        const auto rotation = f.Append(mesh, KindOf<RotationValue>);
        // The rotation leaves no two moved corners a GPU rounding apart in sort order.
        const Transform delta{{1.f, 2.f, 3.f}, numeric::AngleAxis(.5f, Normalize(vec3{1.f, 2.f, 3.f})), {2.f, 1.f, .5f}};
        f.Do(UpdateOf<&VectorValue::Value>(vector, delta.P));
        f.Do(UpdateOf<&RotationValue::Value>(rotation, delta.R));
        f.Do(UpdateOf<&TransformElements::Delta, &Transform::S>(transform, delta.S));
        f.Do(Link{mesh, vector, 0u, transform, ParameterSocket<&TransformElements::Delta, &Transform::P>});
        f.Do(Link{mesh, rotation, 0u, transform, ParameterSocket<&TransformElements::Delta, &Transform::R>});
        f.Route(mesh, {f.Node(mesh, 0u), transform});
        expect(Near(f.Shaped(mesh).Vertices, SourcePoints(f, mesh, [&](uint32_t, vec3 p) { return delta.R * (delta.S * p) + delta.P; })));
        f.Do(DeleteNodes{mesh, {transform, vector, rotation}});
    }
}

// A chained display path edits the record its Source node refers to in place, and the Source node's state keeps the geometry a cook from the root reads.
// Disconnecting the displayed path restores its upstream result, and collapsing captures the displayed geometry.
void SourceStates(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto source = f.SourceOf(mesh);
    const auto record = f.Output(mesh);
    const auto shape = f.Shaped(mesh);
    expect(source.StoreId == record && source.State != store::Hash128{});
    const auto extrude = f.Append(mesh, KindOf<procedure::Extrude>);
    f.Route(mesh, {f.Node(mesh, 0u), extrude});
    expect(f.Output(mesh) == record && f.SourceOf(mesh) == source);
    const auto extruded = f.Shaped(mesh);
    expect(extruded != shape);
    f.Do(DeleteNodes{mesh, {extrude}});
    expect(f.Shaped(mesh) == shape && f.SourceOf(mesh) == source);
    f.Undo();
    expect(f.Shaped(mesh) == extruded);
    f.Redo();
    expect(f.Shaped(mesh) == shape);
    const auto boolean = f.Append(mesh, KindOf<procedure::Boolean>);
    f.Route(mesh, {f.Node(mesh, 0u), boolean});
    expect(f.Shaped(mesh) == shape && !r.any_of<NodeError>(boolean));
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Route(mesh, {boolean, subdivide});
    expect(Mesh{f.Store(), f.Output(mesh)}.FaceCount() == 24u);
    f.Do(action::nodes::Unlink{mesh, subdivide, 0u});
    expect(r.get<const GeometryProcedure>(mesh).DisplayNode == boolean && f.Shaped(mesh) == shape);
    f.Undo();
    expect(Mesh{f.Store(), f.Output(mesh)}.FaceCount() == 24u);
    f.Undo();
    expect(f.Shaped(mesh) == shape);
    f.Do(CollapseProcedure{mesh});
    expect(r.get<const CookPlan>(mesh).Nodes.size() == 1u && f.SourceOf(mesh) != source);
    expect(f.SourceOf(mesh).StoreId == f.Output(mesh) && f.Shaped(mesh) == shape);
}

// A viewport operator appends after the displayed node and displays its node, moving the displayed node's readers right.
// Removing the active node activates the displayed node, and deleting the displayed node displays the node its chain reads, or a Join's first input reads.
void DisplayedNode(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto source = f.Shaped(mesh);
    const auto extrude = f.Append(mesh, KindOf<procedure::Extrude>);
    f.Route(mesh, {f.Node(mesh, 0u), extrude});
    f.EnterEdit(Element::Face);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{1u});
    {
        const auto &procedure = r.get<const CookPlan>(mesh);
        expect(procedure.Nodes.size() == 3u && procedure.DisplayNode == 2u && ActiveIndex(r, procedure) == 2u);
        expect(procedure.Nodes[procedure.Inputs[procedure.InputFirst[2u]].FromNode].Entity == extrude);
    }
    const auto extrude_x = r.get<const NodeLayout>(extrude).Position.x;
    f.Do(SetDisplay{mesh, f.Node(mesh, 0u)});
    expect(f.Shaped(mesh) == source);
    f.Do(SelectItems{mesh, {extrude}, {{extrude}}});
    f.Do(SetDisplay{mesh, extrude});
    expect(r.get<const GeometryProcedure>(mesh).DisplayNode == extrude && f.Shaped(mesh) != source);
    f.Do(SetDisplay{mesh, f.Node(mesh, 0u)});
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{1u});
    {
        const auto &procedure = r.get<const CookPlan>(mesh);
        expect(procedure.Nodes.size() == 4u && procedure.DisplayNode == 3u && ActiveIndex(r, procedure) == 3u);
        expect(procedure.Inputs[procedure.InputFirst[3u]].FromNode == 0u && procedure.Inputs[procedure.InputFirst[1u]].FromNode == 0u);
        expect(r.get<const NodeLayout>(extrude).Position.x == extrude_x + AppendGap);
    }
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    f.Do(SelectItems{mesh, {extrude}, {{extrude}}});
    f.Do(DeleteNodes{mesh, {extrude}});
    {
        const auto &procedure = r.get<const CookPlan>(mesh);
        expect(procedure.Nodes.size() == 3u && procedure.DisplayNode == 2u && ActiveIndex(r, procedure) == 2u);
    }
    f.Do(DeleteNodes{mesh, {f.Node(mesh, 2u)}});
    {
        const auto &procedure = r.get<const CookPlan>(mesh);
        expect(procedure.Nodes.size() == 2u && procedure.DisplayNode == 0u && ActiveIndex(r, procedure) == 0u);
    }
    expect(f.Shaped(mesh) == source);
    const auto box = f.Append(mesh, KindOf<primitive::Cube>);
    const auto join = f.Append(mesh, KindOf<Join>);
    f.Do(Link{mesh, f.Node(mesh, 0u), 0u, join, 0u});
    f.Do(Link{mesh, box, f.Out(box), join, 1u});
    f.Do(SetDisplay{mesh, join});
    expect(f.Shaped(mesh) != source);
    f.Do(DeleteNodes{mesh, {join}});
    expect(r.get<const GeometryProcedure>(mesh).DisplayNode == f.Node(mesh, 0u) && f.Shaped(mesh) == source);

    for (const int depth : {0, 2}) {
        const auto owner = f.AddMesh(primitive::Cube{}), root = f.Node(owner, 0u);
        const auto cube = f.Shaped(owner);
        const auto bevel = f.Append(owner, KindOf<action::mesh::Bevel>), tail = f.Append(owner, KindOf<TransformElements>);
        f.Do(UpdateOf<&TransformElements::Delta, &Transform::P>(tail, vec3{1, 2, 3}));
        f.Do(Link{owner, bevel, f.Out(bevel), tail, 0u});
        auto input = bevel;
        uint16_t socket = 0u;
        for (int level = 0; level < depth; ++level) {
            f.Do(action::nodes::GroupNodes{owner, level == 0 ? std::vector{bevel, tail} : std::vector{input}});
            const auto group = r.get<const GeometryProcedure>(owner).Last;
            f.Do(action::nodes::ExposeSocket{group, input, socket, "Geometry"});
            input = group;
            socket = r.get<const SubnetInterface>(group).Ports.back().Id;
        }
        const auto displayed = [&] { return ResolveSocket(r, DisplayLink(r, r.get<const GeometryProcedure>(owner))).Source.FromNode; };
        for (const bool bypass : {false, true}) {
            f.Do(action::nodes::SetBypass{bevel, bypass});
            expect(r.get<const GeometryProcedure>(owner).DisplayNode == root && f.Shaped(owner) == cube);
            f.Do(Link{owner, root, f.Out(root), input, socket});
            expect(displayed() == tail && f.Shaped(owner) != cube);
            const auto result = f.Shaped(owner);
            f.Do(action::nodes::Unlink{owner, tail, 0u});
            expect(r.get<const GeometryProcedure>(owner).DisplayNode == bevel);
            f.Do(Link{owner, bevel, f.Out(bevel), tail, 0u});
            expect(displayed() == tail && f.Shaped(owner) == result);
            f.Do(action::nodes::Unlink{owner, input, socket});
            expect(r.get<const GeometryProcedure>(owner).DisplayNode == root && f.Shaped(owner) == cube);
        }
        if (depth) {
            f.Do(action::nodes::SetBypass{input, true});
            f.Do(Link{owner, root, f.Out(root), input, socket});
            f.Do(action::nodes::Unlink{owner, tail, 0u});
            expect(r.get<const GeometryProcedure>(owner).DisplayNode == input && f.Shaped(owner) == cube);
            f.Do(Link{owner, bevel, f.Out(bevel), tail, 0u});
            f.Do(action::nodes::Unlink{owner, input, socket});
            expect(r.get<const GeometryProcedure>(owner).DisplayNode == root && f.Shaped(owner) == cube);
            f.Do(action::nodes::SetBypass{input, false});
        }
        f.Do(AddNode{owner, KindOf<Join>, {}, ContainerOf(r, bevel)});
        const auto branch = r.get<const GeometryProcedure>(owner).Last;
        f.Do(Link{owner, bevel, f.Out(bevel), branch, 0u});
        f.Do(Link{owner, root, f.Out(root), input, socket});
        expect(r.get<const GeometryProcedure>(owner).DisplayNode == bevel);
        f.Do(Link{owner, tail, f.Out(tail), branch, 1u});
        f.Do(action::nodes::Unlink{owner, branch, 1u});
        expect(r.get<const GeometryProcedure>(owner).DisplayNode == bevel);
    }
}

// Editing input selections keeps Display and the viewport selection independent, and commits or cancels the whole task.
void ViewportSelection(ProcedureFixture &f) {
    auto &r = f.R;
    for (const auto [kind, vertices] : {std::pair{KindOf<procedure::Extrude>, false}, {KindOf<action::mesh::Bevel>, false}, {KindOf<action::mesh::Bevel>, true}, {KindOf<TransformElements>, false}}) {
        const auto mesh = kind == KindOf<action::mesh::Bevel> ? f.AddMesh(primitive::Cube{}) : f.AddGenerated(primitive::Cube{});
        const auto node = f.Append(mesh, kind);
        if (vertices) f.Do(UpdateOf<&action::mesh::Bevel::Vertices>(node, true));
        f.Route(mesh, {f.Node(mesh, 0u), node});
        if (kind == KindOf<TransformElements>) f.Do(action::nodes::GroupNodes{mesh, {node}});
        const auto display = r.get<const GeometryProcedure>(mesh).DisplayNode;
        for (const auto mode : {InteractionMode::Object, InteractionMode::Edit}) {
            f.Do(action::view::SetInteractionMode{mode});
            if (mode == InteractionMode::Edit) {
                f.Do(action::view::SetEditMode{Element::Vertex});
                f.Do(action::selection::ApplyEditElementClick{false, ElementHit{mesh, *GetMesh(r, mesh).VertexAt(0u)}});
            }
            for (const bool cancel : {true, false}) {
                const auto before = f.Shaped(mesh);
                const auto prior_elements = EditorGeometrySelection(f.Store(), f.Output(mesh));
                const auto present = f.P->History.Present;
                const auto selected = SelectedIn(r);
                const auto element = r.get<const EditMode>(state::Document).Value;
                const auto edit = [&](auto a) {
                    action::Emit(std::move(a));
                    f.P->Frame(action::Drain());
                    f.Audit();
                    expect(f.P->HasStaged() && f.P->History.Present == present);
                    expect(r.get<const GeometryProcedure>(mesh).DisplayNode == display);
                };
                edit(action::nodes::EditSelection{node, f.Object(mesh)});
                const auto input = r.get<const SelectionEditing>(state::Document).Mesh;
                expect(GetMesh(r, input).FaceCount() == 6u);
                edit(action::view::SetEditMode{Element::Face});
                edit(action::selection::DeselectAll{});
                expect(SelectedCount(r.get<const InlineSelection>(node).Faces) == 0u);
                f.Render();
                const Mesh cage = GetMesh(r, input);
                const auto hit = f.Pick(FacingFaceCenters(cage, f.View().CameraPosition).front(), Element::Face);
                expect(fatal(hit.has_value()));
                expect(hit->MeshEntity == input);
                edit(action::selection::ApplyEditElementClick{false, hit});
                expect(SelectedCount(r.get<const InlineSelection>(node).Faces) == 1u);
                const auto output = GetMesh(r, mesh);
                expect(output.HasClosedSurface());
                if (kind == KindOf<action::mesh::Bevel>) {
                    const auto shaped = f.Shaped(mesh);
                    const auto selected = f.Store().GetSelectedElements(f.Output(input), Element::Vertex);
                    for (const auto v : cage.vertices()) {
                        if (selected.Contains(*v)) continue;
                        const auto p = test::Position(cage, v);
                        expect(std::ranges::contains(shaped.Vertices, Corner{p.x, p.y, p.z}));
                    }
                }
                edit(action::selection::SelectAll{});
                expect(SelectedCount(r.get<const InlineSelection>(node).Faces) == 6u);
                if (cancel) {
                    action::Cancel();
                    f.P->Frame(action::Drain());
                    expect(f.P->History.Present == present && f.Shaped(mesh) == before);
                } else {
                    f.Do(action::nodes::FinishSelection{});
                    expect(f.P->History.Nodes[*f.P->History.Present].Parent == present);
                }
                expect(!f.P->HasStaged() && !r.all_of<SelectionEditing>(state::Document));
                expect(SelectedIn(r) == selected && r.get<const Interaction>(state::Document).Mode == mode && r.get<const EditMode>(state::Document).Value == element);
                expect(r.get<const GeometryProcedure>(mesh).DisplayNode == display);
                const auto restored = EditorGeometrySelection(f.Store(), f.Output(mesh));
                expect(restored.Vertices == prior_elements.Vertices && restored.Edges == prior_elements.Edges && restored.Faces == prior_elements.Faces);
            }
        }
    }
}

// An Unsubdivide node's first pass publishes the selection the kernel publishes, and its second pass reads that selection as the viewport operator's second iteration does.
Graph UnsubdivideSelection(ProcedureFixture &f) {
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Plane{});
    f.EnterEdit(Element::Vertex);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{.Cuts = 3u});
    const auto drawn = f.Output(mesh);
    const auto selected = EditorGeometrySelection(meshes, drawn);
    const auto pass = [&](uint32_t clone, const GeometrySelection &input) {
        auto task = UnsubdivideTask(f.R, Mesh{meshes, clone}, input);
        expect(fatal(task.has_value()));
        return PublishedKernel(f, clone, std::move(*task));
    };
    GeometrySelection first, second;
    const auto once = Reference(f, drawn, [&](uint32_t clone) { return first = pass(clone, selected); });
    const auto twice = Reference(f, drawn, [&](uint32_t clone) { return second = pass(clone, pass(clone, selected)); });
    f.Do(action::mesh::Unsubdivide{1u});
    const auto published = EditorGeometrySelection(meshes, f.Output(mesh));
    expect(published.Vertices == first.Vertices && published.Edges == first.Edges && published.Faces == first.Faces);
    expect(f.Shaped(mesh) == once.Output);
    f.Undo();
    f.Do(action::mesh::Unsubdivide{2u});
    return {
        mesh,
        [mesh, twice, second](ProcedureFixture &f) { expect(f.Shaped(mesh) == twice.Output && EditorGeometrySelection(f.Store(), f.Output(mesh)).Vertices == second.Vertices); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Unsubdivide::Iterations>(f.Node(mesh, 2u), 1u)); },
    };
}

// A change to a node committed in an earlier step restores the record to the result of the node before it.
void ColdRecook(ProcedureFixture &f) {
    const auto subdivide = Build(f, {primitive::Cube{}, {{KindOf<action::mesh::Subdivide>}, {KindOf<procedure::Extrude>}}, {{0, 1, 0}, {1, 2, 0}}, 2}).Nodes[1];
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide, 2u));
}

// A change to the first node after the Source restores the record to the Source node's result, with the position edit and the topology node after it run again.
void NodeResultRecook(ProcedureFixture &f) {
    const auto subdivide = Build(f, {primitive::Cube{}, {{KindOf<action::mesh::Subdivide>}, {KindOf<ToSphere>, {Set<&ToSphere::Factor>(.5f)}}, {KindOf<procedure::Extrude>}}, {{0, 1, 0}, {1, 2, 0}, {2, 3, 0}}, 3}).Nodes[1];
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide, 2u));
}

// Deleting the displayed tip node restores the record in place to its input node's result with one cook, which leaves the Source node's state untouched.
void DeleteTipRestores(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto root = f.Node(mesh, 0u);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    const auto extrude = f.Append(mesh, KindOf<procedure::Extrude>);
    f.Route(mesh, {root, subdivide});
    f.Route(mesh, {subdivide, extrude});
    const auto record = f.Output(mesh);
    const auto source = f.SourceOf(mesh);
    const auto cooks = Cooks(f.R);
    f.Do(DeleteNodes{mesh, {extrude}});
    expect(Cooks(f.R) == cooks + 1u);
    expect(f.Output(mesh) == record && f.SourceOf(mesh) == source);
}

// Grouping changes presentation only, and nested interfaces edit the same underlying fields.
void SceneNetworks(ProcedureFixture &f) {
    using namespace action::nodes;
    const auto cube = f.AddCube();
    const auto light = f.Add(action::object::AddLight{std::make_unique<ObjectCreateInfo>()});
    f.Do(AssignReference{{{light}, ReferenceKind::Parent}, {cube}});
    f.Do(action::animation::AddAnimation{"Animation"});
    for (const auto network : {FrameScene(f.R), state::Document}) {
        const auto items = NetworkItems(f.R, network);
        const auto shapes = Live::DrawnShapes(f);
        const auto cooks = Cooks(f.R);
        std::vector<state::Entity> groups;
        auto members = items;
        for (int depth = 0; depth < 2; ++depth) {
            f.Do(GroupItems{network, members});
            const auto group = f.R.get<const NetworkActive>(network).Item.Entity;
            groups.push_back(group);
            members = {{group}};
            f.Do(MoveNodes{network, {{{group}, {uint32_t(depth + 2), 3u}}}});
            expect(NetworkLeaves(f.R, group) == items);
            if (network != state::Document) {
                f.Do(ExposePort{group, PromotedField{light, state::Key<PunctualLight>(), action::detail::FieldOffset<&PunctualLight::Intensity>()}, std::make_unique<std::string>("Intensity")});
                f.Do(SetPortValue{group, f.R.get<const SubnetInterface>(group).Ports.back().Id, {std::bit_cast<uint32_t>(17.f + depth)}});
                expect(f.R.get<const PunctualLight>(light).Intensity == 17.f + depth);
            }
        }
        for (const auto group : groups | std::views::reverse) f.Do(UngroupItems{group});
        expect(NetworkItems(f.R, network) == items && Live::DrawnShapes(f) == shapes && Cooks(f.R) == cooks);
        expect(ParentOrNull(f.R, light) == cube);
    }
}

// Interactive and read-only drawing produce the same UI on every page, including after history navigation, and the history panel edits a recorded action.
void CanvasEquivalence(ProcedureFixture &f) {
    const auto mesh = f.AddGenerated(primitive::Cube{});
    const auto node = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Route(mesh, {f.Node(mesh, 0), node});
    f.Do(action::nodes::GroupNodes{mesh, {node}});
    const auto subnet = f.R.get<const GeometryProcedure>(mesh).Last;
    f.Do(action::physics::AddJointDef{});
    f.Do(action::animation::AddAnimation{"Animation"});
    auto *context = ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = {1400, 900};
    io.Fonts->Build();
    NodesPanel panel{{"Nodes"}};
    Panel parameters{"Parameters"};
    const auto frame = [&](bool interactive = false) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({950, 850});
        DrawNodesPanel(f.R, panel, interactive);
        ImGui::SetNextWindowPos({950, 0});
        ImGui::SetNextWindowSize({450, 850});
        DrawParametersPanel(f.R, panel, parameters, interactive);
        ImGui::Render();
        std::vector<std::byte> drawn;
        for (const auto *list : ImGui::GetDrawData()->CmdLists) {
            drawn.append_range(std::as_bytes(std::span{list->VtxBuffer.Data, size_t(list->VtxBuffer.Size)}));
            drawn.append_range(std::as_bytes(std::span{list->IdxBuffer.Data, size_t(list->IdxBuffer.Size)}));
        }
        expect(!drawn.empty());
        return drawn;
    };
    const auto cooks = Cooks(f.R);
    for (const auto network : {FrameScene(f.R), state::Document, mesh, subnet}) {
        EnterNetwork(panel, network);
        frame();
        frame();
        const auto drawn = frame(true);
        expect(frame() == drawn);
        f.Undo();
        f.Redo();
        expect(frame() == drawn);
        if (network == state::Document) {
            for (const auto item : NetworkItems(f.R, network)) {
                f.Do(SelectItems{network, item, {item}});
                frame();
            }
        }
    }
    expect(Cooks(f.R) == cooks);
    const auto object = f.AddEmpty("Object choice");
    f.AddEmpty("Other");
    const auto picked = f.Do(action::selection::Pick{false, ObjectHit{{object}}});
    const auto hit = [&]() -> std::optional<SelectionHit> & {
        return std::get<action::selection::Pick>(std::get<action::selection::Action>(f.P->DraftOf(picked).RecordedActions.front())).Hit;
    };
    HistoryPanel history{{"History"}};
    const auto history_frame = [&](ImGuiID activate = 0) {
        if (activate) ImGui::ActivateItemByID(activate);
        ImGui::NewFrame();
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({900, 900});
        project::DrawHistoryPanel(*f.P, history, true);
        ImGui::Render();
        f.P->Frame(action::Drain());
        f.Audit();
    };
    history_frame();
    const int index = 0;
    const auto action_id = ImHashData(&index, sizeof(index), ImHashData(&picked, sizeof(picked), ImGui::FindWindowByName("History")->ID));
    const auto present_id = ImHashStr("##present", 0, ImHashStr("Hit", 0, action_id));
    history_frame(present_id);
    expect(!hit());
    history_frame(present_id);
    expect(hit().has_value());
    expect(HitEntity(*hit()) == state::Null);
    expect(!f.R.all_of<Selected>(state::Document));
    f.Step(true);
    ImGui::DestroyContext(context);
}

state::Entity AnimatedAssembly(ProcedureFixture &f) {
    const auto root = f.AddEmpty("Assembly");
    const auto mesh = f.AddGenerated(primitive::Cube{});
    const auto operand = f.Object(mesh);
    const auto result = f.AddGenerated(primitive::Plane{});
    const auto info = f.Append(result, KindOf<procedure::ObjectInfo>);
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(info, operand));
    f.Do(SetDisplay{result, info});
    const auto light = f.Add(action::object::AddLight{std::make_unique<ObjectCreateInfo>()});
    for (const auto child : {operand, f.Object(result), light}) f.Do(action::object::SetParent{child, root});
    f.R.patch<Animations>(state::Document, [](auto &a) { a.Names = {"Animation"}; a.Active = 0; });
    f.R.emplace<AnimationClips>(root, std::vector<AnimationClip>{{0, {{animation::Target<&Transform::P>(), AnimationInterpolation::Linear, {0, 1}, {0, 0, 0, 2, 0, 0}, {0, 0, 0}}}}});
    f.R.emplace<AnimationClips>(light, std::vector<AnimationClip>{{0, {{animation::Target<&PunctualLight::Intensity>(), AnimationInterpolation::Linear, {0, 1}, {2, 8}, {2}}}}});
    f.Commit("Animate assembly");
    return root;
}
state::Entity Mechanism(ProcedureFixture &f) {
    const auto root = f.AddEmpty("Mechanism"), a = f.AddCube(), b = f.AddCube();
    for (const auto child : {a, b}) {
        f.Do(action::object::SetParent{child, root});
        f.Do(action::physics::SetMotionType{action::physics::SetMotionType::Type::Kinematic, child});
    }
    f.Do(UpdateOf<&Transform::P>(b, vec3{3, 0, 0}));
    const auto definition = f.R.create();
    f.R.emplace<PhysicsJointDef>(definition);
    f.R.emplace<PhysicsJoint>(a, b, definition);
    f.R.emplace<TriggerNodes>(root, std::vector{a, b});
    ModalModes modes;
    modes.Freqs = {440};
    modes.T60s = {1};
    modes.Shapes = {{{1, 0, 0}}};
    modes.Positions = {{-1, -1, -1}};
    modes.Vertices = {0};
    f.R.emplace<ModalModes>(a, modes);
    f.R.emplace<SoundVerticesModel>(a, SoundVerticesModel::Modal);
    f.R.emplace<ModalGain>(a);
    f.Commit("Bind mechanism domains");
    return root;
}
state::Entity ImportedRig(ProcedureFixture &f) {
    f.Do(action::io::LoadGltf{RiggedSimple});
    auto root = *f.R.view<const ArmatureObject>().begin();
    while (ParentOrNull(f.R, root) != state::Null) root = ParentOrNull(f.R, root);
    return root;
}

// Detaching a nested instance preserves its geometry and animated appearance.
void Hierarchies(ProcedureFixture &f, Fixture &other) {
    for (const auto [name, build] : std::array<std::pair<const char *, state::Entity (*)(ProcedureFixture &)>, 3>{{{"Assembly", AnimatedAssembly}, {"Mechanism", Mechanism}, {"Rig", ImportedRig}}}) {
        Scenario(f, other, std::string{"Hierarchy"} + name, [&](ProcedureFixture &f) {
            const auto source = build(f);
            const auto inner = f.Add(action::object::InstantiateHierarchy{source});
            const auto root = f.Add(action::object::InstantiateHierarchy{inner});
            f.Do(UpdateOf<&Transform::P>(root, vec3{2, 0, 0}));
            const auto members = f.R.get<const HierarchyInstance>(root).Members;
            auto edited = state::Null;
            for (const auto &member : members) {
                if (const auto *object = f.R.try_get<const Instance>(member.Entity); object && DrawnStoreId(f.R, object->Entity)) {
                    expect(f.Shaped(object->Entity) == f.Shaped(f.R.get<const Instance>(member.Source).Entity));
                    edited = HierarchySource(f.R, member.Entity);
                }
            }
            for (const auto &member : members) f.Do(action::object::SetVisible{false, HierarchySource(f.R, member.Entity)});
            f.Do(action::selection::Select{inner});
            f.Do(action::object::Delete{});
            f.Do(action::selection::DeselectAll{});
            std::array<std::vector<std::byte>, 2> images;
            for (int frame = 0; frame < 2; ++frame) {
                f.Do(action::timeline::SetFrame{1 + frame * 30});
                if (Render) images[frame] = f.Image();
            }
            f.Do(action::object::MakeIndependent{root});
            for (int frame = 0; frame < 2; ++frame) {
                f.Do(action::timeline::SetFrame{1 + frame * 30});
                if (Render) expect(f.Image() == images[frame]);
            }
            const auto independent = Live::DrawnShapes(f);
            f.Do(UpdateOf<&Transform::P>(edited, vec3{9, 0, 0}));
            expect(Live::DrawnShapes(f) == independent);
            if (Render) expect(f.Image() == images.back());
        });
    }
}

void ObjectDependencies(ProcedureFixture &f) {
    const auto a = f.AddGenerated(primitive::Cube{});
    const auto root = f.Node(a, 0u);
    const auto b = f.AddGenerated(primitive::Plane{});
    const auto bi = f.Append(b, KindOf<procedure::ObjectInfo>);
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(bi, f.Object(a)));
    f.Do(SetDisplay{b, bi});
    const auto c = f.AddGenerated(primitive::Plane{});
    const auto ci = f.Append(c, KindOf<procedure::ObjectInfo>);
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(ci, f.Object(b)));
    f.Do(SetDisplay{c, ci});
    auto cooks = Cooks(f.R);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(root, vec3{2.f, .5f, 1.f}));
    expect(Cooks(f.R) == cooks + 3u);
    expect(f.Shaped(a) == f.Shaped(b) && f.Shaped(a) == f.Shaped(c));
    const auto dormant = f.Append(a, KindOf<primitive::Cube>);
    cooks = Cooks(f.R);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(dormant, vec3{4.f, 1.f, 2.f}));
    expect(Cooks(f.R) == cooks);
    const auto cycle = f.Append(a, KindOf<procedure::ObjectInfo>);
    cooks = Cooks(f.R);
    f.Fails(UpdateOf<&procedure::ObjectInfo::Object>(cycle, f.Object(c)));
    expect(f.R.get<const procedure::ObjectInfo>(cycle).Object == state::Null && Cooks(f.R) == cooks);
    f.Do(action::selection::Select{f.Object(c)});
    f.Do(action::selection::ExtendActive{f.Object(a)});
    const auto node_count = f.R.get<const CookPlan>(a).Nodes.size();
    f.Fails(action::mesh::Boolean{BooleanOperation::Difference});
    expect(f.R.get<const CookPlan>(a).Nodes.size() == node_count && Cooks(f.R) == cooks);
    f.Do(SetDisplay{b, f.Node(b, 0u)});
    cooks = Cooks(f.R);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(root, vec3{1.f, 1.f, 3.f}));
    expect(Cooks(f.R) == cooks + 1u);
    expect(f.Shaped(b) == f.Shaped(c));
    f.Do(SetDisplay{b, bi});
    expect(f.Shaped(a) == f.Shaped(b) && f.Shaped(a) == f.Shaped(c));
    const auto object_a = f.Object(a);
    f.Do(action::selection::Select{object_a});
    f.Do(action::object::Delete{});
    expect(f.R.all_of<NodeError>(bi));
}

void HiddenSharedGeometry(ProcedureFixture &f) {
    const auto original = f.AddCube();
    const auto mesh = f.R.get<const Instance>(original).Entity;
    const auto shape = f.Shaped(mesh);
    const auto hidden = f.Add(action::object::DuplicateLinked{});
    f.Do(action::object::SetVisible{false, hidden});
    f.Do(action::selection::Select{original});
    f.Do(action::object::Delete{});
    expect(f.R.valid(mesh) && f.R.get<const Instance>(hidden).Entity == mesh);
    expect(f.Shaped(mesh) == shape);
    f.Do(action::object::SetVisible{true, hidden});
    expect(f.Shaped(mesh) == shape);
    f.Do(action::selection::Select{hidden});
    f.Do(action::object::Delete{});
    expect(!f.R.valid(mesh));
}

void ProceduralHierarchies(ProcedureFixture &f) {
    auto &r = f.R;
    const auto source = f.AddEmpty("Prototype");
    const auto light = f.Add(action::object::AddLight{std::make_unique<ObjectCreateInfo>()});
    f.Do(action::object::SetParent{light, source});
    const auto mesh = f.AddGenerated(primitive::Plane{});
    const auto producer = f.Object(mesh);
    const auto points = f.Append(mesh, KindOf<MeshToPoints>);
    const auto instances = f.Append(mesh, KindOf<InstanceOnPoints>);
    f.Do(UpdateOf<&InstanceOnPoints::Prototype>(instances, source));
    f.Route(mesh, {f.Node(mesh, 0u), points, instances});
    const auto roots = r.get<const ProcedureInstances>(mesh).Entities;
    expect(roots.size() == 4u);
    for (const auto root : roots) {
        expect(r.all_of<HierarchyInstance>(root));
        expect(ParentOrNull(r, root) == producer);
        expect(r.get<const HierarchyInstance>(root).Members.size() == 2u);
    }
    f.Do(UpdateOf<&PunctualLight::Intensity>(light, 23.f));
    expect(r.get<const ProcedureInstances>(mesh).Entities == roots);
    for (const auto root : roots) {
        const auto &members = r.get<const HierarchyInstance>(root).Members;
        const auto copied = std::ranges::find(members, light, &HierarchyInstance::Member::Source)->Entity;
        expect(r.get<const PunctualLight>(copied).Intensity == 23.f);
    }
    f.Fails(UpdateOf<&InstanceOnPoints::Prototype>(instances, producer));
    const auto realize = f.Append(mesh, KindOf<RealizeInstances>);
    f.Do(Link{mesh, instances, f.Out(instances), realize, 0});
    f.Do(SetDisplay{mesh, realize});
    expect(r.all_of<NodeError>(realize));
    expect(r.get<const ProcedureInstances>(mesh).Entities == roots);
    f.Do(SetDisplay{mesh, f.Node(mesh, 0u)});
    for (const auto root : roots) expect(!r.valid(root));
    f.Do(action::selection::Select{source});
    f.Do(action::object::Delete{});
    expect(r.all_of<HierarchyDefinition>(source));
    expect(r.valid(light));
    f.Do(SetDisplay{mesh, instances});
    expect(r.get<const ProcedureInstances>(mesh).Entities.size() == 4u);
    f.Do(SetDisplay{mesh, f.Node(mesh, 0u)});
    f.Do(UpdateOf<&InstanceOnPoints::Prototype>(instances, state::Null));
    expect(!r.valid(source) && !r.valid(light));
}

void LinkedSkinPoses(ProcedureFixture &f) {
    f.Do(action::io::LoadGltf{RiggedSimple});
    auto &r = f.R;
    const auto skinned = *r.view<const ArmatureModifier>().begin();
    const auto modifier = r.get<const ArmatureModifier>(skinned);
    const auto first = modifier.ArmatureObjectEntity;
    f.Do(action::selection::Select{first});
    const auto second = f.Add(action::object::DuplicateLinked{});
    if (r.all_of<HierarchyInstance>(second)) f.Do(action::object::MakeIndependent{second});
    f.Do(action::selection::Select{skinned});
    const auto mesh_copy = f.Add(action::object::DuplicateLinked{});
    r.patch<ArmatureModifier>(mesh_copy, [&](auto &binding) { binding.ArmatureObjectEntity = second; });
    f.Commit("Bind the copied skin to its armature occurrence");
    const auto target = f.Add(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Transform = {.P = {3.f, 0.f, 0.f}}})});
    const auto second_bone = r.get<const ArmatureObject>(second).BoneEntities.front();
    f.Do(action::bone::AddConstraint{action::bone::BoneConstraintKind::CopyTransforms, second_bone});
    f.Do(action::bone::SetConstraintTarget{0u, target, second_bone});
    const auto check = [&] {
        const auto &a = r.get<const ArmaturePoseState>(first), &b = r.get<const ArmaturePoseState>(second);
        const auto &layout = r.get<const PosedLayout>(r.get<const Instance>(skinned).Entity);
        expect(std::ranges::contains(layout.Armatures, first) && std::ranges::contains(layout.Armatures, second));
        expect(!a.GpuDeformRanges.empty() && a.GpuDeformRanges.size() == b.GpuDeformRanges.size()) << fatal;
        const auto &buffers = r.Resources.get<const GpuBuffers>();
        for (uint32_t skin = 0; skin < a.GpuDeformRanges.size(); ++skin) {
            expect(a.GpuDeformRanges[skin].Offset != b.GpuDeformRanges[skin].Offset);
            const auto one = buffers.ArmatureDeformBuffer.Get(a.GpuDeformRanges[skin]);
            const auto two = buffers.ArmatureDeformBuffer.Get(b.GpuDeformRanges[skin]);
            std::vector<mat4> expected(two.size());
            ComputeDeformMatrices(r.get<const Armature>(modifier.ArmatureEntity), skin, b.BonePoseWorld, expected);
            expect(std::ranges::equal(two, expected));
            expect(!std::ranges::equal(one, two));
        }
    };
    check();
    f.Undo();
    f.Redo();
    check();
}

using action::mesh::DeleteMode;
const std::array<GraphData, 23> DataGraphs{{
    {"ExtrudeSubdivide", {primitive::Plane{}, {{KindOf<procedure::Extrude>}, {KindOf<action::mesh::Subdivide>, {Set<&action::mesh::Subdivide::Cuts>(2u)}}}, {{0, 1, 0}, {1, 2, 0}}, 2}, nullptr, [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 2u), 3u)); }},
    {"ScalePositions", {primitive::Cube{}, {{KindOf<Position>}, {KindOf<VectorMath>, {Set<&VectorMath::Operation>(VectorMathOp::Scale), Set<&VectorMath::Scale>(2.f)}}, {KindOf<SetPosition>, {Set<&SetPosition::Offset>(false)}}}, {{1, 2, 0}, {2, 3, 1}, {0, 3, 0}}, 3}, [](ProcedureFixture &f, const BuiltGraph &g) { expect(Near(f.Shaped(g.Mesh).Vertices, SourcePoints(f, g.Mesh, [](uint32_t, vec3 p) { return 2.f * p; }))); }, [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&VectorMath::Scale>(f.Node(mesh, 2u), 3.f)); }},
    {"CompareSelectsExtrusion", {primitive::Cube{}, {{KindOf<Position>}, {KindOf<Compare>, {Set<&Compare::Component>(VectorComponent::Z)}}, {KindOf<procedure::Extrude>}}, {{1, 2, 0}, {2, 3, 1}, {0, 3, 0}}, 3}, ExpectUpperExtruded, [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&Compare::Component>(f.Node(mesh, 2u), VectorComponent::X)); }},
    {"NamedAttributeReads", {primitive::Plane{}, {{KindOf<procedure::NamedAttribute>, {Named("offset"), Set<&procedure::NamedAttribute::Type>(AttributeType::Vec3)}}, {KindOf<SetPosition>}}, {{1, 2, 1}, {0, 2, 0}}, 2, AddOffsets}, [](ProcedureFixture &f, const BuiltGraph &g) { expect(Near(f.Shaped(g.Mesh).Vertices, SourcePoints(f, g.Mesh, [](uint32_t v, vec3 p) { return p + OffsetOf(v); }))); }, [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&SetPosition::Offset>(f.Node(mesh, 2u), false)); }},
    // The rings at the major radius have zero curvature and stay with either comparison.
    {"CurvatureSelectsOuterTorus", {primitive::Torus{}, {{KindOf<Curvature>, {Set<&Curvature::Kind>(CurvatureKind::Gaussian)}}, {KindOf<Compare>, {Set<&Compare::B>(.1f)}}, {KindOf<action::mesh::Delete>}}, {{1, 2, 0}, {2, 3, 1}, {0, 3, 0}}, 3}, [](ProcedureFixture &f, const BuiltGraph &g) {
         const auto [outer, inner] = OuterInner(f, g.Mesh);
         expect(outer == 0u && inner > 0u) << "outer" << outer << "inner" << inner; }, [](ProcedureFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&Compare::Operation>(f.Node(mesh, 2u), CompareOp::Less));
         const auto [outer, inner] = OuterInner(f, mesh);
         expect(inner == 0u && outer > 0u) << "outer" << outer << "inner" << inner; }},
    {"StorePromotes", {primitive::Plane{}, {{KindOf<Position>}, {KindOf<StoreNamedAttribute>, {Named("rest")}}, {KindOf<action::mesh::Subdivide>, {Set<&action::mesh::Subdivide::Cuts>(2u)}}}, {{1, 2, 1}, {0, 2, 0}, {2, 3, 0}}, 3}, [](ProcedureFixture &f, const BuiltGraph &g) { expect(RestMatches(f, g.Mesh)); }, [](ProcedureFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 3u), 3u));
         expect(RestMatches(f, mesh)); }},
    // Vector Math on a vertex and a face attribute errs and forwards A, so Set Position offsets each vertex by its position.
    {"DomainMismatch", {primitive::Cube{}, {{KindOf<Position>}, {KindOf<Normal>, {Set<&Normal::Domain>(AttributeDomain::Face)}}, {KindOf<VectorMath>}, {KindOf<SetPosition>}}, {{1, 3, 0}, {2, 3, 1}, {3, 4, 1}, {0, 4, 0}}, 4}, [](ProcedureFixture &f, const BuiltGraph &g) {
         expect(f.R.any_of<NodeError>(g.Nodes[3]));
         expect(Near(f.Shaped(g.Mesh).Vertices, SourcePoints(f, g.Mesh, [](uint32_t, vec3 p) { return 2.f * p; }))); }, [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&Normal::Domain>(f.Node(mesh, 2u), AttributeDomain::Vertex)); }},
    {"JoinGenerated", {primitive::Cube{}, {{KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{2.f, .5f, 1.f})}}, {KindOf<Join>}}, {{0, 2, 0}, {1, 2, 1}}, 2, AddWeights}, ExpectJoinedDefaults, [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 1u), vec3{.5f, .5f, .5f})); }},
    {"BooleanIntersection", {primitive::Cube{}, {{KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{2.f, .25f, .75f})}}, {KindOf<procedure::Boolean>, {Set<&procedure::Boolean::Operation>(BooleanOperation::Intersection)}}}, {{0, 2, 0}, {1, 2, 1}}, 2}, [](ProcedureFixture &f, const BuiltGraph &g) { expect(std::abs(OutputVolume(f, g.Mesh) - 1.5) < 1e-4); }, [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&procedure::Boolean::Operation>(f.Node(mesh, 2u), BooleanOperation::Union)); }},
    {"BoundaryFillsHole", {primitive::Cube{}, {{KindOf<Position>}, {KindOf<Compare>, {Set<&Compare::Component>(VectorComponent::Z), Set<&Compare::B>(.9f)}}, {KindOf<action::mesh::Delete>, {Set<&action::mesh::Delete::Mode>(DeleteMode::Faces)}}, {KindOf<Boundary>}, {KindOf<action::mesh::Fill>}}, {{1, 2, 0}, {2, 3, 1}, {4, 5, 1}, {0, 3, 0}, {3, 5, 0}}, 5}, [](ProcedureFixture &f, const BuiltGraph &g) {
         const Mesh output{f.Store(), f.Output(g.Mesh)};
         expect(output.FaceCount() == 6u && output.HasClosedSurface()) << "faces" << output.FaceCount();
         expect(std::abs(test::EnclosedVolume(output).value_or(-1.0) - 8.0) < 1e-4); },
     // Deleting the top vertices takes the sides too, and Fill declines the loop that already bounds the remaining face.
     [](ProcedureFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&action::mesh::Delete::Mode>(f.Node(mesh, 3u), DeleteMode::Vertices));
         ExpectFaces(f, mesh, 1u);
     }},
    {"LinkedDeletesComponent", {primitive::Cube{}, {{KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{.5f, .5f, .5f})}}, {KindOf<Join>}, {KindOf<Position>}, {KindOf<Compare>, {Set<&Compare::Component>(VectorComponent::Z), Set<&Compare::B>(.9f)}}, {KindOf<Linked>}, {KindOf<action::mesh::Delete>}}, {{0, 2, 0}, {1, 2, 1}, {2, 6, 0}, {3, 4, 0}, {4, 5, 0}, {5, 6, 1}}, 6}, [](ProcedureFixture &f, const BuiltGraph &g) {
         const Mesh output{f.Store(), f.Output(g.Mesh)};
         expect(output.VertexCount() == 8u && output.FaceCount() == 6u) << "vertices" << output.VertexCount() << "faces" << output.FaceCount();
         expect(std::abs(test::EnclosedVolume(output).value_or(-1.0) - 1.0) < 1e-4); },
     // A threshold above every vertex seeds nothing, so both cubes stay.
     [](ProcedureFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&Compare::B>(f.Node(mesh, 4u), 10.f));
         ExpectFaces(f, mesh, 12u);
     }},
    // Seeding Adjacent, at More by one step, with the plane's three vertex columns past x = 1.5 grows the seed by the column beside it, which Delete removes with the seed.
    {"AdjacentDeletesRing", {primitive::Plane{.HalfExtents = {4.f, 4.f}, .Subdivisions = {8u, 8u}}, {{KindOf<Position>}, {KindOf<Compare>, {Set<&Compare::B>(1.5f)}}, {KindOf<Adjacent>}, {KindOf<action::mesh::Delete>}}, {{0, 4, 0}, {1, 2, 0}, {2, 3, 0}, {3, 4, 1}}, 4}, [](ProcedureFixture &f, const BuiltGraph &g) { ExpectColumnsThrough(f, g.Mesh, 0); },
     // Less shrinks the seed by its column beside the rest, so Delete removes the two columns past x = 2.5.
     [](ProcedureFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&Adjacent::Mode>(f.Node(mesh, 3u), AdjacentMode::Less));
         ExpectColumnsThrough(f, mesh, 2);
     }},
    // The primitive shades flat, which the patches would keep as creases, so a Shade node supplies the smooth normals.
    {"SmoothSurfaceIcoSphere", {primitive::IcoSphere{.Subdivisions = 1}, {{KindOf<Shade>}, {KindOf<SmoothSurface>}}, {{0, 1, 0}, {1, 2, 0}}, 2}, ExpectSmoothedTowardSphere, [](ProcedureFixture &f, state::Entity mesh) {
         const auto vertices = Mesh{f.Store(), f.Output(mesh)}.VertexCount();
         f.Do(UpdateOf<&SmoothSurface::Cuts>(f.Node(mesh, 2u), 4u));
         expect(Mesh{f.Store(), f.Output(mesh)}.VertexCount() > vertices);
     }},
    // A level set of one minus the distance from the origin is a sphere, which the level shrinks.
    {"LevelSetSphere", {primitive::Cube{}, {{KindOf<Position>}, {KindOf<VectorMath>, {Set<&VectorMath::Operation>(VectorMathOp::Length)}}, {KindOf<FloatMath>, {Set<&FloatMath::Operation>(FloatMathOp::Multiply), Set<&FloatMath::B>(-1.f)}}, {KindOf<FloatMath>, {Set<&FloatMath::B>(.6f)}}, {KindOf<LevelSet>}}, {{1, 2, 0}, {2, 3, 0}, {3, 4, 0}, {4, 5, 0}}, 5}, [](ProcedureFixture &f, const BuiltGraph &g) { ExpectSphere(f, g.Mesh, .6); }, [](ProcedureFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&LevelSet::Level>(f.Node(mesh, 5u), .2f));
         ExpectSphere(f, mesh, .4); }},
    {"SubdivisionSurfaceCube", {primitive::Cube{}, {{KindOf<SubdivisionSurface>}}, {{0, 1, 0}}, 1, AddHeightHeat}, ExpectCatmullClarkCube, [](ProcedureFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&SubdivisionSurface::Levels>(f.Node(mesh, 1u), 2u));
         expect(Mesh{f.Store(), f.Output(mesh)}.VertexCount() == 98u);
     }},
    // A cylinder seen along its axis projects to its cap polygon, and seen across it to a rectangle.
    {"ProjectCylinder", {primitive::Cylinder{.Slices = 32}, {{KindOf<Project>, {Set<&Project::Normal>(vec3{0.f, 1.f, 0.f})}}}, {{0, 1, 0}}, 1}, [](ProcedureFixture &f, const BuiltGraph &g) { ExpectOneFaceOfArea(f, g.Mesh, 16.0 * std::sin(std::numbers::pi / 16.0)); }, [](ProcedureFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&Project::Normal>(f.Node(mesh, 1u), vec3{0.f, 0.f, 1.f}));
         ExpectOneFaceOfArea(f, mesh, 4.0); }},
    {"MinkowskiBoxes", {primitive::Cube{}, {{KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{.5f, .25f, .125f})}}, {KindOf<Minkowski>}}, {{1, 2, 1}, {0, 2, 0}}, 2}, [](ProcedureFixture &f, const BuiltGraph &g) { expect(std::abs(OutputVolume(f, g.Mesh) - 3.0 * 2.5 * 2.25) < 1e-3) << "sum volume" << OutputVolume(f, g.Mesh); }, [](ProcedureFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 1u), vec3{1.f, .5f, .25f}));
         expect(std::abs(OutputVolume(f, mesh) - 4.0 * 3.0 * 2.5) < 1e-3) << "grown sum volume" << OutputVolume(f, mesh); }},
    {"SourceInstances", {primitive::Plane{}, {{KindOf<MeshToPoints>}, {KindOf<InstanceOnPoints>}}, {{0, 1, 0}, {1, 2, 0}, {0, 2, 1}}, 2}, ExpectSourceInstances, UnlinkInstance},
    // The source clone takes lower blocks than the generated cube and the Join appends them after the cube's, so the joined block table is not in ascending order.
    // A Compare linked into a Delete still selects the cube's upper vertices in its higher block.
    {"JoinedSelection", {primitive::Cube{}, {{KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{1.f, 2.f, 1.f})}}, {KindOf<Join>}, {KindOf<Position>}, {KindOf<Compare>, {Set<&Compare::Component>(VectorComponent::Y), Set<&Compare::B>(1.5f)}}, {KindOf<action::mesh::Delete>}}, {{1, 2, 0}, {0, 2, 1}, {3, 4, 0}, {4, 5, 1}, {2, 5, 0}}, 5}, ExpectJoinedSelection, [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&Compare::B>(f.Node(mesh, 4u), .5f)); }},
    // A plane past the torus's outer equator holds one equator edge, whose edge loop is the 32-edge equator, and Delete removes the 64 faces beside it.
    // Tilting the plane toward the tube holds one edge across the equator instead, whose loop is the 16-edge tube ring with 32 faces beside it.
    {"EdgeLoopDeletesRing", {primitive::Torus{}, {{KindOf<PlaneSide>, {Set<&PlaneSide::Point>(vec3{0.f, 0.f, 1.48f}), Set<&PlaneSide::Normal>(vec3{.0985f, 0.f, 1.f}), Set<&PlaneSide::Side>(PlaneSideKind::Positive), Set<&PlaneSide::Domain>(AttributeDomain::Edge)}}, {KindOf<procedure::EdgeLoop>}, {KindOf<action::mesh::Delete>, {Set<&action::mesh::Delete::Mode>(DeleteMode::Edges)}}}, {{1, 2, 0}, {2, 3, 1}, {0, 3, 0}}, 3}, [](ProcedureFixture &f, const BuiltGraph &g) { ExpectFaces(f, g.Mesh, 32u * 16u - 64u); }, [](ProcedureFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&PlaneSide::Normal>(f.Node(mesh, 1u), vec3{0.f, .1989f, 1.f}));
         ExpectFaces(f, mesh, 32u * 16u - 32u); }},
    // A face field selects in the face domain, so Mark Sharp smooths the top and +x faces.
    {"FaceFieldMarksSharp", {primitive::Cube{}, {{KindOf<PlaneSide>, {Set<&PlaneSide::Point>(vec3{-.5f, 0.f, -.5f}), Set<&PlaneSide::Normal>(vec3{1.f, 0.f, 1.f}), Set<&PlaneSide::Side>(PlaneSideKind::Positive), Set<&PlaneSide::Domain>(AttributeDomain::Face)}}, {KindOf<MarkSharp>, {Set<&MarkSharp::Sharp>(false)}}}, {{1, 2, 1}, {0, 2, 0}}, 2}, [](ProcedureFixture &f, const BuiltGraph &g) {
         const auto &meshes = f.Store();
         const Mesh marked{meshes, f.Output(g.Mesh)};
         std::vector<vec3> smooth;
         for (const auto face : marked.faces()) {
             if (!meshes.Arenas().FaceSharpness.Get({test::Address(marked, face), 1u})[0]) smooth.push_back(test::Normal(marked, face));
         }
         std::ranges::sort(smooth, {}, [](vec3 n) { return std::array{n.x, n.y, n.z}; });
         expect(smooth == std::vector<vec3>{{0.f, 0.f, 1.f}, {1.f, 0.f, 0.f}}); }, nullptr},
    {"FaceFieldFills", {primitive::Cube{}, {{KindOf<PlaneSide>, {Set<&PlaneSide::Point>(vec3{-.5f, 0.f, -.5f}), Set<&PlaneSide::Normal>(vec3{1.f, 0.f, 1.f}), Set<&PlaneSide::Side>(PlaneSideKind::Positive), Set<&PlaneSide::Domain>(AttributeDomain::Face)}}, {KindOf<action::mesh::Fill>}}, {{1, 2, 1}, {0, 2, 0}}, 2}, [](ProcedureFixture &f, const BuiltGraph &g) { ExpectFaces(f, g.Mesh, 5u); }, [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&PlaneSide::Side>(f.Node(mesh, 1u), PlaneSideKind::Negative)); }},
    {"StoreFaceValue", {primitive::Cube{}, {{KindOf<FloatValue>, {Set<&FloatValue::Value>(2.5f)}}, {KindOf<StoreNamedAttribute>, {Set<&StoreNamedAttribute::Domain>(AttributeDomain::Face), Named("heat")}}}, {{1, 2, 1}, {0, 2, 0}}, 2}, [](ProcedureFixture &f, const BuiltGraph &g) {
         const auto &meshes = f.Store();
         const auto heat = meshes.FindAttribute(f.Output(g.Mesh), "heat");
         expect(fatal(heat != InvalidOffset));
         expect(meshes.GetAttribute(heat).Domain == AttributeDomain::Face && meshes.GetAttribute(heat).Type == AttributeType::Float);
         const auto values = meshes.AttributeView<float>(heat);
         for (const auto face : Mesh{meshes, f.Output(g.Mesh)}.faces()) expect(values[*face] == 2.5f); }, nullptr},
}};
constexpr std::array<GraphRow, 13> BuiltGraphs{{
    {"CapInset", CapInset},
    {"StoreSelectionDelete", StoreSelectionDelete},
    {"HeatSubdivide", HeatSubdivide},
    {"CrossSectionOfCone", CrossSectionOfCone},
    {"SplitBySolid", SplitBySolid},
    {"HullOfObjects", HullOfObjects},
    {"BooleanAction", BooleanAction},
    {"GeneratorEdit", GeneratorEdit},
    {"ObjectInfo", ObjectInfo},
    {"InstancesBorrowedPublish", Instances<true, false>},
    {"InstancesGeneratedRealize", Instances<false, true>},
    {"NodeEditing", NodeEditing},
    {"UnsubdivideSelection", UnsubdivideSelection},
}};
} // namespace

void TestProcedures() {
    ProcedureFixture f;
    Fixture other;
    OperatorKinds(f, other);
    for (const auto &row : DataGraphs) {
        Survives(f, other, row.Name, [&row](ProcedureFixture &f) {
            const auto built = Build(f, row.Description);
            return Graph{built.Mesh, [&row, built](ProcedureFixture &f) { if (row.Oracle) row.Oracle(f, built); }, row.Edit};
        });
    }
    for (const auto &row : BuiltGraphs) Survives(f, other, row.Name, row.Build);
    TestOperators(f, other);
    Hierarchies(f, other);
    Scenario(f, other, "SceneNetworks", SceneNetworks);
    Scenario(f, other, "CanvasEquivalence", CanvasEquivalence);
    Scenario(f, other, "ParameterSockets", ParameterSockets);
    Scenario(f, other, "SourceStates", SourceStates);
    Scenario(f, other, "DisplayedNode", DisplayedNode);
    Scenario(f, other, "ViewportSelection", ViewportSelection);
    Scenario(f, other, "ColdRecook", ColdRecook);
    Scenario(f, other, "NodeResultRecook", NodeResultRecook);
    Scenario(f, other, "ExtrudeDisplacement", ExtrudeDisplacement);
    Scenario(f, other, "DeleteTipRestores", DeleteTipRestores);
    Scenario(f, other, "HiddenSharedGeometry", HiddenSharedGeometry);
    Scenario(f, other, "ProceduralHierarchies", ProceduralHierarchies);
    Scenario(f, other, "LinkedSkinPoses", LinkedSkinPoses);
    Scenario(f, other, "ObjectDependencies", ObjectDependencies);
}
