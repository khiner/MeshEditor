#include "GraphFixture.h"
#include "TopologyCases.h"
#include "animation/Fields.h"
#include "animation/Keying.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/ModalModes.h"
#include "gltf/SourceAssets.h"
#include "graph/NodeView.h"
#include "graph/OccurrenceImages.h"
#include "mesh/Unsubdivide.h"
#include "physics/PhysicsSystem.h"
#include "project/HistoryUi.h"
#include "render/GpuBufferOps.h"
#include "render/MaterialComponents.h"
#include "scene/Instance.h"
#include "scene/SceneGraph.h"
#include "viewport/FrameState.h"
#include "viewport/ViewportRenderGpu.h"
#include <imgui_internal.h>

using boost::ut::expect, boost::ut::fatal;

namespace {
using namespace graph;
using action::UpdateOf;
using action::nodes::AddNode, action::nodes::DeleteNodes, action::nodes::DuplicateNodes, action::nodes::InsertNode, action::nodes::Link, action::nodes::SelectItems, action::nodes::SetAttributeName, action::nodes::SetDisplay, action::nodes::SetInlineSelection;
using ::test::Op, ::test::Patchwork, ::test::Region, ::test::SelectionWhere, ::test::TopologyCases;

std::vector<Corner> SourcePoints(GraphFixture &f, state::Entity mesh_entity, auto &&transform) {
    return f.WithSource(mesh_entity, [&](uint32_t id) {
        const Mesh mesh{f.Store(), id};
        std::vector<Corner> points;
        for (const auto v : mesh.vertices()) {
            const auto p = transform(*v, test::Position(mesh, v));
            points.push_back({p.x, p.y, p.z});
        }
        std::ranges::sort(points);
        return points;
    });
}
bool Near(const std::vector<Corner> &a, const std::vector<Corner> &b) {
    return a.size() == b.size() && std::ranges::equal(a, b, [](const Corner &p, const Corner &q) {
               return std::abs(p[0] - q[0]) < 1e-5f && std::abs(p[1] - q[1]) < 1e-5f && std::abs(p[2] - q[2]) < 1e-5f;
           });
}
std::vector<Corner> ShapePositions(const auto &shape) {
    std::vector<Corner> points;
    for (const auto p : primitive::CreateMesh(shape).Data.Positions) points.push_back({p.x, p.y, p.z});
    std::ranges::sort(points);
    return points;
}

// Steps each scalar parameter of the node's kind up and down through the node panel's field update, clamped by the field's spec.
template<typename K> void SweepParameters(GraphFixture &f, state::Entity node) {
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
void SweepParameters(GraphFixture &f, state::Entity node, uint16_t kind) {
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
// Whether the path, flattened into straight pieces, passes through the interior of the rectangle.
bool Enters(std::span<const vec2> path, const CanvasGeometry::Rect &rect) {
    constexpr float Inset{1e-3f};
    constexpr int Pieces{32};
    const auto im = [](vec2 p) { return ImVec2{p.x, p.y}; };
    for (size_t i = 1u; i < path.size(); i += 3u) {
        auto a = im(path[i - 1u]);
        for (int k = 1; k <= Pieces; ++k) {
            const auto b = ImBezierCubicCalc(im(path[i - 1u]), im(path[i]), im(path[i + 1u]), im(path[i + 2u]), float(k) / float(Pieces));
            float enter = 0.f, leave = 1.f;
            for (const auto [p, q] : {std::pair{a.x - b.x, a.x - rect.Min.x - Inset}, std::pair{b.x - a.x, rect.Max.x - Inset - a.x}, std::pair{a.y - b.y, a.y - rect.Min.y - Inset}, std::pair{b.y - a.y, rect.Max.y - Inset - a.y}}) {
                if (p == 0.f) leave = q < 0.f ? -1.f : leave;
                else if (p < 0.f) enter = std::max(enter, q / p);
                else leave = std::min(leave, q / p);
            }
            if (enter <= leave) return true;
            a = b;
        }
    }
    return false;
}
bool Overlap(const CanvasGeometry::Rect &a, const CanvasGeometry::Rect &b) { return a.Min.x < b.Max.x && b.Min.x < a.Max.x && a.Min.y < b.Max.y && b.Min.y < a.Max.y; }

// Each wire of a subgraph's page is the authored link it stands for.
void CheckSubgraphWires(const state::Registry &a, state::Entity page) {
    const auto view = ProjectGraph(a, page, {});
    for (uint32_t n = 0u; n < view.Nodes.size(); ++n) {
        const auto entity = view.Nodes[n].Item;
        for (uint16_t slot = 0u; slot < InputsOf(view, n); ++slot) {
            const auto wire = view.Wires[view.InputFirst[n] + slot];
            const auto authored = InputOf(a, entity, a.any_of<SubgraphNode, Subgraph>(entity) ? SocketId(view, n, slot) : slot);
            expect(authored == (wire.FromNode == NoNode ? SocketLink{} : SocketLink{view.Nodes[wire.FromNode].Item, SocketId(view, wire.FromNode, wire.FromSocket)}));
        }
    }
}
// Draws the panels read-only until two frames in a row draw the same.
// A frame sizes its windows by the content of the frame before, and a scrollbar that comes or goes changes the width content wraps at.
std::vector<std::byte> Settled(Fixture &f, PanelContext &panels) {
    auto drawn = DrawPanels(f, panels);
    for (int frame = 0; frame < 4; ++frame) {
        auto next = DrawPanels(f, panels);
        if (next == drawn) break;
        drawn = std::move(next);
    }
    return drawn;
}
// The texels of an image of 32-bit float channels, as their floats.
std::vector<float> Texels(MTL::Texture *image) {
    const auto format = image->pixelFormat();
    const auto stride = format == MTL::PixelFormatR32Float ? 1u : format == MTL::PixelFormatRG32Float ? 2u :
                                                                                                        4u;
    const auto width = uint32_t(image->width()), height = uint32_t(image->height());
    std::vector<float> texels(size_t(width) * height * stride);
    image->getBytes(texels.data(), width * stride * sizeof(float), MTL::Region::Make2D(0, 0, width, height), 0);
    return texels;
}
} // namespace

// A display path of only topology tasks on one Source node draws what those tasks make of a clone of the Source, and a faulted node passes its input through.
// With EveryStep, the fresh-cook and canvas checks follow.
void GraphFixture::CheckStep() {
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
            const auto &[node, kind, parent, in_builtin_subgraph] = plan.Nodes[n];
            // A built-in subgraph's nodes hold their parameters in the registry of built-in subgraphs.
            if (in_builtin_subgraph) {
                runs = false;
                break;
            }
            if (kind == KindOf<graph::Extrude> && R.get<const graph::Extrude>(node).OffsetScale != 0.f) runs = false;
            if (kind == KindOf<graph::Source> && !source) source = R.get<const SourceState>(node);
            else if (std::ranges::contains(TopologyCases, std::optional{kind}, &::test::TopologyCase::Kind)) path.push_back(plan.Nodes[n]);
            else runs = false;
        }
        // A Source node holding no mesh faults, so its path has no kernel reference.
        if (!runs || !source || path.empty() || (source->State == store::Hash128{} && !meshes.TryGet(source->StoreId))) continue;
        const auto id = CloneSource(*source);
        cloned = true;
        for (const auto &[node, kind, parent, in_builtin_subgraph] : path) {
            if (Fault(node)) continue;
            const auto *inline_selection = R.try_get<const InlineSelection>(node);
            const auto listed = inline_selection && !inline_selection->All ? std::optional{LiveHandles(meshes, id, *inline_selection)} : std::nullopt;
            const TopologyInput input{
                .In = listed ? std::optional<SelectionSource>{*listed} : std::nullopt,
                .Domain = inline_selection ? inline_selection->Domain : Element::Vertex,
            };
            // A hull and a beautify plan on the host as the cook's GPU planners do.
            const Mesh record{meshes, id};
            const auto selection = listed ? *listed : AllHandles(record);
            std::vector<MeshTopologyTask> tasks;
            VisitKind(kind, [&]<typename K> {
                if constexpr (std::same_as<K, action::mesh::ConvexHull>) {
                    tasks = ConvexHullTasks(R, record, selection);
                } else if constexpr (std::same_as<K, action::mesh::BeautifyFaces>) {
                    if (auto task = BeautifyFaceTask(R, record, selection)) tasks.push_back(std::move(*task));
                } else if constexpr (requires(const K &p) { PlanTopology(R, p, id, input); }) {
                    auto planned = PlanTopology(R, R.get<const K>(node), id, input);
                    if constexpr (std::same_as<decltype(planned), MeshTopologyTask>) tasks.push_back(std::move(planned));
                    else if (planned) tasks.push_back(std::move(*planned));
                }
            });
            // A plan without a task leaves the record as the cook's node leaves it.
            if (!tasks.empty()) ExecuteGeometryTopologyStages(R, tasks);
        }
        expect(Describe(meshes, id) == Shaped(mesh));
        meshes.Release(id);
    }
    // The present node holds mesh storage as it stood before the clones.
    if (cloned) P->History.Revert();
    if (EveryStep) CheckEnd();
}

// The page the Nodes panel keeps across steps draws what a page built from nothing draws.
// Every page of the Nodes panel draws the same Nodes and Properties panels interactively as read-only, and drawing leaves the document unchanged.
// A page at its automatic layout keeps its node boxes and lanes apart and draws each wire clear of the box of every node but its two, and a subgraph's wires are its authored links.
void GraphFixture::CheckCanvas() {
    auto &panels = *Panels;
    const auto document = P->History.MaterializeLive();
    const auto kept = Settled(*this, panels);
    panels.Nodes.Page.reset();
    expect(Settled(*this, panels) == kept) << "the kept page differs from a page built from nothing";
    // The pages draw in a Nodes panel of their own, which leaves the kept page as it is.
    NodesPanel pages_panel{{"Nodes"}};
    std::swap(panels.Nodes, pages_panel);
    std::vector<std::pair<state::Entity, std::vector<state::Entity>>> pages{{state::Document, {}}};
    for (const auto scene : SortedEntities(R.view<const Scene>())) pages.push_back({scene, {}});
    for (const auto graph : SortedEntities(R.view<const GraphNodes>())) {
        pages.push_back({graph, {}});
        for (const auto node : NodesOf(R, graph))
            if (R.all_of<SubgraphNode>(node)) pages.push_back({graph, {node}});
    }
    for (const auto &[graph, path] : pages) {
        EnterGraph(panels.Nodes, graph, state::Null, path);
        Settled(*this, panels);
        const auto drawn = DrawPanels(*this, panels, true);
        expect(DrawPanels(*this, panels) == drawn) << "interactive and read-only drawing differ";
        const auto page = path.empty() ? graph : path.back();
        const auto subgraph = R.all_of<SubgraphNode>(page) ? SubgraphOf(R, page) : SubgraphRef{&R, R.all_of<Subgraph>(page) ? page : state::Null};
        if (subgraph.Entity != state::Null) CheckSubgraphWires(*subgraph.Registry, subgraph.Entity);
        if (subgraph.Entity != state::Null ? subgraph.Registry->all_of<GraphLayout>(subgraph.Entity) : R.all_of<GraphLayout>(graph)) continue;
        const auto geometry = DrawnGeometry(R, panels.Nodes);
        auto cells = geometry.Nodes;
        cells.append_range(geometry.Lanes);
        for (size_t i = 0u; i < cells.size(); ++i) {
            for (size_t j = i + 1u; j < cells.size(); ++j) expect(!Overlap(cells[i], cells[j]));
        }
        for (const auto &wire : geometry.Wires) {
            for (uint32_t n = 0u; n < geometry.Nodes.size(); ++n) {
                if (n != wire.From && n != wire.To) expect(!Enters(wire.Path, geometry.Nodes[n]));
            }
        }
    }
    std::swap(panels.Nodes, pages_panel);
    expect(P->History.MaterializeLive() == document) << "drawing changed the document";
}

// Each graph mesh draws what a cook of its graph from nothing makes, and each image a Bake or filter occurrence keeps equals an evaluation of the occurrence from nothing.
// The kept images return afterwards, so the check leaves the evaluations' state as it found it.
void GraphFixture::CheckFreshCooks() {
    auto &images = CurrentImages(R);
    auto kept = std::exchange(images.Images, {});
    std::vector<state::Entity> drawn;
    for (const auto [mesh, plan] : R.view<const CookPlan>().each()) {
        // A mesh that publishes instances draws them as objects of their own, which a cook that publishes nothing leaves out.
        const auto *instances = R.try_get<const GraphInstances>(mesh);
        if (plan.ResultNode != NoNode && DrawnStoreId(R, mesh) && (!instances || instances->Entities.empty())) drawn.push_back(mesh);
    }
    std::ranges::sort(drawn);
    if (!drawn.empty()) {
        ReadAtFrame(R, drawn, R.get<const TimelinePlayback>(state::Document).CurrentFrame, [&](std::span<const uint32_t> records) {
            for (size_t i = 0; i < drawn.size(); ++i) {
                const auto fresh = records[i] == InvalidOffset ? Shape{} : Describe(Store(), records[i]);
                expect(fresh == Shaped(drawn[i])) << "mesh" << state::Index(drawn[i]) << "draws a stale cook";
            }
        });
    }
    WaitForRender(R);
    for (const auto &[key, image] : kept) {
        const auto fresh = OccurrenceImageOf(R, key.first, key.second);
        expect(fresh.has_value() == (image.Views != nullptr)) << "mesh" << state::Index(key.first) << "keeps a stale image";
        if (fresh && image.Views) expect(Texels(*image.Views->Texture) == Texels(fresh->Image)) << "mesh" << state::Index(key.first) << "keeps a stale image";
    }
    images.Images = std::move(kept);
    // A frame settles and lays out what the evaluations' restores left, which an image comparison's frame does under rendering.
    if (!::Render) {
        P->Settle();
        LayOutMeshes(R);
    }
}

namespace {
// Every operator kind cooks like its kernel on the patchwork at every parameter value, each generator kind like its primitive, and a fan-out cooks the read branch alone.
void OperatorKinds(GraphFixture &f, Fixture &other) {
    // Rows sharing a kind run its scenario once, under the first row's name.
    for (const auto &c : TopologyCases) {
        const auto kind = c.Kind;
        if (!kind || &c != &*std::ranges::find(TopologyCases, kind, &::test::TopologyCase::Kind)) continue;
        Scenario(f, other, std::string{c.Name}, [kind](GraphFixture &f) {
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
        Scenario(f, other, std::visit([]<typename K>(const K &) { return std::string{field::TypeLabel<K>.c_str()}; }, shape), [shape](GraphFixture &f) {
            auto &meshes = f.Store();
            const auto mesh = f.AddMesh(primitive::Plane{});
            const auto source = f.SourceOf(mesh);
            const auto generator = f.Append(mesh, uint16_t(KindOf<primitive::Plane> + shape.index()));
            const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
            f.Route(mesh, {generator, subdivide});
            // The generated mesh holds texture coordinate set 0 while a material it draws reads the set.
            const auto texcoords = [&] { return (meshes.Get(f.Output(mesh)).CornerAttributes & MeshAttributeBit_TexCoord0) != 0u; };
            expect(!texcoords());
            f.Do(action::object::AddMaterial{0u, mesh});
            const auto material = MaterialAt(f.R, meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(f.Output(mesh)).PrimitiveMaterials)[0]);
            constexpr auto strength = uint16_t(offsetof(PBRMaterial, Anisotropy) + offsetof(Anisotropy, Strength));
            f.Do(action::object::UpdateMaterial<float>{material, strength, .5f});
            expect(texcoords());
            // The set lies in the unit square, and each face maps to a polygon of nonzero area.
            const Mesh output{meshes, f.Output(mesh)};
            const auto &record = meshes.Get(f.Output(mesh));
            for (const auto face : output.faces()) {
                const auto uvs = test::FaceHalfedges(output, face) | std::views::transform([&](Mesh::HH h) { return meshes.Arenas().CornerUvs[0].Get(record.CornerUvs[0], *h); }) | std::ranges::to<std::vector>();
                float twice_area = 0.f;
                for (size_t i = 0; i < uvs.size(); ++i) twice_area += uvs[i].x * uvs[(i + 1) % uvs.size()].y - uvs[(i + 1) % uvs.size()].x * uvs[i].y;
                expect(std::ranges::all_of(uvs, [](vec2 uv) { return uv.x >= 0.f && uv.x <= 1.f && uv.y >= 0.f && uv.y <= 1.f; }));
                expect(twice_area != 0.f);
            }
            f.Do(action::object::UpdateMaterial<float>{material, strength, 0.f});
            expect(!texcoords());
            const auto generated = CreateMesh(f.R, primitive::CreateMesh(shape)).StoreId;
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
    Scenario(f, other, "FanOut", [](GraphFixture &f) {
        const auto mesh = f.AddMesh(primitive::Plane{});
        const auto source = f.SourceOf(mesh);
        const auto first = f.Append(mesh, KindOf<graph::Extrude>);
        const auto second = f.Append(mesh, KindOf<graph::Extrude>);
        f.Do(Link{mesh, f.Node(mesh, 0u), 0u, first, 0u});
        f.Do(Link{mesh, f.Node(mesh, 0u), 0u, second, 0u});
        for (const auto read : {first, second}) {
            f.Do(SetDisplay{mesh, read});
            expect(!f.Fault(first) && !f.Fault(second));
            expect(f.SourceOf(mesh) == source);
        }
    });
}

// A graph on `Mesh`, an oracle for its output beyond the kernels each step checks, and an edit that changes a copy's output.
struct GraphCase {
    state::Entity Mesh;
    std::function<void(GraphFixture &)> Oracle{};
    std::function<void(GraphFixture &, state::Entity)> Edit{};
};
struct GraphRow {
    std::string_view Name;
    GraphCase (*Build)(GraphFixture &);
};
// The link from an op output into a consumer input that calls its program at each element, the default of the read that takes its place in the stored graph, and the input's domain.
struct CalledInput {
    uint32_t Link;
    OpLiteral Default;
    AttributeDomain Domain{AttributeDomain::Vertex};
};
struct GraphData {
    std::string_view Name;
    GraphDescription Description;
    void (*Oracle)(GraphFixture &, const BuiltGraph &);
    void (*Edit)(GraphFixture &, state::Entity);
    std::optional<CalledInput> Called{};
    // The face counts of the output and of the edited copy's output.
    std::optional<uint32_t> Faces{}, EditedFaces{};
};

// The graph's output matches its oracle, and a duplicate of its object cooks the same and edits apart.
void Survives(GraphFixture &f, Fixture &other, std::string_view name, auto &&build) {
    Scenario(f, other, std::string{name}, [&build](GraphFixture &f) {
        auto &r = f.R;
        const auto graph = build(f);
        const auto mesh = graph.Mesh;
        if (graph.Oracle) graph.Oracle(f);
        if (!graph.Edit) return;
        const auto output = f.Shaped(mesh);
        f.Do(action::selection::Select{f.Object(mesh)});
        f.Do(action::object::Duplicate{});
        const auto copy = r.get<const ObjectGeometry>(FindActiveEntity(r)).Entity;
        expect(copy != mesh && f.Output(copy) != f.Output(mesh) && f.Shaped(copy) == output);
        graph.Edit(f, copy);
        expect(f.Shaped(copy) != output && f.Shaped(mesh) == output);
    });
}

vec3 OffsetOf(uint32_t v) { return {float(v % 7u), 1.f, -float(v % 3u)}; }
bool RestMatches(GraphFixture &f, state::Entity mesh) {
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
void ExpectColumnsThrough(GraphFixture &f, state::Entity mesh, int last) {
    const Mesh output{f.Store(), f.Output(mesh)};
    const auto columns = uint32_t(last + 5);
    expect(output.VertexCount() == 9u * columns && output.FaceCount() == 8u * (columns - 1u)) << "vertices" << output.VertexCount() << "faces" << output.FaceCount();
    for (const auto v : output.vertices()) expect(test::Position(output, v).x <= float(last)) << "vertex x" << test::Position(output, v).x;
}

std::vector<uint16_t> PortIds(const state::Registry &r, state::Entity use) {
    std::vector<uint16_t> ids;
    for (const auto &port : InterfaceOf(r, use)->Ports) ids.push_back(port.Id);
    return ids;
}
// Feeds the cube's Source node and a generated bar through it into the built-in subgraph's first two ports, displays its third, and returns the mesh and the port ids.
std::pair<state::Entity, std::vector<uint16_t>> CubeAndBarInto(GraphFixture &f, BuiltinSubgraphId builtin) {
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto source = f.Node(mesh, 0u);
    const auto box = f.Append(mesh, KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{2.f, .5f, .5f})});
    f.Do(AddNode{mesh, action::nodes::BuiltinSubgraphNode{builtin}});
    const auto node = ActiveNodeOf(f.R, mesh);
    expect(SubgraphNodeName(f.R, node) == BuiltinSubgraphs[size_t(builtin)].Name);
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

void AddOffsets(GraphFixture &f, uint32_t source) {
    AddValues<vec3>(f.Store(), source, "offset", [](const Mesh &, he::VH v) { return OffsetOf(*v); });
}
// Gives each source vertex a "weight" of 5 over the attribute's default of -1.
void AddWeights(GraphFixture &f, uint32_t source) {
    AddValues<float>(f.Store(), source, "weight", [](const Mesh &, he::VH) { return 5.f; }, AttributeValue{std::bit_cast<uint32_t>(-1.f), 0u, 0u, 0u});
}
void AddHeightHeat(GraphFixture &f, uint32_t source) {
    AddValues<float>(f.Store(), source, "heat", [](const Mesh &m, he::VH v) { return test::Position(m, v).z; });
}

void MoveFirstVertex(GraphFixture &f, uint32_t source) {
    const PositionOperationTarget target{.StoreId = source, .Selection = GeometrySelection{.Vertices = {**Mesh{f.Store(), source}.vertices().begin()}}};
    ExecuteGeometryPositions(f.R, target, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {.3f, -.5f, .1f}}}});
}

void ExpectUpperExtruded(GraphFixture &f, const BuiltGraph &g) {
    const auto &meshes = f.Store();
    const auto extruded = EditedSource(f, g.Mesh, [&](uint32_t clone) {
        const Mesh cube{meshes, clone};
        const auto upper = SelectionWhere(cube, [&](he::VH v) { return test::Position(cube, v).z > 0.f; });
        expect(upper.Faces.size() == 1u);
        f.Run(clone, {{.Op = MeshTopologyOp::ExtrudeRegion, .Selection = upper}});
    });
    expect(f.Shaped(g.Mesh) == extruded);
}
// The output vertices beyond and within the unit major radius around the Y axis.
std::pair<uint32_t, uint32_t> OuterInner(GraphFixture &f, state::Entity mesh) {
    uint32_t outer = 0u, inner = 0u;
    for (const auto p : f.Shaped(mesh).Vertices) {
        const auto radius = std::sqrt(p[0] * p[0] + p[2] * p[2]);
        if (radius > 1.001f) ++outer;
        if (radius < .999f) ++inner;
    }
    return {outer, inner};
}
void ExpectJoinedDefaults(GraphFixture &f, const BuiltGraph &g) {
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
void ExpectSmoothedTowardSphere(GraphFixture &f, const BuiltGraph &g) {
    const auto coarse = f.WithSource(g.Mesh, [&](uint32_t id) { return test::EnclosedVolume(Mesh{f.Store(), id}).value_or(-1.0); });
    const Mesh output{f.Store(), f.Output(g.Mesh)};
    const auto volume = test::EnclosedVolume(output).value_or(-1.0), sphere = 4.0 / 3.0 * std::numbers::pi;
    expect(output.HasClosedSurface() && volume > coarse && std::abs(volume - sphere) < .5 * std::abs(coarse - sphere)) << "coarse" << coarse << "smoothed" << volume;
}
void ExpectSphere(GraphFixture &f, state::Entity mesh, double radius) {
    const Mesh output{f.Store(), f.Output(mesh)};
    const auto volume = test::EnclosedVolume(output).value_or(-1.0), expected = 4.0 / 3.0 * std::numbers::pi * radius * radius * radius;
    expect(output.HasClosedSurface() && std::abs(volume - expected) < .03 * expected) << "volume" << volume << "vs" << expected;
}
// One Catmull-Clark level turns the cube into 24 quads whose corners sit at five ninths, with the heat of each corner blending into the new points.
void ExpectCatmullClarkCube(GraphFixture &f, const BuiltGraph &g) {
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
void ExpectOneFaceOfArea(GraphFixture &f, state::Entity mesh, double expected) {
    const Mesh output{f.Store(), f.Output(mesh)};
    double area = 0.0;
    for (const auto face : output.faces()) area += Length(test::AreaVector(output, face));
    expect(output.FaceCount() == 1u && std::abs(area - expected) < 1e-3 * expected) << "faces" << output.FaceCount() << "area" << area << "vs" << expected;
}
void ExpectSourceInstances(GraphFixture &f, const BuiltGraph &g) {
    const auto &r = f.R;
    const auto plane = ShapePositions(primitive::Plane{});
    const auto &entities = r.get<const GraphInstances>(g.Mesh).Entities;
    expect(!f.Fault(g.Nodes[2]));
    expect(fatal(entities.size() == plane.size()));
    const auto instanced = r.get<const ObjectGeometry>(entities.front()).Entity;
    const auto placed = r.get<const MeshHandle>(instanced).StoreId;
    expect(instanced != g.Mesh && placed != f.Output(g.Mesh) && Near(Describe(f.Store(), placed).Vertices, plane));
    expect(Near(SourcePoints(f, g.Mesh, [](uint32_t, vec3 p) { return p; }), plane) && f.Shaped(g.Mesh).Vertices != plane);
}
// An instancer without an Instance input faults and passes its points through.
void UnlinkInstance(GraphFixture &f, state::Entity mesh) {
    const auto instancer = f.Node(mesh, 2u);
    f.Do(action::nodes::Unlink{mesh, instancer, 1u});
    expect(f.Fault(instancer) == NodeFault::NoMesh);
    const Mesh points{f.Store(), f.Output(mesh)};
    const auto *instances = f.R.try_get<const GraphInstances>(mesh);
    expect(points.VertexCount() == 4u && points.FaceCount() == 0u && (!instances || instances->Entities.empty()));
}
// The joined vertices share one block, and the Delete still removes every vertex at or above y = 1.5.
void ExpectJoinedSelection(GraphFixture &f, const BuiltGraph &g) {
    const auto &meshes = f.Store();
    expect(!f.Fault(g.Nodes[5]) && !f.Fault(g.Nodes[6]));
    const auto id = f.Output(g.Mesh);
    expect(meshes.GetBlockList(id, MeshStore::ElementDomain::Vertex).Blocks.size() == 1u);
    const Mesh output{meshes, id};
    expect(output.VertexCount() == 12u);
    for (const auto v : output.vertices()) expect(test::Position(output, v).y < 1.5f);
}

// A sphere's cap above y = 0.9, with a vertex moved before its Source node's state is kept, insets through position, extract and if greater ops and Inset.
GraphCase CapInset(GraphFixture &f) {
    const auto [mesh, nodes] = Build(f, {primitive::UVSphere{.Slices = 128u, .Stacks = 64u}, {{"position"}, {"extract", {Literal("index", 1)}}, {"ifgreater", {Literal("value2", .9f)}}, {KindOf<action::mesh::Inset>}}, {{1, 2, 0, "in"}, {2, 3, 0, "value1"}, {3, 4, 1}, {0, 4, 0}}, 4, MoveFirstVertex});
    const auto inset = nodes[4];
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
        [mesh](GraphFixture &f) {
            const auto inset = EditedSource(f, mesh, [&](uint32_t clone) {
                const Mesh sphere{f.Store(), clone};
                const auto cap = SelectionWhere(sphere, [&](he::VH v) { return test::Position(sphere, v).y > .9f; });
                f.Run(clone, {{.Op = MeshTopologyOp::InsetRegion, .Param0 = .06f, .Flags = TopologyFlagEvenOffset, .Selection = cap}});
            });
            expect(f.Shaped(mesh) == inset);
        },
        [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Inset::Thickness>(f.Node(mesh, 4u), .2f)); },
    };
}

// A Store Selection of the cube's upper face survives a later subdivision, so a Delete reading it through a geometric property op removes the four faces it became.
GraphCase StoreSelectionDelete(GraphFixture &f) {
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto store = f.Append(mesh, KindOf<StoreSelection>);
    const auto named = f.Append(mesh, "geompropvalue", {Literal("default", false)});
    const auto remove = f.Append(mesh, KindOf<action::mesh::Delete>);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Do(SetAttributeName{store, "upper"});
    Text("geomprop", "upper")(f, named);
    f.Do(UpdateOf<&action::mesh::Delete::Mode>(remove, action::mesh::DeleteMode::OnlyFaces));
    const Mesh cube{meshes, f.Output(mesh)};
    GeometrySelection upper;
    for (const auto face : cube.faces())
        if (test::Centroid(cube, face).y > .5f) upper.Faces.push_back(*face);
    f.Do(SetInlineSelection{store, std::make_unique<InlineSelection>(SelectionOf(upper, Element::Face))});
    f.Do(Link{mesh, named, f.Out(named), remove, 1u});
    f.Route(mesh, {f.Node(mesh, 0u), store, remove});
    expect(Mesh{meshes, f.Output(mesh)}.FaceCount() == 5u);
    f.Do(InsertNode{mesh, subdivide, remove, 0u, 0u});
    return {
        mesh,
        [mesh, named, remove](GraphFixture &f) {
            const auto &meshes = f.Store();
            expect(!f.Fault(named) && !f.Fault(remove));
            const auto deleted = EditedSource(f, mesh, [&](uint32_t clone) {
                f.Run(clone, {Op(MeshTopologyOp::Subdivide, 1.f, TopologyFlagSelectAll)});
                const Mesh subdivided{meshes, clone};
                GeometrySelection upper;
                for (const auto face : subdivided.faces())
                    if (test::Centroid(subdivided, face).y > .99f) upper.Faces.push_back(*face);
                expect(upper.Faces.size() == 4u);
                f.Run(clone, {{.Op = MeshTopologyOp(uint32_t(action::mesh::DeleteMode::OnlyFaces)), .Selection = upper}});
            });
            expect(f.Shaped(mesh) == deleted);
        },
        [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 4u), 2u)); },
    };
}

std::map<uint32_t, float> Heat(const MeshStore &meshes, uint32_t id) {
    std::map<uint32_t, float> heat;
    const auto view = meshes.AttributeView<float>(meshes.FindAttribute(id, "heat"));
    meshes.Arenas().Vertices.ForEach(meshes.Get(id).Vertices, [&](ElementHandle v, ElementAddress) { heat.emplace(v.Value, view[v.Value]); });
    return heat;
}
// Retained vertices keep their heat, and an edge midpoint takes the mean of two corners and a face center the mean of four.
GraphCase HeatSubdivide(GraphFixture &f) {
    auto &meshes = f.Store();
    const auto mesh = f.AddPrepared(primitive::Cube{}, [&](GraphFixture &, uint32_t id) {
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
    return {mesh, [mesh, original](GraphFixture &f) {
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

// The built-in Cross Section subgraph across a cone's Y axis leaves a disc whose radius shrinks with height, and the section moves with the plane.
GraphCase CrossSectionOfCone(GraphFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cylinder{.TopRadius = 0.f, .Slices = 32});
    const auto source = f.Node(mesh, 0u);
    f.Do(AddNode{mesh, action::nodes::BuiltinSubgraphNode{graph::BuiltinSubgraphId::CrossSection}});
    const auto section = graph::ActiveNodeOf(f.R, mesh);
    const auto ports = PortIds(f.R, section);
    expect(fatal(ports.size() == 4u));
    f.Do(Link{mesh, source, f.Out(source), section, ports[0]});
    f.Do(action::nodes::SetPortValue{section, ports[2], {0u, std::bit_cast<uint32_t>(1.f), 0u, 0u}});
    f.Do(SetDisplay{mesh, section, ports[3]});
    // A section of radius r is the 32-gon of area 16 r^2 sin(pi / 16).
    const auto disc = [](GraphFixture &f, state::Entity mesh, double radius) { ExpectOneFaceOfArea(f, mesh, 16.0 * radius * radius * std::sin(std::numbers::pi / 16.0)); };
    return {
        mesh,
        [mesh, disc](GraphFixture &f) { disc(f, mesh, .5f); },
        [disc, point = ports[1]](GraphFixture &f, state::Entity mesh) {
            f.Do(action::nodes::SetPortValue{f.R.get<const GeometryGraph>(mesh).DisplayNode, point, {0u, std::bit_cast<uint32_t>(-.5f), 0u, 0u}});
            disc(f, mesh, .75f);
        },
    };
}

// The split subgraph's outside and inside of the cube cut by a bar hold the cube's volume between them.
GraphCase SplitBySolid(GraphFixture &f) {
    const auto [mesh, ports] = CubeAndBarInto(f, BuiltinSubgraphId::SplitBySolid);
    expect(ports.size() == 4u);
    return {
        mesh,
        [mesh](GraphFixture &f) {
            const auto volume = OutputVolume(f, mesh);
            expect(std::abs(volume - 6.0) < 1e-4) << "outside volume" << volume;
        },
        [inside = ports[3]](GraphFixture &f, state::Entity mesh) {
            f.Do(SetDisplay{mesh, f.Node(mesh, 2u), inside});
            const auto volume = OutputVolume(f, mesh);
            expect(std::abs(volume - 2.0) < 1e-4) << "inside volume" << volume;
        },
    };
}
// The built-in Hull of Objects subgraph over the cube and a bar through it adds a frustum of 7/3 at each end.
GraphCase HullOfObjects(GraphFixture &f) {
    const auto [mesh, ports] = CubeAndBarInto(f, BuiltinSubgraphId::HullOfObjects);
    expect(ports.size() == 3u);
    return {
        mesh,
        [mesh](GraphFixture &f) {
            const Mesh hull{f.Store(), f.Output(mesh)};
            const auto volume = test::EnclosedVolume(hull).value_or(-1.0);
            expect(std::abs(volume - (8.0 + 14.0 / 3.0)) < 1e-3) << "hull volume" << volume;
            expect(hull.VertexCount() == 16u && hull.FaceCount() == 28u) << "hull elements" << hull.VertexCount() << hull.FaceCount();
            expect(hull.HasClosedSurface() && test::ClosedComponentsOutward(hull));
        },
        [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 1u), vec3{3.f, .5f, .5f})); },
    };
}

// The boolean action makes the active object's mesh the active cube minus the other selected cube, in the active object's space.
GraphCase BooleanAction(GraphFixture &f) {
    const auto &r = f.R;
    const auto target = f.AddMesh(primitive::Cube{});
    const auto active = FindActiveEntity(r);
    f.Do(action::object::AddMeshPrimitive{primitive::Cube{}, std::make_unique<MeshObjectCreateInfo>(MeshObjectCreateInfo{.Transform = {.P = {1.f, 1.f, 1.f}}, .Select = MeshObjectCreateInfo::SelectBehavior::Additive})});
    expect(FindActiveEntity(r) == active);
    f.Do(action::mesh::Boolean{BooleanOperation::Difference});
    return {target, [target, active](GraphFixture &f) {
                expect(std::abs(OutputVolume(f, target) - 7.0) < 1e-4);
                for (const auto e : f.R.view<const ObjectGeometry>())
                    if (e != active) expect(Mesh{f.Store(), f.Output(f.R.get<const ObjectGeometry>(e).Entity)}.VertexCount() == 8u);
            }};
}

GraphCase GeneratorEdit(GraphFixture &f) {
    const auto mesh = f.AddGenerated(primitive::Cube{});
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 0u), vec3{2.f, .5f, 1.f}));
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 0u), vec3{1.f, 1.f, 3.f}));
    return {mesh, [mesh](GraphFixture &f) { expect(Near(f.Shaped(mesh).Vertices, ShapePositions(primitive::Cube{{1.f, 1.f, 3.f}}))); }};
}

// An Object Info node reading its own graph's mesh errs and forwards the source.
// One reading a mesh with a graph of its own cooks after that graph, so a resize there reaches the join.
GraphCase ObjectInfo(GraphFixture &f) {
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto plane = f.Shaped(mesh);
    const auto info = f.Append(mesh, KindOf<graph::ObjectInfo>, {Set<&graph::ObjectInfo::Object>(f.Object(mesh))});
    const auto join = f.Append(mesh, KindOf<Join>);
    f.Do(Link{mesh, f.Node(mesh, 0u), 0u, join, 0u});
    f.Do(Link{mesh, info, 0u, join, 0u});
    f.Do(SetDisplay{mesh, join});
    expect(f.Fault(info) && !f.Fault(join));
    expect(f.Shaped(mesh) == plane);
    const auto cube = f.AddMesh(primitive::Cube{});
    const auto box = f.Append(cube, KindOf<primitive::Cube>);
    f.Do(SetDisplay{cube, box});
    f.Do(UpdateOf<&graph::ObjectInfo::Object>(info, f.Object(cube)));
    expect(!f.Fault(info));
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(box, vec3{.5f, .25f, .5f}));
    return {mesh, [mesh](GraphFixture &f) {
                auto points = ShapePositions(primitive::Plane{});
                points.append_range(ShapePositions(primitive::Cube{{.5f, .25f, .5f}}));
                std::ranges::sort(points);
                expect(Mesh{f.Store(), f.Output(mesh)}.FaceCount() == 7u && Near(f.Shaped(mesh).Vertices, points));
            }};
}

// Instances on the plane's points publish as entities or realize as geometry, following scale, plane and cube edits.
template<bool Borrowed, bool Realize> GraphCase Instances(GraphFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddGenerated(primitive::Plane{});
    const auto object = FindActiveEntity(r);
    const auto cube_mesh = f.AddGenerated(primitive::Cube{});
    const auto to_points = f.Append(mesh, KindOf<MeshToPoints>);
    const auto instancer = f.Append(mesh, KindOf<InstanceOnPoints>);
    const auto instance = f.Append(mesh, Borrowed ? KindOf<graph::ObjectInfo> : KindOf<primitive::Cube>);
    if constexpr (Borrowed) f.Do(UpdateOf<&graph::ObjectInfo::Object>(instance, f.Object(cube_mesh)));
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
    GraphCase graph{mesh, [=](GraphFixture &f) {
                        const auto &r = f.R;
                        const auto plane = ShapePositions(primitive::Plane{{2.f, 3.f}}), cube = ShapePositions(primitive::Cube{{.5f, .25f, .5f}});
                        if constexpr (Realize) {
                            std::vector<Corner> realized;
                            for (const auto &p : plane)
                                for (const auto &c : cube) realized.push_back({p[0] + .5f * c[0], p[1] + .5f * c[1], p[2] + .5f * c[2]});
                            std::ranges::sort(realized);
                            expect(Near(f.Shaped(mesh).Vertices, realized));
                        } else {
                            const auto &entities = r.get<const GraphInstances>(mesh).Entities;
                            expect(entities.size() == 4u);
                            std::vector<Corner> points;
                            for (const auto e : entities) {
                                expect(ParentOrNull(r, e) == object);
                                const auto &t = r.get<const Transform>(e);
                                expect(t.S == vec3{.5f, .5f, .5f});
                                points.push_back({t.P.x, t.P.y, t.P.z});
                                expect(Near(f.Shaped(r.get<const ObjectGeometry>(e).Entity).Vertices, cube));
                            }
                            std::ranges::sort(points);
                            expect(Near(points, plane));
                        }
                    }};
    if constexpr (Realize) graph.Edit = [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&InstanceOnPoints::Scale>(f.Node(mesh, 2u), vec3{.25f, .25f, .25f})); };
    return graph;
}

// An instance of the prototype subgraph publishes at each of the plane's points, with a copy of each object the subgraph holds, and follows an edit of one.
GraphCase InstancesOfPrototype(GraphFixture &f) {
    auto &r = f.R;
    const std::array cubes{f.Object(f.AddGenerated(primitive::Cube{})), f.Object(f.AddGenerated(primitive::Cube{}))};
    f.Do(action::nodes::Collapse{FrameScene(r), {cubes.begin(), cubes.end()}});
    const auto prototype = FindActiveNode(r);
    const auto mesh = f.AddGenerated(primitive::Plane{});
    const auto object = FindActiveEntity(r);
    const auto to_points = f.Append(mesh, KindOf<MeshToPoints>);
    const auto instancer = f.Append(mesh, KindOf<InstanceOnPoints>);
    f.Route(mesh, {f.Node(mesh, 0u), to_points, instancer});
    f.Do(UpdateOf<&InstanceOnPoints::Prototype>(instancer, prototype));
    f.Do(UpdateOf<&Transform::P>(cubes[1], vec3{0.f, 2.f, 0.f}));
    return {mesh, [=](GraphFixture &f) {
                const auto &r = f.R;
                const auto &entities = r.get<const GraphInstances>(mesh).Entities;
                expect(entities.size() == 4u);
                std::vector<Corner> points;
                for (const auto e : entities) {
                    expect(ParentOrNull(r, e) == object && r.get<const Instance>(e).SubgraphEntity == prototype);
                    const auto &t = r.get<const Transform>(e);
                    points.push_back({t.P.x, t.P.y, t.P.z});
                    for (const auto cube : cubes) {
                        const auto copy = CopyFor(r.get<const Instance>(e).Copies, cube);
                        expect(copy != cube && ParentOrNull(r, copy) == e && r.get<const Transform>(copy) == r.get<const Transform>(cube) && r.get<const ObjectGeometry>(copy).Entity == r.get<const ObjectGeometry>(cube).Entity);
                    }
                }
                std::ranges::sort(points);
                expect(Near(points, ShapePositions(primitive::Plane{})));
            }};
}

// Selecting, duplicating, inserting and deleting nodes leaves the output the chain they form.
GraphCase NodeEditing(GraphFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto source = f.Shaped(mesh);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    const auto triangulate = f.Append(mesh, KindOf<action::mesh::SplitFaces>);
    const auto position = f.Append(mesh, "position");
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide, 3u));
    const auto shown = r.get<const GeometryGraph>(mesh).DisplayNode;
    f.Do(SelectItems{mesh, subdivide, {subdivide, triangulate}});
    f.Do(action::nodes::MoveNodes{mesh, {{subdivide, {12u, 4u}}, {triangulate, {20u, 4u}}}});
    expect(r.get<const GeometryGraph>(mesh).DisplayNode == shown && f.Shaped(mesh) == source);
    f.Do(DuplicateNodes{mesh, {subdivide, triangulate}});
    const auto &graph = r.get<const CookPlan>(mesh);
    expect(graph.Nodes.size() == 6u && r.get<const action::mesh::Subdivide>(graph.Nodes[4].Entity).Cuts == 3u);
    expect(!GraphPosition(r, mesh, graph.Nodes[4].Entity));
    expect(f.Shaped(mesh) == source);
    f.Route(mesh, {f.Node(mesh, 0u), triangulate});
    f.Fails(InsertNode{mesh, position, triangulate, 0u, 0u});
    // A node with geometry sockets never goes into a link into an op input.
    const auto sum = f.Append(mesh, "add", {From("in1", position)});
    f.Fails(InsertNode{mesh, subdivide, sum, f.OpSocket(sum, "in1"), 0u});
    f.Do(SelectItems{mesh, subdivide, {subdivide}});
    f.Stage(InsertNode{mesh, subdivide, triangulate, 0u, 40u});
    f.CommitGesture();
    expect(GraphPosition(r, mesh, subdivide) == uvec2{12u, 4u} && GraphPosition(r, mesh, triangulate) == uvec2{60u, 4u});
    const auto refined = f.Shaped(mesh);
    f.Do(action::nodes::SetBypass{subdivide, true});
    expect(f.Shaped(mesh) != refined);
    expect(InputOf(r, triangulate, 0u).FromNode == subdivide);
    f.Do(action::nodes::SetBypass{subdivide, false});
    expect(f.Shaped(mesh) == refined);
    f.Do(DeleteNodes{mesh, {subdivide}});
    expect(!GraphPosition(r, mesh, subdivide) && GraphPosition(r, mesh, triangulate));
    f.Undo();
    f.Do(DeleteNodes{mesh, {subdivide, triangulate}});
    expect(f.Shaped(mesh) == source && r.get<const CookPlan>(mesh).DisplayNode == 0u);
    f.Undo();
    f.Do(action::nodes::UnplaceNode{mesh, subdivide});
    expect(!GraphPosition(r, mesh, subdivide) && GraphPosition(r, mesh, triangulate));
    f.Do(action::nodes::AutoLayout{mesh});
    expect(!r.all_of<GraphLayout>(mesh));
    return {mesh};
}

float WidthAt(const Mesh &mesh, he::VH v) {
    const auto p = test::Position(mesh, v);
    return .05f * (p.x + p.y + p.z) + .2f;
}

// The source's shape after the topology op over every element, whose first parameter reads the attribute `parameter` adds to the clone.
Shape TopologyReference(GraphFixture &f, state::Entity mesh, MeshTopologyOp op, auto &&parameter) {
    return EditedSource(f, mesh, [&](uint32_t clone) {
        auto task = Op(op, 0.f, TopologyFlagSelectAll);
        task.Param0Attribute = parameter(clone);
        f.Run(clone, {task});
    });
}

// Editing input selections keeps Display and the viewport selection independent, and commits or cancels the whole task.
void ViewportSelection(GraphFixture &f) {
    auto &r = f.R;
    // A collapsed node's selection exposed as a port is the subgraph node's own selection.
    // In a crowded edit a dense mesh is also in edit mode, and its cluster LOD builds once the selection edit's gesture ends.
    for (const auto [kind, vertices, port, crowded] : {std::tuple{KindOf<graph::Extrude>, false, false, false}, {KindOf<action::mesh::Bevel>, false, false, false}, {KindOf<action::mesh::Bevel>, false, false, true}, {KindOf<action::mesh::Bevel>, true, false, false}, {KindOf<TransformElements>, false, false, false}, {KindOf<TransformElements>, false, true, false}}) {
        const auto mesh = kind == KindOf<action::mesh::Bevel> ? f.AddMesh(primitive::Cube{}) : f.AddGenerated(primitive::Cube{});
        const auto node = f.Append(mesh, kind);
        if (vertices) f.Do(UpdateOf<&action::mesh::Bevel::Vertices>(node, true));
        f.Route(mesh, {f.Node(mesh, 0u), node});
        if (kind == KindOf<TransformElements>) f.Do(action::nodes::Collapse{mesh, {node}});
        const auto use = ActiveNodeOf(r, mesh);
        std::optional<uint16_t> selection_port;
        if (port) {
            const auto subgraph = UserSubgraph(r.get<const SubgraphNode>(use).SubgraphId);
            f.Do(action::nodes::ExposeSocket{subgraph, node, uint16_t(std::ranges::find(KindSockets(kind), true, &Socket::Selection) - KindSockets(kind).begin()), {}});
            selection_port = r.get<const SubgraphInterface>(subgraph).Ports.back().Id;
        }
        const auto selection = [&]() -> const InlineSelection & { return port ? *NodePortOf(r, use, *selection_port)->Selection : r.get<const InlineSelection>(node); };
        const auto display = r.get<const GeometryGraph>(mesh).DisplayNode;
        for (const auto mode : {InteractionMode::Object, InteractionMode::Edit}) {
            f.Do(action::view::SetInteractionMode{mode});
            if (mode == InteractionMode::Edit) {
                f.Do(action::view::SetEditMode{Element::Vertex});
                f.Do(action::selection::ApplyEditElementClick{false, ElementHit{mesh, *GetMesh(r, mesh).VertexAt(0u)}});
            }
            for (const bool cancel : {true, false}) {
                // The crowded edit commits a selection edit of one step, whose cook keeps the whole mesh storage with that LOD.
                if (crowded && !cancel && mode == InteractionMode::Edit) {
                    f.AddGenerated(primitive::UVSphere{.Radius = .25f, .Slices = 64u, .Stacks = 32u});
                    f.Do(action::selection::ExtendActive{f.Object(mesh)});
                    action::Emit(action::nodes::EditSelection{node, f.Object(mesh), r.get<const GeometryGraph>(mesh).DisplayPath});
                    f.P->Frame();
                    action::Emit(action::selection::DeselectAll{});
                    f.P->Frame();
                    f.Do(action::nodes::FinishSelection{});
                }
                const auto before = f.Shaped(mesh);
                const auto prior_elements = EditorGeometrySelection(f.Store(), f.Output(mesh));
                const auto present = f.P->History.Present;
                const auto selected = SelectedIn(r);
                const auto element = r.get<const EditMode>(state::Document).Value;
                const auto edit = [&](auto a) {
                    action::Emit(std::move(a));
                    f.P->Frame();
                    f.Audit();
                    expect(f.P->HasStaged() && f.P->History.Present == present);
                    expect(r.get<const GeometryGraph>(mesh).DisplayNode == display);
                };
                edit(port ? action::nodes::EditSelection{use, f.Object(mesh), {}, selection_port} : action::nodes::EditSelection{node, f.Object(mesh), r.get<const GeometryGraph>(mesh).DisplayPath});
                const auto input = r.get<const SelectionEditing>(state::Document).Mesh;
                expect(GetMesh(r, input).FaceCount() == 6u);
                edit(action::view::SetEditMode{Element::Face});
                edit(action::selection::DeselectAll{});
                expect(SelectedCount(selection().Faces) == 0u);
                f.Render();
                const Mesh cage = GetMesh(r, input);
                const auto hit = f.Pick(FacingFaceCenters(cage, f.View().CameraPosition).front(), Element::Face);
                expect(fatal(hit.has_value()));
                expect(hit->MeshEntity == input);
                edit(action::selection::ApplyEditElementClick{false, hit});
                expect(SelectedCount(selection().Faces) == 1u);
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
                expect(SelectedCount(selection().Faces) == 6u);
                if (cancel) {
                    action::Cancel();
                    f.P->Frame();
                    expect(f.P->History.Present == present && f.Shaped(mesh) == before);
                } else {
                    f.Do(action::nodes::FinishSelection{});
                    expect(f.P->History.Nodes[*f.P->History.Present].Parent == present);
                }
                expect(!f.P->HasStaged() && !r.all_of<SelectionEditing>(state::Document));
                expect(SelectedIn(r) == selected && r.get<const Interaction>(state::Document).Mode == mode && r.get<const EditMode>(state::Document).Value == element);
                expect(r.get<const GeometryGraph>(mesh).DisplayNode == display);
                const auto restored = EditorGeometrySelection(f.Store(), f.Output(mesh));
                expect(restored.Vertices == prior_elements.Vertices && restored.Edges == prior_elements.Edges && restored.Faces == prior_elements.Faces);
            }
        }
    }
}

// An Unsubdivide node's first pass publishes the selection the kernel publishes, and its second pass reads that selection as the viewport operator's second iteration does.
GraphCase UnsubdivideSelection(GraphFixture &f) {
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
        [mesh, twice, second](GraphFixture &f) { expect(f.Shaped(mesh) == twice.Output && EditorGeometrySelection(f.Store(), f.Output(mesh)).Vertices == second.Vertices); },
        [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Unsubdivide::Iterations>(f.Node(mesh, 2u), 1u)); },
    };
}

// A history begun on a live document resolves every state its Source nodes and node results refer to, which the cooks after it read.
void BeginKeepsStates(GraphFixture &f, const std::filesystem::path &next) {
    const auto chained = Build(f, {primitive::Cube{}, {{KindOf<action::mesh::Subdivide>}}, {{0, 1, 0}}, 1});
    // A graph joining its Source node's cube with a box generator, which reads the Source node's state on each cook.
    const auto joined = Build(f, {primitive::Cube{}, {{KindOf<primitive::Cube>}, {KindOf<Join>}}, {{0, 2, 0}, {1, 2, 0}}, 2});
    const auto output = f.Shaped(joined.Mesh);
    for (int i = 0; i < 9; ++i) f.AddMesh(primitive::Cube{});
    expect(f.P->Begin(next));
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(chained.Nodes[1], 2u));
    expect(!f.Fault(chained.Nodes[0]) && Mesh{f.Store(), f.Output(chained.Mesh)}.FaceCount() == 54u);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(joined.Nodes[1], vec3{2.f, 1.f, 1.f}));
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(joined.Nodes[1], vec3{1.f, 1.f, 1.f}));
    expect(!f.Fault(joined.Nodes[0]) && f.Shaped(joined.Mesh) == output);
}

// A gesture's steps leave the coarse cluster groups they invalidate dirty, and its commit repairs them.
void GestureLodRepair(GraphFixture &f) {
    using namespace action::mesh;
    const auto mesh = f.AddGenerated(primitive::UVSphere{.Slices = 96, .Stacks = 48});
    const auto smooth = f.Append(mesh, KindOf<SmoothVertices>);
    f.Route(mesh, {f.Node(mesh, 0u), smooth});
    const auto &meshes = f.Store();
    const auto dirty = [&] { return meshes.HasDirtyGroups(meshes.Get(f.Output(mesh))); };
    const auto groups = [&] { return meshes.ClusterGroupCount(meshes.Get(f.Output(mesh))); };
    const auto built = groups();
    expect(built > 0u && !dirty());
    for (const float factor : {.2f, .35f, .5f}) {
        f.Stage(UpdateOf<&SmoothVertices::Factor>(smooth, factor));
        expect(dirty());
    }
    f.CommitGesture();
    expect(!dirty() && groups() == built);
}

// A gesture on a graph whose cook starts from a new record of its root commits the state its recorded actions reach.
// A copied graph starts so, since its nodes hold no state of its own record.
void GestureOnNewRecord(GraphFixture &f) {
    using namespace action::mesh;
    const auto stage = [&](state::Entity object, auto &&updates) {
        f.Do(action::selection::Select{object});
        const auto copy = f.Add(action::object::Duplicate{});
        const auto node = NodesOf(f.R, f.R.get<const ObjectGeometry>(copy).Entity)[1];
        updates(node);
        f.CommitGesture();
    };
    const auto attribute = f.AddGenerated(primitive::Cube{});
    f.Route(attribute, {f.Node(attribute, 0u), f.Append(attribute, KindOf<graph::StoreNamedAttribute>)});
    stage(f.Object(attribute), [&](state::Entity node) {
        for (const auto domain : {AttributeDomain::Vertex, AttributeDomain::Face}) f.Stage(UpdateOf<&graph::StoreNamedAttribute::Domain>(node, domain));
    });
    const auto smoothed = f.AddGenerated(primitive::UVSphere{});
    f.Route(smoothed, {f.Node(smoothed, 0u), f.Append(smoothed, KindOf<SmoothVertices>)});
    stage(f.Object(smoothed), [&](state::Entity node) {
        for (const float factor : {.2f, .35f}) f.Stage(UpdateOf<&SmoothVertices::Factor>(node, factor));
    });
}

// The history panel edits a recorded action's selection hit.
void HistoryPanelEdit(GraphFixture &f) {
    const auto object = f.AddEmpty("Object choice");
    f.AddEmpty("Other");
    const auto picked = f.Do(action::selection::Pick{false, ObjectHit{{object}}});
    const auto hit = [&]() -> std::optional<SelectionHit> & {
        return std::get<action::selection::Pick>(std::get<action::selection::Action>(project::DraftOf(*f.P, picked).RecordedActions.front())).Hit;
    };
    HistoryPanel history{{"History"}};
    ImGui::SetCurrentContext(f.Panels->Context);
    const auto history_frame = [&](ImGuiID activate = 0) {
        if (activate) ImGui::ActivateItemByID(activate);
        ImGui::NewFrame();
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({900, 900});
        project::DrawHistoryPanel(*f.P, history, true);
        ImGui::Render();
        f.P->Frame();
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
}

state::Entity AnimatedAssembly(GraphFixture &f) {
    const auto root = f.AddEmpty("Assembly");
    const auto mesh = f.AddGenerated(primitive::Cube{});
    const auto operand = f.Object(mesh);
    const auto result = f.AddGenerated(primitive::Plane{});
    const auto info = f.Append(result, KindOf<graph::ObjectInfo>, {Set<&graph::ObjectInfo::Object>(operand)});
    f.Do(SetDisplay{result, info});
    const auto light = f.Add(action::object::AddLight{std::make_unique<ObjectCreateInfo>()});
    for (const auto child : {operand, f.Object(result), light}) f.Do(action::object::SetParent{child, root});
    f.R.patch<Animations>(state::Document, [](auto &a) { a.Names = {"Animation"}; a.Active = 0; });
    f.R.emplace<AnimationClips>(root, std::vector<AnimationClip>{{0, {{animation::Target<&Transform::P>(), AnimationInterpolation::Linear, {0, 1}, {0, 0, 0, 2, 0, 0}, {0, 0, 0}}}}});
    f.R.emplace<AnimationClips>(light, std::vector<AnimationClip>{{0, {{animation::Target<&SceneLight::Intensity>(), AnimationInterpolation::Linear, {0, 1}, {2, 8}, {2}}}}});
    f.Commit("Animate assembly");
    return root;
}
state::Entity Mechanism(GraphFixture &f) {
    const auto root = f.AddEmpty("Mechanism"), a = f.AddCube(), b = f.AddCube();
    for (const auto child : {a, b}) {
        f.Do(action::object::SetParent{child, root});
        f.Do(action::physics::SetMotionType{action::physics::SetMotionType::Type::Kinematic, child});
    }
    f.Do(UpdateOf<&Transform::P>(b, vec3{3, 0, 0}));
    const auto description = f.R.create();
    f.R.emplace<JointDescription>(description);
    f.R.emplace<PhysicsJoint>(a, b, description);
    f.R.emplace<TriggerNodes>(root, std::vector{a, b});
    ModalModes modes;
    modes.Freqs = {440};
    modes.T60s = {1};
    modes.Shapes = {{{1, 0, 0}}};
    modes.Positions = {{-1, -1, -1}};
    modes.Vertices = {0};
    // The first sound vertices enter Excite mode, which marks them, and the next ones are marked in that mode.
    for (const auto object : {a, b}) {
        f.R.emplace<ModalModes>(object, modes);
        f.R.emplace<SoundVerticesModel>(object, SoundVerticesModel::Modal);
        f.R.emplace<ModalGain>(object);
        f.Commit("Bind mechanism domains");
        expect(f.R.get<const Interaction>(state::Document).Mode == InteractionMode::Excite);
        expect(f.Store().GetSelectedElements(f.Output(f.R.get<const ObjectGeometry>(object).Entity), Element::Vertex).Count() == 1u);
    }
    return root;
}
state::Entity ImportedRig(GraphFixture &f) {
    f.Do(action::io::LoadGltf{RiggedSimple});
    auto root = *f.R.view<const ArmatureObject>().begin();
    while (ParentOrNull(f.R, root) != state::Null) root = ParentOrNull(f.R, root);
    return root;
}

// An instance of a subgraph that holds an instance keeps its geometry and animated appearance when the placed subgraphs go, and through each way to objects of their own: make unique, expand, and expand of all levels.
void SubgraphInstances(GraphFixture &f, Fixture &other) {
    for (const auto [name, build] : std::array<std::pair<const char *, state::Entity (*)(GraphFixture &)>, 3>{{{"Assembly", AnimatedAssembly}, {"Mechanism", Mechanism}, {"Rig", ImportedRig}}}) {
        Scenario(f, other, std::string{"Instance"} + name, [&](GraphFixture &f) {
            auto &r = f.R;
            const auto source = build(f);
            const auto inner = f.Instantiate(source);
            // Three instances of a subgraph that holds an instance, one for each way to objects of their own.
            const auto first = f.Instantiate(inner);
            const auto inner_subgraph = r.get<const Instance>(inner).SubgraphEntity, outer_subgraph = r.get<const Instance>(first).SubgraphEntity;
            const std::array roots{first, f.Add(action::object::Instantiate{outer_subgraph}), f.Add(action::object::Instantiate{outer_subgraph})};
            for (size_t i = 0; i < roots.size(); ++i) f.Do(UpdateOf<&Transform::P>(roots[i], vec3{2.f * float(i + 1u), 0, 0}));
            std::array<std::vector<SourceCopy>, 3> shown;
            std::vector<std::pair<state::Entity, Shape>> drawn;
            auto original = state::Null;
            for (size_t i = 0; i < roots.size(); ++i) {
                shown[i] = r.get<const Instance>(roots[i]).Copies;
                for (const auto &pair : shown[i]) {
                    if (const auto *object = r.try_get<const ObjectGeometry>(pair.CopyEntity); object && DrawnStoreId(r, object->Entity)) {
                        expect(f.Shaped(object->Entity) == f.Shaped(r.get<const ObjectGeometry>(pair.SourceEntity).Entity));
                        drawn.emplace_back(pair.CopyEntity, f.Shaped(object->Entity));
                        original = SourceOfCopy(r, pair.CopyEntity);
                    }
                }
            }
            // Deleting the placed subgraphs leaves their items in the document for the instances, which stay as they are.
            f.Do(action::nodes::DeleteItems{{inner_subgraph, outer_subgraph}});
            expect(r.valid(original) && !RootMembership(r, original));
            for (size_t i = 0; i < roots.size(); ++i) expect(r.get<const Instance>(roots[i]).Copies == shown[i]);
            f.Do(action::selection::DeselectAll{});
            // The second image shows Object mode, which a restore keeps where the objects have sound vertices.
            const auto mode = r.get<const Interaction>(state::Document).Mode;
            std::array<std::vector<std::byte>, 2> images;
            for (int frame = 0; frame < 2; ++frame) {
                f.Do(action::timeline::SetFrame{1 + frame * 30});
                if (frame == 1 && mode != InteractionMode::Object) f.Do(action::view::SetInteractionMode{InteractionMode::Object});
                if (Render) images[frame] = f.Image();
            }
            const auto subgraphs = r.view<const graph::SubgraphItems>().size();
            const auto active = FindActiveEntity(r);
            // Make unique acts on one level: the copy of the instance inside stays an instance, whose copies keep sharing the geometry.
            const auto nested = CopyFor(shown[0], inner);
            f.Do(action::nodes::MakeUnique{roots[0]});
            expect(!r.all_of<Instance>(roots[0]) && r.valid(original) && r.get<const Instance>(nested).SubgraphEntity == inner_subgraph);
            for (const auto &pair : r.get<const Instance>(nested).Copies) {
                expect(r.get<const CopyOf>(pair.CopyEntity).ObjectEntity == nested && CopyFor(shown[0], CopyFor(r.get<const Instance>(inner).Copies, pair.SourceEntity)) == pair.CopyEntity);
                if (const auto *object = r.try_get<const ObjectGeometry>(pair.SourceEntity); object && r.all_of<graph::GeometryGraph>(object->Entity)) expect(r.get<const ObjectGeometry>(pair.CopyEntity).Entity == object->Entity);
            }
            f.Do(action::nodes::MakeUnique{nested});
            // Expand also acts on one level and leaves plain objects, and with all levels it expands the instance inside too.
            f.Do(action::nodes::Expand{roots[1]});
            expect(r.all_of<Instance>(CopyFor(shown[1], inner)));
            f.Do(action::nodes::Expand{CopyFor(shown[1], inner)});
            f.Do(action::nodes::Expand{roots[2], true});
            // Expand selects what replaced the instance, and the images show the object that was active before.
            f.Do(action::selection::Select{active});
            f.Do(action::selection::DeselectAll{});
            for (int frame = 1; frame >= 0; --frame) {
                f.Do(action::timeline::SetFrame{1 + frame * 30});
                if (frame == 0 && mode != InteractionMode::Object) f.Do(action::view::SetInteractionMode{mode});
                if (Render) expect(f.Image() == images[frame]);
            }
            // Each copy is an object of its own with the geometry it drew, and a geometry with a graph is its alone.
            // The document drops the subgraphs it kept for the instances, and only the two that make unique made remain.
            std::vector<state::Entity> geometries;
            for (const auto &[copy, shape] : drawn) {
                const auto geometry = r.get<const ObjectGeometry>(copy).Entity;
                expect(f.Shaped(geometry) == shape);
                if (r.all_of<graph::GeometryGraph>(geometry)) geometries.push_back(geometry);
            }
            std::ranges::sort(geometries);
            expect(std::ranges::adjacent_find(geometries) == geometries.end());
            for (const auto &copies : shown) {
                for (const auto &pair : copies) expect(r.valid(pair.CopyEntity) && !r.any_of<CopyOf, Instance>(pair.CopyEntity));
            }
            expect(!r.valid(original) && !r.valid(inner) && r.view<const graph::SubgraphItems>().size() == subgraphs);
        });
    }
    // Each instance of a subgraph of two objects shows both, follows an edit of either, and keeps its own value of a port through an edit of the subgraph.
    Scenario(f, other, "InstancePorts", [](GraphFixture &f) {
        auto &r = f.R;
        const auto a = f.AddCube(), b = f.AddCube();
        f.Do(UpdateOf<&Transform::P>(b, vec3{3, 0, 0}));
        const auto camera = f.Add(action::object::AddCamera{std::make_unique<ObjectCreateInfo>(), {}});
        const auto objects = r.view<const ObjectKind>().size();
        f.Do(action::nodes::Collapse{FrameScene(r), {a, b, camera}});
        const auto subgraph = FindActiveNode(r);
        // Collapsing copies nothing.
        expect(r.view<const ObjectKind>().size() == objects && r.view<const CopyOf>().empty());
        const std::array instances{f.Add(action::object::Instantiate{subgraph}), f.Add(action::object::Instantiate{subgraph})};
        for (size_t i = 0; i < 2u; ++i) f.Do(UpdateOf<&Transform::P>(instances[i], vec3{0, 5.f * float(i + 1u), 0}));
        const auto copy = [&](size_t i, state::Entity item) { return CopyFor(r.get<const Instance>(instances[i]).Copies, item); };
        const auto held = [&](state::Entity e) { return r.get<const Transform>(e).P; };
        for (size_t i = 0; i < 2u; ++i) {
            for (const auto item : {a, b}) {
                expect(fatal(copy(i, item) != item));
                expect(ParentOrNull(r, copy(i, item)) == instances[i] && r.get<const ObjectGeometry>(copy(i, item)).Entity == r.get<const ObjectGeometry>(item).Entity && held(copy(i, item)) == held(item));
            }
        }
        // An edit of an item reaches every instance, through the item or through a copy of it.
        f.Do(UpdateOf<&Transform::P>(a, vec3{1, 0, 0}));
        expect(held(copy(0u, a)) == vec3{1, 0, 0} && held(copy(1u, a)) == vec3{1, 0, 0});
        f.Do(UpdateOf<&Transform::P>(copy(0u, a), vec3{2, 0, 0}));
        expect(held(a) == vec3{2, 0, 0} && held(copy(1u, a)) == vec3{2, 0, 0});
        f.Do(action::object::SetProjection{true, copy(0u, camera)});
        expect(r.all_of<Orthographic>(camera) && r.all_of<Orthographic>(copy(0u, camera)) && r.all_of<Orthographic>(copy(1u, camera)));
        // A hit on a copy in the viewport is a hit on the object whose Instance shows it.
        for (size_t i = 0; i < 2u; ++i) expect(HitEntity(ResolveHits(r, std::array{copy(i, a)}, false).front()) == instances[i]);
        // A port's value on one instance is that instance's alone, set at the port or through its copy's field, and an edit of the subgraph leaves it.
        f.Do(action::nodes::ExposePort{subgraph, graph::PromotedField{b, state::Key<Transform>(), uint16_t(offsetof(Transform, P))}, std::make_unique<std::string>("Position")});
        const auto port = r.get<const graph::SubgraphInterface>(subgraph).Ports.front().Id;
        std::array<uint32_t, 4> words{};
        const vec3 own{7, 0, 0};
        std::memcpy(words.data(), &own, sizeof(own));
        f.Do(action::nodes::SetPortValue{instances[0], port, words});
        expect(held(copy(0u, b)) == own && held(copy(1u, b)) == vec3{3, 0, 0} && held(b) == vec3{3, 0, 0});
        // The Properties panel draws a copy with a field of the instance's own, and one without.
        for (const auto item : {a, b}) {
            f.Do(action::selection::Select{copy(0u, item)});
            DrawPanels(f, *f.Panels);
        }
        // A duplicate of an object alone has nothing below it, and takes the object's place: below its parent, at its local transform and in the subgraph that holds it.
        // Duplicated with a selected object below it, it holds that object's duplicate, which has an Instance of the same subgraph with the same port values.
        const auto above = f.AddEmpty("Above");
        f.Do(action::object::SetParent{instances[0], above});
        for (const bool linked : {false, true}) {
            const auto duplicated = [&] { return linked ? f.Add(action::object::DuplicateLinked{}) : f.Add(action::object::Duplicate{}); };
            f.Do(action::selection::Select{above});
            const auto alone = duplicated();
            expect(std::ranges::empty(Children{&r, alone}));
            f.Do(action::object::Delete{});
            f.Do(action::selection::Select{instances[0]});
            const auto sibling = duplicated();
            expect(ParentOrNull(r, sibling) == above && r.get<const Transform>(sibling) == r.get<const Transform>(instances[0]));
            f.Do(action::object::Delete{});
            f.Do(action::selection::Select{b});
            const auto twin = duplicated();
            expect(SubgraphHolding(r, twin) == subgraph && copy(0u, twin) != twin && copy(1u, twin) != twin);
            f.Do(action::object::Delete{});
            f.Do(action::selection::Select{instances[0]});
            f.Do(action::selection::ExtendActive{above});
            const auto duplicate = duplicated();
            const auto below = *Children{&r, duplicate}.begin();
            expect(fatal(r.all_of<Instance>(below)));
            expect(r.get<const Instance>(below).SubgraphEntity == subgraph && held(CopyFor(r.get<const Instance>(below).Copies, b)) == own);
            f.Do(action::selection::Select{below});
            f.Do(action::object::Delete{});
        }
        f.Do(UpdateOf<&Transform::P>(b, vec3{4, 0, 0}));
        expect(held(copy(0u, b)) == own && held(copy(1u, b)) == vec3{4, 0, 0});
        f.Do(UpdateOf<&Transform::P>(copy(1u, b), vec3{9, 0, 0}));
        expect(held(copy(1u, b)) == vec3{9, 0, 0} && held(copy(0u, b)) == own && held(b) == vec3{4, 0, 0});
        f.Do(action::nodes::ClearPortValue{instances[0], port});
        expect(held(copy(0u, b)) == vec3{4, 0, 0} && held(copy(1u, b)) == vec3{9, 0, 0});
        // A copy hangs below the copy of its item's parent, and below the object of the Instance where the parent is no item of the subgraph.
        const auto outside = f.AddEmpty("Outside");
        f.Do(action::object::SetParent{a, outside});
        f.Do(action::object::SetParent{b, a});
        for (size_t i = 0; i < 2u; ++i) expect(ParentOrNull(r, copy(i, a)) == instances[i] && ParentOrNull(r, copy(i, b)) == copy(i, a));
        // The deleted subgraph stays in the document for its instances, and goes with the last of them.
        const auto copies = r.get<const Instance>(instances[1]).Copies;
        f.Do(action::nodes::DeleteItems{{subgraph}});
        expect(r.valid(a) && !RootMembership(r, a) && graph::GraphRoot(r, subgraph) == state::Document && r.get<const Instance>(instances[1]).Copies == copies && held(copy(1u, b)) == vec3{9, 0, 0});
        for (const auto instance : instances) {
            f.Do(action::selection::Select{instance});
            f.Do(action::object::Delete{});
        }
        expect(!r.valid(a) && !r.valid(b) && !r.valid(subgraph));
    });
}

// An instance's copy of an object draws the object's geometry, whose graph reads the objects it refers to.
// A copy that becomes an object of its own, by making the instance unique or by duplicating the hierarchy, draws its own geometry.
// That geometry's graph and the subgraphs it uses refer to the copies, and keep the result.
void ContextualSubgraph(GraphFixture &f) {
    auto &r = f.R;
    // The nodes of the mesh's graph and of each subgraph it uses.
    const auto nodes_of = [&](state::Entity mesh) {
        auto nodes = NodesOf(r, mesh);
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (const auto *use = r.try_get<const SubgraphNode>(nodes[i]); use && UserSubgraph(use->SubgraphId) != state::Null) nodes.append_range(NodesOf(r, UserSubgraph(use->SubgraphId)));
        }
        return nodes;
    };
    for (const bool collapsed : {false, true}) {
        for (const bool instanced : {false, true}) {
            const auto root = f.AddEmpty("Assembly");
            const auto mesh = f.AddGenerated(primitive::Cube{});
            const auto object = f.Object(mesh), operand = f.Object(f.AddGenerated(primitive::Cube{}));
            f.Do(UpdateOf<&Transform::P>(operand, vec3{.5f, .25f, 0.f}));
            const auto boolean = f.Append(mesh, KindOf<graph::Boolean>, {Set<&graph::Boolean::Operation>(BooleanOperation::Union)});
            const auto info = f.Append(mesh, KindOf<graph::ObjectInfo>, {Set<&graph::ObjectInfo::Object>(operand)});
            f.Do(UpdateOf<&graph::Boolean::Space>(boolean, object));
            f.Do(Link{mesh, f.Node(mesh, 0u), f.Out(f.Node(mesh, 0u)), boolean, 0u});
            f.Do(Link{mesh, info, f.Out(info), boolean, 1u});
            f.Do(SetDisplay{mesh, boolean});
            if (collapsed) f.Do(action::nodes::Collapse{mesh, {boolean, info}});
            for (const auto child : {operand, object}) f.Do(action::object::SetParent{child, root});
            const auto shape = f.Shaped(mesh);
            // The copies of the graph's object and of the operand, as objects of their own.
            auto own_object = state::Null, own_operand = state::Null;
            if (instanced) {
                const auto instance = f.Instantiate(root);
                f.Do(UpdateOf<&Transform::P>(instance, vec3{2.f, 0.f, 0.f}));
                const auto copies = r.get<const Instance>(instance).Copies;
                own_object = CopyFor(copies, object);
                own_operand = CopyFor(copies, operand);
                expect(r.get<const ObjectGeometry>(own_object).Entity == mesh);
                f.Do(action::nodes::MakeUnique{instance});
            } else {
                f.Do(action::selection::Select{object});
                for (const auto e : {operand, root}) f.Do(action::selection::ExtendActive{e});
                const auto copy = f.Add(action::object::Duplicate{});
                f.Do(UpdateOf<&Transform::P>(copy, vec3{2.f, 0.f, 0.f}));
                for (const auto child : Children{&r, copy}) (NodesOf(r, r.get<const ObjectGeometry>(child).Entity).size() > 1u ? own_object : own_operand) = child;
            }
            const auto own = r.get<const ObjectGeometry>(own_object).Entity;
            expect(own != mesh && f.Shaped(own) == shape);
            const auto authored = nodes_of(mesh);
            for (const auto node : nodes_of(own)) {
                expect(!std::ranges::contains(authored, node));
                if (const auto *read = r.try_get<const graph::ObjectInfo>(node)) expect(read->Object == own_operand);
                if (const auto *combined = r.try_get<const graph::Boolean>(node)) expect(combined->Space == own_object);
            }
        }
    }
}

// A selected part of an object, such as a bone's joint, duplicates nothing on its own.
void DuplicateSkipsParts(GraphFixture &f) {
    ImportedRig(f);
    const auto part = *f.R.view<const BoneSubPartOf>().begin();
    f.Do(action::selection::Select{part});
    for (const bool linked : {false, true}) {
        const auto living = f.R.Living.size();
        if (linked) f.Do(action::object::DuplicateLinked{});
        else f.Do(action::object::Duplicate{});
        expect(f.R.Living.size() == living && FindActiveEntity(f.R) == part);
    }
}
void LinkedSkinPoses(GraphFixture &f) {
    f.Do(action::io::LoadGltf{RiggedSimple});
    auto &r = f.R;
    const auto skinned = *r.view<const ArmatureModifier>().begin();
    const auto modifier = r.get<const ArmatureModifier>(skinned);
    const auto first = modifier.ArmatureObjectEntity;
    f.Do(action::selection::Select{first});
    const auto second = f.Add(action::object::DuplicateLinked{});
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
        const auto &layout = r.get<const PosedLayout>(r.get<const ObjectGeometry>(skinned).Entity);
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

// The mesh's source after the position edit over every vertex, which reads the parameter attribute holding `value` at each vertex when given.
Shape PositionEdited(GraphFixture &f, state::Entity mesh, PositionEditOp op, float factor, uint32_t repeat, PositionOperationOptions options, float (*value)(const Mesh &, he::VH) = nullptr) {
    auto &meshes = f.Store();
    return EditedSource(f, mesh, [&](uint32_t clone) {
        if (value) options.Input = meshes.AttributeRef(AddValues<float>(meshes, clone, "parameter", value));
        ExecuteGeometryPositions(f.R, {.StoreId = clone, .Selection = AllHandles(Mesh{meshes, clone})}, op, factor, repeat, options);
    });
}
// Shrink/Fatten moves each vertex by the width the ops compute from its position.
// The kernel reads the factor at each vertex from the attribute, past the constant factor's zero check.
void ExpectWidthsShrunk(GraphFixture &f, const BuiltGraph &g) { expect(f.Shaped(g.Mesh) == PositionEdited(f, g.Mesh, PositionEditOp::ShrinkFatten, 1.f, 1u, {}, WidthAt)); }
// Smooth moves each vertex by the factor the ops compute from its height, which a repeated smooth reads once from its input.
void ExpectSmoothFactors(GraphFixture &f, const BuiltGraph &g) {
    const auto smooth = f.R.get<const action::mesh::SmoothVertices>(g.Nodes[5]);
    expect(Near(f.Shaped(g.Mesh).Vertices, PositionEdited(f, g.Mesh, PositionEditOp::Smooth, smooth.Factor, smooth.Repeat, {}, [](const Mesh &mesh, he::VH v) { return .25f * test::Position(mesh, v).z + .5f; }).Vertices));
}
// Randomize moves each vertex by the amount the ops compute from its distance to the origin.
void ExpectRandomAmounts(GraphFixture &f, const BuiltGraph &g) {
    const auto randomize = f.R.get<const action::mesh::Randomize>(g.Nodes[4]);
    expect(Near(f.Shaped(g.Mesh).Vertices, PositionEdited(f, g.Mesh, PositionEditOp::Randomize, randomize.Amount, 1u, {.Randomize = PositionRandomize{randomize.Uniform, randomize.Normal, randomize.Seed}}, [](const Mesh &mesh, he::VH v) { return .1f * Length(test::Position(mesh, v)); }).Vertices));
}
// Set Position doubles the smoothed positions and raises them by one.
void ExpectSmoothedOffsets(GraphFixture &f, const BuiltGraph &g) {
    const auto offset = PositionEdited(f, g.Mesh, PositionEditOp::Smooth, .5f, 1u, {}).Vertices | std::views::transform([](const Corner &p) { return Corner{2.f * p[0], 2.f * p[1], 2.f * p[2] + 1.f}; }) | std::ranges::to<std::vector>();
    expect(Near(f.Shaped(g.Mesh).Vertices, offset));
}
// Poke offsets each face of the source by `offset` at the face.
void ExpectPokes(GraphFixture &f, state::Entity mesh, float (*offset)(const Mesh &, he::FH)) {
    expect(f.Shaped(mesh) == TopologyReference(f, mesh, MeshTopologyOp::Poke, [&](uint32_t clone) { return f.Store().AttributeRef(AddValues<float, AttributeDomain::Face>(f.Store(), clone, "parameter", offset)); }));
}
// A face reads the mean of its vertices' positions, so each face's offset is the width at its center.
void ExpectCenterPokes(GraphFixture &f, const BuiltGraph &g) {
    expect(!f.Fault(g.Nodes[1]));
    ExpectPokes(f, g.Mesh, [](const Mesh &m, he::FH face) {
        vec3 sum{};
        float count = 0.f;
        for (const auto v : test::FaceVertices(m, face)) {
            sum += test::Position(m, v);
            count += 1.f;
        }
        const auto center = sum * (1.f / count);
        return .05f * (center.x + center.y + center.z) + .2f;
    });
}
void ExpectCuts(GraphFixture &f, state::Entity mesh, float cuts) {
    expect(f.Shaped(mesh) == EditedSource(f, mesh, [&](uint32_t clone) { f.Run(clone, {Op(MeshTopologyOp::Subdivide, cuts, TopologyFlagSelectAll)}); }));
}

// Whether the shapes are equal, with their sorted vertex positions equal bit for bit.
bool SameBits(const Shape &a, const Shape &b) {
    const auto bits = [](const Shape &shape) { return shape.Vertices | std::views::join | std::views::transform([](float c) { return std::bit_cast<uint32_t>(c); }) | std::ranges::to<std::vector>(); };
    return a == b && bits(a) == bits(b);
}
using action::mesh::DeleteMode;
constexpr auto Cuts = ParameterSocket<&action::mesh::Subdivide::Cuts>;
// The stored output type of an op node.
NodeUpdate OutputType(mx::Type type) {
    return [type](GraphFixture &f, state::Entity node) { f.Do(action::nodes::SetOpOutputType{node, type}); };
}
// The node's selection applies to the domain and selects every element while unlinked.
NodeUpdate SelectsOn(Element domain) {
    return [domain](GraphFixture &f, state::Entity node) { f.Do(action::nodes::SetInlineSelection{node, std::make_unique<InlineSelection>(InlineSelection{.Domain = domain, .All = true})}); };
}
// Plane Side selects the positions in front of the plane through (-0.5, 0, -0.5) facing +x and +z.
const std::vector<NodeUpdate> FrontOfDiagonal{Literal("point", vec3{-.5f, 0.f, -.5f}), Literal("normal", vec3{1.f, 0.f, 1.f}), Literal("side", 0)};
// A Convert Domain node's source domain, target domain and mode.
NodeUpdate Converts(std::string_view source, std::string_view target, std::string_view mode) {
    return [source, target, mode](GraphFixture &f, state::Entity node) {
        for (const auto [input, value] : {std::pair{"source", source}, std::pair{"target", target}, std::pair{"mode", mode}}) f.Do(OpInputOf(node, input, std::string{value}));
    };
}
// Set Position sets each vertex to twice its position.
const GraphDescription DoubledPositions{primitive::Cube{}, {{"position"}, {"multiply", {Literal("in2", 2.f)}}, {KindOf<SetPosition>, {Set<&SetPosition::Offset>(false)}}}, {{1, 2, 0, "in1"}, {2, 3, 1}, {0, 3, 0}}, 3};
// Ops before, between and after two Convert Domain nodes offset the positions.
const GraphDescription ConvertedOffsets{primitive::Cube{}, {{"position"}, {"multiply", {Literal("in2", 2.f)}}, {"convertdomain", {Converts("vertex", "face", "mean")}}, {"add", {Literal("in2", vec3{0.f, 0.f, 1.f})}}, {"convertdomain", {Converts("face", "vertex", "max")}}, {"multiply", {Literal("in2", .25f)}}, {KindOf<SetPosition>}}, {{1, 2, 0, "in1"}, {2, 3, 0, "in"}, {3, 4, 0, "in1"}, {4, 5, 0, "in"}, {5, 6, 0, "in1"}, {6, 7, 1}, {0, 7, 0}}, 7};
// The same graph with each conversion's result written by a Store node on its domain and read back.
const GraphDescription StoredOffsets{primitive::Cube{}, {{"position"}, {"multiply", {Literal("in2", 2.f)}}, {"convertdomain", {Converts("vertex", "face", "mean")}}, {KindOf<StoreNamedAttribute>, {Named("a"), Set<&StoreNamedAttribute::Domain>(AttributeDomain::Face)}}, {"geompropvalue", {Text("geomprop", "a"), Literal("default", vec3{})}}, {"add", {Literal("in2", vec3{0.f, 0.f, 1.f})}}, {"convertdomain", {Converts("face", "vertex", "max")}}, {KindOf<StoreNamedAttribute>, {Named("b")}}, {"geompropvalue", {Text("geomprop", "b"), Literal("default", vec3{})}}, {"multiply", {Literal("in2", .25f)}}, {KindOf<SetPosition>}}, {{1, 2, 0, "in1"}, {2, 3, 0, "in"}, {3, 4, 1}, {0, 4, 0}, {5, 6, 0, "in1"}, {6, 7, 0, "in"}, {7, 8, 1}, {4, 8, 0}, {9, 10, 0, "in1"}, {10, 11, 1}, {8, 11, 0}}, 11};
// The rotation leaves no two moved corners a GPU rounding apart in sort order.
const Transform ConstantDelta{{1.f, 2.f, 3.f}, numeric::AngleAxis(.5f, Normalize(vec3{1.f, 2.f, 3.f})), {2.f, 1.f, .5f}};
// A geompropvalue op that finds no attribute of its name and type shows that it found none, and Set Position reads its default.
void ExpectProbeDefault(GraphFixture &f, const BuiltGraph &g) {
    expect(f.Fault(g.Nodes[1]) == NodeFault::NotFound && !f.Fault(g.Nodes[2]));
    expect(Near(f.Shaped(g.Mesh).Vertices, SourcePoints(f, g.Mesh, [](uint32_t, vec3 p) { return p + vec3{0.f, 0.f, 1.f}; })));
}
// A program on corners reads each corner's shading normal as the viewport shades it: its face's on a flat face, its sector's beside a sharp edge, and else its vertex's.
void ExpectCornerNormals(GraphFixture &f, const BuiltGraph &g) {
    expect(!f.Fault(g.Nodes[10]));
    const auto &meshes = f.Store();
    const auto id = f.Output(g.Mesh);
    expect(meshes.Get(id).Classification == uint32_t(CornerClassMode::Mixed));
    const auto shading = meshes.GetCornerNormalView(id);
    const auto stored = meshes.FindAttribute(id, "shading");
    expect(fatal(stored != InvalidOffset));
    const Mesh output{meshes, id};
    std::set<std::array<float, 3>> normals;
    for (const auto face : output.faces()) {
        for (const auto corner : test::FaceHalfedges(output, face)) {
            const auto expected = Normalize(shading[*corner]);
            const auto words = test::AttributeWords(meshes, stored, *corner);
            expect(Length(vec3{std::bit_cast<float>(words[0]), std::bit_cast<float>(words[1]), std::bit_cast<float>(words[2])} - expected) < 1e-5f);
            normals.insert({std::round(expected.x * 1e3f), std::round(expected.y * 1e3f), std::round(expected.z * 1e3f)});
        }
    }
    // Flat, sector and vertex normals give the corners more distinct normals than the six faces have.
    expect(normals.size() > 6u);
}
// Four points on a line, each with a color of its own, which a record without faces holds on its vertices.
MeshSource ColoredPoints() {
    MeshSource points;
    auto &colors = points.Attrs.Colors0.emplace();
    points.Attrs.Colors0ComponentCount = 4u;
    for (uint32_t i = 0u; i < 4u; ++i) {
        points.Data.Positions.emplace_back(float(i), 0.f, 0.f);
        colors.emplace_back(float(i) / 4.f, .25f, .5f, 1.f);
    }
    points.Data.Edges = {{0u, 1u}, {1u, 2u}, {2u, 3u}};
    return points;
}
// A color read on the vertices of a record without faces is each vertex's own color.
void ExpectVertexColors(GraphFixture &f, const BuiltGraph &g) {
    const auto &meshes = f.Store();
    const auto id = f.Output(g.Mesh);
    const auto stored = meshes.FindAttribute(id, "color");
    expect(fatal(stored != InvalidOffset));
    const auto &a = meshes.Arenas();
    for (const auto v : Mesh{meshes, id}.vertices()) {
        const auto words = test::AttributeWords(meshes, stored, *v);
        const auto expected = a.VertexColors.Get(meshes.Get(id).VertexColors, *v);
        for (uint32_t c = 0u; c < 4u; ++c) expect(std::bit_cast<float>(words[c]) == expected[c]);
    }
}
// A Bevel without a selection of its own evaluates its linked selection on edges, as a selection the Domain combo sets to edges does, which selects other elements than one on vertices.
void ExpectNaturalEdges(GraphFixture &f, const BuiltGraph &g) {
    const auto natural = f.Shaped(g.Mesh);
    const auto shaped = [&](Element domain) {
        SelectsOn(domain)(f, g.Nodes[2]);
        return f.Shaped(g.Mesh);
    };
    expect(natural == shaped(Element::Edge) && natural != shaped(Element::Vertex));
}
const std::array<GraphData, 56> DataGraphs{{
    {"ExtrudeSubdivide", {primitive::Plane{}, {{KindOf<graph::Extrude>}, {KindOf<action::mesh::Subdivide>, {Set<&action::mesh::Subdivide::Cuts>(2u)}}}, {{0, 1, 0}, {1, 2, 0}}, 2}, nullptr, [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 2u), 3u)); }},
    {"ScalePositions", DoubledPositions, [](GraphFixture &f, const BuiltGraph &g) { expect(Near(f.Shaped(g.Mesh).Vertices, SourcePoints(f, g.Mesh, [](uint32_t, vec3 p) { return 2.f * p; }))); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 2u), "in2", 3.f); }, CalledInput{1u, vec3{}}},
    // The selection evaluates on vertices in place of Extrude's faces, so its program reads the positions where they are stored.
    {"CompareSelectsExtrusion", {primitive::Cube{}, {{"position"}, {"extract", {Literal("index", 2)}}, {"ifgreater", {Literal("value2", 0.f)}}, {KindOf<graph::Extrude>, {SelectsOn(Element::Vertex)}}}, {{1, 2, 0, "in"}, {2, 3, 0, "value1"}, {3, 4, 1}, {0, 4, 0}}, 4}, ExpectUpperExtruded, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 2u), "index", 0); }, CalledInput{2u, false}},
    {"NamedAttributeReads", {primitive::Plane{}, {{"geompropvalue", {Text("geomprop", "offset"), Literal("default", vec3{})}}, {KindOf<SetPosition>}}, {{1, 2, 1}, {0, 2, 0}}, 2, AddOffsets}, [](GraphFixture &f, const BuiltGraph &g) { expect(Near(f.Shaped(g.Mesh).Vertices, SourcePoints(f, g.Mesh, [](uint32_t v, vec3 p) { return p + OffsetOf(v); }))); }, [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&SetPosition::Offset>(f.Node(mesh, 2u), false)); }, CalledInput{0u, vec3{}}},
    // The rings at the major radius have zero curvature and stay with either comparison.
    {"CurvatureSelectsOuterTorus", {primitive::Torus{}, {{"curvature", {Text("kind", "gaussian")}}, {"ifgreater", {Literal("value2", .1f)}}, {KindOf<action::mesh::Delete>}}, {{1, 2, 0, "value1"}, {2, 3, 1}, {0, 3, 0}}, 3}, [](GraphFixture &f, const BuiltGraph &g) {
         const auto [outer, inner] = OuterInner(f, g.Mesh);
         expect(outer == 0u && inner > 0u) << "outer" << outer << "inner" << inner; }, [](GraphFixture &f, state::Entity mesh) {
         // The curvature moves to the second operand in one gesture, so the comparison takes the curvatures below the threshold.
         const auto curvature = f.Node(mesh, 1u), compare = f.Node(mesh, 2u);
         const auto first = f.OpSocket(compare, "value1");
         f.Stage(action::nodes::Unlink{mesh, compare, first}, OpInputOf(compare, "value1", action::nodes::OpLiteral{mx::Type::Float, {std::bit_cast<uint32_t>(.1f)}}), Link{mesh, curvature, f.Out(curvature), compare, f.OpSocket(compare, "value2")});
         f.CommitGesture();
         const auto [outer, inner] = OuterInner(f, mesh);
         expect(inner == 0u && outer > 0u) << "outer" << outer << "inner" << inner; }, CalledInput{1u, false}},
    {"StorePromotes", {primitive::Plane{}, {{"position"}, {KindOf<StoreNamedAttribute>, {Named("rest")}}, {KindOf<action::mesh::Subdivide>, {Set<&action::mesh::Subdivide::Cuts>(2u)}}}, {{1, 2, 1}, {0, 2, 0}, {2, 3, 0}}, 3}, [](GraphFixture &f, const BuiltGraph &g) { expect(RestMatches(f, g.Mesh)); }, [](GraphFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 3u), 3u));
         expect(RestMatches(f, mesh)); }},
    // An and op reads a vertex and a face field on Delete's vertices, where the face field holds at a vertex when a face around it does.
    // The selection is the 16 boundary vertices, and deleting them leaves the middle four faces.
    // Deleting faces reads both fields on faces, where the vertex field holds at a face when all its vertices do, which no face's do.
    {"DomainMismatch", {primitive::Plane{.Subdivisions = {4u, 4u}}, {{"boundary", {Text("domain", "vertex")}}, {"boundary", {Text("domain", "face")}}, {"and"}, {KindOf<action::mesh::Delete>}}, {{1, 3, 0, "in1"}, {2, 3, 0, "in2"}, {3, 4, 1}, {0, 4, 0}}, 4}, [](GraphFixture &f, const BuiltGraph &g) { expect(!f.Fault(g.Nodes[3])); }, [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Delete::Mode>(f.Node(mesh, 4u), DeleteMode::Faces)); }, {}, 4u, 16u},
    {"JoinGenerated", {primitive::Cube{}, {{KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{2.f, .5f, 1.f})}}, {KindOf<Join>}}, {{0, 2, 0}, {1, 2, 0}}, 2, AddWeights}, ExpectJoinedDefaults, [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 1u), vec3{.5f, .5f, .5f})); }},
    {"BooleanIntersection", {primitive::Cube{}, {{KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{2.f, .25f, .75f})}}, {KindOf<graph::Boolean>, {Set<&graph::Boolean::Operation>(BooleanOperation::Intersection)}}}, {{0, 2, 0}, {1, 2, 1}}, 2}, [](GraphFixture &f, const BuiltGraph &g) { expect(std::abs(OutputVolume(f, g.Mesh) - 1.5) < 1e-4); }, [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&graph::Boolean::Operation>(f.Node(mesh, 2u), BooleanOperation::Union)); }},
    // Deleting the top vertices takes the sides too, and Fill declines the loop that already bounds the remaining face.
    {"BoundaryFillsHole", {primitive::Cube{}, {{"position"}, {"extract", {Literal("index", 2)}}, {"ifgreater", {Literal("value2", .9f)}}, {KindOf<action::mesh::Delete>, {Set<&action::mesh::Delete::Mode>(DeleteMode::Faces)}}, {"boundary"}, {KindOf<action::mesh::Fill>}}, {{1, 2, 0, "in"}, {2, 3, 0, "value1"}, {3, 4, 1}, {5, 6, 1}, {0, 4, 0}, {4, 6, 0}}, 6}, [](GraphFixture &f, const BuiltGraph &g) {
         const Mesh output{f.Store(), f.Output(g.Mesh)};
         expect(output.FaceCount() == 6u && output.HasClosedSurface()) << "faces" << output.FaceCount();
         expect(std::abs(test::EnclosedVolume(output).value_or(-1.0) - 8.0) < 1e-4); }, [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Delete::Mode>(f.Node(mesh, 4u), DeleteMode::Vertices)); }, CalledInput{2u, false}, {}, 1u},
    // A threshold above every vertex seeds nothing, so both cubes stay.
    {"LinkedDeletesComponent", {primitive::Cube{}, {{KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{.5f, .5f, .5f})}}, {KindOf<Join>}, {"position"}, {"extract", {Literal("index", 2)}}, {"ifgreater", {Literal("value2", .9f)}}, {"linked"}, {KindOf<action::mesh::Delete>}}, {{0, 2, 0}, {1, 2, 0}, {2, 7, 0}, {3, 4, 0, "in"}, {4, 5, 0, "value1"}, {5, 6, 0}, {6, 7, 1}}, 7}, [](GraphFixture &f, const BuiltGraph &g) {
         const Mesh output{f.Store(), f.Output(g.Mesh)};
         expect(output.VertexCount() == 8u && output.FaceCount() == 6u) << "vertices" << output.VertexCount() << "faces" << output.FaceCount();
         expect(std::abs(test::EnclosedVolume(output).value_or(-1.0) - 1.0) < 1e-4); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 5u), "value2", 10.f); }, {}, {}, 12u},
    // Seeding Adjacent, at More by one step, with the plane's three vertex columns past x = 1.5 grows the seed by the column beside it, which Delete removes with the seed.
    {"AdjacentDeletesRing", {primitive::Plane{.HalfExtents = {4.f, 4.f}, .Subdivisions = {8u, 8u}}, {{"position"}, {"extract", {Literal("index", 0)}}, {"ifgreater", {Literal("value2", 1.5f)}}, {"adjacent"}, {KindOf<action::mesh::Delete>}}, {{0, 5, 0}, {1, 2, 0, "in"}, {2, 3, 0, "value1"}, {3, 4, 0}, {4, 5, 1}}, 5}, [](GraphFixture &f, const BuiltGraph &g) { ExpectColumnsThrough(f, g.Mesh, 0); },
     // Less shrinks the seed by its column beside the rest, so Delete removes the two columns past x = 2.5.
     [](GraphFixture &f, state::Entity mesh) {
         Text("mode", "less")(f, f.Node(mesh, 4u));
         ExpectColumnsThrough(f, mesh, 2);
     }},
    // Plane Side holds the edge on the +x border of the row of faces at the greatest z, whose ring is the row's five edges along z, which Adjacent grows by the 13 edges touching them.
    // Deleting the 18 edges deletes the eight faces of the two rows at the greatest z.
    // Every ring edge touches an unselected edge, so shrinking the ring selects nothing.
    {"EdgeRingAdjacentDeletes", {primitive::Plane{.HalfExtents = {2.f, 2.f}, .Subdivisions = {4u, 4u}}, {{"planeside", {Literal("point", vec3{2.625f, 0.f, 0.f}), Literal("normal", vec3{2.f, 0.f, 1.f}), Literal("side", 0)}}, {"edgering"}, {"adjacent", {Text("domain", "edge")}}, {KindOf<action::mesh::Delete>, {Set<&action::mesh::Delete::Mode>(DeleteMode::Edges)}}}, {{1, 2, 0}, {2, 3, 0}, {3, 4, 1}, {0, 4, 0}}, 4}, nullptr, [](GraphFixture &f, state::Entity mesh) { Text("mode", "less")(f, f.Node(mesh, 3u)); }, {}, 8u, 16u},
    // The open box's rim is its boundary, and the box is the component holding it, so Delete removes the box and keeps the cube inside it.
    // No face has only rim edges, so the face components hold nothing and Delete keeps both meshes.
    {"BoundaryLinkedDeletes", {primitive::Cube{}, {{"position"}, {"extract", {Literal("index", 2)}}, {"ifgreater", {Literal("value2", .9f)}}, {KindOf<action::mesh::Delete>, {Set<&action::mesh::Delete::Mode>(DeleteMode::Faces)}}, {KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{.5f, .5f, .5f})}}, {KindOf<Join>}, {"boundary"}, {"linked"}, {KindOf<action::mesh::Delete>}}, {{1, 2, 0, "in"}, {2, 3, 0, "value1"}, {3, 4, 1}, {0, 4, 0}, {4, 6, 0}, {5, 6, 0}, {7, 8, 0}, {8, 9, 1}, {6, 9, 0}}, 9}, nullptr, [](GraphFixture &f, state::Entity mesh) { Text("domain", "face")(f, f.Node(mesh, 8u)); }, {}, 6u, 11u},
    // The primitive shades flat, which the patches would keep as creases, so a Shade node supplies the smooth normals.
    {"SmoothSurfaceIcoSphere", {primitive::IcoSphere{.Subdivisions = 1}, {{KindOf<Shade>}, {KindOf<SmoothSurface>}}, {{0, 1, 0}, {1, 2, 0}}, 2}, ExpectSmoothedTowardSphere, [](GraphFixture &f, state::Entity mesh) {
         const auto vertices = Mesh{f.Store(), f.Output(mesh)}.VertexCount();
         f.Do(UpdateOf<&SmoothSurface::Cuts>(f.Node(mesh, 2u), 4u));
         expect(Mesh{f.Store(), f.Output(mesh)}.VertexCount() > vertices);
     }},
    // The cylinder's caps are polygons, which Smooth Surface triangulates before it subdivides: 12 cap triangles and 8 side quads, each into the square of one more than the cuts.
    {"SmoothSurfacePolygons", {primitive::Cylinder{.Slices = 8u}, {{KindOf<SmoothSurface>}}, {{0, 1, 0}}, 1}, [](GraphFixture &f, const BuiltGraph &g) { expect(Mesh{f.Store(), f.Output(g.Mesh)}.HasClosedSurface()); }, [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&SmoothSurface::Cuts>(f.Node(mesh, 1u), 1u)); }, {}, 180u, 80u},
    // Flipping the faces on one side of a plane leaves the corners along the seam without opposites, so Smooth Surface rejects the mesh as not closed.
    // Flipping every face gives each corner an opposite again.
    {"SmoothSurfaceFlippedSide", {primitive::Cylinder{.Slices = 8u}, {{"planeside", {Literal("normal", vec3{1.f, 0.f, 1.f}), Literal("side", 0)}}, {KindOf<action::mesh::FlipNormals>}, {KindOf<SmoothSurface>}}, {{1, 2, 1}, {0, 2, 0}, {2, 3, 0}}, 3}, [](GraphFixture &f, const BuiltGraph &g) { expect(f.Fault(g.Nodes[3]) == NodeFault::NotClosed); }, [](GraphFixture &f, state::Entity mesh) {
         SetLiteral(f, f.Node(mesh, 1u), "point", vec3{-5.f, 0.f, -5.f});
         expect(!f.Fault(f.Node(mesh, 3u))); }, {}, {}, 180u},
    // The seam's corners without opposites make the Boolean reject the cylinder as not closed.
    // Flipping no face leaves the closed cylinder, which the cube holds, so the intersection is the cylinder.
    {"BooleanFlippedSide", {primitive::Cylinder{.Slices = 8u}, {{"planeside", {Literal("normal", vec3{1.f, 0.f, 1.f}), Literal("side", 0)}}, {KindOf<action::mesh::FlipNormals>}, {KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{2.f})}}, {KindOf<graph::Boolean>, {Set<&graph::Boolean::Operation>(BooleanOperation::Intersection)}}}, {{1, 2, 1}, {0, 2, 0}, {2, 4, 0}, {3, 4, 1}}, 4}, [](GraphFixture &f, const BuiltGraph &g) { expect(f.Fault(g.Nodes[4]) == NodeFault::NotClosed); }, [](GraphFixture &f, state::Entity mesh) {
         SetLiteral(f, f.Node(mesh, 1u), "point", vec3{5.f, 0.f, 5.f});
         expect(!f.Fault(f.Node(mesh, 4u)));
         expect(std::abs(OutputVolume(f, mesh) - 4.0 * std::numbers::sqrt2) < 1e-4); }},
    // A level set of one minus the distance from the origin is a sphere, which the level shrinks.
    {"LevelSetSphere", {primitive::Cube{}, {{"position"}, {"magnitude"}, {"multiply", {Literal("in2", -1.f)}}, {"add", {Literal("in2", .6f)}}, {KindOf<LevelSet>}}, {{1, 2, 0, "in"}, {2, 3, 0, "in1"}, {3, 4, 0, "in1"}, {4, 5, 0}}, 5}, [](GraphFixture &f, const BuiltGraph &g) { ExpectSphere(f, g.Mesh, .6); }, [](GraphFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&LevelSet::Level>(f.Node(mesh, 5u), .2f));
         ExpectSphere(f, mesh, .4); }},
    {"SubdivisionSurfaceCube", {primitive::Cube{}, {{KindOf<SubdivisionSurface>}}, {{0, 1, 0}}, 1, AddHeightHeat}, ExpectCatmullClarkCube, [](GraphFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&SubdivisionSurface::Levels>(f.Node(mesh, 1u), 2u));
         expect(Mesh{f.Store(), f.Output(mesh)}.VertexCount() == 98u);
     }},
    // A cylinder seen along its axis projects to its cap polygon, and seen across it to a rectangle.
    {"ProjectCylinder", {primitive::Cylinder{.Slices = 32}, {{KindOf<Project>, {Set<&Project::Normal>(vec3{0.f, 1.f, 0.f})}}}, {{0, 1, 0}}, 1}, [](GraphFixture &f, const BuiltGraph &g) { ExpectOneFaceOfArea(f, g.Mesh, 16.0 * std::sin(std::numbers::pi / 16.0)); }, [](GraphFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&Project::Normal>(f.Node(mesh, 1u), vec3{0.f, 0.f, 1.f}));
         ExpectOneFaceOfArea(f, mesh, 4.0); }},
    {"MinkowskiBoxes", {primitive::Cube{}, {{KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{.5f, .25f, .125f})}}, {KindOf<Minkowski>}}, {{1, 2, 1}, {0, 2, 0}}, 2}, [](GraphFixture &f, const BuiltGraph &g) { expect(std::abs(OutputVolume(f, g.Mesh) - 3.0 * 2.5 * 2.25) < 1e-3) << "sum volume" << OutputVolume(f, g.Mesh); }, [](GraphFixture &f, state::Entity mesh) {
         f.Do(UpdateOf<&Minkowski::Operation>(f.Node(mesh, 2u), MinkowskiOperation::Difference));
         expect(std::abs(OutputVolume(f, mesh) - 1.0 * 1.5 * 1.75) < 1e-3); }},
    {"SourceInstances", {primitive::Plane{}, {{KindOf<MeshToPoints>}, {KindOf<InstanceOnPoints>}}, {{0, 1, 0}, {1, 2, 0}, {0, 2, 1}}, 2}, ExpectSourceInstances, UnlinkInstance},
    // A Join takes every link into its one input socket, in order, and Unlink removes the link at its index.
    {"JoinLinks", {primitive::Cube{}, {{KindOf<primitive::Cube>}, {KindOf<primitive::Cube>}, {KindOf<Join>}}, {{0, 3, 0}, {1, 3, 0}, {2, 3, 0}}, 3}, nullptr, [](GraphFixture &f, state::Entity mesh) {
         const auto join = f.Node(mesh, 3u);
         const auto links = f.R.get<const NodeInputs>(join).Links;
         f.Do(action::nodes::Unlink{mesh, join, 0u, 1u});
         expect(f.R.get<const NodeInputs>(join).Links == std::vector{links[0], links[2]}); }, {}, 18u, 12u},
    // A Boolean subtracts every Mesh 2 from Mesh 1.
    // Two crossing 1 x 1 bars through the cube remove 2 + 2 - 1 of its volume of 8.
    {"BooleanCutters", {primitive::Cube{}, {{KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{.5f, .5f, 2.f})}}, {KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{2.f, .5f, .5f})}}, {KindOf<graph::Boolean>}}, {{0, 3, 0}, {1, 3, 1}, {2, 3, 1}}, 3}, [](GraphFixture &f, const BuiltGraph &g) { expect(std::abs(OutputVolume(f, g.Mesh) - 5.0) < 1e-3) << OutputVolume(f, g.Mesh); }, [](GraphFixture &f, state::Entity mesh) {
         f.Do(action::nodes::Unlink{mesh, f.Node(mesh, 3u), 1u, 0u});
         expect(std::abs(OutputVolume(f, mesh) - 6.0) < 1e-3) << OutputVolume(f, mesh); }},
    // The source clone takes lower blocks than the generated cube and the Join appends them after the cube's, so the joined block table is not in ascending order.
    // A comparison linked into a Delete still selects the cube's upper vertices in its higher block.
    {"JoinedSelection", {primitive::Cube{}, {{KindOf<primitive::Cube>, {Set<&primitive::Cube::HalfExtents>(vec3{1.f, 2.f, 1.f})}}, {KindOf<Join>}, {"position"}, {"extract", {Literal("index", 1)}}, {"ifgreater", {Literal("value2", 1.5f)}}, {KindOf<action::mesh::Delete>}}, {{1, 2, 0}, {0, 2, 0}, {3, 4, 0, "in"}, {4, 5, 0, "value1"}, {5, 6, 1}, {2, 6, 0}}, 6}, ExpectJoinedSelection, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 5u), "value2", .5f); }, CalledInput{4u, false}},
    // An edge lies in front of a plane when both its vertices do.
    // A plane past the torus's outer equator holds one equator edge, whose edge loop is the 32-edge equator, and Delete removes the 64 faces beside it.
    // Tilting the plane toward the tube holds one edge across the equator instead, whose loop is the 16-edge tube ring with 32 faces beside it.
    {"EdgeLoopDeletesRing", {primitive::Torus{}, {{"planeside", {Literal("point", vec3{0.f, 0.f, 1.48f}), Literal("normal", vec3{.0985f, 0.f, 1.f}), Literal("side", 0)}}, {"convertdomain", {Converts("vertex", "edge", "all")}}, {"edgeloop"}, {KindOf<action::mesh::Delete>, {Set<&action::mesh::Delete::Mode>(DeleteMode::Edges)}}}, {{1, 2, 0, "in"}, {2, 3, 0}, {3, 4, 1}, {0, 4, 0}}, 4}, nullptr, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 1u), "normal", vec3{0.f, .1989f, 1.f}); }, {}, 32u * 16u - 64u, 32u * 16u - 32u},
    // A selection the Domain combo sets to faces in place of Mark Sharp's edges reads the face field of the faces whose vertices all lie in front of the plane, so Mark Sharp smooths the top and +x faces.
    {"FaceFieldMarksSharp", {primitive::Cube{}, {{"planeside", FrontOfDiagonal}, {"convertdomain", {Converts("vertex", "face", "all")}}, {KindOf<MarkSharp>, {Set<&MarkSharp::Sharp>(false), SelectsOn(Element::Face)}}}, {{1, 2, 0, "in"}, {2, 3, 1}, {0, 3, 0}}, 3}, [](GraphFixture &f, const BuiltGraph &g) {
         const auto &meshes = f.Store();
         const Mesh marked{meshes, f.Output(g.Mesh)};
         std::vector<vec3> smooth;
         for (const auto face : marked.faces()) {
             if (!meshes.Arenas().FaceSharpness.Get({test::Address(marked, face), 1u})[0]) smooth.push_back(test::Normal(marked, face));
         }
         std::ranges::sort(smooth, {}, [](vec3 n) { return std::array{n.x, n.y, n.z}; });
         expect(smooth == std::vector<vec3>{{0.f, 0.f, 1.f}, {1.f, 0.f, 0.f}}); }, nullptr},
    {"FaceFieldFills", {primitive::Cube{}, {{"planeside", FrontOfDiagonal}, {"convertdomain", {Converts("vertex", "face", "all")}}, {KindOf<action::mesh::Fill>, {SelectsOn(Element::Face)}}}, {{1, 2, 0, "in"}, {2, 3, 1}, {0, 3, 0}}, 3}, nullptr, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 1u), "side", 1); }, {}, 5u},
    {"WidthsShrinkFatten", {primitive::Cube{}, {{"position"}, {"dotproduct", {Literal("in2", vec3{.05f, .05f, .05f})}}, {"add", {Literal("in2", .2f)}}, {KindOf<action::mesh::ShrinkFatten>}}, {{1, 2, 0, "in1"}, {2, 3, 0, "in1"}, {3, 4, ParameterSocket<&action::mesh::ShrinkFatten::Distance>}, {0, 4, 0}}, 4}, ExpectWidthsShrunk, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 3u), "in2", .3f); }, CalledInput{2u, 0.f}},
    // A position edit whose selection reads a stored-field op moves the selected vertices.
    {"BoundaryMoves", {primitive::Plane{.Subdivisions = {4u, 4u}}, {{"boundary", {Text("domain", "vertex")}}, {"position"}, {"multiply", {Literal("in2", .5f)}}, {KindOf<SetPosition>, {Set<&SetPosition::Offset>(false)}}}, {{1, 4, 2}, {2, 3, 0, "in1"}, {3, 4, 1}, {0, 4, 0}}, 4}, [](GraphFixture &f, const BuiltGraph &g) { expect(f.Shaped(g.Mesh) != f.WithSource(g.Mesh, [&](uint32_t id) { return Describe(f.Store(), id); })); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 3u), "in2", .25f); }, CalledInput{0u, false}},
    // A constant passed through a dot op sets the cuts from the value store, and an add of two literals evaluates them at one point.
    {"ConstantCuts", {primitive::Plane{}, {{"constant", {Literal("value", 2)}}, {"dot"}, {KindOf<action::mesh::Subdivide>}}, {{1, 2, 0, "in"}, {2, 3, Cuts}, {0, 3, 0}}, 3}, [](GraphFixture &f, const BuiltGraph &g) { ExpectCuts(f, g.Mesh, 2.f); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 1u), "value", 3); }},
    {"SumCuts", {primitive::Plane{}, {{"add", {Literal("in1", 1), Literal("in2", 1)}}, {KindOf<action::mesh::Subdivide>}}, {{1, 2, Cuts}, {0, 2, 0}}, 2}, [](GraphFixture &f, const BuiltGraph &g) { ExpectCuts(f, g.Mesh, 2.f); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 1u), "in2", 2); }},
    // Each torus quad between the outer two rings cuts three sides twice and its outer side three times, and a partition table fills it.
    // The torus quads are planar, so the subdivided surface encloses the same volume.
    {"SubdivideToLength", {primitive::Torus{.MajorSegments = 12, .MinorSegments = 6}, {{KindOf<action::mesh::Subdivide>, {Set<&action::mesh::Subdivide::MaxLength>(.24f)}}}, {{0, 1, 0}}, 1}, [](GraphFixture &f, const BuiltGraph &g) {
         const auto [faces, source] = f.WithSource(g.Mesh, [&](uint32_t id) { const Mesh mesh{f.Store(), id}; return std::pair{mesh.FaceCount(), test::EnclosedVolume(mesh).value_or(-1.0)}; });
         const Mesh output{f.Store(), f.Output(g.Mesh)};
         const auto volume = test::EnclosedVolume(output).value_or(-1.0);
         expect(output.HasClosedSurface() && output.FaceCount() > faces && std::abs(volume - source) < 1e-5 * source) << "faces" << output.FaceCount() << "volume" << volume << "vs" << source; }, [](GraphFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::MaxLength>(f.Node(mesh, 1u), .3f)); }},
    {"StoreFaceValue", {primitive::Cube{}, {{"constant", {Literal("value", 2.5f)}}, {KindOf<StoreNamedAttribute>, {Set<&StoreNamedAttribute::Domain>(AttributeDomain::Face), Named("heat")}}}, {{1, 2, 1}, {0, 2, 0}}, 2}, [](GraphFixture &f, const BuiltGraph &g) {
         const auto &meshes = f.Store();
         const auto heat = meshes.FindAttribute(f.Output(g.Mesh), "heat");
         expect(fatal(heat != InvalidOffset));
         expect(meshes.GetAttribute(heat).Domain == AttributeDomain::Face && meshes.GetAttribute(heat).Type == AttributeType::Float);
         const auto values = meshes.AttributeView<float>(heat);
         for (const auto face : Mesh{meshes, f.Output(g.Mesh)}.faces()) expect(values[*face] == 2.5f); }, nullptr},
    // The smooth before Set Position records its edit over every vertex unsubmitted, and Set Position's program reads the smoothed positions.
    {"OffsetPositions", {primitive::Cube{}, {{KindOf<action::mesh::SmoothVertices>}, {"position"}, {"multiply", {Literal("in2", 2.f)}}, {"add", {Literal("in2", vec3{0.f, 0.f, 1.f})}}, {KindOf<SetPosition>, {Set<&SetPosition::Offset>(false)}}}, {{0, 1, 0}, {1, 5, 0}, {2, 3, 0, "in1"}, {3, 4, 0, "in1"}, {4, 5, 1}}, 5}, ExpectSmoothedOffsets, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 4u), "in2", vec3{0.f, 0.f, 2.f}); }, CalledInput{4u, vec3{}}},
    // Shrink/Fatten reads the normals of the sheared positions before Set Position, and the three edits record unsubmitted.
    {"ShrunkOffsetPositions", {primitive::Cube{}, {{KindOf<graph::Shear>}, {"position"}, {"dotproduct", {Literal("in2", vec3{.05f, .05f, .05f})}}, {"add", {Literal("in2", .2f)}}, {KindOf<action::mesh::ShrinkFatten>}, {"position"}, {"multiply", {Literal("in2", 2.f)}}, {"add", {Literal("in2", vec3{0.f, 0.f, 1.f})}}, {KindOf<SetPosition>, {Set<&SetPosition::Offset>(false)}}}, {{0, 1, 0}, {1, 5, 0}, {2, 3, 0, "in1"}, {3, 4, 0, "in1"}, {4, 5, ParameterSocket<&action::mesh::ShrinkFatten::Distance>}, {5, 9, 0}, {6, 7, 0, "in1"}, {7, 8, 0, "in1"}, {8, 9, 1}}, 9}, nullptr, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 4u), "in2", .3f); }, CalledInput{4u, 0.f}},
    // Set Position offsets the sheared positions along their normals, and Subdivide follows the two unsubmitted edits.
    {"NormalOffsetsSubdivide", {primitive::Cube{}, {{KindOf<graph::Shear>}, {"normal"}, {"multiply", {Literal("in2", .25f)}}, {KindOf<SetPosition>}, {KindOf<action::mesh::Subdivide>}}, {{0, 1, 0}, {1, 4, 0}, {2, 3, 0, "in1"}, {3, 4, 1}, {4, 5, 0}}, 5}, nullptr, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 3u), "in2", .5f); }, CalledInput{3u, vec3{}}},
    // Each vertex of the cube's source rises by a tenth of its index among the live vertices, which are its handles.
    {"IndexOffsets", {primitive::Cube{}, {{"elementindex"}, {"convert", {OutputType(mx::Type::Float)}}, {"multiply", {Literal("in1", vec3{0.f, 0.f, .1f})}}, {KindOf<SetPosition>}}, {{1, 2, 0, "in"}, {2, 3, 0, "in2"}, {3, 4, 1}, {0, 4, 0}}, 4}, [](GraphFixture &f, const BuiltGraph &g) { expect(Near(f.Shaped(g.Mesh).Vertices, SourcePoints(f, g.Mesh, [](uint32_t v, vec3 p) { return p + vec3{0.f, 0.f, .1f * float(v)}; }))); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 3u), "in1", vec3{.1f, 0.f, 0.f}); }, CalledInput{2u, vec3{}}},
    {"SmoothFactors", {primitive::Cube{}, {{"position"}, {"extract", {Literal("index", 2)}}, {"multiply", {Literal("in2", .25f)}}, {"add", {Literal("in2", .5f)}}, {KindOf<action::mesh::SmoothVertices>}}, {{1, 2, 0, "in"}, {2, 3, 0, "in1"}, {3, 4, 0, "in1"}, {4, 5, ParameterSocket<&action::mesh::SmoothVertices::Factor>}, {0, 5, 0}}, 5}, ExpectSmoothFactors, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 4u), "in2", .25f); }, CalledInput{3u, 0.f}},
    // A repeated smooth whose factor reads positions reads it once from its input, as a stored attribute.
    {"RepeatedSmoothFactors", {primitive::Cube{}, {{"position"}, {"extract", {Literal("index", 2)}}, {"multiply", {Literal("in2", .25f)}}, {"add", {Literal("in2", .5f)}}, {KindOf<action::mesh::SmoothVertices>, {Set<&action::mesh::SmoothVertices::Repeat>(3u)}}}, {{1, 2, 0, "in"}, {2, 3, 0, "in1"}, {3, 4, 0, "in1"}, {4, 5, ParameterSocket<&action::mesh::SmoothVertices::Factor>}, {0, 5, 0}}, 5}, ExpectSmoothFactors, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 4u), "in2", .25f); }},
    // Noise read at each vertex's unset position offsets the sphere's vertices, which lie off the noise lattice.
    {"NoiseOffsets", {primitive::UVSphere{}, {{"noise3d", {OutputType(mx::Type::Vector3)}}, {KindOf<SetPosition>}}, {{1, 2, 1}, {0, 2, 0}}, 2}, [](GraphFixture &f, const BuiltGraph &g) { expect(f.Shaped(g.Mesh) != f.WithSource(g.Mesh, [&](uint32_t id) { return Describe(f.Store(), id); })); }, nullptr},
    {"RandomAmounts", {primitive::Cube{}, {{"position"}, {"magnitude"}, {"multiply", {Literal("in2", .1f)}}, {KindOf<action::mesh::Randomize>}}, {{1, 2, 0, "in"}, {2, 3, 0, "in1"}, {3, 4, ParameterSocket<&action::mesh::Randomize::Amount>}, {0, 4, 0}}, 4}, ExpectRandomAmounts, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 3u), "in2", .2f); }, CalledInput{2u, 0.f}},
    // Deleting faces reads its selection on faces, which reads a face attribute and fills its face masks, so Delete removes the faces past index 2.
    {"FaceIndexDeletes", {primitive::Cube{}, {{"elementindex"}, {KindOf<StoreNamedAttribute>, {Set<&StoreNamedAttribute::Domain>(AttributeDomain::Face), Named("index")}}, {"geompropvalue", {Text("geomprop", "index"), Literal("default", 0)}}, {"ifgreater", {Literal("value2", 2)}}, {KindOf<action::mesh::Delete>, {Set<&action::mesh::Delete::Mode>(DeleteMode::Faces)}}}, {{1, 2, 1}, {0, 2, 0}, {3, 4, 0, "value1"}, {4, 5, 1}, {2, 5, 0}}, 5}, nullptr, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 4u), "value2", 0); }, CalledInput{3u, false, AttributeDomain::Face}, 3u},
    // The stored-field ops' results cook bit for bit as the graph that stores each with a Store node and reads it back.
    {"ConvertedOffsets", ConvertedOffsets, [](GraphFixture &f, const BuiltGraph &g) {
         const auto stored = Build(f, StoredOffsets);
         const auto shape = f.Shaped(g.Mesh);
         expect(shape != f.WithSource(g.Mesh, [&](uint32_t id) { return Describe(f.Store(), id); }));
         expect(SameBits(f.Shaped(stored.Mesh), shape)); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 2u), "in2", 3.f); }},
    // A negative integer into Cuts clamps to its least value.
    {"NegativeCuts", {primitive::Plane{}, {{"add", {Literal("in1", -3), Literal("in2", 1)}}, {KindOf<action::mesh::Subdivide>}}, {{1, 2, Cuts}, {0, 2, 0}}, 2}, [](GraphFixture &f, const BuiltGraph &g) { ExpectCuts(f, g.Mesh, 1.f); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 1u), "in2", 5); }},
    // Bevel's vertex widths are the widths the ops compute from each vertex's position.
    {"WidthsBevelVertices", {primitive::Cube{}, {{"position"}, {"dotproduct", {Literal("in2", vec3{.05f, .05f, .05f})}}, {"add", {Literal("in2", .2f)}}, {KindOf<action::mesh::Bevel>, {Set<&action::mesh::Bevel::Vertices>(true)}}}, {{1, 2, 0, "in1"}, {2, 3, 0, "in1"}, {3, 4, ParameterSocket<&action::mesh::Bevel::Width>}, {0, 4, 0}}, 4}, [](GraphFixture &f, const BuiltGraph &g) { expect(f.Shaped(g.Mesh) == TopologyReference(f, g.Mesh, MeshTopologyOp::BevelVertices, [&](uint32_t clone) { return f.Store().AttributeRef(AddValues<float>(f.Store(), clone, "parameter", WidthAt)); })); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 3u), "in2", .3f); }},
    // Poke's offset evaluates on faces, so it reads each face's mean of the vertex widths.
    {"CenterPokes", {primitive::Cube{}, {{"position"}, {"dotproduct", {Literal("in2", vec3{.05f, .05f, .05f})}}, {"add", {Literal("in2", .2f)}}, {KindOf<action::mesh::Poke>}}, {{1, 2, 0, "in1"}, {2, 3, 0, "in1"}, {3, 4, ParameterSocket<&action::mesh::Poke::Offset>}, {0, 4, 0}}, 4}, ExpectCenterPokes, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 3u), "in2", .3f); }},
    // A face normal's z sets each face's offset.
    {"NormalPokes", {primitive::Cube{}, {{"normal"}, {"dotproduct", {Literal("in2", vec3{0.f, 0.f, .3f})}}, {KindOf<action::mesh::Poke>}}, {{1, 2, 0, "in1"}, {2, 3, ParameterSocket<&action::mesh::Poke::Offset>}, {0, 3, 0}}, 3}, [](GraphFixture &f, const BuiltGraph &g) { ExpectPokes(f, g.Mesh, [](const Mesh &m, he::FH face) { return .3f * test::Normal(m, face).z; }); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 2u), "in2", vec3{0.f, 0.f, .5f}); }},
    // Each face reads its own ordinal, which is its handle on the cube's source.
    {"OrdinalPokes", {primitive::Cube{}, {{"elementindex"}, {"convert", {OutputType(mx::Type::Float)}}, {"multiply", {Literal("in1", .1f)}}, {KindOf<action::mesh::Poke>}}, {{1, 2, 0, "in"}, {2, 3, 0, "in2"}, {3, 4, ParameterSocket<&action::mesh::Poke::Offset>}, {0, 4, 0}}, 4}, [](GraphFixture &f, const BuiltGraph &g) { ExpectPokes(f, g.Mesh, [](const Mesh &, he::FH face) { return .1f * float(*face); }); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 3u), "in1", .2f); }},
    // Two constants set the translation and the rotation of Transform Elements, beside its own scale.
    {"ConstantTransform", {primitive::Cube{}, {{"constant", {Literal("value", ConstantDelta.P)}}, {"constant", {Literal("value", std::bit_cast<vec4>(ConstantDelta.R))}}, {KindOf<TransformElements>, {Set<&TransformElements::Delta, &Transform::S>(ConstantDelta.S)}}}, {{1, 3, ParameterSocket<&TransformElements::Delta, &Transform::P>}, {2, 3, ParameterSocket<&TransformElements::Delta, &Transform::R>}, {0, 3, 0}}, 3}, [](GraphFixture &f, const BuiltGraph &g) { expect(Near(f.Shaped(g.Mesh).Vertices, SourcePoints(f, g.Mesh, [](uint32_t, vec3 p) { return ConstantDelta.R * (ConstantDelta.S * p) + ConstantDelta.P; }))); }, [](GraphFixture &f, state::Entity mesh) { SetLiteral(f, f.Node(mesh, 1u), "value", vec3{-1.f, 0.f, 0.f}); }},
    {"ProbeDefault", {primitive::Cube{}, {{"geompropvalue", {Text("geomprop", "probe"), Literal("default", vec3{0.f, 0.f, 1.f})}}, {KindOf<SetPosition>}}, {{1, 2, 1}, {0, 2, 0}}, 2}, ExpectProbeDefault, nullptr},
    // Only the top face's normal and none of the vertices' normals rise past 0.7.
    {"ConsumerDomains", {primitive::Cube{}, {{"normal"}, {"extract", {Literal("index", 2)}}, {"ifgreater", {Literal("value2", .7f)}}, {KindOf<action::mesh::Delete>, {Set<&action::mesh::Delete::Mode>(DeleteMode::Faces), SelectsOn(Element::Face)}}}, {{1, 2, 0, "in"}, {2, 3, 0, "value1"}, {3, 4, 1}, {0, 4, 0}}, 4}, nullptr, [](GraphFixture &f, state::Entity mesh) { SelectsOn(Element::Vertex)(f, f.Node(mesh, 4u)); }, {}, 5u, 6u},
    // The smooth cube's top edges are sharp and its +x face flat.
    {"CornerNormals", {primitive::Cube{}, {{KindOf<Shade>}, {"position"}, {"extract", {Literal("index", 2)}}, {"ifgreater", {Literal("value2", .5f)}}, {KindOf<MarkSharp>}, {"position"}, {"extract", {Literal("index", 0)}}, {"ifgreater", {Literal("value2", .5f)}}, {KindOf<MarkSharp>, {SelectsOn(Element::Face)}}, {"normal"}, {KindOf<StoreNamedAttribute>, {Named("shading"), Set<&StoreNamedAttribute::Domain>(AttributeDomain::Corner)}}}, {{0, 1, 0}, {1, 5, 0}, {2, 3, 0, "in"}, {3, 4, 0, "value1"}, {4, 5, 1}, {5, 9, 0}, {6, 7, 0, "in"}, {7, 8, 0, "value1"}, {8, 9, 1}, {9, 11, 0}, {10, 11, 1}}, 11}, ExpectCornerNormals, nullptr},
    {"PointColors", {ColoredPoints(), {{"geomcolor", {OutputType(mx::Type::Color4)}}, {KindOf<StoreNamedAttribute>, {Named("color")}}}, {{1, 2, 1}, {0, 2, 0}}, 2}, ExpectVertexColors, nullptr},
    // The positions in front of the plane x + z = 0 are two vertices, five edges and two faces of the cube.
    {"NaturalEdges", {primitive::Cube{}, {{"planeside", {Literal("normal", vec3{1.f, 0.f, 1.f}), Literal("side", 0)}}, {KindOf<action::mesh::Bevel>}}, {{1, 2, 1}, {0, 2, 0}}, 2}, ExpectNaturalEdges, nullptr},
}};
const auto BoxTextured = std::filesystem::path{MESHEDITOR_SOURCE_DIR} / "external/glTF-Sample-Assets/Models/BoxTextured/glTF/BoxTextured.gltf";
// A Join appends the primitive tables of a textured box and its subdivided copy, and a generated cube's faces read one primitive past them.
GraphCase JoinedMaterials(GraphFixture &f) {
    auto &r = f.R;
    f.Do(action::io::LoadGltf{BoxTextured});
    const auto mesh = SortedEntities(r.view<const GeometryGraph>()).front();
    const auto source = f.Node(mesh, 0u);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    const auto cube = f.Append(mesh, KindOf<primitive::Cube>);
    const auto join = f.Append(mesh, KindOf<Join>);
    f.Do(Link{mesh, source, f.Out(source), subdivide, 0u});
    for (const auto input : {source, subdivide, cube}) f.Do(Link{mesh, input, f.Out(input), join, 0u});
    f.Do(SetDisplay{mesh, join});
    // The joined record's face count of each primitive, and its primitives' materials.
    const auto joined = [](GraphFixture &f, state::Entity mesh) {
        const auto &record = f.Store().Get(f.Output(mesh));
        const auto &a = f.Store().Arenas();
        std::map<uint32_t, uint32_t> faces;
        for (const auto face : Mesh{f.Store(), f.Output(mesh)}.faces()) ++faces[a.FacePrimitives.Get(record.FacePrimitives, *face)];
        return std::pair{faces, a.PrimitiveMaterials.Get(record.PrimitiveMaterials) | std::ranges::to<std::vector>()};
    };
    return {
        mesh,
        [mesh, joined](GraphFixture &f) {
            const auto material = f.WithSource(mesh, [&](uint32_t id) { return f.Store().Arenas().PrimitiveMaterials.Get(f.Store().Get(id).PrimitiveMaterials)[0]; });
            expect(material != 0u);
            expect(joined(f, mesh) == std::pair{std::map<uint32_t, uint32_t>{{0u, 12u}, {1u, 48u}, {2u, 6u}}, std::vector{material, material, 0u}});
        },
        [joined](GraphFixture &f, state::Entity mesh) {
            f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 1u), 2u));
            expect(joined(f, mesh).first == std::map<uint32_t, uint32_t>{{0u, 12u}, {1u, 108u}, {2u, 6u}});
        },
    };
}
constexpr std::array<GraphRow, 15> BuiltGraphs{{
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
    {"InstancesOfPrototype", InstancesOfPrototype},
    {"NodeEditing", NodeEditing},
    {"UnsubdivideSelection", UnsubdivideSelection},
    {"JoinedMaterials", JoinedMaterials},
}};

// The graph with the called link's op output stored on the input's domain by a Store node on the consumer's geometry and read back by a geompropvalue op.
GraphDescription Stored(GraphDescription graph, const CalledInput &called) {
    const auto [from, to, socket, input] = graph.Links[called.Link];
    const auto store = uint32_t(graph.Nodes.size() + 1u), read = store + 1u;
    graph.Nodes.push_back({KindOf<StoreNamedAttribute>, {Named("called"), Set<&StoreNamedAttribute::Domain>(called.Domain)}});
    graph.Nodes.push_back({"geompropvalue", {Text("geomprop", "called"), Literal("default", called.Default)}});
    auto &geometry = *std::ranges::find_if(graph.Links, [&](const auto &link) { return link.To == to && link.Socket == 0u && link.Input.empty(); });
    const auto source = std::exchange(geometry.From, store);
    graph.Links[called.Link] = {from, store, 1u};
    graph.Links.push_back({source, store, 0u});
    graph.Links.push_back({read, to, socket, input});
    return graph;
}
// Cooks every node after the graph's Source node again in one cook, which gathers consecutive position edits.
void Recook(GraphFixture &f, const GraphDescription &graph, const BuiltGraph &built) {
    const auto first = built.Nodes[std::ranges::find_if(graph.Links, [](const auto &link) { return link.From == 0u && link.Socket == 0u && link.Input.empty(); })->To];
    f.Do(action::nodes::SetBypass{first, true});
    f.Do(action::nodes::SetBypass{first, false});
}
// A consumer input that calls its op program at each element cooks bit for bit as the graph with the program's output stored and read back.
// It creates no temporary attribute, so a graph whose only temporaries could be its op outputs leaves no released attribute slot, which holds no record.
void CalledEqualsStored(GraphFixture &f, const GraphData &row) {
    const auto called = Build(f, row.Description);
    Recook(f, row.Description, called);
    const bool fields = std::ranges::any_of(row.Description.Nodes, [](const auto &node) { return std::holds_alternative<std::string_view>(node.Kind) && DefinitionOf(std::get<std::string_view>(node.Kind))->Stored; });
    uint32_t released = 0u;
    f.Store().ForEachNamedAttribute(InvalidOffset, [&](uint32_t, const auto &) { ++released; });
    expect(fields || released == 0u);
    const auto shape = f.Shaped(called.Mesh);
    const auto description = Stored(row.Description, *row.Called);
    const auto stored = Build(f, description);
    Recook(f, description, stored);
    expect(SameBits(f.Shaped(stored.Mesh), shape));
}

// Adds a subgraph node of the subgraph to the other mesh, whose ports read the other mesh's Source node where the subgraph's read its own mesh's, and displays its port.
state::Entity AddSubgraphNodeOf(GraphFixture &f, state::Entity use, state::Entity other, std::optional<uint16_t> port) {
    auto &r = f.R;
    const auto source = f.Node(GraphOf(r, use), 0u), other_source = f.Node(other, 0u);
    f.Do(AddNode{other, action::nodes::UserSubgraphNode{UserSubgraph(r.get<const SubgraphNode>(use).SubgraphId)}});
    const auto node = ActiveNodeOf(r, other);
    for (const auto &own : r.get<const SubgraphNode>(use).Ports) {
        if (own.Source.FromNode == source) f.Do(Link{other, other_source, f.Out(other_source), node, own.PortId});
    }
    f.Do(SetDisplay{other, node, port});
    return node;
}
// The graph's nodes other than its Source node as one shared subgraph, whose occurrence cooks bit for bit as the nodes inline.
// Two subgraph nodes of it in a chain on one mesh cook as the graph twice inline, also after a literal edit inside the subgraph, and a second mesh's subgraph node of it cooks as the nodes inline there.
void SharedSubgraph(GraphFixture &f, const GraphData &row) {
    auto &r = f.R;
    const auto float_literal = [&](state::Entity node) -> std::optional<uint16_t> {
        const auto *op = r.try_get<const graph::Op>(node);
        if (!op) return {};
        const auto literal = std::ranges::find(op->Literals, mx::Type::Float, &std::pair<const std::string, mx::Type>::second);
        return literal == op->Literals.end() ? std::nullopt : std::optional{f.OpSocket(node, literal->first)};
    };
    // Raises the literal by a quarter and returns its prior word.
    const auto raise = [&](state::Entity node, uint16_t input) {
        const auto word = ParameterDefault(r, node, input)->Words[0];
        f.Do(OpInputOf(node, SocketOf(r, node, input).Name, action::nodes::OpLiteral{mx::Type::Float, {std::bit_cast<uint32_t>(std::bit_cast<float>(word) + .25f)}}));
        return word;
    };
    const auto displayed_index = row.Description.Display;
    // The graph twice inline: copies of its nodes read the first result where the originals read the Source node.
    const auto chain = Build(f, row.Description);
    const auto inline_once = f.Shaped(chain.Mesh);
    const auto displayed = chain.Nodes[displayed_index];
    const SocketLink result{displayed, *ResultSocket(r, displayed)};
    const bool chains = SocketOf(r, result.FromNode, result.FromSocket).Types == TypeBit(DataType::Geometry);
    Shape inline_twice, inline_raised;
    std::optional<size_t> literal;
    if (chains) {
        const std::vector originals(chain.Nodes.begin() + 1, chain.Nodes.end());
        f.Do(DuplicateNodes{chain.Mesh, originals});
        const auto nodes = NodesOf(r, chain.Mesh);
        const auto copy_of = [&](size_t i) { return nodes[nodes.size() - originals.size() + i]; };
        for (size_t i = 0; i < originals.size(); ++i) {
            const auto links = r.get<const NodeInputs>(originals[i]).Links;
            for (uint16_t slot = 0; slot < links.size(); ++slot) {
                if (links[slot].FromNode == chain.Nodes[0]) f.Do(Link{chain.Mesh, result.FromNode, result.FromSocket, copy_of(i), SlotSocketOf(r, originals[i], slot)});
            }
        }
        f.Do(SetDisplay{chain.Mesh, copy_of(displayed_index - 1u)});
        inline_twice = f.Shaped(chain.Mesh);
        for (size_t i = 0; i < originals.size() && !literal; ++i) {
            if (const auto input = float_literal(originals[i])) {
                literal = i;
                raise(originals[i], *input);
                raise(copy_of(i), *input);
                inline_raised = f.Shaped(chain.Mesh);
            }
        }
    }
    const auto a = Build(f, row.Description);
    const auto mesh = a.Mesh;
    f.Do(action::nodes::Collapse{mesh, {a.Nodes.begin() + 1, a.Nodes.end()}});
    expect(SameBits(f.Shaped(mesh), inline_once));
    if (!chains) return;
    const auto first = ActiveNodeOf(r, mesh);
    const auto subgraph = UserSubgraph(r.get<const SubgraphNode>(first).SubgraphId);
    const SocketLink shown{a.Nodes[displayed_index], result.FromSocket};
    const auto shows = [&](const SubgraphPort &port) { return port.Direction == SocketDirection::Out && port.Source() == shown; };
    if (std::ranges::none_of(r.get<const SubgraphInterface>(subgraph).Ports, shows)) f.Do(action::nodes::ExposeSocket{subgraph, shown.FromNode, shown.FromSocket, {}});
    const auto port = std::ranges::find_if(r.get<const SubgraphInterface>(subgraph).Ports, shows)->Id;
    // A second mesh's subgraph node of the subgraph reads that mesh's Source node at the same ports.
    const auto other = Build(f, {row.Description.Shape, {}, {}, 0u, row.Description.Prepare});
    AddSubgraphNodeOf(f, first, other.Mesh, port);
    expect(SameBits(f.Shaped(other.Mesh), inline_once));
    // Each input port of the second subgraph node reads the first's result where the first's reads the Source node.
    f.Do(AddNode{mesh, action::nodes::UserSubgraphNode{subgraph}});
    const auto second = ActiveNodeOf(r, mesh);
    for (const auto &own : r.get<const SubgraphNode>(first).Ports) {
        if (own.Source.FromNode == a.Nodes[0]) f.Do(Link{mesh, first, port, second, own.PortId});
    }
    f.Do(SetDisplay{mesh, second, port});
    expect(SameBits(f.Shaped(mesh), inline_twice));
    // A literal edit of the subgraph's op reaches both occurrences.
    if (literal) {
        const auto node = NodesOf(r, subgraph)[*literal];
        raise(node, *float_literal(node));
        expect(SameBits(f.Shaped(mesh), inline_raised));
    }
}

// How a timed graph's scenario builds it: on one mesh, as a subgraph two meshes instance, or on a mesh another mesh's Join reads through an Object Info node, which a subgraph can hold.
enum class TimedForm : uint8_t { Inline,
                                 Shared,
                                 Operand,
                                 DefinedOperand };
struct TimedGraph {
    std::string_view Name;
    GraphDescription Description;
    TimedForm Form{TimedForm::Inline};
};
// Set Position moves each vertex along a direction by the time.
const GraphDescription TimeMoves{primitive::Cube{}, {{"position"}, {"time"}, {"multiply", {Literal("in1", vec3{0.f, .5f, 1.f})}}, {"add"}, {KindOf<SetPosition>, {Set<&SetPosition::Offset>(false)}}}, {{1, 4, 0, "in1"}, {2, 3, 0, "in2"}, {3, 4, 0, "in2"}, {4, 5, 1}, {0, 5, 0}}, 5};
// Set Position sets each vertex to the mean, over the faces around it, of the time-moved positions' face means.
const GraphDescription TimeConverts{primitive::Cube{}, {{"position"}, {"time"}, {"multiply", {Literal("in1", vec3{0.f, .5f, 1.f})}}, {"add"}, {"convertdomain", {Converts("vertex", "face", "mean")}}, {"convertdomain", {Converts("face", "vertex", "mean")}}, {KindOf<SetPosition>, {Set<&SetPosition::Offset>(false)}}}, {{1, 4, 0, "in1"}, {2, 3, 0, "in2"}, {3, 4, 0, "in2"}, {4, 5, 0, "in"}, {5, 6, 0, "in"}, {6, 7, 1}, {0, 7, 0}}, 7};
const std::array<TimedGraph, 11> TimedGraphs{{
    {"TimeSetsPositions", TimeMoves},
    // A Join with one input passes Set Position's edit through, which repairs the render records itself.
    {"TimeSetsPositionsJoined", {primitive::Cube{}, {{"position"}, {"time"}, {"multiply", {Literal("in1", vec3{0.f, .5f, 1.f})}}, {"add"}, {KindOf<SetPosition>, {Set<&SetPosition::Offset>(false)}}, {KindOf<Join>}}, {{1, 4, 0, "in1"}, {2, 3, 0, "in2"}, {3, 4, 0, "in2"}, {4, 5, 1}, {0, 5, 0}, {5, 6, 0}}, 6}},
    // The vertices right of a threshold that rises with the frame are deleted.
    {"FrameSelects", {primitive::Plane{.Subdivisions = {4u, 4u}}, {{"position"}, {"extract", {Literal("index", 0)}}, {"frame"}, {"multiply", {Literal("in2", .25f)}}, {"subtract", {Literal("in2", 1.f)}}, {"ifgreater"}, {KindOf<action::mesh::Delete>}}, {{1, 2, 0, "in"}, {2, 6, 0, "value1"}, {3, 4, 0, "in1"}, {4, 5, 0, "in1"}, {5, 6, 0, "value2"}, {6, 7, 1}, {0, 7, 0}}, 7}},
    {"TimePokesFaces", {primitive::Cube{}, {{"normal"}, {"extract", {Literal("index", 0)}}, {"time"}, {"multiply"}, {KindOf<action::mesh::Poke>}}, {{1, 2, 0, "in"}, {2, 4, 0, "in1"}, {3, 4, 0, "in2"}, {4, 5, ParameterSocket<&action::mesh::Poke::Offset>}, {0, 5, 0}}, 5}},
    {"TimeMidChain", {primitive::UVSphere{.Slices = 16, .Stacks = 8}, {{KindOf<action::mesh::SmoothVertices>}, {KindOf<action::mesh::Subdivide>}, {KindOf<ToSphere>, {Set<&ToSphere::Factor>(.5f)}}, {"position"}, {"time"}, {"multiply", {Literal("in1", vec3{0.f, 0.f, .5f})}}, {"add"}, {KindOf<SetPosition>, {Set<&SetPosition::Offset>(false)}}, {KindOf<action::mesh::Subdivide>}, {KindOf<Shade>, {Set<&Shade::Mode>(ShadeMode::Flat)}}}, {{0, 1, 0}, {1, 2, 0}, {2, 3, 0}, {3, 8, 0}, {4, 7, 0, "in1"}, {5, 6, 0, "in2"}, {6, 7, 0, "in2"}, {7, 8, 1}, {8, 9, 0}, {9, 10, 0}}, 10}},
    {"TimeSharedSubgraph", TimeMoves, TimedForm::Shared},
    {"TimeOperand", TimeMoves, TimedForm::Operand},
    {"TimeDefinedOperand", TimeMoves, TimedForm::DefinedOperand},
    {"TimeConvertsDomains", TimeConverts},
    {"TimeConvertsDomainsShared", TimeConverts, TimedForm::Shared},
    {"Untimed", DoubledPositions},
}};
// Builds the graph in the form, and returns the meshes whose output the graph decides and the built nodes, starting with the first mesh's Source node.
std::pair<std::vector<state::Entity>, std::vector<state::Entity>> BuildTimed(GraphFixture &f, const GraphDescription &graph, TimedForm form) {
    auto &r = f.R;
    const auto built = Build(f, graph);
    if (form == TimedForm::Operand || form == TimedForm::DefinedOperand) {
        const auto object = f.Object(built.Mesh);
        const auto reader = Build(f, {primitive::Cube{}, {{KindOf<graph::ObjectInfo>, {Set<&graph::ObjectInfo::Object>(object)}}, {KindOf<Join>}}, {{0, 2, 0}, {1, 2, 0}}, 2});
        if (form == TimedForm::DefinedOperand) f.Do(action::nodes::Collapse{reader.Mesh, {reader.Nodes[1], reader.Nodes[2]}});
        return {{built.Mesh, reader.Mesh}, built.Nodes};
    }
    if (form == TimedForm::Inline) return {{built.Mesh}, built.Nodes};
    f.Do(action::nodes::Collapse{built.Mesh, {built.Nodes.begin() + 1, built.Nodes.end()}});
    const auto other = Build(f, {graph.Shape, {}, {}, 0u}).Mesh;
    AddSubgraphNodeOf(f, ActiveNodeOf(r, built.Mesh), other, r.get<const GeometryGraph>(built.Mesh).DisplayPort);
    return {{built.Mesh, other}, built.Nodes};
}

// The cook at each frame equals the cook of the graph with its frame and time ops replaced by constants of the frame and its time, at every frame a user reaches.
// Playing adds the two history nodes of its start and its stop at any length, and its frames after the second add no history or trie storage.
void Timed(GraphFixture &f, const TimedGraph &row) {
    using action::timeline::OffsetFrame, action::timeline::SetFrame, action::timeline::TogglePlay;
    auto &r = f.R;
    constexpr std::array Frames{1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto constant = row.Description;
    std::vector<std::pair<uint32_t, bool>> replaced; // Built node index, and whether it reads the time
    for (uint32_t i = 0; i < constant.Nodes.size(); ++i) {
        const auto *op = std::get_if<std::string_view>(&constant.Nodes[i].Kind);
        if (!op || (*op != "frame" && *op != "time")) continue;
        replaced.emplace_back(i + 1u, *op == "time");
        constant.Nodes[i] = {"constant", {Literal("value", 0.f)}};
    }
    // The outputs at each frame, from the graph with constants.
    std::map<int, std::vector<Shape>> expected;
    {
        const auto [meshes, nodes] = BuildTimed(f, constant, row.Form);
        for (const auto frame : Frames) {
            for (const auto [node, time] : replaced) SetLiteral(f, nodes[node], "value", time ? animation::FrameSeconds(r, frame) : float(frame));
            for (const auto mesh : meshes) expected[frame].push_back(f.Shaped(mesh));
        }
    }
    const auto [meshes, nodes] = BuildTimed(f, row.Description, row.Form);
    const auto frame_now = [&] { return r.get<const TimelinePlayback>(state::Document).CurrentFrame; };
    const auto expect_outputs = [&] {
        const auto &outputs = expected.at(frame_now());
        for (size_t i = 0; i < meshes.size(); ++i) expect(SameBits(f.Shaped(meshes[i]), outputs[i]));
    };
    expect_outputs();
    f.Do(SetFrame{4});
    expect_outputs();
    f.Do(OffsetFrame{1});
    expect_outputs();
    f.Do(OffsetFrame{-2});
    expect_outputs();
    // Undo and redo restore each frame's outputs.
    f.Undo();
    expect(frame_now() == 5);
    expect_outputs();
    f.Redo();
    expect(frame_now() == 3);
    expect_outputs();
    // Plays `ticks` frames from the current one, then stops, and returns the history nodes it added.
    // The history's owned bytes and the trie node bytes stay at or below the most either of the first two frames leaves, since a frame owns no copy of a page it leaves unchanged.
    auto &frame_state = r.Resources.get<FrameState>();
    const auto play = [&](uint32_t ticks) {
        const auto before = f.P->History.Nodes.size();
        f.Do(TogglePlay{frame_now()});
        frame_state.FixedFrameStep = true;
        std::pair<uint64_t, uint64_t> held{};
        for (uint32_t tick = 0; tick < ticks; ++tick) {
            f.P->Frame();
            expect_outputs();
            const std::pair sizes{f.P->History.Stats().OwnedBytes, store::SharedNodeBytes()};
            if (tick > 1u) expect(sizes.first <= held.first && sizes.second <= held.second);
            else held = {std::max(held.first, sizes.first), std::max(held.second, sizes.second)};
        }
        frame_state.FixedFrameStep = false;
        f.Do(TogglePlay{frame_now()});
        expect_outputs();
        return f.P->History.Nodes.size() - before;
    };
    play(3u);
    expect(frame_now() == 6);
    expect(play(1u) == 2u);
    expect(play(4u) == 2u);
    expect(frame_now() == 11);
    // A frame change while the mesh shows another node leaves no state of the earlier frame for the node shown again.
    if (row.Form == TimedForm::Inline) {
        const auto display = nodes[row.Description.Display];
        f.Do(action::nodes::SetBypass{display, true});
        f.Do(action::nodes::SetBypass{display, false});
        f.Do(SetDisplay{meshes[0], nodes[0]});
        f.Do(OffsetFrame{1});
        f.Do(SetDisplay{meshes[0], display});
        expect_outputs();
    }
}

// A static hull collider fit to a graph's cooked mesh, under a falling cube, bakes the same reached by playing frame by frame as reached by one jump, and a jump back reads the baked frames.
// Playing bakes one more frame each tick and keeps the frames baked before it.
// A collider whose mesh reads the time bakes each frame with that frame's geometry.
void TimeCollider(GraphFixture &f, const GraphDescription &graph) {
    using action::timeline::SetFrame, action::timeline::TogglePlay;
    auto &r = f.R;
    const auto mesh = Build(f, graph).Mesh;
    f.Do(action::physics::SetMotionType{action::physics::SetMotionType::Type::Static, f.Object(mesh)});
    expect(std::holds_alternative<physics::ConvexHull>(r.get<const ColliderShape>(f.Object(mesh)).Shape));
    const auto body = f.AddCube();
    f.Do(UpdateOf<&Transform::P>(body, vec3{0.f, 3.5f, 0.f}));
    f.Do(action::physics::SetMotionType{action::physics::SetMotionType::Type::Dynamic});
    const auto timed = !r.view<const TimeReads>().empty();
    const auto poses = [&] { return r.get<const BodyPoseCache>(body).Frames; };
    const auto same = [](const std::vector<CachedPose> &a, const std::vector<CachedPose> &b) {
        return std::ranges::equal(a, b, [](const CachedPose &p, const CachedPose &q) { return p.P == q.P && p.R == q.R; });
    };
    constexpr int Last{24};
    f.Do(SetFrame{Last});
    expect(physics::BakedThrough(r) == uint32_t(Last));
    const auto jumped = poses();
    // The rising collider lifts the cube after it lands.
    if (timed) expect(jumped.back().P.y > std::ranges::min(jumped | std::views::transform([](const CachedPose &p) { return p.P.y; })));
    f.Do(SetFrame{1});
    physics::InvalidateCache(r);
    f.Do(TogglePlay{1});
    auto &frame_state = r.Resources.get<FrameState>();
    frame_state.FixedFrameStep = true;
    for (int frame = 2; frame <= Last; ++frame) {
        const auto before = poses();
        f.P->Frame();
        expect(physics::BakedThrough(r) == uint32_t(frame));
        auto after = poses();
        after.resize(before.size());
        expect(same(after, before));
    }
    frame_state.FixedFrameStep = false;
    f.Do(TogglePlay{Last});
    expect(same(poses(), jumped));
    f.Do(SetFrame{5});
    expect(physics::BakedThrough(r) == uint32_t(Last));
}

constexpr std::array<std::pair<std::string_view, AttributeDomain>, 4> DomainNames{{{"vertex", AttributeDomain::Vertex}, {"edge", AttributeDomain::Edge}, {"face", AttributeDomain::Face}, {"corner", AttributeDomain::Corner}}};
// Each field type Convert Domain takes, with the modes it admits.
struct ConvertedType {
    mx::Type Type;
    std::vector<std::pair<std::string_view, DomainReduction>> Modes;
};
// The modes of a numeric type.
const std::vector<std::pair<std::string_view, DomainReduction>> Arithmetic{{"mean", DomainReduction::Mean}, {"min", DomainReduction::Min}, {"max", DomainReduction::Max}};
const std::array<ConvertedType, 8> ConvertedTypes{{
    {mx::Type::Boolean, {{"any", DomainReduction::Any}, {"all", DomainReduction::All}}},
    {mx::Type::Integer, Arithmetic},
    {mx::Type::Float, Arithmetic},
    {mx::Type::Vector2, Arithmetic},
    {mx::Type::Vector3, Arithmetic},
    {mx::Type::Vector4, Arithmetic},
    {mx::Type::Color3, Arithmetic},
    {mx::Type::Color4, Arithmetic},
}};
// The live elements of the domain, by ascending handle, where corners are the corners of faces.
std::vector<uint32_t> Elements(const Mesh &mesh, AttributeDomain domain) {
    std::vector<uint32_t> handles;
    if (domain == AttributeDomain::Vertex) {
        for (const auto v : mesh.vertices()) handles.push_back(*v);
    } else if (domain == AttributeDomain::Edge) {
        for (const auto e : mesh.edges()) handles.push_back(*e);
    } else {
        for (const auto face : mesh.faces()) {
            if (domain == AttributeDomain::Face) handles.push_back(*face);
            else
                for (const auto h : test::FaceHalfedges(mesh, face)) handles.push_back(*h);
        }
    }
    std::ranges::sort(handles);
    return handles;
}
// The elements of domain `from` related to element t of domain `to`, by the adjacency each conversion reduces over.
std::vector<uint32_t> Related(const Mesh &mesh, AttributeDomain from, AttributeDomain to, uint32_t t) {
    using D = AttributeDomain;
    const auto &c = mesh.GetConnectivity();
    if (from == to) return {t};
    std::vector<uint32_t> related;
    if (to == D::Vertex) {
        if (from == D::Edge) {
            for (const auto e : mesh.edges()) {
                const auto [a, b] = test::EdgeVertices(mesh, e);
                if (*a == t || *b == t) related.push_back(*e);
            }
            return related;
        }
        for (const auto face : mesh.faces()) {
            for (const auto h : test::FaceHalfedges(mesh, face))
                if (*test::ToVertex(mesh, h) == t) related.push_back(from == D::Face ? *face : *h);
        }
    } else if (to == D::Edge) {
        const auto first = test::Halfedge(mesh, Mesh::EH{t});
        if (from == D::Vertex) {
            const auto [a, b] = test::EdgeVertices(mesh, Mesh::EH{t});
            return {*a, *b};
        }
        for (const auto h : {first, test::Opposite(mesh, first)}) {
            if (!h || !test::Face(mesh, h)) continue;
            if (from == D::Face) related.push_back(*test::Face(mesh, h));
            else related.insert(related.end(), {*c.Previous(h), *h});
        }
    } else if (to == D::Face) {
        for (const auto h : test::FaceHalfedges(mesh, Mesh::FH{t})) related.push_back(from == D::Vertex ? *test::ToVertex(mesh, h) : from == D::Edge ? *c.Edge(h) :
                                                                                                                                                       *h);
    } else if (from == D::Vertex) {
        related.push_back(*test::ToVertex(mesh, Mesh::HH{t}));
    } else if (const auto face = test::Face(mesh, Mesh::HH{t})) {
        if (from == D::Face) related.push_back(*face);
        else related.insert(related.end(), {*c.Edge(Mesh::HH{t}), *c.Edge(c.Next(Mesh::HH{t}))});
    }
    return related;
}
// The input value of a Convert Domain test at element `handle` of the domain with index `domain`, whose floats are integers so that sums are exact in any order.
AttributeValue InputValue(AttributeType type, uint32_t domain, uint32_t handle) {
    if (type == AttributeType::Bool) return {uint32_t((handle * 7u + domain * 3u + 1u) % 5u < 2u), 0u, 0u, 0u};
    if (type == AttributeType::Int) return {std::bit_cast<uint32_t>(int32_t((handle * 13u + domain * 7u + 5u) % 17u) - 8), 0u, 0u, 0u};
    AttributeValue words{};
    for (uint32_t c = 0u; c < AttributeComponents(type); ++c) words[c] = std::bit_cast<uint32_t>(float(int32_t((handle * 11u + c * 5u + domain * 3u + 3u) % 19u) - 9));
    return words;
}
// The reduction of the values, and zero for none.
AttributeValue Reduce(AttributeType type, DomainReduction reduction, std::span<const AttributeValue> values) {
    if (values.empty()) return {};
    if (type == AttributeType::Bool) {
        bool reduced = values.front()[0] != 0u;
        for (const auto &value : values) reduced = reduction == DomainReduction::All ? reduced && value[0] != 0u : reduced || value[0] != 0u;
        return {uint32_t(reduced), 0u, 0u, 0u};
    }
    if (type == AttributeType::Int) {
        const auto integer = [](const AttributeValue &value) { return std::bit_cast<int32_t>(value[0]); };
        if (reduction != DomainReduction::Mean) {
            const auto [least, greatest] = std::ranges::minmax(values | std::views::transform(integer));
            return {std::bit_cast<uint32_t>(reduction == DomainReduction::Min ? least : greatest), 0u, 0u, 0u};
        }
        int64_t sum = 0;
        for (const auto &value : values) sum += integer(value);
        const auto count = int64_t(values.size()), quotient = sum / count, remainder = sum % count;
        return {std::bit_cast<uint32_t>(int32_t(2 * std::abs(remainder) >= count ? quotient + (sum < 0 ? -1 : 1) : quotient)), 0u, 0u, 0u};
    }
    AttributeValue reduced{};
    for (uint32_t c = 0u; c < AttributeComponents(type); ++c) {
        float value = std::bit_cast<float>(values.front()[c]);
        for (const auto &next : values | std::views::drop(1)) {
            const auto component = std::bit_cast<float>(next[c]);
            value = reduction == DomainReduction::Mean ? value + component : reduction == DomainReduction::Min ? std::min(value, component) :
                                                                                                                 std::max(value, component);
        }
        reduced[c] = std::bit_cast<uint32_t>(reduction == DomainReduction::Mean ? value * (1.f / float(values.size())) : value);
    }
    return reduced;
}
// The attribute's words at each live element of its domain.
std::vector<AttributeValue> StoredWords(GraphFixture &f, state::Entity mesh, std::string_view name) {
    const auto &meshes = f.Store();
    const auto id = f.Output(mesh);
    const auto attribute = meshes.FindAttribute(id, name);
    if (attribute == InvalidOffset) return {};
    return Elements(Mesh{meshes, id}, meshes.GetAttribute(attribute).Domain) | std::views::transform([&](uint32_t handle) { return test::AttributeWords(meshes, attribute, handle); }) | std::ranges::to<std::vector>();
}
// The named attribute of the mesh's output is on the domain and holds `expected(output, element)` at each live element of the domain.
void ExpectStored(GraphFixture &f, state::Entity mesh, std::string_view name, AttributeDomain domain, auto &&expected) {
    const auto &meshes = f.Store();
    const auto id = f.Output(mesh);
    const auto attribute = meshes.FindAttribute(id, name);
    expect(fatal(attribute != InvalidOffset)) << name;
    expect(meshes.GetAttribute(attribute).Domain == domain) << name;
    const Mesh output{meshes, id};
    for (const auto element : Elements(output, domain)) expect(test::AttributeWords(meshes, attribute, element) == expected(output, element)) << name << element;
}
// Applies the actions, in order, as one frame's gesture.
void Gesture(GraphFixture &f, auto &&add) {
    action::Begin(action::Phase::Record);
    add();
    f.P->Frame();
    f.Audit();
}

// For each mode, pair of domains and type Convert Domain admits, its output at each element equals the reduction of the input at the related elements, on meshes with boundary edges, irregular valence, triangles, quads and n-gons.
void DomainConversions(GraphFixture &f) {
    const std::array<PrimitiveParameters, 5> shapes{primitive::Plane{.Subdivisions = {3u, 2u}}, primitive::Circle{.Segments = 5}, primitive::IcoSphere{.Subdivisions = 1}, primitive::Cylinder{.Slices = 5}, primitive::UVSphere{.Slices = 6, .Stacks = 4}};
    for (const auto &shape : shapes) {
        for (const auto &[type, modes] : ConvertedTypes) {
            const auto storage = *ops::StorageOf(type);
            const auto mesh = f.AddPrepared(shape, [&](GraphFixture &f, uint32_t source) {
                auto &meshes = f.Store();
                for (uint32_t d = 0u; d < DomainNames.size(); ++d) {
                    const auto &[name, domain] = DomainNames[d];
                    const auto attribute = meshes.AddAttribute(source, domain, storage, std::format("in.{}", name)).value();
                    for (const auto handle : Elements(Mesh{meshes, source}, domain)) test::SetAttributeWords(meshes, attribute, handle, InputValue(storage, d, handle));
                }
            });
            struct Conversion {
                uint32_t Source, Target;
                std::string_view Mode;
                DomainReduction Reduction;
                std::string Name;
            };
            std::vector<Conversion> conversions;
            for (uint32_t from = 0u; from < DomainNames.size(); ++from) {
                for (uint32_t to = 0u; to < DomainNames.size(); ++to) {
                    for (const auto &[mode, reduction] : modes) conversions.push_back({from, to, mode, reduction, std::format("out.{}.{}.{}", from, to, mode)});
                }
            }
            // One read per source domain, then a Convert Domain node and a Store node for each conversion.
            Gesture(f, [&] {
                for (uint32_t d = 0u; d < DomainNames.size(); ++d) action::AddToGesture(AddNode{mesh, action::nodes::OpNode{"geompropvalue"}});
                for (size_t i = 0u; i < conversions.size(); ++i) {
                    action::AddToGesture(AddNode{mesh, action::nodes::OpNode{"convertdomain"}});
                    action::AddToGesture(AddNode{mesh, action::nodes::KindNode{KindOf<StoreNamedAttribute>}});
                }
            });
            const auto nodes = NodesOf(f.R, mesh);
            const auto read = [&](uint32_t d) { return nodes[1u + d]; };
            const auto convert = [&](size_t i) { return nodes[1u + DomainNames.size() + 2u * i]; };
            const auto store = [&](size_t i) { return nodes[2u + DomainNames.size() + 2u * i]; };
            Gesture(f, [&] {
                for (uint32_t d = 0u; d < DomainNames.size(); ++d) {
                    action::AddToGesture(OpInputOf(read(d), "geomprop", std::format("in.{}", DomainNames[d].first)));
                    action::AddToGesture(OpInputOf(read(d), "default", action::nodes::OpLiteral{type, std::vector<uint32_t>(mx::Words(type))}));
                }
                auto geometry = SocketLink{f.Node(mesh, 0u), f.Out(f.Node(mesh, 0u))};
                for (size_t i = 0u; i < conversions.size(); ++i) {
                    const auto &[from, to, mode, reduction, name] = conversions[i];
                    for (const auto [input, value] : {std::pair{"source", DomainNames[from].first}, std::pair{"target", DomainNames[to].first}, std::pair{"mode", mode}}) action::AddToGesture(OpInputOf(convert(i), input, std::string{value}));
                    action::AddToGesture(Link{mesh, read(from), f.Out(read(from)), convert(i), f.OpSocket(convert(i), "in")});
                    action::AddToGesture(SetAttributeName{store(i), name});
                    action::AddToGesture(UpdateOf<&StoreNamedAttribute::Domain>(store(i), DomainNames[to].second));
                    action::AddToGesture(Link{mesh, geometry.FromNode, geometry.FromSocket, store(i), 0u});
                    action::AddToGesture(Link{mesh, convert(i), f.Out(convert(i)), store(i), 1u});
                    geometry = {store(i), f.Out(store(i))};
                }
                action::AddToGesture(SetDisplay{mesh, geometry.FromNode});
            });
            const auto &meshes = f.Store();
            for (const auto &[from, to, mode, reduction, name] : conversions) {
                ExpectStored(f, mesh, name, DomainNames[to].second, [&](const Mesh &output, uint32_t t) {
                    return Reduce(storage, reduction, Related(output, DomainNames[from].second, DomainNames[to].second, t) | std::views::transform([&](uint32_t s) { return InputValue(storage, from, s); }) | std::ranges::to<std::vector>());
                });
                expect(meshes.GetAttribute(meshes.FindAttribute(f.Output(mesh), name)).Type == storage);
            }
        }
    }
}

// A graph whose consumer reads a field of another domain through link `Link`, which converts from `Source` to `Target` with `Mode`, and the attribute its Store node writes, if any.
struct ConvertedLink {
    std::string_view Name;
    GraphDescription Description;
    uint32_t Link;
    std::string_view Source, Target, Mode;
    std::string_view Stored{};
};
// The graph with a Convert Domain node in the link from its source.
GraphDescription Explicit(GraphDescription graph, const ConvertedLink &row) {
    const auto convert = uint32_t(graph.Nodes.size() + 1u);
    graph.Nodes.push_back({"convertdomain", {Converts(row.Source, row.Target, row.Mode)}});
    const auto from = std::exchange(graph.Links[row.Link].From, convert);
    graph.Links.push_back({from, convert, 0u, "in"});
    return graph;
}
// Plane Side selects the positions above z = 0.5.
const std::vector<NodeUpdate> TopHalf{Literal("point", vec3{0.f, 0.f, .5f}), Literal("side", 0)};
const std::array<ConvertedLink, 8> ConvertedLinks{{
    // A face selection reads the mean of each face's vertex positions.
    {"SelectionReadsFacePositions", {primitive::Cube{}, {{"position"}, {"extract", {Literal("index", 2)}}, {"ifgreater", {Literal("value2", 0.f)}}, {KindOf<action::mesh::Delete>, {Set<&action::mesh::Delete::Mode>(DeleteMode::Faces)}}}, {{1, 2, 0, "in"}, {2, 3, 0, "value1"}, {3, 4, 1}, {0, 4, 0}}, 4}, 0u, "vertex", "face", "mean"},
    // A vertex selection holds where any face around the vertex is selected, and a face selection where all its vertices are.
    {"SelectionReadsFaceField", {primitive::Cube{}, {{"planeside", TopHalf}, {"convertdomain", {Converts("vertex", "face", "all")}}, {KindOf<action::mesh::Delete>}}, {{1, 2, 0, "in"}, {2, 3, 1}, {0, 3, 0}}, 3}, 1u, "face", "vertex", "any"},
    {"SelectionReadsVertexField", {primitive::Cube{}, {{"planeside", TopHalf}, {KindOf<StoreNamedAttribute>, {Named("top")}}, {"geompropvalue", {Text("geomprop", "top"), Literal("default", false)}}, {KindOf<action::mesh::Delete>, {Set<&action::mesh::Delete::Mode>(DeleteMode::Faces)}}}, {{1, 2, 1}, {0, 2, 0}, {3, 4, 1}, {2, 4, 0}}, 4}, 2u, "vertex", "face", "all"},
    // Poke's per-face offset reads the mean of each face's vertex heights.
    {"PokeReadsVertexHeat", {primitive::Cube{}, {{"geompropvalue", {Text("geomprop", "heat"), Literal("default", 0.f)}}, {KindOf<action::mesh::Poke>}}, {{1, 2, ParameterSocket<&action::mesh::Poke::Offset>}, {0, 2, 0}}, 2, AddHeightHeat}, 0u, "vertex", "face", "mean"},
    // Set Position reads the mean of the stored normals of the faces around each vertex.
    {"PositionsReadFaceNormals", {primitive::Cube{}, {{"normal"}, {KindOf<StoreNamedAttribute>, {Named("facing"), Set<&StoreNamedAttribute::Domain>(AttributeDomain::Face)}}, {"geompropvalue", {Text("geomprop", "facing"), Literal("default", vec3{})}}, {KindOf<SetPosition>}}, {{1, 2, 1}, {0, 2, 0}, {3, 4, 1}, {2, 4, 0}}, 4}, 2u, "face", "vertex", "mean"},
    {"EdgesStoreVertexHeat", {primitive::Cube{}, {{"geompropvalue", {Text("geomprop", "heat"), Literal("default", 0.f)}}, {KindOf<StoreNamedAttribute>, {Named("edge heat"), Set<&StoreNamedAttribute::Domain>(AttributeDomain::Edge)}}}, {{1, 2, 1}, {0, 2, 0}}, 2, AddHeightHeat}, 0u, "vertex", "edge", "mean", "edge heat"},
    // A corner reads its vertex's position and its face's attribute through the point.
    {"CornersStorePositions", {primitive::Cube{}, {{"position"}, {KindOf<StoreNamedAttribute>, {Named("corner position"), Set<&StoreNamedAttribute::Domain>(AttributeDomain::Corner)}}}, {{1, 2, 1}, {0, 2, 0}}, 2}, 0u, "vertex", "corner", "mean", "corner position"},
    {"CornersStoreFaceIndex", {primitive::Cube{}, {{"elementindex"}, {KindOf<StoreNamedAttribute>, {Named("index"), Set<&StoreNamedAttribute::Domain>(AttributeDomain::Face)}}, {"geompropvalue", {Text("geomprop", "index"), Literal("default", 0)}}, {KindOf<StoreNamedAttribute>, {Named("corner index"), Set<&StoreNamedAttribute::Domain>(AttributeDomain::Corner)}}}, {{1, 2, 1}, {0, 2, 0}, {3, 4, 1}, {2, 4, 0}}, 4}, 2u, "face", "corner", "max", "corner index"},
}};
// A consumer that reads a field of another domain cooks bit for bit as the graph with an explicit Convert Domain node of the default mode in its link.
void AutomaticEqualsExplicit(GraphFixture &f, const ConvertedLink &row) {
    const auto automatic = Build(f, row.Description);
    const auto shape = f.Shaped(automatic.Mesh);
    const auto converted = Build(f, Explicit(row.Description, row));
    expect(!f.Fault(converted.Nodes.back()));
    expect(SameBits(f.Shaped(converted.Mesh), shape));
    if (row.Stored.empty()) return;
    const auto stored = StoredWords(f, automatic.Mesh, row.Stored);
    expect(!stored.empty() && StoredWords(f, converted.Mesh, row.Stored) == stored);
}

// Each element's rank among the live elements of its domain, by handle.
std::map<uint32_t, uint32_t> Ordinals(const Mesh &mesh, AttributeDomain domain) {
    std::map<uint32_t, uint32_t> ordinals;
    for (const auto handle : Elements(mesh, domain)) ordinals.emplace(handle, uint32_t(ordinals.size()));
    return ordinals;
}

// A grid of `count` quads along each side whose vertex (x, z) holds the texture coordinates `uv(x, z)` and the position `position(x, z)`.
MeshSource UvGrid(uint32_t count, auto &&uv, auto &&position) {
    MeshSource source;
    auto &uvs = source.Attrs.TexCoords0.emplace();
    for (uint32_t z = 0u; z <= count; ++z) {
        for (uint32_t x = 0u; x <= count; ++x) {
            uvs.push_back(uv(x, z));
            source.Data.Positions.push_back(position(x, z));
        }
    }
    for (uint32_t z = 0u; z < count; ++z) {
        for (uint32_t x = 0u; x < count; ++x) {
            const auto v = z * (count + 1u) + x;
            source.Data.AddFace(std::array{v, v + count + 1u, v + count + 2u, v + 1u});
        }
    }
    return source;
}
// The unit square of texture coordinates as a grid of quads whose positions are their texture coordinates.
MeshSource UvSquare(uint32_t quads) {
    const auto uv = [quads](uint32_t x, uint32_t z) { return vec2{float(x) / float(quads), float(z) / float(quads)}; };
    return UvGrid(quads, uv, [&](uint32_t x, uint32_t z) { return vec3{uv(x, z).x, 0.f, uv(x, z).y}; });
}
// A grid whose vertices hold two texture coordinate sets and a color, which its corners store, with values whose means are exact.
MeshSource UvColorGrid() {
    auto source = UvGrid(2u, [](uint32_t x, uint32_t z) { return vec2{float(x) / 2.f, float(z) / 2.f}; }, [](uint32_t x, uint32_t z) { return vec3{float(x) - 1.f, 0.f, float(z) - 1.f}; });
    auto &second = source.Attrs.TexCoords1.emplace();
    auto &colors = source.Attrs.Colors0.emplace();
    source.Attrs.Colors0ComponentCount = 4u;
    for (uint32_t z = 0u; z < 3u; ++z) {
        for (uint32_t x = 0u; x < 3u; ++x) {
            second.emplace_back(float(x), 2.f * float(z));
            colors.emplace_back(float(x) / 4.f, float(z) / 4.f, .5f, 1.f);
        }
    }
    return source;
}
// A field the last of `Nodes` computes on the shape, which a Store node writes under the row's name on the domain, and its value at each element of the stored mesh.
struct StoredFieldRow {
    std::string Name;
    std::variant<PrimitiveParameters, MeshSource> Shape;
    std::vector<GraphDescription::Node> Nodes;
    std::vector<GraphDescription::Link> Links;
    AttributeDomain Domain;
    std::function<AttributeValue(const Mesh &, uint32_t element)> Expected;
};
std::vector<StoredFieldRow> StoredFieldRows() {
    using D = AttributeDomain;
    std::vector<StoredFieldRow> rows;
    // Element Index gives each element its ordinal in its own domain.
    for (const auto &[name, domain] : DomainNames) {
        rows.push_back({std::format("index {}", name), primitive::Cylinder{.Slices = 5}, {{"elementindex"}}, {}, domain, [domain](const Mesh &m, uint32_t element) { return AttributeValue{Ordinals(m, domain).at(element)}; }});
    }
    // Convert Domain's input evaluates on its source domain, so each vertex takes the greatest ordinal of the faces around it.
    rows.push_back({"around", primitive::Cylinder{.Slices = 5}, {{"elementindex"}, {"convertdomain", {Converts("face", "vertex", "max")}}}, {{1, 2, 0, "in"}}, D::Vertex, [](const Mesh &m, uint32_t vertex) {
                        const auto faces = Ordinals(m, D::Face);
                        return AttributeValue{std::ranges::max(Related(m, D::Face, D::Vertex, vertex) | std::views::transform([&](uint32_t face) { return faces.at(face); }))};
                    }});
    // Plane Side reads the position on its consumer's domain: each vertex's, each edge's midpoint and each face's center.
    constexpr vec3 Normal{1.f, 1.f, 0.f};
    for (const auto &[name, domain] : DomainNames) {
        if (domain == D::Corner) continue;
        for (int32_t side = 0; side < 3; ++side) {
            rows.push_back({std::format("side {} {}", side, name), primitive::Cube{}, {{"planeside", {Literal("normal", Normal), Literal("side", side)}}}, {}, domain, [domain, side, n = Normalize(Normal)](const Mesh &m, uint32_t element) {
                                const auto vertices = Related(m, D::Vertex, domain, element);
                                vec3 center{};
                                for (const auto v : vertices) center += test::Position(m, Mesh::VH{v});
                                center /= float(vertices.size());
                                const auto distance = Dot(n, center), bound = 1e-4f * std::max(1.f, Length(center));
                                const auto held = distance > bound ? 0 : distance < -bound ? 1 :
                                                                                             2;
                                return AttributeValue{uint32_t(held == side)};
                            }});
        }
    }
    // A source the consumer's domain lacks evaluates on the domain that stores it and is read as the mean of its values there.
    // The second texture coordinates read with a computed set index or nested in a test node, a ramp's unset input reads the first, and a color reads the corners that hold it, read or nested, each against the mean of the corners' values.
    // The texture coordinates of a set at a corner, with V up as the driver reads them.
    const auto texcoord = [](const Mesh &m, uint32_t set, uint32_t corner) {
        const auto uv = m.GetStore().Arenas().CornerUvs[set].Get(m.GetStore().Get(m.GetStoreId()).CornerUvs[set], corner);
        return vec2{uv.x, 1.f - uv.y};
    };
    const auto second = [texcoord](const Mesh &m, uint32_t corner) {
        const auto uv = texcoord(m, 1u, corner);
        return vec4{uv.x, uv.y, 0.f, 0.f};
    };
    const auto ramp = [texcoord](const Mesh &m, uint32_t corner) { return vec4{std::clamp(texcoord(m, 0u, corner).x, 0.f, 1.f), 0.f, 0.f, 0.f}; };
    const auto color = [](const Mesh &m, uint32_t corner) { return m.GetStore().Arenas().CornerColors.Get(m.GetStore().Get(m.GetStoreId()).CornerColors, corner); };
    // Rows of the read on each domain, whose field holds the first `components` of the mean of the value at the corners of each element.
    const auto reads = [&](std::string_view name, std::vector<GraphDescription::Node> nodes, std::vector<GraphDescription::Link> links, std::initializer_list<D> domains, uint32_t components, std::function<vec4(const Mesh &, uint32_t)> value) {
        for (const auto domain : domains) {
            const auto domain_name = std::ranges::find(DomainNames, domain, &std::pair<std::string_view, D>::second)->first;
            rows.push_back({std::format("{} {}", name, domain_name), UvColorGrid(), nodes, links, domain, [domain, components, value](const Mesh &m, uint32_t element) {
                                const auto corners = Related(m, D::Corner, domain, element);
                                vec4 mean{};
                                for (const auto corner : corners) mean += value(m, corner);
                                mean /= float(corners.size());
                                AttributeValue words{};
                                for (uint32_t c = 0u; c < components; ++c) words[c] = std::bit_cast<uint32_t>(mean[c]);
                                return words;
                            }});
        }
    };
    reads("second", {{"add", {Literal("in1", 0), Literal("in2", 1)}}, {"texcoord", {OutputType(mx::Type::Vector2)}}}, {{1, 2, 0, "index"}}, {D::Vertex, D::Face}, 2u, second);
    reads("nested second", {{"testsecondtexcoord"}}, {}, {D::Vertex, D::Face}, 2u, second);
    reads("ramp", {{"ramplr", {Literal("valuer", 1.f)}}}, {}, {D::Vertex, D::Face}, 1u, ramp);
    reads("color", {{"geomcolor", {OutputType(mx::Type::Color4)}}}, {}, {D::Vertex, D::Face, D::Edge}, 4u, color);
    reads("nested color", {{"testcolor"}}, {}, {D::Vertex, D::Face}, 4u, color);
    return rows;
}
// Each row's field holds its value at each element of its domain, and no node of its graph faults.
void StoredFields(GraphFixture &f) {
    for (const auto &row : StoredFieldRows()) {
        auto nodes = row.Nodes;
        auto links = row.Links;
        const auto store = uint32_t(nodes.size() + 1u);
        nodes.push_back({KindOf<StoreNamedAttribute>, {Named(row.Name), Set<&StoreNamedAttribute::Domain>(row.Domain)}});
        links.push_back({store - 1u, store, 1u});
        links.push_back({0u, store, 0u});
        const auto g = Build(f, {row.Shape, std::move(nodes), std::move(links), store});
        expect(std::ranges::none_of(g.Nodes, [&](state::Entity node) { return f.Fault(node).has_value(); })) << row.Name;
        ExpectStored(f, g.Mesh, row.Name, row.Domain, row.Expected);
    }
}

// The texels of the image the occurrence of a Bake node or a filter holds, as their floats.
std::vector<float> BakedTexels(GraphFixture &f, state::Entity node) {
    const auto image = OccurrenceImageOf(f.R, GraphOf(f.R, node), std::array{node});
    expect(fatal(image.has_value()));
    return Texels(image->Image);
}
// The texels of a Width by Height image of Stride floats per texel blurred by a kernel `size` wide in texture coordinates along each axis in turn, each texel the weighted mean of the texels around it, clamped to the image.
// A box kernel weighs a texel by its overlap with the box, and a Gaussian one by a Gaussian whose deviation is a sixth of the kernel's width.
std::vector<float> Blurred(std::span<const float> texels, uint32_t width, uint32_t height, uint32_t stride, float size, bool gaussian) {
    // A pass of a kernel `extent` texels wide, where `at(t, k)` is the texel k texels from texel t along the pass's axis, clamped to the image.
    const auto pass = [&](std::span<const float> in, float extent, auto &&at) {
        std::vector<float> out(in.size());
        const auto radius = int(std::ceil(gaussian ? .5f * extent : .5f * extent + .5f));
        const auto deviation = double(extent) / 6.0;
        for (uint32_t t = 0u; t < width * height; ++t) {
            for (uint32_t c = 0u; c < stride; ++c) {
                double sum = 0.0, total = 0.0;
                for (int k = -radius; k <= radius; ++k) {
                    const double weight = gaussian ? std::exp(-double(k * k) / (2.0 * deviation * deviation)) : std::max(0.f, std::min(float(k) + .5f, .5f * extent) - std::max(float(k) - .5f, -.5f * extent));
                    sum += weight * in[at(t, k) * stride + c];
                    total += weight;
                }
                out[t * stride + c] = float(sum / total);
            }
        }
        return out;
    };
    const auto rows = pass(texels, size * float(width), [&](uint32_t t, int k) { return t / width * width + uint32_t(std::clamp(int(t % width) + k, 0, int(width) - 1)); });
    return pass(rows, size * float(height), [&](uint32_t t, int k) { return uint32_t(std::clamp(int(t / width) + k, 0, int(height) - 1)) * width + t % width; });
}
// Bakes each value without a margin on a Width by Height grid over the mesh's layout, and returns the Bake nodes.
// A Set Position reads every image, which demands the bakes.
std::vector<state::Entity> BakeEach(GraphFixture &f, state::Entity mesh, std::span<const state::Entity> values, uint32_t width, uint32_t height) {
    const auto root = f.Node(mesh, 0u);
    std::vector<state::Entity> bakes;
    std::optional<state::Entity> offset;
    for (const auto value : values) {
        const auto bake = bakes.emplace_back(f.Append(mesh, KindOf<Bake>, {Set<&Bake::Width>(width), Set<&Bake::Height>(height), Set<&Bake::Margin>(0u)}));
        f.Do(Link{mesh, root, f.Out(root), bake, 0u});
        f.Do(Link{mesh, value, f.Out(value), bake, 1u});
        const auto read = f.Append(mesh, "image", {From("file", bake), Literal("default", vec3{})});
        offset = offset ? f.Append(mesh, "add", {From("in1", *offset), From("in2", read)}) : read;
    }
    const auto set = f.Append(mesh, KindOf<SetPosition>);
    f.Do(Link{mesh, *offset, f.Out(*offset), set, 1u});
    f.Route(mesh, {root, set});
    return bakes;
}
// The texture coordinates of the unit square's mesh as an op, and an op separating them into their components.
std::pair<state::Entity, state::Entity> UvComponents(GraphFixture &f) {
    const auto mesh = f.AddSourced(UvSquare(4)).first;
    const auto uv = f.Append(mesh, "texcoord", {OutputType(mx::Type::Vector2)});
    return {uv, f.Append(mesh, "separate2", {From("in", uv)})};
}
// One where the coordinate along the axis exceeds the threshold, and else zero.
state::Entity Step(GraphFixture &f, state::Entity components, std::string_view axis, float threshold) {
    const auto mesh = GraphOf(f.R, components);
    const auto step = f.Append(mesh, "ifgreater", {Literal("value2", threshold), Literal("in1", 1.f), Literal("in2", 0.f)});
    f.Do(Link{mesh, components, f.OpSocket(components, axis), step, f.OpSocket(step, "value1")});
    return step;
}
state::Entity Blur(GraphFixture &f, state::Entity input, float size, std::string filter) {
    return f.Append(GraphOf(f.R, input), "blur", {From("in", input), Literal("size", size), Text("filtertype", std::move(filter))});
}
state::Entity HeightNormal(GraphFixture &f, state::Entity height, float scale) { return f.Append(GraphOf(f.R, height), "heighttonormal", {From("in", height), Literal("scale", scale)}); }
constexpr uint32_t BakeWidth{37}, BakeHeight{29};
// A bake of the field, an image read of it, or another image.
state::Entity BakeOf(GraphFixture &f, state::Entity value) { return BakeEach(f, GraphOf(f.R, value), std::array{value}, BakeWidth, BakeHeight).front(); }

// What an image row checks of the node its graph builds: a bake of the field read back at its texel centers holds the field, a blur holds one host blur of its input's texels, and a derivative holds the normal of the forward differences of its height's texels.
enum class ImageProperty : uint8_t { Readback,
                                     Blur,
                                     Derivative };
struct ImageRow {
    std::string_view Name;
    ImageProperty Property;
    state::Entity (*Build)(GraphFixture &);
    float Parameter{}; // A blur's size, or a derivative's scale
    bool Gaussian{};
};
const std::array<ImageRow, 13> ImageRows{{
    {"NoiseReadback", ImageProperty::Readback, [](GraphFixture &f) { const auto [uv, xy] = UvComponents(f); return f.Append(GraphOf(f.R, uv), "noise2d", {From("texcoord", uv), OutputType(mx::Type::Float), Literal("amplitude", 4.f)}); }},
    {"PositionReadback", ImageProperty::Readback, [](GraphFixture &f) { return f.Append(f.AddSourced(UvSquare(4)).first, "position"); }},
    {"NormalReadback", ImageProperty::Readback, [](GraphFixture &f) { return f.Append(f.AddSourced(UvSquare(4)).first, "normal"); }},
    {"BoxStep", ImageProperty::Blur, [](GraphFixture &f) { return Blur(f, Step(f, UvComponents(f).second, "outx", .625f), .25f, "box"); }, .25f},
    {"GaussianStep", ImageProperty::Blur, [](GraphFixture &f) { return Blur(f, Step(f, UvComponents(f).second, "outy", .3125f), .25f, "gaussian"); }, .25f, true},
    {"BlurredBlur", ImageProperty::Blur, [](GraphFixture &f) { return Blur(f, Blur(f, Step(f, UvComponents(f).second, "outx", .625f), .25f, "box"), .25f, "box"); }, .25f},
    {"ImageBox", ImageProperty::Blur, [](GraphFixture &f) { return Blur(f, BakeOf(f, Step(f, UvComponents(f).second, "outx", .625f)), .2f, "box"); }, .2f},
    {"ImageGaussian", ImageProperty::Blur, [](GraphFixture &f) { return Blur(f, BakeOf(f, Step(f, UvComponents(f).second, "outy", .3125f)), .15f, "gaussian"); }, .15f, true},
    {"TextureBox", ImageProperty::Blur, [](GraphFixture &f) {
         f.Do(action::io::LoadGltf{BoxTextured});
         const auto blur = f.Append(SortedEntities(f.R.view<const GeometryGraph>()).front(), "blur", {Literal("size", .02f)});
         f.Do(OpInputOf(blur, "in", SortedEntities(f.R.view<const MaterializedTexture>()).front()));
         return blur; }, .02f},
    {"LinearNormal", ImageProperty::Derivative, [](GraphFixture &f) { const auto uv = UvComponents(f).first; return HeightNormal(f, f.Append(GraphOf(f.R, uv), "dotproduct", {From("in1", uv), Literal("in2", vec2{.8f, -.5f})}), 4.f); }, 4.f},
    {"SquareNormal", ImageProperty::Derivative, [](GraphFixture &f) {
         const auto xy = UvComponents(f).second;
         const auto mesh = GraphOf(f.R, xy);
         const auto square = f.Append(mesh, "multiply");
         for (const auto *input : {"in1", "in2"}) f.Do(Link{mesh, xy, f.OpSocket(xy, "outx"), square, f.OpSocket(square, input)});
         return HeightNormal(f, square, 4.f); }, 4.f},
    {"WaveNormal", ImageProperty::Derivative, [](GraphFixture &f) {
         const auto xy = UvComponents(f).second;
         const auto mesh = GraphOf(f.R, xy);
         const auto phase = f.Append(mesh, "multiply", {Literal("in2", 2.f * std::numbers::pi_v<float>)});
         f.Do(Link{mesh, xy, f.OpSocket(xy, "outy"), phase, f.OpSocket(phase, "in1")});
         return HeightNormal(f, f.Append(mesh, "multiply", {From("in1", f.Append(mesh, "sin", {From("in", phase)})), Literal("in2", .2f)}), 1.f); }, 1.f},
    {"ImageHeightNormal", ImageProperty::Derivative, [](GraphFixture &f) {
         const auto step = Step(f, UvComponents(f).second, "outx", .5f);
         const auto mesh = GraphOf(f.R, step);
         return HeightNormal(f, f.Append(mesh, "image", {From("file", BakeOf(f, Blur(f, step, .25f, "box"))), Literal("default", 0.f), Text("uaddressmode", "clamp"), Text("vaddressmode", "clamp")}), 4.f); }, 4.f},
}};
// The sRGB-encoded bytes of the document texture, decoded to linear floats.
std::vector<float> DecodedTexture(GraphFixture &f, state::Entity texture) {
    const auto &picture = *std::ranges::find(f.R.Resources.get<const TextureStore>().Textures, f.R.get<const MaterializedTexture>(texture).SamplerSlot, &TextureEntry::SamplerSlot);
    return ReadbackTextureRgba8(f.R.Resources.get<const mtl::Context>(), picture).value() | std::views::transform([](std::byte b) {
               const auto c = float(b) / 255.f;
               return c <= .04045f ? c / 12.92f : std::pow((c + .055f) / 1.055f, 2.4f);
           }) |
        std::ranges::to<std::vector>();
}
bool Near(std::span<const float> a, std::span<const float> b, float tolerance) {
    return a.size() == b.size() && std::ranges::equal(a, b, [&](float x, float y) { return std::abs(x - y) <= tolerance; });
}
void CheckImage(GraphFixture &f, const ImageRow &row) {
    const auto node = row.Build(f);
    const auto mesh = GraphOf(f.R, node);
    expect(!f.Fault(node));
    switch (row.Property) {
        case ImageProperty::Readback: {
            const auto bake = BakeOf(f, node);
            const auto texels = BakedTexels(f, bake);
            // The read takes the type of the bake's values.
            const auto read = f.Append(mesh, "image", {From("file", bake), texels.size() == BakeWidth * BakeHeight ? Literal("default", 0.f) : Literal("default", vec3{}), Text("filtertype", "closest")});
            expect(texels == BakedTexels(f, BakeOf(f, read)) && size_t(std::ranges::count(texels, 0.f)) < texels.size());
            break;
        }
        case ImageProperty::Blur: {
            const auto image = OccurrenceImageOf(f.R, mesh, std::array{node});
            expect(fatal(image.has_value()));
            const auto blurred = Texels(image->Image);
            const auto width = image->Width, height = image->Height, stride = uint32_t(blurred.size()) / (width * height);
            // The input's texels: a texture's, an image's, or a field's on the blur's grid.
            const auto &op = f.R.get<const graph::Op>(node);
            const auto input = InputOf(f.R, node, f.OpSocket(node, "in")).FromNode;
            const auto texels = op.Images.contains("in")                                                                                                 ? DecodedTexture(f, op.Images.find("in")->second) :
                f.R.get<const GraphNode>(input).Kind == KindOf<Bake> || (f.R.all_of<graph::Op>(input) && f.R.get<const graph::Op>(input).Node == "blur") ? BakedTexels(f, input) :
                                                                                                                                                           BakedTexels(f, BakeEach(f, mesh, std::array{input}, width, height).front());
            expect(Near(blurred, Blurred(texels, width, height, stride, row.Parameter, row.Gaussian), 2e-3f));
            break;
        }
        case ImageProperty::Derivative: {
            constexpr uint32_t Width{64}, Height{48};
            const auto height_node = InputOf(f.R, node, f.OpSocket(node, "in")).FromNode;
            const auto bakes = BakeEach(f, mesh, std::array{node, height_node}, Width, Height);
            const auto normals = BakedTexels(f, bakes[0]), heights = BakedTexels(f, bakes[1]);
            const auto stride = uint32_t(normals.size()) / (Width * Height), height_stride = uint32_t(heights.size()) / (Width * Height);
            // Each texel differs toward the next texel right and down, short of the last column and row, and V runs up.
            uint32_t held = 0u, checked = 0u;
            for (uint32_t t = 0u; t < Width * Height; ++t) {
                if (t % Width == Width - 1u || t / Width == Height - 1u) continue;
                const auto h = [&](uint32_t at) { return heights[at * height_stride]; };
                const auto gu = (h(t + 1u) - h(t)) * float(Width), gv = (h(t) - h(t + Width)) * float(Height);
                const auto n = Normalize(vec3{-row.Parameter / 16.f * gu, -row.Parameter / 16.f * gv, 1.f}) * .5f + .5f;
                ++checked;
                held += std::abs(normals[t * stride] - n.x) < 1e-4f && std::abs(normals[t * stride + 1u] - n.y) < 1e-4f && std::abs(normals[t * stride + 2u] - n.z) < 1e-4f;
            }
            expect(held == checked) << held << checked;
            break;
        }
    }
}

} // namespace

void TestGraphs() {
    GraphFixture f;
    Fixture other;
    OperatorKinds(f, other);
    for (const auto &row : DataGraphs) {
        Survives(f, other, row.Name, [&row](GraphFixture &f) {
            const auto built = Build(f, row.Description);
            const auto faces = [](GraphFixture &f, state::Entity mesh, std::optional<uint32_t> expected) {
                if (!expected) return;
                const auto count = Mesh{f.Store(), f.Output(mesh)}.FaceCount();
                expect(count == *expected) << "faces" << count;
            };
            GraphCase graph{built.Mesh, [&row, built, faces](GraphFixture &f) { if (row.Oracle) row.Oracle(f, built); faces(f, built.Mesh, row.Faces); }};
            if (row.Edit) graph.Edit = [&row, faces](GraphFixture &f, state::Entity mesh) { row.Edit(f, mesh); faces(f, mesh, row.EditedFaces); };
            return graph;
        });
        if (row.Called) Scenario(f, other, std::format("{}Called", row.Name), [&row](GraphFixture &f) { CalledEqualsStored(f, row); });
        Scenario(f, other, std::format("{}Shared", row.Name), [&row](GraphFixture &f) { SharedSubgraph(f, row); });
    }
    boost::ut::test("DomainConversions") = [&f] {
        const TestDir dir{"/tmp/mesheditor-scratch/DomainConversions"};
        f.Begin(dir.Path);
        DomainConversions(f);
    };
    for (const auto &row : ConvertedLinks) Scenario(f, other, std::string{row.Name}, [&row](GraphFixture &f) { AutomaticEqualsExplicit(f, row); });
    Scenario(f, other, "StoredFields", StoredFields);
    for (const auto &row : ImageRows) Scenario(f, other, std::string{row.Name}, [&row](GraphFixture &f) { CheckImage(f, row); });
    for (const auto &row : TimedGraphs) Scenario(f, other, std::string{row.Name}, [&row](GraphFixture &f) { Timed(f, row); });
    Scenario(f, other, "TimeCollider", [](GraphFixture &f) { TimeCollider(f, TimeMoves); });
    Scenario(f, other, "UntimedCollider", [](GraphFixture &f) { TimeCollider(f, DoubledPositions); });
    for (const auto &row : BuiltGraphs) Survives(f, other, row.Name, row.Build);
    TestOperators(f, other);
    SubgraphInstances(f, other);
    Scenario(f, other, "HistoryPanelEdit", HistoryPanelEdit);
    Scenario(f, other, "ViewportSelection", ViewportSelection);
    boost::ut::test("BeginKeepsStates") = [&f] {
        const TestDir dir{"/tmp/mesheditor-scratch/BeginKeepsStates"}, next{"/tmp/mesheditor-scratch/BeginKeepsStates-next"};
        f.Begin(dir.Path);
        BeginKeepsStates(f, next.Path);
    };
    Scenario(f, other, "GestureLodRepair", GestureLodRepair);
    Scenario(f, other, "GestureOnNewRecord", GestureOnNewRecord);
    Scenario(f, other, "LinkedSkinPoses", LinkedSkinPoses);
    Scenario(f, other, "ContextualSubgraph", ContextualSubgraph);
    Scenario(f, other, "DuplicateSkipsParts", DuplicateSkipsParts);
}
