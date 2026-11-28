#include "graph/BuiltinSubgraphs.h"
#include "action/Nodes.h"
#include "graph/Graph.h"
#include "graph/OpNodes.h"
#include "scene/Entity.h"
#include "state/Registry.h"

namespace graph {
namespace {
using action::nodes::Action, action::nodes::AddNode, action::nodes::ExposeSocket, action::nodes::Link, action::nodes::SetOpInput;

state::Entity Inner(state::Registry &r, state::Entity subgraph, uint16_t kind) {
    action::nodes::Apply(r, Action{AddNode{subgraph, action::nodes::KindNode{kind}}});
    return r.get<const GraphNodes>(subgraph).Last;
}
uint16_t Expose(state::Registry &r, state::Entity subgraph, state::Entity node, uint16_t socket, const char *name) {
    action::nodes::Apply(r, Action{ExposeSocket{subgraph, node, socket, name}});
    return r.get<const SubgraphInterface>(subgraph).Ports.back().Id;
}
// An op node of the MaterialX node, with its texts and literals set by input name.
state::Entity InnerOp(state::Registry &r, state::Entity subgraph, std::string node, std::initializer_list<std::pair<std::string_view, std::string_view>> texts = {}, std::initializer_list<std::pair<std::string_view, int32_t>> integers = {}) {
    action::nodes::Apply(r, Action{AddNode{subgraph, action::nodes::OpNode{std::move(node)}}});
    const auto op = r.get<const GraphNodes>(subgraph).Last;
    for (const auto [input, text] : texts) action::nodes::Apply(r, Action{SetOpInput{op, std::string{input}, std::make_unique<action::nodes::OpInputValue>(std::string{text})}});
    for (const auto [input, value] : integers) action::nodes::Apply(r, Action{SetOpInput{op, std::string{input}, std::make_unique<action::nodes::OpInputValue>(action::nodes::OpLiteral{mx::Type::Integer, {std::bit_cast<uint32_t>(value)}})}});
    return op;
}
uint16_t Out(const state::Registry &r, state::Entity node) { return InputCountOf(r, node); }
void Wire(state::Registry &r, state::Entity subgraph, state::Entity from, state::Entity to, uint16_t socket) {
    action::nodes::Apply(r, Action{Link{subgraph, from, Out(r, from), to, socket}});
}

void SplitBySolid(state::Registry &r, state::Entity subgraph) {
    const auto outside = Inner(r, subgraph, KindOf<Boolean>);
    const auto inside = Inner(r, subgraph, KindOf<Boolean>);
    r.replace<Boolean>(outside, Boolean{.Operation = BooleanOperation::Difference});
    r.replace<Boolean>(inside, Boolean{.Operation = BooleanOperation::Intersection});
    action::nodes::Apply(r, Action{Link{subgraph, subgraph, Expose(r, subgraph, outside, 0u, "Solid"), inside, 0u}});
    action::nodes::Apply(r, Action{Link{subgraph, subgraph, Expose(r, subgraph, outside, 1u, "Cutter"), inside, 1u}});
    Expose(r, subgraph, outside, Out(r, outside), "Outside");
    Expose(r, subgraph, inside, Out(r, inside), "Inside");
}
void HullOfObjects(state::Registry &r, state::Entity subgraph) {
    const auto join = Inner(r, subgraph, KindOf<Join>);
    const auto hull = Inner(r, subgraph, KindOf<action::mesh::ConvexHull>);
    Wire(r, subgraph, join, hull, 0u);
    Expose(r, subgraph, join, 0u, "Mesh 1");
    Expose(r, subgraph, join, 0u, "Mesh 2");
    Expose(r, subgraph, hull, Out(r, hull), "Hull");
}
// A subdivide cutting the edge rings through the selected edges Cuts times and selecting the new loops, with the subgraph's Mesh, Edges and Cuts ports.
state::Entity CutRings(state::Registry &r, state::Entity subgraph) {
    const auto selected = InnerOp(r, subgraph, "selection");
    const auto ring = InnerOp(r, subgraph, "edgering");
    const auto cut = Inner(r, subgraph, KindOf<action::mesh::Subdivide>);
    r.replace<action::mesh::Subdivide>(cut, action::mesh::Subdivide{.NewLoopsOnly = true});
    Wire(r, subgraph, selected, ring, 0u);
    Wire(r, subgraph, ring, cut, 1u);
    Expose(r, subgraph, cut, 0u, "Mesh");
    Expose(r, subgraph, selected, 0u, "Edges");
    Expose(r, subgraph, cut, ParameterSocket<&action::mesh::Subdivide::Cuts>, "Cuts");
    return cut;
}
void LoopCut(state::Registry &r, state::Entity subgraph) {
    const auto cut = CutRings(r, subgraph);
    Expose(r, subgraph, cut, Out(r, cut), "Mesh");
}
void LoopCutAndSlide(state::Registry &r, state::Entity subgraph) {
    const auto cut = CutRings(r, subgraph);
    const auto slide = Inner(r, subgraph, KindOf<EdgeSlide>);
    r.emplace<PublishSelection>(cut);
    r.emplace<InlineSelection>(slide, InlineSelection{.Domain = Element::Edge, .Published = true});
    r.replace<EdgeSlide>(slide, EdgeSlide{.Factor = 0.f});
    Wire(r, subgraph, cut, slide, 0u);
    Expose(r, subgraph, slide, ParameterSocket<&EdgeSlide::Factor>, "Factor");
    Expose(r, subgraph, slide, ParameterSocket<&EdgeSlide::Direction>, "Direction");
    Expose(r, subgraph, slide, ParameterSocket<&EdgeSlide::Reference>, "Reference");
    Expose(r, subgraph, slide, Out(r, slide), "Mesh");
}
void FillHoles(state::Registry &r, state::Entity subgraph) {
    const auto boundary = InnerOp(r, subgraph, "boundary");
    const auto fill = Inner(r, subgraph, KindOf<action::mesh::Fill>);
    r.replace<action::mesh::Fill>(fill, action::mesh::Fill{.Sides = 4u});
    Wire(r, subgraph, boundary, fill, 1u);
    Expose(r, subgraph, fill, 0u, "Mesh");
    Expose(r, subgraph, fill, ParameterSocket<&action::mesh::Fill::Sides>, "Sides");
    Expose(r, subgraph, fill, Out(r, fill), "Mesh");
}
// A bisect feeding a fill of its output's boundary edges on the plane, with the subgraph's Mesh, Point and Normal ports.
struct Capped {
    state::Entity Bisect, Fill;
    uint16_t Point, Normal;
};
// Links the subgraph's Point and Normal ports into the Plane Side op's plane.
void LinkPlane(state::Registry &r, state::Entity subgraph, const Capped &capped, state::Entity side) {
    const auto &declared = *DefinitionOf(r.get<const Op>(side).Node);
    action::nodes::Apply(r, Action{Link{subgraph, subgraph, capped.Point, side, *InputSocket(declared, "point")}});
    action::nodes::Apply(r, Action{Link{subgraph, subgraph, capped.Normal, side, *InputSocket(declared, "normal")}});
}
Capped CappedBisect(state::Registry &r, state::Entity subgraph, const Bisect &parameters) {
    const auto bisect = Inner(r, subgraph, KindOf<Bisect>);
    const auto on = InnerOp(r, subgraph, "planeside");
    // An edge lies on the plane when both of its vertices do.
    const auto edges = InnerOp(r, subgraph, "convertdomain", {{"source", "vertex"}, {"target", "edge"}, {"mode", "all"}});
    const auto boundary = InnerOp(r, subgraph, "boundary");
    const auto both = InnerOp(r, subgraph, "and");
    const auto fill = Inner(r, subgraph, KindOf<action::mesh::Fill>);
    r.replace<Bisect>(bisect, parameters);
    Wire(r, subgraph, bisect, fill, 0u);
    // The and op's inputs are its first two sockets.
    Wire(r, subgraph, on, edges, 0u);
    Wire(r, subgraph, edges, both, 0u);
    Wire(r, subgraph, boundary, both, 1u);
    Wire(r, subgraph, both, fill, 1u);
    Expose(r, subgraph, bisect, 0u, "Mesh");
    const Capped capped{bisect, fill, Expose(r, subgraph, bisect, ParameterSocket<&Bisect::Point>, "Point"), Expose(r, subgraph, bisect, ParameterSocket<&Bisect::Normal>, "Normal")};
    LinkPlane(r, subgraph, capped, on);
    return capped;
}
void BisectFill(state::Registry &r, state::Entity subgraph) {
    const auto capped = CappedBisect(r, subgraph, Bisect{.ClearInner = true});
    Expose(r, subgraph, capped.Bisect, ParameterSocket<&Bisect::ClearInner>, "Clear Inner");
    Expose(r, subgraph, capped.Bisect, ParameterSocket<&Bisect::ClearOuter>, "Clear Outer");
    Expose(r, subgraph, capped.Fill, Out(r, capped.Fill), "Mesh");
    Expose(r, subgraph, capped.Fill, Out(r, capped.Fill) + 1u, "Fill");
}
// The region where the plane crosses the mesh: the side behind it capped on the plane, then its vertices behind the plane deleted.
void CrossSection(state::Registry &r, state::Entity subgraph) {
    const auto capped = CappedBisect(r, subgraph, Bisect{.Normal = {0, 0, 1}, .ClearOuter = true});
    // Side 1 is behind the plane.
    const auto behind = InnerOp(r, subgraph, "planeside", {}, {{"side", 1}});
    const auto clear = Inner(r, subgraph, KindOf<action::mesh::Delete>);
    LinkPlane(r, subgraph, capped, behind);
    Wire(r, subgraph, capped.Fill, clear, 0u);
    Wire(r, subgraph, behind, clear, 1u);
    Expose(r, subgraph, clear, Out(r, clear), "Section");
}
// The side behind the plane cleared, then joined to its reflection with reversed winding and welded where the halves meet on the plane.
void Symmetrize(state::Registry &r, state::Entity subgraph) {
    const auto bisect = Inner(r, subgraph, KindOf<Bisect>);
    const auto mirror = Inner(r, subgraph, KindOf<TransformElements>);
    const auto flip = Inner(r, subgraph, KindOf<action::mesh::FlipNormals>);
    const auto join = Inner(r, subgraph, KindOf<Join>);
    const auto weld = Inner(r, subgraph, KindOf<action::mesh::Merge>);
    r.replace<Bisect>(bisect, Bisect{.ClearInner = true});
    r.replace<TransformElements>(mirror, TransformElements{.Mirror = {1, 0, 0}});
    // The cut snaps the vertices it keeps on the plane onto it, so each meets its reflection within rounding.
    r.replace<action::mesh::Merge>(weld, action::mesh::Merge{.Mode = action::mesh::MergeMode::ByDistance, .Distance = 1e-5f});
    Wire(r, subgraph, bisect, join, 0u);
    Wire(r, subgraph, bisect, mirror, 0u);
    Wire(r, subgraph, mirror, flip, 0u);
    Wire(r, subgraph, flip, join, 0u);
    Wire(r, subgraph, join, weld, 0u);
    Expose(r, subgraph, bisect, 0u, "Mesh");
    action::nodes::Apply(r, Action{Link{subgraph, subgraph, Expose(r, subgraph, bisect, ParameterSocket<&Bisect::Point>, "Point"), mirror, ParameterSocket<&TransformElements::Pivot>}});
    action::nodes::Apply(r, Action{Link{subgraph, subgraph, Expose(r, subgraph, bisect, ParameterSocket<&Bisect::Normal>, "Normal"), mirror, ParameterSocket<&TransformElements::Mirror>}});
    Expose(r, subgraph, weld, Out(r, weld), "Mesh");
}
} // namespace

const std::array<BuiltinSubgraphSpec, BuiltinSubgraphCount> BuiltinSubgraphs{{
    {"Split by Solid", KindGroup::Combine, "Splits a closed mesh by a closed cutter into the part outside the cutter and the part inside it", SplitBySolid},
    {"Hull of Objects", KindGroup::Combine, "Builds the convex hull of two meshes together", HullOfObjects},
    {"Loop Cut", KindGroup::Operators, "Cuts the edge rings through the selected edges and selects the new loops", LoopCut},
    {"Loop Cut and Slide", KindGroup::Operators, "Cuts the edge rings through the selected edges and slides the new loops along their faces", LoopCutAndSlide},
    {"Fill Holes", KindGroup::Operators, "Fills each boundary loop of at most Sides edges", FillHoles},
    {"Bisect Fill", KindGroup::Operators, "Cuts the mesh along a plane, deletes the vertices on the cleared sides and fills the cut on the plane", BisectFill},
    {"Cross Section", KindGroup::Operators, "Leaves the region where a plane crosses the mesh", CrossSection},
    {"Symmetrize", KindGroup::Operators, "Replaces the side behind a plane with the reflection of the side in front, welded where the halves meet", Symmetrize},
}};

namespace {
struct Built {
    std::unique_ptr<state::Registry> Registry;
    std::array<state::Entity, BuiltinSubgraphCount> Subgraphs;
};
// Each built-in subgraph is built once, in BuiltinSubgraphs order, so its node entities are the same in every session.
const Built &BuiltSubgraphs() {
    static const Built built = [] {
        Built b{std::make_unique<state::Registry>(), {}};
        auto &r = *b.Registry;
        r.Resources.emplace<BuiltinSubgraphLiterals>();
        for (size_t i = 0; i < BuiltinSubgraphs.size(); ++i) {
            const auto subgraph = b.Subgraphs[i] = r.create();
            r.emplace<Subgraph>(subgraph);
            r.emplace<GraphNodes>(subgraph);
            r.emplace<SubgraphInterface>(subgraph);
            r.emplace<Name>(subgraph, BuiltinSubgraphs[i].Name);
            BuiltinSubgraphs[i].Define(r, subgraph);
        }
        // A built-in subgraph's page selects nothing.
        r.clear<GraphActive>();
        return b;
    }();
    return built;
}
} // namespace

const state::Registry &BuiltinSubgraphRegistry() { return *BuiltSubgraphs().Registry; }
SubgraphRef BuiltinSubgraph(BuiltinSubgraphId id) { return {&BuiltinSubgraphRegistry(), BuiltSubgraphs().Subgraphs[size_t(id)]}; }
} // namespace graph
