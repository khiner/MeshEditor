#include "procedure/Macros.h"
#include "action/Nodes.h"
#include "procedure/OpNodes.h"
#include "procedure/Procedure.h"
#include "scene/Entity.h"
#include "state/Registry.h"

namespace procedure {
namespace {
using action::nodes::Action, action::nodes::AddNode, action::nodes::ExposeSocket, action::nodes::Link, action::nodes::SetOpInput;

state::Entity Inner(state::Registry &r, state::Entity definition, uint16_t kind) {
    action::nodes::Apply(r, Action{AddNode{definition, action::nodes::KindNode{kind}}});
    return r.get<const NetworkNodes>(definition).Last;
}
uint16_t Expose(state::Registry &r, state::Entity definition, state::Entity node, uint16_t socket, const char *name) {
    action::nodes::Apply(r, Action{ExposeSocket{definition, node, socket, name}});
    return r.get<const SubnetInterface>(definition).Ports.back().Id;
}
// An op node of the MaterialX node, with its texts and literals set by input name.
state::Entity InnerOp(state::Registry &r, state::Entity definition, std::string node, std::initializer_list<std::pair<std::string_view, std::string_view>> texts = {}, std::initializer_list<std::pair<std::string_view, int32_t>> integers = {}) {
    action::nodes::Apply(r, Action{AddNode{definition, action::nodes::OpNode{std::move(node)}}});
    const auto op = r.get<const NetworkNodes>(definition).Last;
    for (const auto [input, text] : texts) action::nodes::Apply(r, Action{SetOpInput{op, std::string{input}, std::make_unique<action::nodes::OpInputValue>(std::string{text})}});
    for (const auto [input, value] : integers) action::nodes::Apply(r, Action{SetOpInput{op, std::string{input}, std::make_unique<action::nodes::OpInputValue>(action::nodes::OpLiteral{mx::Type::Integer, {std::bit_cast<uint32_t>(value)}})}});
    return op;
}
uint16_t Out(const state::Registry &r, state::Entity node) { return InputCountOf(r, node); }
void Wire(state::Registry &r, state::Entity definition, state::Entity from, state::Entity to, uint16_t socket) {
    action::nodes::Apply(r, Action{Link{definition, from, Out(r, from), to, socket}});
}

void SplitBySolid(state::Registry &r, state::Entity definition) {
    const auto outside = Inner(r, definition, KindOf<Boolean>);
    const auto inside = Inner(r, definition, KindOf<Boolean>);
    r.replace<Boolean>(outside, Boolean{.Operation = BooleanOperation::Difference});
    r.replace<Boolean>(inside, Boolean{.Operation = BooleanOperation::Intersection});
    action::nodes::Apply(r, Action{Link{definition, definition, Expose(r, definition, outside, 0u, "Solid"), inside, 0u}});
    action::nodes::Apply(r, Action{Link{definition, definition, Expose(r, definition, outside, 1u, "Cutter"), inside, 1u}});
    Expose(r, definition, outside, Out(r, outside), "Outside");
    Expose(r, definition, inside, Out(r, inside), "Inside");
}
void HullOfObjects(state::Registry &r, state::Entity definition) {
    const auto join = Inner(r, definition, KindOf<Join>);
    const auto hull = Inner(r, definition, KindOf<action::mesh::ConvexHull>);
    Wire(r, definition, join, hull, 0u);
    Expose(r, definition, join, 0u, "Mesh 1");
    Expose(r, definition, join, 0u, "Mesh 2");
    Expose(r, definition, hull, Out(r, hull), "Hull");
}
// A subdivide cutting the edge rings through the selected edges Cuts times and selecting the new loops, with the definition's Mesh, Edges and Cuts ports.
state::Entity CutRings(state::Registry &r, state::Entity definition) {
    const auto selected = InnerOp(r, definition, "selection");
    const auto ring = InnerOp(r, definition, "edgering");
    const auto cut = Inner(r, definition, KindOf<action::mesh::Subdivide>);
    r.replace<action::mesh::Subdivide>(cut, action::mesh::Subdivide{.NewLoopsOnly = true});
    Wire(r, definition, selected, ring, 0u);
    Wire(r, definition, ring, cut, 1u);
    Expose(r, definition, cut, 0u, "Mesh");
    Expose(r, definition, selected, 0u, "Edges");
    Expose(r, definition, cut, ParameterSocket<&action::mesh::Subdivide::Cuts>, "Cuts");
    return cut;
}
void LoopCut(state::Registry &r, state::Entity definition) {
    const auto cut = CutRings(r, definition);
    Expose(r, definition, cut, Out(r, cut), "Mesh");
}
void LoopCutAndSlide(state::Registry &r, state::Entity definition) {
    const auto cut = CutRings(r, definition);
    const auto slide = Inner(r, definition, KindOf<EdgeSlide>);
    r.emplace<PublishSelection>(cut);
    r.emplace<InlineSelection>(slide, InlineSelection{.Domain = Element::Edge, .Published = true});
    r.replace<EdgeSlide>(slide, EdgeSlide{.Factor = 0.f});
    Wire(r, definition, cut, slide, 0u);
    Expose(r, definition, slide, ParameterSocket<&EdgeSlide::Factor>, "Factor");
    Expose(r, definition, slide, ParameterSocket<&EdgeSlide::Direction>, "Direction");
    Expose(r, definition, slide, ParameterSocket<&EdgeSlide::Reference>, "Reference");
    Expose(r, definition, slide, Out(r, slide), "Mesh");
}
void FillHoles(state::Registry &r, state::Entity definition) {
    const auto boundary = InnerOp(r, definition, "boundary");
    const auto fill = Inner(r, definition, KindOf<action::mesh::Fill>);
    r.replace<action::mesh::Fill>(fill, action::mesh::Fill{.Sides = 4u});
    Wire(r, definition, boundary, fill, 1u);
    Expose(r, definition, fill, 0u, "Mesh");
    Expose(r, definition, fill, ParameterSocket<&action::mesh::Fill::Sides>, "Sides");
    Expose(r, definition, fill, Out(r, fill), "Mesh");
}
// A bisect feeding a fill of its output's boundary edges on the plane, with the definition's Mesh, Point and Normal ports.
struct Capped {
    state::Entity Bisect, Fill;
    uint16_t Point, Normal;
};
// Links the definition's Point and Normal ports into the Plane Side op's plane.
void LinkPlane(state::Registry &r, state::Entity definition, const Capped &capped, state::Entity side) {
    const auto &declared = *DefinitionOf(r.get<const Op>(side).Node);
    action::nodes::Apply(r, Action{Link{definition, definition, capped.Point, side, *InputSocket(declared, "point")}});
    action::nodes::Apply(r, Action{Link{definition, definition, capped.Normal, side, *InputSocket(declared, "normal")}});
}
Capped CappedBisect(state::Registry &r, state::Entity definition, const Bisect &parameters) {
    const auto bisect = Inner(r, definition, KindOf<Bisect>);
    const auto on = InnerOp(r, definition, "planeside");
    // An edge lies on the plane when both of its vertices do.
    const auto edges = InnerOp(r, definition, "convertdomain", {{"source", "vertex"}, {"target", "edge"}, {"mode", "all"}});
    const auto boundary = InnerOp(r, definition, "boundary");
    const auto both = InnerOp(r, definition, "and");
    const auto fill = Inner(r, definition, KindOf<action::mesh::Fill>);
    r.replace<Bisect>(bisect, parameters);
    Wire(r, definition, bisect, fill, 0u);
    // The and op's inputs are its first two sockets.
    Wire(r, definition, on, edges, 0u);
    Wire(r, definition, edges, both, 0u);
    Wire(r, definition, boundary, both, 1u);
    Wire(r, definition, both, fill, 1u);
    Expose(r, definition, bisect, 0u, "Mesh");
    const Capped capped{bisect, fill, Expose(r, definition, bisect, ParameterSocket<&Bisect::Point>, "Point"), Expose(r, definition, bisect, ParameterSocket<&Bisect::Normal>, "Normal")};
    LinkPlane(r, definition, capped, on);
    return capped;
}
void BisectFill(state::Registry &r, state::Entity definition) {
    const auto capped = CappedBisect(r, definition, Bisect{.ClearInner = true});
    Expose(r, definition, capped.Bisect, ParameterSocket<&Bisect::ClearInner>, "Clear Inner");
    Expose(r, definition, capped.Bisect, ParameterSocket<&Bisect::ClearOuter>, "Clear Outer");
    Expose(r, definition, capped.Fill, Out(r, capped.Fill), "Mesh");
}
// The region where the plane crosses the mesh: the side behind it capped on the plane, then its vertices behind the plane deleted.
void CrossSection(state::Registry &r, state::Entity definition) {
    const auto capped = CappedBisect(r, definition, Bisect{.Normal = {0, 0, 1}, .ClearOuter = true});
    // Side 1 is behind the plane.
    const auto behind = InnerOp(r, definition, "planeside", {}, {{"side", 1}});
    const auto clear = Inner(r, definition, KindOf<action::mesh::Delete>);
    LinkPlane(r, definition, capped, behind);
    Wire(r, definition, capped.Fill, clear, 0u);
    Wire(r, definition, behind, clear, 1u);
    Expose(r, definition, clear, Out(r, clear), "Section");
}
// The side behind the plane cleared, then joined to its reflection with reversed winding and welded where the halves meet on the plane.
void Symmetrize(state::Registry &r, state::Entity definition) {
    const auto bisect = Inner(r, definition, KindOf<Bisect>);
    const auto mirror = Inner(r, definition, KindOf<TransformElements>);
    const auto flip = Inner(r, definition, KindOf<action::mesh::FlipNormals>);
    const auto join = Inner(r, definition, KindOf<Join>);
    const auto weld = Inner(r, definition, KindOf<action::mesh::Merge>);
    r.replace<Bisect>(bisect, Bisect{.ClearInner = true});
    r.replace<TransformElements>(mirror, TransformElements{.Mirror = {1, 0, 0}});
    // The cut snaps the vertices it keeps on the plane onto it, so each meets its reflection within rounding.
    r.replace<action::mesh::Merge>(weld, action::mesh::Merge{.Mode = action::mesh::MergeMode::ByDistance, .Distance = 1e-5f});
    Wire(r, definition, bisect, join, 0u);
    Wire(r, definition, bisect, mirror, 0u);
    Wire(r, definition, mirror, flip, 0u);
    Wire(r, definition, flip, join, 0u);
    Wire(r, definition, join, weld, 0u);
    Expose(r, definition, bisect, 0u, "Mesh");
    action::nodes::Apply(r, Action{Link{definition, definition, Expose(r, definition, bisect, ParameterSocket<&Bisect::Point>, "Point"), mirror, ParameterSocket<&TransformElements::Pivot>}});
    action::nodes::Apply(r, Action{Link{definition, definition, Expose(r, definition, bisect, ParameterSocket<&Bisect::Normal>, "Normal"), mirror, ParameterSocket<&TransformElements::Mirror>}});
    Expose(r, definition, weld, Out(r, weld), "Mesh");
}
} // namespace

const std::array<Macro, MacroCount> Macros{{
    {"Split by Solid", SplitBySolid},
    {"Hull of Objects", HullOfObjects},
    {"Loop Cut", LoopCut},
    {"Loop Cut and Slide", LoopCutAndSlide},
    {"Fill Holes", FillHoles},
    {"Bisect Fill", BisectFill},
    {"Cross Section", CrossSection},
    {"Symmetrize", Symmetrize},
}};

namespace {
struct Built {
    std::unique_ptr<state::Registry> Registry;
    std::array<state::Entity, MacroCount> Definitions;
};
// Each built-in definition is built once, in Macros order, so its node entities are the same in every session.
const Built &Builtins() {
    static const Built built = [] {
        Built b{std::make_unique<state::Registry>(), {}};
        auto &r = *b.Registry;
        r.Resources.emplace<LibraryLiterals>();
        for (size_t i = 0; i < Macros.size(); ++i) {
            const auto definition = b.Definitions[i] = r.create();
            r.emplace<SubnetDefinition>(definition);
            r.emplace<NetworkNodes>(definition);
            r.emplace<SubnetInterface>(definition);
            r.emplace<Name>(definition, Macros[i].Name);
            Macros[i].Define(r, definition);
        }
        // A built-in definition's page selects nothing.
        r.clear<NetworkActive>();
        return b;
    }();
    return built;
}
} // namespace

const state::Registry &Library() { return *Builtins().Registry; }
DefinitionRef BuiltinDefinition(std::string_view name) {
    const auto found = std::ranges::find(Macros, name, &Macro::Name);
    return found == Macros.end() ? DefinitionRef{} : DefinitionRef{&Library(), Builtins().Definitions[size_t(found - Macros.begin())]};
}
} // namespace procedure
