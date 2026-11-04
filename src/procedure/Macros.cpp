#include "procedure/Macros.h"
#include "action/Nodes.h"
#include "procedure/Procedure.h"
#include "state/Registry.h"

namespace procedure {
namespace {
using action::nodes::Action, action::nodes::AddNode, action::nodes::ExposeSocket, action::nodes::Link;

state::Entity Inner(state::Registry &r, state::Entity mesh_entity, state::Entity subnet, uint16_t kind, uvec2 position) {
    action::nodes::Apply(r, Action{AddNode{mesh_entity, kind, position, subnet}});
    return r.get<const GeometryProcedure>(mesh_entity).Last;
}
uint16_t Expose(state::Registry &r, state::Entity subnet, state::Entity node, uint16_t socket, const char *name) {
    action::nodes::Apply(r, Action{ExposeSocket{subnet, node, socket, name}});
    return r.get<const SubnetInterface>(subnet).Ports.back().Id;
}
uint16_t Out(const state::Registry &r, state::Entity node) { return InputCount[r.get<const ProcedureNode>(node).Kind]; }
void Wire(state::Registry &r, state::Entity mesh_entity, state::Entity from, state::Entity to, uint16_t socket) {
    action::nodes::Apply(r, Action{Link{mesh_entity, from, Out(r, from), to, socket}});
}

void SplitBySolid(state::Registry &r, state::Entity mesh_entity, state::Entity subnet) {
    const auto outside = Inner(r, mesh_entity, subnet, KindOf<Boolean>, {AppendGap, 2u});
    const auto inside = Inner(r, mesh_entity, subnet, KindOf<Boolean>, {AppendGap, 2u + AppendGap});
    r.replace<Boolean>(outside, Boolean{.Operation = BooleanOperation::Difference});
    r.replace<Boolean>(inside, Boolean{.Operation = BooleanOperation::Intersection});
    action::nodes::Apply(r, Action{Link{mesh_entity, subnet, Expose(r, subnet, outside, 0u, "Solid"), inside, 0u}});
    action::nodes::Apply(r, Action{Link{mesh_entity, subnet, Expose(r, subnet, outside, 1u, "Cutter"), inside, 1u}});
    Expose(r, subnet, outside, Out(r, outside), "Outside");
    Expose(r, subnet, inside, Out(r, inside), "Inside");
}
void HullOfObjects(state::Registry &r, state::Entity mesh_entity, state::Entity subnet) {
    const auto join = Inner(r, mesh_entity, subnet, KindOf<Join>, {AppendGap, 2u});
    const auto hull = Inner(r, mesh_entity, subnet, KindOf<action::mesh::ConvexHull>, {2u * AppendGap, 2u});
    Wire(r, mesh_entity, join, hull, 0u);
    Expose(r, subnet, join, 0u, "Mesh 1");
    Expose(r, subnet, join, 1u, "Mesh 2");
    Expose(r, subnet, hull, Out(r, hull), "Hull");
}
// A subdivide cutting the edge rings through the selected edges Cuts times and selecting the new loops, with the subnet's Mesh, Edges and Cuts ports.
state::Entity CutRings(state::Registry &r, state::Entity mesh_entity, state::Entity subnet) {
    const auto selected = Inner(r, mesh_entity, subnet, KindOf<SelectionAttribute>, {AppendGap, 2u + AppendGap});
    const auto ring = Inner(r, mesh_entity, subnet, KindOf<EdgeRing>, {2u * AppendGap, 2u + AppendGap});
    const auto cut = Inner(r, mesh_entity, subnet, KindOf<action::mesh::Subdivide>, {3u * AppendGap, 2u});
    r.replace<action::mesh::Subdivide>(cut, action::mesh::Subdivide{.NewLoopsOnly = true});
    Wire(r, mesh_entity, selected, ring, 0u);
    Wire(r, mesh_entity, ring, cut, 1u);
    Expose(r, subnet, cut, 0u, "Mesh");
    Expose(r, subnet, selected, 0u, "Edges");
    Expose(r, subnet, cut, ParameterSocket<&action::mesh::Subdivide::Cuts>, "Cuts");
    return cut;
}
void LoopCut(state::Registry &r, state::Entity mesh_entity, state::Entity subnet) {
    const auto cut = CutRings(r, mesh_entity, subnet);
    Expose(r, subnet, cut, Out(r, cut), "Mesh");
}
void LoopCutAndSlide(state::Registry &r, state::Entity mesh_entity, state::Entity subnet) {
    const auto cut = CutRings(r, mesh_entity, subnet);
    const auto slide = Inner(r, mesh_entity, subnet, KindOf<EdgeSlide>, {4u * AppendGap, 2u});
    r.emplace<PublishSelection>(cut);
    r.emplace<InlineSelection>(slide, InlineSelection{.Domain = Element::Edge, .Published = true});
    r.replace<EdgeSlide>(slide, EdgeSlide{.Factor = 0.f});
    Wire(r, mesh_entity, cut, slide, 0u);
    Expose(r, subnet, slide, ParameterSocket<&EdgeSlide::Factor>, "Factor");
    Expose(r, subnet, slide, ParameterSocket<&EdgeSlide::Direction>, "Direction");
    Expose(r, subnet, slide, ParameterSocket<&EdgeSlide::Reference>, "Reference");
    Expose(r, subnet, slide, Out(r, slide), "Mesh");
}
void FillHoles(state::Registry &r, state::Entity mesh_entity, state::Entity subnet) {
    const auto boundary = Inner(r, mesh_entity, subnet, KindOf<Boundary>, {AppendGap, 2u + AppendGap});
    const auto fill = Inner(r, mesh_entity, subnet, KindOf<action::mesh::Fill>, {2u * AppendGap, 2u});
    r.replace<action::mesh::Fill>(fill, action::mesh::Fill{.Sides = 4u});
    Wire(r, mesh_entity, boundary, fill, 1u);
    Expose(r, subnet, fill, 0u, "Mesh");
    Expose(r, subnet, fill, ParameterSocket<&action::mesh::Fill::Sides>, "Sides");
    Expose(r, subnet, fill, Out(r, fill), "Mesh");
}
// A bisect feeding a fill of its output's boundary edges on the plane, with the subnet's Mesh, Point and Normal ports.
struct Capped {
    state::Entity Bisect, Fill;
    uint16_t Point, Normal;
};
Capped CappedBisect(state::Registry &r, state::Entity mesh_entity, state::Entity subnet, const Bisect &parameters) {
    const auto bisect = Inner(r, mesh_entity, subnet, KindOf<Bisect>, {AppendGap, 2u});
    const auto on = Inner(r, mesh_entity, subnet, KindOf<PlaneSide>, {AppendGap, 2u + AppendGap});
    const auto boundary = Inner(r, mesh_entity, subnet, KindOf<Boundary>, {AppendGap, 2u + 2u * AppendGap});
    const auto both = Inner(r, mesh_entity, subnet, KindOf<BooleanMath>, {2u * AppendGap, 2u + AppendGap});
    const auto fill = Inner(r, mesh_entity, subnet, KindOf<action::mesh::Fill>, {3u * AppendGap, 2u});
    r.replace<Bisect>(bisect, parameters);
    r.replace<PlaneSide>(on, PlaneSide{.Domain = AttributeDomain::Edge});
    Wire(r, mesh_entity, bisect, fill, 0u);
    Wire(r, mesh_entity, on, both, 0u);
    Wire(r, mesh_entity, boundary, both, ParameterSocket<&BooleanMath::B>);
    Wire(r, mesh_entity, both, fill, 1u);
    Expose(r, subnet, bisect, 0u, "Mesh");
    const Capped capped{bisect, fill, Expose(r, subnet, bisect, ParameterSocket<&Bisect::Point>, "Point"), Expose(r, subnet, bisect, ParameterSocket<&Bisect::Normal>, "Normal")};
    action::nodes::Apply(r, Action{Link{mesh_entity, subnet, capped.Point, on, ParameterSocket<&PlaneSide::Point>}});
    action::nodes::Apply(r, Action{Link{mesh_entity, subnet, capped.Normal, on, ParameterSocket<&PlaneSide::Normal>}});
    return capped;
}
void BisectFill(state::Registry &r, state::Entity mesh_entity, state::Entity subnet) {
    const auto capped = CappedBisect(r, mesh_entity, subnet, Bisect{.ClearInner = true});
    Expose(r, subnet, capped.Bisect, ParameterSocket<&Bisect::ClearInner>, "Clear Inner");
    Expose(r, subnet, capped.Bisect, ParameterSocket<&Bisect::ClearOuter>, "Clear Outer");
    Expose(r, subnet, capped.Fill, Out(r, capped.Fill), "Mesh");
}
// The region where the plane crosses the mesh: the side behind it capped on the plane, then its vertices behind the plane deleted.
void CrossSection(state::Registry &r, state::Entity mesh_entity, state::Entity subnet) {
    const auto capped = CappedBisect(r, mesh_entity, subnet, Bisect{.Normal = {0, 0, 1}, .ClearOuter = true});
    const auto behind = Inner(r, mesh_entity, subnet, KindOf<PlaneSide>, {3u * AppendGap, 2u + AppendGap});
    const auto clear = Inner(r, mesh_entity, subnet, KindOf<action::mesh::Delete>, {4u * AppendGap, 2u});
    r.replace<PlaneSide>(behind, PlaneSide{.Side = PlaneSideKind::Negative});
    action::nodes::Apply(r, Action{Link{mesh_entity, subnet, capped.Point, behind, ParameterSocket<&PlaneSide::Point>}});
    action::nodes::Apply(r, Action{Link{mesh_entity, subnet, capped.Normal, behind, ParameterSocket<&PlaneSide::Normal>}});
    Wire(r, mesh_entity, capped.Fill, clear, 0u);
    Wire(r, mesh_entity, behind, clear, 1u);
    Expose(r, subnet, clear, Out(r, clear), "Section");
}
// The side behind the plane cleared, then joined to its reflection with reversed winding and welded where the halves meet on the plane.
void Symmetrize(state::Registry &r, state::Entity mesh_entity, state::Entity subnet) {
    const auto bisect = Inner(r, mesh_entity, subnet, KindOf<Bisect>, {AppendGap, 2u});
    const auto mirror = Inner(r, mesh_entity, subnet, KindOf<TransformElements>, {2u * AppendGap, 2u + AppendGap});
    const auto flip = Inner(r, mesh_entity, subnet, KindOf<action::mesh::FlipNormals>, {3u * AppendGap, 2u + AppendGap});
    const auto join = Inner(r, mesh_entity, subnet, KindOf<Join>, {4u * AppendGap, 2u});
    const auto weld = Inner(r, mesh_entity, subnet, KindOf<action::mesh::Merge>, {5u * AppendGap, 2u});
    r.replace<Bisect>(bisect, Bisect{.ClearInner = true});
    r.replace<TransformElements>(mirror, TransformElements{.Mirror = {1, 0, 0}});
    // The cut snaps the vertices it keeps on the plane onto it, so each meets its reflection within rounding.
    r.replace<action::mesh::Merge>(weld, action::mesh::Merge{.Mode = action::mesh::MergeMode::ByDistance, .Distance = 1e-5f});
    Wire(r, mesh_entity, bisect, join, 0u);
    Wire(r, mesh_entity, bisect, mirror, 0u);
    Wire(r, mesh_entity, mirror, flip, 0u);
    Wire(r, mesh_entity, flip, join, 1u);
    Wire(r, mesh_entity, join, weld, 0u);
    Expose(r, subnet, bisect, 0u, "Mesh");
    action::nodes::Apply(r, Action{Link{mesh_entity, subnet, Expose(r, subnet, bisect, ParameterSocket<&Bisect::Point>, "Point"), mirror, ParameterSocket<&TransformElements::Pivot>}});
    action::nodes::Apply(r, Action{Link{mesh_entity, subnet, Expose(r, subnet, bisect, ParameterSocket<&Bisect::Normal>, "Normal"), mirror, ParameterSocket<&TransformElements::Mirror>}});
    Expose(r, subnet, weld, Out(r, weld), "Mesh");
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
} // namespace procedure
