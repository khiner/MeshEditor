#pragma once

#include "Field.h"
#include "action/ActionSchema.h"
#include "mesh/TopologyOperations.h"
#include "numeric/vec2.h"
#include "state/Entity.h"
#include <numbers>

namespace action::mesh {
struct Hide {
    bool Unselected{false};
};
struct Reveal {
    bool Select{true};
};
// Match delete topology op values.
enum class DeleteMode : uint32_t { Vertices,
                                   Edges,
                                   Faces,
                                   OnlyEdgesAndFaces,
                                   OnlyFaces,
                                   Loose };
struct Delete {
    DeleteMode Mode{DeleteMode::Vertices};
};

using MergeMode = GeometryMergeMode;
struct Merge {
    MergeMode Mode{MergeMode::Center};
    float Distance{0.0001f};
};

enum class ExtrudeMode : uint8_t { Region,
                                   Edges,
                                   FacesIndividual,
                                   Vertices };
struct Extrude {
    ExtrudeMode Mode{ExtrudeMode::Region};
    vec3 Offset{};
    float OffsetScale{};
    bool AlongNormals{true};
};
struct Duplicate {};
struct Split {};
using SeparateMode = GeometrySeparateMode;
// Splits selected geometry, connected components, or face material groups into objects.
struct Separate {
    SeparateMode Mode{SeparateMode::Selected};
};

enum class DissolveMode : uint8_t { Vertices,
                                    Edges,
                                    Faces,
                                    Limited,
                                    Degenerate };
struct Dissolve {
    DissolveMode Mode{DissolveMode::Edges};
    float Angle{0.0872665f};
    float Distance{0.0001f};
    bool KeepVertices{};
    bool AllBoundaries{};
    bool DelimitMaterials{};
    bool DelimitSharpEdges{};
    bool DelimitUVs{};
};

struct Subdivide {
    uint32_t Cuts{1};
    float MaxLength{};
    bool NewLoopsOnly{};
};

enum class SplitTarget : uint8_t { Triangle,
                                   Convex,
                                   Planar };
struct SplitFaces {
    SplitTarget Target{SplitTarget::Triangle};
    float Angle{0.0872665f};
};
struct Unsubdivide {
    uint32_t Iterations{2};
};
// Collapse every selected edge whose surviving vertex stays within Tolerance of each face plane it touches, keeping a subset of the vertices.
// Then raise the tolerance until the region's triangle count falls to Ratio of it.
struct Simplify {
    float Tolerance{};
    float Ratio{1.f};
};
struct BeautifyFaces {};
struct TrisToQuads {};
struct Poke {
    float Offset{0.f};
};
struct FlipNormals {};
struct RecalculateNormals {};
struct EdgeSplit {
    bool SelectCopies{};
};
struct Inset {
    float Thickness{0.1f};
    float Depth{0.f};
    bool Individual{false};
    bool Even{true};
};
// Creates an edge from two vertices, fills selected loops of at most Sides edges or every loop when Sides is zero, or forms a face from selected points.
struct Fill {
    uint32_t Sides{};
};

// Appends an Extrude node of Steps steps, each rotating Angle over Steps about Axis through Center and sliding Offset over Steps along Axis.
struct Revolve {
    uint32_t Steps{9};
    float Angle{std::numbers::pi_v<float> / 2};
    vec3 Axis{0.f, 0.f, 1.f};
    vec3 Center{};
    float Offset{0.f};
    bool ViewSpace{false};
};
struct Bisect {
    vec3 Point{};
    vec3 Normal{1.f, 0.f, 0.f};
    bool ClearInner{false};
    bool ClearOuter{false};
    bool ViewSpace{false};
};
// The local axis normal to a mirror plane through the mesh origin.
enum class SnapSymmetryAxis : uint8_t { X,
                                        Y,
                                        Z };
// Match reflected mesh-local positions within Threshold.
// Negative selects the negative side at factor zero.
struct SnapSymmetry {
    SnapSymmetryAxis Axis{SnapSymmetryAxis::X};
    float Threshold{0.05f}, Factor{0.5f};
    bool Negative{false}, Center{true};
};
struct Solidify {
    float Thickness{0.1f};
};
struct Wireframe {
    float Thickness{0.02f};
    float Offset{0.f};
    bool Replace{true}, Boundary{true}, Even{true}, Relative{false};
};
// Splits faces along chords between their consecutive selected vertices.
struct ConnectVertices {};
// Cuts every edge crossing the screen segment from Start to End, in viewport pixels.
struct Knife {
    vec2 Start, End;
};

struct BridgeEdgeLoops {};
// Fills one closed loop of selected boundary or loose edges with a grid of quads, Span edges along its first side.
struct GridFill {
    uint32_t Span{0};
};
// Replaces the selection with the convex hull of the selected vertices, keeping the selected edges and vertices that unselected geometry uses.
struct ConvexHull {};
struct EdgeRotate {};
struct Bevel {
    float Width{0.1f};
    uint32_t Segments{1};
    bool Vertices{false};
};
struct SmoothVertices {
    float Factor{0.5f};
    uint32_t Repeat{1};
    bool X{true}, Y{true}, Z{true};
};
enum class EdgeLoopInterpolation : uint8_t { Cubic,
                                             Linear };
struct SpaceEvenly {
    EdgeLoopInterpolation Interpolation{EdgeLoopInterpolation::Cubic};
    float Factor{1.f};
    bool X{true}, Y{true}, Z{true};
};
struct RelaxEdgeLoops {
    EdgeLoopInterpolation Interpolation{EdgeLoopInterpolation::Cubic};
    uint32_t Iterations{1};
    bool EvenSpacing{true};
};
enum class CircleFit : uint8_t { LeastSquares,
                                 Contract };
// Radius zero fits the source radius.
struct Circularize {
    CircleFit Method{CircleFit::LeastSquares};
    float Factor{1.f}, Radius{0.f}, Angle{0.f};
    bool Regular{true}, X{true}, Y{true}, Z{true};
};
enum class CurveElevation : uint8_t { None,
                                      Raise,
                                      Lower };
// Interpolate unselected loop vertices through selected control points.
// Extend includes the remaining loop, preserving open endpoints and limiting sparse closed selections.
struct CurveBetweenSelected {
    EdgeLoopInterpolation Interpolation{EdgeLoopInterpolation::Cubic};
    CurveElevation Elevation{CurveElevation::None};
    float Factor{1.f};
    bool Extend{false}, Regular{true}, X{true}, Y{true}, Z{true};
};
struct ShrinkFatten {
    float Distance{0.1f};
    bool Even{false};
};
// Blends selected vertices toward their average local radius about the shared selection center.
struct ToSphere {
    float Factor{1.f};
};
struct PushPull {
    float Distance{0.1f};
};
// Axis is the shear plane's normal and AxisOrtho the displacement direction.
// Local uses the active object orientation around the shared selection center.
enum class ShearAxis : uint8_t { X,
                                 Y,
                                 Z };
struct Shear {
    float Angle{0.2f};
    ShearAxis Axis{ShearAxis::Z}, AxisOrtho{ShearAxis::X};
    bool Local{false};
};
// AutoRange computes Min and Max from each edited mesh's selected vertices after OffsetAngle.
struct Warp {
    float Angle{2 * std::numbers::pi_v<float>}, OffsetAngle{0.f};
    vec3 Center{};
    bool AutoRange{true};
    float Min{-1.f}, Max{1.f};
};
// Bend along the rolled view X axis from Center over Radius world units.
// Clamp fixes the start side and rigidly rotates the end side.
struct Bend {
    float Angle{0.78539816f}, Radius{1.f}, OffsetAngle{0.f};
    vec3 Center{};
    bool Clamp{true};
};
// Seed and canonical vertex handle determine offsets independently of selection.
struct Randomize {
    float Amount{0.1f}, Uniform{0.f}, Normal{0.f};
    uint32_t Seed{0u};
};
// Both append a node whose Reference vertex is the mesh's active vertex in Vertex mode.
// Choose the incident edge nearest Direction in world space.
// Even uses the edge length of the Reference vertex, or else of the first selected vertex.
struct VertexSlide {
    float Factor{0.5f};
    vec3 Direction{1, 0, 0};
    bool Even{false}, Flipped{false}, Clamp{true};
    bool ViewSpace{false};
};
// Direction selects the world-space positive side at the Reference vertex, or else at the first selected vertex.
// Even uses the rail length at that vertex.
struct EdgeSlide {
    float Factor{0.5f};
    vec3 Direction{1, 0, 0};
    bool Even{false}, Flipped{false}, Clamp{true};
    bool ViewSpace{false};
};
struct MakePlanarFaces {
    float Factor{1.f};
    uint32_t Repeat{1};
};
enum class FlattenMethod : uint8_t { LeastSquares,
                                     FaceNormals,
                                     View };
// Flatten face regions and remaining edge components independently.
struct Flatten {
    FlattenMethod Method{FlattenMethod::LeastSquares};
    float Factor{1.f};
    bool X{true}, Y{true}, Z{true};
};
enum class CornerAttribute : uint8_t { UVs,
                                       Colors };
enum class CornerCycle : uint8_t { Forward,
                                   Backward,
                                   Reverse };
struct CycleCornerAttribute {
    CornerAttribute Attribute{CornerAttribute::UVs};
    CornerCycle Cycle{CornerCycle::Forward};
    uint32_t UVSet{0};
};
// Combine two selected objects into the active mesh in its local space, preserving the other object.
struct Boolean {
    BooleanOperation Operation{BooleanOperation::Difference};
};

using Action = std::variant<
    Delete, Merge, Extrude, Duplicate, Split, Separate, Dissolve, Subdivide, SplitFaces, TrisToQuads, Poke, FlipNormals, EdgeSplit, Inset, Fill,
    Revolve, Bisect, Solidify, ConnectVertices, Knife, BridgeEdgeLoops, GridFill, ConvexHull, EdgeRotate, Bevel, SmoothVertices, SpaceEvenly, RelaxEdgeLoops, CurveBetweenSelected, Circularize, MakePlanarFaces, Flatten, Wireframe, CycleCornerAttribute, ShrinkFatten, ToSphere, PushPull, RecalculateNormals, BeautifyFaces, Shear, Warp, Bend, Randomize, VertexSlide, EdgeSlide, Unsubdivide, SnapSymmetry, Hide, Reveal, Simplify, Boolean>;
// Placement-drag operators latch a translate for the drag that follows and manage their own previews.
constexpr bool Places(const auto &a) {
    using A = std::decay_t<decltype(a)>;
    if constexpr (std::same_as<A, EdgeSplit>) return a.SelectCopies;
    else return std::same_as<A, Extrude> || std::same_as<A, Duplicate>;
}

void Apply(state::Registry &, const Action &);
// Whether a later step of a stage of the action edits the nodes its first step appended, since only the action sets their parameters.
bool Patches(const Action &);
// The edit-mode meshes with a selection in the viewport's edit element domain, which the operators edit.
std::vector<state::Entity> SelectedEditMeshes(const state::Registry &);
} // namespace action::mesh

template<> inline constexpr field::SpecTable<action::mesh::Delete> Specs<action::mesh::Delete>{{&action::mesh::Delete::Mode, {.Help = "Elements to delete, including the faces of deleted vertices or edges. Loose deletes only selected wire edges and isolated vertices."}}};
template<> inline constexpr field::SpecTable<action::mesh::Merge> Specs<action::mesh::Merge>{
    {&action::mesh::Merge::Mode, {.Help = "Merge target: selection center, first or last selected vertex, each connected run's center, or each pair within Distance"}},
    {&action::mesh::Merge::Distance, {.Min = 0, .Max = 10, .Speed = 0.0001f, .Digits = 4, .Help = "Largest vertex separation that By Distance merges, in local units"}},
};
template<> inline constexpr field::SpecTable<action::mesh::Extrude> Specs<action::mesh::Extrude>{
    {&action::mesh::Extrude::Mode, {.Help = "Elements to extrude: selected faces as one region, each face individually, selected edges, or selected vertices"}},
    {&action::mesh::Extrude::Offset, {.Speed = .01f, .Help = "Direction multiplied by Offset Scale when Along Normals is disabled. A connected field supplies directions on input vertices, or input faces for Individual Faces."}},
    {&action::mesh::Extrude::OffsetScale, {.Speed = .01f, .Help = "Displacement multiplier, or distance for a unit direction. Fields use the same input domain as Offset."}},
    {&action::mesh::Extrude::AlongNormals, {.Help = "Use the input region's average face normal, individual face normals, or vertex normals. A connected Offset field overrides this default."}},
};
template<> inline constexpr field::SpecTable<action::mesh::Dissolve> Specs<action::mesh::Dissolve>{
    {&action::mesh::Dissolve::Mode, {.Help = "Elements to dissolve. Limited dissolves edges flatter than Angle, and Degenerate collapses edges and faces smaller than Distance."}},
    {&action::mesh::Dissolve::Angle, {.Min = 0, .Max = std::numbers::pi_v<float>, .Unit = FieldUnit::Radians, .Help = "Largest angle between adjacent faces that Limited dissolves"}},
    {&action::mesh::Dissolve::Distance, {.Min = 0, .Max = 10, .Speed = 0.0001f, .Digits = 4, .Help = "Largest edge length that Degenerate collapses, in local units"}},
    {&action::mesh::Dissolve::KeepVertices, {.Help = "Keep vertices left between two remaining edges when dissolving edges"}},
    {&action::mesh::Dissolve::AllBoundaries, {.Help = "In Limited, dissolve every selected vertex between two remaining edges, whatever their angle"}},
    {&action::mesh::Dissolve::DelimitMaterials, {.Help = "In Limited, keep edges between faces of different materials"}},
    {&action::mesh::Dissolve::DelimitSharpEdges, {.Help = "In Limited, keep edges marked sharp"}},
    {&action::mesh::Dissolve::DelimitUVs, {.Help = "In Limited, keep edges where the UV coordinates on either side differ"}},
};
template<> inline constexpr field::SpecTable<action::mesh::Subdivide> Specs<action::mesh::Subdivide>{
    {&action::mesh::Subdivide::Cuts, {.Min = 1, .Help = "Number of cuts in each selected edge"}},
    {&action::mesh::Subdivide::MaxLength, {.Min = 0, .Speed = 0.01f, .Help = "Longest piece a selected edge is cut into, in place of the cut count when positive"}},
    {&action::mesh::Subdivide::NewLoopsOnly, {.Help = "Select only the new loops across the cut edges"}},
};
template<> inline constexpr field::SpecTable<action::mesh::Fill> Specs<action::mesh::Fill>{{&action::mesh::Fill::Sides, {.Min = 0, .Help = "Largest number of edges in a loop to fill. Zero fills every loop."}}};
template<> inline constexpr field::SpecTable<action::mesh::SpaceEvenly> Specs<action::mesh::SpaceEvenly>{{&action::mesh::SpaceEvenly::Factor, {.Min = 0, .Max = 1}}};
template<> inline constexpr field::SpecTable<action::mesh::Flatten> Specs<action::mesh::Flatten>{{&action::mesh::Flatten::Factor, {.Min = 0, .Max = 1}}};
template<> inline constexpr field::SpecTable<action::mesh::RelaxEdgeLoops> Specs<action::mesh::RelaxEdgeLoops>{{&action::mesh::RelaxEdgeLoops::Iterations, {.Min = 1, .Max = 1000}}};
template<> inline constexpr field::SpecTable<action::mesh::SnapSymmetry> Specs<action::mesh::SnapSymmetry>{
    {&action::mesh::SnapSymmetry::Threshold, {.Min = 0, .Max = 10, .Speed = 0.001f, .Digits = 4}},
    {&action::mesh::SnapSymmetry::Factor, {.Min = 0, .Max = 1}},
};
template<> inline constexpr field::SpecTable<action::mesh::Unsubdivide> Specs<action::mesh::Unsubdivide>{{&action::mesh::Unsubdivide::Iterations, {.Min = 1, .Max = 1000}}};
template<> inline constexpr field::SpecTable<action::mesh::Simplify> Specs<action::mesh::Simplify>{{&action::mesh::Simplify::Tolerance, {.Min = 0, .Help = "Largest distance a collapsed vertex may move from each of its face planes"}}, {&action::mesh::Simplify::Ratio, {.Min = 0, .Max = 1, .Help = "Fraction of the selected region's triangles to keep after the tolerance pass"}}};
template<> inline constexpr field::SpecTable<action::mesh::Poke> Specs<action::mesh::Poke>{{&action::mesh::Poke::Offset, {.Min = -100, .Max = 100, .Speed = 0.01f, .Help = "Distance each center vertex moves along its face normal, in local units"}}};
template<> inline constexpr field::SpecTable<action::mesh::EdgeSplit> Specs<action::mesh::EdgeSplit>{{&action::mesh::EdgeSplit::SelectCopies, {.Help = "Select only the first copy at each split vertex, the side a rip moves"}}};
template<> inline constexpr field::SpecTable<action::mesh::Inset> Specs<action::mesh::Inset>{
    {&action::mesh::Inset::Thickness, {.Min = 0, .Max = 100, .Speed = 0.01f, .Help = "Inset width from the face boundary, in local units"}},
    {&action::mesh::Inset::Depth, {.Min = -100, .Max = 100, .Speed = 0.01f, .Help = "Distance the inset faces move along their normals, in local units"}},
    {&action::mesh::Inset::Individual, {.Help = "Inset each selected face individually instead of the selection as one region"}},
    {&action::mesh::Inset::Even, {.Help = "Keep the inset width even across corners"}},
};
template<> inline constexpr field::SpecTable<action::mesh::Revolve> Specs<action::mesh::Revolve>{
    {&action::mesh::Revolve::Steps, {.Min = 1, .Help = "Number of extrusion steps, which divide the rotation and the displacement evenly. More than one step, or a rotation, needs Region mode and unlinked Offset inputs."}},
    {&action::mesh::Revolve::Angle, {.Min = -2 * std::numbers::pi_v<float>, .Max = 2 * std::numbers::pi_v<float>, .Unit = FieldUnit::Radians, .Help = "Total rotation about Axis across all steps"}},
    {&action::mesh::Revolve::Axis, {.Speed = 0.01f, .Help = "Rotation axis direction in local space"}},
    {&action::mesh::Revolve::Center, {.Speed = 0.01f, .Help = "Point the axis passes through, in local units"}},
    {&action::mesh::Revolve::Offset, {.Min = -100, .Max = 100, .Speed = 0.01f, .Help = "Total distance along Axis across all steps, in local units"}},
};
template<> inline constexpr field::SpecTable<action::mesh::Bisect> Specs<action::mesh::Bisect>{
    {&action::mesh::Bisect::Point, {.Speed = 0.01f, .Help = "Point on the cutting plane, in local units"}},
    {&action::mesh::Bisect::Normal, {.Speed = 0.01f, .Help = "Cutting plane normal in local space. A zero normal faults the node."}},
    {&action::mesh::Bisect::ClearInner, {.Help = "Delete the vertices behind the plane, opposite its normal, with their edges and faces"}},
    {&action::mesh::Bisect::ClearOuter, {.Help = "Delete the vertices in front of the plane, along its normal, with their edges and faces"}},
};
template<> inline constexpr field::SpecTable<action::mesh::Solidify> Specs<action::mesh::Solidify>{{&action::mesh::Solidify::Thickness, {.Min = -100, .Max = 100, .Speed = 0.01f, .Help = "Shell thickness behind the faces, in local units. A negative thickness extends in front of them."}}};
template<> inline constexpr field::SpecTable<action::mesh::Wireframe> Specs<action::mesh::Wireframe>{
    {&action::mesh::Wireframe::Thickness, {.Min = 0, .Max = 100, .Speed = 0.01f, .Help = "Strut width and depth in local units"}},
    {&action::mesh::Wireframe::Offset, {.Min = -1, .Max = 1, .Speed = 0.01f, .Help = "Strut position along the face normals: -1 is fully behind the faces, 0 centered on them, and 1 fully in front"}},
    {&action::mesh::Wireframe::Replace, {.Help = "Replace selected faces with struts instead of adding to them"}},
    {&action::mesh::Wireframe::Boundary, {.Help = "Also add struts along the selection's boundary edges"}},
    {&action::mesh::Wireframe::Even, {.Help = "Keep the strut width even across sharp corners"}},
    {&action::mesh::Wireframe::Relative, {.Help = "Scale each strut's width with the size of its face"}},
};
template<> inline constexpr field::SpecTable<action::mesh::GridFill> Specs<action::mesh::GridFill>{{&action::mesh::GridFill::Span, {.Min = 0}}};
template<> inline constexpr field::SpecTable<action::mesh::Bevel> Specs<action::mesh::Bevel>{
    {&action::mesh::Bevel::Width, {.Min = 0, .Max = 100, .Speed = 0.01f, .Help = "Bevel offset from each beveled element along its adjacent faces, in local units"}},
    {&action::mesh::Bevel::Segments, {.Min = 1, .Help = "Number of faces across each bevel"}},
    {&action::mesh::Bevel::Vertices, {.Help = "Bevel selected vertices instead of selected edges"}},
};
template<> inline constexpr field::SpecTable<action::mesh::SmoothVertices> Specs<action::mesh::SmoothVertices>{
    {&action::mesh::SmoothVertices::Factor, {.Min = -10, .Max = 10, .Speed = 0.01f, .Help = "Fraction of the distance each vertex moves toward its neighbors' mean per iteration. A negative factor moves it away."}},
    {&action::mesh::SmoothVertices::Repeat, {.Min = 1, .Max = 1000, .Help = "Number of smoothing iterations"}},
    {&action::mesh::SmoothVertices::X, {.Help = "Smooth along the local X axis"}},
    {&action::mesh::SmoothVertices::Y, {.Help = "Smooth along the local Y axis"}},
    {&action::mesh::SmoothVertices::Z, {.Help = "Smooth along the local Z axis"}},
};
template<> inline constexpr field::SpecTable<action::mesh::MakePlanarFaces> Specs<action::mesh::MakePlanarFaces>{
    {&action::mesh::MakePlanarFaces::Factor, {.Min = -10, .Max = 10, .Speed = 0.01f}},
    {&action::mesh::MakePlanarFaces::Repeat, {.Min = 1, .Max = 10000}},
};
template<> inline constexpr field::SpecTable<action::mesh::ToSphere> Specs<action::mesh::ToSphere>{{&action::mesh::ToSphere::Factor, {.Min = 0, .Max = 1, .Speed = 0.01f}}};
template<> inline constexpr field::SpecTable<action::mesh::CycleCornerAttribute> Specs<action::mesh::CycleCornerAttribute>{
    {&action::mesh::CycleCornerAttribute::Attribute, {.Help = "The corner attribute to permute"}},
    {&action::mesh::CycleCornerAttribute::Cycle, {.Help = "Shift each value one corner forward or backward, or reverse the order"}},
    {&action::mesh::CycleCornerAttribute::UVSet, {.Min = 0, .Max = 3, .Help = "The UV set to permute"}},
};
template<> inline constexpr field::SpecTable<action::mesh::Shear> Specs<action::mesh::Shear>{{&action::mesh::Shear::Angle, {.Min = -2 * std::numbers::pi_v<float>, .Max = 2 * std::numbers::pi_v<float>, .Unit = FieldUnit::Radians}}};
template<> inline constexpr field::SpecTable<action::mesh::Warp> Specs<action::mesh::Warp>{
    {&action::mesh::Warp::Angle, {.Unit = FieldUnit::Radians}},
    {&action::mesh::Warp::OffsetAngle, {.Unit = FieldUnit::Radians}},
};
template<> inline constexpr field::SpecTable<action::mesh::Bend> Specs<action::mesh::Bend>{
    {&action::mesh::Bend::Angle, {.Unit = FieldUnit::Radians}},
    {&action::mesh::Bend::OffsetAngle, {.Unit = FieldUnit::Radians}},
};
template<> inline constexpr field::SpecTable<action::mesh::Randomize> Specs<action::mesh::Randomize>{
    {&action::mesh::Randomize::Uniform, {.Min = 0, .Max = 1, .Speed = .01f, .Help = "Blend of each offset's length from random at 0 to the full Amount at 1"}},
    {&action::mesh::Randomize::Amount, {.Speed = .01f, .Help = "Largest offset in local units"}},
    {&action::mesh::Randomize::Normal, {.Min = 0, .Max = 1, .Speed = .01f, .Help = "Blend of each offset's direction from random at 0 to the vertex normal at 1"}},
    {&action::mesh::Randomize::Seed, {.Help = "Random seed"}},
};
template<> inline constexpr field::SpecTable<action::mesh::ShrinkFatten> Specs<action::mesh::ShrinkFatten>{
    {&action::mesh::ShrinkFatten::Distance, {.Help = "Distance each vertex moves along its normal, in local units. A negative distance shrinks."}},
    {&action::mesh::ShrinkFatten::Even, {.Help = "Keep the surface thickness even across sharp corners"}},
};
template<> inline constexpr field::SpecTable<action::mesh::SplitFaces> Specs<action::mesh::SplitFaces>{
    {&action::mesh::SplitFaces::Target, {.Help = "Split until each part is a triangle, is convex, or is planar within Angle"}},
    {&action::mesh::SplitFaces::Angle, {.Min = 0, .Max = std::numbers::pi_v<float>, .Unit = FieldUnit::Radians}},
};

template<> inline constexpr field::SpecTable<action::mesh::CurveBetweenSelected> Specs<action::mesh::CurveBetweenSelected>{{&action::mesh::CurveBetweenSelected::Factor, {.Min = 0, .Max = 1}}};

template<> inline constexpr field::SpecTable<action::mesh::Circularize> Specs<action::mesh::Circularize>{
    {&action::mesh::Circularize::Factor, {.Min = 0, .Max = 1}},
    {&action::mesh::Circularize::Radius, {.Min = 0}},
    {&action::mesh::Circularize::Angle, {.Unit = FieldUnit::Radians}},
};
