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
// Loose removes selected edges without faces and selected isolated vertices.
enum class DeleteMode : uint32_t { Vertices,
                                   Edges,
                                   Faces,
                                   OnlyEdgesAndFaces,
                                   OnlyFaces,
                                   Loose };
struct Delete {
    DeleteMode Mode{DeleteMode::Vertices};
};

// Merge selected vertices at their center, first vertex, or last vertex.
// Collapse merges each connected run at its center; ByDistance merges pairs within Distance.
using MergeMode = GeometryMergeMode;
struct Merge {
    MergeMode Mode{MergeMode::Center};
    float Distance{0.0001f};
};

// Extrudes the selection and latches a translate for the drag that follows.
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
// Duplicates the selected faces onto copied vertices and latches a translate for the drag that follows.
struct Duplicate {};
struct Split {};
using SeparateMode = GeometrySeparateMode;
// Splits selected geometry, connected components, or face material groups into objects.
struct Separate {
    SeparateMode Mode{SeparateMode::Selected};
};

// Dissolve selected topology into faces.
// Limited uses Angle; Degenerate uses Distance.
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
};

struct Triangulate {};
enum class BeautifyMethod : uint8_t { Area,
                                      Angle };
struct Unsubdivide {
    uint32_t Iterations{2};
};
// Reduce selected edges toward Ratio of the region triangle count.
// Preserve material boundaries, attribute discontinuities, and sharp edges; skip unsafe collapses.
struct Decimate {
    float Ratio{0.5f};
};
struct BeautifyFaces {
    BeautifyMethod Method{BeautifyMethod::Area};
};
struct TrisToQuads {};
// Fans each selected face around a center vertex lifted by Offset along the face normal.
struct Poke {
    float Offset{0.f};
};
struct FlipNormals {};
struct RecalculateNormals {
    bool Inside{false};
};
struct EdgeSplit {};
// Insets the selected region, or each face on its own, by Thickness and lifts it by Depth.
struct Inset {
    float Thickness{0.1f};
    float Depth{0.f};
    bool Individual{false};
    bool Even{true};
};
// Creates an edge from two vertices, fills selected loops, or forms a face from selected points.
struct Fill {};
// Cuts the edge ring through the active edge Cuts times.
struct LoopCut {
    uint32_t Cuts{1};
};

// Extrudes the selection Steps times, each step rotating Angle about Axis through Center and sliding Offset along Axis.
struct Spin {
    uint32_t Steps{9};
    float Angle{std::numbers::pi_v<float> / 2};
    vec3 Axis{0.f, 0.f, 1.f};
    vec3 Center{};
    float Offset{0.f};
    bool ViewSpace{false};
};
struct ExtrudeRepeat {
    uint32_t Steps{1};
    vec3 Offset{0.f, 0.f, 1.f};
};
// Cuts the mesh along the plane through Point with normal Normal and deletes the faces on the cleared sides.
struct Bisect {
    vec3 Point{};
    vec3 Normal{1.f, 0.f, 0.f};
    bool ClearInner{false};
    bool ClearOuter{false};
    bool ViewSpace{false};
};
// Mirrors the mesh's positive side of Axis onto its negative side across the mesh origin and welds the vertices on the plane.
enum class SymmetrizeAxis : uint8_t { X,
                                      Y,
                                      Z };
struct Symmetrize {
    SymmetrizeAxis Axis{SymmetrizeAxis::X};
    bool Negative{false};
};
// Match reflected mesh-local positions within Threshold.
// Factor blends both sides; Negative selects the negative side at factor zero.
struct SnapSymmetry {
    SymmetrizeAxis Axis{SymmetrizeAxis::X};
    float Threshold{0.05f}, Factor{0.5f};
    bool Negative{false}, Center{true};
};
// Thickens the selected faces by Thickness toward their back with a rim along the region boundary.
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

// Bridges two selected open chains or closed loops of boundary or loose edges with a strip of faces.
struct BridgeEdgeLoops {};
// Fills one closed loop of selected boundary or loose edges with a grid of quads, Span edges along its first side.
struct GridFill {
    uint32_t Span{0};
};
// Fills every boundary loop of at most Sides edges.
struct FillHoles {
    uint32_t Sides{4};
};
struct ConvexHull {};
struct EdgeRotate {};
// Splits the faces along the selected edges and latches a translate for the split side.
struct Rip {};
struct Bevel {
    float Width{0.1f};
    uint32_t Segments{1};
    bool Vertices{false};
};
// Moves selected vertices toward the mean of their edge neighbors, once per iteration.
struct SmoothVertices {
    float Factor{0.5f};
    uint32_t Repeat{1};
    bool X{true}, Y{true}, Z{true};
};
enum class EdgeLoopInterpolation : uint8_t { Cubic,
                                             Linear };
// Redistributes selected edge chains by arc length, keeping open endpoints and junctions fixed.
struct SpaceEvenly {
    EdgeLoopInterpolation Interpolation{EdgeLoopInterpolation::Cubic};
    float Factor{1.f};
    bool X{true}, Y{true}, Z{true};
};
// Smooths alternating vertices along selected chains, retaining open endpoints and junctions.
struct RelaxEdgeLoops {
    EdgeLoopInterpolation Interpolation{EdgeLoopInterpolation::Cubic};
    uint32_t Iterations{1};
    bool EvenSpacing{true};
};
enum class CircleFit : uint8_t { LeastSquares,
                                 Contract };
// Fit selected boundaries and wire chains to planar circles.
// Radius zero fits the source radius; Angle rotates around the fitted normal.
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
// Offsets selected vertices along their normals, using selected face normals in face mode.
struct ShrinkFatten {
    float Distance{0.1f};
    bool Even{false};
};
// Blends selected vertices toward their average local radius about the shared selection center.
struct ToSphere {
    float Factor{1.f};
};
// Moves selected vertices toward the shared selection center by Distance (negative moves outward).
struct PushPull {
    float Distance{0.1f};
};
// Axis specifies the shear-plane normal; AxisOrtho specifies displacement.
// Local uses the active object orientation around the shared selection center.
enum class ShearAxis : uint8_t { X,
                                 Y,
                                 Z };
struct Shear {
    float Angle{0.2f};
    ShearAxis Axis{ShearAxis::Z}, AxisOrtho{ShearAxis::X};
    bool Local{false};
};
// Warp around Center in the viewport orientation.
// Compute automatic bounds from each edited mesh's selected vertices after OffsetAngle.
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
// Preserve corner UVs and colors in both slide operations.
// Choose the incident edge nearest Direction in world space.
// Even uses the active selected vertex edge length, or the first selected vertex.
struct VertexSlide {
    float Factor{0.5f};
    vec3 Direction{1, 0, 0};
    bool Even{false}, Flipped{false}, Clamp{true};
    bool ViewSpace{false};
};
// Slide selected edge chains along adjacent faces.
// Direction selects the world-space positive side; Even uses the active or first selected vertex.
struct EdgeSlide {
    float Factor{0.5f};
    vec3 Direction{1, 0, 0};
    bool Even{false}, Flipped{false}, Clamp{true};
    bool ViewSpace{false};
};
// Flattens selected polygons toward their original planes, averaging at shared vertices.
struct MakePlanarFaces {
    float Factor{1.f};
    uint32_t Repeat{1};
};
enum class FlattenMethod : uint8_t { LeastSquares,
                                     FaceNormals,
                                     View };
// Flatten face regions and remaining edge components independently.
// X/Y/Z constrain local coordinates.
struct Flatten {
    FlattenMethod Method{FlattenMethod::LeastSquares};
    float Factor{1.f};
    bool X{true}, Y{true}, Z{true};
};
// Splits warped selected polygons along legal diagonals above the normal angle limit.
struct SplitNonplanarFaces {
    float Angle{0.0872665f};
};
struct SplitConcaveFaces {};
struct RotateUVs {
    uint32_t UVSet{0};
    bool CounterClockwise{false};
};
struct ReverseUVs {
    uint32_t UVSet{0};
};
struct RotateColors {
    bool CounterClockwise{false};
};
struct ReverseColors {};
// Combine two selected objects into the active mesh in its local space.
// Preserve the other object.
struct Boolean {
    BooleanOperation Operation{BooleanOperation::Difference};
};

using Action = std::variant<
    Delete, Merge, Extrude, Duplicate, Split, Separate, Dissolve, Subdivide, Triangulate, TrisToQuads, Poke, FlipNormals, EdgeSplit, Inset, Fill, LoopCut,
    Spin, ExtrudeRepeat, Bisect, Symmetrize, Solidify, ConnectVertices, Knife, BridgeEdgeLoops, GridFill, FillHoles, ConvexHull, EdgeRotate, Rip, Bevel, SmoothVertices, SpaceEvenly, RelaxEdgeLoops, CurveBetweenSelected, Circularize, MakePlanarFaces, Flatten, Wireframe, SplitNonplanarFaces, SplitConcaveFaces, RotateUVs, ReverseUVs, RotateColors, ReverseColors, ShrinkFatten, ToSphere, PushPull, RecalculateNormals, BeautifyFaces, Shear, Warp, Bend, Randomize, VertexSlide, EdgeSlide, Unsubdivide, SnapSymmetry, Hide, Reveal, Decimate, Boolean>;

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
template<> inline constexpr field::SpecTable<action::mesh::Subdivide> Specs<action::mesh::Subdivide>{{&action::mesh::Subdivide::Cuts, {.Min = 1, .Help = "Number of cuts in each selected edge"}}};
template<> inline constexpr field::SpecTable<action::mesh::SpaceEvenly> Specs<action::mesh::SpaceEvenly>{{&action::mesh::SpaceEvenly::Factor, {.Min = 0, .Max = 1}}};
template<> inline constexpr field::SpecTable<action::mesh::Flatten> Specs<action::mesh::Flatten>{{&action::mesh::Flatten::Factor, {.Min = 0, .Max = 1}}};
template<> inline constexpr field::SpecTable<action::mesh::RelaxEdgeLoops> Specs<action::mesh::RelaxEdgeLoops>{{&action::mesh::RelaxEdgeLoops::Iterations, {.Min = 1, .Max = 1000}}};
template<> inline constexpr field::SpecTable<action::mesh::SnapSymmetry> Specs<action::mesh::SnapSymmetry>{
    {&action::mesh::SnapSymmetry::Threshold, {.Min = 0, .Max = 10, .Speed = 0.001f, .Digits = 4}},
    {&action::mesh::SnapSymmetry::Factor, {.Min = 0, .Max = 1}},
};
template<> inline constexpr field::SpecTable<action::mesh::Unsubdivide> Specs<action::mesh::Unsubdivide>{{&action::mesh::Unsubdivide::Iterations, {.Min = 1, .Max = 1000}}};
template<> inline constexpr field::SpecTable<action::mesh::Decimate> Specs<action::mesh::Decimate>{{&action::mesh::Decimate::Ratio, {.Min = 0, .Max = 1, .Help = "Fraction of the selected region's triangles to keep"}}};
template<> inline constexpr field::SpecTable<action::mesh::Poke> Specs<action::mesh::Poke>{{&action::mesh::Poke::Offset, {.Min = -100, .Max = 100, .Speed = 0.01f, .Help = "Distance each center vertex moves along its face normal, in local units"}}};
template<> inline constexpr field::SpecTable<action::mesh::Inset> Specs<action::mesh::Inset>{
    {&action::mesh::Inset::Thickness, {.Min = 0, .Max = 100, .Speed = 0.01f, .Help = "Inset width from the face boundary, in local units"}},
    {&action::mesh::Inset::Depth, {.Min = -100, .Max = 100, .Speed = 0.01f, .Help = "Distance the inset faces move along their normals, in local units"}},
    {&action::mesh::Inset::Individual, {.Help = "Inset each selected face individually instead of the selection as one region"}},
    {&action::mesh::Inset::Even, {.Help = "Keep the inset width even across corners"}},
};
template<> inline constexpr field::SpecTable<action::mesh::LoopCut> Specs<action::mesh::LoopCut>{{&action::mesh::LoopCut::Cuts, {.Min = 1}}};
template<> inline constexpr field::SpecTable<action::mesh::Spin> Specs<action::mesh::Spin>{
    {&action::mesh::Spin::Steps, {.Min = 1, .Help = "Number of extrusion steps"}},
    {&action::mesh::Spin::Angle, {.Min = -2 * std::numbers::pi_v<float>, .Max = 2 * std::numbers::pi_v<float>, .Unit = FieldUnit::Radians, .Help = "Total rotation about Axis across all steps"}},
    {&action::mesh::Spin::Axis, {.Speed = 0.01f, .Help = "Rotation axis direction in local space"}},
    {&action::mesh::Spin::Center, {.Speed = 0.01f, .Help = "Point the axis passes through, in local units"}},
    {&action::mesh::Spin::Offset, {.Min = -100, .Max = 100, .Speed = 0.01f, .Help = "Total distance along Axis across all steps, in local units"}},
};
template<> inline constexpr field::SpecTable<action::mesh::ExtrudeRepeat> Specs<action::mesh::ExtrudeRepeat>{
    {&action::mesh::ExtrudeRepeat::Steps, {.Min = 1, .Help = "Number of extrusions"}},
    {&action::mesh::ExtrudeRepeat::Offset, {.Speed = 0.01f, .Help = "Translation per step, in local units"}},
};
template<> inline constexpr field::SpecTable<action::mesh::Bisect> Specs<action::mesh::Bisect>{
    {&action::mesh::Bisect::Point, {.Speed = 0.01f, .Help = "Point on the cutting plane, in local units"}},
    {&action::mesh::Bisect::Normal, {.Speed = 0.01f, .Help = "Cutting plane normal in local space. A zero normal faults the node."}},
    {&action::mesh::Bisect::ClearInner, {.Help = "Delete faces behind the plane, opposite its normal"}},
    {&action::mesh::Bisect::ClearOuter, {.Help = "Delete faces in front of the plane, along its normal"}},
};
template<> inline constexpr field::SpecTable<action::mesh::Symmetrize> Specs<action::mesh::Symmetrize>{
    {&action::mesh::Symmetrize::Axis, {.Help = "Local axis normal to the mirror plane through the mesh origin"}},
    {&action::mesh::Symmetrize::Negative, {.Help = "Mirror the negative side onto the positive instead of the reverse"}},
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
template<> inline constexpr field::SpecTable<action::mesh::FillHoles> Specs<action::mesh::FillHoles>{{&action::mesh::FillHoles::Sides, {.Min = 0, .Help = "Largest number of edges in a boundary loop to fill. Zero fills every hole."}}};
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
template<> inline constexpr field::SpecTable<action::mesh::RotateUVs> Specs<action::mesh::RotateUVs>{{&action::mesh::RotateUVs::UVSet, {.Min = 0, .Max = 3}}};
template<> inline constexpr field::SpecTable<action::mesh::ReverseUVs> Specs<action::mesh::ReverseUVs>{{&action::mesh::ReverseUVs::UVSet, {.Min = 0, .Max = 3}}};
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
template<> inline constexpr field::SpecTable<action::mesh::RecalculateNormals> Specs<action::mesh::RecalculateNormals>{{&action::mesh::RecalculateNormals::Inside, {.Help = "Point normals inside the mesh instead of outside"}}};
template<> inline constexpr field::SpecTable<action::mesh::SplitNonplanarFaces> Specs<action::mesh::SplitNonplanarFaces>{{&action::mesh::SplitNonplanarFaces::Angle, {.Min = 0, .Max = std::numbers::pi_v<float>, .Unit = FieldUnit::Radians}}};

template<> inline constexpr field::SpecTable<action::mesh::CurveBetweenSelected> Specs<action::mesh::CurveBetweenSelected>{{&action::mesh::CurveBetweenSelected::Factor, {.Min = 0, .Max = 1}}};

template<> inline constexpr field::SpecTable<action::mesh::Circularize> Specs<action::mesh::Circularize>{
    {&action::mesh::Circularize::Factor, {.Min = 0, .Max = 1}},
    {&action::mesh::Circularize::Radius, {.Min = 0}},
    {&action::mesh::Circularize::Angle, {.Unit = FieldUnit::Radians}},
};
