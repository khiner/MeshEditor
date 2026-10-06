#pragma once

#include "Field.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "state/Entity.h"
#include "viewport/RenderView.h"

#include <memory>
#include <variant>

// Edit-mode topology operators over the selected meshes' element selections.
namespace action::mesh {
struct InsetPreviewCache;
struct Hide {
    bool Unselected{false};
};
struct Reveal {
    bool Select{true};
};
// Values match the delete topology ops.
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

// Merges the selected vertices: all into one at their center or at the first or last selected vertex.
// Collapse merges each connected run into its center, and ByDistance merges every pair within Distance.
enum class MergeMode : uint8_t { Center,
                                 First,
                                 Last,
                                 Collapse,
                                 ByDistance };
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
};
// Duplicates the selected faces onto copied vertices and latches a translate for the drag that follows.
struct Duplicate {};
// Detaches the selected faces from the rest of the mesh.
struct Split {};
enum class SeparateMode : uint8_t { Selected,
                                    LooseParts,
                                    Material };
// Splits selected geometry, connected components, or face material groups into objects.
struct Separate {
    SeparateMode Mode{SeparateMode::Selected};
};

// Joins the faces around the selected vertices, across the selected edges, or of each selected region into one face.
// Limited joins across edges flatter than Angle, and Degenerate collapses edges shorter than Distance.
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
// Collapses selected edges toward Ratio of the affected region's triangle count.
// Material/attribute seams and sharp edges stay fixed; unsafe collapses are skipped.
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
// Splits vertices along the selected edges so the faces on each side come apart.
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
    float Angle{1.5707963f};
    vec3 Axis{0.f, 0.f, 1.f};
    vec3 Center{};
    float Offset{0.f};
};
// Extrudes the selection Steps times, each step moving by Offset.
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
};
// Mirrors the mesh's positive side of Axis onto its negative side across the mesh origin and welds the vertices on the plane.
enum class SymmetrizeAxis : uint8_t { X,
                                      Y,
                                      Z };
struct Symmetrize {
    SymmetrizeAxis Axis{SymmetrizeAxis::X};
    bool Negative{false};
};
// Matches nearest reflected positions within Threshold in mesh-local space.
// Factor blends both sides; Negative chooses the negative side at factor zero.
struct SnapSymmetry {
    SymmetrizeAxis Axis{SymmetrizeAxis::X};
    float Threshold{0.05f}, Factor{0.5f};
    bool Negative{false}, Center{true};
};
// Thickens the selected faces by Thickness toward their back with a rim along the region boundary.
struct Solidify {
    float Thickness{0.1f};
};
// Builds solid struts around selected face edges.
struct Wireframe {
    float Thickness{0.02f};
    float Offset{0.f};
    bool Replace{true}, Boundary{true}, Even{true}, Relative{false};
};
// Splits faces along chords between their consecutive selected vertices.
struct ConnectVertices {};
// Cuts every edge crossing the screen segment from Start to End, in pixels of View.
struct Knife {
    vec2 Start, End;
    std::unique_ptr<RenderView> View;
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
// Adds the convex hull of the selected vertices as triangles.
struct ConvexHull {};
struct EdgeRotate {};
// Splits the faces along the selected edges and latches a translate for the split side.
struct Rip {};
// Bevels selected edges or vertices by Width with Segments across the profile.
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
// Fits selected surface boundaries and isolated wire chains to planar circles.
// Radius zero fits the source radius; Angle rotates around the fitted normal.
struct Circularize {
    CircleFit Method{CircleFit::LeastSquares};
    float Factor{1.f}, Radius{0.f}, Angle{0.f};
    bool Regular{true}, X{true}, Y{true}, Z{true};
};
enum class CurveElevation : uint8_t { None,
                                      Raise,
                                      Lower };
// Fits unselected loop vertices through selected control points. Extend includes the
// rest of each loop, keeping open endpoints fixed and limiting sparse closed selections.
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
// Axis is the shear plane normal; AxisOrtho is its displacement direction.
// Local uses the active object's orientation around the shared selection center.
enum class ShearAxis : uint8_t { X,
                                 Y,
                                 Z };
struct Shear {
    float Angle{0.2f};
    ShearAxis Axis{ShearAxis::Z}, AxisOrtho{ShearAxis::X};
    bool Local{false};
};
// Warps around Center in the recorded orientation (world XY by default).
// Automatic bounds use each edited mesh's selected vertices after OffsetAngle.
struct Warp {
    float Angle{6.2831853f}, OffsetAngle{0.f};
    vec3 Center{};
    bool AutoRange{true};
    float Min{-1.f}, Max{1.f};
    std::unique_ptr<quat> Orientation{std::make_unique<quat>()};
};
// Bends from Center along the rolled view's X axis over Radius world units.
// Clamp leaves the start side fixed and rigidly rotates the end side.
struct Bend {
    float Angle{0.78539816f}, Radius{1.f}, OffsetAngle{0.f};
    vec3 Center{};
    bool Clamp{true};
    std::unique_ptr<quat> Orientation{std::make_unique<quat>()};
};
// Local-space random offsets keyed by Seed and canonical vertex handle, so
// changing the selection does not change an existing vertex's random sample.
struct Randomize {
    float Amount{0.1f}, Uniform{0.f}, Normal{0.f};
    uint32_t Seed{0u};
};
// Both slide operations preserve the UV and color values attached to each corner.
// Chooses each vertex's incident edge nearest Direction in world space.
// Even uses the active selected vertex's edge length, or the first selected vertex.
struct VertexSlide {
    float Factor{0.5f};
    vec3 Direction{1, 0, 0};
    bool Even{false}, Flipped{false}, Clamp{true};
};
// Slides selected edge chains along their neighboring faces. Direction chooses
// the positive side in world space; Even uses the active or first selected vertex.
struct EdgeSlide {
    float Factor{0.5f};
    vec3 Direction{1, 0, 0};
    bool Even{false}, Flipped{false}, Clamp{true};
};
// Flattens selected polygons toward their original planes, averaging at shared vertices.
struct MakePlanarFaces {
    float Factor{1.f};
    uint32_t Repeat{1};
};
enum class FlattenMethod : uint8_t { LeastSquares,
                                     FaceNormals,
                                     View };
// Flattens selected face regions and remaining edge components independently.
// ViewNormal is a recorded world-space direction; X/Y/Z constrain local coordinates.
struct Flatten {
    FlattenMethod Method{FlattenMethod::LeastSquares};
    float Factor{1.f};
    vec3 ViewNormal{0, 0, 1};
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

using Action = std::variant<
    Delete, Merge, Extrude, Duplicate, Split, Separate, Dissolve, Subdivide, Triangulate, TrisToQuads, Poke, FlipNormals, EdgeSplit, Inset, Fill, LoopCut,
    Spin, ExtrudeRepeat, Bisect, Symmetrize, Solidify, ConnectVertices, Knife, BridgeEdgeLoops, GridFill, FillHoles, ConvexHull, EdgeRotate, Rip, Bevel, SmoothVertices, SpaceEvenly, RelaxEdgeLoops, CurveBetweenSelected, Circularize, MakePlanarFaces, Flatten, Wireframe, SplitNonplanarFaces, SplitConcaveFaces, RotateUVs, ReverseUVs, RotateColors, ReverseColors, ShrinkFatten, ToSphere, PushPull, RecalculateNormals, BeautifyFaces, Shear, Warp, Bend, Randomize, VertexSlide, EdgeSlide, Unsubdivide, SnapSymmetry, Hide, Reveal, Decimate>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
// Reuses the staged inset's topology when only its continuous parameters move.
// Returns false without writing if topology, targets, or selection changed.
bool UpdateInsetPreview(state::Scene &, state::Entity viewport, const Inset &, InsetPreviewCache &);
} // namespace action::mesh

template<> inline constexpr FieldSpec Spec<action::mesh::Merge, "Distance">{.Min = 0, .Max = 10, .Speed = 0.0001f, .Digits = 4};
template<> inline constexpr FieldSpec Spec<action::mesh::Dissolve, "Angle">{.Min = 0, .Max = 3.14159265f, .Digits = 1, .Unit = FieldUnit::Radians};
template<> inline constexpr FieldSpec Spec<action::mesh::Dissolve, "Distance">{.Min = 0, .Max = 10, .Speed = 0.0001f, .Digits = 4};
template<> inline constexpr FieldSpec Spec<action::mesh::Subdivide, "Cuts">{.Min = 1};
template<> inline constexpr FieldSpec Spec<action::mesh::SpaceEvenly, "Factor">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<action::mesh::Flatten, "Factor">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<action::mesh::RelaxEdgeLoops, "Iterations">{.Min = 1, .Max = 1000};
template<> inline constexpr FieldSpec Spec<action::mesh::SnapSymmetry, "Threshold">{.Min = 0, .Max = 10, .Speed = 0.001f, .Digits = 4};
template<> inline constexpr FieldSpec Spec<action::mesh::SnapSymmetry, "Factor">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<action::mesh::Unsubdivide, "Iterations">{.Min = 1, .Max = 1000};
template<> inline constexpr FieldSpec Spec<action::mesh::Decimate, "Ratio">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<action::mesh::Poke, "Offset">{.Min = -100, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Inset, "Thickness">{.Min = 0, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Inset, "Depth">{.Min = -100, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::LoopCut, "Cuts">{.Min = 1};
template<> inline constexpr FieldSpec Spec<action::mesh::Spin, "Steps">{.Min = 1};
template<> inline constexpr FieldSpec Spec<action::mesh::Spin, "Angle">{.Min = -6.2831853f, .Max = 6.2831853f, .Digits = 1, .Unit = FieldUnit::Radians};
template<> inline constexpr FieldSpec Spec<action::mesh::Spin, "Axis">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Spin, "Center">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Spin, "Offset">{.Min = -100, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::ExtrudeRepeat, "Steps">{.Min = 1};
template<> inline constexpr FieldSpec Spec<action::mesh::ExtrudeRepeat, "Offset">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Bisect, "Point">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Bisect, "Normal">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Solidify, "Thickness">{.Min = -100, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Wireframe, "Thickness">{.Min = 0, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Wireframe, "Offset">{.Min = -1, .Max = 1, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::GridFill, "Span">{.Min = 0};
template<> inline constexpr FieldSpec Spec<action::mesh::FillHoles, "Sides">{.Min = 0};
template<> inline constexpr FieldSpec Spec<action::mesh::Bevel, "Width">{.Min = 0, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Bevel, "Segments">{.Min = 1};
template<> inline constexpr FieldSpec Spec<action::mesh::SmoothVertices, "Factor">{.Min = -10, .Max = 10, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::SmoothVertices, "Repeat">{.Min = 1, .Max = 1000};
template<> inline constexpr FieldSpec Spec<action::mesh::MakePlanarFaces, "Factor">{.Min = -10, .Max = 10, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::MakePlanarFaces, "Repeat">{.Min = 1, .Max = 10000};
template<> inline constexpr FieldSpec Spec<action::mesh::ToSphere, "Factor">{.Min = 0, .Max = 1, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::RotateUVs, "UVSet">{.Min = 0, .Max = 3};
template<> inline constexpr FieldSpec Spec<action::mesh::ReverseUVs, "UVSet">{.Min = 0, .Max = 3};
template<> inline constexpr FieldSpec Spec<action::mesh::Shear, "Angle">{.Min = -6.2831853f, .Max = 6.2831853f, .Digits = 1, .Unit = FieldUnit::Radians};
template<> inline constexpr FieldSpec Spec<action::mesh::Warp, "Angle">{.Digits = 1, .Unit = FieldUnit::Radians};
template<> inline constexpr FieldSpec Spec<action::mesh::Warp, "OffsetAngle">{.Digits = 1, .Unit = FieldUnit::Radians};
template<> inline constexpr FieldSpec Spec<action::mesh::Bend, "Angle">{.Digits = 1, .Unit = FieldUnit::Radians};
template<> inline constexpr FieldSpec Spec<action::mesh::Bend, "OffsetAngle">{.Digits = 1, .Unit = FieldUnit::Radians};
template<> inline constexpr FieldSpec Spec<action::mesh::Randomize, "Uniform">{.Min = 0, .Max = 1, .Speed = .01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Randomize, "Amount">{.Speed = .01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Randomize, "Normal">{.Min = 0, .Max = 1, .Speed = .01f};
template<> inline constexpr FieldSpec Spec<action::mesh::SplitNonplanarFaces, "Angle">{.Min = 0, .Max = 3.14159265f, .Digits = 1, .Unit = FieldUnit::Radians};

template<> inline constexpr FieldSpec Spec<action::mesh::CurveBetweenSelected, "Factor">{.Min = 0, .Max = 1};

template<> inline constexpr FieldSpec Spec<action::mesh::Circularize, "Factor">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<action::mesh::Circularize, "Radius">{.Min = 0};
template<> inline constexpr FieldSpec Spec<action::mesh::Circularize, "Angle">{.Digits = 1, .Unit = FieldUnit::Radians};
