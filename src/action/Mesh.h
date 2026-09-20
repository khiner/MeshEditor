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
// Values match the delete topology ops.
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
                                   FacesIndividual };
struct Extrude {
    ExtrudeMode Mode{ExtrudeMode::Region};
};
// Duplicates the selected faces onto copied vertices and latches a translate for the drag that follows.
struct Duplicate {};
// Detaches the selected faces from the rest of the mesh.
struct Split {};
// Moves the selected faces into a new mesh object.
struct Separate {};

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
};

struct Subdivide {
    uint32_t Cuts{1};
};

struct Triangulate {};
struct TrisToQuads {};
// Fans each selected face around a center vertex lifted by Offset along the face normal.
struct Poke {
    float Offset{0.f};
};
struct FlipNormals {};
// Splits vertices along the selected edges so the faces on each side come apart.
struct EdgeSplit {};
// Insets the selected region, or each face on its own, by Thickness and lifts it by Depth.
struct Inset {
    float Thickness{0.1f};
    float Depth{0.f};
    bool Individual{false};
    bool Even{true};
};
// Fills each closed loop of selected boundary edges with a face.
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
// Thickens the selected faces by Thickness toward their back with a rim along the region boundary.
struct Solidify {
    float Thickness{0.1f};
};
// Splits faces along chords between their consecutive selected vertices.
struct ConnectVertices {};
// Cuts every edge crossing the screen segment from Start to End, in pixels of View.
struct Knife {
    vec2 Start, End;
    std::unique_ptr<RenderView> View;
};

// Bridges two closed loops of selected boundary edges with a strip of faces.
struct BridgeEdgeLoops {};
// Fills one closed loop of selected boundary edges with a grid of quads, Span edges along its first side.
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
// Bevels the selected edges, or the selected vertices, by Width with Segments across each strip.
struct Bevel {
    float Width{0.1f};
    uint32_t Segments{1};
    bool Vertices{false};
};

using Action = std::variant<
    Delete, Merge, Extrude, Duplicate, Split, Separate, Dissolve, Subdivide, Triangulate, TrisToQuads, Poke, FlipNormals, EdgeSplit, Inset, Fill, LoopCut,
    Spin, ExtrudeRepeat, Bisect, Symmetrize, Solidify, ConnectVertices, Knife, BridgeEdgeLoops, GridFill, FillHoles, ConvexHull, EdgeRotate, Rip, Bevel>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
// Adopts every entity's preview as its mesh, releasing the base record.
void CommitPreviews(state::Scene &);
} // namespace action::mesh

template<> inline constexpr FieldSpec Spec<action::mesh::Merge, "Distance">{.Min = 0, .Max = 10, .Speed = 0.0001f, .Digits = 4};
template<> inline constexpr FieldSpec Spec<action::mesh::Dissolve, "Angle">{.Min = 0, .Max = 3.14159265f, .Digits = 1, .Unit = FieldUnit::Radians};
template<> inline constexpr FieldSpec Spec<action::mesh::Dissolve, "Distance">{.Min = 0, .Max = 10, .Speed = 0.0001f, .Digits = 4};
template<> inline constexpr FieldSpec Spec<action::mesh::Subdivide, "Cuts">{.Min = 1, .Max = 32};
template<> inline constexpr FieldSpec Spec<action::mesh::Poke, "Offset">{.Min = -100, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Inset, "Thickness">{.Min = 0, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Inset, "Depth">{.Min = -100, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::LoopCut, "Cuts">{.Min = 1, .Max = 32};
template<> inline constexpr FieldSpec Spec<action::mesh::Spin, "Steps">{.Min = 1, .Max = 256};
template<> inline constexpr FieldSpec Spec<action::mesh::Spin, "Angle">{.Min = -6.2831853f, .Max = 6.2831853f, .Digits = 1, .Unit = FieldUnit::Radians};
template<> inline constexpr FieldSpec Spec<action::mesh::Spin, "Axis">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Spin, "Center">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Spin, "Offset">{.Min = -100, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::ExtrudeRepeat, "Steps">{.Min = 1, .Max = 256};
template<> inline constexpr FieldSpec Spec<action::mesh::ExtrudeRepeat, "Offset">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Bisect, "Point">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Bisect, "Normal">{.Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Solidify, "Thickness">{.Min = -100, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::GridFill, "Span">{.Min = 0, .Max = 256};
template<> inline constexpr FieldSpec Spec<action::mesh::FillHoles, "Sides">{.Min = 0, .Max = 1000};
template<> inline constexpr FieldSpec Spec<action::mesh::Bevel, "Width">{.Min = 0, .Max = 100, .Speed = 0.01f};
template<> inline constexpr FieldSpec Spec<action::mesh::Bevel, "Segments">{.Min = 1, .Max = 16};
