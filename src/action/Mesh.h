#pragma once

#include "gpu/MeshTopologyOp.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "state/Entity.h"
#include "viewport/RenderView.h"

#include <memory>
#include <variant>

// Edit-mode topology operators over the selected meshes' element selections.
namespace action::mesh {
// Removes the selected elements, with `Op` naming which dependents go with them.
struct Delete {
    MeshTopologyOp Op{MeshTopologyOp::DeleteVertices};
};

// Merges the selected vertices: all into one at their center or at the first or last selected vertex.
// Collapse merges each connected run into its center, and ByDistance merges every pair within Distance.
struct Merge {
    enum class Mode : uint8_t { Center,
                                First,
                                Last,
                                Collapse,
                                ByDistance };
    Mode Value{Mode::Center};
    float Distance{0.0001f};
};

// Extrudes the selection and latches a translate for the drag that follows.
struct Extrude {
    enum class Mode : uint8_t { Region,
                                Edges,
                                FacesIndividual };
    Mode Value{Mode::Region};
};
// Duplicates the selected faces onto copied vertices and latches a translate for the drag that follows.
struct Duplicate {};
// Detaches the selected faces from the rest of the mesh.
struct Split {};
// Moves the selected faces into a new mesh object.
struct Separate {};

// Joins the faces around the selected vertices, across the selected edges, or of each selected region into one face.
// Limited joins across edges flatter than Angle, and Degenerate collapses edges shorter than Distance.
struct Dissolve {
    enum class Mode : uint8_t { Vertices,
                                Edges,
                                Faces,
                                Limited,
                                Degenerate };
    Mode Value{Mode::Edges};
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
struct Symmetrize {
    uint8_t Axis{0};
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

// The most recent committed mesh operator and its history node, for rerunning it with edited parameters.
struct LastOperation {
    Action Value;
    int Node;
};
} // namespace action::mesh
