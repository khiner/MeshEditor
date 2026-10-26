#pragma once

#include "state/Slot.h"

namespace action {
struct DragFieldStart;
struct Errors;
namespace mesh {
struct Delete;
struct Merge;
struct Extrude;
struct Duplicate;
struct Split;
struct Dissolve;
struct Subdivide;
struct Triangulate;
struct Decimate;
struct TrisToQuads;
struct Poke;
struct FlipNormals;
struct RecalculateNormals;
struct EdgeSplit;
struct Inset;
struct Spin;
struct ExtrudeRepeat;
struct Bisect;
struct Symmetrize;
struct Solidify;
struct Wireframe;
struct FillHoles;
struct ConvexHull;
struct Rip;
struct Bevel;
struct SmoothVertices;
struct ShrinkFatten;
struct Randomize;
struct SpaceEvenly;
struct RelaxEdgeLoops;
struct CurveBetweenSelected;
struct Circularize;
struct MakePlanarFaces;
struct SnapSymmetry;
struct SplitNonplanarFaces;
struct SplitConcaveFaces;
struct ConnectVertices;
struct Fill;
struct BridgeEdgeLoops;
struct GridFill;
struct EdgeRotate;
struct BeautifyFaces;
struct Unsubdivide;
struct LoopCut;
struct RotateUVs;
struct ReverseUVs;
struct RotateColors;
struct ReverseColors;
} // namespace mesh

using ActionSchema = state::Schema<state::ActionBase, state::ActionCount, DragFieldStart, Errors, mesh::Delete, mesh::Merge, mesh::Extrude, mesh::Duplicate, mesh::Split, mesh::Dissolve, mesh::Subdivide, mesh::Triangulate, mesh::Decimate, mesh::TrisToQuads, mesh::Poke, mesh::FlipNormals, mesh::RecalculateNormals, mesh::EdgeSplit, mesh::Inset, mesh::Spin, mesh::ExtrudeRepeat, mesh::Bisect, mesh::Symmetrize, mesh::Solidify, mesh::Wireframe, mesh::FillHoles, mesh::ConvexHull, mesh::Rip, mesh::Bevel, mesh::SmoothVertices, mesh::ShrinkFatten, mesh::Randomize, mesh::SpaceEvenly, mesh::RelaxEdgeLoops, mesh::CurveBetweenSelected, mesh::Circularize, mesh::MakePlanarFaces, mesh::SnapSymmetry, mesh::SplitNonplanarFaces, mesh::SplitConcaveFaces, mesh::ConnectVertices, mesh::Fill, mesh::BridgeEdgeLoops, mesh::GridFill, mesh::EdgeRotate, mesh::BeautifyFaces, mesh::Unsubdivide, mesh::LoopCut, mesh::RotateUVs, mesh::ReverseUVs, mesh::RotateColors, mesh::ReverseColors>;
} // namespace action
template<typename T>
    requires action::ActionSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = action::ActionSchema::Of<T>();
