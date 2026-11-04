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
struct SplitFaces;
struct Simplify;
struct TrisToQuads;
struct Poke;
struct FlipNormals;
struct RecalculateNormals;
struct EdgeSplit;
struct Inset;
struct Revolve;
struct Bisect;
struct Solidify;
struct Wireframe;
struct ConvexHull;
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
struct ConnectVertices;
struct Fill;
struct BridgeEdgeLoops;
struct GridFill;
struct EdgeRotate;
struct BeautifyFaces;
struct Unsubdivide;
struct CycleCornerAttribute;
} // namespace mesh

using ActionSchema = state::Schema<state::ActionBase, state::ActionCount, DragFieldStart, Errors, mesh::Delete, mesh::Merge, mesh::Extrude, mesh::Duplicate, mesh::Split, mesh::Dissolve, mesh::Subdivide, mesh::SplitFaces, mesh::Simplify, mesh::TrisToQuads, mesh::Poke, mesh::FlipNormals, mesh::RecalculateNormals, mesh::EdgeSplit, mesh::Inset, mesh::Revolve, mesh::Bisect, mesh::Solidify, mesh::Wireframe, mesh::ConvexHull, mesh::Bevel, mesh::SmoothVertices, mesh::ShrinkFatten, mesh::Randomize, mesh::SpaceEvenly, mesh::RelaxEdgeLoops, mesh::CurveBetweenSelected, mesh::Circularize, mesh::MakePlanarFaces, mesh::SnapSymmetry, mesh::ConnectVertices, mesh::Fill, mesh::BridgeEdgeLoops, mesh::GridFill, mesh::EdgeRotate, mesh::BeautifyFaces, mesh::Unsubdivide, mesh::CycleCornerAttribute>;
} // namespace action
template<typename T>
    requires action::ActionSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = action::ActionSchema::Of<T>();
