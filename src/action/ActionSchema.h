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
} // namespace mesh

using ActionSchema = state::Schema<state::ActionBase, state::ActionCount, DragFieldStart, Errors, mesh::Delete, mesh::Merge, mesh::Extrude, mesh::Duplicate, mesh::Split, mesh::Dissolve, mesh::Subdivide, mesh::Triangulate, mesh::Decimate, mesh::TrisToQuads, mesh::Poke, mesh::FlipNormals, mesh::RecalculateNormals, mesh::EdgeSplit, mesh::Inset, mesh::Spin, mesh::ExtrudeRepeat, mesh::Bisect, mesh::Symmetrize, mesh::Solidify, mesh::Wireframe, mesh::FillHoles, mesh::ConvexHull, mesh::Rip, mesh::Bevel, mesh::SmoothVertices, mesh::ShrinkFatten, mesh::Randomize>;
} // namespace action
template<typename T>
    requires action::ActionSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = action::ActionSchema::Of<T>();
