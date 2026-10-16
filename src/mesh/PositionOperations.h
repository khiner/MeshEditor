#pragma once

#include "Range.h"
#include "gpu/VertexPositionEditPushConstants.h"
#include "mesh/GeometrySelection.h"
#include "mesh/MeshClosure.h"
#include <optional>

struct MeshStore;
struct MeshPipelines;

// Selections and traversal exclusions are canonical handles or masks, which operations read on the host.
// World, Center, Direction and reference frames are supplied by the caller.
struct PositionOperationTarget {
    uint32_t StoreId;
    SelectionSource Selection, Excluded;
    Transform World{};
    uint32_t Reference{InvalidOffset};
};
struct PositionWarp {
    quat Orientation{1, 0, 0, 0};
    vec3 Center{};
    float OffsetAngle{}, Min{-1.f}, Max{1.f};
    bool AutoRange{true};
};
struct PositionBend {
    quat Orientation{1, 0, 0, 0};
    vec3 Center{};
    float OffsetAngle{}, Radius{1.f};
};
struct PositionTransform {
    Transform Delta{};
    vec3 Pivot{};
};
struct PositionRandomize {
    float Uniform{}, Normal{};
    uint32_t Seed{};
};
struct PositionSlide {
    vec3 Direction{1, 0, 0};
    bool Even{}, Flipped{};
};
struct PositionSymmetry {
    float Threshold{.05f};
    bool Center{true};
};
struct PositionCurve {
    bool Extend{};
};
struct PositionCircle {
    float Radius{}, Angle{};
};
struct PositionOperationOptions {
    uint32_t Axes{7u}, Flags{};
    vec3 Direction{}, Gradient{}, Center{};
    std::optional<PositionWarp> Warp;
    std::optional<PositionBend> Bend;
    std::optional<PositionRandomize> Randomize;
    std::optional<PositionSlide> Slide, EdgeSlide;
    std::optional<PositionSymmetry> SnapSymmetry;
    std::optional<PositionCurve> Curve;
    std::optional<PositionCircle> Circle;
    std::optional<PositionTransform> Transform;
    std::optional<NamedAttributeRef> Attribute; // A vec3 vertex attribute the positions take, or add with PositionEditAttributeOffset
    NamedAttributeRef Factor{}; // A float vertex attribute holding the factor per vertex of a smooth, shrink/fatten or randomize, or none to read the factor
};
struct PositionOperationChange {
    uint32_t TargetIndex;
    std::vector<uint32_t> Vertices;
    std::vector<Range> Ranges;
};
// Encodes canonical writes and captures their pages.
// The caller submits the chain and refreshes canonical derived geometry before subsequent operations.
std::vector<PositionOperationChange> EncodePositionOperations(MeshStore &, MeshPipelines &, mtl::ComputeChain &, std::span<const PositionOperationTarget>, PositionEditOp, float factor, uint32_t repeat = 1u, const PositionOperationOptions & = {});
struct FaceAttributeOperationChange {
    uint32_t TargetIndex;
    ClosureSeed Faces;
};
// Submitting completes canonical UV/color permutation and UV tangent invalidation.
// Returned face work belongs to the caller's chain and needs no canonical refresh.
std::vector<FaceAttributeOperationChange> EncodeFaceAttributeOperation(MeshStore &, MeshPipelines &, mtl::ComputeChain &, std::span<const PositionOperationTarget>, bool colors, uint32_t uv_set, uint32_t operation);
