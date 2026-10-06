#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"

enum VertexPositionEditFlags : uint32_t {
    PositionEditSelectedFaceNormals = 1u,
    PositionEditEvenOffset = 2u,
    PositionEditWarpAutoRange = 4u,
    PositionEditBendClamp = 8u,
    PositionEditSlideEven = 16u,
    PositionEditSlideFlipped = 32u,
    PositionEditSlideUnclamped = 64u,
    PositionEditSymmetryNegative = 128u,
    PositionEditCurveCubic = 256u,
    PositionEditRelaxEven = 512u,
    PositionEditFlattenNormals = 1024u,
    PositionEditFlattenView = 2048u,
    PositionEditCurveRegular = 4096u,
    PositionEditCurveRaise = 8192u,
    PositionEditCurveLower = 16384u,
    PositionEditCircleContract = 32768u,
};

enum class PositionEditOp : uint32_t {
    Copy,
    SnapSymmetry,
    Smooth,
    ShrinkFatten,
    Planar,
    ToSphere,
    PushPull,
    Shear,
    Warp,
    Bend,
    Randomize,
    VertexSlide,
    EdgeSlide,
    SpaceEvenly,
    RelaxEdgeLoops,
};

struct VertexPositionEditPushConstants {
    ConnectivityRef Connectivity;
    SlotOffset Handles, Positions;
    uint32_t VertexSlot, CornerSlot, FaceCount, Count, Axes;
    float Factor;
    ElementWork Faces DEFAULT();
    SlotOffset Planes;
    uint32_t PlaneCount DEFAULT(), FaceNormalSlot DEFAULT(InvalidSlot);
    uint32_t VertexNormalSlot DEFAULT(InvalidSlot), FaceSelectionSlot DEFAULT(InvalidSlot), Flags DEFAULT();
    vec3 Center DEFAULT();
    SlotOffset ReductionBlocks, ReductionResult;
    vec3 Direction DEFAULT(), Gradient DEFAULT();
    SlotOffset Parameters;
    uint32_t ChainCount DEFAULT();
    PositionEditOp Operation DEFAULT(PositionEditOp::Copy);
};
static_assert(sizeof(VertexPositionEditPushConstants) == 212);

struct EdgeChain {
    uint32_t InputOffset, OutputOffset, Count, Closed;
    uint32_t WorkOffset DEFAULT();
    uint32_t PhaseOffset DEFAULT(InvalidOffset);
};
static_assert(sizeof(EdgeChain) == 24);

struct PositionPlane {
    vec3 X, Y, InverseX, InverseY;
    vec2 Offset;
};
struct WarpParameters {
    PositionPlane Plane;
    float Minimum, Maximum;
};
static_assert(sizeof(WarpParameters) == 64);
struct BendParameters {
    PositionPlane Plane;
    float Radius, Pivot;
};
static_assert(sizeof(BendParameters) == 64);
struct RandomizeParameters {
    float Uniform, Normal;
    uint32_t Seed;
};
static_assert(sizeof(RandomizeParameters) == 12);
struct VertexSlideParameters {
    vec3 Direction, Scale;
    uint32_t Reference;
};
static_assert(sizeof(VertexSlideParameters) == 28);
struct EdgeSlideDirections {
    vec3 Positive, Negative;
};
static_assert(sizeof(EdgeSlideDirections) == 24);

struct PositionReducePushConstants {
    SlotOffset Input, Output;
    uint32_t Count;
    float Scale;
    uint32_t Bounds DEFAULT();
};
static_assert(sizeof(PositionReducePushConstants) == 28);
