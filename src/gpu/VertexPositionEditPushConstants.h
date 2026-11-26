#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"
#include "gpu/OpDriver.h"
#include "gpu/Transform.h"

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
    PositionEditFlattenDirection = 2048u,
    PositionEditCurveRegular = 4096u,
    PositionEditCurveRaise = 8192u,
    PositionEditCurveLower = 16384u,
    PositionEditCircleContract = 32768u,
    PositionEditAttributeOffset = 65536u,
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
    Transform,
    Attribute,
};

struct VertexPositionEditPushConstants {
    ConnectivityRef Connectivity;
    SlotOffset Handles, Positions;
    uint32_t VertexSlot, Count, Axes;
    float Factor;
    ElementWork Faces DEFAULT();
    SlotOffset Planes;
    uint32_t PlaneCount DEFAULT(), FaceNormalSlot DEFAULT(InvalidSlot);
    uint32_t VertexNormalSlot DEFAULT(InvalidSlot), Flags DEFAULT();
    vec3 Center DEFAULT();
    SlotOffset ReductionBlocks, ReductionResult;
    vec3 Direction DEFAULT(), Gradient DEFAULT();
    SlotOffset Parameters;
    uint32_t ChainCount DEFAULT();
    PositionEditOp Operation DEFAULT(PositionEditOp::Copy);
    NamedAttributeRef Input DEFAULT(); // A vertex attribute of the per-vertex input, a vec3 for Attribute and a float factor otherwise, or none to read Factor
};

// One record's edited vertices: the listing pass writes every live vertex of Blocks into Handles, and the write pass stores each handle's position from Positions.
struct EditedVerticesJob {
    DomainBlocks Blocks DEFAULT();
    SlotOffset Handles DEFAULT(), Positions DEFAULT();
    SlotOffset VertexTable DEFAULT(); // The record's vertex table run
    uint32_t Count DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot);
};
// Each tile of the listing pass is one vertex block of a job, and each tile of the write pass EditedVerticesTile of its handles.
GPU_CONSTANT uint32_t EditedVerticesTile = 256u;

// A gather whose per-vertex input is the first output of the call's program at each vertex, one thread per listed vertex in tiles of JobTileThreads.
struct OpPositionsJob {
    VertexPositionEditPushConstants Edit;
    OpCall Call DEFAULT();
};

struct EdgeChain {
    uint32_t InputOffset, OutputOffset, Count, Closed;
    uint32_t WorkOffset DEFAULT();
    uint32_t PhaseOffset DEFAULT(InvalidOffset);
};

struct PositionPlane {
    vec3 X, Y, InverseX, InverseY;
    vec2 Offset;
};
struct WarpParameters {
    PositionPlane Plane;
    float Minimum, Maximum;
};
struct BendParameters {
    PositionPlane Plane;
    float Radius, Pivot;
};
struct RandomizeParameters {
    float Uniform, Normal;
    uint32_t Seed;
};
struct VertexSlideParameters {
    vec3 Direction, Scale;
    uint32_t Reference;
};
struct EdgeSlideDirections {
    vec3 Positive, Negative;
};

struct PositionReducePushConstants {
    SlotOffset Input, Output;
    uint32_t Count;
    float Scale;
    uint32_t Bounds DEFAULT();
};

struct TransformPositionParameters {
    Transform Frame, Delta;
    vec3 Pivot, Mirror;
};
