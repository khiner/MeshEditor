#pragma once

#include "Range.h"
#include "gpu/VertexPositionEditPushConstants.h"
#include "mesh/GeometrySelection.h"
#include "mesh/MeshClosure.h"

struct MeshStore;
struct MeshPipelines;
namespace mtl {
struct ComputePipeline;
template<typename P> struct Driver;
using ComputeDriver = Driver<ComputePipeline>;
} // namespace mtl

struct PositionOperationTarget {
    uint32_t StoreId;
    SelectionSource Selection, Excluded;
    Transform World{};
    uint32_t Reference{InvalidOffset};
    // Every element of the record is selected, and Selection is unread.
    bool Every{};
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
// A unit Mirror reflects across the plane through Pivot with that normal before Delta, and a zero Mirror reflects nothing.
struct PositionTransform {
    Transform Delta{};
    vec3 Pivot{}, Mirror{};
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
// A program an edit calls at each vertex through the driver, whose call reads the named attribute table and the image table the edit uploads from Attributes and Images.
struct PositionProgram {
    const mtl::ComputeDriver *Driver;
    OpCall Call;
    std::vector<OpAttributeRef> Attributes;
    std::vector<OpImageTexture> Images;
};
// A Place edit's vec3 vertex attributes of each vertex's two directions, and the scalar of the second, where the edit's factor is the scalar of the first.
struct PositionPlacement {
    NamedAttributeRef First, Second;
    float SecondFactor{};
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
    std::optional<PositionPlacement> Placement;
    // The per-vertex input from a vertex attribute or a program's first output: the vec3 positions of an Attribute edit, or the factor of a smooth, shrink/fatten or randomize, which reads `factor` without one.
    std::variant<std::monostate, NamedAttributeRef, PositionProgram> Input;
    std::optional<float> SphereRadius{}; // The mean distance a sphere blend scales toward, or none for the mean distance of the selected vertices
};
// The vertices a position operation moves.
struct PositionOperationChange {
    std::vector<uint32_t> Vertices; // Their ascending handles, or none when Every lists them on the GPU
    std::vector<Range> Ranges; // Their handle runs
    bool Every{}; // Every live vertex of the record
};
// Whether the operation plans its edit from the mesh on the host.
bool PlansFromMesh(PositionEditOp, const PositionOperationOptions &);
// The host plan of an edit planned from the mesh.
struct PositionPlan;
// Records the reads of the plan of an edit planned from the mesh on the chain, which gather with the other pending lanes' reads, and calls `then` with the plan after the submits that complete them.
void RecordPositionPlan(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, const PositionOperationTarget &, PositionEditOp, const PositionOperationOptions &, std::function<void(std::shared_ptr<const PositionPlan>)> then);
// The caller submits the chain and refreshes canonical derived geometry before subsequent operations.
// An edit planned from the mesh reads `plan`, or else reads its plan on a chain of its own.
std::optional<PositionOperationChange> EncodePositionOperations(MeshStore &, MeshPipelines &, mtl::ComputeChain &, const PositionOperationTarget &, PositionEditOp, float factor, uint32_t repeat = 1u, const PositionOperationOptions & = {}, const PositionPlan *plan = nullptr);
struct FaceAttributeOperationChange {
    ClosureSeed Faces;
};
// Submitting the chain completes the canonical UV or color permutation and the UV tangent invalidation.
// Returned face work belongs to the caller's chain and needs no canonical refresh.
std::optional<FaceAttributeOperationChange> EncodeFaceAttributeOperation(MeshStore &, MeshPipelines &, mtl::ComputeChain &, const PositionOperationTarget &, bool colors, uint32_t uv_set, uint32_t operation);
