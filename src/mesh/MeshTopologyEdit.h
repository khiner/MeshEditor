#pragma once

#include "mesh/ConnectivityEditWork.h"
#include "mesh/MeshClosure.h"
#include "mesh/MeshTopology.h"
#include "mesh/TopologyOutputHandles.h"

namespace mtl {
struct ComputeChain;
} // namespace mtl
struct PrimitiveListReferences;
struct SpatialFaceWork;
struct TopologyReadView;

// Initial scratch capacity for a local topology edit and render repair.
inline constexpr uint32_t TopologyScratchWords = 64u << 10;

// Record sparse topology edits on the action chain using shared scratch and source clones.
// Retain source corner and triangle identities until render repair completes.
// The document transaction restores canonical storage on cancellation.
struct MeshTopologyEdit {
    // Construct tasks on chain.
    // KeepSelectedFaces and Boolean create meshes, and other tasks edit their source.
    // An unchanged edit has no Output.
    static std::vector<MeshTopologyEdit> Construct(state::Registry &, mtl::ComputeChain &, std::span<const MeshTopologyTask>);
    MeshTopologyEdit(MeshTopologyEdit &&) noexcept;
    ~MeshTopologyEdit();
    // The edits of one construction share publication's submit.
    // The corner class writes and normals stay recorded on the chain, and the chain's next submit runs them.
    // Until then the host reads no normals or corner sectors and writes nothing those passes read.
    static void PublishAll(state::Registry &, std::span<MeshTopologyEdit>);
    // Completes the edits in order, retires the in-place edits' sources, and records the refresh of the selection aggregates every edit changed in one update.
    // Completion submits the chain and releases the corner sector payloads the repaired blocks leave unused.
    // The host reads the refreshed selection once the chain submits again.
    static void FinishAll(state::Registry &, std::span<MeshTopologyEdit *const>);

    GeometryTopologyResult Result(const MeshStore &) const;

    mtl::ComputeChain &Chain;
    uint32_t SourceId;
    uint32_t StoreId; // The destination owner, and equals SourceId for in-place edits.
    bool OwnsSource{}; // A boolean's source is a combined copy of its meshes, which finishing releases.
    MeshTopologyOp Op;
    uint32_t OriginalClassMode{};
    std::unique_ptr<TopologyOutputHandles> Output;
    std::unique_ptr<ConnectivityEditWork> Repair;
    // Old triangles in the entire affected shading neighborhood, including unselected faces whose corner equivalence/normals can change.
    FaceTriangles ChangedTriangles;
    // The corner blocks of a record with sector roots that a local corner repair may retire before pose caches observe the new canonical membership.
    std::vector<uint32_t> OldCornerBlocks;
    ElementWork AddedTriangles{};
    ElementWork RetiredEdges{};
    uint32_t AddedTriangleCount{};
    uint32_t FirstTriangle{};
    // New triangle t inherits ownership from the chain scratch word SourceTriangles.Offset + t - FirstTriangle.
    // Those source identities stay reserved until the edit finishes.
    Range SourceTriangles{};
    // Handles of newly created vertices, listed in the chain's scratch when they are not one run.
    // Retained output vertices live in Output->Vertices, and overlay ownership only needs new ones.
    ElementHandleRange NewVertices{};
    // Blocks whose edges and faces can border the repaired vertices after publication: the neighborhood's and the new ones.
    std::array<std::vector<uint32_t>, 2> RepairedBlocks; // Edges, faces

private:
    struct Closures;
    struct Prepared;
    // Publication's recorded passes use the plan's buffers until Complete.
    std::unique_ptr<Prepared> Plan;
    bool Published{}, Finished{};
    std::optional<GeometryMasks> SelectionOutput;
    Element StoreSelectionSource;
    MeshTopologyEdit(mtl::ComputeChain &, const MeshTopologyTask &);
    // Records the task's source closures, or returns none when the task selects no source.
    // A face list task takes its parsed face list, and a spatial task takes its finished face query.
    std::optional<Closures> RecordClosures(state::Registry &, const MeshTopologyTask &, std::optional<PrimitiveListReferences>, const SpatialFaceWork *);
    // Reads the submitted closures and returns whether the task has source elements to edit.
    bool FinishClosures(const MeshTopologyTask &, Closures &);
    // Lays out the operator's job and workspaces for its count passes, an in-place edit reading its source through `view`.
    void PrepareCounts(state::Registry &, const MeshTopologyTask &, Closures &, const std::shared_ptr<const TopologyReadView> &view);
    // Reads the output counts and identity plan once the chain has submitted them.
    void ReadCounts();
};
