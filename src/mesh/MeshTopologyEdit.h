#pragma once

#include "gpu/ElementHandleRange.h"
#include "gpu/ElementWork.h"
#include "gpu/MeshTopologyOp.h"
#include "mesh/ConnectivityEditWork.h"
#include "mesh/MeshClosure.h"
#include "mesh/MeshTopology.h"
#include "mesh/TopologyOutputHandles.h"
#include "metal/Buffer.h"

namespace mtl { struct ComputeChain; }

// The scratch words a topology transaction's chain starts with.
// They hold a local edit and its render repair.
inline constexpr uint32_t TopologyScratchWords = 64u << 10;

// Sparse topology emission and incidence repair, recorded into a chain the edits of one action share.
// Construction gathers every edit's source closure in one submit and runs every operator through its output counts in a second.
// Publication inserts the outputs, emits them and repairs incidence, edges, fans and corner classes through one submit.
// New edges and fan items are inserted for host bounds and trimmed to the counts that submit reports.
// The corner class writes and the normals it then records run with the chain's next submit.
// Source corner/triangle identities remain reserved until render dependencies have been repaired.
// The owning document transaction restores canonical storage on cancellation.
struct MeshTopologyEdit {
    // Constructs the edits of `tasks` on `chain`, which is the only way to obtain an edit.
    // A KeepSelectedFaces task copies its faces into a new mesh, and every other task edits its source in place.
    // An edit whose task selects no source or produces no change has no Output.
    static std::vector<MeshTopologyEdit> Construct(state::Scene &, mtl::ComputeChain &, std::span<const MeshTopologyTask>, bool capture_inset_basis = false);
    MeshTopologyEdit(MeshTopologyEdit &&) noexcept;
    ~MeshTopologyEdit();
    // The edits of one construction share publication's submit.
    // The corner class writes and normals stay recorded on the chain, and the chain's next submit runs them.
    // Until then the host reads no normals or corner sectors and writes nothing those passes read.
    static void PublishAll(state::Scene &, std::span<MeshTopologyEdit>);
    // Completes the edits in order, retires the in-place edits' sources, and refreshes the selection aggregates every edit changed in one update.
    // Completion submits the chain and releases the corner sector payloads the repaired blocks leave unused.
    static void FinishAll(state::Scene &, std::span<MeshTopologyEdit *const>);

    mtl::ComputeChain &Chain;
    uint32_t SourceId;
    uint32_t StoreId; // The destination owner, and equals SourceId for in-place edits.
    MeshTopologyOp Op;
    uint32_t OriginalClassMode{};
    std::unique_ptr<TopologyOutputHandles> Output;
    std::unique_ptr<ConnectivityEditWork> Repair;
    // Old triangles in the entire affected shading neighborhood, including
    // unselected faces whose corner equivalence/normals can change.
    FaceTriangles ChangedTriangles;
    // Payloads that a local corner repair may retire before pose caches can
    // observe the new canonical membership.
    std::vector<uint32_t> OldNormalPayloadBlocks;
    ElementWork AddedTriangles{};
    ElementWork RetiredEdges{};
    uint32_t AddedTriangleCount{};
    uint32_t FirstTriangle{};
    // New triangle t inherits ownership from SourceTriangles[t - FirstTriangle].
    // Those source identities stay reserved until the edit finishes.
    mtl::Buffer SourceTriangles;
    // Handles of newly created vertices, listed in NewVertexList when they are not one run.
    // Retained output vertices live in Output->Vertices, and overlay ownership only needs new ones.
    ElementHandleRange NewVertices{};
    mtl::Buffer NewVertexList;
    // Optional GPU-resident source basis for a staged inset. Output order is
    // local to this edit and includes retained and newly created vertices.
    mtl::Buffer InsetBasis;
    // Blocks whose edges and faces can border the repaired vertices after publication: the neighborhood's and the new ones.
    std::array<std::vector<uint32_t>,2> RepairedBlocks; // Edges, faces
private:
    struct Closures;
    struct Prepared;
    // Publication's recorded passes use the plan's buffers until Complete.
    std::unique_ptr<Prepared> Plan;
    bool Published{}, Finished{};
    MeshTopologyEdit(mtl::ComputeChain &, const MeshTopologyTask &);
    // Records the task's source closures, or returns none when the task selects no source.
    std::optional<Closures> RecordClosures(state::Scene &, const MeshTopologyTask &);
    // Reads the submitted closures and records the operator through its output counts and identity plan.
    void RecordCounts(state::Scene &, const MeshTopologyTask &, Closures &);
    // Reads the output counts and identity plan once the chain has submitted them.
    void ReadCounts(bool capture_inset_basis);
};
