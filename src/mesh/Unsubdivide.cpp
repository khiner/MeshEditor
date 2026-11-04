#include "mesh/Unsubdivide.h"
#include "Profile.h"
#include "gpu/UnsubdividePushConstants.h"
#include "mesh/AttributeFields.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"
#include "state/Registry.h"

std::optional<MeshTopologyTask> UnsubdivideTask(state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) {
    const profile::CpuScope scope{"PlanUnsubdivide"};
    if (selection.Vertices.empty()) return {};
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto &a = meshes.Arenas();
    const auto id = mesh.GetStoreId();
    const auto &record = meshes.Get(id);
    const uint32_t selected = uint32_t(selection.Vertices.size()), vertices = a.Vertices.HandleBound(record.Vertices), faces = a.FaceTriangles.HandleBound(record.FaceData);
    const uint32_t slots = std::bit_ceil(16u * selected);
    // The planner's scratch is sized once for everything but the list, whose length the scans decide.
    const uint64_t scratch_words = uint64_t(UnsubdividePolygonWords + 4u) * selected + 2ull * vertices + 2ull * (uint64_t(faces) + 1u) + 3ull * slots + 3ull * TileCount(std::max(selected, faces) + 1u, BlockElements) + 8u;
    mtl::ComputeChain chain{meshes.BufferContext(), scratch_words};
    auto &scratch = chain.Scratch;
    UnsubdividePushConstants pc{
        .Connectivity = meshes.GetConnectivityRef(id),
        .FaceCount = mesh.FaceCount(),
        .FaceBound = faces,
        .ScratchSlot = scratch.Buffer.Slot,
        .Selection = scratch.Allocate(selection.Vertices).Offset,
        .SelectionCount = selected,
        .Eligible = TakeScratch(chain, selected).Offset,
        .Tags = TakeScratch(chain, selected).Offset,
        .State = TakeScratch(chain, vertices).Offset,
        .Chain = TakeScratch(chain, vertices).Offset,
        .Polygons = TakeScratch(chain, uint64_t(UnsubdividePolygonWords) * selected).Offset,
        .Changed = TakeScratch(chain, 1u).Offset,
        .Removed = TakeScratch(chain, uint64_t(selected) + 1u).Offset,
        .FaceWords = TakeScratch(chain, uint64_t(faces) + 1u).Offset,
        .FaceFlags = TakeScratch(chain, uint64_t(faces) + 1u).Offset,
        .Table = TakeScratch(chain, uint64_t(KeyTableWords) * slots).Offset,
        .TableMask = slots - 1u,
        .Duplicate = TakeScratch(chain, 1u).Offset,
    };
    chain.Threads(pipelines[MeshPass::UnsubdivideEligible], pc, selected);
    chain.Submit();
    const auto eligible_flags = scratch.Get({pc.Eligible, selected});
    std::vector<uint32_t> eligible, eligible_index;
    for (uint32_t i = 0u; i < selected; ++i) {
        if (eligible_flags[i] & 1u) {
            eligible.push_back(selection.Vertices[i]);
            eligible_index.push_back(i);
        }
    }
    if (eligible.empty()) return {};
    const auto labels = ComponentLabels(meshes, pipelines, id, Element::Vertex, eligible, FieldAdjacency::Domain, true);
    std::vector<uint32_t> tags(selected, 0u);
    for (uint32_t k = 0u; k < eligible.size(); ++k) tags[eligible_index[k]] = labels[k] & 1u;
    scratch.Buffer.Update(std::as_bytes(std::span{tags}), uint64_t(pc.Tags) * sizeof(uint32_t));
    FillScratch(chain, pipelines, pc.State, vertices, UnsubdivideStateNone);
    chain.Threads(pipelines[MeshPass::UnsubdividePolygon], pc, selected);
    const auto rounds = RunUntilUnchanged(chain, pipelines[MeshPass::UnsubdivideDecide], pc, selected, pc.Changed);
    FillScratch(chain, pipelines, pc.FaceWords, faces + 1u, 0u);
    FillScratch(chain, pipelines, pc.FaceFlags, faces + 1u, 0u);
    FillScratch(chain, pipelines, pc.Table, KeyTableWords * slots, InvalidOffset);
    FillScratch(chain, pipelines, pc.Duplicate, 1u, 0u);
    chain.Threads(pipelines[MeshPass::UnsubdivideCount], pc, selected + 1u);
    if (faces) chain.Threads(pipelines[MeshPass::UnsubdivideFaceWords], pc, faces);
    ScanScratch(chain, pipelines, pc.Removed, selected + 1u);
    ScanScratch(chain, pipelines, pc.FaceWords, faces + 1u);
    ScanScratch(chain, pipelines, pc.FaceFlags, faces + 1u);
    if (faces) {
        chain.Threads(pipelines[MeshPass::UnsubdivideInsert], pc, faces);
        chain.Threads(pipelines[MeshPass::UnsubdivideVerify], pc, faces);
    }
    chain.Submit();
    const uint32_t removed = scratch.Get({pc.Removed + selected, 1u})[0], words = scratch.Get({pc.FaceWords + faces, 1u})[0], touched = scratch.Get({pc.FaceFlags + faces, 1u})[0];
    profile::RecordCounter("UnsubdivideRounds", rounds);
    profile::RecordCounter("UnsubdivideRemovedVertices", removed);
    if (!removed || scratch.Get({pc.Duplicate, 1u})[0]) return {};
    const uint32_t length = pc.FaceCount ? 2u + removed + touched + words : 1u + removed;
    pc.List = TakeScratch(chain, length).Offset;
    pc.Parameter = touched;
    chain.Threads(pipelines[MeshPass::UnsubdivideList], pc, std::max(selected, pc.FaceCount ? faces : 0u));
    chain.Submit();
    const auto list = scratch.Get({pc.List, length});
    MeshTopologyTask task{.SourceId = id, .Op = pc.FaceCount ? MeshTopologyOp::ReplaceFaces : MeshTopologyOp::DissolveVertices, .Flags = pc.FaceCount ? 0u : TopologyFlagListSelects, .List = {list.begin(), list.end()}};
    task.Selection = selection;
    return task;
}
