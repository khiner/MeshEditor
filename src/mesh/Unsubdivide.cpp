#include "mesh/Unsubdivide.h"
#include "Profile.h"
#include "gpu/UnsubdivideJob.h"
#include "mesh/AttributeFields.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"
#include "state/Registry.h"

namespace {
// An Unsubdivide plan between its submits.
struct UnsubdivideWork {
    uint32_t StoreId;
    GeometrySelection Selection;
    UnsubdivideJob Pc;
    uint32_t Selected, Vertices, Faces, Slots;
    std::vector<uint32_t> EligibleIndex{};
    uint32_t Rounds{};
    std::function<void(std::optional<MeshTopologyTask>)> Then;
};
// Records the round of the decide pass, and the counts and the list once a round changes nothing.
void RecordDecide(const MeshPipelines &pipelines, mtl::ComputeChain &chain, std::shared_ptr<UnsubdivideWork> work) {
    auto &pc = work->Pc;
    chain.Scratch.GetMutable({pc.Changed, 1u})[0] = 0u;
    chain.Lane(work->StoreId, [&] { mtl::GatherPass(chain, pipelines[MeshPass::UnsubdivideDecide], pc, work->Selected); });
    chain.AfterSubmit([&pipelines, &chain, work] {
        ++work->Rounds;
        auto &pc = work->Pc;
        if (chain.Scratch.Get({pc.Changed, 1u})[0]) return RecordDecide(pipelines, chain, work);
        const auto selected = work->Selected, faces = work->Faces, slots = work->Slots;
        chain.Lane(work->StoreId, [&] {
            FillScratch(chain, pipelines, std::to_array<ScratchFillRun>({{pc.FaceWords, faces + 1u, 0u}, {pc.FaceFlags, faces + 1u, 0u}, {pc.Table, KeyTableWords * slots, InvalidOffset}, {pc.Duplicate, 1u, 0u}}));
            mtl::GatherPass(chain, pipelines[MeshPass::UnsubdivideCount], pc, selected + 1u);
            if (faces) mtl::GatherPass(chain, pipelines[MeshPass::UnsubdivideFaceWords], pc, faces);
            ScanScratch(chain, pipelines, std::array{Range{pc.Removed, selected + 1u}, Range{pc.FaceWords, faces + 1u}, Range{pc.FaceFlags, faces + 1u}});
            if (faces) {
                mtl::GatherPass(chain, pipelines[MeshPass::UnsubdivideInsert], pc, faces);
                mtl::GatherPass(chain, pipelines[MeshPass::UnsubdivideVerify], pc, faces);
            }
        });
        chain.AfterSubmit([&pipelines, &chain, work] {
            auto &pc = work->Pc;
            const auto selected = work->Selected, faces = work->Faces;
            const auto &scratch = chain.Scratch;
            const uint32_t removed = scratch.Get({pc.Removed + selected, 1u})[0], words = scratch.Get({pc.FaceWords + faces, 1u})[0], touched = scratch.Get({pc.FaceFlags + faces, 1u})[0];
            profile::RecordCounter("UnsubdivideRounds", work->Rounds);
            profile::RecordCounter("UnsubdivideRemovedVertices", removed);
            if (!removed || scratch.Get({pc.Duplicate, 1u})[0]) return work->Then({});
            const uint32_t length = pc.FaceCount ? 2u + removed + touched + words : 1u + removed;
            pc.List = TakeScratch(chain, length).Offset;
            pc.Parameter = touched;
            chain.Lane(work->StoreId, [&] { mtl::GatherPass(chain, pipelines[MeshPass::UnsubdivideList], pc, std::max(selected, pc.FaceCount ? faces : 0u)); });
            chain.AfterSubmit([&chain, work, length] {
                const auto &pc = work->Pc;
                const auto list = chain.Scratch.Get({pc.List, length});
                MeshTopologyTask task{.SourceId = work->StoreId, .Op = pc.FaceCount ? MeshTopologyOp::ReplaceFaces : MeshTopologyOp::DissolveVertices, .Flags = pc.FaceCount ? 0u : TopologyFlagListSelects, .List = {list.begin(), list.end()}};
                task.Selection = std::move(work->Selection);
                work->Then(std::move(task));
            });
        });
    });
}
} // namespace

void RecordUnsubdivideTask(state::Registry &r, mtl::ComputeChain &chain, const Mesh &mesh, const GeometrySelection &selection, std::function<void(std::optional<MeshTopologyTask>)> then) {
    const profile::CpuScope scope{"PlanUnsubdivide"};
    if (selection.Vertices.empty()) return then({});
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto &a = meshes.Arenas();
    const auto id = mesh.GetStoreId();
    const auto &record = meshes.Get(id);
    const uint32_t selected = uint32_t(selection.Vertices.size()), vertices = a.Vertices.HandleBound(record.Vertices), faces = a.FaceTriangles.HandleBound(record.FaceData);
    const uint32_t slots = std::bit_ceil(16u * selected);
    // The planner's scratch is reserved once for everything but the list, whose length the scans decide.
    chain.Scratch.ReserveAdditional(uint64_t(UnsubdividePolygonWords + 4u) * selected + 2ull * vertices + 2ull * (uint64_t(faces) + 1u) + 3ull * slots + 3ull * TileCount(std::max(selected, faces) + 1u, BlockElements) + 8u);
    auto &scratch = chain.Scratch;
    auto work = std::make_shared<UnsubdivideWork>(UnsubdivideWork{
        .StoreId = id,
        .Selection = selection,
        .Pc = {
            .Connectivity = meshes.GetConnectivityRef(id),
            .FaceCount = mesh.FaceCount(),
            .FaceBound = faces,
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
        },
        .Selected = selected,
        .Vertices = vertices,
        .Faces = faces,
        .Slots = slots,
        .Then = std::move(then),
    });
    chain.Lane(id, [&] { mtl::GatherPass(chain, pipelines[MeshPass::UnsubdivideEligible], work->Pc, selected); });
    chain.AfterSubmit([&meshes, &pipelines, &chain, work] {
        const auto selected = work->Selected;
        const auto eligible_flags = chain.Scratch.Get({work->Pc.Eligible, selected});
        std::vector<uint32_t> eligible;
        for (uint32_t i = 0u; i < selected; ++i) {
            if (!(eligible_flags[i] & 1u)) continue;
            eligible.push_back(work->Selection.Vertices[i]);
            work->EligibleIndex.push_back(i);
        }
        if (eligible.empty()) return work->Then({});
        RecordComponentLabels(meshes, pipelines, chain, work->StoreId, Element::Vertex, eligible, FieldAdjacency::Domain, true, [&pipelines, &chain, work](std::vector<uint32_t> labels) {
            auto &pc = work->Pc;
            std::vector<uint32_t> tags(work->Selected, 0u);
            for (uint32_t k = 0u; k < labels.size(); ++k) tags[work->EligibleIndex[k]] = labels[k] & 1u;
            chain.Scratch.Buffer.Update(std::as_bytes(std::span{tags}), uint64_t(pc.Tags) * sizeof(uint32_t));
            chain.Lane(work->StoreId, [&] {
                FillScratch(chain, pipelines, std::array{ScratchFillRun{pc.State, work->Vertices, UnsubdivideStateNone}});
                mtl::GatherPass(chain, pipelines[MeshPass::UnsubdividePolygon], pc, work->Selected);
            });
            RecordDecide(pipelines, chain, work);
        });
    });
}

std::optional<MeshTopologyTask> UnsubdivideTask(state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) {
    return mtl::Drained<std::optional<MeshTopologyTask>>(r.Resources.get<const MeshStore>().BufferContext(), [&](mtl::ComputeChain &chain, auto then) { RecordUnsubdivideTask(r, chain, mesh, selection, std::move(then)); });
}
