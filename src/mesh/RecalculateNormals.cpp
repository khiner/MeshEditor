#include "mesh/RecalculateNormals.h"
#include "gpu/RecalculateNormalsJob.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"
void RecordFaceFlips(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, const GeometrySelection &selection, std::function<void(std::vector<uint32_t>)> then) {
    if (selection.Faces.empty()) return then({});
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const uint32_t faces = a.FaceTriangles.HandleBound(record.FaceData), halfedges = a.FaceCorners.HandleBound(record.FaceCorners);
    // The planner's scratch is reserved once: the neighbor, label and component runs, then the selection and its flips.
    chain.Scratch.ReserveAdditional(uint64_t(halfedges) + uint64_t(1u + NormalComponentWords) * faces + 2ull * selection.Faces.size() + 16u);
    auto &scratch = chain.Scratch;
    const RecalculateNormalsJob pc{
        .Connectivity = meshes.GetConnectivityRef(id),
        .Blocks = BlocksOf(meshes, id, MeshStore::ElementDomain::Face),
        .HalfedgeBlocks = BlocksOf(meshes, id, MeshStore::ElementDomain::Halfedge),
        .VertexSlot = a.Vertices.Buffer.Slot,
        .Neighbors = TakeScratch(chain, halfedges).Offset,
        .Labels = TakeScratch(chain, faces).Offset,
        .Components = TakeScratch(chain, uint64_t(NormalComponentWords) * faces).Offset,
        .Changed = TakeScratch(chain, 1u).Offset,
        .Selection = scratch.Allocate(selection.Faces).Offset,
        .SelectionCount = uint32_t(selection.Faces.size()),
        .Flips = TakeScratch(chain, selection.Faces.size()).Offset,
    };
    const uint32_t face_threads = pc.Blocks.Count * MeshElementBlockSize, halfedge_threads = pc.HalfedgeBlocks.Count * MeshElementBlockSize;
    // Each pass gathers with the pending lanes' planners through the same pipeline.
    const auto pass = [&pipelines, &chain, pc](MeshPass pipeline, uint32_t threads) { mtl::GatherPass(chain, pipelines[pipeline], pc, threads); };
    chain.Lane(id, [&] {
        FillScratch(chain, pipelines, std::array{ScratchFillRun{pc.Components, NormalComponentWords * faces, 0u}});
        pass(MeshPass::RecalculateNormalNeighbors, halfedge_threads);
        pass(MeshPass::RecalculateNormalLabelInit, face_threads);
    });
    RecordUntilUnchanged(chain, id, pc.Changed, [pass, face_threads] { pass(MeshPass::RecalculateNormalLink, face_threads); }, [&chain, id, pass, pc, face_threads, faces = selection.Faces, then = std::move(then)] {
        chain.Lane(id, [&] {
            pass(MeshPass::RecalculateNormalMarkSeeds, pc.SelectionCount);
            pass(MeshPass::RecalculateNormalBounds, face_threads);
            pass(MeshPass::RecalculateNormalVolumes, face_threads);
            pass(MeshPass::RecalculateNormalFlips, pc.SelectionCount);
        });
        chain.AfterSubmit([&chain, pc, faces, then] {
            const auto flips = chain.Scratch.Get({pc.Flips, pc.SelectionCount});
            std::vector<uint32_t> result;
            for (uint32_t i = 0u; i < faces.size(); ++i) {
                if (flips[i]) result.push_back(faces[i]);
            }
            then(std::move(result));
        });
    });
}
