#include "mesh/RecalculateNormals.h"
#include "Profile.h"
#include "gpu/RecalculateNormalsPushConstants.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"

std::vector<uint32_t> RecalculateFaceFlips(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, const GeometrySelection &selection) {
    if (selection.Faces.empty()) return {};
    const profile::CpuScope scope{"RecalculateFaceFlips"};
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const uint32_t faces = a.FaceTriangles.HandleBound(record.FaceData), halfedges = a.FaceCorners.HandleBound(record.FaceCorners);
    // The planner's scratch is sized once: the neighbor, label and component runs, then the selection and its flips.
    const uint64_t scratch_words = uint64_t(halfedges) + uint64_t(1u + NormalComponentWords) * faces + 2ull * selection.Faces.size() + 16u;
    mtl::ComputeChain chain{meshes.BufferContext(), scratch_words};
    auto &scratch = chain.Scratch;
    const RecalculateNormalsPushConstants pc{
        .Connectivity = meshes.GetConnectivityRef(id),
        .Blocks = BlocksOf(meshes, id, MeshStore::ElementDomain::Face),
        .HalfedgeBlocks = BlocksOf(meshes, id, MeshStore::ElementDomain::Halfedge),
        .VertexSlot = a.Vertices.Buffer.Slot,
        .ScratchSlot = scratch.Buffer.Slot,
        .Neighbors = TakeScratch(chain, halfedges).Offset,
        .Labels = TakeScratch(chain, faces).Offset,
        .Components = TakeScratch(chain, uint64_t(NormalComponentWords) * faces).Offset,
        .Changed = TakeScratch(chain, 1u).Offset,
        .Selection = scratch.Allocate(selection.Faces).Offset,
        .SelectionCount = uint32_t(selection.Faces.size()),
        .Flips = TakeScratch(chain, selection.Faces.size()).Offset,
    };
    const uint32_t face_threads = pc.Blocks.Count * MeshElementBlockSize, halfedge_threads = pc.HalfedgeBlocks.Count * MeshElementBlockSize;
    chain.Concurrent([&] {
        FillScratch(chain, pipelines, pc.Components, NormalComponentWords * faces, 0u);
        chain.Threads(pipelines[MeshPass::RecalculateNormalNeighbors], pc, halfedge_threads);
        chain.Threads(pipelines[MeshPass::RecalculateNormalLabelInit], pc, face_threads);
    });
    profile::RecordCounter("NormalOrientationRounds", RunUntilUnchanged(chain, pipelines[MeshPass::RecalculateNormalLink], pc, face_threads, pc.Changed));
    chain.Threads(pipelines[MeshPass::RecalculateNormalMarkSeeds], pc, pc.SelectionCount);
    chain.Threads(pipelines[MeshPass::RecalculateNormalBounds], pc, face_threads);
    chain.Threads(pipelines[MeshPass::RecalculateNormalVolumes], pc, face_threads);
    chain.Threads(pipelines[MeshPass::RecalculateNormalFlips], pc, pc.SelectionCount);
    chain.Submit();
    const auto flips = scratch.Get({pc.Flips, pc.SelectionCount});
    std::vector<uint32_t> result;
    for (uint32_t i = 0u; i < selection.Faces.size(); ++i) {
        if (flips[i]) result.push_back(selection.Faces[i]);
    }
    return result;
}
