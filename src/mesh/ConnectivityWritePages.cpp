#include "mesh/ConnectivityWritePages.h"

#include "mesh/MeshStore.h"
#include "mesh/PageFootprint.h"
#include "state/Scene.h"

void CaptureConnectivityPrepareWrites(state::Scene &r, const MeshConnectivityJob &job, std::span<const uint32_t> vertex_blocks, std::span<const uint32_t> halfedge_blocks, std::span<const uint32_t> face_blocks) {
    const auto &a = r.Context.get<MeshStore>().Arenas();
    PageFootprint pages;
    pages.Add(a.OutgoingHalfedges.Buffer, vertex_blocks, BlockBytes<uint32_t>);
    for (const auto *buffer : {&a.HalfedgeFaces.Buffer, &a.OppositeHalfedges.Buffer, &a.HalfedgeEdges.Buffer})
        pages.Add(*buffer, halfedge_blocks, BlockBytes<uint32_t>);
    if (!job.FaceStarts) pages.Add(a.FaceRanges.Buffer, face_blocks, BlockBytes<uvec2>);
    pages.CaptureWrites();
}
