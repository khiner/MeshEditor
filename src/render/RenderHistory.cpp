#include "render/GpuBuffers.h"
#include "render/MeshletBuildGpu.h"
#include "mesh/MeshStore.h"
#include "project/store/History.h"
#include "state/Scene.h"
#include <cstring>
#include <stdexcept>

namespace {
const store::Records::Codec LodDepthCodec{
    [](const void *) -> uint64_t { return 1u; },
    [](void *,uint64_t count) { if (count!=1u) throw std::logic_error("Invalid LOD depth history length."); },
    [](const void *value,uint64_t index,std::vector<std::byte> &out) {
        if (index) throw std::out_of_range("LOD depth history index.");
        out.resize(sizeof(uint32_t));
        std::memcpy(out.data(),value,sizeof(uint32_t));
    },
    [](void *value,uint64_t index,std::span<const std::byte> bytes) {
        if (index || bytes.size()!=sizeof(uint32_t)) throw std::invalid_argument("Invalid LOD depth history value.");
        std::memcpy(value,bytes.data(),sizeof(uint32_t));
    },
    [](void *value,uint64_t index) {
        if (index) throw std::out_of_range("LOD depth history index.");
        *static_cast<uint32_t *>(value)=0u;
    },
};
}

void GpuBuffers::Track(store::History &history) {
    MeshHistory=std::make_unique<store::Records>(Meshes);
    MeshHistory->Trie.CollectChanged=true;
    history.Track(*MeshHistory,"render.meshes",0);
    LodDepthHistory=std::make_unique<store::Records>(&MeshletLodDepth,LodDepthCodec,1u);
    history.Track(*LodDepthHistory,"render.lodDepth",0);
    VertexBuffer.Track(history,"render.vertices");
    FaceIndexBuffer.Track(history,"render.faceIndices");
    EdgeIndexBuffer.Track(history,"render.edgeIndices");
    VertexIndexBuffer.Track(history,"render.vertexIndices");
    Meshlets.Track(history,"render.meshlets");
    MeshletSpatialNodes.Buffer.Track(history,"render.spatialNodes");
    ActiveMeshlets.Nodes.Track(history,"render.membershipNodes");
    ActiveMeshlets.Leaves.Track(history,"render.membershipLeaves");
    MeshletTriangleIds.Track(history,"render.triangleIds");
    MeshletVertexCorners.Track(history,"render.corners");
    MeshletLocalTriangles.Track(history,"render.triangles");
    ClusterGroups.Track(history,"render.groups");
    LodNodes.Track(history,"render.nodes");
    MeshletLodLeaves.Buffer.Track(history,"render.leaves");
    LodParents.Buffer.Track(history,"render.parents");
    GroupLinks.Buffer.Track(history,"render.groupLinks");
    GroupClusterIds.Track(history,"render.groupIds");
    Primitives.Track(history,"render.primitives");
    PrimitiveRoutes.Track(history,"render.primitiveRoutes");
    MeshRecords.TrackAllocator(history,"render.meshRecords");
    for (uint32_t i=0u;i<ElementMeshlets.size();++i) {
        auto &owner=ElementMeshlets[i];
        const auto name="render.elementOwners"+std::to_string(i);
        owner.Values.Track(history,name+".values");
        owner.Blocks.Buffer.Track(history,name+".blocks");
        owner.Owners.Buffer.Track(history,name+".owners");
    }
}

void GpuBuffers::RefreshMeshBinding(state::Scene &r,uint32_t id) {
    if (id>=Meshes.size() || !Meshes[id]) return;
    const auto &meshes=r.Context.get<const MeshStore>();
    auto &mb=*Meshes[id];
    mb.Vertices.Slot=mb.StoreId==InvalidOffset ? VertexBuffer.Buffer.Slot : meshes.Arenas().Vertices.Buffer.Slot;
    mb.FaceIndices.Slot=mb.StoreId==InvalidOffset ? FaceIndexBuffer.Buffer.Slot : meshes.Arenas().FaceCorners.Buffer.Slot;
    mb.EdgeIndices.Slot=EdgeIndexBuffer.Buffer.Slot;
    mb.VertexIndices.Slot=VertexIndexBuffer.Buffer.Slot;
    if (mb.MeshRecord.Count) {
        MeshRecords.Mirror(mb.MeshRecord);
        MeshRecords.GetMutable(mb.MeshRecord)[0]=mb.StoreId==InvalidOffset ? MeshRecord{
            .VertexSlot=mb.Vertices.Slot,.IndexSlotOffset=mb.FaceIndices,.ModelSlot=Instances.TransformBuffer.Slot,
            .VertexCountOrHeadImageSlot=mb.Vertices.Count,.InstanceStateSlot=Instances.StateBuffer.Slot,.VertexOffset=mb.Vertices.Offset,
        } : BuildMeshRecord(*this,mb,meshes,id,mb.RenderTopology==0u,mb.RenderTopology==1u);
    }
}

std::vector<uint32_t> GpuBuffers::RestoreMeshBindings(state::Scene &r) {
    if (!MeshHistory) return {};
    std::vector<uint32_t> ids;
    for (const auto id:MeshHistory->Trie.TakeChanged()) {
        RefreshMeshBinding(r,uint32_t(id));
        ids.push_back(uint32_t(id));
    }
    PreludeStale=true;
    return ids;
}
