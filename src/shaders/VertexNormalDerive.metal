#ifndef VERTEXNORMALDERIVE_MSL
#define VERTEXNORMALDERIVE_MSL

// Derives face, smooth-vertex, and normal-sector shading normals in deterministic corner order.
// Fan items use Blender's corner-angle weighting of unit face normals.
#include "Bindless.metal"
#include "gpu/NormalDeriveEntry.h"
#include "NormalSectorTraversal.metal"
#include "gpu/NormalDerivePushConstants.h"
#include "ElementWorkShared.metal"

struct DeriveContext {
    device const BindlessSet &B;
    constant NormalDerivePushConstants &Pc;

    device packed_float3 *FaceNormals() const { return BindlessBufferMutable(packed_float3, B.Buffer, Pc.FaceNormalSlot); }
    device packed_float3 *VertexNormals() const { return BindlessBufferMutable(packed_float3, B.Buffer, Pc.VertexNormalSlot); }

    float3 Position(NormalDeriveEntry entry, uint vertex_id) const {
        const uint posed = PoseAttributeIndex(B, Pc.PositionNodesSlot, entry.PositionNamespace, vertex_id);
        if (posed != InvalidOffset) return float3(BindlessBuffer(packed_float3, B.Buffer, Pc.PositionSlot)[posed]);
        float3 position = float3(BindlessBuffer(Vertex, B.VertexBuffer, entry.Vertices.Slot)[vertex_id].Position);
        if (entry.Morph.BlocksSlot != InvalidSlot) {
            const uint index = ElementAttributeIndex(B, entry.Morph, vertex_id, entry.MorphTargetIndex);
            position += float3(BindlessBuffer(MorphTargetVertex, B.MorphTargetBuffer, entry.Morph.ValuesSlot)[index].PositionDelta);
        }
        return position;
    }

    float3 CornerPosition(NormalDeriveEntry entry, uint h) const {
        return Position(entry, BindlessBuffer(uint, B.IndexBuffer, entry.Corners.Slot)[h]);
    }

    // Returns the corner-angle-weighted face normal, or zero for a degenerate face or corner.
    float3 FanContributionKnownOwner(NormalDeriveEntry entry, uint h, float3 position, uint face) const {
        const ConnectivityView conn{B, entry.Connectivity, entry.FaceCount};
        const uint posed = PoseAttributeIndex(B, Pc.FaceNormalNodesSlot, entry.FaceNormalNamespace, face);
        const float3 fn = posed != InvalidOffset ? float3(FaceNormals()[posed]) :
            float3(BindlessBuffer(packed_float3, B.Buffer, Pc.BaseFaceNormalSlot)[face]);
        if (all(fn == float3(0))) return float3(0);
        const uint2 loop = conn.FaceHalfedges(face);
        const float3 e_prev = CornerPosition(entry, h == loop.x ? loop.y-1u : h-1u) - position;
        const float3 e_next = CornerPosition(entry, h+1u < loop.y ? h+1u : loop.x) - position;
        const float d = length(e_prev) * length(e_next);
        if (d == 0.0f) return float3(0);
        return fn * acos(clamp(dot(e_prev, e_next) / d, -1.0f, 1.0f));
    }

    float3 FanContribution(NormalDeriveEntry entry, uint h, float3 position) const {
        return FanContributionKnownOwner(entry,h,position,ConnectivityView{B,entry.Connectivity,entry.FaceCount}.HalfedgeFace(h));
    }

    float3 GatherSectorNormal(NormalDeriveEntry entry, uint root, float3 position) const {
        const ConnectivityView conn{B, entry.Connectivity, entry.FaceCount};
        const uint v = BindlessBuffer(uint, B.IndexBuffer, entry.Corners.Slot)[root];
        const uint limit = conn.Incoming(v).y;
        const NormalSectorTraversal sectors{conn,
            BindlessBuffer(uchar, B.Buffer, Pc.FaceSharpnessSlot), BindlessBuffer(uchar, B.Buffer, Pc.EdgeSharpnessSlot)};
        float3 n = FanContribution(entry, root, position);
        sectors.VisitOthers(root,limit,[&](uint h) { n += FanContribution(entry,h,position); });
        return NormalizeOrZero(n);
    }

    // Use one normalized vector-area normal for every triangle of a non-planar face.
    float3 FaceNormal(NormalDeriveEntry entry, uint f) const {
        const ConnectivityView conn{B, entry.Connectivity, entry.FaceCount};
        const uint2 range = conn.FaceHalfedges(f);
        const float3 origin = CornerPosition(entry, range.x);
        float3 n = float3(0);
        float3 previous = CornerPosition(entry, range.x + 1u) - origin;
        for (uint h = range.x + 2u; h < range.y; ++h) {
            const float3 next = CornerPosition(entry, h) - origin;
            n += cross(previous, next);
            previous = next;
        }
        return NormalizeOrZero(n);
    }
};

kernel void VertexNormalDeriveKernel(
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant NormalDerivePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const DeriveContext ctx{bindless, pc};
    const bool sparse = pc.Work.Storage.Slot != InvalidSlot;
    const uint2 tile = sparse ? uint2(pc.EntryIndex, 0u) : uint2(BindlessBuffer(packed_uint2, bindless.Buffer, pc.TileMapSlot)[pc.FirstTile + group_id]);
    const NormalDeriveEntry entry = BindlessBuffer(NormalDeriveEntry, bindless.Buffer, pc.EntriesSlot)[tile.x];
    if (pc.Phase == 0u) {
        uint i = sparse ? WorkElement(bindless, pc.Work, group_id * 256u + local_id) : tile.y * 256u + local_id;
        if (i == InvalidOffset) return;
        if (!sparse) {
            const ElementWork work = entry.FacesWork;
            if (work.Storage.Slot != InvalidSlot) {
                if (i >= entry.FaceWorkCount) return;
                i = WorkGroupElement(bindless,work,i);
                if (i == InvalidOffset) return;
            } else {
                const MeshElementBlock block = BindlessBuffer(MeshElementBlock,bindless.Buffer,entry.FaceBlocksSlot)[tile.y];
                if ((block.Live[local_id/32u] & (1u << (local_id%32u))) == 0u) return;
            }
        }
        ctx.FaceNormals()[entry.FaceNormalNamespace == InvalidOffset ? i : PoseAttributeIndex(bindless, pc.FaceNormalNodesSlot, entry.FaceNormalNamespace, i)] = packed_float3(ctx.FaceNormal(entry, i));
        return;
    }
    const ConnectivityView conn{bindless, entry.Connectivity, entry.FaceCount};
    const uint pair_lane = local_id & 1u;
    for (uint segment = 0u; segment < 2u; ++segment) {
        const uint ordinal = segment * 128u + local_id / 2u;
        uint i = sparse ? WorkElement(bindless, pc.Work, group_id * 256u + ordinal) : tile.y * 256u + ordinal;
        bool active = i != InvalidOffset;
        if (active && !sparse) {
            const ElementWork work = entry.VerticesWork;
            if (work.Storage.Slot != InvalidSlot) {
                active = i < entry.VertexWorkCount;
                if (active) { i = WorkGroupElement(bindless,work,i); active = i != InvalidOffset; }
            } else {
                const MeshElementBlock block = BindlessBuffer(MeshElementBlock,bindless.Buffer,entry.VertexBlocksSlot)[tile.y];
                active = (block.Live[ordinal/32u] & (1u << (ordinal%32u))) != 0u;
            }
        }
        const uint2 fan = active ? conn.Incoming(i) : uint2(0);
        const float3 position = active ? ctx.Position(entry,i) : float3(0);
        const uint rounds = simd_max((fan.y + 1u) / 2u);
        float3 normal = float3(0);
        for (uint round = 0u; round < rounds; ++round) {
            const uint index = round * 2u + pair_lane;
            float3 contribution = float3(0);
            if (index < fan.y) {
                const uint2 item = conn.FanItem(fan.x + index);
                const uint h = item.x;
                contribution = ctx.FanContributionKnownOwner(entry,h,position,item.y);
                if (entry.HasSectors && CornerSectorRoot(bindless, pc.CornerSectors, h) == h) {
                    const packed_float3 sector = packed_float3(ctx.GatherSectorNormal(entry,h,position));
                    const uint record = ElementAttributeIndex(bindless, pc.NormalSectors, h);
                    if (entry.SectorNamespace != InvalidOffset) {
                        BindlessBufferMutable(packed_float3, bindless.Buffer, pc.PosedSectorValuesSlot)[PoseAttributeIndex(bindless, pc.PosedSectorNodesSlot, entry.SectorNamespace, record)] = sector;
                    } else BindlessBufferMutable(NormalSector, bindless.Buffer, pc.NormalSectors.ValuesSlot)[record].Normal = sector;
                }
            }
            const float3 pair = contribution + simd_shuffle_xor(contribution,1u);
            if (pair_lane == 0u) normal += pair;
        }
        if (active && pair_lane == 0u) {
            ctx.VertexNormals()[entry.VertexNormalNamespace == InvalidOffset ? i : PoseAttributeIndex(bindless, pc.VertexNormalNodesSlot, entry.VertexNormalNamespace, i)] = packed_float3(NormalizeOrZero(normal));
        }
    }
}

#endif
