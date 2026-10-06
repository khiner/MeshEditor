#include "ConnectivityRead.metal"
#include "gpu/RecalculateNormalsPushConstants.h"

struct NormalRecalcContext {
    device const BindlessSet &B;
    constant RecalculateNormalsPushConstants &Pc;
    device const packed_uint2 *Faces() const { return reinterpret_cast<device const packed_uint2 *>(BindlessBuffer(uint,B.Buffer,Pc.Faces.Slot)+Pc.Faces.Offset); }
    device const packed_uint4 *Groups() const { return reinterpret_cast<device const packed_uint4 *>(BindlessBuffer(uint,B.Buffer,Pc.Groups.Slot)+Pc.Groups.Offset); }
    uint2 Tile(uint tile) const { return uint2(reinterpret_cast<device const packed_uint2 *>(BindlessBuffer(uint,B.Buffer,Pc.Tiles.Slot)+Pc.Tiles.Offset)[tile]); }
    device packed_float4 *PartialCenters() const { return reinterpret_cast<device packed_float4 *>(BindlessBufferMutable(uint,B.Buffer,Pc.PartialCenters.Slot)+Pc.PartialCenters.Offset); }
    device packed_float3 *Centers() const { return reinterpret_cast<device packed_float3 *>(BindlessBufferMutable(uint,B.Buffer,Pc.Centers.Slot)+Pc.Centers.Offset); }
    device NormalOrientationCandidate *Candidates() const { return reinterpret_cast<device NormalOrientationCandidate *>(BindlessBufferMutable(uint,B.Buffer,Pc.Candidates.Slot)+Pc.Candidates.Offset); }
    float3 Position(uint h) const { return float3(BindlessBuffer(Vertex,B.VertexBuffer,Pc.VertexSlot)[BindlessBuffer(uint,B.IndexBuffer,Pc.CornerSlot)[h]].Position); }
    ConnectivityView Topology() const { return {B,Pc.Connectivity,Pc.FaceCount}; }
};

inline float4 NormalCenterSum(float4 value,uint lane,threadgroup float4 *sums) {
    sums[lane]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=128u;stride;stride>>=1u) {
        if (lane<stride) sums[lane]+=sums[lane+stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return sums[0];
}

kernel void RecalculateNormalFaceCenters(uint lane [[thread_index_in_threadgroup]],uint tile [[threadgroup_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],constant RecalculateNormalsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const NormalRecalcContext ctx{b,pc};
    const uint2 item=ctx.Tile(tile);
    const uint4 group=uint4(ctx.Groups()[item.x]);
    float4 value=0.f;
    if (item.y+lane<group.x+group.y) {
        const uint face=ctx.Faces()[item.y+lane].x;
        const uint2 loop=ctx.Topology().FaceHalfedges(face);
        float3 center=0.f,area_normal=0.f;
        float weight=0.f;
        const float3 origin=ctx.Position(loop.x);
        for (uint h=loop.x;h<loop.y;++h) {
            const float3 p=ctx.Position(h),prev=ctx.Position(h==loop.x ? loop.y-1u : h-1u),next=ctx.Position(h+1u==loop.y ? loop.x : h+1u);
            const float w=distance(prev,p)+distance(p,next);
            center+=w*p; weight+=w;
            area_normal+=cross(p-origin,next-origin);
        }
        if (weight>0.f) center/=weight;
        const float area=length(area_normal)*.5f;
        value=float4(center*area,area);
    }
    threadgroup float4 sums[256];
    const float4 sum=NormalCenterSum(value,lane,sums);
    if (!lane) ctx.PartialCenters()[tile]=packed_float4(sum);
}

kernel void RecalculateNormalCenters(uint lane [[thread_index_in_threadgroup]],uint component [[threadgroup_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],constant RecalculateNormalsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const NormalRecalcContext ctx{b,pc};
    const uint4 group=uint4(ctx.Groups()[component]);
    float4 value=0.f;
    for (uint i=lane;i<group.w;i+=256u) value+=float4(ctx.PartialCenters()[group.z+i]);
    threadgroup float4 sums[256];
    const float4 sum=NormalCenterSum(value,lane,sums);
    // Match recalc_face_normals_find_index's area-weighted center, including its face-count factor.
    if (!lane) ctx.Centers()[component]=packed_float3(sum.w>0.f ? sum.xyz/(sum.w*float(group.y)) : float3(0.f));
}

inline NormalOrientationCandidate NormalCandidateEmpty() { return {packed_float3(FLT_EPSILON,-FLT_MAX,-FLT_MAX),InvalidOffset,0u}; }
inline bool NormalCandidateBetter(NormalOrientationCandidate a,NormalOrientationCandidate b) {
    if (a.Score.x!=b.Score.x) return a.Score.x>b.Score.x;
    if (a.Score.y!=b.Score.y) return a.Score.y>b.Score.y;
    if (a.Score.z!=b.Score.z) return a.Score.z>b.Score.z;
    return a.Rank<b.Rank;
}
inline NormalOrientationCandidate NormalCandidateReduce(NormalOrientationCandidate value,uint lane,threadgroup NormalOrientationCandidate *values) {
    values[lane]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=128u;stride;stride>>=1u) {
        if (lane<stride && NormalCandidateBetter(values[lane+stride],values[lane])) values[lane]=values[lane+stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return values[0];
}

kernel void RecalculateNormalCandidates(uint lane [[thread_index_in_threadgroup]],uint tile [[threadgroup_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],constant RecalculateNormalsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const NormalRecalcContext ctx{b,pc};
    const uint2 item=ctx.Tile(tile);
    const uint4 group=uint4(ctx.Groups()[item.x]);
    auto best=NormalCandidateEmpty();
    if (item.y+lane<group.x+group.y) {
        const uint2 face=uint2(ctx.Faces()[item.y+lane]);
        const uint2 loop=ctx.Topology().FaceHalfedges(face.x);
        const float3 center=float3(ctx.Centers()[item.x]);
        const float3 normal=float3(BindlessBuffer(packed_float3,b.Buffer,pc.NormalSlot)[face.x]);
        for (uint h=loop.x;h<loop.y;++h) {
            const float3 p=ctx.Position(h),delta=p-center;
            const float dist=dot(delta,delta);
            if (dist<FLT_EPSILON) continue;
            float3 next=ctx.Position(h+1u==loop.y ? loop.x : h+1u)-p,prev=ctx.Position(h==loop.x ? loop.y-1u : h-1u)-p;
            const float a=length(next),z=length(prev);
            if (a<=FLT_EPSILON || z<=FLT_EPSILON) continue;
            next/=a; prev/=z;
            float3 loop_normal=cross(next,prev);
            const float n=length(loop_normal);
            if (n<=FLT_EPSILON) continue;
            loop_normal/=n;
            if (dot(loop_normal,normal)<0.f) loop_normal=-loop_normal;
            const float3 direction=delta*rsqrt(dist);
            const float alignment=dot(direction,loop_normal);
            const NormalOrientationCandidate candidate{packed_float3(dist,max(dot(direction,next),dot(direction,prev)),abs(alignment)),h,uint(alignment<0.f)^face.y};
            if (NormalCandidateBetter(candidate,best)) best=candidate;
        }
    }
    threadgroup NormalOrientationCandidate values[256];
    const auto result=NormalCandidateReduce(best,lane,values);
    if (!lane) ctx.Candidates()[tile]=result;
}

kernel void RecalculateNormalOrientation(uint lane [[thread_index_in_threadgroup]],uint component [[threadgroup_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],constant RecalculateNormalsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const NormalRecalcContext ctx{b,pc};
    const uint4 group=uint4(ctx.Groups()[component]);
    auto best=NormalCandidateEmpty();
    for (uint i=lane;i<group.w;i+=256u) {
        const auto candidate=ctx.Candidates()[group.z+i];
        if (NormalCandidateBetter(candidate,best)) best=candidate;
    }
    threadgroup NormalOrientationCandidate values[256];
    const auto result=NormalCandidateReduce(best,lane,values);
    if (!lane) BindlessBufferMutable(uint,b.Buffer,pc.Orientations.Slot)[pc.Orientations.Offset+component]=result.Flip;
}
