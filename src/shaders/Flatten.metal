#include "Bindless.metal"
#include "gpu/VertexPositionEditPushConstants.h"
#include "gpu/Vertex.h"

inline float4 FlattenSum(float4 value,uint lane,threadgroup float4 *work) {
    work[lane]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step=128u;step;step>>=1u) {
        if (lane<step) work[lane]+=work[lane+step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float4 result=work[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// The smallest covariance eigenvector is the least-squares plane normal.
// Jacobi rotations solve the fixed 3x3 system after the linear reduction.
inline float3 FlattenLeastSquares(float3 diagonal,float3 off) {
    const float scale=max(diagonal.x,max(diagonal.y,diagonal.z));
    if (!(scale>0.f) || !isfinite(scale)) return float3(0.f);
    diagonal/=scale; off/=scale;
    float3x3 a=float3x3(float3(diagonal.x,off.x,off.y),float3(off.x,diagonal.y,off.z),float3(off.y,off.z,diagonal.z));
    float3x3 basis=float3x3(1.f);
    for (uint sweep=0u;sweep<12u;++sweep) for (uint p=0u;p<2u;++p) for (uint q=p+1u;q<3u;++q) {
        const float cross=a[q][p];
        if (abs(cross)<=1e-7f) continue;
        const float tau=(a[q][q]-a[p][p])/(2.f*cross);
        const float t=copysign(1.f,tau)/(abs(tau)+sqrt(1.f+tau*tau));
        const float c=rsqrt(1.f+t*t),s=t*c;
        a[p][p]-=t*cross; a[q][q]+=t*cross; a[p][q]=a[q][p]=0.f;
        for (uint r=0u;r<3u;++r) if (r!=p && r!=q) {
            const float x=a[p][r],y=a[q][r];
            a[p][r]=a[r][p]=c*x-s*y; a[q][r]=a[r][q]=s*x+c*y;
        }
        const float3 x=basis[p],y=basis[q]; basis[p]=c*x-s*y; basis[q]=s*x+c*y;
    }
    uint minimum=a[1][1]<a[0][0] ? 1u : 0u;
    if (a[2][2]<a[minimum][minimum]) minimum=2u;
    return normalize(basis[minimum]);
}

// Every group in a dispatch has disjoint vertices. Subsequent batches observe
// earlier projections when face regions or remaining wire groups share a vertex.
kernel void FlattenGroups(
    uint lane [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexPositionEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const auto words=BindlessBuffer(uint,b.Buffer,pc.Parameters.Slot)+pc.Parameters.Offset;
    const uint offset=BindlessBuffer(uint,b.Buffer,pc.Planes.Slot)[pc.Planes.Offset+group];
    const uint count=words[offset],faces=words[offset+1u];
    const auto vertices=words+offset+2u;
    const auto face_offsets=vertices+count;
    auto positions=reinterpret_cast<device packed_float3 *>(BindlessBufferMutable(uint,b.Buffer,pc.Positions.Slot)+pc.Positions.Offset);
    threadgroup float4 work[256];
    threadgroup float3 normal;
    const float3 origin=float3(positions[vertices[0]]);
    float3 sum=0.f;
    for (uint k=lane;k<count;k+=256u) sum+=float3(positions[vertices[k]])-origin;
    const float3 center=origin+FlattenSum(float4(sum,0.f),lane,work).xyz/float(count);
    if (pc.Flags & PositionEditFlattenView) {
        if (!lane) normal=pc.Direction;
    } else if ((pc.Flags & PositionEditFlattenNormals) && faces) {
        const auto normals=BindlessBuffer(packed_float3,b.Buffer,pc.FaceNormalSlot);
        float3 sum=0.f;
        for (uint k=lane;k<faces;k+=256u) {
            const auto face=words+face_offsets[k];
            const uint n=face[1]; const auto corners=face+2u;
            float3 area=0.f,previous=float3(positions[corners[n-1u]])-center;
            for (uint j=0u;j<n;++j) {
                const float3 current=float3(positions[corners[j]])-center;
                area+=cross(previous,current); previous=current;
            }
            sum+=float3(normals[face[0]])*length(area);
        }
        const float3 total=FlattenSum(float4(sum,0.f),lane,work).xyz;
        if (!lane) normal=dot(total,total)>0.f ? normalize(total) : float3(0,0,1);
    } else {
        float3 diagonal=0.f,off=0.f;
        for (uint k=lane;k<count;k+=256u) {
            const float3 d=float3(positions[vertices[k]])-center;
            diagonal+=d*d; off+=float3(d.x*d.y,d.x*d.z,d.y*d.z);
        }
        diagonal=FlattenSum(float4(diagonal,0.f),lane,work).xyz;
        off=FlattenSum(float4(off,0.f),lane,work).xyz;
        if (!lane) normal=FlattenLeastSquares(diagonal,off);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k=lane;k<count;k+=256u) {
        const float3 before=float3(positions[vertices[k]]);
        float3 delta=pc.Factor*dot(before-center,normal)*normal;
        for (uint axis=0u;axis<3u;++axis) if (!(pc.Axes & (1u<<axis))) delta[axis]=0.f;
        positions[vertices[k]]=packed_float3(before-delta);
    }
}
