#include "Bindless.metal"
#include "gpu/VertexPositionEditPushConstants.h"

// One invocation per independent boundary chain. Fitting reads shared GPU
// scratch; dependency batches preserve order where boundaries meet.
kernel void Circularize(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexPositionEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i>=pc.ChainCount) return;
    const auto words=BindlessBuffer(uint,b.Buffer,pc.Parameters.Slot);
    const auto chain=reinterpret_cast<device const EdgeChain *>(words+pc.Parameters.Offset)[i];
    const uint n=chain.Count;
    const auto inputs=words+chain.InputOffset;
    const auto handles=BindlessBuffer(uint,b.Buffer,pc.Handles.Slot)+pc.Handles.Offset;
    auto positions=reinterpret_cast<device packed_float3 *>(BindlessBufferMutable(uint,b.Buffer,pc.Positions.Slot)+pc.Positions.Offset);
    const auto normals=BindlessBuffer(packed_float3,b.Buffer,pc.VertexNormalSlot);
    const float3 origin=float3(positions[inputs[0]]);
    float3 center=0.f,normal_sum=0.f;
    for (uint k=0u;k<n;++k) { center+=float3(positions[inputs[k]])-origin; normal_sum+=float3(normals[handles[inputs[k]]]); }
    center=origin+center/float(n);
    float3 normal=0.f,previous=float3(positions[inputs[n-1u]])-center;
    for (uint k=0u;k<n;++k) {
        const float3 current=float3(positions[inputs[k]])-center;
        normal+=cross(previous,current); previous=current;
    }
    const float normal_length=length(normal);
    if (!(normal_length>=1e-6f) || !isfinite(normal_length)) return;
    normal/=normal_length;
    const bool reverse=dot(normal,normal_sum)<0.f;
    if (reverse) normal=-normal;
    uint axis=abs(normal.y)<abs(normal.x) ? 1u : 0u;
    if (abs(normal.z)<abs(normal[axis])) axis=2u;
    float3 reference=0.f; reference[axis]=1.f;
    const float3 x=normalize(cross(reference,normal)),y=cross(normal,x);
    const auto rank=[&](uint k) { return inputs[reverse ? n-1u-k : k]; };
    const auto point=[&](uint k) { const float3 p=float3(positions[rank(k)])-center; return float2(dot(p,x),dot(p,y)); };
    float2 fitted=0.f;
    float radius=0.f;
    if (pc.Flags & PositionEditCircleContract) {
        float perimeter=0.f;
        float2 previous=point(n-1u);
        for (uint k=0u;k<n;++k) {
            const float2 current=point(k); const float distance=length(current-previous);
            fitted+=(previous+current)*distance; perimeter+=distance; previous=current;
        }
        if (perimeter>0.f) fitted*=.5f/perimeter;
        float squared=INFINITY;
        for (uint k=0u;k<n;++k) { const float2 d=point(k)-fitted; squared=min(squared,dot(d,d)); }
        radius=sqrt(squared);
    } else {
        radius=1.f;
        bool converged=false;
        for (uint step=0u;step<500u;++step) {
            float3x3 matrix=float3x3(0.f); float3 rhs=0.f;
            for (uint k=0u;k<n;++k) {
                const float2 d=fitted-point(k); const float distance=length(d);
                if (distance<1e-6f) continue;
                const float3 row=float3(d/distance,-1.f);
                for (uint column=0u;column<3u;++column) matrix[column]+=row*row[column];
                rhs+=row*(radius-distance);
            }
            const float scale=max(matrix[0][0],max(matrix[1][1],matrix[2][2]));
            if (!(scale>0.f) || !isfinite(scale)) break;
            matrix/=scale; rhs/=scale;
            const float3 a=cross(matrix[1],matrix[2]),c=cross(matrix[0],matrix[1]),bb=cross(matrix[2],matrix[0]);
            const float determinant=dot(matrix[0],a);
            if (!(abs(determinant)>1e-15f) || !isfinite(determinant)) break;
            const float3 delta=float3(dot(a,rhs),dot(bb,rhs),dot(c,rhs))/determinant;
            fitted+=delta.xy; radius+=delta.z;
            if (all(abs(delta)<float3(1e-6f))) { converged=true; break; }
        }
        if (!converged) {
            fitted=0.f; radius=0.f;
            for (uint k=0u;k<n;++k) radius+=length(point(k));
            radius/=float(n);
        }
    }
    if (pc.Direction.x>0.f) radius=pc.Direction.x;
    float step=0.f,start=0.f;
    const bool regular=pc.Flags & PositionEditCurveRegular;
    // Precise atan2 preserves quadrants when an exact right angle produces signed zero.
    if (regular) {
        float angle=2.f*M_PI_F;
        if (!chain.Closed) {
            angle=0.f; float2 before=point(0u)-fitted;
            for (uint k=1u;k<n;++k) {
                const float2 after=point(k)-fitted;
                angle+=precise::atan2(before.x*after.y-before.y*after.x,dot(before,after)); before=after;
            }
            angle=clamp(angle,-2.f*M_PI_F,2.f*M_PI_F);
        }
        step=angle/float(n-uint(!chain.Closed));
        float2 sum=0.f;
        for (uint k=0u;k<n;++k) {
            const float2 d=point(k)-fitted;
            const float deviation=precise::atan2(d.y,d.x)-step*float(k);
            sum+=float2(cos(deviation),sin(deviation));
        }
        start=precise::atan2(sum.y,sum.x);
    }
    for (uint k=0u;k<n;++k) {
        const float2 d=point(k)-fitted;
        const float angle=(regular ? start+step*float(k) : precise::atan2(d.y,d.x))-pc.Direction.y;
        const float2 target=fitted+radius*float2(cos(angle),sin(angle));
        const float3 before=float3(positions[rank(k)]);
        const float3 circle=center+x*target.x+y*target.y;
        float3 after=mix(before,circle,pc.Factor);
        for (uint axis=0u;axis<3u;++axis) if (!(pc.Axes & (1u<<axis))) after[axis]=before[axis];
        positions[rank(k)]=packed_float3(after);
    }
}
