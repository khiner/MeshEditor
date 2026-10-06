#include "Bindless.metal"
#include "gpu/Vertex.h"
#include "gpu/VertexPositionEditPushConstants.h"

inline float EdgeCurveLength(device const float *knots,uint k) {
    const float h=knots[k+1u]-knots[k];
    return h>0.f ? h : 1e-8f;
}

template<typename Position>
void SolveEdgeCurve(uint n,bool closed,device const float *knots,device float *upper,
                    device packed_float4 *solution,Position position) {
    // Small periodic systems have coincident lower and upper neighbors.
    if (closed && n<=2u) {
        const float a=EdgeCurveLength(knots,0u),b=EdgeCurveLength(knots,n-1u);
        const float3 c=n==1u ? float3(0.f) : 3.f*(position(1u)-position(0u))/(a*b);
        solution[0]=packed_float4(float4(c,0.f));
        if (n==2u) solution[1]=packed_float4(float4(-c,0.f));
        return;
    }
    // Solve xyz together. The fourth lane solves the rank-one correction
    // for the cyclic matrix; open chains have zero endpoint curvature.
    float previous_upper=0.f;
    float4 previous_rhs=0.f;
    for (uint k=0u;k<n;++k) {
        float lower=0.f,diagonal=1.f,higher=0.f;
        float4 rhs=0.f;
        if (closed || (k>0u && k+1u<n)) {
            const uint prev=(k+n-1u)%n,next=(k+1u)%n;
            lower=EdgeCurveLength(knots,prev); higher=EdgeCurveLength(knots,k);
            diagonal=2.f*(lower+higher);
            rhs.xyz=3.f*((position(next)-position(k))/higher-(position(k)-position(prev))/lower);
            if (closed && (k==0u || k+1u==n)) {
                const float end=k==0u ? lower : higher;
                diagonal-=end; rhs.w=end;
            }
        }
        const float inverse=1.f/(diagonal-lower*previous_upper);
        upper[k]=previous_upper=higher*inverse;
        previous_rhs=(rhs-lower*previous_rhs)*inverse;
        solution[k]=packed_float4(previous_rhs);
    }
    for (uint k=n-1u;k-->0u;) solution[k]=packed_float4(float4(solution[k])-upper[k]*float4(solution[k+1u]));
    if (closed) {
        const float4 ends=float4(solution[0])+float4(solution[n-1u]);
        const float3 correction=ends.xyz/(1.f+ends.w);
        for (uint k=0u;k<n;++k) {
            float4 value=float4(solution[k]); value.xyz-=correction*value.w; solution[k]=packed_float4(value);
        }
    }
}

template<typename Position>
float3 SampleEdgeCurve(uint n,uint segment,float target,bool cubic,device const float *knots,
                        device const packed_float4 *solution,Position position) {
    const uint next=(segment+1u)%n;
    const float dt=target-knots[segment],h=EdgeCurveLength(knots,segment);
    const float3 a=position(segment),z=position(next);
    if (!cubic) return mix(a,z,dt/h);
    const float3 c=float4(solution[segment]).xyz,last=float4(solution[next]).xyz;
    const float3 slope=(z-a)/h-h*(last+2.f*c)/3.f;
    return a+dt*(slope+dt*(c+dt*(last-c)/(3.f*h)));
}

// One invocation per independent chain: a linear sweep measures arc length,
// solves the natural/periodic cubic, and samples in increasing distance order.
// Only handle metadata comes from the host; all geometry stays in GPU buffers.
kernel void SpaceEvenlyGather(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexPositionEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i>=pc.ChainCount) return;
    auto storage=BindlessBufferMutable(uint,b.Buffer,pc.Parameters.Slot);
    const auto chain=reinterpret_cast<device const EdgeChain *>(storage+pc.Parameters.Offset)[i];
    const uint n=chain.Count,segments=n-uint(!chain.Closed);
    const auto handles=storage+chain.InputOffset;
    const auto vertices=BindlessBuffer(Vertex,b.VertexBuffer,pc.VertexSlot);
    const auto position=[&](uint k) { return float3(vertices[handles[k]].Position); };
    auto knots=reinterpret_cast<device float *>(storage+chain.WorkOffset);
    auto upper=knots+n+1u;
    auto solution=reinterpret_cast<device packed_float4 *>(upper+n);
    knots[0]=0.f;
    bool distinct=false;
    for (uint k=0u;k<segments;++k) {
        const float distance=length(position((k+1u)%n)-position(k));
        knots[k+1u]=knots[k]+distance;
        distinct|=distance>1e-6f;
    }
    const float total=knots[segments];
    const bool valid=distinct && isfinite(total);
    const bool cubic=valid && (pc.Flags & PositionEditCurveCubic);
    if (cubic) SolveEdgeCurve(n,chain.Closed,knots,upper,solution,position);
    auto output=reinterpret_cast<device packed_float3 *>(BindlessBufferMutable(uint,b.Buffer,pc.Positions.Slot)+pc.Positions.Offset);
    uint segment=0u;
    const uint first=chain.Closed ? 0u : 1u,end=chain.Closed ? n : n-1u;
    for (uint k=first;k<end;++k) {
        const float3 before=position(k);
        float3 after=before;
        if (valid) {
            const float target=total*(float(k)/float(segments));
            while (segment+1u<segments && knots[segment+1u]<=target) ++segment;
            after=SampleEdgeCurve(n,segment,target,cubic,knots,solution,position);
            after=mix(before,after,pc.Factor);
            for (uint axis=0u;axis<3u;++axis) if (!(pc.Axes & (1u<<axis))) after[axis]=before[axis];
        }
        output[chain.OutputOffset+k-first]=packed_float3(after);
    }
}

// Each phase reads a fixed set of knots, then publishes its relaxed points
// together before the next phase. Canonical writes use the shared position pass.
kernel void RelaxEdgeLoopsGather(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexPositionEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i>=pc.ChainCount) return;
    auto storage=BindlessBufferMutable(uint,b.Buffer,pc.Parameters.Slot);
    const auto chain=reinterpret_cast<device const EdgeChain *>(storage+pc.Parameters.Offset)[i];
    const uint n=chain.Count,first=chain.Closed ? 0u : 1u,end=chain.Closed ? n : n-1u;
    auto state=reinterpret_cast<device packed_float3 *>(storage+chain.WorkOffset);
    auto knots=reinterpret_cast<device float *>(state+n);
    auto targets=knots+n+1u;
    auto upper=targets+n;
    auto solution=reinterpret_cast<device packed_float4 *>(upper+n);
    const auto vertices=BindlessBuffer(Vertex,b.VertexBuffer,pc.VertexSlot);
    for (uint k=0u;k<n;++k) state[k]=vertices[storage[chain.InputOffset+k]].Position;
    auto output=reinterpret_cast<device packed_float3 *>(BindlessBufferMutable(uint,b.Buffer,pc.Positions.Slot)+pc.Positions.Offset)+chain.OutputOffset;
    for (uint k=first;k<end;++k) output[k-first]=state[k];
    uint cursor=chain.PhaseOffset;
    for (uint phase=0u;phase<2u;++phase) {
        const uint knot_count=storage[cursor++],point_count=storage[cursor++];
        const auto anchors=storage+cursor,points=anchors+knot_count;
        cursor+=knot_count+point_count;
        if (!point_count) continue;
        const uint total=knot_count+point_count;
        uint nk=0u,np=0u;
        float distance=0.f;
        float3 previous=0.f;
        for (uint k=0u;k<total;++k) {
            const uint v=k%2u==0u ? anchors[k/2u] : k+1u==total ? anchors[knot_count-1u] : points[k/2u];
            const float3 current=float3(state[v]);
            if (k) distance+=length(current-previous);
            previous=current;
            if (k%2u==0u || k+1u==total) knots[nk++]=distance;
            else targets[np++]=distance;
        }
        if (!isfinite(distance) || !(distance>0.f)) continue;
        if (pc.Flags & PositionEditRelaxEven)
            for (uint k=0u;k<point_count;++k) targets[k]=.5f*(knots[k]+knots[k+1u]);
        const uint count=knot_count-uint(chain.Closed),segments=knot_count-1u;
        const auto position=[&](uint k) { return float3(state[anchors[k]]); };
        const bool cubic=pc.Flags & PositionEditCurveCubic;
        if (cubic) SolveEdgeCurve(count,chain.Closed,knots,upper,solution,position);
        uint segment=0u;
        for (uint k=0u;k<point_count;++k) {
            while (segment+1u<segments && knots[segment+1u]<=targets[k]) ++segment;
            const float3 sampled=SampleEdgeCurve(count,segment,targets[k],cubic,knots,solution,position);
            output[points[k]-first]=packed_float3(.5f*(float3(state[points[k]])+sampled));
        }
        for (uint k=0u;k<point_count;++k) state[points[k]]=output[points[k]-first];
    }
}

// Paths without read/write conflicts run together. Later batches observe the positions
// written by preceding crossing paths, before the common canonical write pass.
kernel void CurveBetweenSelected(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexPositionEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i>=pc.ChainCount) return;
    auto storage=BindlessBufferMutable(uint,b.Buffer,pc.Parameters.Slot);
    const auto chain=reinterpret_cast<device const EdgeChain *>(storage+pc.Parameters.Offset)[i];
    const uint n=chain.Count,segments=n-uint(!chain.Closed);
    const auto inputs=storage+chain.InputOffset;
    auto positions=reinterpret_cast<device packed_float3 *>(BindlessBufferMutable(uint,b.Buffer,pc.Positions.Slot)+pc.Positions.Offset);
    const auto position=[&](uint k) { return float3(positions[inputs[k]]); };
    auto distances=reinterpret_cast<device float *>(storage+chain.WorkOffset);
    auto knots=distances+n+1u,upper=knots+n+1u;
    auto solution=reinterpret_cast<device packed_float4 *>(upper+n);
    distances[0]=0.f;
    for (uint k=0u;k<segments;++k) distances[k+1u]=distances[k]+length(position((k+1u)%n)-position(k));
    const float total=distances[segments];
    if (!(total>0.f) || !isfinite(total)) return;
    if ((pc.Flags & PositionEditCurveRegular) && total>1e-8f)
        for (uint k=0u;k<n;++k) distances[k]=total*(float(k)/float(segments));
    const auto phase=storage+chain.PhaseOffset;
    const uint count=phase[0],point_count=phase[1];
    const auto anchors=phase+2u,points=anchors+count,surface=points+point_count;
    for (uint k=0u;k<count;++k) knots[k]=distances[anchors[k]];
    if (chain.Closed) knots[count]=total;
    const auto anchor_position=[&](uint k) { return position(anchors[k]); };
    const bool cubic=pc.Flags & PositionEditCurveCubic;
    if (cubic) SolveEdgeCurve(count,chain.Closed,knots,upper,solution,anchor_position);
    const auto handles=BindlessBuffer(uint,b.Buffer,pc.Handles.Slot)+pc.Handles.Offset;
    const auto normals=BindlessBuffer(packed_float3,b.Buffer,pc.VertexNormalSlot);
    uint segment=0u;
    const uint knot_segments=count-uint(!chain.Closed);
    for (uint k=0u;k<point_count;++k) {
        const auto point=points[k];
        const float target=distances[point];
        while (segment+1u<knot_segments && knots[segment+1u]<=target) ++segment;
        const float3 before=position(point);
        float3 after=SampleEdgeCurve(count,segment,target,cubic,knots,solution,anchor_position);
        const float3 delta=after-before;
        if (surface[k] && dot(delta,delta)>1e-16f) {
            const float elevation=dot(delta,float3(normals[handles[inputs[point]]]));
            if ((pc.Flags & PositionEditCurveRaise) && elevation < -1e-8f) continue;
            if ((pc.Flags & PositionEditCurveLower) && elevation > 1e-8f) continue;
        }
        after=mix(before,after,pc.Factor);
        for (uint axis=0u;axis<3u;++axis) if (!(pc.Axes & (1u<<axis))) after[axis]=before[axis];
        positions[inputs[point]]=packed_float3(after);
    }
}
