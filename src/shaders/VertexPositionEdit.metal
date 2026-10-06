#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
#include "gpu/VertexPositionEditPushConstants.h"
#include "gpu/Vertex.h"

inline float3 SnapSymmetryPosition(uint i,uint v,float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    const uint partner=BindlessBuffer(uint,b.Buffer,pc.Parameters.Slot)[pc.Parameters.Offset+i];
    const auto vertices=BindlessBuffer(Vertex,b.VertexBuffer,pc.VertexSlot);
    const uint axis=ctz(pc.Axes);
    float3 position=before;
    if (partner==v) position[axis]=0.f;
    else {
        const float3 other=float3(vertices[partner].Position);
        bool primary=position[axis]>other[axis] || (position[axis]==other[axis] && v<partner);
        if (pc.Flags & PositionEditSymmetryNegative) primary=!primary;
        const float3 source=primary ? position : other;
        float3 mirrored=primary ? other : position; mirrored[axis]=-mirrored[axis];
        position=mix(source,mirrored,pc.Factor);
        if (!primary) position[axis]=-position[axis];
    }
    return position;
}

// Blender's bmo_smooth_vert_exec: average each incident edge's other endpoint,
// including unselected neighbors, before any selected vertex is written.
inline float3 SmoothPosition(uint v,float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    const auto vertices = BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot);
    const auto corners = BindlessBuffer(uint, b.IndexBuffer, pc.CornerSlot);
    const ConnectivityView topology{b, pc.Connectivity, pc.FaceCount};
    float3 sum = 0.f;
    uint count = 0u;
    topology.ForEachIncidentEdge(v, [&](uint edge) {
        const uint h = topology.EdgeHalfedge(edge);
        const uint a = corners[h], b = corners[topology.Previous(h)];
        sum += float3(vertices[a == v ? b : a].Position);
        ++count;
    });
    float3 position = count ? before + pc.Factor * (sum / float(count) - before) : before;
    for (uint axis = 0u; axis < 3u; ++axis)
        if (!(pc.Axes & (1u << axis))) position[axis] = before[axis];
    return position;
}

// Blender's VertsToTransData and BM_vert_calc_shell_factor_ex: face mode uses
// selected faces for the normal; even thickness uses selected faces when present.
inline float3 ShrinkFattenPosition(uint v,float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    const auto vertices = BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot);
    const auto corners = BindlessBuffer(uint, b.IndexBuffer, pc.CornerSlot);
    const auto normals = BindlessBuffer(packed_float3, b.Buffer, pc.FaceNormalSlot);
    const auto selected = BindlessBuffer(uint, b.Buffer, pc.FaceSelectionSlot);
    const ConnectivityView topology{b, pc.Connectivity, pc.FaceCount};
    float3 normal = float3(BindlessBuffer(packed_float3, b.Buffer, pc.VertexNormalSlot)[v]);
    // Blender's bm_vert_calc_normals uses the radial direction when the full fan has no normal.
    if (all(normal == float3(0.f))) normal = NormalizeOrZero(before);
    const auto angle = [&](uint h, uint face) {
        const uint2 loop = topology.FaceHalfedges(face);
        const uint prev = h == loop.x ? loop.y - 1u : h - 1u, next = h + 1u == loop.y ? loop.x : h + 1u;
        const float3 a = NormalizeOrZero(float3(vertices[corners[prev]].Position) - before);
        const float3 z = NormalizeOrZero(float3(vertices[corners[next]].Position) - before);
        return acos(clamp(dot(a, z), -1.f, 1.f));
    };
    const auto is_selected = [&](uint face) { return (selected[face / 32u] & (1u << (face % 32u))) != 0u; };
    if (pc.Flags & PositionEditSelectedFaceNormals) {
        float3 sum = 0.f;
        uint count = 0u;
        for (const auto item : topology.Fan(v)) if (is_selected(item.y)) {
            sum += float3(normals[item.y]) * angle(item.x, item.y);
            ++count;
        }
        if (count) normal = NormalizeOrZero(sum);
    }
    float shell = 1.f;
    if (pc.Flags & PositionEditEvenOffset) {
        float2 sum = 0.f, weight = 0.f;
        uint count_selected = 0u;
        for (const auto item : topology.Fan(v)) {
            const float cosine = abs(dot(normal, float3(normals[item.y])));
            const float factor = cosine < 1e-8f ? 1.f : 1.f / cosine;
            const float a = angle(item.x, item.y);
            sum.x += factor * a;
            weight.x += a;
            if (is_selected(item.y)) {
                sum.y += factor * a;
                weight.y += a;
                ++count_selected;
            }
        }
        const uint at = count_selected ? 1u : 0u;
        if (weight[at] > 0.f) shell = sum[at] / weight[at];
    }
    return before + normal * (pc.Factor * shell);
}

// Blender's bmo_planar_faces_exec keeps these original planes for every iteration.
kernel void PlanarFacePlanes(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexPositionEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.PlaneCount) return;
    const ElementWorkDomain faces{bindless, pc.Faces, 0u};
    const uint face = faces.Handle(i);
    const ConnectivityView topology{bindless, pc.Connectivity, pc.FaceCount};
    const auto range = topology.FaceHalfedges(face);
    const auto vertices = BindlessBuffer(Vertex, bindless.VertexBuffer, pc.VertexSlot);
    const auto corners = BindlessBuffer(uint, bindless.IndexBuffer, pc.CornerSlot);
    const auto position = [&](uint h) { return float3(vertices[corners[h]].Position); };
    float3 center = 0.f;
    float total = 0.f;
    for (uint h = range.x; h < range.y; ++h) {
        const float3 p = position(h);
        const float weight = distance(p, position(h == range.x ? range.y - 1u : h - 1u)) +
            distance(p, position(h + 1u == range.y ? range.x : h + 1u));
        center += weight * p;
        total += weight;
    }
    if (total > 0.f) center /= total;
    const float3 normal = float3(BindlessBuffer(packed_float3, bindless.Buffer, pc.FaceNormalSlot)[face]);
    auto planes = reinterpret_cast<device packed_float4 *>(BindlessBufferMutable(uint, bindless.Buffer, pc.Planes.Slot) + pc.Planes.Offset);
    planes[i] = packed_float4(float4(normal, -dot(normal, center)));
}

inline float3 PlanarPosition(uint v,float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    const ConnectivityView topology{b, pc.Connectivity, pc.FaceCount};
    const ElementWorkDomain faces{b, pc.Faces, 0u};
    const auto planes = reinterpret_cast<device const packed_float4 *>(BindlessBuffer(uint, b.Buffer, pc.Planes.Slot) + pc.Planes.Offset);
    float3 target = 0.f;
    uint count = 0u;
    for (const auto item : topology.Fan(v)) {
        const uint index = faces.Index(item.y);
        if (index == InvalidOffset) continue;
        const float4 plane = float4(planes[index]);
        const float3 projected = before - plane.xyz * (dot(plane.xyz, before) + plane.w);
        target += (projected - target) / float(++count);
    }
    const float3 delta = target - before;
    const float3 position = count && dot(delta, delta) > 1e-10f ? before + pc.Factor * delta : before;
    return position;
}

// Radius and bounds share a fixed reduction tree; only the combine operation differs.
inline float2 PositionStatistics(float2 value,uint lane,bool bounds,threadgroup float2 *partials) {
    const float2 total=bounds ? float2(simd_min(value.x),simd_max(value.y)) : float2(simd_sum(value.x),0.f);
    if ((lane&31u)==0u) partials[lane/32u]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 result=bounds ? float2(INFINITY,-INFINITY) : float2(0.f);
    if (lane==0u) for (uint group=0u;group<8u;++group)
        result=bounds ? float2(min(result.x,partials[group].x),max(result.y,partials[group].y)) : result+partials[group];
    return result;
}

kernel void PositionStatisticsGather(
    uint i [[thread_position_in_grid]],uint lane [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexPositionEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const bool bounds=pc.Operation==PositionEditOp::Warp;
    float2 value=bounds ? float2(INFINITY,-INFINITY) : float2(0.f);
    if (i<pc.Count) {
        const uint v=BindlessBuffer(uint,b.Buffer,pc.Handles.Slot)[pc.Handles.Offset+i];
        const float3 position=float3(BindlessBuffer(Vertex,b.VertexBuffer,pc.VertexSlot)[v].Position);
        if (bounds) {
            device const auto &warp=*reinterpret_cast<device const WarpParameters *>(BindlessBuffer(uint,b.Buffer,pc.Parameters.Slot)+pc.Parameters.Offset);
            value=float2(dot(position,float3(warp.Plane.X))+warp.Plane.Offset.x);
        } else value.x=distance(position,float3(pc.Center));
    }
    threadgroup float2 partials[8];
    const float2 total=PositionStatistics(value,lane,bounds,partials);
    if (lane==0u) reinterpret_cast<device packed_float2 *>(BindlessBufferMutable(float,b.Buffer,pc.ReductionBlocks.Slot)+pc.ReductionBlocks.Offset)[group]=packed_float2(total);
}

kernel void PositionStatisticsReduce(
    uint i [[thread_position_in_grid]],uint lane [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant PositionReducePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const auto input=reinterpret_cast<device const packed_float2 *>(BindlessBuffer(float,b.Buffer,pc.Input.Slot)+pc.Input.Offset);
    const float2 value=i<pc.Count ? float2(input[i]) : pc.Bounds ? float2(INFINITY,-INFINITY) : float2(0.f);
    threadgroup float2 partials[8];
    const float2 total=PositionStatistics(value,lane,pc.Bounds!=0u,partials)*pc.Scale;
    if (lane==0u) reinterpret_cast<device packed_float2 *>(BindlessBufferMutable(float,b.Buffer,pc.Output.Slot)+pc.Output.Offset)[group]=packed_float2(total);
}

// Blender's To Sphere averages local distances to the shared pivot across all edited meshes.
inline float3 ToSpherePosition(float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    const float3 delta = before - float3(pc.Center);
    const float radius = length(delta), mean = BindlessBuffer(float, b.Buffer, pc.ReductionResult.Slot)[pc.ReductionResult.Offset];
    return float3(pc.Center) + NormalizeOrZero(delta) * mix(radius, mean, pc.Factor);
}

inline float3 PushPullPosition(float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    return before + pc.Factor * NormalizeOrZero(float3(pc.Center) - before);
}

// The host transforms the world displacement vector and distance covector into
// each primary's local space, including nonuniform and reflected object scales.
inline float3 ShearPosition(float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    const float distance = dot(before - float3(pc.Center), float3(pc.Gradient));
    return before + float3(pc.Direction) * (pc.Factor * distance);
}

// Blender's object_warp_transverts, with tangent continuation outside the range.
inline float3 WarpPosition(float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    device const WarpParameters &warp=*reinterpret_cast<device const WarpParameters *>(BindlessBuffer(uint,b.Buffer,pc.Parameters.Slot)+pc.Parameters.Offset);
    float lo=warp.Minimum,hi=warp.Maximum;
    if (pc.Flags&PositionEditWarpAutoRange) {
        const auto bounds=BindlessBuffer(float,b.Buffer,pc.ReductionResult.Slot)+pc.ReductionResult.Offset;
        lo=bounds[0]; hi=bounds[1];
    }
    float3 position=before;
    if (hi>lo) {
        const float x=dot(before,float3(warp.Plane.X))+warp.Plane.Offset.x,y=dot(before,float3(warp.Plane.Y))+warp.Plane.Offset.y;
        const float angle=-pc.Factor,phi=(clamp(x,lo,hi)-(lo+.5f*(hi-lo)))/(hi-lo)*angle;
        float2 warped{-sin(phi)*y,cos(phi)*y};
        if (x<lo) warped+=float2(-cos(.5f*angle),sin(.5f*angle))*(lo-x);
        else if (x>hi) warped+=float2(cos(.5f*angle),sin(.5f*angle))*(x-hi);
        position+=float3(warp.Plane.InverseX)*(warped.x-x)+float3(warp.Plane.InverseY)*(warped.y-y);
    }
    return position;
}

// Blender's transdata_elem_bend, evaluated in its bend plane. The half-angle
// identity avoids subtracting nearly equal cosines around a distant pivot.
inline float3 BendPosition(float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    device const BendParameters &bend=*reinterpret_cast<device const BendParameters *>(BindlessBuffer(uint,b.Buffer,pc.Parameters.Slot)+pc.Parameters.Offset);
    const float x=dot(before,float3(bend.Plane.X))+bend.Plane.Offset.x;
    const float y=dot(before,float3(bend.Plane.Y))+bend.Plane.Offset.y;
    float factor=x/bend.Radius;
    if (pc.Flags&PositionEditBendClamp) factor=clamp(factor,0.f,1.f);
    const float delta=bend.Radius*factor,angle=-pc.Factor*factor;
    const float sine=sin(angle),half_sine=sin(.5f*angle),cosine_minus_one=-2.f*half_sine*half_sine;
    const float dx=-delta+cosine_minus_one*(x-delta)-sine*(y-bend.Pivot);
    const float dy=sine*(x-delta)+cosine_minus_one*(y-bend.Pivot);
    const float3 position=before+float3(bend.Plane.InverseX)*dx+float3(bend.Plane.InverseY)*dy;
    return position;
}

inline uint PositionRandomHash(uint value) {
    value^=value>>16u; value*=0x7feb352du;
    value^=value>>15u; value*=0x846ca68bu;
    return value^(value>>16u);
}

// Independent integer streams permit one thread per selected canonical handle.
// The 24-bit conversion stays in [0,1), including after float rounding.
inline float PositionRandom(uint handle,uint seed,uint stream) {
    return float(PositionRandomHash(handle^PositionRandomHash(seed+stream))>>8u)*(1.f/16777216.f);
}

inline float3 RandomizePosition(uint v,float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    device const RandomizeParameters &settings=*reinterpret_cast<device const RandomizeParameters *>(BindlessBuffer(uint,b.Buffer,pc.Parameters.Slot)+pc.Parameters.Offset);
    const float magnitude=mix(PositionRandom(v,settings.Seed,0x9e3779b9u),1.f,settings.Uniform);
    const float z=2.f*PositionRandom(v,settings.Seed,0x243f6a88u)-1.f;
    const float angle=6.28318530718f*PositionRandom(v,settings.Seed,0xb7e15162u);
    const float radius=sqrt(max(0.f,1.f-z*z));
    float3 direction{radius*cos(angle),radius*sin(angle),z};
    if (settings.Normal>0.f) {
        float3 normal=float3(BindlessBuffer(packed_float3,b.Buffer,pc.VertexNormalSlot)[v]);
        if (all(normal==float3(0.f))) normal=NormalizeOrZero(before);
        const float cosine=dot(direction,normal);
        if (cosine<0.f) normal=-normal;
        const float alignment=clamp(abs(cosine),0.f,1.f);
        // Blender's nearest-hemisphere spherical interpolation, with the same
        // near-parallel linear branch and zero-normal behavior.
        float a=1.f-settings.Normal,c=settings.Normal;
        if (alignment<.9999f) {
            const float omega=acos(alignment),denominator=sin(omega);
            a=sin(a*omega)/denominator; c=sin(c*omega)/denominator;
        }
        direction=a*direction+c*normal;
    }
    return before+direction*(pc.Factor*magnitude);
}

inline float3 VertexSlideDelta(uint v,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    device const VertexSlideParameters &settings=*reinterpret_cast<device const VertexSlideParameters *>(BindlessBuffer(uint,b.Buffer,pc.Parameters.Slot)+pc.Parameters.Offset);
    const auto vertices=BindlessBuffer(Vertex,b.VertexBuffer,pc.VertexSlot);
    const auto corners=BindlessBuffer(uint,b.IndexBuffer,pc.CornerSlot);
    const float3 before=float3(vertices[v].Position);
    const ConnectivityView topology{b,pc.Connectivity,pc.FaceCount};
    float best=-INFINITY;
    uint best_edge=InvalidOffset;
    float3 result=0.f;
    topology.ForEachIncidentEdge(v,[&](uint edge) {
        const uint h=topology.EdgeHalfedge(edge),a=corners[h],z=corners[topology.Previous(h)];
        const float3 delta=float3(vertices[a==v ? z : a].Position)-before;
        const float score=dot(float3(settings.Direction),NormalizeOrZero(delta*float3(settings.Scale)));
        if (score>best || (score==best && edge<best_edge)) {
            best=score; best_edge=edge; result=delta;
        }
    });
    return result;
}

kernel void VertexSlideReference(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexPositionEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    device const VertexSlideParameters &settings=*reinterpret_cast<device const VertexSlideParameters *>(BindlessBuffer(uint,b.Buffer,pc.Parameters.Slot)+pc.Parameters.Offset);
    BindlessBufferMutable(float,b.Buffer,pc.ReductionResult.Slot)[pc.ReductionResult.Offset]=length(VertexSlideDelta(settings.Reference,b,pc));
}

inline float3 VertexSlidePosition(uint v,float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    const float3 delta=VertexSlideDelta(v,b,pc);
    float3 position=before+delta*pc.Factor;
    if (pc.Flags&PositionEditSlideEven) {
        const float edge_length=length(delta);
        const float distance=pc.Factor*BindlessBuffer(float,b.Buffer,pc.ReductionResult.Slot)[pc.ReductionResult.Offset];
        position=before;
        if (edge_length>1.1920928955078125e-7f)
            position+=(pc.Flags&PositionEditSlideFlipped ? delta : float3(0.f))+
                delta*((pc.Flags&PositionEditSlideFlipped ? -distance : distance)/edge_length);
    }
    return position;
}

inline float3 EdgeSlidePosition(uint i,float3 before,device const BindlessSet &b,constant VertexPositionEditPushConstants &pc) {
    device const auto &rails=reinterpret_cast<device const EdgeSlideDirections *>(BindlessBuffer(uint,b.Buffer,pc.Parameters.Slot)+pc.Parameters.Offset)[i];
    const float3 positive=float3(rails.Positive),negative=float3(rails.Negative);
    float3 delta=0.f;
    if (pc.Flags&PositionEditSlideEven) {
        const float length_between=length(positive-negative);
        if (length_between>1.1920928955078125e-7f) {
            const bool flipped=pc.Flags&PositionEditSlideFlipped;
            const float reference=BindlessBuffer(float,b.Buffer,pc.ReductionResult.Slot)[pc.ReductionResult.Offset];
            const float distance=reference*((flipped ? pc.Factor : -pc.Factor)+1.f)*.5f;
            const float t=min(length_between,distance)/length_between;
            const float3 a=flipped ? negative : positive,z=flipped ? positive : negative,line=z-a;
            const float middle=-dot(a,line)/dot(line,line);
            if (t<middle) delta=abs(middle)<1.1920928955078125e-7f ? float3(0.f) : a*(1.f-t/middle);
            else delta=abs(1.f-middle)<1.1920928955078125e-7f ? z : z*((t-middle)/(1.f-middle));
        }
    } else if (pc.Flags&PositionEditSlideUnclamped) {
        delta=all(positive==float3(0.f)) ? -negative*pc.Factor : positive*pc.Factor;
    } else delta=pc.Factor<0.f ? negative*(-pc.Factor) : positive*pc.Factor;
    return before+delta;
}

// All per-vertex transforms share the same canonical read and scratch write.
// Operation is uniform across the dispatch; only its own transform is evaluated.
kernel void PositionVerticesGather(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant VertexPositionEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i>=pc.Count) return;
    const uint v=BindlessBuffer(uint,b.Buffer,pc.Handles.Slot)[pc.Handles.Offset+i];
    const float3 before=float3(BindlessBuffer(Vertex,b.VertexBuffer,pc.VertexSlot)[v].Position);
    float3 position=before;
    switch (pc.Operation) {
        case PositionEditOp::SnapSymmetry: position=SnapSymmetryPosition(i,v,before,b,pc); break;
        case PositionEditOp::Smooth: position=SmoothPosition(v,before,b,pc); break;
        case PositionEditOp::ShrinkFatten: position=ShrinkFattenPosition(v,before,b,pc); break;
        case PositionEditOp::Planar: position=PlanarPosition(v,before,b,pc); break;
        case PositionEditOp::ToSphere: position=ToSpherePosition(before,b,pc); break;
        case PositionEditOp::PushPull: position=PushPullPosition(before,b,pc); break;
        case PositionEditOp::Shear: position=ShearPosition(before,b,pc); break;
        case PositionEditOp::Warp: position=WarpPosition(before,b,pc); break;
        case PositionEditOp::Bend: position=BendPosition(before,b,pc); break;
        case PositionEditOp::Randomize: position=RandomizePosition(v,before,b,pc); break;
        case PositionEditOp::VertexSlide: position=VertexSlidePosition(v,before,b,pc); break;
        case PositionEditOp::EdgeSlide: position=EdgeSlidePosition(i,before,b,pc); break;
        default: break;
    }
    auto output=reinterpret_cast<device packed_float3 *>(BindlessBufferMutable(uint,b.Buffer,pc.Positions.Slot)+pc.Positions.Offset);
    output[i]=packed_float3(position);
}

kernel void WriteEditedPositions(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant VertexPositionEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.Count) return;
    const uint v = BindlessBuffer(uint, bindless.Buffer, pc.Handles.Slot)[pc.Handles.Offset + i];
    const auto positions = reinterpret_cast<device const packed_float3 *>(BindlessBuffer(uint, bindless.Buffer, pc.Positions.Slot) + pc.Positions.Offset);
    BindlessBufferMutable(Vertex, bindless.VertexBuffer, pc.VertexSlot)[v].Position = positions[i];
}
