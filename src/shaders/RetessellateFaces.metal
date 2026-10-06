#include "Bindless.metal"
#include "ElementWorkShared.metal"
#include "TransformUtils.metal"
#include "gpu/PolygonTriangulation.h"
#include "gpu/RetessellateFacesPushConstants.h"

kernel void RetessellateFaces(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant RetessellateFacesPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i>=pc.Count) return;
    const uint2 entry=uint2(reinterpret_cast<device const packed_uint2 *>(BindlessBuffer(uint,b.Buffer,pc.Faces.Slot)+pc.Faces.Offset)[i]);
    const uint2 loop=uint2(BindlessBuffer(packed_uint2,b.Buffer,pc.FaceRangesSlot)[entry.x]);
    const uint n=loop.y-loop.x;
    const uint first=BindlessBuffer(uint,b.ObjectIdBuffer,pc.FaceTrianglesSlot)[entry.x];
    const auto corners=BindlessBuffer(uint,b.IndexBuffer,pc.CornerSlot);
    if (pc.Tangents.ValuesSlot!=InvalidSlot) {
        bool moved=false,changed=false;
        for (uint h=loop.x;h<loop.y;++h) moved|=WorkRank(b,pc.ChangedVertices,corners[h])!=InvalidOffset;
        if (moved) {
            auto tangents=BindlessBufferMutable(packed_float4,b.CornerTangentBuffer,pc.Tangents.ValuesSlot);
            for (uint h=loop.x;h<loop.y;++h) {
                const uint at=ElementAttributeIndex(b,pc.Tangents,h);
                if (all(float4(tangents[at])==0.f)) continue;
                tangents[at]=packed_float4(0.f);
                changed=true;
            }
        }
        if (changed) for (uint t=0u;t<n-2u;++t) MarkWork(b,pc.ChangedTriangles,first+t);
    }
    if (n<=3u) return;
    device uint *scratch=BindlessBufferMutable(uint,b.Buffer,pc.Scratch.Slot)+pc.Scratch.Offset+4u*entry.y;
    device uint *next=scratch,*previous=scratch+n;
    device vec2 *points=reinterpret_cast<device vec2 *>(scratch+2u*n);
    const auto vertices=BindlessBuffer(Vertex,b.VertexBuffer,pc.VertexSlot);
    device packed_uint3 *triangles=BindlessBufferMutable(packed_uint3,b.Buffer,pc.TriangleSlot);
    TriangulatePolygon(n,[&](uint corner) {
        const uint v=corners[loop.x+corner];
        const float3 base=float3(vertices[v].Position);
        const bool selected=pc.ApplyTransform && pc.SelectionSlot!=InvalidSlot &&
            (BindlessBuffer(uint,b.Buffer,pc.SelectionSlot)[v/32u]&(1u<<(v%32u)));
        return vec3(selected ? trs_inverse_transform_point(pc.Primary,
            apply_edit_transform(trs_transform_point(pc.Primary,base),pc.Pivot,pc.Delta)) : base);
    },points,next,previous,[&](uvec3 local,uint t) {
        const uint3 triangle=uint3(local)+loop.x;
        if (all(uint3(triangles[first+t])==triangle)) return;
        triangles[first+t]=packed_uint3(triangle);
        MarkWork(b,pc.ChangedTriangles,first+t);
    });
}
