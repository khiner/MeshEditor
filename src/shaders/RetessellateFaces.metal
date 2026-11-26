#include "ElementWorkShared.metal"
#include "gpu/PolygonTriangulation.h"
#include "gpu/RetessellateFacesJob.h"

kernel void RetessellateFaces(JOB_TABLE_ARGS(RetessellateFacesJob), uint simd_lane [[thread_index_in_simdgroup]], uint simd_width [[threads_per_simdgroup]]) {
    JOB_THREAD(job, i);
    const bool active=i<job.Count;
    const ElementValues<const uint> corners{BindlessBuffer(uint,b.IndexBuffer,job.CornerSlot),BindlessBuffer(uint,b.Buffer,job.Tables.Halfedges.Slot)+job.Tables.Halfedges.Offset};
    const auto vertices=BindlessBuffer(Vertex,b.VertexBuffer,job.VertexSlot);
    const ElementValues<packed_uint3> triangles{BindlessBufferMutable(packed_uint3,b.Buffer,job.TriangleSlot),BindlessBuffer(uint,b.Buffer,job.Tables.Triangles.Slot)+job.Tables.Triangles.Offset};
    device uint *first_triangles=BindlessBufferMutable(uint,b.ObjectIdBuffer,job.FaceTrianglesSlot);
    const bool dense=job.Faces.Slot==InvalidSlot;
    // Entry k's face handle and scratch offset.
    const auto entry_of=[&](uint k) { return dense ? uint2(k,0u) : uint2(reinterpret_cast<device const packed_uint2 *>(BindlessBuffer(uint,b.Buffer,job.Faces.Slot)+job.Faces.Offset)[k]); };
    // Calls `visit` with entry k's face corner count, corner positions, scratch and triangle writer.
    const auto polygon=[&](uint k,auto visit) {
        const uint2 entry=entry_of(k);
        const uint face=HandleAddress<MeshElementBlockSize>(b,job.Tables.Faces,entry.x);
        const uint2 loop=uint2(BindlessBuffer(packed_uint2,b.Buffer,job.FaceRangesSlot)[face]);
        const uint n=loop.y-loop.x;
        const uint first=first_triangles[face];
        device uint *scratch=dense ? nullptr : BindlessBufferMutable(uint,b.Buffer,job.Scratch.Slot)+job.Scratch.Offset+6u*entry.y;
        return visit(n,[&](uint corner) {
            const uint v=HandleAddress<MeshElementBlockSize>(b,job.Tables.Vertices,corners[loop.x+corner]);
            return vec3(float3(vertices[v].Position));
        },scratch,reinterpret_cast<device float *>(scratch+3u*n),[&](uvec3 local,uint t) {
            const uint3 triangle=uint3(local)+loop.x;
            if (all(uint3(triangles[first+t])==triangle)) return;
            triangles[first+t]=packed_uint3(triangle);
            MarkWork(b,job.ChangedTriangles,first+t);
        });
    };
    bool triangulated=false;
    if (active) {
        const uint face=HandleAddress<MeshElementBlockSize>(b,job.Tables.Faces,entry_of(i).x);
        const uint2 loop=uint2(BindlessBuffer(packed_uint2,b.Buffer,job.FaceRangesSlot)[face]);
        const uint n=loop.y-loop.x;
        // A dense record's triangles follow its face loops, two fewer per face than its corners.
        if (dense) {
            const uint first=loop.x-2u*i;
            first_triangles[face]=first;
            if (n==3u) triangles[first]=packed_uint3(loop.x,loop.x+1u,loop.x+2u);
        }
        if (job.Tangents.Slot!=InvalidSlot) {
            bool moved=false,changed=false;
            for (uint h=loop.x;h<loop.y;++h) moved|=WorkRank(b,job.ChangedVertices,corners[h])!=InvalidOffset;
            if (moved) {
                auto tangents=BindlessBufferMutable(packed_float4,b.CornerTangentBuffer,job.Tangents.Slot);
                for (uint h=loop.x;h<loop.y;++h) {
                    const uint at=job.Tangents.Offset+h;
                    if (all(float4(tangents[at])==0.f)) continue;
                    tangents[at]=packed_float4(0.f);
                    changed=true;
                }
            }
            const uint first=first_triangles[face];
            if (changed) for (uint t=0u;t<n-2u;++t) MarkWork(b,job.ChangedTriangles,first+t);
        }
        triangulated=dense ? n>3u&&n<=PolygonThreadCorners : n>3u;
    }
    TriangulatePolygons(triangulated,i,simd_lane,simd_width,polygon);
}
