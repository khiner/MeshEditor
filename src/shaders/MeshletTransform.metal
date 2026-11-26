#include "gpu/MeshletLimit.h"
#include "MeshletNonTriangle.metal"
#include "gpu/MaterialPattern.h"
#include "gpu/MeshVertexConstant.h"
#include "gpu/VisibilityId.h"
#include "VertexTransform.metal"

constant bool NonTriangleTopology [[function_constant(uint(MeshVertexConstant::NonTriangleTopology))]];
constant bool SplitPatternsConstant [[function_constant(uint(MeshVertexConstant::SplitPatterns))]];
constant bool SplitPatterns = is_function_constant_defined(SplitPatternsConstant) && SplitPatternsConstant;

// Emit one attribute-carrying vertex per corner because indexed mesh output produces nondeterministic attributes on this driver.
// Opaque meshlets emit welded positions and fetch attributes during shading.
using MeshletOutput = metal::mesh<MeshletVertexVaryings, void, uint(MeshletLimit::MaxTriangles) * 3u, uint(MeshletLimit::MaxTriangles), metal::topology::triangle>;
using MeshletPatternOutput = metal::mesh<MeshletPatternVaryings, void, uint(MeshletLimit::MaxTriangles) * 3u, uint(MeshletLimit::MaxTriangles), metal::topology::triangle>;
using MeshletVisibilityOutput = metal::mesh<MeshletPositionVaryings, MeshletVisibilityPrimitiveVaryings, uint(MeshletLimit::MaxVertices), uint(MeshletLimit::MaxTriangles), metal::topology::triangle>;

inline uint MeshletVisibilityId(uint visible_index, uint triangle) {
    return (visible_index << uint(VisibilityId::TriangleBits)) | triangle;
}

// A forward vertex carries no object-space surface, and a pattern vertex carries its corner's and its triangle's visibility id, which keys its fragments.
// Where the frame splits patterned materials from the others, the pattern pipelines draw only triangles of patterned materials.
inline bool DrawsPatterned(const thread MeshletVertexVaryings &) { return false; }
inline bool DrawsPatterned(const thread MeshletPatternVaryings &) { return true; }
// A forward vertex carries its object's and element's pick identity and its facing, which the pattern pipelines' fragments read none of.
inline void SetIdentity(thread MeshletVertexVaryings &out, uint object_id, uint element_id, int facing) {
    out.ObjectId = object_id;
    out.ElementId = element_id;
    out.Facing = facing;
}
inline void SetIdentity(thread MeshletPatternVaryings &, uint, uint, int) {}
inline void SetObjectSurface(thread MeshletVertexVaryings &, const thread Scene &, DrawData, uint, uint, bool, float3, uint) {}
inline void SetObjectSurface(thread MeshletPatternVaryings &out, const thread Scene &scene, DrawData draw, uint corner, uint vertex_id, bool coarse, float3 coarse_normal, uint key) {
    out.Key = key;
    out.ObjectPosition = scene.GetLocalPosition(draw, vertex_id);
    out.ObjectNormal = CornerNormal(scene, draw, corner, vertex_id, coarse ? InvalidOffset : scene.CornerFace(draw, corner), coarse, coarse_normal);
    out.ObjectTangent = half3(draw.CornerTangent.Slot != InvalidSlot ? scene.CornerTangent(draw.CornerTangent, corner).xyz : float3(0.0f));
}

template<typename V, typename Output>
inline void ForwardMesh(
    Output output, uint thread_index, uint3 threadgroup_position, device const BindlessSet &bindless, constant SceneViewUBO &view,
    constant ViewportThemeUbo &theme, constant WorkspaceLights &workspace, constant MeshletDrawPushConstants &pc
) {
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(scene, pc, threadgroup_position.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    const uint topology = NonTriangleTopology ?
        MeshletPrimitiveTopology(work.Meshlet) : uint(MeshPrimitiveTopology::Triangle);
    const bool triangles_topology = topology == uint(MeshPrimitiveTopology::Triangle);
    output.set_primitive_count(work.Meshlet.TriangleCount * (triangles_topology ? 1u : 2u));
    if (thread_index >= work.Meshlet.TriangleCount * (triangles_topology ? 3u : 6u)) return;
    V out;
    if (triangles_topology) {
        const bool coarse = MeshletCoarse(work.Meshlet);
        const uint local_triangle = thread_index / 3u;
        // Coarse triangles have no source-triangle identity.
        const uint triangle = coarse ? 0u : work.Payloads.TriangleIds[work.Meshlet.TriangleOffset + local_triangle];
        const MeshletTriangleCorners corners = ResolveMeshletCorners(scene, work.Draw, work.Payloads, work.Meshlet, triangle, local_triangle);
        const uint vertex_index = corners.CornerIds[thread_index % 3u];
        const uint vertex_id = corners.VertexIds[thread_index % 3u];
        const bool flat_face = (work.Payloads.LocalTriangles[work.Meshlet.LocalTriangleOffset + local_triangle][0] & uint(MeshletGeometryEncoding::FlatTriangleBit)) != 0u;
        SetVertexVaryings(out, TransformVertex(scene, work.Draw, vertex_index, vertex_index, vertex_id, false, !flat_face, coarse, corners.CoarseNormal));
        SetObjectSurface(out, scene, work.Draw, vertex_index, vertex_id, coarse, corners.CoarseNormal, MeshletVisibilityId(work.VisibleIndex, local_triangle));
        const Transform world = MeshletWorld(scene, work.Draw);
        const auto face = coarse ? MeshletCoarseFace(scene, work.Primitive, work.Instance, work.Draw, world) :
                                   MeshletFace(scene, work.Draw, work.Instance, world, triangle, flat_face);
        out.FlatWorldNormal = face.FlatWorldNormal;
        out.FaceOverlayFlags = face.FaceOverlayFlags;
        out.MaterialIndex = face.MaterialIndex;
        out.WorldScale = face.WorldScale;
        SetIdentity(out, face.ObjectId, face.ElementId, BackfaceCullFacing(view, VisibilityDoubleSided(view, scene.Materials(view.MaterialSlot)[face.MaterialIndex]), float3(world.S)));
        out.Topology = topology;
        out.PointCoord = float2(0.0f);
    } else {
        const uint element_index = thread_index / 6u;
        const uint corner = line_quad_corner(thread_index);
        const uint element = work.Payloads.TriangleIds[work.Meshlet.TriangleOffset + element_index];
        const uint vertex_id = NonTriangleVertexId(work.Payloads, work.Meshlet, topology, element_index, corner);
        SetVertexVaryings(out, TransformVertex(scene, work.Draw, vertex_id, vertex_id, vertex_id, false, true));
        SetObjectSurface(out, scene, work.Draw, vertex_id, vertex_id, false, float3(0.0f), 0u);
        out.Position = NonTrianglePosition(scene, work.Payloads, work.Draw, work.Meshlet, topology, element_index, corner);
        out.FlatWorldNormal = float3(0.0f);
        out.FaceOverlayFlags = 0u;
        out.MaterialIndex = MeshletPrimitiveMaterialIndex(scene, work.Primitive);
        const float3 scale = float3(MeshletWorld(scene, work.Draw).S);
        out.WorldScale = (abs(scale.x) + abs(scale.y) + abs(scale.z)) / 3.0f;
        SetIdentity(out, work.Instance.ObjectId, scene.ElementPickId(work.Draw, topology == uint(MeshPrimitiveTopology::Line), element), 0);
        out.Topology = topology;
        out.PointCoord = PointQuadCorner(corner) * 0.5f + 0.5f;
        // Alpha zero marks an unselected instance for fill recoloring during shading.
        out.Color = scene.View.InteractionMode == InteractionMode::Object && scene.View.ShowOverlays != 0u ?
            scene.ObjectSelectionColor(scene.InstanceState(work.Draw), float4(0.0f)) : float4(0.0f);
    }
    if (MeshletElementHidden(scene,work,pc,thread_index/(triangles_topology ? 3u : 6u))) out.Position=float4(0.f);
    if (SplitPatterns) {
        const bool patterned = triangles_topology && BindlessBuffer(MaterialPattern, bindless.Buffer, view.PatternSlot)[out.MaterialIndex].Program.Index != InvalidOffset;
        if (patterned != DrawsPatterned(out)) out.Position = float4(0.0f);
    }
    output.set_vertex(thread_index, out);
    output.set_index(thread_index, thread_index);
}

[[mesh]] void MeshletForwardMesh(
    MeshletOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    SCENE_ARGS,
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    ForwardMesh<MeshletVertexVaryings>(output, thread_index, threadgroup_position, bindless, view, theme, workspace, pc);
}

// The forward vertices of patterned materials' triangles, with their surface in object space and their triangle's key.
[[mesh]] void MeshletPatternMesh(
    MeshletPatternOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    SCENE_ARGS,
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    ForwardMesh<MeshletPatternVaryings>(output, thread_index, threadgroup_position, bindless, view, theme, workspace, pc);
}

[[mesh]] void MeshletVisibilityMesh(
    MeshletVisibilityOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    SCENE_ARGS,
    constant MeshletDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(scene, pc, threadgroup_position.x);
    if (!work.Valid) {
        output.set_primitive_count(0u);
        return;
    }
    const uint topology = MeshletPrimitiveTopology(work.Meshlet);
    if (topology == uint(MeshPrimitiveTopology::Triangle) && thread_index < work.Meshlet.VertexCount) {
        const uint source_vertex = MeshletSourceVertex(work.Payloads, work.Meshlet, thread_index);
        const uint vertex_id = MeshletVertexId(scene, work.Draw, topology, source_vertex);
        output.set_vertex(thread_index, MeshletPositionVaryings{MeshletPosition(scene, work.Draw, MeshletWorld(scene, work.Draw), vertex_id)});
    } else if (topology != uint(MeshPrimitiveTopology::Triangle) && thread_index < work.Meshlet.TriangleCount * 4u) {
        const uint element = thread_index / 4u;
        const uint corner = thread_index & 3u;
        output.set_vertex(thread_index, MeshletPositionVaryings{NonTrianglePosition(scene, work.Payloads, work.Draw, work.Meshlet, topology, element, corner)});
    }
    const bool triangles_topology = topology == uint(MeshPrimitiveTopology::Triangle);
    const uint primitive_count = work.Meshlet.TriangleCount * (triangles_topology ? 1u : 2u);
    if (thread_index < primitive_count) {
        const uint element = triangles_topology ? thread_index : thread_index / 2u;
        const bool hidden = MeshletElementHidden(scene,work,pc,element);
        for (uint corner = 0u; corner < 3u; ++corner) {
            uint index = 0u;
            if (!hidden) {
                if (triangles_topology) index = work.Payloads.LocalTriangles[work.Meshlet.LocalTriangleOffset + thread_index][corner] & uint(MeshletGeometryEncoding::LocalIndexMask);
                else index = element * 4u + line_quad_corner((thread_index & 1u) * 3u + corner);
            }
            output.set_index(thread_index * 3u + corner, index);
        }
        output.set_primitive(thread_index, MeshletVisibilityPrimitiveVaryings{
            MeshletVisibilityId(work.VisibleIndex, thread_index)
        });
    }
    output.set_primitive_count(primitive_count);
}
