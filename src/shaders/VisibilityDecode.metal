#ifndef VISIBILITY_DECODE_MSL
#define VISIBILITY_DECODE_MSL

#include "MeshletResolve.metal"
#include "Varyings.metal"
#include "VisibilityShadingPushConstants.metal"
#include "ArmatureDeform.metal"
#include "CornerClass.metal"
#include "CornerClassEncoding.metal"
#include "MorphDeform.metal"
#include "MeshletNonTriangle.metal"

constant uint VisibilityBackground = 0xffffffffu;
constant uint VisibilityPhaseBit = 0x80000000u;
constant uint VisibilityTriangleMask = 0x3fu;
constant uint VisibilityIndexMask = 0x01ffffffu;

struct DecodedVisibility {
    MeshVaryings V;
    float2 UvDx[4];
    float2 UvDy[4];
    uint ObjectId;
    uint ElementId;
    uint InstanceFlags;
    uint Topology;
    float2 PointCoord;
    bool Valid;
};

struct ResolvedVisibility {
    InstanceRecord Instance;
    MeshletRecord Meshlet;
    PrimitiveRecord Primitive;
    DrawData Draw;
    uint Triangle;
    uint FaceId;
    uint LocalTriangle;
    bool Valid;
};

struct VisibilityMetadata {
    uint ObjectId;
    uint ElementId;
    uint InstanceFlags;
    bool Valid;
};

inline float3 VisibilityPoseWorldPos(
    const thread Scene &scene, DrawData draw, Vertex vert, uint idx,
    uint model_slot, uint armature_slot, uint morph_slot
) {
    float3 normal = float3(0);
    float3 pos = float3(vert.Position);
    ApplyMorphDeform(scene, draw, pos, idx, morph_slot);
    const float3 local_pos = ApplyArmatureDeform(scene, draw, pos, idx, normal, armature_slot);
    return trs_transform_point(scene.Models(model_slot)[draw.FirstInstance], local_pos);
}

inline float3 VisibilityApplyNormalOffset(
    const thread Scene &scene, DrawData draw, uint vertex_id, float3 normal, float2 offset
) {
    const float3 n = dot(normal, normal) > 0.0f ? normal : float3(0, 0, 1);
    const uint tri = (vertex_id / 3u) * 3u;
    const uint k = vertex_id - tri;
    device const uint *indices = scene.Indices(draw.IndexSlotOffset.Slot);
    const float3 p0 = scene.GetLocalPosition(draw, indices[draw.IndexSlotOffset.Offset + tri + k]);
    float3 ref;
    const float3 e1 = scene.GetLocalPosition(draw, indices[draw.IndexSlotOffset.Offset + tri + (k + 1u) % 3u]) - p0;
    const float3 r1 = e1 - n * dot(e1, n);
    const float l1 = length(r1);
    if (l1 > 1e-3f * length(e1)) {
        ref = r1 / l1;
    } else {
        const float3 e2 = scene.GetLocalPosition(draw, indices[draw.IndexSlotOffset.Offset + tri + (k + 2u) % 3u]) - p0;
        const float3 r2 = e2 - n * dot(e2, n);
        const float l2 = length(r2);
        if (l2 > 1e-3f * length(e2)) {
            ref = r2 / l2;
        } else {
            const float3 axis = abs(n.x) < 0.5f ? float3(1, 0, 0) : float3(0, 1, 0);
            ref = normalize(cross(n, axis));
        }
    }
    return cos(offset.x) * n + sin(offset.x) * (cos(offset.y) * ref + sin(offset.y) * cross(n, ref));
}

inline float3 VisibilityCornerNormal(const thread Scene &scene, DrawData draw, uint vertex_id, uint idx, uint face_id) {
    const uint value = draw.CornerClassOffset == INVALID_OFFSET ? CornerClass_Vertex << CornerClassEncoding_TagShift :
        draw.CornerClassOffset == CornerClassEncoding_UniformFaceOffset ? CornerClass_Face << CornerClassEncoding_TagShift :
                                                                          scene.CornerClasses(scene.View.CornerClassSlot)[draw.CornerClassOffset + vertex_id];
    const uint tag = value >> CornerClassEncoding_TagShift;
    float3 normal;
    if (tag == CornerClass_Vertex) {
        normal = scene.GetVertexNormal(draw, idx);
    } else if (tag == CornerClass_Face) {
        normal = draw.PosedFaceNormalOffset != INVALID_OFFSET ?
            float3(scene.PosedFaceNormals(scene.View.PosedFaceNormalSlot)[draw.PosedFaceNormalOffset + face_id - 1u]) :
            float3(scene.BaseFaceNormals(scene.View.BaseFaceNormalSlot)[draw.BaseFaceNormalOffset + face_id - 1u]);
    } else {
        const uint seam = value & CornerClassEncoding_IndexMask;
        normal = draw.PosedSeamNormalOffset != INVALID_OFFSET ?
            float3(scene.PosedSeamNormals(scene.View.PosedSeamNormalSlot)[draw.PosedSeamNormalOffset + seam]) :
            float3(scene.BaseSeamNormals(scene.View.BaseSeamNormalSlot)[draw.BaseSeamNormalOffset + seam]);
    }
    if (draw.CustomCornerMaskOffset != INVALID_OFFSET) {
        const uint corner = draw.CornerBase + vertex_id;
        const uint2 mask = uint2(scene.CustomCornerMasks(scene.View.CustomCornerMaskSlot)[draw.CustomCornerMaskOffset + corner / 32u]);
        const uint bit = 1u << (corner % 32u);
        if ((mask.x & bit) != 0u) {
            const uint packed = mask.y + popcount(mask.x & (bit - 1u));
            normal = VisibilityApplyNormalOffset(
                scene, draw, vertex_id, normal,
                float2(scene.CustomCornerNormals(scene.View.CustomCornerNormalSlot)[draw.CustomCornerNormalOffset + packed])
            );
        }
    }
    if (draw.MorphShadingAuthored != 0u) {
        normal = NormalizeOrZero(normal + float3(scene.PosedMorphNormalDeltas(scene.View.PosedMorphNormalDeltaSlot)[draw.PosedPositionOffset + idx]));
    }
    return normal;
}

inline MeshVaryings VisibilityCorner(
    const thread Scene &scene, DrawData draw, uint vertex_index, uint idx, uint face_id, bool shading_normal,
    bool velocity_output
) {
    MeshVaryings out{};
    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    const uint model_slot = scene.View.ModelSlotOverride != INVALID_SLOT ? scene.View.ModelSlotOverride : draw.ModelSlot;
    const Transform world = scene.Models(model_slot)[draw.FirstInstance];
    const float3 local_pos = scene.GetLocalPosition(draw, idx);
    const float3 world_pos = apply_object_pending_transform(scene, draw, trs_transform_point(world, local_pos));
    out.WorldNormal = shading_normal ? trs_transform_normal(world, VisibilityCornerNormal(scene, draw, vertex_index, idx, face_id)) : float3(0.0f);
    out.WorldPosition = world_pos;
    out.Color = float4(0.8f, 0.8f, 0.8f, 1.0f);
    out.VertexColor = draw.CornerColorOffset != INVALID_OFFSET ?
        float4(scene.CornerColors(scene.View.CornerColorSlot)[draw.CornerColorOffset + vertex_index]) : float4(1.0f);
    device const packed_float2 *uvs = scene.CornerUvs(scene.View.CornerUvSlot);
    out.TexCoord0 = draw.CornerUvOffsets[0] != INVALID_OFFSET ? float2(uvs[draw.CornerUvOffsets[0] + vertex_index]) : float2(0);
    out.TexCoord1 = draw.CornerUvOffsets[1] != INVALID_OFFSET ? float2(uvs[draw.CornerUvOffsets[1] + vertex_index]) : float2(0);
    out.TexCoord2 = draw.CornerUvOffsets[2] != INVALID_OFFSET ? float2(uvs[draw.CornerUvOffsets[2] + vertex_index]) : float2(0);
    out.TexCoord3 = draw.CornerUvOffsets[3] != INVALID_OFFSET ? float2(uvs[draw.CornerUvOffsets[3] + vertex_index]) : float2(0);
    const float4 vertex_tangent = draw.CornerTangentOffset != INVALID_OFFSET ?
        float4(scene.CornerTangents(scene.View.CornerTangentSlot)[draw.CornerTangentOffset + vertex_index]) : float4(0, 0, 0, 1);
    float3 tangent = vertex_tangent.xyz;
    if (dot(tangent, tangent) > 1e-8f) {
        tangent = normalize(tangent);
        float3 tangent_dummy_pos = float3(0.0f);
        ApplyArmatureDeform(scene, draw, tangent_dummy_pos, idx, tangent);
        out.WorldTangent = float4(normalize(trs_transform_normal(world, tangent)), vertex_tangent.w);
    } else {
        out.WorldTangent = float4(0, 0, 0, 1);
    }
    out.Position = scene.ViewProj() * float4(world_pos, 1.0f);
    if (velocity_output) {
        const float3 prev = VisibilityPoseWorldPos(scene, draw, vert, idx, scene.View.PrevModelSlot, scene.View.PrevArmatureDeformSlot, scene.View.PrevMorphWeightsSlot);
        const float3 next = VisibilityPoseWorldPos(scene, draw, vert, idx, scene.View.NextModelSlot, scene.View.NextArmatureDeformSlot, scene.View.NextMorphWeightsSlot);
        out.MotionPrev = prev - world_pos;
        out.MotionNext = next - world_pos;
    }
    return out;
}

inline MeshVaryings VisibilityWorkspaceCorner(
    const thread Scene &scene, DrawData draw, uint vertex_index, uint idx, uint face_id, bool shading_normal
) {
    MeshVaryings out{};
    const uint model_slot = scene.View.ModelSlotOverride != INVALID_SLOT ? scene.View.ModelSlotOverride : draw.ModelSlot;
    const Transform world = scene.Models(model_slot)[draw.FirstInstance];
    const float3 world_pos = apply_object_pending_transform(scene, draw, trs_transform_point(world, scene.GetLocalPosition(draw, idx)));
    out.WorldNormal = shading_normal ? trs_transform_normal(world, VisibilityCornerNormal(scene, draw, vertex_index, idx, face_id)) : float3(0.0f);
    out.WorldPosition = world_pos;
    out.Position = scene.ViewProj() * float4(world_pos, 1.0f);
    return out;
}

template<typename T>
inline T PerspectiveValue(float3 weights, T a, T b, T c) {
    return weights.x * a + weights.y * b + weights.z * c;
}

struct PerspectiveWeights {
    float3 Value;
    float3 Dx;
    float3 Dy;
};

inline PerspectiveWeights TriangleWeights(float2 pixel, float4 c0, float4 c1, float4 c2, float2 viewport_size) {
    const float2 p0 = ndc_to_uv(c0.xy / c0.w) * viewport_size;
    const float2 p1 = ndc_to_uv(c1.xy / c1.w) * viewport_size;
    const float2 p2 = ndc_to_uv(c2.xy / c2.w) * viewport_size;
    const float denominator = (p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y);
    const float inv_denominator = 1.0f / denominator;
    const float3 lambda = {
        ((p1.y - p2.y) * (pixel.x - p2.x) + (p2.x - p1.x) * (pixel.y - p2.y)) * inv_denominator,
        ((p2.y - p0.y) * (pixel.x - p2.x) + (p0.x - p2.x) * (pixel.y - p2.y)) * inv_denominator,
        0.0f,
    };
    const float3 l = {lambda.x, lambda.y, 1.0f - lambda.x - lambda.y};
    const float3 lx = {
        (p1.y - p2.y) * inv_denominator,
        (p2.y - p0.y) * inv_denominator,
        (p0.y - p1.y) * inv_denominator,
    };
    const float3 ly = {
        (p2.x - p1.x) * inv_denominator,
        (p0.x - p2.x) * inv_denominator,
        (p1.x - p0.x) * inv_denominator,
    };
    const float3 inv_w = {1.0f / c0.w, 1.0f / c1.w, 1.0f / c2.w};
    const float d = dot(l, inv_w);
    const float dx = dot(lx, inv_w);
    const float dy = dot(ly, inv_w);
    const float inv_d2 = 1.0f / (d * d);
    return {
        l * inv_w / d,
        (lx * inv_w * d - l * inv_w * dx) * inv_d2,
        (ly * inv_w * d - l * inv_w * dy) * inv_d2,
    };
}

inline void DecodeUv(
    thread float2 &value, thread float2 &dx, thread float2 &dy,
    PerspectiveWeights weights, float2 a, float2 b, float2 c
) {
    value = PerspectiveValue(weights.Value, a, b, c);
    dx = PerspectiveValue(weights.Dx, a, b, c);
    dy = PerspectiveValue(weights.Dy, a, b, c);
}

struct VisibilityCoverageValues {
    PerspectiveWeights Weights;
    float4 VertexColor;
    float2 PointCoord;
    float3 WorldNormal;
    float3 WorldPosition;
};

struct VisibilityTextureCoordinates {
    float2 Value;
    float2 Dx;
    float2 Dy;
};

inline VisibilityCoverageValues DecodeVisibilityCoverage(
    const thread Scene &scene, const thread ResolvedVisibility &resolved, float2 pixel,
    constant SceneViewUBO &view, uint meshlet_vertex_slot
) {
    float4 clip[3];
    float4 vertex_color[3];
    float2 point_coord[3]{};
    float3 world_normal[3]{};
    float3 world_position[3];
    const uint topology = MeshletPrimitiveTopology(resolved.Meshlet);
    const bool triangle_topology = topology == MeshPrimitiveTopology_Triangle;
    const uint logical_element = resolved.LocalTriangle / 2u;
    const Transform world = MeshletWorld(scene, resolved.Draw);
    for (uint corner = 0u; corner < 3u; ++corner) {
        const uint quad_corner = line_quad_corner((resolved.LocalTriangle & 1u) * 3u + corner);
        const uint vertex_index = triangle_topology ?
            (resolved.Triangle - resolved.Primitive.FirstTriangle) * 3u + corner :
            NonTriangleVertexId(
                scene.B, meshlet_vertex_slot, resolved.Meshlet, topology, logical_element, quad_corner
            );
        const uint vertex_id = triangle_topology ?
            scene.Indices(resolved.Draw.IndexSlotOffset.Slot)[resolved.Draw.IndexSlotOffset.Offset + vertex_index] : vertex_index;
        const float3 world_pos = apply_object_pending_transform(
            scene, resolved.Draw, trs_transform_point(world, scene.GetLocalPosition(resolved.Draw, vertex_id))
        );
        world_position[corner] = world_pos;
        if (!triangle_topology) {
            world_normal[corner] = trs_transform_normal(world, scene.GetVertexNormal(resolved.Draw, vertex_id));
        }
        clip[corner] = triangle_topology ? scene.ViewProj() * float4(world_pos, 1.0f) : NonTrianglePosition(
            scene, scene.B, meshlet_vertex_slot, resolved.Draw, resolved.Meshlet,
            topology, logical_element, quad_corner
        );
        vertex_color[corner] = resolved.Draw.CornerColorOffset != INVALID_OFFSET ?
            float4(scene.CornerColors(scene.View.CornerColorSlot)[resolved.Draw.CornerColorOffset + vertex_index]) : float4(1.0f);
        if (!triangle_topology) point_coord[corner] = PointQuadCorners[quad_corner] * 0.5f + 0.5f;
    }
    const PerspectiveWeights weights = TriangleWeights(pixel, clip[0], clip[1], clip[2], float2(view.ViewportSize));
    return {
        weights,
        PerspectiveValue(weights.Value, vertex_color[0], vertex_color[1], vertex_color[2]),
        PerspectiveValue(weights.Value, point_coord[0], point_coord[1], point_coord[2]),
        PerspectiveValue(weights.Value, world_normal[0], world_normal[1], world_normal[2]),
        PerspectiveValue(weights.Value, world_position[0], world_position[1], world_position[2]),
    };
}

inline VisibilityTextureCoordinates DecodeVisibilityTextureCoordinates(
    const thread Scene &scene, const thread ResolvedVisibility &resolved,
    const thread VisibilityCoverageValues &coverage, uint uv_set
) {
    if (MeshletPrimitiveTopology(resolved.Meshlet) != MeshPrimitiveTopology_Triangle) return {};
    const uint set = min(uv_set, 3u);
    const uint offset = resolved.Draw.CornerUvOffsets[set];
    float2 uv[3]{};
    if (offset != INVALID_OFFSET) {
        device const packed_float2 *uvs = scene.CornerUvs(scene.View.CornerUvSlot);
        for (uint corner = 0u; corner < 3u; ++corner) {
            const uint vertex_index = (resolved.Triangle - resolved.Primitive.FirstTriangle) * 3u + corner;
            uv[corner] = float2(uvs[offset + vertex_index]);
        }
    }
    VisibilityTextureCoordinates result;
    DecodeUv(result.Value, result.Dx, result.Dy, coverage.Weights, uv[0], uv[1], uv[2]);
    return result;
}

inline ResolvedVisibility ResolveVisibilityPrimitive(
    uint id,
    device const BindlessSet &bindless,
    VisibilityShadingPushConstants pc
) {
    ResolvedVisibility result{};
    if (id == VisibilityBackground) return result;
    const uint visible_index = (id >> 6u) & VisibilityIndexMask;
    const uint visible_slot = (id & VisibilityPhaseBit) != 0u ? pc.Phase2VisibleMeshletSlot : pc.VisibleMeshletSlot;
    const VisibleMeshlet visible = BindlessBuffer(VisibleMeshlet, bindless.Buffer, visible_slot)[visible_index];
    const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[visible.Instance];
    result.Instance = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot];
    result.Meshlet = BindlessBuffer(MeshletRecord, bindless.Buffer, pc.MeshletSlot)[visible.Meshlet];
    result.Primitive = BindlessBuffer(PrimitiveRecord, bindless.Buffer, pc.PrimitiveSlot)[result.Meshlet.Primitive];
    result.Draw = MeshletDraw(result.Primitive, result.Instance, instance_slot);
    result.LocalTriangle = id & VisibilityTriangleMask;
    result.Valid = true;
    return result;
}

inline ResolvedVisibility ResolveVisibilityId(
    uint id,
    device const BindlessSet &bindless,
    constant SceneViewUBO &view,
    constant ViewportTheme &theme,
    constant WorkspaceLights &workspace,
    VisibilityShadingPushConstants pc
) {
    ResolvedVisibility result = ResolveVisibilityPrimitive(id, bindless, pc);
    if (!result.Valid) return result;
    const Scene scene{bindless, view, theme, workspace};
    const uint topology = MeshletPrimitiveTopology(result.Meshlet);
    const uint logical_element = topology == MeshPrimitiveTopology_Triangle ?
        result.LocalTriangle : result.LocalTriangle / 2u;
    result.Triangle = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot)[
        result.Meshlet.TriangleOffset + logical_element
    ];
    result.FaceId = topology == MeshPrimitiveTopology_Triangle ?
        scene.ObjectIds(result.Draw.ObjectIdSlot)[result.Draw.FaceIdOffset + result.Triangle - result.Primitive.FirstTriangle] :
        result.Triangle + 1u;
    return result;
}

inline VisibilityMetadata DecodeVisibilityMetadata(
    uint id,
    device const BindlessSet &bindless,
    constant SceneViewUBO &view,
    constant ViewportTheme &theme,
    constant WorkspaceLights &workspace,
    VisibilityShadingPushConstants pc
) {
    const ResolvedVisibility resolved = ResolveVisibilityId(id, bindless, view, theme, workspace, pc);
    if (!resolved.Valid) return {};
    return {
        resolved.Instance.ObjectId,
        resolved.Instance.ElementIdOffset + resolved.FaceId,
        resolved.Instance.Flags,
        true,
    };
}

inline DecodedVisibility DecodeVisibilityId(
    uint id, float2 pixel,
    device const BindlessSet &bindless,
    constant SceneViewUBO &view,
    constant ViewportTheme &theme,
    constant WorkspaceLights &workspace,
    VisibilityShadingPushConstants pc,
    bool velocity_output = false
) {
    DecodedVisibility result{};
    if (id == VisibilityBackground) return result;

    const Scene scene{bindless, view, theme, workspace};
    const ResolvedVisibility resolved = ResolveVisibilityId(id, bindless, view, theme, workspace, pc);
    const InstanceRecord instance = resolved.Instance;
    const MeshletRecord meshlet = resolved.Meshlet;
    const PrimitiveRecord primitive = resolved.Primitive;
    const DrawData draw = resolved.Draw;
    const uint triangle = resolved.Triangle;
    const uint face_id = resolved.FaceId;
    const uint topology = MeshletPrimitiveTopology(meshlet);
    if (topology != MeshPrimitiveTopology_Triangle) {
        const uint logical_element = resolved.LocalTriangle / 2u;
        MeshVaryings corners[3];
        float2 point_coords[3];
        for (uint corner = 0u; corner < 3u; ++corner) {
            const uint quad_corner = line_quad_corner((resolved.LocalTriangle & 1u) * 3u + corner);
            const uint vertex_id = NonTriangleVertexId(
                bindless, pc.MeshletVertexSlot, meshlet, topology, logical_element, quad_corner
            );
            corners[corner] = VisibilityCorner(scene, draw, vertex_id, vertex_id, 0u, true, velocity_output);
            corners[corner].Position = NonTrianglePosition(
                scene, bindless, pc.MeshletVertexSlot, draw, meshlet, topology, logical_element, quad_corner
            );
            point_coords[corner] = PointQuadCorners[quad_corner] * 0.5f + 0.5f;
        }
        const PerspectiveWeights weights = TriangleWeights(
            pixel, corners[0].Position, corners[1].Position, corners[2].Position, float2(view.ViewportSize)
        );
        result.V.Position = float4(pixel, 0.0f, 1.0f);
        result.V.WorldNormal = PerspectiveValue(weights.Value, corners[0].WorldNormal, corners[1].WorldNormal, corners[2].WorldNormal);
        result.V.WorldPosition = PerspectiveValue(weights.Value, corners[0].WorldPosition, corners[1].WorldPosition, corners[2].WorldPosition);
        // A selected object's fill recolors in the shading, so alpha zero marks an unselected instance.
        result.V.Color = view.InteractionMode == InteractionMode_Object && view.ShowOverlays != 0u ?
            scene.ObjectSelectionColor(scene.InstanceState(draw), float4(0.0f)) : float4(0.0f);
        result.V.VertexColor = draw.CornerColorOffset != INVALID_OFFSET ?
            PerspectiveValue(weights.Value, corners[0].VertexColor, corners[1].VertexColor, corners[2].VertexColor) : float4(1.0f);
        result.V.WorldTangent = float4(0, 0, 0, 1);
        result.V.MotionPrev = PerspectiveValue(weights.Value, corners[0].MotionPrev, corners[1].MotionPrev, corners[2].MotionPrev);
        result.V.MotionNext = PerspectiveValue(weights.Value, corners[0].MotionNext, corners[1].MotionNext, corners[2].MotionNext);
        result.V.FlatWorldNormal = float3(0.0f);
        result.V.FaceOverlayFlags = 0u;
        result.V.MaterialIndex = MeshletPrimitiveMaterialIndex(scene, primitive);
        const float3 scale = float3(MeshletWorld(scene, draw).S);
        result.V.WorldScale = (scale.x + scale.y + scale.z) / 3.0f;
        result.ObjectId = instance.ObjectId;
        result.ElementId = instance.ElementIdOffset + face_id;
        result.InstanceFlags = instance.Flags;
        result.Topology = topology;
        result.PointCoord = PerspectiveValue(weights.Value, point_coords[0], point_coords[1], point_coords[2]);
        result.Valid = true;
        return result;
    }
    const uchar packed_first = BindlessBuffer(uchar, bindless.Buffer, pc.MeshletLocalTriangleSlot)[MeshletLocalTriangleOffset(meshlet) + resolved.LocalTriangle * 3u];
    const bool flat_face = (packed_first & MeshletGeometryEncoding_FlatTriangleBit) != 0u;

    MeshVaryings corners[3];
    for (uint corner = 0u; corner < 3u; ++corner) {
        const uint vertex_index = (triangle - primitive.FirstTriangle) * 3u + corner;
        const uint vertex_id = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + vertex_index];
        corners[corner] = VisibilityCorner(scene, draw, vertex_index, vertex_id, face_id, !flat_face, velocity_output);
    }
    const PerspectiveWeights weights = TriangleWeights(
        pixel, corners[0].Position, corners[1].Position, corners[2].Position, float2(view.ViewportSize)
    );
    result.V.Position = float4(pixel, 0.0f, 1.0f);
    result.V.WorldNormal = PerspectiveValue(weights.Value, corners[0].WorldNormal, corners[1].WorldNormal, corners[2].WorldNormal);
    result.V.WorldPosition = PerspectiveValue(weights.Value, corners[0].WorldPosition, corners[1].WorldPosition, corners[2].WorldPosition);
    result.V.Color = PerspectiveValue(weights.Value, corners[0].Color, corners[1].Color, corners[2].Color);
    result.V.VertexColor = draw.CornerColorOffset != INVALID_OFFSET ?
        PerspectiveValue(weights.Value, corners[0].VertexColor, corners[1].VertexColor, corners[2].VertexColor) : float4(1.0f);
    result.V.WorldTangent = draw.CornerTangentOffset != INVALID_OFFSET ?
        PerspectiveValue(weights.Value, corners[0].WorldTangent, corners[1].WorldTangent, corners[2].WorldTangent) : float4(0, 0, 0, 1);
    result.V.MotionPrev = PerspectiveValue(weights.Value, corners[0].MotionPrev, corners[1].MotionPrev, corners[2].MotionPrev);
    result.V.MotionNext = PerspectiveValue(weights.Value, corners[0].MotionNext, corners[1].MotionNext, corners[2].MotionNext);
    if (draw.CornerUvOffsets[0] != INVALID_OFFSET) {
        DecodeUv(result.V.TexCoord0, result.UvDx[0], result.UvDy[0], weights, corners[0].TexCoord0, corners[1].TexCoord0, corners[2].TexCoord0);
    }
    if (draw.CornerUvOffsets[1] != INVALID_OFFSET) {
        DecodeUv(result.V.TexCoord1, result.UvDx[1], result.UvDy[1], weights, corners[0].TexCoord1, corners[1].TexCoord1, corners[2].TexCoord1);
    }
    if (draw.CornerUvOffsets[2] != INVALID_OFFSET) {
        DecodeUv(result.V.TexCoord2, result.UvDx[2], result.UvDy[2], weights, corners[0].TexCoord2, corners[1].TexCoord2, corners[2].TexCoord2);
    }
    if (draw.CornerUvOffsets[3] != INVALID_OFFSET) {
        DecodeUv(result.V.TexCoord3, result.UvDx[3], result.UvDy[3], weights, corners[0].TexCoord3, corners[1].TexCoord3, corners[2].TexCoord3);
    }

    const Transform world = MeshletWorld(scene, draw);
    const MeshletFaceValues face = MeshletFace(scene, draw, primitive, instance, world, triangle, flat_face);
    result.V.FlatWorldNormal = face.FlatWorldNormal;
    result.V.FaceOverlayFlags = face.FaceOverlayFlags;
    result.V.MaterialIndex = face.MaterialIndex;
    result.V.WorldScale = face.WorldScale;
    result.ObjectId = face.ObjectId;
    result.ElementId = face.ElementId;
    result.InstanceFlags = instance.Flags;
    result.Topology = uint(MeshPrimitiveTopology_Triangle);
    result.PointCoord = float2(0.0f);
    result.Valid = true;
    return result;
}

inline DecodedVisibility DecodeWorkspaceVisibilityId(
    uint id, float2 pixel,
    device const BindlessSet &bindless,
    constant SceneViewUBO &view,
    constant ViewportTheme &theme,
    constant WorkspaceLights &workspace,
    VisibilityShadingPushConstants pc
) {
    DecodedVisibility result{};
    const ResolvedVisibility resolved = ResolveVisibilityId(id, bindless, view, theme, workspace, pc);
    if (!resolved.Valid) return result;

    const Scene scene{bindless, view, theme, workspace};
    const uchar packed_first = BindlessBuffer(uchar, bindless.Buffer, pc.MeshletLocalTriangleSlot)[
        MeshletLocalTriangleOffset(resolved.Meshlet) + resolved.LocalTriangle * 3u
    ];
    const bool flat_face = (packed_first & MeshletGeometryEncoding_FlatTriangleBit) != 0u;
    MeshVaryings corners[3];
    for (uint corner = 0u; corner < 3u; ++corner) {
        const uint vertex_index = (resolved.Triangle - resolved.Primitive.FirstTriangle) * 3u + corner;
        const uint vertex_id = scene.Indices(resolved.Draw.IndexSlotOffset.Slot)[resolved.Draw.IndexSlotOffset.Offset + vertex_index];
        corners[corner] = VisibilityWorkspaceCorner(
            scene, resolved.Draw, vertex_index, vertex_id, resolved.FaceId, !flat_face
        );
    }
    const PerspectiveWeights weights = TriangleWeights(
        pixel, corners[0].Position, corners[1].Position, corners[2].Position, float2(view.ViewportSize)
    );
    result.V.Position = float4(pixel, 0.0f, 1.0f);
    result.V.WorldNormal = PerspectiveValue(weights.Value, corners[0].WorldNormal, corners[1].WorldNormal, corners[2].WorldNormal);
    result.V.WorldPosition = PerspectiveValue(weights.Value, corners[0].WorldPosition, corners[1].WorldPosition, corners[2].WorldPosition);
    result.V.Color = float4(0.8f, 0.8f, 0.8f, 1.0f);
    const Transform world = MeshletWorld(scene, resolved.Draw);
    const MeshletFaceValues face = MeshletFace(
        scene, resolved.Draw, resolved.Primitive, resolved.Instance, world, resolved.Triangle, flat_face
    );
    result.V.FlatWorldNormal = face.FlatWorldNormal;
    result.V.FaceOverlayFlags = face.FaceOverlayFlags;
    result.Valid = true;
    return result;
}

inline DecodedVisibility DecodeVisibility(
    float2 pixel, texture2d<uint, access::read> visibility,
    device const BindlessSet &bindless,
    constant SceneViewUBO &view,
    constant ViewportTheme &theme,
    constant WorkspaceLights &workspace,
    VisibilityShadingPushConstants pc,
    bool velocity_output = false
) {
    return DecodeVisibilityId(
        visibility.read(uint2(pixel)).r, pixel, bindless, view, theme, workspace, pc, velocity_output
    );
}

#endif
