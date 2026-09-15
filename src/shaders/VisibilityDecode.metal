#ifndef VISIBILITY_DECODE_MSL
#define VISIBILITY_DECODE_MSL

#include "MeshletResolve.metal"
#include "Varyings.metal"
#include "gpu/VisibilityShadingPushConstants.h"
#include "MeshletNonTriangle.metal"
#include "VertexTransform.metal"
#include "gpu/VisibilityId.h"

constant uint VisibilityBackground = InvalidOffset;
constant uint VisibilityTriangleMask = (1u << uint(VisibilityId::TriangleBits)) - 1u;
constant uint VisibilityIndexMask = (1u << uint(VisibilityId::IndexBits)) - 1u;

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
    constant SceneViewUBO &view, VisibilityShadingPushConstants pc
) {
    float4 clip[3];
    float4 vertex_color[3];
    float2 point_coord[3]{};
    float3 world_normal[3]{};
    float3 world_position[3];
    const uint topology = MeshletPrimitiveTopology(resolved.Meshlet);
    const bool triangle_topology = topology == uint(MeshPrimitiveTopology::Triangle);
    const uint logical_element = resolved.LocalTriangle / 2u;
    const Transform world = MeshletWorld(scene, resolved.Draw);
    const uint3 corner_ids = MeshletCornerIds(
        scene.B, pc.MeshletVertexSlot, pc.MeshletLocalTriangleSlot, resolved.Meshlet, resolved.Primitive,
        resolved.Triangle, resolved.LocalTriangle
    );
    for (uint corner = 0u; corner < 3u; ++corner) {
        const uint quad_corner = line_quad_corner((resolved.LocalTriangle & 1u) * 3u + corner);
        const uint vertex_index = triangle_topology ?
            corner_ids[corner] :
            NonTriangleVertexId(
                scene.B, pc.MeshletVertexSlot, resolved.Meshlet, topology, logical_element, quad_corner
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
            scene, scene.B, pc.MeshletVertexSlot, resolved.Draw, resolved.Meshlet,
            topology, logical_element, quad_corner
        );
        vertex_color[corner] = resolved.Draw.CornerColorOffset != InvalidOffset ?
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
    const thread VisibilityCoverageValues &coverage, uint uv_set, VisibilityShadingPushConstants pc
) {
    if (MeshletPrimitiveTopology(resolved.Meshlet) != uint(MeshPrimitiveTopology::Triangle)) return {};
    const uint set = min(uv_set, 3u);
    const uint offset = resolved.Draw.CornerUvOffsets[set];
    float2 uv[3]{};
    if (offset != InvalidOffset) {
        device const packed_float2 *uvs = scene.CornerUvs(scene.View.CornerUvSlot);
        const uint3 corner_ids = MeshletCornerIds(
            scene.B, pc.MeshletVertexSlot, pc.MeshletLocalTriangleSlot, resolved.Meshlet, resolved.Primitive,
            resolved.Triangle, resolved.LocalTriangle
        );
        for (uint corner = 0u; corner < 3u; ++corner) uv[corner] = float2(uvs[offset + corner_ids[corner]]);
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
    const uint visible_index = (id >> uint(VisibilityId::TriangleBits)) & VisibilityIndexMask;
    const VisibleMeshlet visible = BindlessBuffer(VisibleMeshlet, bindless.Buffer, pc.VisibleMeshletSlot)[visible_index];
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
    if (MeshletCoarse(result.Meshlet)) return result;
    const Scene scene{bindless, view, theme, workspace};
    const uint topology = MeshletPrimitiveTopology(result.Meshlet);
    const uint logical_element = topology == uint(MeshPrimitiveTopology::Triangle) ?
        result.LocalTriangle : result.LocalTriangle / 2u;
    result.Triangle = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot)[
        result.Meshlet.TriangleOffset + logical_element
    ];
    result.FaceId = topology == uint(MeshPrimitiveTopology::Triangle) ?
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

// `attributes` adds the interpolated colors, UVs, and tangents that lit shading reads.
inline DecodedVisibility DecodeVisibilityId(
    uint id, float2 pixel,
    device const BindlessSet &bindless,
    constant SceneViewUBO &view,
    constant ViewportTheme &theme,
    constant WorkspaceLights &workspace,
    VisibilityShadingPushConstants pc,
    bool attributes = true
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
    if (topology != uint(MeshPrimitiveTopology::Triangle)) {
        const uint logical_element = resolved.LocalTriangle / 2u;
        MeshVaryings corners[3];
        float2 point_coords[3];
        for (uint corner = 0u; corner < 3u; ++corner) {
            const uint quad_corner = line_quad_corner((resolved.LocalTriangle & 1u) * 3u + corner);
            const uint vertex_id = NonTriangleVertexId(
                bindless, pc.MeshletVertexSlot, meshlet, topology, logical_element, quad_corner
            );
            corners[corner] = TransformVertex(scene, draw, vertex_id, vertex_id, vertex_id, false, true);
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
        // Alpha zero marks an unselected instance for fill recoloring during shading.
        result.V.Color = view.InteractionMode == InteractionMode::Object && view.ShowOverlays != 0u ?
            scene.ObjectSelectionColor(scene.InstanceState(draw), float4(0.0f)) : float4(0.0f);
        result.V.VertexColor = draw.CornerColorOffset != InvalidOffset ?
            PerspectiveValue(weights.Value, corners[0].VertexColor, corners[1].VertexColor, corners[2].VertexColor) : float4(1.0f);
        result.V.WorldTangent = float4(0, 0, 0, 1);
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
    const bool coarse = MeshletCoarse(meshlet);
    const bool flat_face = (packed_first & uint(MeshletGeometryEncoding::FlatTriangleBit)) != 0u;

    const MeshletTriangleCorners triangle_corners = ResolveMeshletCorners(
        scene, draw, pc.MeshletVertexSlot, pc.MeshletLocalTriangleSlot, meshlet, primitive, triangle, resolved.LocalTriangle
    );
    MeshVaryings corners[3];
    for (uint corner = 0u; corner < 3u; ++corner) {
        corners[corner] = TransformVertex(
            scene, draw, triangle_corners.CornerIds[corner], triangle_corners.CornerIds[corner], triangle_corners.VertexIds[corner],
            false, !flat_face, coarse, triangle_corners.CoarseNormal
        );
    }
    const PerspectiveWeights weights = TriangleWeights(
        pixel, corners[0].Position, corners[1].Position, corners[2].Position, float2(view.ViewportSize)
    );
    result.V.Position = float4(pixel, 0.0f, 1.0f);
    result.V.WorldNormal = PerspectiveValue(weights.Value, corners[0].WorldNormal, corners[1].WorldNormal, corners[2].WorldNormal);
    result.V.WorldPosition = PerspectiveValue(weights.Value, corners[0].WorldPosition, corners[1].WorldPosition, corners[2].WorldPosition);
    result.V.Color = attributes ? PerspectiveValue(weights.Value, corners[0].Color, corners[1].Color, corners[2].Color) : corners[0].Color;
    if (attributes) {
        result.V.VertexColor = draw.CornerColorOffset != InvalidOffset ?
            PerspectiveValue(weights.Value, corners[0].VertexColor, corners[1].VertexColor, corners[2].VertexColor) : float4(1.0f);
        result.V.WorldTangent = draw.CornerTangentOffset != InvalidOffset ?
            PerspectiveValue(weights.Value, corners[0].WorldTangent, corners[1].WorldTangent, corners[2].WorldTangent) : float4(0, 0, 0, 1);
        if (draw.CornerUvOffsets[0] != InvalidOffset) {
            DecodeUv(result.V.TexCoord0, result.UvDx[0], result.UvDy[0], weights, corners[0].TexCoord0, corners[1].TexCoord0, corners[2].TexCoord0);
        }
        if (draw.CornerUvOffsets[1] != InvalidOffset) {
            DecodeUv(result.V.TexCoord1, result.UvDx[1], result.UvDy[1], weights, corners[0].TexCoord1, corners[1].TexCoord1, corners[2].TexCoord1);
        }
        if (draw.CornerUvOffsets[2] != InvalidOffset) {
            DecodeUv(result.V.TexCoord2, result.UvDx[2], result.UvDy[2], weights, corners[0].TexCoord2, corners[1].TexCoord2, corners[2].TexCoord2);
        }
        if (draw.CornerUvOffsets[3] != InvalidOffset) {
            DecodeUv(result.V.TexCoord3, result.UvDx[3], result.UvDy[3], weights, corners[0].TexCoord3, corners[1].TexCoord3, corners[2].TexCoord3);
        }
    }

    const Transform world = MeshletWorld(scene, draw);
    const MeshletFaceValues face = coarse ? MeshletCoarseFace(scene, primitive, instance, world) :
                                            MeshletFace(scene, draw, primitive, instance, world, triangle, flat_face);
    result.V.FlatWorldNormal = face.FlatWorldNormal;
    result.V.FaceOverlayFlags = face.FaceOverlayFlags;
    result.V.MaterialIndex = face.MaterialIndex;
    result.V.WorldScale = face.WorldScale;
    result.ObjectId = face.ObjectId;
    result.ElementId = face.ElementId;
    result.InstanceFlags = instance.Flags;
    result.Topology = uint(uint(MeshPrimitiveTopology::Triangle));
    result.PointCoord = float2(0.0f);
    result.Valid = true;
    return result;
}

inline DecodedVisibility DecodeVisibility(
    float2 pixel, texture2d<uint, access::read> visibility,
    device const BindlessSet &bindless,
    constant SceneViewUBO &view,
    constant ViewportTheme &theme,
    constant WorkspaceLights &workspace,
    VisibilityShadingPushConstants pc
) {
    return DecodeVisibilityId(
        visibility.read(uint2(pixel)).r, pixel, bindless, view, theme, workspace, pc
    );
}

#endif
