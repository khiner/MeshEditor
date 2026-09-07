#ifndef VISIBILITYMOTION_MSL
#define VISIBILITYMOTION_MSL

#include "VisibilityDecode.metal"
#include "MotionBlurTilesFlattenPushConstants.metal"
#include "MotionBlurShared.metal"

inline bool SameTransform(Transform a, Transform b) {
    return all(float3(a.P) == float3(b.P)) && all(float4(a.R) == float4(b.R)) && all(float3(a.S) == float3(b.S));
}

inline float3 ShutterPosition(const thread Scene &scene, DrawData draw, uint vertex_id) {
    float3 position = float3(scene.Vertices(draw.VertexSlot)[draw.VertexOffset + vertex_id].Position);
    ApplyMorphDeform(scene, draw, position, vertex_id, scene.View.MorphWeightsSlot);
    float3 normal = float3(0.0f);
    position = ApplyArmatureDeform(scene, draw, position, vertex_id, normal);
    return apply_object_pending_transform(scene, draw, trs_transform_point(MeshletWorld(scene, draw), position));
}

// Reject shutter endpoints behind the camera instead of producing unbounded or NaN streaks.
inline float2 ShutterMotion(float2 uv, float4 clip) {
    return clip.w > 1e-6f && all(isfinite(clip)) ? uv - ndc_to_uv(clip.xy / clip.w) : float2(0.0f);
}

inline float4 VisibilityMotion(
    const thread Scene &scene, const thread Scene &previous, const thread Scene &next,
    constant MotionBlurTilesFlattenPushConstants &pc, uint id, float depth, float2 uv
) {
    const float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    const float4 homogeneous = pc.InvViewProj.Unpack() * float4(ndc, depth, 1.0f);
    const float3 world = homogeneous.xyz / homogeneous.w;
    float4 prev_world = float4(world, 1.0f), next_world = prev_world;
    if (id == VisibilityBackground) {
        // An orthographic world background is a uniform direction, with no screen-space motion.
        if (pc.CameraMotion == 0u || scene.View.ScreenPixelScale < 0.0f) return float4(0.0f);
        prev_world = next_world = float4(WorldBackgroundDirection(scene, ndc), 0.0f);
    } else {
        const ResolvedVisibility resolved = ResolveVisibilityId(id, scene.B, scene.View, scene.Theme, scene.Workspace, pc.Visibility);
        const DrawData draw = resolved.Draw;
        const Transform current = MeshletWorld(scene, draw);
        const Transform prev = MeshletWorld(previous, draw), after = MeshletWorld(next, draw);
        const bool deformed = draw.BoneDeformOffset != INVALID_OFFSET || draw.MorphDeformOffset != INVALID_OFFSET;
        if (!deformed && SameTransform(current, prev) && SameTransform(current, after)) {
            if (pc.CameraMotion == 0u) return float4(0.0f);
            // Static geometry needs only depth reprojection, regardless of its tessellation.
        } else if (!deformed && scene.View.IsTransforming == 0u && all(abs(float3(current.S)) > 1e-20f)) {
            const float3 local = trs_inverse_transform_point(current, world);
            prev_world = float4(trs_transform_point(prev, local), 1.0f);
            next_world = float4(trs_transform_point(after, local), 1.0f);
        } else {
            const uint topology = MeshletPrimitiveTopology(resolved.Meshlet);
            const bool triangle = topology == MeshPrimitiveTopology_Triangle;
            const uint3 corners = MeshletCornerIds(scene.B, pc.Visibility.MeshletVertexSlot, pc.Visibility.MeshletLocalTriangleSlot,
                resolved.Meshlet, resolved.Primitive, resolved.Triangle, resolved.LocalTriangle);
            float4 clip[3];
            float3 prev_position[3], next_position[3];
            for (uint i = 0; i < 3; ++i) {
                const uint quad_corner = line_quad_corner((resolved.LocalTriangle & 1u) * 3u + i);
                const uint vertex_id = triangle ? scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + corners[i]] :
                    NonTriangleVertexId(scene.B, pc.Visibility.MeshletVertexSlot, resolved.Meshlet, topology, resolved.LocalTriangle / 2u, quad_corner);
                clip[i] = triangle ? MeshletPosition(scene, draw, current, vertex_id) :
                    NonTrianglePosition(scene, scene.B, pc.Visibility.MeshletVertexSlot, draw, resolved.Meshlet, topology, resolved.LocalTriangle / 2u, quad_corner);
                prev_position[i] = ShutterPosition(previous, draw, vertex_id);
                next_position[i] = ShutterPosition(next, draw, vertex_id);
            }
            const float3 weights = TriangleWeights(uv * float2(scene.View.ViewportSize), clip[0], clip[1], clip[2], float2(scene.View.ViewportSize)).Value;
            prev_world = float4(PerspectiveValue(weights, prev_position[0], prev_position[1], prev_position[2]), 1.0f);
            next_world = float4(PerspectiveValue(weights, next_position[0], next_position[1], next_position[2]), 1.0f);
        }
    }
    return float4(-ShutterMotion(uv, previous.ViewProj() * prev_world), ShutterMotion(uv, next.ViewProj() * next_world));
}

#endif
