#ifndef OBJECT_EXTRAS_TRANSFORM_MSL
#define OBJECT_EXTRAS_TRANSFORM_MSL

#include "Bindless.metal"
#include "brdf.metal"

constant uint VCLASS_NONE = 0u;
constant uint VCLASS_BILLBOARD = 1u;
constant uint VCLASS_SPOT_CONE = 2u;
constant uint VCLASS_SCREENSPACE = 3u;
constant uint VCLASS_GROUNDPOINT = 4u;

template<typename SetT>
inline float3 ObjectExtrasWorldPos(const thread SceneT<SetT> &scene, Transform world, float3 vert_pos, uint vertex_class) {
    const float3 object_origin = float3(world.P);
    const float3 camera_position = float3(scene.View.CameraPosition);
    if (vertex_class == VCLASS_BILLBOARD || vertex_class == VCLASS_SCREENSPACE) {
        const float3 to_camera_vec = camera_position - object_origin;
        const float dist = length(to_camera_vec);
        const float3 to_camera = dist > 0.0f ? to_camera_vec / dist : float3(0, 0, 1);
        const float3 up = abs(to_camera.y) > 0.999f ? float3(0, 0, 1) : float3(0, 1, 0);
        const float3 right = normalize(cross(up, to_camera));
        const float3 basis_up = cross(to_camera, right);
        // Positive ScreenPixelScale applies perspective distance; negative values give an orthographic pixel scale.
        float scale = 1.0f;
        if (vertex_class == VCLASS_SCREENSPACE) {
            scale = scene.View.ScreenPixelScale > 0.0f ? dist * scene.View.ScreenPixelScale : -scene.View.ScreenPixelScale;
        }
        return object_origin + (right * vert_pos.x + basis_up * vert_pos.y) * scale;
    }
    if (vertex_class == VCLASS_SPOT_CONE) {
        // Collapse non-silhouette spot-cone spokes to the apex, matching Blender's overlay_extra_vert.glsl.
        const float4x4 M = trs_to_mat4(world);
        const float3 world_pos = (M * float4(vert_pos, 1.0f)).xyz;

        const float2 perp = float2(vert_pos.y, -vert_pos.x);
        const float incr_angle = (360.0f / 32.0f) * (Pi / 180.0f);
        const float2 slope = float2(cos(incr_angle), sin(incr_angle));
        const float3 p0 = (M * float4(vert_pos.xy * slope.x + perp * slope.y, vert_pos.z, 1.0f)).xyz;
        const float3 p1 = (M * float4(vert_pos.xy * slope.x - perp * slope.y, vert_pos.z, 1.0f)).xyz;

        const float3 edge = object_origin - world_pos;
        const float3 n0 = normalize(cross(edge, p0 - world_pos));
        const float3 n1 = normalize(cross(edge, world_pos - p1));

        const float3 V = normalize(camera_position - world_pos);

        const bool facing0 = dot(n0, V) > 0.0f;
        const bool facing1 = dot(n1, V) > 0.0f;
        return (facing0 == facing1) ? object_origin : world_pos;
    }
    if (vertex_class == VCLASS_GROUNDPOINT) {
        // Interpolate vert_pos.y from the ground plane to the object origin while keeping the ground diamond pixel-sized.
        const float scale = scene.View.ScreenPixelScale > 0.0f
            ? length(camera_position - object_origin) * scene.View.ScreenPixelScale
            : -scene.View.ScreenPixelScale;
        return float3(object_origin.x + vert_pos.x * scale, object_origin.y * vert_pos.y, object_origin.z + vert_pos.z * scale);
    }

    return trs_transform_point(world, vert_pos);
}

#endif
