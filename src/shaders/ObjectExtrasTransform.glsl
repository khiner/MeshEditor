// Shared vertex-class transform for object extras (cameras, lights, empties).
// Requires: Bindless.glsl included, `draw`, `vert`, `world` already loaded.

const uint VCLASS_NONE = 0u;
const uint VCLASS_BILLBOARD = 1u;
const uint VCLASS_SPOT_CONE = 2u;
const uint VCLASS_SCREENSPACE = 3u;
const uint VCLASS_GROUNDPOINT = 4u;

uint GetVertexClass(DrawData draw, uint idx) {
    if (draw.VertexClass.Slot != INVALID_SLOT) {
        return uint(VertexClassBuffers[draw.VertexClass.Slot].Classes[draw.VertexClass.Offset + idx]);
    }
    return VCLASS_NONE;
}

vec3 ObjectExtrasWorldPos(DrawData draw, Vertex vert, WorldMatrix world, uint idx) {
    const uint vertex_class = GetVertexClass(draw, idx);
    const vec3 object_origin = vec3(world.M[3]);
    if (vertex_class == VCLASS_BILLBOARD || vertex_class == VCLASS_SCREENSPACE) {
        // Billboard: rotate local XY-plane vertices to face the camera.
        const vec3 to_camera_vec = SceneViewUBO.CameraPosition - object_origin;
        const float dist = length(to_camera_vec);
        const vec3 to_camera = dist > 0.0 ? to_camera_vec / dist : vec3(0, 0, 1);
        const vec3 up = abs(to_camera.y) > 0.999 ? vec3(0, 0, 1) : vec3(0, 1, 0);
        const vec3 right = normalize(cross(up, to_camera));
        const vec3 basis_up = cross(to_camera, right);
        // Screenspace: maintain constant pixel size regardless of distance/zoom.
        // ScreenPixelScale > 0 means perspective (multiply by distance),
        // ScreenPixelScale < 0 means orthographic (use abs value directly).
        float scale = 1.0;
        if (vertex_class == VCLASS_SCREENSPACE) {
            scale = SceneViewUBO.ScreenPixelScale > 0.0 ? dist * SceneViewUBO.ScreenPixelScale : -SceneViewUBO.ScreenPixelScale;
        }

        return object_origin + (right * vert.Position.x + basis_up * vert.Position.y) * scale;
    }
    if (vertex_class == VCLASS_SPOT_CONE) {
        // Spot cone silhouette detection (matches Blender's overlay_extra_vert.glsl).
        // Only spoke edges at facing transitions (silhouette) are visible; others collapse to the apex.
        const vec3 world_pos = vec3(world.M * vec4(vert.Position, 1.0));

        // Compute adjacent points on the cone base circle using 2D rotation (no atan needed).
        const vec2 perp = vec2(vert.Position.y, -vert.Position.x);
        const float incr_angle = radians(360.0 / 32.0);
        const vec2 slope = vec2(cos(incr_angle), sin(incr_angle));
        const vec3 p0 = vec3(world.M * vec4(vert.Position.xy * slope.x + perp * slope.y, vert.Position.z, 1.0));
        const vec3 p1 = vec3(world.M * vec4(vert.Position.xy * slope.x - perp * slope.y, vert.Position.z, 1.0));

        // Face normals of the two adjacent cone triangles.
        const vec3 edge = object_origin - world_pos;
        const vec3 n0 = normalize(cross(edge, p0 - world_pos));
        const vec3 n1 = normalize(cross(edge, world_pos - p1));

        // View direction from surface to camera.
        const vec3 V = normalize(SceneViewUBO.CameraPosition - world_pos);

        // Discard non-silhouette edges by collapsing to the apex.
        const bool facing0 = dot(n0, V) > 0.0;
        const bool facing1 = dot(n1, V) > 0.0;
        return (facing0 == facing1) ? object_origin : world_pos;
    }
    if (vertex_class == VCLASS_GROUNDPOINT) {
        // Ground-plane projection. vert.Position.y interpolates between ground (0) and object origin (1).
        // vert.Position.xz are screenspace-scaled offsets (constant pixel size for ground diamond).
        const float scale = SceneViewUBO.ScreenPixelScale > 0.0
            ? length(SceneViewUBO.CameraPosition - object_origin) * SceneViewUBO.ScreenPixelScale
            : -SceneViewUBO.ScreenPixelScale;
        return vec3(object_origin.x + vert.Position.x * scale, object_origin.y * vert.Position.y, object_origin.z + vert.Position.z * scale);
    }

    return vec3(world.M * vec4(vert.Position, 1.0));
}
