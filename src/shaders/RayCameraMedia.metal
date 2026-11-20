#include "RayTracing.h"

// A boundary's first hit from an interior point is an exit. Only volume instances use mask bit 2.
kernel void RayCameraMedia(
    device const BindlessSet &bindless [[buffer(0)]], constant SceneViewUBO &view [[buffer(2)]],
    constant ViewportThemeUbo &theme [[buffer(3)]], constant WorkspaceLights &workspace [[buffer(4)]],
    instance_acceleration_structure acceleration [[buffer(5)]], device const RayTracingInstance *instances [[buffer(6)]],
    device const uint *triangles [[buffer(7)]], device float *environment [[buffer(8)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const RayGeometry rt{acceleration, instances, triangles};
    RayMediumState media{};
    float distances[RayMediumCapacity];
    const float3 direction = normalize(float3(0.75487766f, 0.312349f, 0.57821f));
    for (uint instance = 0u; instance < view.RayInstanceCount; ++instance) {
        if (instances[instance].VolumeMaterial == InvalidOffset) continue;
        ray r{float3(view.CameraPosition), direction, 0.0f, INFINITY};
        intersector<triangle_data, instancing> query;
        query.assume_geometry_type(geometry_type::triangle);
        while (r.min_distance < r.max_distance) {
            const auto hit = query.intersect(r, acceleration, 2u);
            if (hit.type == intersection_type::none) break;
            if (hit.instance_id == instance && !RayElementHidden(scene, rt, hit.instance_id, hit.primitive_id)) {
                const auto surface = DecodeRaySurface(scene, rt, hit.instance_id, hit.primitive_id, hit.triangle_barycentric_coord);
                if (dot(surface.GeometricNormal, direction) > 0.0f && media.Count < RayMediumCapacity) {
                    uint at = media.Count++;
                    while (at > 0u && distances[at - 1u] < hit.distance) {
                        distances[at] = distances[at - 1u];
                        media.Materials[at] = media.Materials[at - 1u];
                        media.Objects[at] = media.Objects[at - 1u];
                        --at;
                    }
                    distances[at] = hit.distance;
                    media.Materials[at] = instances[instance].VolumeMaterial;
                    media.Objects[at] = surface.ObjectId;
                }
                break;
            }
            r.min_distance = nextafter(hit.distance, INFINITY);
        }
    }
    *reinterpret_cast<device RayMediumState *>(environment + RayEnvironmentCount) = media;
}
