// Adapted from KhronosGroup/glTF-Sample-Renderer (punctual.glsl)

#ifndef PUNCTUAL_MSL
#define PUNCTUAL_MSL

#include "Bindless.metal"
#include "PunctualLightType.metal"

constant float LIGHT_EPSILON = 1e-4f;

// Guard zero-length vectors / zero distance to avoid NaN/Inf.
inline float3 safeNormalize(float3 v, float3 fallback) {
    const float len = length(v);
    return len > LIGHT_EPSILON ? (v / len) : fallback;
}

inline float getRangeAttenuation(float range, float distance_to_light) {
    const float safe_distance = max(distance_to_light, LIGHT_EPSILON);
    if (range <= 0.0f) return 1.0f / pow(safe_distance, 2.0f);
    return max(min(1.0f - pow(safe_distance / range, 4.0f), 1.0f), 0.0f) / pow(safe_distance, 2.0f);
}

inline float getSpotAttenuation(float3 point_to_light, float3 spot_direction, float outerConeCos, float innerConeCos) {
    const float3 light_to_point = safeNormalize(-point_to_light, float3(0.0f, 0.0f, 1.0f));
    const float3 spot_dir = safeNormalize(spot_direction, float3(0.0f, 0.0f, -1.0f));
    const float actualCos = dot(spot_dir, light_to_point);
    if (actualCos > outerConeCos) {
        if (actualCos < innerConeCos) {
            const float angular_attenuation = (actualCos - outerConeCos) / (innerConeCos - outerConeCos);
            return angular_attenuation * angular_attenuation;
        }
        return 1.0f;
    }
    return 0.0f;
}

inline float3 getLightEmissionDirection(Transform wt) {
    // Emission axis is transform local -Z.
    const float3 local_plus_z = quat_rotate(float4(wt.R), float3(0.0f, 0.0f, 1.0f));
    return -safeNormalize(local_plus_z, float3(0.0f, 0.0f, 1.0f));
}

inline float3 getPointToLight(PunctualLight light, Transform wt, float3 world_position, float3 emission_direction) {
    // Directional point-to-light vectors are opposite emission direction.
    if (light.Type == PunctualLightType_Directional) return -emission_direction;
    return float3(wt.P) - world_position;
}

template<typename SetT>
inline float3 getLightIntensity(
    const thread SceneT<SetT> &scene, PunctualLight light, float3 worldPosition,
    thread float3 &L, thread float3 &point_to_light
) {
    const Transform wt = scene.Models(light.TransformSlotOffset.Slot)[light.TransformSlotOffset.Offset];
    const float3 emission_direction = getLightEmissionDirection(wt);
    point_to_light = getPointToLight(light, wt, worldPosition, emission_direction);

    L = safeNormalize(point_to_light, -emission_direction);

    float range_attenuation = 1.0f;
    float spot_attenuation = 1.0f;
    if (light.Type != PunctualLightType_Directional) {
        range_attenuation = getRangeAttenuation(light.Range, length(point_to_light));
    }
    if (light.Type == PunctualLightType_Spot) {
        spot_attenuation = getSpotAttenuation(point_to_light, emission_direction, light.OuterConeCos, light.InnerConeCos);
    }
    return range_attenuation * spot_attenuation * light.Intensity * float3(light.Color);
}

#endif
