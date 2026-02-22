// Adapted from KhronosGroup/glTF-Sample-Renderer (punctual.glsl), pulled 2026-02-16.

#ifndef PUNCTUAL_GLSL
#define PUNCTUAL_GLSL

const float LIGHT_EPSILON = 1e-4;
const uint LIGHT_TYPE_DIRECTIONAL = 0u;
const uint LIGHT_TYPE_SPOT = 2u;

// Guard zero-length vectors / zero distance to avoid NaN/Inf.
vec3 safeNormalize(vec3 v, vec3 fallback) {
    const float len = length(v);
    return len > LIGHT_EPSILON ? (v / len) : fallback;
}

float getRangeAttenuation(float range, float distance_to_light) {
    const float safe_distance = max(distance_to_light, LIGHT_EPSILON);
    if (range <= 0.0) return 1.0 / pow(safe_distance, 2.0);
    return max(min(1.0 - pow(safe_distance / range, 4.0), 1.0), 0.0) / pow(safe_distance, 2.0);
}

float getSpotAttenuation(vec3 point_to_light, vec3 spot_direction, float outerConeCos, float innerConeCos) {
    const vec3 light_to_point = safeNormalize(-point_to_light, vec3(0.0, 0.0, 1.0));
    const vec3 spot_dir = safeNormalize(spot_direction, vec3(0.0, 0.0, -1.0));
    const float actualCos = dot(spot_dir, light_to_point);
    if (actualCos > outerConeCos) {
        if (actualCos < innerConeCos) {
            const float angular_attenuation = (actualCos - outerConeCos) / (innerConeCos - outerConeCos);
            return angular_attenuation * angular_attenuation;
        }
        return 1.0;
    }
    return 0.0;
}

vec3 getLightEmissionDirection(const WorldTransform wt) {
    // Emission axis is transform local -Z.
    const vec3 local_plus_z = quat_rotate(wt.Rotation, vec3(0.0, 0.0, 1.0));
    return -safeNormalize(local_plus_z, vec3(0.0, 0.0, 1.0));
}

vec3 getPointToLight(PunctualLight light, const WorldTransform wt, vec3 world_position, vec3 emission_direction) {
    // Directional point-to-light vectors are opposite emission direction.
    if (light.Type == LIGHT_TYPE_DIRECTIONAL) return -emission_direction;
    return wt.Position - world_position;
}

vec3 getLightIntensity(PunctualLight light, vec3 worldPosition, out vec3 L) {
    const WorldTransform wt = ModelBuffers[nonuniformEXT(light.TransformSlotOffset.Slot)].Models[light.TransformSlotOffset.Offset];
    const vec3 emission_direction = getLightEmissionDirection(wt);
    const vec3 point_to_light = getPointToLight(light, wt, worldPosition, emission_direction);

    L = safeNormalize(point_to_light, -emission_direction);

    float range_attenuation = 1.0;
    float spot_attenuation = 1.0;
    if (light.Type != LIGHT_TYPE_DIRECTIONAL) {
        range_attenuation = getRangeAttenuation(light.Range, length(point_to_light));
    }
    if (light.Type == LIGHT_TYPE_SPOT) {
        spot_attenuation = getSpotAttenuation(point_to_light, emission_direction, light.OuterConeCos, light.InnerConeCos);
    }
    return range_attenuation * spot_attenuation * light.Intensity * light.Color;
}

#endif
