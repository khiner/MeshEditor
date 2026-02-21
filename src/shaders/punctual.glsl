// Adapted from KhronosGroup/glTF-Sample-Renderer (punctual.glsl), pulled 2026-02-16.

#ifndef PUNCTUAL_GLSL
#define PUNCTUAL_GLSL

float getRangeAttenuation(float range, float distance_to_light) {
    if (range <= 0.0) return 1.0 / pow(distance_to_light, 2.0);
    return max(min(1.0 - pow(distance_to_light / range, 4.0), 1.0), 0.0) / pow(distance_to_light, 2.0);
}

float getSpotAttenuation(vec3 pointToLight, vec3 spotDirection, float outerConeCos, float innerConeCos) {
    const float actualCos = dot(normalize(spotDirection), normalize(-pointToLight));
    if (actualCos > outerConeCos) {
        if (actualCos < innerConeCos) {
            const float angular_attenuation = (actualCos - outerConeCos) / (innerConeCos - outerConeCos);
            return angular_attenuation * angular_attenuation;
        }
        return 1.0;
    }
    return 0.0;
}

vec3 getLightIntensity(PunctualLight light, vec3 worldPosition, out vec3 L) {
    const WorldTransform wt = ModelBuffers[nonuniformEXT(light.TransformSlotOffset.Slot)].Models[light.TransformSlotOffset.Offset];
    const vec3 forward = quat_rotate(wt.Rotation, vec3(0.0, 0.0, 1.0));

    const vec3 point_to_light = light.Type == 0u ? -forward : wt.Position - worldPosition;
    L = normalize(point_to_light);

    float range_attenuation = 1.0;
    float spot_attenuation = 1.0;
    if (light.Type != 0u) {
        range_attenuation = getRangeAttenuation(light.Range, length(point_to_light));
    }
    if (light.Type == 2u) {
        spot_attenuation = getSpotAttenuation(point_to_light, forward, light.OuterConeCos, light.InnerConeCos);
    }
    return range_attenuation * spot_attenuation * light.Intensity * light.Color;
}

#endif
