// Adapted from KhronosGroup/glTF-Sample-Renderer (punctual.glsl), pulled 2026-02-16.

#ifndef PUNCTUAL_GLSL
#define PUNCTUAL_GLSL

float getRangeAttenuation(float range, float distanceToLight) {
    const float d = max(distanceToLight, 1e-4);
    const float invSq = 1.0 / (d * d);
    if (range <= 0.0) return invSq;
    const float x = d / range;
    const float rangeWindow = max(min(1.0 - pow(x, 4.0), 1.0), 0.0);
    return invSq * rangeWindow;
}

float getSpotAttenuation(vec3 pointToLight, vec3 spotDirection, float outerConeCos, float innerConeCos) {
    const vec3 lightToPoint = normalize(-pointToLight);
    const vec3 spotDir = normalize(spotDirection);
    const float actualCos = dot(spotDir, lightToPoint);
    if (innerConeCos <= outerConeCos + 1e-5) return actualCos > outerConeCos ? 1.0 : 0.0;
    if (actualCos <= outerConeCos) return 0.0;
    if (actualCos >= innerConeCos) return 1.0;
    return smoothstep(outerConeCos, innerConeCos, actualCos);
}

vec3 getLightIntensity(PunctualLight light, vec3 worldPosition, out vec3 L) {
    const WorldTransform wt = ModelBuffers[nonuniformEXT(light.TransformSlotOffset.Slot)].Models[light.TransformSlotOffset.Offset];
    const vec3 forward = quat_rotate(wt.Rotation, vec3(0.0, 0.0, 1.0));

    if (light.Type == 0u) {
        L = normalize(forward);
        return light.Color * light.Intensity;
    }

    const vec3 pointToLight = wt.Position - worldPosition;
    const float distanceToLight = length(pointToLight);
    L = distanceToLight > 1e-5 ? pointToLight / distanceToLight : vec3(0.0, 0.0, 1.0);

    float attenuation = getRangeAttenuation(light.Range, distanceToLight);
    if (light.Type == 2u) {
        attenuation *= getSpotAttenuation(L, -forward, light.OuterConeCos, light.InnerConeCos);
    }
    return light.Color * light.Intensity * attenuation;
}

#endif
