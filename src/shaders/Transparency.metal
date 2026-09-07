#ifndef TRANSPARENCY_MSL
#define TRANSPARENCY_MSL

#include "Varyings.metal"

// Keep the nearest four surfaces in raster-depth order. Deeper layers retain their
// coverage and mean radiance in a weighted tail rather than disappearing at capacity.
constant uint TransparencyLayers = 4u;
struct TransparencyValues {
    half4 Colors [[raster_order_group(0)]] [TransparencyLayers];
    float Depths [[raster_order_group(0)]] [TransparencyLayers];
    half4 Tail [[raster_order_group(0)]];
};
struct TransparencyStore { TransparencyValues Values [[imageblock_data]]; };

fragment TransparencyStore TransparencyInitFragment() {
    TransparencyStore out{};
    for (uint i = 0; i < TransparencyLayers; ++i) out.Values.Depths[i] = INFINITY;
    return out;
}

inline bool TransparencyBefore(float depth, half4 color, float other_depth, half4 other_color) {
    if (depth != other_depth) return depth < other_depth;
    for (uint i = 0; i < 4u; ++i) if (color[i] != other_color[i]) return color[i] < other_color[i];
    return false;
}

inline TransparencyStore StoreTransparency(TransparencyValues values, float4 color, float depth) {
    if (color.a <= 0.0f) return {values};
    half4 incoming = half4(float4(color.rgb * color.a, color.a));
    for (uint i = 0; i < TransparencyLayers; ++i) {
        if (TransparencyBefore(depth, incoming, values.Depths[i], values.Colors[i])) {
            const half4 displaced_color = values.Colors[i];
            const float displaced_depth = values.Depths[i];
            values.Colors[i] = incoming;
            values.Depths[i] = depth;
            incoming = displaced_color;
            depth = displaced_depth;
            if (!isfinite(depth)) return {values};
        }
    }
    if (isfinite(depth)) {
        const float alpha = float(incoming.a);
        if (alpha > 0.0f) {
            // Optical density supplies both the color weight and total transmittance,
            // keeping the explicit image block within Metal's 64-byte sample limit.
            const float weight = -log(max(1.0f - alpha, 1e-6f));
            const float total = float(values.Tail.a) + weight;
            const float3 mean = mix(float3(values.Tail.rgb), float3(incoming.rgb) / alpha, weight / total);
            values.Tail = half4(float4(mean, min(total, 65504.0f)));
        }
    }
    return {values};
}

fragment half4 TransparencyResolveFragment(
    TransparencyValues values [[imageblock_data]],
    half4 background [[color(0), raster_order_group(0)]]
) {
    float4 color = float4(background);
    const float transmittance = exp(-float(values.Tail.a));
    const float tail_alpha = 1.0f - transmittance;
    if (values.Tail.a > 0.0f) {
        color = float4(float3(values.Tail.rgb) * tail_alpha, tail_alpha) + transmittance * color;
    }
    for (uint i = TransparencyLayers; i > 0u; --i) {
        const float4 layer = float4(values.Colors[i - 1u]);
        if (layer.a > 0.0f) color = layer + (1.0f - layer.a) * color;
    }
    return half4(color);
}

#endif
