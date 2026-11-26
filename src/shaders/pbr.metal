// Fast math that keeps Inf and NaN semantics.
#pragma METAL fp math_mode(relaxed)
#include "PbrLighting.h"
#include "PreviewSurface.h"
#include "Transparency.metal"

// Scene lights and image-based lighting of a rasterized surface.
inline float4 ShadeRaster(const thread Scene &scene, constant SceneViewUBO &view, const thread PbrContext &ctx, uint topology, float2 point_coord) {
    float4 base_color, unlit;
    if (ShadeUnlit<true>(scene, view, ctx, topology, point_coord, base_color, unlit)) return unlit;
    const PbrClosure cl = EvaluatePbrMaterial<true>(scene, view, ctx, base_color, ctx.LightLut);
    const PbrAreaTerms area = AreaTerms(view, cl, ctx.LightLut);
    float3 direct(0);
    ForEachLight(scene, view, ctx, [&](LightRecord, thread PbrLight &light) __attribute__((always_inline)) {
        const PbrLightColor l = ReflectLight(cl, light, area, ctx.LightLut, true);
        direct += l.Diffuse + l.Glossy;
    });
    const PbrImageLight image = ImageLight(scene, view, ctx, cl);
    return PbrSurfaceColor<true>(scene, view, ctx, cl, direct + image.Diffuse + image.Glossy, image.Occlusion);
}

// Shades a transparent fragment, with ray-traced shadows under alpha blending and with raster lighting otherwise, and stores it in its pixel's transparency layers.
template<typename Varyings>
inline TransparencyStore ShadeTransparent(Varyings meshlet_in, TransparencyValues values, uint topology, uint pattern_word, device const BindlessSet &bindless, constant SceneViewUBO &view, constant ViewportThemeUbo &theme,
    constant WorkspaceLights &workspace, instance_acceleration_structure acceleration, device const RayTracingInstance *instances, device const uint *triangles, device const float *environment, texture2d_array<float> light_lut) {
    const MeshVaryings in = FromMeshletVaryings(meshlet_in);
    const Scene scene{bindless, view, theme, workspace};
    PbrContext ctx{scene, in};
    ctx.LightLut = light_lut;
    ctx.PatternWord = pattern_word;
    float4 shaded;
    if (EnableAlphaBlend) {
        const RayScene rays{acceleration, instances, triangles, environment, light_lut};
        const RayMediumState media = RayCameraState(rays);
        ctx.Media = &media;
        ctx.Opaque = ctx.RayTracing = true;
        ctx.GeometricNormal = normalize(cross(dfdy(in.WorldPosition), dfdx(in.WorldPosition)));
        shaded = ShadeSecondary<true>(scene, ctx, topology, meshlet_in.PointCoord, rays);
        shaded.rgb *= RayMediumAttenuation(scene, media, distance(float3(view.CameraPosition), in.WorldPosition));
    } else shaded = ShadeRaster(scene, view, ctx, topology, meshlet_in.PointCoord);
    return StoreTransparency(values, shaded, in.Position.z);
}
// The arguments of a transparent fragment, whose ray tracing arguments a frame binds under alpha blending.
#define TRANSPARENT_FRAGMENT_ARGS TransparencyValues values [[imageblock_data]], SCENE_ARGS, instance_acceleration_structure acceleration [[buffer(5), function_constant(EnableAlphaBlend)]], device const RayTracingInstance *instances [[buffer(6), function_constant(EnableAlphaBlend)]], device const uint *triangles [[buffer(7), function_constant(EnableAlphaBlend)]], device const float *environment [[buffer(8), function_constant(EnableAlphaBlend)]], texture2d_array<float> light_lut [[texture(15)]]

fragment TransparencyStore PbrTransparentFragment(MeshletVertexVaryings meshlet_in [[stage_in]], TRANSPARENT_FRAGMENT_ARGS) {
    const uint topology = NonTriangleTopology ? meshlet_in.Topology : uint(MeshPrimitiveTopology::Triangle);
    return ShadeTransparent(meshlet_in, values, topology, 0u, bindless, view, theme, workspace, acceleration, instances, triangles, environment, light_lut);
}

// The transparent fragments of the materials that compute inputs, each reading the words the pattern list pass computed at it.
fragment TransparencyStore PbrTransparentPatternFragment(MeshletPatternVaryings meshlet_in [[stage_in]], TRANSPARENT_FRAGMENT_ARGS) {
    const uint pattern_word = FragmentPatternWord(bindless, view, uint2(meshlet_in.Position.xy), meshlet_in.Key);
    return ShadeTransparent(meshlet_in, values, uint(MeshPrimitiveTopology::Triangle), pattern_word, bindless, view, theme, workspace, acceleration, instances, triangles, environment, light_lut);
}

// The material preview's lit sphere, whose pixels of a material that computes inputs read the words the preview's pattern pass computed.
fragment float4 PbrPreviewFragment(
    QuadVaryings quad [[stage_in]],
    texture2d<uint, access::read> triangles [[texture(0)]],
    SCENE_ARGS,
    device const PreviewCorner *corners [[buffer(BufferIndex_PreviewCorners)]],
    constant MaterialPreviewPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    texture2d_array<float> light_lut [[texture(15)]]
) {
    const uint2 pixel = uint2(quad.Position.xy);
    const uint triangle = triangles.read(pixel).r;
    if (triangle == VisibilityBackground) discard_fragment();
    const DecodedSurface decoded = DecodePreviewSurface(quad.Position.xy, triangle, corners, view, pc.Material);
    const Scene scene{bindless, view, theme, workspace};
    PbrContext ctx{scene, decoded.V};
    ctx.LightLut = light_lut;
    ctx.SetSurface(decoded, PixelPatternWord(view, decoded.Topology, pixel));
    return ShadeRaster(scene, view, ctx, uint(MeshPrimitiveTopology::Triangle), decoded.PointCoord);
}

fragment float4 PbrVisibilityFragment(
    QuadVaryings quad [[stage_in]],
    texture2d<uint, access::read> visibility [[texture(0)]],
    SCENE_ARGS,
    constant VisibilityShadingPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    texture2d_array<float> light_lut [[texture(15)]]
) {
    DecodedSurface decoded = DecodeVisibility(quad.Position.xy, visibility, bindless, view, theme, workspace, pc);
    if (!decoded.Valid) discard_fragment();
    const Scene scene{bindless, view, theme, workspace};
    PbrContext ctx{scene, decoded.V};
    ctx.LightLut = light_lut;
    ctx.SetSurface(decoded, PixelPatternWord(view, decoded.Topology, uint2(quad.Position.xy)));
    return ShadeRaster(scene, view, ctx, NonTriangleTopology ? decoded.Topology : uint(MeshPrimitiveTopology::Triangle), decoded.PointCoord);
}
