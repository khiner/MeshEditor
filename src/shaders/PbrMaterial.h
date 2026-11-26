#pragma once
// Adapted from KhronosGroup/glTF-Sample-Renderer.
#include "PbrLobes.h"
#include "gpu/MaterialPattern.h"
#include "gpu/PbrFeature.h"

constant bool TiledLights [[function_constant(uint(PbrFeature::TiledLights))]];
constant bool EnableSceneLights [[function_constant(uint(PbrFeature::SceneLights))]];
constant bool EnableAreaLights [[function_constant(uint(PbrFeature::AreaLights))]];
constant bool MultipleShadowSamples [[function_constant(uint(PbrFeature::MultipleShadowSamples))]];
constant bool OpaqueShadows [[function_constant(uint(PbrFeature::OpaqueShadows))]];
constant bool CullShadowBackfaces [[function_constant(uint(PbrFeature::CullShadowBackfaces))]];
constant bool EnableTransmission [[function_constant(uint(PbrFeature::Transmission))]];
constant bool EnableDiffuseTrans [[function_constant(uint(PbrFeature::DiffuseTrans))]];
constant bool EnableClearcoat [[function_constant(uint(PbrFeature::Clearcoat))]];
constant bool EnableSheen [[function_constant(uint(PbrFeature::Sheen))]];
constant bool EnableAnisotropy [[function_constant(uint(PbrFeature::Anisotropy))]];
constant bool EnableIridescence [[function_constant(uint(PbrFeature::Iridescence))]];
constant bool TransmissionPrepass [[function_constant(uint(PbrFeature::TransmissionPrepass))]];
constant bool NonTriangleTopology [[function_constant(uint(PbrFeature::NonTriangleTopology))]];

constant bool EnableOpaqueReceiver [[function_constant(uint(PbrFeature::OpaqueReceiver))]];
constant bool EnableAlphaBlend [[function_constant(uint(PbrFeature::AlphaBlend))]];
constant bool TraceDiffuse [[function_constant(uint(PbrFeature::TraceDiffuse))]];
constant bool PreparePrimary [[function_constant(uint(PbrFeature::PreparePrimary))]];
constant bool TraceReflections [[function_constant(uint(PbrFeature::TraceReflections))]];
constant bool EnablePatterns [[function_constant(uint(PbrFeature::Patterns))]];
constant bool PatternResume [[function_constant(uint(PbrFeature::PatternResume))]];

// TransmissionPrepass omits transmission materials and exposure to prevent attachment self-sampling.

struct NormalInfo {
    float3 ng;
    float3 t;
    float3 b;
    float3 n;
    float3 ntex;
};


inline float clampedDot(float3 x, float3 y) { return clamp(dot(x, y), 0.0f, 1.0f); }

// An attenuation distance of zero or less means no attenuation.
inline float3 applyVolumeAttenuation(float3 radiance, float dist, float3 attenuation_color, float attenuation_distance) {
    if (attenuation_distance <= 0.0f) return radiance;
    return pow(attenuation_color, float3(dist / attenuation_distance)) * radiance;
}

// Rotates the tangent frame by the KHR_texture_transform UV rotation to align sampled normal-map components.
inline float3x3 GetNormalMapTBN(float3 t, float3 b, float3 n, float uv_rotation) {
    const float c = cos(uv_rotation);
    const float s = sin(uv_rotation);
    return float3x3(c * t - s * b, s * t + c * b, n);
}

// The first word of the inputs a pixel's pattern computed, for a triangle surface's pixel in a view whose materials compute inputs, and else zero.
// A material without a pattern has no computed input, so its pixels read none of the words.
inline uint PixelPatternWord(constant SceneViewUBO &view, uint topology, uint2 pixel) {
    return EnablePatterns && topology == uint(MeshPrimitiveTopology::Triangle) ? PatternPixelWord(view, pixel) : 0u;
}

// The first word of the inputs a transparent fragment's pattern computed, found in its pixel's fragment list by its triangle's key, or zero for a fragment the list lacks.
inline uint FragmentPatternWord(device const BindlessSet &b, constant SceneViewUBO &view, uint2 pixel, uint key) {
    device const uint *words = BindlessBuffer(uint, b.Buffer, view.PatternWordSlot);
    for (uint entry = words[view.PatternFragmentHeads + pixel.y * view.PatternWidth + pixel.x]; entry != PatternListEnd; entry = words[view.PatternFragmentEntries + 2u * entry + 1u]) {
        if (words[view.PatternFragmentEntries + 2u * entry] == key) return view.PatternFragmentWords + entry * view.PatternStride;
    }
    return 0u;
}

// A surface point's occlusion, and the factor on the light beneath the coat, which a texture's strength scales.
struct PbrOcclusion {
    float Value, Scale;
};

#ifdef FILTER_AREA_SHADOWS
#include "RayShadowNeighborhood.h"
#endif
// The surface point a material is evaluated at.
// Its copy of the scene serves its own lookups, and lighting code reads the caller's scene.
struct PbrContext {
    Scene S;
    const thread MeshVaryings &In;
    float2 UvDx[4];
    float2 UvDy[4];
    bool ExplicitGradients;
    bool Secondary, RayTracing, Opaque;
    float3 Outgoing, PositionDx, PositionDy, GeometricNormal;
    float Footprint;
    uint SampleOffset;
    const thread RayMediumState *Media;
#ifdef FILTER_AREA_SHADOWS
    ShadowNeighborhood ShadowNeighbors;
#endif
    texture2d_array<float> LightLut;
    // The first word of the inputs the surface point's pattern computed in the pattern word buffer, or zero where none did.
    uint PatternWord;

    void SetSurface(const thread DecodedSurface &decoded, uint pattern_word) {
        ExplicitGradients = true;
        PatternWord = pattern_word;
        GeometricNormal = decoded.GeometricNormal;
        PositionDx = decoded.PositionDx;
        PositionDy = decoded.PositionDy;
        for (uint set = 0; set < 4; ++set) {
            UvDx[set] = decoded.UvDx[set];
            UvDy[set] = decoded.UvDy[set];
        }
    }

    float3 ViewDirection() const { return Secondary ? Outgoing : RayTracing && S.View.ScreenPixelScale < 0.0f ? transpose(S.View.ViewRotation.Unpack()) * float3(0, 0, 1) :
                                                                                                                normalize(float3(S.View.CameraPosition) - In.WorldPosition); }

    float2 GetUv(TextureInfo tex) const {
        return ApplyUvTransform(RayUv(In, tex.TexCoord), float2(tex.UvOffset), float2(tex.UvScale), tex.UvRotation);
    }
    float4 SampleTexture(TextureInfo tex) const {
        const float2 uv = GetUv(tex);
        if (!ExplicitGradients) return S.SampleTex(tex.Slot, uv);
        const uint set = min(tex.TexCoord, 3u);
        const float2 scale = float2(tex.UvScale);
        const float footprint = Secondary ? Footprint : 1.0f;
        return S.SampleTexGrad(tex.Slot, uv, TransformUvGradient(UvDx[set] * footprint, scale, tex.UvRotation), TransformUvGradient(UvDy[set] * footprint, scale, tex.UvRotation));
    }

    // Whether the surface point's pattern computes the input, and its computed value.
    uint MaterialPatternInputs() const { return BindlessBuffer(MaterialPattern, S.B.Buffer, S.View.PatternSlot)[In.MaterialIndex].Inputs; }
    bool Computes(PbrInput input) const { return EnablePatterns && PatternWord != 0u && ((MaterialPatternInputs() >> uint(input)) & 1u) != 0u; }
    device const float *PatternValue(PbrInput input) const { return BindlessBuffer(float, S.B.Buffer, S.View.PatternWordSlot) + PatternWord + PbrInputWord(MaterialPatternInputs(), input); }
    float Computed(PbrInput input) const { return *PatternValue(input); }
    float3 Computed3(PbrInput input) const {
        device const float *value = PatternValue(input);
        return float3(value[0], value[1], value[2]);
    }

    // Whether the surface point samples a texture that scales the input: the material holds one, and no pattern computes the input.
    bool Samples(TextureInfo tex, PbrInput input) const { return tex.Slot != InvalidSlot && !Computes(input); }

    float2 MetallicRoughness(device const PBRMaterial &m) const {
        float2 values(m.MetallicFactor, m.RoughnessFactor);
        if (Samples(m.MetallicRoughnessTexture, PbrInput::Metallic) || Samples(m.MetallicRoughnessTexture, PbrInput::Roughness)) values *= SampleTexture(m.MetallicRoughnessTexture).bg;
        if (Computes(PbrInput::Metallic)) values.x = Computed(PbrInput::Metallic);
        if (Computes(PbrInput::Roughness)) values.y = Computed(PbrInput::Roughness);
        return values;
    }

    float4 SpecularColorStrength(device const PBRMaterial &m) const {
        float4 specular(float3(m.Specular.ColorFactor), m.Specular.Factor);
        if (Samples(m.Specular.ColorTexture, PbrInput::SpecularColor)) specular.rgb *= SampleTexture(m.Specular.ColorTexture).rgb;
        if (Samples(m.Specular.Texture, PbrInput::Specular)) specular.a *= SampleTexture(m.Specular.Texture).a;
        if (Computes(PbrInput::SpecularColor)) specular.rgb = Computed3(PbrInput::SpecularColor);
        if (Computes(PbrInput::Specular)) specular.a = Computed(PbrInput::Specular);
        return specular;
    }

    float Transmission(device const PBRMaterial &m) const {
        float transmission = m.Transmission.Factor;
        if (m.Transmission.Texture.Slot != InvalidSlot) transmission *= SampleTexture(m.Transmission.Texture).r;
        return transmission;
    }
    // The volume's thickness in world space.
    float Thickness(device const PBRMaterial &m) const {
        float thickness = m.Volume.ThicknessFactor * In.WorldScale;
        if (Samples(m.Volume.ThicknessTexture, PbrInput::Thickness)) thickness *= SampleTexture(m.Volume.ThicknessTexture).g;
        if (Computes(PbrInput::Thickness)) thickness = Computed(PbrInput::Thickness) * In.WorldScale;
        return thickness;
    }
    PbrOcclusion Occlusion(device const PBRMaterial &m) const {
        if (Computes(PbrInput::Occlusion)) {
            const float occlusion = Computed(PbrInput::Occlusion);
            return {occlusion, occlusion};
        }
        if (m.OcclusionTexture.Slot == InvalidSlot) return {1.0f, 1.0f};
        const float occlusion = SampleTexture(m.OcclusionTexture).r;
        return {occlusion, 1.0f + m.OcclusionStrength * (occlusion - 1.0f)};
    }

    template<bool Raster> NormalInfo GetNormalInfo(device const PBRMaterial &material) const {
        const float2 uv = RayUv(In, material.NormalTexture.TexCoord);
        float3 ng = normalize(ShadingWorldNormal(In));
        float3 t;
        float3 b;
        if (length(In.WorldTangent.xyz) > 1e-8f) {
            t = normalize(In.WorldTangent.xyz - ng * dot(ng, In.WorldTangent.xyz));
            b = normalize(cross(ng, t) * In.WorldTangent.w);
        } else {
            float3 pos_dx = PositionDx, pos_dy = PositionDy;
            float2 uv_dx = UvDx[min(material.NormalTexture.TexCoord, 3u)];
            float2 uv_dy = UvDy[min(material.NormalTexture.TexCoord, 3u)];
            if constexpr (Raster) {
                pos_dx = dfdx(In.WorldPosition);
                pos_dy = dfdy(In.WorldPosition);
                uv_dx = dfdx(uv);
                uv_dy = dfdy(uv);
            }
            const float uv_scale = max(max(abs(uv_dx.x), abs(uv_dx.y)), max(abs(uv_dy.x), abs(uv_dy.y)));
            if (uv_scale > 0.0f) {
                uv_dx /= uv_scale;
                uv_dy /= uv_scale;
            }
            const float det = uv_dx.x * uv_dy.y - uv_dy.x * uv_dx.y;
            const float3 derivative_t = sign(det) * (uv_dy.y * pos_dx - uv_dx.y * pos_dy);
            const float3 derivative_b = sign(det) * (uv_dx.x * pos_dy - uv_dy.x * pos_dx);
            const float3 projected_t = derivative_t - ng * dot(ng, derivative_t);
            if (det != 0.0f && dot(projected_t, projected_t) > 0.0f) {
                t = normalize(projected_t);
                const float handedness = dot(cross(ng, t), derivative_b) < 0.0f ? -1.0f : 1.0f;
                b = cross(ng, t) * handedness;
            } else {
                const float3 axis = abs(ng.x) < abs(ng.y) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
                t = normalize(cross(axis, ng));
                b = cross(ng, t);
            }
        }

        if (Raster ? !IsFrontFacing(S, ng, In.WorldPosition) : dot(ng, ViewDirection()) < 0.0f) {
            t *= -1.0f;
            b *= -1.0f;
            ng *= -1.0f;
        }

        if (Computes(PbrInput::Normal)) {
            const float3 ntex = normalize(Computed3(PbrInput::Normal));
            return {.ng = ng, .t = t, .b = b, .n = normalize(GetNormalMapTBN(t, b, ng, 0.0f) * ntex), .ntex = ntex};
        }
        if (material.NormalTexture.Slot != InvalidSlot) {
            float3 ntex = SampleTexture(material.NormalTexture).rgb * 2.0f - float3(1.0f);
            ntex *= float3(material.NormalScale, material.NormalScale, 1.0f);
            ntex = normalize(ntex);
            return {.ng = ng, .t = t, .b = b, .n = normalize(GetNormalMapTBN(t, b, ng, material.NormalTexture.UvRotation) * ntex), .ntex = ntex};
        }
        return {.ng = ng, .t = t, .b = b, .n = ng, .ntex = float3(0.0f, 0.0f, 1.0f)};
    }
};

// What the lighting reads from a material at a surface point: the material's parameters and the weight of each lobe.
// A lobe outside the feature mask keeps its zero weight.
struct PbrClosure {
    device const PBRMaterial *Material;
    float4 BaseColor;
    float3 V;
    NormalInfo Normal;
    float3 N;
    float NdotV, Metallic, Roughness;
    float3 SheenColor;
    float SheenRoughness;
    bool HasSheen;
    float SpecularWeight;
    float3 SpecularColor;
    float Transmission, DiffuseTransmission;
    float3 DiffuseTransmissionColor;
    float Thickness;
    float3 TransmissionRay; // From the entry point to the exit point of the volume
    float Clearcoat, ClearcoatRoughness;
    float3 ClearcoatN;
    float ClearcoatNdotV;
    bool HasClearcoat;
    float AnisotropyStrength;
    float2 AnisotropyDirection;
    float3 AnisotropicT, AnisotropicB;
    bool HasAnisotropy;
    float Iridescence, IridescenceThickness;

    // Layering by directional albedo at the view direction, with the coat over the sheen over the metal and dielectric mix.
    float3 DiffuseWeight, DiffuseTransmissionWeight, TransmissionWeight;
    float3 SpecularF0, SpecularF90, SpecularAlbedo; // The base GGX lobe, with its weight and multiple scattering
    float3 SheenWeight;
    float SheenAlpha, SheenAlbedo; // The directional albedo of the sheen lobe before its weight
    float CoatF0, CoatF90, CoatAlbedo; // The coat GGX lobe, with its weight and multiple scattering
    float CoatThroughput; // The part of the light the coat passes to the layers beneath, including emission
};

// Weights every lobe by layering at the view direction, with the coat over the sheen over the metal and dielectric mix.
// `terms` and `coat_terms` are the GGX albedo terms of the base and the coat, and `attenuation` is the transmittance of the volume through its thickness.
inline void LayerLobes(thread PbrClosure &cl, float2 terms, float2 coat_terms, float3 dielectric_f0, float3 iridescent_dielectric, float3 iridescent_metal, float3 attenuation) {
    constexpr float CoatF0 = 0.04f; // An ior 1.5 dielectric
    float base_weight = 1.0f;
    cl.CoatF0 = cl.CoatF90 = cl.CoatAlbedo = 0.0f;
    if (cl.HasClearcoat) {
        const float coat_scale = cl.Clearcoat * GgxCompensation(float3(CoatF0), float3(1), coat_terms).x;
        cl.CoatF0 = coat_scale * CoatF0;
        cl.CoatF90 = coat_scale;
        cl.CoatAlbedo = cl.CoatF0 * coat_terms.x + cl.CoatF90 * coat_terms.y;
        base_weight = 1.0f - cl.CoatAlbedo;
    }
    cl.CoatThroughput = base_weight;
    cl.SheenAlpha = SheenAlpha(cl.SheenRoughness);
    // A smooth sheen seen at grazing angles reflects more light than it receives, so its weight scales its albedo down to one.
    cl.SheenAlbedo = cl.HasSheen ? SheenAlbedo(cl.SheenRoughness, cl.NdotV) : 0.0f;
    cl.SheenWeight = cl.HasSheen ? base_weight * cl.SheenColor / max(cl.SheenAlbedo, 1.0f) : float3(0);
    if (cl.HasSheen) base_weight *= 1.0f - max3(cl.SheenColor.r, cl.SheenColor.g, cl.SheenColor.b) * min(cl.SheenAlbedo, 1.0f);

    const float iridescence = EnableIridescence && cl.Iridescence > 0.0f ? cl.Iridescence : 0.0f;
    const PbrBaseLayer layer = BaseLayer(terms, base_weight, cl.BaseColor.rgb, cl.Metallic, dielectric_f0, cl.SpecularWeight, iridescence, iridescent_dielectric, iridescent_metal);
    cl.SpecularF0 = layer.SpecularF0;
    cl.SpecularF90 = layer.SpecularF90;
    cl.SpecularAlbedo = layer.SpecularF0 * terms.x + layer.SpecularF90 * terms.y;
    // Specular transmission, diffuse transmission and diffuse reflection share the layer beneath the dielectric reflection.
    const float transmitted = layer.Diffuse * cl.Transmission;
    const float diffuse = layer.Diffuse - transmitted;
    cl.TransmissionWeight = transmitted * cl.BaseColor.rgb * attenuation;
    cl.DiffuseTransmissionWeight = diffuse * cl.DiffuseTransmission * cl.DiffuseTransmissionColor * attenuation;
    cl.DiffuseWeight = diffuse * (1.0f - cl.DiffuseTransmission) * cl.BaseColor.rgb;
}

// The parts of an integrator are always inlined, so each integrator compiles as one function.
// Each part takes the bound view as its own argument, which shades faster than reading it through the scene.

// Shades a surface that takes no lighting: a colored point or line, a point or line without a normal, or an unlit material.
// Returns false for a lit surface, with its base color in `base_color`.
// Fragment shading discards culled back faces, fragments outside a point's disk and alpha-masked fragments.
template<bool Fragment>
__attribute__((always_inline)) inline bool ShadeUnlit(const thread Scene &scene, constant SceneViewUBO &view, const thread PbrContext &ctx, uint topology, float2 point_coord, thread float4 &base_color, thread float4 &unlit) {
    const thread MeshVaryings &in = ctx.In;
    // Discarded lanes continue so neighboring texture-gradient calculations remain valid at masked edges.
    if constexpr (Fragment)
        if (topology == uint(MeshPrimitiveTopology::Point) && length(point_coord - float2(0.5f)) > 0.5f) discard_fragment();

    if (topology != uint(MeshPrimitiveTopology::Triangle) && in.Color.a > 0.0f && view.DebugChannel == DebugChannel::None) {
        unlit = float4(in.Color.rgb, 1.0f);
        return true;
    }

    device const PBRMaterial &material = scene.Materials(view.MaterialSlot)[in.MaterialIndex];
    const float3 world_normal = ShadingWorldNormal(in);
    if constexpr (Fragment)
        if (!ctx.Secondary && CullBackfaces(view, material.DoubleSided) && !IsFrontFacing(scene, world_normal, in.WorldPosition)) discard_fragment();

    base_color = float4(material.BaseColorFactor);
    if (material.BaseColorTexture.Slot != InvalidSlot) base_color *= ctx.SampleTexture(material.BaseColorTexture);
    if (ctx.Computes(PbrInput::BaseColor)) base_color.rgb = ctx.Computed3(PbrInput::BaseColor);
    base_color *= in.VertexColor;
    if (material.AlphaMode == MaterialAlphaMode::Opaque) base_color.a = 1.0f;

    const bool no_normal = topology != uint(MeshPrimitiveTopology::Triangle) && dot(world_normal, world_normal) < 1e-12f;
    if (no_normal || (material.Unlit != 0u && view.DebugChannel == DebugChannel::None)) {
        if (material.AlphaMode == MaterialAlphaMode::Mask) {
            if constexpr (Fragment)
                if (!ctx.Secondary && base_color.a < material.AlphaCutoff) discard_fragment();
            base_color.a = 1.0f;
        }
        float3 color = base_color.rgb;
        if (no_normal) {
            float3 emissive = float3(material.EmissiveFactor) * material.EmissiveStrength;
            if (material.EmissiveTexture.Slot != InvalidSlot) emissive *= ctx.SampleTexture(material.EmissiveTexture).rgb;
            color += emissive;
        }
        unlit = TransmissionPrepass || ctx.Secondary ? float4(color, base_color.a) : float4(color * view.Exposure, base_color.a);
        return true;
    }
    return false;
}

// Evaluates a lit surface's material at the surface point.
// Fragment evaluation discards the transmissive fragments of the transmission prepass and takes gradients from the raster.
template<bool Fragment>
__attribute__((always_inline)) inline PbrClosure EvaluatePbrMaterial(const thread Scene &scene, constant SceneViewUBO &view, const thread PbrContext &ctx, float4 base_color, texture2d_array<float> lut) {
    const thread MeshVaryings &in = ctx.In;
    device const PBRMaterial &material = scene.Materials(view.MaterialSlot)[in.MaterialIndex];
    PbrClosure cl;
    cl.Material = &material;
    cl.BaseColor = base_color;

    const float3 v = ctx.ViewDirection();
    const NormalInfo normal_info = ctx.GetNormalInfo<Fragment>(material);
    const float3 n = normal_info.n;
    cl.V = v;
    cl.Normal = normal_info;
    cl.N = n;
    cl.NdotV = clampedDot(n, v);

    const float2 mr = clamp(ctx.MetallicRoughness(material), 0.0f, 1.0f);
    cl.Metallic = mr.x;
    cl.Roughness = mr.y;

    cl.SheenColor = float3(0.0f);
    cl.SheenRoughness = 0.0f;
    if (EnableSheen) {
        cl.SheenColor = float3(material.Sheen.ColorFactor);
        if (ctx.Samples(material.Sheen.ColorTexture, PbrInput::SheenColor)) cl.SheenColor *= ctx.SampleTexture(material.Sheen.ColorTexture).rgb;
        cl.SheenRoughness = clamp(material.Sheen.RoughnessFactor, 0.0f, 1.0f);
        if (ctx.Samples(material.Sheen.RoughnessTexture, PbrInput::SheenRoughness)) cl.SheenRoughness *= ctx.SampleTexture(material.Sheen.RoughnessTexture).a;
        if (ctx.Computes(PbrInput::SheenColor)) cl.SheenColor = ctx.Computed3(PbrInput::SheenColor);
        if (ctx.Computes(PbrInput::SheenRoughness)) cl.SheenRoughness = clamp(ctx.Computed(PbrInput::SheenRoughness), 0.0f, 1.0f);
    }
    cl.HasSheen = EnableSheen && any(cl.SheenColor > float3(0.0f));

    const float4 specular = ctx.SpecularColorStrength(material);
    const float f0_ior_t = (material.Ior - 1.0f) / (material.Ior + 1.0f);
    const float f0_ior = f0_ior_t * f0_ior_t;
    const float3 f0_dielectric = min(float3(f0_ior) * specular.rgb, float3(1.0f));
    cl.SpecularWeight = specular.a;
    cl.SpecularColor = specular.rgb;

    cl.Transmission = EnableTransmission && !ctx.Opaque ? ctx.Transmission(material) : 0.0f;
    // The transmission prepass drops transmissive fragments so it captures the opaque geometry behind them.
    if constexpr (Fragment)
        if (TransmissionPrepass && cl.Transmission > 0.0f) discard_fragment();
    cl.DiffuseTransmission = 0.0f;
    cl.DiffuseTransmissionColor = float3(0.0f);
    if (EnableDiffuseTrans) {
        cl.DiffuseTransmission = material.DiffuseTransmission.Factor;
        if (material.DiffuseTransmission.Texture.Slot != InvalidSlot) cl.DiffuseTransmission *= ctx.SampleTexture(material.DiffuseTransmission.Texture).a;
        cl.DiffuseTransmissionColor = float3(material.DiffuseTransmission.ColorFactor);
        if (material.DiffuseTransmission.ColorTexture.Slot != InvalidSlot) cl.DiffuseTransmissionColor *= ctx.SampleTexture(material.DiffuseTransmission.ColorTexture).rgb;
    }
    // KHR_materials_volume thickness converts from model to world space for Beer's law.
    cl.Thickness = ctx.Thickness(material);
    cl.TransmissionRay = cl.Transmission > 0.0f ? normalize(refract(-v, n, 1.0f / material.Ior)) * cl.Thickness : float3(0);

    cl.Clearcoat = 0.0f;
    cl.ClearcoatRoughness = 0.0f;
    cl.ClearcoatN = normal_info.ng;
    cl.ClearcoatNdotV = 0.0f;
    if (EnableClearcoat) {
        cl.Clearcoat = material.Clearcoat.Factor;
        if (ctx.Samples(material.Clearcoat.Texture, PbrInput::Clearcoat)) cl.Clearcoat *= ctx.SampleTexture(material.Clearcoat.Texture).r;
        cl.ClearcoatRoughness = clamp(material.Clearcoat.RoughnessFactor, 0.0f, 1.0f);
        if (ctx.Samples(material.Clearcoat.RoughnessTexture, PbrInput::ClearcoatRoughness)) cl.ClearcoatRoughness *= ctx.SampleTexture(material.Clearcoat.RoughnessTexture).g;
        if (ctx.Computes(PbrInput::Clearcoat)) cl.Clearcoat = ctx.Computed(PbrInput::Clearcoat);
        if (ctx.Computes(PbrInput::ClearcoatRoughness)) cl.ClearcoatRoughness = ctx.Computed(PbrInput::ClearcoatRoughness);
        cl.ClearcoatRoughness = clamp(cl.ClearcoatRoughness, 0.0f, 1.0f);
        if (ctx.Computes(PbrInput::ClearcoatNormal)) {
            cl.ClearcoatN = normalize(GetNormalMapTBN(normal_info.t, normal_info.b, normal_info.ng, 0.0f) * normalize(ctx.Computed3(PbrInput::ClearcoatNormal)));
        } else if (material.Clearcoat.NormalTexture.Slot != InvalidSlot) {
            float3 cc_ntex = ctx.SampleTexture(material.Clearcoat.NormalTexture).rgb * 2.0f - float3(1.0f);
            cc_ntex *= float3(material.Clearcoat.NormalScale, material.Clearcoat.NormalScale, 1.0f);
            cl.ClearcoatN = normalize(GetNormalMapTBN(normal_info.t, normal_info.b, normal_info.ng, material.Clearcoat.NormalTexture.UvRotation) * normalize(cc_ntex));
        }
        cl.ClearcoatNdotV = clampedDot(cl.ClearcoatN, v);
    }
    cl.HasClearcoat = EnableClearcoat && cl.Clearcoat > 0.0f;

    cl.AnisotropyStrength = 0.0f;
    cl.AnisotropyDirection = float2(1.0f, 0.0f);
    cl.AnisotropicT = float3(0.0f);
    cl.AnisotropicB = float3(0.0f);
    if (EnableAnisotropy) {
        cl.AnisotropyStrength = material.Anisotropy.Strength;
        cl.AnisotropyDirection = float2(cos(material.Anisotropy.Rotation), sin(material.Anisotropy.Rotation));
        if (ctx.Samples(material.Anisotropy.Texture, PbrInput::AnisotropyRotation) || ctx.Samples(material.Anisotropy.Texture, PbrInput::AnisotropyStrength)) {
            const float3 anisotropySample = ctx.SampleTexture(material.Anisotropy.Texture).rgb;
            const float2 texDir = anisotropySample.xy * 2.0f - float2(1.0f);
            const float2 dir = cl.AnisotropyDirection;
            cl.AnisotropyDirection = normalize(float2x2(float2(dir.x, dir.y), float2(-dir.y, dir.x)) * texDir);
            cl.AnisotropyStrength *= anisotropySample.z;
        }
        if (ctx.Computes(PbrInput::AnisotropyRotation)) {
            const float rotation = ctx.Computed(PbrInput::AnisotropyRotation);
            cl.AnisotropyDirection = float2(cos(rotation), sin(rotation));
        }
        if (ctx.Computes(PbrInput::AnisotropyStrength)) cl.AnisotropyStrength = ctx.Computed(PbrInput::AnisotropyStrength);
        cl.AnisotropyStrength = clamp(cl.AnisotropyStrength, 0.0f, 1.0f);
        // AnisotropicB uses the geometric normal to match the reference implementation.
        cl.AnisotropicT = normalize(float3x3(normal_info.t, normal_info.b, normal_info.n) * float3(cl.AnisotropyDirection, 0.0f));
        cl.AnisotropicB = cross(normal_info.ng, cl.AnisotropicT);
    }
    cl.HasAnisotropy = EnableAnisotropy && cl.AnisotropyStrength > 0.0f;

    cl.Iridescence = 0.0f;
    cl.IridescenceThickness = 0.0f;
    float3 iridescence_fresnel_dielectric = float3(0.0f);
    float3 iridescence_fresnel_metallic = float3(0.0f);
    if (EnableIridescence) {
        cl.Iridescence = material.Iridescence.Factor;
        if (ctx.Samples(material.Iridescence.Texture, PbrInput::Iridescence)) cl.Iridescence *= ctx.SampleTexture(material.Iridescence.Texture).r;
        if (ctx.Computes(PbrInput::Iridescence)) cl.Iridescence = ctx.Computed(PbrInput::Iridescence);
        cl.Iridescence = clamp(cl.Iridescence, 0.0f, 1.0f);
        cl.IridescenceThickness = material.Iridescence.ThicknessMaximum;
        if (ctx.Computes(PbrInput::IridescenceThickness)) {
            cl.IridescenceThickness = ctx.Computed(PbrInput::IridescenceThickness);
        } else if (material.Iridescence.ThicknessTexture.Slot != InvalidSlot) {
            const float t = ctx.SampleTexture(material.Iridescence.ThicknessTexture).g;
            cl.IridescenceThickness = mix(material.Iridescence.ThicknessMinimum, material.Iridescence.ThicknessMaximum, t);
        }
        // Iridescence Fresnel evaluates once at NdotV to match the reference implementation.
        iridescence_fresnel_dielectric = evalIridescence(1.0f, material.Iridescence.Ior, cl.NdotV, cl.IridescenceThickness, f0_dielectric);
        iridescence_fresnel_metallic = evalIridescence(1.0f, material.Iridescence.Ior, cl.NdotV, cl.IridescenceThickness, base_color.rgb);
        if (cl.IridescenceThickness == 0.0f) cl.Iridescence = 0.0f;
    }
    const float2 coat_terms = cl.HasClearcoat ? GgxAlbedoTerms(lut, cl.ClearcoatRoughness, cl.ClearcoatNdotV) : float2(0);
    const float3 attenuation = cl.Transmission > 0.0f || cl.DiffuseTransmission > 0.0f ? applyVolumeAttenuation(float3(1), cl.Thickness, float3(material.Volume.AttenuationColor), material.Volume.AttenuationDistance) : float3(1);
    LayerLobes(cl, GgxAlbedoTerms(lut, cl.Roughness, cl.NdotV), coat_terms, f0_dielectric * cl.SpecularWeight, iridescence_fresnel_dielectric, iridescence_fresnel_metallic, attenuation);
    return cl;
}
