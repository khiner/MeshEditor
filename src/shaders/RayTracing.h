#pragma once
#include "gpu/RayEnvironment.h"
#include "RayHistory.h"
#include "VisibilityDecode.metal"
#include "gpu/RayTracingRecord.h"
#include "gpu/Hash.h"
#include <metal_raytracing>
using namespace raytracing;

struct RayGeometry {
    instance_acceleration_structure Acceleration;
    device const RayTracingInstance *Instances;
    device const uint *Triangles;
};
struct RayScene {
    instance_acceleration_structure Acceleration;
    device const RayTracingInstance *Instances;
    device const uint *Triangles;
    device const float *Environment;
    texture2d_array<float> Lut;
    texture2d<float, access::read> Feedback, HistoryGeometry, HistoryNormal;
    texture2d<uint, access::read> HistoryMaterial;
    device const RayMotionTransform *Motion;
    float4x4 PreviousViewProj;
    bool HistoryValid = false, ApplyMotion = false;
    float4x4 PreviousClipToWorld;
    texture2d<float, access::read> OpaqueRadiance, OpaqueGeometry, OpaqueNormal;
    texture2d<uint, access::read> OpaqueIdentity;
    float4x4 ClipToWorld;
    bool OpaqueValid = false;
    RayGeometry Geometry() const { return {Acceleration, Instances, Triangles}; }
};

struct RayTriangle {
    DrawData Draw;
    uint Triangle, ObjectId;
    uint Face(const thread Scene &scene) const { return scene.TriangleFace(Draw, Triangle); }
    uint Material(const thread Scene &scene) const { return MeshletFaceMaterialIndex(scene, Draw, Face(scene)); }
};
inline RayTriangle ResolveRayTriangle(const thread Scene &scene, const thread RayGeometry &rt, uint instance_id, uint primitive_id) {
    const RayTracingInstance mapping = rt.Instances[instance_id];
    const InstanceRecord instance = scene.InstanceRecords(scene.View.InstanceRecordSlot)[mapping.InstanceSlot];
    const MeshRecord mesh = scene.MeshRecords(scene.View.MeshRecordSlot)[instance.Mesh];
    return {ComposeDraw(mesh, instance, mapping.InstanceSlot, uint(MeshPrimitiveTopology::Triangle)),
            rt.Triangles[mapping.TriangleOffset + primitive_id], instance.ObjectId};
}
// Match raster visibility without rebuilding the acceleration structure for hide/reveal.
inline bool RayElementHidden(const thread Scene &scene, const thread RayGeometry &rt, uint instance_id, uint primitive_id) {
    if (scene.View.InteractionMode != InteractionMode::Edit) return false;
    const auto hit = ResolveRayTriangle(scene, rt, instance_id, primitive_id);
    return EditElementHidden(scene, hit.Draw, Element::Face, hit.Face(scene));
}
inline float3 RayGeometricNormal(const thread Scene &scene, DrawData draw, float3 edge1, float3 edge2) {
    const float3 normal = normalize(cross(edge1, edge2));
    const float3 scale = float3(MeshletWorld(scene, draw).S);
    float determinant_sign = scale.x * scale.y * scale.z;
    if (scene.View.IsTransforming != 0u && scene.View.InteractionMode != InteractionMode::Edit && (scene.InstanceState(draw) & STATE_SELECTED) != 0u) {
        const float3 pending_scale = float3(scene.View.PendingScale);
        determinant_sign *= pending_scale.x * pending_scale.y * pending_scale.z;
    }
    return determinant_sign < 0.0f ? -normal : normal;
}
// Inlining attribute decoding into traversal loops produced intermittent glass speckles.
__attribute__((noinline)) DecodedSurface DecodeRaySurface(const thread Scene &scene, const thread RayGeometry &rt, uint instance_id, uint primitive_id, float2 bary) {
    const auto hit = ResolveRayTriangle(scene, rt, instance_id, primitive_id);
    const thread DrawData &draw = hit.Draw;
    MeshVaryings corners[3];
    for (uint i = 0; i < 3; ++i) {
        const uint corner = TriangleCornerHandle(scene.B, draw.TriangleSlot, draw.Connectivity, hit.Triangle, i);
        corners[i] = TransformVertex(scene, draw, corner, corner, scene.CornerVertex(draw, corner));
    }
    const float3 weights = float3(1.0f - bary.x - bary.y, bary);
    DecodedSurface result{};
    result.Valid = true;
    result.ObjectId = hit.ObjectId;
    result.InstanceSlot = draw.FirstInstance;
    result.V = corners[0];
    result.V.WorldPosition = PerspectiveValue(weights, corners[0].WorldPosition, corners[1].WorldPosition, corners[2].WorldPosition);
    result.V.WorldNormal = PerspectiveValue(weights, corners[0].WorldNormal, corners[1].WorldNormal, corners[2].WorldNormal);
    result.V.WorldTangent = PerspectiveValue(weights, corners[0].WorldTangent, corners[1].WorldTangent, corners[2].WorldTangent);
    result.V.VertexColor = draw.CornerColor.Slot != InvalidSlot ?
        PerspectiveValue(weights, corners[0].VertexColor, corners[1].VertexColor, corners[2].VertexColor) : float4(1.0f);
    result.V.FaceOverlayFlags = 0u;
    result.PositionDx = corners[1].WorldPosition - corners[0].WorldPosition;
    result.PositionDy = corners[2].WorldPosition - corners[0].WorldPosition;
    result.GeometricNormal = RayGeometricNormal(scene, draw, result.PositionDx, result.PositionDy);
    const PerspectiveWeights uv_weights{weights, float3(-1, 1, 0), float3(-1, 0, 1)};
    DecodeUv(result.V.TexCoord0, result.UvDx[0], result.UvDy[0], uv_weights, corners[0].TexCoord0, corners[1].TexCoord0, corners[2].TexCoord0);
    DecodeUv(result.V.TexCoord1, result.UvDx[1], result.UvDy[1], uv_weights, corners[0].TexCoord1, corners[1].TexCoord1, corners[2].TexCoord1);
    DecodeUv(result.V.TexCoord2, result.UvDx[2], result.UvDy[2], uv_weights, corners[0].TexCoord2, corners[1].TexCoord2, corners[2].TexCoord2);
    DecodeUv(result.V.TexCoord3, result.UvDx[3], result.UvDy[3], uv_weights, corners[0].TexCoord3, corners[1].TexCoord3, corners[2].TexCoord3);
    return result;
}
inline float2 RayUv(const thread MeshVaryings &v, uint set) {
    if (set == 1u) return v.TexCoord1;
    if (set == 2u) return v.TexCoord2;
    if (set == 3u) return v.TexCoord3;
    return v.TexCoord0;
}
inline float4 RaySurfaceTexel(const thread Scene &scene, const thread DecodedSurface &surface, TextureInfo tex) {
    const float2 uv = ApplyUvTransform(RayUv(surface.V, tex.TexCoord), float2(tex.UvOffset), float2(tex.UvScale), tex.UvRotation);
    return scene.SampleTexLod(tex.Slot, uv, 0.0f);
}
inline float RaySurfaceAlpha(const thread Scene &scene, const thread DecodedSurface &surface, device const PBRMaterial &material) {
    float alpha = material.BaseColorFactor.a * surface.V.VertexColor.a;
    if (material.BaseColorTexture.Slot != InvalidSlot) alpha *= RaySurfaceTexel(scene, surface, material.BaseColorTexture).a;
    return alpha;
}
// Offset by representable coordinates rather than a fixed scene-unit bias (Ray Tracing Gems, chapter 6).
inline float3 RayOrigin(float3 p, float3 n, float3 direction) {
    n = dot(n, direction) < 0.0f ? -n : n;
    const int3 offset = int3(256.0f * n);
    const float3 moved = as_type<float3>(as_type<int3>(p) + select(offset, -offset, p < 0.0f));
    return select(moved, p + n * (1.0f / 65536.0f), abs(p) < (1.0f / 32.0f));
}
inline bool RayOpaqueFeedback(const thread Scene &scene, const thread RayScene &rt, float3 position, float3 direction,
    uint instance, uint primitive, uint material, thread float3 &light, thread float3 &normal) {
    const RayTracingInstance mapping = rt.Instances[instance];
    const float3 outgoing = scene.View.ScreenPixelScale < 0.0f ? transpose(scene.View.ViewRotation.Unpack()) * float3(0, 0, 1) : float3(scene.View.CameraPosition) - position;
    return RayOpaqueFeedback(rt.OpaqueRadiance, rt.OpaqueGeometry, rt.OpaqueNormal, rt.OpaqueIdentity,
        scene.View.ViewProj.Unpack(), rt.ClipToWorld, outgoing, position, direction,
        mapping.InstanceSlot, rt.Triangles[mapping.TriangleOffset + primitive], material, light, normal);
}
// A traced surface, and the instance, primitive and barycentric coordinates DecodeRaySurface decodes it from, which a cached surface has none of.
struct RayHit {
    DecodedSurface Surface;
    uint4 Identity;
};
inline RayHit TraceRayHit(const thread Scene &scene, const thread RayScene &rt, float3 origin, float3 direction, bool skip_transmission = false, thread float3 *cached_light = nullptr, thread bool *cache_hit = nullptr) {
    const RayGeometry geometry = rt.Geometry();
    ray r{origin, direction, 0.0f, INFINITY};
    intersector<triangle_data, instancing> query;
    query.assume_geometry_type(geometry_type::triangle);
    if (cached_light) query.set_triangle_front_facing_winding(winding::clockwise);
    while (r.min_distance < r.max_distance) {
        const auto hit = query.intersect(r, rt.Acceleration);
        if (hit.type == intersection_type::none) return {};
        if (RayElementHidden(scene, geometry, hit.instance_id, hit.primitive_id)) {
            r.min_distance = nextafter(hit.distance, INFINITY);
            continue;
        }
        if (skip_transmission) {
            device const PBRMaterial &material = scene.Materials(scene.View.MaterialSlot)[ResolveRayTriangle(scene, geometry, hit.instance_id, hit.primitive_id).Material(scene)];
            if (!material.Unlit && material.Transmission.Factor > 0.0f) {
                r.min_distance = nextafter(hit.distance, INFINITY);
                continue;
            }
        }
        if (cached_light && rt.OpaqueValid) {
            const uint material_index = ResolveRayTriangle(scene, geometry, hit.instance_id, hit.primitive_id).Material(scene);
            device const PBRMaterial &m = scene.Materials(scene.View.MaterialSlot)[material_index];
            if (m.AlphaMode == MaterialAlphaMode::Opaque && (m.DoubleSided || hit.triangle_front_facing)) {
                DecodedSurface cached{};
                cached.V.WorldPosition = origin + hit.distance * direction;
                if (RayOpaqueFeedback(scene, rt, cached.V.WorldPosition, direction, hit.instance_id, hit.primitive_id, material_index, *cached_light, cached.GeometricNormal)) {
                    if ((dot(cached.GeometricNormal, direction) < 0.0f) != hit.triangle_front_facing) cached.GeometricNormal *= -1.0f;
                    cached.Valid = *cache_hit = true;
                    cached.V.MaterialIndex = material_index;
                    cached.InstanceSlot = rt.Instances[hit.instance_id].InstanceSlot;
                    return {cached, uint4(0)};
                }
            }
        }
        DecodedSurface surface = DecodeRaySurface(scene, geometry, hit.instance_id, hit.primitive_id, hit.triangle_barycentric_coord);
        device const PBRMaterial &material = scene.Materials(scene.View.MaterialSlot)[surface.V.MaterialIndex];
        bool accepted = material.DoubleSided != 0u || (!material.Unlit && material.Transmission.Factor > 0.0f && material.Volume.ThicknessFactor > 0.0f) || dot(surface.GeometricNormal, direction) < 0.0f;
        if (accepted && material.AlphaMode == MaterialAlphaMode::Mask)
            accepted = RaySurfaceAlpha(scene, surface, material) >= material.AlphaCutoff;
        if (accepted) return {surface, uint4(hit.instance_id, hit.primitive_id, as_type<uint2>(hit.triangle_barycentric_coord))};
        r.min_distance = nextafter(hit.distance, INFINITY);
    }
    return {};
}
__attribute__((noinline)) float RayOcclusionAlpha(const thread Scene &scene, const thread RayGeometry &rt, uint instance, uint primitive, float2 barycentric, device const PBRMaterial &material) {
    const DecodedSurface surface = DecodeRaySurface(scene, rt, instance, primitive, barycentric);
    return RaySurfaceAlpha(scene, surface, material);
}
inline bool RayOpaqueOccluded(instance_acceleration_structure acceleration, float3 origin, float3 direction, float maximum, bool cull_backfaces) {
    if (maximum <= 0.0f) return false;
    intersector<instancing> query;
    query.assume_geometry_type(geometry_type::triangle);
    query.force_opacity(forced_opacity::opaque);
    query.accept_any_intersection(true);
    query.set_triangle_front_facing_winding(winding::clockwise);
    query.set_triangle_cull_mode(cull_backfaces ? triangle_cull_mode::back : triangle_cull_mode::none);
    return query.intersect(ray{origin, direction, 0.0f, maximum}, acceleration, 0xff).type != intersection_type::none;
}
inline bool RayOccluded(const thread Scene &scene, const thread RayGeometry &rt, float3 origin, float3 direction, float maximum, bool opaque, bool cull_backfaces) {
    if (opaque) return RayOpaqueOccluded(rt.Acceleration, origin, direction, maximum, cull_backfaces);
    if (maximum <= 0.0f) return false;
    intersector<triangle_data, instancing> query;
    query.assume_geometry_type(geometry_type::triangle);
    query.set_triangle_front_facing_winding(winding::clockwise);
    ray r{origin, direction, 0.0f, maximum};
    while (r.min_distance < r.max_distance) {
        const auto hit = query.intersect(r, rt.Acceleration, 0xff);
        if (hit.type == intersection_type::none) return false;
        if (RayElementHidden(scene, rt, hit.instance_id, hit.primitive_id)) {
            r.min_distance = nextafter(hit.distance, INFINITY);
            continue;
        }
        const uint material_index = ResolveRayTriangle(scene, rt, hit.instance_id, hit.primitive_id).Material(scene);
        device const PBRMaterial &material = scene.Materials(scene.View.MaterialSlot)[material_index];
        const bool front = hit.triangle_front_facing;
        if (material.DoubleSided || front) {
            if (material.AlphaMode != MaterialAlphaMode::Mask) return true;
            const float alpha = RayOcclusionAlpha(scene, rt, hit.instance_id, hit.primitive_id, hit.triangle_barycentric_coord, material);
            if (alpha >= material.AlphaCutoff) return true;
        }
        r.min_distance = nextafter(hit.distance, INFINITY);
    }
    return false;
}
inline uint RayHash(uint x) { return HashMix(x); }
inline float RayRandom(thread uint &state) {
    state = RayHash(state + 0x9e3779b9u);
    return float(state >> 8) * (1.0f / 16777216.0f);
}
inline float3x3 RayFrame(float3 n) {
    const float3 t = normalize(cross(abs(n.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0), n));
    return float3x3(t, cross(n, t), n);
}

// eta is incident IOR / transmitted IOR; the normal faces the incident medium.
inline float RayDielectricFresnel(float cosine, float eta) {
    if (eta == 1.0f) return 0.0f;
    cosine = clamp(cosine, 0.0f, 1.0f);
    const float sin2 = eta * eta * max(0.0f, 1.0f - cosine * cosine);
    if (sin2 >= 1.0f) return 1.0f;
    const float transmitted = sqrt(1.0f - sin2);
    const float rs = (eta * cosine - transmitted) / (eta * cosine + transmitted);
    const float rp = (cosine - eta * transmitted) / (cosine + eta * transmitted);
    return 0.5f * (rs * rs + rp * rp);
}

inline device const RayMediumState &RayCameraState(const thread RayScene &rt) {
    return *reinterpret_cast<device const RayMediumState *>(rt.Environment + RayEnvironmentCount);
}
inline float3 RayMediumAttenuation(const thread Scene &scene, const thread RayMediumState &media, float length) {
    if (media.Count == 0u || length <= 0.0f) return float3(1);
    device const PBRMaterial &material = scene.Materials(scene.View.MaterialSlot)[media.Materials[media.Count - 1u]];
    if (material.Volume.AttenuationDistance <= 0.0f) return float3(1);
    const float3 color = clamp(float3(material.Volume.AttenuationColor), 0.0f, 1.0f);
    return isinf(length) ? select(float3(0), float3(1), color >= 1.0f) : pow(color, float3(length / material.Volume.AttenuationDistance));
}
inline uint RayMediumFind(const thread RayMediumState &media, uint object) {
    uint found = media.Count;
    for (uint i = 0u; i < media.Count; ++i)
        if (media.Objects[i] == object) found = i;
    return found;
}
inline void RayMediumRemove(thread RayMediumState &media, uint index) {
    if (index >= media.Count) return;
    for (uint i = index + 1u; i < media.Count; ++i) {
        media.Materials[i - 1u] = media.Materials[i];
        media.Objects[i - 1u] = media.Objects[i];
    }
    --media.Count;
}

// Straight connections approximate glass shadows; coverage alpha remains untinted visibility.
template<bool Glass = true>
inline float3 RayTransmittance(const thread Scene &scene, const thread RayGeometry &rt, float3 origin, float3 direction, float maximum, bool transmit_glass, RayMediumState media = {}) {
    if (maximum <= 0.0f) return float3(1);
    float3 weight(1);
    float previous = 0.0f;
    ray r{origin, direction, 0.0f, maximum};
    intersector<triangle_data, instancing> query;
    query.assume_geometry_type(geometry_type::triangle);
    query.set_triangle_front_facing_winding(winding::clockwise);
    uint layer = 0u;
    while (layer < 64u) {
        const auto hit = query.intersect(r, rt.Acceleration, 0xff);
        if (hit.type == intersection_type::none) {
            if constexpr (Glass) return weight * RayMediumAttenuation(scene, media, maximum - previous);
            else return weight;
        }
        if (RayElementHidden(scene, rt, hit.instance_id, hit.primitive_id)) {
            r.min_distance = nextafter(hit.distance, INFINITY);
            continue;
        }
        ++layer;
        const auto triangle = ResolveRayTriangle(scene, rt, hit.instance_id, hit.primitive_id);
        const thread DrawData &draw = triangle.Draw;
        const uint material_index = triangle.Material(scene);
        device const PBRMaterial &material = scene.Materials(scene.View.MaterialSlot)[material_index];
        const bool glass = Glass && !material.Unlit && material.Transmission.Factor > 0.0f;
        const bool solid = glass && material.Volume.ThicknessFactor > 0.0f;
        const bool front = hit.triangle_front_facing;
        r.min_distance = nextafter(hit.distance, INFINITY);
        if (!solid && !material.DoubleSided && !front) continue;
        const bool blend = material.AlphaMode == MaterialAlphaMode::Blend;
        if ((!glass || !transmit_glass) && !blend && material.AlphaMode != MaterialAlphaMode::Mask) return float3(0);
        // An untextured shadow boundary needs coverage, tint and the geometric normal.
        // Skip shading normals, tangents and UV derivatives on this path.
        DecodedSurface surface{};
        if (draw.CornerColor.Slot == InvalidSlot && material.BaseColorTexture.Slot == InvalidSlot &&
            (!glass || (material.MetallicRoughnessTexture.Slot == InvalidSlot && material.Transmission.Texture.Slot == InvalidSlot))) {
            surface.V.MaterialIndex = material_index;
            surface.V.VertexColor = float4(1.0f);
            surface.ObjectId = triangle.ObjectId;
            if (glass && transmit_glass) {
                const Transform world = MeshletWorld(scene, draw);
                float3 positions[3];
                for (uint i = 0; i < 3; ++i) {
                    const uint corner = TriangleCornerHandle(scene.B, draw.TriangleSlot, draw.Connectivity, triangle.Triangle, i);
                    const float3 local = scene.GetLocalPosition(draw, scene.CornerVertex(draw, corner));
                    positions[i] = apply_object_pending_transform(scene, draw, trs_transform_point(world, local));
                }
                surface.GeometricNormal = RayGeometricNormal(scene, draw, positions[1] - positions[0], positions[2] - positions[0]);
            }
        } else surface = DecodeRaySurface(scene, rt, hit.instance_id, hit.primitive_id, hit.triangle_barycentric_coord);
        const float alpha = material.AlphaMode == MaterialAlphaMode::Opaque ? 1.0f : clamp(RaySurfaceAlpha(scene, surface, material), 0.0f, 1.0f);
        if ((material.AlphaMode == MaterialAlphaMode::Mask && alpha < material.AlphaCutoff) || (blend && alpha == 0.0f)) continue;
        const float segment = hit.distance - previous;
        if constexpr (Glass) weight *= RayMediumAttenuation(scene, media, segment);
        previous = hit.distance;
        if (!glass || !transmit_glass) {
            weight *= blend ? 1.0f - alpha : 0.0f;
        } else {
            float metallic = material.MetallicFactor;
            float transmission = material.Transmission.Factor;
            float3 tint = material.BaseColorFactor.rgb * surface.V.VertexColor.rgb;
            if (material.MetallicRoughnessTexture.Slot != InvalidSlot) metallic *= RaySurfaceTexel(scene, surface, material.MetallicRoughnessTexture).b;
            transmission *= 1.0f - saturate(metallic);
            if (material.Transmission.Texture.Slot != InvalidSlot) transmission *= RaySurfaceTexel(scene, surface, material.Transmission.Texture).r;
            if (material.BaseColorTexture.Slot != InvalidSlot) tint *= RaySurfaceTexel(scene, surface, material.BaseColorTexture).rgb;
            // Both sides use the exterior incidence angle: the straight-ray approximation does not model internal focusing or TIR.
            const float f = RayDielectricFresnel(abs(dot(surface.GeometricNormal, direction)), 1.0f / material.Ior);
            const float reflection = solid ? f : 2.0f * f / (1.0f + f);
            const float3 passed = saturate(transmission) * tint * (1.0f - reflection);
            weight *= blend ? mix(float3(1), passed, alpha) : passed;
            if (solid) {
                const bool entering = dot(surface.GeometricNormal, direction) < 0.0f;
                if (entering && media.Count < RayMediumCapacity) {
                    media.Materials[media.Count] = material_index;
                    media.Objects[media.Count++] = surface.ObjectId;
                } else if (!entering) {
                    const uint found = RayMediumFind(media, surface.ObjectId);
                    if (found == media.Count && material.Volume.AttenuationDistance > 0.0f)
                        weight *= pow(float3(material.Volume.AttenuationColor), float3(segment / material.Volume.AttenuationDistance));
                    RayMediumRemove(media, found);
                }
            }
        }
        if (max(weight.x, max(weight.y, weight.z)) <= 0.0f) return float3(0);
    }
    return float3(0);
}
