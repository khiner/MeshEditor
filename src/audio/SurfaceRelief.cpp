#include "SurfaceRelief.h"

#include "ContactSurface.h"
#include "gltf/SourceTexture.h"
#include "mesh/Mesh.h"

#include <entt/entity/registry.hpp>
#include <glm/common.hpp>

#include <cmath>
#include <numbers>

namespace {
// Length over which the integrated relief leaks back to zero, mesh-local, so a node's scale does not move it.
// A sampled normal map is only approximately a gradient field, so integrating it drifts, and leaking holds that error out while passing every feature the contact filter resolves.
constexpr float ReliefLeakLength{1e-2f};

// Bilinear tangent-space normal at a texel position, wrapping at the edges.
vec3 SampleNormal(const DecodedImage &image, float x, float y) {
    const auto wrap = [](int v, int n) { return ((v % n) + n) % n; };
    const int w = int(image.Width), h = int(image.Height);
    const int x0 = wrap(int(std::floor(x)), w), y0 = wrap(int(std::floor(y)), h);
    const int x1 = wrap(x0 + 1, w), y1 = wrap(y0 + 1, h);
    const float fx = x - std::floor(x), fy = y - std::floor(y);
    const auto *pixels = reinterpret_cast<const uint8_t *>(image.Pixels.data());
    const auto texel = [&](int px, int py) {
        const auto *p = pixels + (size_t(py) * size_t(w) + size_t(px)) * 4;
        return vec3{float(p[0]), float(p[1]), float(p[2])} / 127.5f - 1.f;
    };
    const vec3 top = glm::mix(texel(x0, y0), texel(x1, y0), fx);
    const vec3 bottom = glm::mix(texel(x0, y1), texel(x1, y1), fx);
    return glm::mix(top, bottom, fy);
}
} // namespace

void UpdateSurfaceRelief(entt::registry &r, entt::entity mesh_entity, bool geometry_changed) {
    // A surface names its own map only to override the one the mesh's material already carries.
    // Both resolve to a source image, so the two paths agree on what a track is derived from.
    const auto *surface = r.try_get<const ContactSurface>(mesh_entity);
    const auto normal_map = [&]() -> std::optional<gltf::NormalMapRef> {
        if (!surface || !surface->NormalTexture) return gltf::MeshMaterialNormalMap(r, mesh_entity);
        const auto &nt = *surface->NormalTexture;
        const auto image = gltf::TextureImageIndex(r, nt.Texture);
        if (!image) return {};
        return gltf::NormalMapRef{.Image = *image, .TexCoord = nt.TexCoord, .Scale = nt.Scale};
    }();
    if (!normal_map) {
        r.remove<SurfaceRelief>(mesh_entity);
        return;
    }
    // Measuring the parameterization walks every triangle, so a surface edit that left the map alone stops here.
    const auto source_key = HashParams(0xff51afd7ed558ccdull, normal_map->Image, normal_map->TexCoord, normal_map->Scale);
    const auto *existing = r.try_get<const SurfaceRelief>(mesh_entity);
    if (!geometry_changed && existing && existing->SourceKey == source_key) return;

    // Lengths stay mesh-local, so one track serves every node instancing it, each sizing it by its own world scale.
    const float length_per_uv = LocalLengthPerUv(r, mesh_entity, normal_map->TexCoord);
    // The track is fixed by the map, its texel size, and its scale, so a mesh edit that left the parameterization alone keeps it.
    const auto key = HashParams(0x2545f4914f6cdd1dull, normal_map->Image, length_per_uv, normal_map->Scale);
    if (existing && existing->Key == key) return;
    const auto image = length_per_uv > 0 ? gltf::DecodeImageRgba8(r, normal_map->Image) : std::nullopt;
    if (!image || image->Width == 0 || image->Height == 0) {
        r.remove<SurfaceRelief>(mesh_entity);
        return;
    }

    // Walk a straight path across the map, one texel of surface per sample.
    // The direction is irrational in texel space, so the path covers the map rather than repeating a row.
    // Along a fixed line, directional relief sounds the same crossed either way, and a feature is heard at the rate the speed implies rather than when the object crosses it.
    // TODO: sample the map along the contact's own path, from its UV on the closest triangle and its direction of travel in that triangle's UV gradient.
    // That moves the integration to the audio thread and turns the contact filter into a mip level over a 2-D slope field, so the pool holds whole maps.
    // One 2048-square map as a two-channel slope pyramid costs about 22 MB against a track's 256 KB, and every pool entry stays resident, which turns MaxSurfaceTracks into a memory budget.
    const float texel = length_per_uv / float(image->Width);
    constexpr float Slope = std::numbers::phi_v<float> - 1; // 1/phi, the least well approximated by a ratio of texel counts
    const float dir_x = 1.f / std::sqrt(1 + Slope * Slope), dir_y = Slope * dir_x;
    const float leak = std::exp(-texel / ReliefLeakLength);

    std::vector<float> heights(TrackSamples);
    float x = 0, y = 0, height = 0;
    for (uint32_t i = 0; i < TrackSamples; ++i) {
        // A tangent-space normal is the surface gradient, n proportional to (-dh/du, -dh/dv, 1), with X and Y scaled as the core material property scales them.
        const vec3 n = SampleNormal(*image, x, y);
        const float nz = std::max(n.z, 1e-3f);
        const float slope = -normal_map->Scale * (n.x * dir_x + n.y * dir_y) / nz;
        height = height * leak + slope * texel;
        heights[i] = height;
        x += dir_x;
        y += dir_y;
    }

    r.emplace_or_replace<SurfaceRelief>(mesh_entity, SurfaceRelief{std::make_shared<const RoughnessTrack>(MakeProfileTrack(heights, texel)), key, source_key});
}
