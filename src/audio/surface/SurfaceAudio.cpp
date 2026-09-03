#include "SurfaceAudio.h"

#include "Reactive.h"
#include "TransformMath.h"
#include "audio/ContactScene.h"
#include "audio/Fft.h"
#include "audio/ModalAudio.h"
#include "gltf/SourceTexture.h"
#include "numeric/vec2.h"
#include "physics/PhysicsContact.h"
#include "physics/PhysicsTypes.h"
#include "render/MaterialComponents.h"
#include "viewport/ViewportEvents.h"

#include <entt/entity/registry.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <unordered_map>

/***** Mesoscale relief, sampled from a normal map *****/

namespace {
// The alternatives a contact can be resolved under, each named by its own environment variable and read once.
// All are off by default.
const struct Gates {
    bool ContactSprings, SlidingRegistration, RelaxDamping, ContactTilt;
} Gate{
    .ContactSprings = std::getenv("CONTACT_SPRINGS") != nullptr,
    .SlidingRegistration = std::getenv("SLIDING_REGISTRATION") != nullptr,
    .RelaxDamping = std::getenv("RELAX_DAMPING") != nullptr,
    .ContactTilt = std::getenv("CONTACT_TILT") != nullptr,
};

// A pool key quantized to quarter octaves, so one entry serves every contact whose value wobbles within the bucket.
// Maps nonpositive frequencies to octave key zero.
double QuarterOctave(double x) { return x > 0 ? std::exp2(std::round(std::log2(x) * 4) / 4) : 0.0; }

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
    const vec3 top = numeric::Mix(texel(x0, y0), texel(x1, y0), fx);
    const vec3 bottom = numeric::Mix(texel(x0, y1), texel(x1, y1), fx);
    return numeric::Mix(top, bottom, fy);
}
} // namespace

void UpdateSurfaceRelief(entt::registry &r, entt::entity node_entity, entt::entity mesh_entity, bool geometry_changed) {
    // A surface names its own map only to override the one the mesh's material already supplies.
    const auto *surface = r.try_get<const ContactSurface>(node_entity);
    const auto normal_map = [&]() -> std::optional<gltf::NormalMapRef> {
        if (!surface || !surface->NormalTexture) return gltf::MeshMaterialNormalMap(r, mesh_entity);
        const auto &nt = *surface->NormalTexture;
        const auto image = gltf::TextureImageIndex(r, nt.Texture);
        if (!image) return {};
        return gltf::NormalMapRef{.Image = *image, .TexCoord = nt.TexCoord, .Scale = nt.Scale};
    }();
    if (!normal_map) {
        r.remove<SurfaceRelief>(node_entity);
        return;
    }
    // Measuring the parameterization walks every triangle, so a surface edit that left the map alone stops here.
    const auto source_key = HashParams(0xff51afd7ed558ccdull, normal_map->Image, normal_map->TexCoord, normal_map->Scale);
    const auto *existing = r.try_get<const SurfaceRelief>(node_entity);
    if (!geometry_changed && existing && existing->SourceKey == source_key) return;

    // Lengths stay mesh-local, so one track serves every node instancing it, each sizing it by its own world scale.
    const float length_per_uv = LocalLengthPerUv(r, mesh_entity, normal_map->TexCoord);
    // The track is fixed by the map, its texel size, and its scale, so a mesh edit that left the parameterization alone keeps it.
    const auto key = HashParams(0x2545f4914f6cdd1dull, normal_map->Image, length_per_uv, normal_map->Scale);
    if (existing && existing->Key == key) return;
    const auto image = length_per_uv > 0 ? gltf::DecodeImageRgba8(r, normal_map->Image) : std::nullopt;
    if (!image || image->Width == 0 || image->Height == 0) {
        r.remove<SurfaceRelief>(node_entity);
        return;
    }

    // Walk a straight path across the map, one texel of surface per sample.
    // The direction is irrational in texel space, so the path covers the map rather than repeating a row.
    // TODO: Sample along the contact path using the closest triangle's UV gradient.
    constexpr float Slope = std::numbers::phi_v<float> - 1; // 1/phi, the least well approximated by a ratio of texel counts
    const float dir_x = 1.f / std::sqrt(1 + Slope * Slope), dir_y = Slope * dir_x;
    const vec2 step_uv{dir_x / float(image->Width), dir_y / float(image->Height)};
    const float step_uv_length = numeric::Length(step_uv);
    const float step_length = length_per_uv * step_uv_length;
    const vec2 travel = step_uv / step_uv_length;
    const float leak = std::exp(-step_length / ReliefLeakLength);

    std::vector<float> heights(TrackSamples);
    float x = 0, y = 0, height = 0;
    for (uint32_t i = 0; i < TrackSamples; ++i) {
        // A tangent-space normal satisfies n proportional to (-dh/du, -dh/dv, 1).
        const vec3 n = SampleNormal(*image, x, y);
        const float nz = std::max(n.z, 1e-3f);
        const float slope = -normal_map->Scale * (n.x * travel.x + n.y * travel.y) / nz;
        height = height * leak + slope * step_length;
        heights[i] = height;
        x += dir_x;
        y += dir_y;
    }

    r.emplace_or_replace<SurfaceRelief>(node_entity, SurfaceRelief{std::make_shared<const RoughnessTrack>(MakeProfileTrack(heights, step_length)), key, source_key});
}

/***** Scene contact inputs *****/

namespace {
// The model's reactive tracking, kept apart from the collision path's.
namespace surface_changes {
struct SurfaceEdit {};
struct SurfaceMaterial {};
struct SurfaceGeometry {};
struct SoundControls {};
} // namespace surface_changes

// Sample spacing of the track synthesized from a surface's roughness parameters, m.
// Uses the surface's shortest wavelength as the track sampling interval.
float SynthesizedFinishSpacing(const ContactSurface &s) { return FinishTrackSpacing(s.ShortWavelength); }

// Content key of a surface's microscale finish track. A measured profile is hashed whole.
uint64_t FinishTrackKey(const ContactSurface &s) {
    if (s.HasMeasuredProfile()) {
        auto key = HashParams(0x9e3779b97f4a7c15ull, s.SampleSpacing, s.Profile.size());
        for (const float h : s.Profile) key = HashParams(key, h);
        return key;
    }
    return HashParams(0x632be59bd9b4e019ull, s.CorrelationLength, s.SpectralSlope, s.ShortWavelength, SynthesizedFinishSpacing(s));
}

uint64_t ContactVoiceId(const SustainedContact &c, uint32_t side) { return c.Id * 2 + side; }

const std::shared_ptr<const RoughnessTrack> &PooledTrack(const SurfaceAudioState &s, int32_t index) { return s.SurfaceTracks[uint32_t(index)].Owned; }

// Each element keeps the tallest crest across the workers' gathers.
void MergeCrests(std::vector<float> &into, std::span<const float> part) {
    if (part.empty()) return;
    if (into.empty()) into.assign(part.size(), -std::numeric_limits<float>::max());
    for (size_t e = 0; e < into.size() && e < part.size(); ++e) into[e] = std::max(into[e], part[e]);
}

SamplePointBlend NearestSamplePoints(const std::vector<vec3> &positions, vec3 local_point) {
    if (positions.size() < 2) return {};
    const auto dist2 = [&](uint32_t i) { const auto d = positions[i] - local_point; return numeric::Dot(d, d); };
    uint32_t first = 0, second = 0;
    float d_first = std::numeric_limits<float>::max(), d_second = d_first;
    for (uint32_t i = 0; i < positions.size(); ++i) {
        const float d = dist2(i);
        if (d < d_first) {
            second = first;
            d_second = d_first;
            first = i;
            d_first = d;
        } else if (d < d_second) {
            second = i;
            d_second = d;
        }
    }
    const float d1 = std::sqrt(d_first), d2 = std::sqrt(d_second);
    const float w = d1 + d2 > 0 ? d2 / (d1 + d2) : 1.f;
    return {{first, second, first}, {w, 1 - w, 0}};
}

SamplePointBlend ShapeBlendAt(const ModalModes &modes, vec3 local_point) {
    if (modes.Indices.size() < 3) return NearestSamplePoints(modes.Positions, local_point);

    SamplePointBlend best;
    float best_distance2 = std::numeric_limits<float>::max();
    for (size_t i = 0; i + 2 < modes.Indices.size(); i += 3) {
        const std::array tri{modes.Indices[i], modes.Indices[i + 1], modes.Indices[i + 2]};
        const auto hit = ClosestPointOnTriangle(local_point, modes.Positions[tri[0]], modes.Positions[tri[1]], modes.Positions[tri[2]]);
        const vec3 offset = hit.Position - local_point;
        if (const float distance2 = numeric::Dot(offset, offset); distance2 < best_distance2) {
            best_distance2 = distance2;
            best = {tri, hit.Weights};
        }
    }
    return best;
}

// The two tracks one body contributes to a contact. Neither depends on which voice reads them.
struct SideTracks {
    ContactTrack Finish, Relief;
};

// A body with no acoustic surface contributes the default finish, so a contact is always fully specified.
SideTracks ResolveSideTracks(const entt::registry &r, ModalAudio &m, const SustainedContactSide &side, entt::entity node, float sample_rate) {
    auto &surface = Surface(m);
    static constexpr ContactSurface DefaultSurface{};
    static const uint64_t default_key = FinishTrackKey(DefaultSurface);
    const auto *authored = r.try_get<const ContactSurface>(node);
    const auto *relief = r.try_get<const SurfaceRelief>(node);
    // An edit whose derived key is not yet computed is hashed here instead.
    const auto *memo = r.try_get<const SurfaceFinishKey>(node);
    const uint64_t finish_key = authored ? (memo ? memo->Value : FinishTrackKey(*authored)) : default_key;
    const auto *node_transform = r.try_get<const WorldTransform>(node);
    const float node_scale = node_transform ? MeanScale(node_transform->S) : 0.f;
    // Both tracks are read at the sweep speed, so a sample advances the same surface distance whatever their spacings are.
    const float step = numeric::Length(side.SweepVelocity) / sample_rate;

    SideTracks out;
    const auto make_track = [&](int32_t index, float sigma, float size) {
        const auto &track = *PooledTrack(surface, index);
        const float spacing = track.Spacing * size;
        const float rate = step / spacing;
        return ContactTrack{
            .Index = index,
            .Rate = rate,
            .Sigma = sigma,
            .Step = step,
            .Spacing = spacing,
        };
    };
    // Microscale finish: a surface's measured profile, or a track synthesized from the roughness parameters it states.
    const auto &s = authored ? *authored : DefaultSurface;
    if (const bool measured = s.HasMeasuredProfile(); measured || s.Roughness > 0) {
        const auto index = AdoptSurfaceTrack(surface, finish_key, [&] {
            return std::make_shared<const RoughnessTrack>(
                measured ? MakeProfileTrack(s.Profile, s.SampleSpacing) : SynthesizeRoughness(s.CorrelationLength, s.SpectralSlope, s.ShortWavelength, SynthesizedFinishSpacing(s), TrackSamples)
            );
        });
        // Scale unit-height synthesized tracks by roughness while preserving measured-profile height.
        if (index >= 0) out.Finish = make_track(index, measured ? PooledTrack(surface, index)->Rms : s.Roughness, 1.f);
    }

    // Mesoscale relief: this body's normal map sampled along its contact path, at its own sweep.
    if (relief && node_scale > 0) {
        const auto index = AdoptSurfaceTrack(surface, relief->Key, [&] { return relief->Track; });
        // One track serves every node instancing the mesh, each sizing it by its own scale.
        if (index >= 0) out.Relief = make_track(index, relief->Track->Rms * node_scale, node_scale);
    }
    return out;
}

// One band of the pair's composite surface.
struct FinishSpec {
    float Correlation{0}, Slope{0}, Sigma{0}, Cutoff{0};
    std::shared_ptr<const RoughnessTrack> Profile;
    float Spacing{0};
    uint64_t Key{0}; // Content key of that trace, which its numbers do not identify.
    uint32_t Side{0};
};

// Uses slot side*2 for finish and side*2+1 for mesoscale relief.
// A zero sigma is absent.
using FinishSpecs = std::array<FinishSpec, 4>;

SubCutoffRoughness CompositeSubCutoff(const FinishSpecs &specs, float spacing) {
    std::array<std::pair<double, double>, 2> bands{}; // Height coefficient and exponent.
    uint32_t band_count = 0;
    double read_width = 0;
    for (uint32_t i = 0; i < specs.size(); i += 2) {
        const auto &spec = specs[i];
        // Split the resolved and synthesized bands at the finish sampling limit.
        const double cutoff = std::max(double(spec.Cutoff), 2 * double(spacing));
        const auto band = UnresolvedBand(spec.Sigma, spec.Correlation, cutoff, spec.Slope);
        if (band.Amplitude <= 0) continue;
        bands[band_count++] = {band.Amplitude, band.Hurst};
        read_width = read_width > 0 ? std::min(read_width, band.MaxWidth) : band.MaxWidth;
    }
    // Two finishes bear inside one contact, so their bands add in variance there.
    // Select the exponent weighted by each band's contribution at the evaluated width.
    double variance = 0, weighted = 0;
    for (uint32_t i = 0; i < band_count; ++i) {
        const auto [amplitude, hurst] = bands[i];
        const double share = amplitude * amplitude * std::pow(read_width, 2 * hurst);
        variance += share;
        weighted += share * hurst;
    }
    if (!(variance > 0)) return {};
    const double hurst = weighted / variance;
    return {.Amplitude = std::sqrt(variance) / std::pow(read_width, hurst), .Hurst = hurst, .MaxWidth = read_width};
}

// One Hertz contact, whose elastic constants belong to the pair rather than to either body.
// Resolved once, so both voices agree on the force between them.
struct ResolvedContact {
    // The contact in this body's own frame, the frame its mode shapes are defined in.
    struct Side {
        // Subsequent values use the frame of this side's model node.
        // Null when the side has nothing to excite.
        entt::entity ModelEntity{entt::null};
        SamplePointBlend Blend{};
        vec3 Normal{0}; // Unit contact normal, directed into this body.
        vec3 SlipDir{0}; // Unit slip direction, which the frictional force acts along.
        // Direction each surface's geometric force drives this body, one per contact side, signed so the two bodies oppose.
        std::array<vec3, 2> SweepDir{};
        double Stiffness{0};
        float SpotStiffness{0};
        float StaticPenetration{0};
        // This side's force scale on the pair's shared element springs, from the stiffness cap.
        float SpringScale{1};
        // Where this side's modes are read for each distributed-excitation bin.
        std::array<SamplePointBlend, MaxSpringBins> SpringBinBlend{};
        int32_t SweepIndex{-1}; // The moving-load sweep table's pool slot. Negative leaves it off.
        float NoiseRms{0}; // Root of the footprint's slip-turnover force variance at the anchor, N, before the side's spring scale.
        float NoiseStiffness{0}; // The footprint's bearing stiffness at the same anchor, N/m after the side's scale.
        // Store sweep half-extent in metres and inverse pitch inertia in kg^-1 m^-2.
        // Zero inertia leaves the face flat.
        float SpringHalfExtent{0};
        float InverseAngularInertia{0};
    };
    std::array<Side, 2> Sides;
    std::array<SideTracks, 2> Tracks; // Each body's finish and relief, adopted once for the pair.
    ContactTrack Turnover{}; // The pair's asperity site population under the quadratised exchange.
    double Stiffness{0}; // k, N/m^(3/2). Zero for a contact bearing on a bed of asperities.
    float StaticPenetration{0};
    float ConformalLength{0}; // u0 = h_rms/Alpha, m. Positive for a contact two faces fix the area of.
    float DampingFactor{0}; // Hunt-Crossley dissipation per unit approach speed, dimensionless.
    uint32_t SpotCount{0}; // Cells of the asperity bed rendered along the track. Positive selects the bed.
    float SpotWeight{0};
    float SpotStiffness{0}; // k of one asperity, N/m^(3/2).
    float CellSpread{0}; // Spread of asperity heights about their cell's local mean, m.
    float SurfaceCompliance{1}; // Share of a surface excursion the bed takes, the bulk having taken the rest.
    float ShearStiffness{0}; // N/m, the patch shear spring the frictional force acts through.
    float FlankStiffness{0}; // N/m, the oblique-flank micro-slip spring the normal channel acts through.
    // The same junction discretised over the bearing population's spread of breakaway strengths. Zero bins run the single element.
    uint32_t FlankBins{0};
    std::array<FlankJunctionBin, MaxFlankBins> FlankBin{};
    float RelaxScale{0}; // N/m^2, the relaxation channel's coefficient, from the contact's second force derivative.
    // The element/spring container (CONTACT_SPRINGS), shared by the pair. Negative leaves the bed path.
    int32_t SpringIndex{-1};
    float SpringRate{0}; // Elements the contact advances per output sample.
    float SlipRate{0}; // Elements the slower surface falls behind the array per output sample.
    uint32_t SlowSide{0}; // Which surface that is.
    uint32_t SpringBins{0}; // Distributed-excitation bins across the footprint. Zero leaves the channel off.
    // The turnover shot process's geometry, all m.
    float JunctionSpacing{0};
    float JunctionTransit{0};
    float FootprintLength{0};
};

// The footprint's slip-turnover fluctuation off a resolved sweep table, the bearing stiffness taking `stiffness_scale`.
void ReadTurnoverNoise(ResolvedContact::Side &side, const SurfaceAudioState &s, int32_t index, float stiffness_scale) {
    if (index < 0) return;
    const auto *table = s.SweepTables[uint32_t(index)].Live.load(std::memory_order_acquire);
    if (!table) return;
    side.NoiseRms = std::sqrt(std::max(table->NoiseVariance, 0.f));
    side.NoiseStiffness = stiffness_scale * std::max(table->NoiseStiffness, 0.f);
}

constexpr double MinContactSamples{6};
// The expected maximum closing rate the stiffness cap is sized at, their simulation value, m/s.
constexpr double MaxImpactSpeed{0.5};

ResolvedContact ResolveContact(const entt::registry &r, ModalAudio &m, const SustainedContact &c, const std::array<ContactNodes, 2> &nodes, const SurfaceSoundControls &controls) {
    auto &surface = Surface(m);
    ResolvedContact out;
    std::array<double, 2> curvature{};
    for (uint32_t i = 0; i < c.Sides.size(); ++i) {
        auto &side = out.Sides[i];
        curvature[i] = SurfaceCurvature(r, nodes[i].Geometry, c.Point).value_or(0.0);
        // Reject sides with inconsistent shape and position counts or missing bodies.
        const auto *modes = r.valid(nodes[i].Model) ? r.try_get<const ModalModes>(nodes[i].Model) : nullptr;
        const auto *transform = modes ? r.try_get<const WorldTransform>(nodes[i].Model) : nullptr;
        if (!transform || modes->Positions.empty() || modes->Shapes.size() != modes->Positions.size()) continue;

        // Physics reports the contact in world space oriented toward the second body.
        // Transform subsequent values into the model node's frame used by the mode shapes.
        side.ModelEntity = nodes[i].Model;
        const auto &wt = *transform;
        const vec3 local_point = InverseTransformPoint(wt, c.Point);
        // The first body is pressed and dragged the other way.
        const float toward = i == 0 ? -1.f : 1.f;
        side.Normal = InverseTransformDir(wt, toward * c.Normal);
        vec3 slip_world = UnitOrZero(c.Slip);
        if (Gate.ContactSprings && numeric::Length(c.Slip) < controls.MinSlipSpeed) {
            const vec3 axis = std::abs(c.Normal.x) < 0.5f ? vec3{1, 0, 0} : vec3{0, 1, 0};
            slip_world = numeric::Normalize(numeric::Cross(c.Normal, axis));
        }
        side.SlipDir = InverseTransformDir(wt, toward * slip_world);
        for (uint32_t j = 0; j < c.Sides.size(); ++j) {
            const float own = j == i ? 1.f : -1.f;
            side.SweepDir[j] = InverseTransformDir(wt, own * UnitOrZero(c.Sides[j].SweepVelocity));
        }
        side.Blend = ShapeBlendAt(*modes, local_point);
    }
    // 1/E* and kappa1 + kappa2 are both sums over the two bodies, so the stiffness and patch belong to the contact.
    const double inv_modulus = InvEffectiveModulus(MaterialOf(r, nodes[0].Surface, nodes[0].Model), MaterialOf(r, nodes[1].Surface, nodes[1].Model));
    out.DampingFactor = 1.5f * std::max(1.f - c.Restitution, 0.f) * controls.ContactDamping;
    const float sample_rate = LiveBank(m).SampleRate;
    for (uint32_t i = 0; i < c.Sides.size(); ++i) out.Tracks[i] = ResolveSideTracks(r, m, c.Sides[i], nodes[i].Surface, sample_rate);

    // The surface gradient a track presents, per unit length along the surface, from its height scale and spacing.
    const auto slope_rms = [&surface](const ContactTrack &t) {
        if (t.Index < 0 || t.Spacing <= 0) return 0.f;
        return t.Sigma * PooledTrack(surface, t.Index)->SlopeRms / t.Spacing;
    };
    static constexpr ContactSurface DefaultSurface{};
    const auto *surface_a = r.try_get<const ContactSurface>(nodes[0].Surface);
    const auto *surface_b = r.try_get<const ContactSurface>(nodes[1].Surface);
    const auto &sa = surface_a ? *surface_a : DefaultSurface;
    const auto &sb = surface_b ? *surface_b : DefaultSurface;
    // Whichever surface is rougher dominates the composite spectrum, so its slope sets the Hurst exponent.
    const double hurst = HurstExponent(sa.Roughness >= sb.Roughness ? sa.SpectralSlope : sb.SpectralSlope);
    const double roughness = std::sqrt(double(sa.Roughness) * sa.Roughness + double(sb.Roughness) * sb.Roughness);
    const auto coeffs = PerssonSeparationCoefficients(hurst, RoughnessBandRatio);
    out.ConformalLength = coeffs.Alpha > 0 ? float(roughness / coeffs.Alpha) : 0.f;
    out.ShearStiffness = roughness > 0 ? float(MindlinShearRatio * c.NormalForce / (0.4 * roughness)) : 0.f;
    const double combined = CombinedCurvature(curvature.front(), curvature.back());
    // Use the smooth Hertz spring for zero-roughness surfaces.
    out.Stiffness = ContactStiffness(inv_modulus, combined);
    out.StaticPenetration = float(StaticPenetration(c.NormalForce, out.Stiffness));
    const double hertz_radius = ContactPatchRadius(c.NormalForce, inv_modulus, combined);
    const double rough_radius = ConformalPatchWidth(combined, roughness, coeffs);
    const double patch_radius = std::max(hertz_radius, rough_radius);
    const double patch_area = std::numbers::pi * patch_radius * patch_radius;
    const double bound_area = c.NominalArea > 0 ? std::min(double(c.NominalArea), patch_area) : patch_area;
    const auto mesoscale_slope = [](const ContactSurface &s, double relief_gradient) {
        if (relief_gradient > 0) return relief_gradient;
        return s.Waviness > 0 && s.WavinessLength > 0 ? double(s.Waviness) / s.WavinessLength : 0.0;
    };
    const auto equal_area_width = [](double area) { return 2.f * std::sqrt(float(area) / std::numbers::pi_v<float>); };
    const auto strip_count = [](double area, double length, double strip_width) {
        return strip_width > 0 && length > 0 ? std::max(area / (length * strip_width), 1.0) : 1.0;
    };
    const double footprint_length = c.NominalExtent > 0 ? double(c.NominalExtent) : double(equal_area_width(bound_area));
    const double meso_a = mesoscale_slope(sa, slope_rms(out.Tracks.front().Relief));
    const double meso_b = mesoscale_slope(sb, slope_rms(out.Tracks.back().Relief));
    const double waviness_gradient = std::sqrt(meso_a * meso_a + meso_b * meso_b);
    const double eligible_area = waviness_gradient > 0 ? RealContactArea(c.NormalForce, bound_area, inv_modulus, waviness_gradient) : bound_area;
    const float eligible_width = equal_area_width(eligible_area);
    const float region_width = std::max({equal_area_width(bound_area), eligible_width, c.NominalExtent});
    const auto size_window = [&surface](ContactTrack &t, float width) {
        if (t.Index < 0 || t.Spacing <= 0) return;
        const auto samples = float(PooledTrack(surface, t.Index)->Heights.size());
        t.Window = std::min(std::max(width / t.Spacing, 2 * t.Rate), samples);
    };
    for (auto &tracks : out.Tracks) {
        size_window(tracks.Finish, region_width);
        size_window(tracks.Relief, region_width);
    }

    double bed_total_spots = 0, bed_height_rms = 0;
    if (out.ConformalLength > 0) {
        // The spectral moments of the gap, which both surfaces contribute to independently.
        const auto track_of = [&surface](const ContactTrack &t) { return PooledTrack(surface, t.Index).get(); };
        const auto usable = [](const ContactTrack &t) { return t.Index >= 0 && t.Spacing > 0; };
        double m2 = 0, m4 = 0;
        for (const auto &tracks : out.Tracks) {
            const auto &t = tracks.Finish;
            if (!usable(t)) continue;
            const auto *track = track_of(t);
            const double gradient = double(t.Sigma) * track->SlopeRms / t.Spacing;
            const double curvature = double(t.Sigma) * track->CurvatureRms / (double(t.Spacing) * t.Spacing);
            m2 += gradient * gradient;
            m4 += curvature * curvature;
        }
        const double peak_density = m2 > 0 ? std::sqrt(m4 / m2) / (2 * std::numbers::pi) : 0.0;
        // Both surfaces contribute to the gap independently, so their curvatures and heights add in quadrature.
        const auto over_finishes = [&](auto &&per_track) {
            double total = 0;
            for (const auto &tracks : out.Tracks) {
                if (!usable(tracks.Finish)) continue;
                const double v = per_track(tracks.Finish, *track_of(tracks.Finish));
                total += v * v;
            }
            return std::sqrt(total);
        };
        double window_floor = 0;
        for (const auto &tracks : out.Tracks) {
            const auto &t = tracks.Finish;
            if (usable(t)) window_floor = std::max(window_floor, double(TrackHalfVarianceWidth(*track_of(t))) * t.Spacing);
        }
        const double cell_window = BedCellWindow(eligible_width, region_width, peak_density, window_floor);
        const auto bed = ResolveAsperityBed(
            c.NormalForce, eligible_width, region_width, inv_modulus, peak_density, cell_window,
            [&](double) {
                return over_finishes([&](const ContactTrack &t, const RoughnessTrack &track) {
                    return double(t.Sigma) * track.CurvatureRms / (double(t.Spacing) * t.Spacing);
                });
            },
            [&](double width) {
                return over_finishes([&](const ContactTrack &t, const RoughnessTrack &track) {
                    return double(t.Sigma) * TrackWindowRms(track, float(width / t.Spacing));
                });
            }
        );
        bed_total_spots = bed.TotalSpots;
        bed_height_rms = bed.HeightRms;
        out.SpotCount = bed.SpotCount;
        out.SpotWeight = float(bed.SpotWeight);
        out.SpotStiffness = float(bed.SpotStiffness);
        out.CellSpread = float(bed.CellSpread);
        out.SurfaceCompliance = float(BedSurfaceShare(inv_modulus, bound_area, bed));
        if (peak_density > 0 && bed.SpotRadius > 0 && bed.HeightRms > 0) {
            const double u = double(bed.Separation);
            const double q = 0.5 * std::erfc(u / std::numbers::sqrt2);
            const double phi = std::exp(-0.5 * u * u) / std::sqrt(2 * std::numbers::pi);
            if (const double compression = bed.HeightRms * (phi / q - u); q > 0 && compression > 0) {
                out.JunctionSpacing = float(1.0 / peak_density);
                out.JunctionTransit = float(std::min(2 * std::sqrt(bed.SpotRadius * compression), double(region_width)));
            }
        }
        float step = 0;
        for (const auto &tracks : out.Tracks) {
            if (usable(tracks.Finish)) step = std::max(step, tracks.Finish.Step);
        }
        if (Gate.ContactSprings) {
            FinishSpecs specs{};
            uint32_t rough_count = 0;
            float spacing = std::numeric_limits<float>::max();
            for (uint32_t side = 0; side < 2; ++side) {
                const auto &s = side ? sb : sa;
                const auto &finish = out.Tracks[side].Finish;
                if (s.HasMeasuredProfile() && finish.Index >= 0) {
                    const auto &track = PooledTrack(surface, finish.Index);
                    if (track->Band.CorrelationLength <= 0 || track->Rms <= 0) continue;
                    specs[side * 2] = {
                        .Correlation = track->Band.CorrelationLength, .Slope = track->Band.SpectralSlope, .Sigma = track->Rms, .Cutoff = track->Cutoff, .Profile = track, .Spacing = track->Spacing, .Key = surface.SurfaceTracks[uint32_t(finish.Index)].Key, .Side = side
                    };
                    spacing = std::min(spacing, track->Spacing);
                } else if (s.Roughness > 0) {
                    specs[side * 2] = {.Correlation = s.CorrelationLength, .Slope = s.SpectralSlope, .Sigma = s.Roughness, .Cutoff = s.ShortWavelength, .Profile = nullptr, .Spacing = 0, .Key = 0, .Side = side};
                    spacing = std::min(spacing, SynthesizedFinishSpacing(s));
                } else {
                    continue;
                }
                ++rough_count;
                const auto &relief = out.Tracks[side].Relief;
                if (relief.Index >= 0) {
                    const auto &track = PooledTrack(surface, relief.Index);
                    const float scale = track->Spacing > 0 ? relief.Spacing / track->Spacing : 1.f;
                    if (track->Band.CorrelationLength > 0 && relief.Sigma > 0) {
                        specs[side * 2 + 1] = {
                            .Correlation = track->Band.CorrelationLength * scale, .Slope = track->Band.SpectralSlope, .Sigma = relief.Sigma, .Cutoff = specs[side * 2].Correlation, .Profile = track, .Spacing = relief.Spacing, .Key = surface.SurfaceTracks[uint32_t(relief.Index)].Key, .Side = side
                        };
                    }
                } else if (s.Waviness > 0 && s.WavinessLength > 0) {
                    specs[side * 2 + 1] = {.Correlation = s.WavinessLength, .Slope = s.SpectralSlope, .Sigma = s.Waviness, .Cutoff = specs[side * 2].Correlation, .Profile = nullptr, .Spacing = 0, .Key = 0, .Side = side};
                }
            }
            const RoughnessTrack *sweep = nullptr;
            for (uint32_t side = 0; side < 2; ++side) {
                const auto &spec = specs[side * 2];
                if (spec.Profile && (!sweep || spec.Profile->Spacing < sweep->Spacing)) sweep = spec.Profile.get();
            }
            if (sweep) spacing = sweep->Spacing;
            if (rough_count > 0 && spacing > 0) {
                const double curv_bucket = QuarterOctave(combined);
                const double reach_width = c.NominalArea > 0 || combined <= 0 ? QuarterOctave(std::max(footprint_length, double(spacing))) : 0.0;
                float min_correlation = std::numeric_limits<float>::max();
                double sigma2 = 0;
                for (uint32_t i = 0; i < specs.size(); ++i) {
                    if (specs[i].Sigma <= 0) continue;
                    // Element width follows the finishes alone, waviness being smooth across elements.
                    if (i % 2 == 0) min_correlation = std::min(min_correlation, specs[i].Correlation);
                    sigma2 += double(specs[i].Sigma) * specs[i].Sigma;
                }
                // Elements of a quarter correlation length.
                const auto element_columns = std::max(uint32_t(std::round(min_correlation / (4 * spacing))), 1u);
                const double engagement_max = 6 * std::sqrt(sigma2);
                const auto sub_cutoff = CompositeSubCutoff(specs, spacing);
                constexpr double strip_correlations = 4.0;
                constexpr uint32_t strip_realizations = 8;
                const double bearing_width = footprint_length > 0 ? bound_area / footprint_length : 0.0;
                const double strip_reach = strip_correlations * min_correlation;
                const double strip_width = bearing_width > 0 ? std::min(bearing_width, strip_reach) : strip_reach;
                const auto rows = fft::DirectLength(std::max(uint32_t(std::llround(strip_width / double(spacing))), 8u));
                const double ribbon_load =
                    double(c.NormalForce) / strip_count(bound_area, footprint_length, double(strip_realizations) * rows * double(spacing));
                const double load_bucket = c.NormalForce > 0 ? QuarterOctave(ribbon_load) : 0.0;
                const auto columns = sweep ? uint32_t(sweep->Heights.size()) : fft::DirectLength(std::max(uint32_t(std::llround(double(SurfaceRepeatLength) / double(spacing))), 4 * element_columns));
                constexpr uint32_t SpringKnots = 56;
                const uint64_t surface_key = HashParams(
                    0x41c7e9b2d5a8f036ull, specs[0].Correlation, specs[0].Slope, specs[0].Sigma, specs[0].Cutoff,
                    specs[1].Correlation, specs[1].Slope, specs[1].Sigma, specs[1].Cutoff,
                    specs[2].Correlation, specs[2].Slope, specs[2].Sigma, specs[2].Cutoff,
                    specs[3].Correlation, specs[3].Slope, specs[3].Sigma, specs[3].Cutoff, spacing,
                    // A trace's numbers are read off it and do not identify it, so its content key joins them.
                    specs[0].Key, specs[1].Key, specs[2].Key, specs[3].Key,
                    inv_modulus, curv_bucket, reach_width, rows, columns, strip_realizations
                );
                const uint64_t key = HashParams(surface_key, load_bucket);
                const auto build_surface = [&] {
                    std::vector<float> gap(rows, 0.f);
                    if (curv_bucket > 0) {
                        for (uint32_t row = 0; row < rows; ++row) {
                            const double y = (double(row) - 0.5 * (rows - 1)) * double(spacing);
                            gap[row] = float(0.5 * curv_bucket * y * y);
                        }
                    }
                    const uint32_t set_count = columns / element_columns;
                    const double element_width = double(element_columns) * spacing;
                    constexpr double FootprintSpans = 8;
                    const uint32_t tiles = reach_width > 0 && set_count > 0 && element_width > 0 ? std::max(1u, uint32_t(std::ceil(FootprintSpans * reach_width / (double(set_count) * element_width)))) : 1u;
                    const uint32_t span_count = set_count * tiles;
                    const double span_length = double(span_count) * element_width;
                    struct SharedBand {
                        RoughnessField Field;
                        float Sigma{0};
                        uint32_t Side{0};
                    };
                    std::vector<SharedBand> shared;
                    std::array<FinishSpec, 4> drawn{};
                    uint32_t drawn_count = 0;
                    for (const auto &spec : specs) {
                        if (spec.Sigma <= 0) continue;
                        // Build measured bands independently because each contains a fixed realization.
                        auto *const same = spec.Profile ? drawn.begin() + drawn_count : std::ranges::find_if(drawn.begin(), drawn.begin() + drawn_count, [&spec](const FinishSpec &d) {
                            return !d.Profile && (!Gate.SlidingRegistration || d.Side == spec.Side) &&
                                d.Correlation == spec.Correlation && d.Slope == spec.Slope && d.Cutoff == spec.Cutoff;
                        });
                        if (same == drawn.begin() + drawn_count) drawn[drawn_count++] = spec;
                        else same->Sigma = float(std::sqrt(double(same->Sigma) * same->Sigma + double(spec.Sigma) * spec.Sigma));
                    }
                    const double ribbon_width = double(strip_realizations) * rows * spacing;
                    struct SharedDraw {
                        std::shared_ptr<const RoughnessTrack> Profile;
                        float ProfileSpacing{0};
                        float Correlation{0}, Slope{0}, Cutoff{0}, Sigma{0};
                        uint32_t Span{0}, Across{0}, Columns{0}, Rows{0}, Realization{0}, Side{0};
                    };
                    std::vector<SharedDraw> to_draw;
                    for (uint32_t i = 0; i < drawn_count; ++i) {
                        if (double(drawn[i].Correlation) <= strip_width) continue;
                        const auto wide = std::max(ribbon_width, strip_correlations * drawn[i].Correlation);
                        const uint32_t realization = Gate.SlidingRegistration ? drawn[i].Side : 0;
                        const double band_length = double(drawn[i].Correlation) * FootprintSpans >= reach_width ? span_length : double(SurfaceRepeatLength);
                        if (const auto &profile = drawn[i].Profile; profile && drawn[i].Spacing > 0) {
                            const auto span = uint32_t(std::min(double(profile->Heights.size()), std::ceil(band_length / drawn[i].Spacing)));
                            const auto across = fft::DirectLength(uint32_t(std::llround(wide / double(drawn[i].Spacing))));
                            if (span < 8 || across < 2) continue;
                            to_draw.push_back({
                                .Profile = profile,
                                .ProfileSpacing = drawn[i].Spacing,
                                .Correlation = drawn[i].Correlation,
                                .Slope = 0,
                                .Cutoff = drawn[i].Cutoff,
                                .Sigma = drawn[i].Sigma,
                                .Span = span,
                                .Across = across,
                                .Columns = 0,
                                .Rows = 0,
                                .Realization = realization,
                                .Side = drawn[i].Side,
                            });
                        } else {
                            const float coarse = drawn[i].Cutoff / SurfaceSamplesPerCutoff;
                            to_draw.push_back({
                                .Profile = nullptr,
                                .ProfileSpacing = 0,
                                .Correlation = drawn[i].Correlation,
                                .Slope = drawn[i].Slope,
                                .Cutoff = drawn[i].Cutoff,
                                .Sigma = drawn[i].Sigma,
                                .Span = 0,
                                .Across = 0,
                                .Columns = fft::DirectLength(uint32_t(std::llround(band_length / coarse))),
                                .Rows = fft::DirectLength(uint32_t(std::llround(wide / coarse))),
                                .Realization = realization,
                                .Side = drawn[i].Side,
                            });
                        }
                        drawn[i].Sigma = 0;
                    }
                    const uint64_t strip_key = [&] {
                        uint64_t k = HashParams(
                            0x7d3fa9c15b2e6084ull, spacing, inv_modulus, curv_bucket, rows, columns, strip_realizations,
                            element_columns, engagement_max, sub_cutoff.Amplitude, sub_cutoff.Hurst, sub_cutoff.MaxWidth
                        );
                        for (uint32_t i = 0; i < drawn_count; ++i) {
                            k = HashParams(k, drawn[i].Correlation, drawn[i].Slope, drawn[i].Sigma, drawn[i].Cutoff, drawn[i].Key, drawn[i].Spacing, drawn[i].Side);
                        }
                        return k;
                    }();
                    const auto build_strips = [&] {
                        constexpr size_t WorkerBudget = 1ull << 30;
                        // A field, the patch it sums from, and the other side's own field where the two are gathered apart.
                        const size_t worker_bytes = (Gate.SlidingRegistration ? 3 : 2) * sizeof(float) * size_t(columns) * rows;
                        const uint32_t affordable = uint32_t(WorkerBudget / worker_bytes);
                        const uint32_t workers = std::min<uint32_t>(
                            {strip_realizations, std::max(1u, std::thread::hardware_concurrency() - 1), std::max(1u, affordable)}
                        );
                        // Permit one over-budget worker because ribbon construction requires at least one worker.
                        if (affordable < strip_realizations) {
                            std::fprintf(
                                stderr, "SPRINGBUDGET workers=%u of %u realizations, %.2f GiB each over a %.2f GiB budget, field %ux%u\n",
                                workers, strip_realizations, double(worker_bytes) / double(1ull << 30), double(WorkerBudget) / double(1ull << 30), columns, rows
                            );
                        }
                        // Divide available threads between realization generation and transforms.
                        SetTransformThreads(std::max(1u, std::max(1u, std::thread::hardware_concurrency() - 1) / workers));
                        // Scale a single synthesized band in place without allocating a sum buffer.
                        uint32_t active = 0, only = 0;
                        for (uint32_t i = 0; i < drawn_count; ++i) {
                            if (drawn[i].Sigma > 0) {
                                ++active;
                                only = i;
                            }
                        }
                        const bool direct = !Gate.SlidingRegistration && active == 1 && !drawn[only].Profile;
                        bool ensemble = active > 0;
                        for (uint32_t i = 0; i < drawn_count; ++i) {
                            if (drawn[i].Sigma > 0 && drawn[i].Profile) ensemble = false;
                        }
                        std::vector<ElementSummits> gathered(workers);
                        std::vector<std::vector<float>> crest_gather(workers);
                        // Gather each side's crests before composing fields, then merge them in worker order.
                        std::vector<std::array<std::vector<float>, 2>> side_crests(workers);
                        const auto run_gather = [&](bool crests_only) {
                            const auto gather = [&, crests_only](uint32_t w, std::span<const float> field) {
                                if (crests_only) GatherElementCrests(crest_gather[w], field, columns, rows, gap, {}, element_columns);
                                else GatherElementSummits(gathered[w], field, columns, rows, gap, {}, element_columns, spacing, inv_modulus);
                            };
                            std::atomic<uint32_t> next{0};
                            std::vector<std::jthread> pool;
                            pool.reserve(workers);
                            for (uint32_t w = 0; w < workers; ++w) {
                                pool.emplace_back([&, w] {
                                    std::vector<float> heights(direct ? 0 : size_t(columns) * rows, 0.f);
                                    // Retain the second field until its crests are gathered, then compose it into the first.
                                    std::vector<float> apart(Gate.SlidingRegistration ? size_t(columns) * rows : 0, 0.f);
                                    for (uint32_t rep = next++; rep < strip_realizations; rep = next++) {
                                        if (direct) {
                                            auto patch = SynthesizeRoughnessPatch(drawn[only].Correlation, drawn[only].Slope, drawn[only].Cutoff, spacing, columns, rows, rep);
                                            // Use one zero-initialized addition per sample to preserve signed zeros.
                                            for (float &h : patch.Heights) h = 0.f + drawn[only].Sigma * h;
                                            gather(w, patch.Heights);
                                            continue;
                                        }
                                        std::ranges::fill(heights, 0.f);
                                        std::ranges::fill(apart, 0.f);
                                        for (uint32_t i = 0; i < drawn_count; ++i) {
                                            if (drawn[i].Sigma <= 0) continue;
                                            auto &into = Gate.SlidingRegistration && drawn[i].Side == 1 ? apart : heights;
                                            const uint32_t realization = Gate.SlidingRegistration ? rep + drawn[i].Side * strip_realizations : rep;
                                            const auto patch = drawn[i].Profile ? SynthesizeProfileField(drawn[i].Profile->Heights, drawn[i].Spacing, rows, realization) : SynthesizeRoughnessPatch(drawn[i].Correlation, drawn[i].Slope, drawn[i].Cutoff, spacing, columns, rows, realization);
                                            if (patch.Columns == columns && patch.Spacing == spacing) {
                                                for (size_t at = 0; at < into.size(); ++at) into[at] += drawn[i].Sigma * patch.Heights[at];
                                            } else {
                                                for (uint32_t col = 0; col < columns; ++col) {
                                                    for (uint32_t row = 0; row < rows; ++row) {
                                                        into[size_t(col) * rows + row] += drawn[i].Sigma * FieldHeightAt(patch, double(col) * spacing, double(row) * spacing);
                                                    }
                                                }
                                            }
                                        }
                                        if (Gate.SlidingRegistration) {
                                            GatherElementCrests(side_crests[w][0], heights, columns, rows, gap, {}, element_columns);
                                            GatherElementCrests(side_crests[w][1], apart, columns, rows, gap, {}, element_columns);
                                            for (size_t at = 0; at < heights.size(); ++at) heights[at] += apart[at];
                                        }
                                        gather(w, heights);
                                    }
                                });
                            }
                        };
                        run_gather(ensemble);
                        auto merged_crests = [&](uint32_t side) {
                            std::vector<float> out;
                            for (const auto &part : side_crests) MergeCrests(out, part[side]);
                            return out;
                        };
                        auto built = std::make_shared<SpringStrips>();
                        if (Gate.SlidingRegistration) built->SideCrests = {merged_crests(0), merged_crests(1)};
                        if (ensemble) {
                            std::vector<float> crests;
                            for (const auto &part : crest_gather) MergeCrests(crests, part);
                            std::array<EnsembleBand, 4> bands{};
                            uint32_t band_count = 0;
                            for (uint32_t i = 0; i < drawn_count; ++i) {
                                if (drawn[i].Sigma > 0) bands[band_count++] = {drawn[i].Correlation, drawn[i].Slope, drawn[i].Cutoff, drawn[i].Sigma};
                            }
                            built->Springs = BuildEnsembleSprings(
                                {bands.data(), band_count}, spacing, columns, rows, gap, curv_bucket, element_columns,
                                strip_realizations, inv_modulus, sub_cutoff, engagement_max, SpringKnots, std::move(crests)
                            );
                            // Use the realized build when a degenerate population has no crest anchor.
                            // Report fallback because the two builds use different bearing statistics.
                            if (built->Springs.Count() == 0) {
                                std::fprintf(stderr, "SPRINGENSEMBLE found no crest population, realized build instead, field %ux%u\n", columns, rows);
                                run_gather(false);
                            }
                        }
                        if (built->Springs.Count() == 0) {
                            // Merge in worker order for deterministic populations across strip assignments.
                            ElementSummits summits;
                            for (const auto &part : gathered) MergeElementSummits(summits, part);
                            built->Springs = BuildElementSprings(summits, sub_cutoff, engagement_max, SpringKnots);
                        }
                        return built;
                    };
                    const auto strips = [&]() -> std::shared_ptr<const SpringStrips> {
                        if (const auto it = surface.SpringStripsByKey.find(strip_key); it != surface.SpringStripsByKey.end()) return it->second;
                        auto built = build_strips();
                        if (surface.SpringStripsByKey.size() >= SurfaceAudioState::MaxContactSprings) surface.SpringStripsByKey.clear();
                        surface.SpringStripsByKey.emplace(strip_key, built);
                        return built;
                    }();
                    // The shared bands, drawn over the footprint's span the strips are free of.
                    for (const auto &d : to_draw) {
                        shared.emplace_back(
                            d.Profile ? SynthesizeProfileField({d.Profile->Heights.data(), d.Span}, d.ProfileSpacing, d.Across, d.Realization) : SynthesizeRoughnessPatch(d.Correlation, d.Slope, d.Cutoff, d.Cutoff / SurfaceSamplesPerCutoff, d.Columns, d.Rows, d.Realization),
                            d.Sigma, d.Side
                        );
                    }
                    auto set = std::make_shared<ContactSpringSet>();
                    set->SideCrestDev = strips->SideCrests;
                    set->Springs = strips->Springs;
                    // Each side's own long bands at each element of the spanned array.
                    // A merged pair keeps them all on the first side, as the draws themselves are merged there.
                    std::array<std::vector<float>, 2> lift{std::vector<float>(span_count, 0.f), std::vector<float>(span_count, 0.f)};
                    for (uint32_t e = 0; e < span_count; ++e) {
                        const double along = (double(e) + 0.5) * element_width;
                        for (const auto &[field, sigma, side] : shared) {
                            lift[Gate.SlidingRegistration ? side : 0][e] += sigma * FieldHeightAt(field, along, 0);
                        }
                    }
                    auto span_crests = [&](std::vector<float> &crest, std::span<const float> band) {
                        if (crest.empty()) return;
                        const auto tile = uint32_t(crest.size());
                        std::vector<float> spanned(span_count);
                        for (uint32_t e = 0; e < span_count; ++e) spanned[e] = crest[e % tile] + band[e];
                        crest = std::move(spanned);
                    };
                    std::vector<float> composite(span_count, 0.f);
                    for (uint32_t e = 0; e < span_count; ++e) composite[e] = lift[0][e] + lift[1][e];
                    span_crests(set->Springs.Crest, composite);
                    for (uint32_t side = 0; side < set->SideCrestDev.size(); ++side) span_crests(set->SideCrestDev[side], lift[side]);
                    // The footprint reads skip on the spanned crests.
                    RefreshCrestBlocks(set->Springs);
                    set->StripWidth = float(strip_realizations) * float(rows) * spacing;
                    set->SubCutoff = sub_cutoff;
                    set->Curvature = curv_bucket;
                    const auto count = set->Springs.Count();
                    const auto half = std::max(count / 2, 1u);
                    uint32_t reach = curv_bucket > 0 ? ElementReach(set->Springs, curv_bucket) : half;
                    if (reach_width > 0) {
                        reach = std::min(reach, uint32_t(0.5 * reach_width / std::max(double(set->Springs.Width), double(spacing))) + 1);
                    }
                    set->Reach = std::min(reach, half);
                    if (set->Reach >= half && count > 2) {
                        std::fprintf(
                            stderr, "SPRINGREACH reach=%u of %u elements over a %g m field, footprint %g m, so the datum does not move\n",
                            set->Reach, count, double(count) * double(set->Springs.Width), reach_width
                        );
                    }
                    return set;
                };
                const auto index = AdoptContactSprings(surface, key, [&] {
                    auto set = [&] {
                        if (const auto it = surface.SpringSurfaceByKey.find(surface_key); it != surface.SpringSurfaceByKey.end()) {
                            return std::make_shared<ContactSpringSet>(*it->second);
                        }
                        auto built = build_surface();
                        if (surface.SpringSurfaceByKey.size() >= SurfaceAudioState::MaxContactSprings) surface.SpringSurfaceByKey.clear();
                        surface.SpringSurfaceByKey.emplace(surface_key, std::make_shared<const ContactSpringSet>(*built));
                        return built;
                    }();
                    const auto count = set->Springs.Count();
                    set->Envelope = BearingDatum(set->Springs, curv_bucket, set->Reach, load_bucket);
                    const auto support_mean = set->Envelope.empty() ? 0.0 : std::accumulate(set->Envelope.begin(), set->Envelope.end(), 0.0) / double(set->Envelope.size());
                    for (auto &dev : set->SideCrestDev) {
                        if (dev.size() != set->Springs.Count() || set->Envelope.size() != dev.size()) {
                            dev.clear();
                            continue;
                        }
                        dev = CrestEnvelope(dev, set->Springs.Width, curv_bucket, set->Reach);
                        const auto mean = float(std::accumulate(dev.begin(), dev.end(), 0.0) / double(dev.size()));
                        for (float &v : dev) v -= mean;
                        double covariance = 0, variance = 0;
                        for (size_t e = 0; e < dev.size(); ++e) {
                            covariance += (double(set->Envelope[e]) - support_mean) * double(dev[e]);
                            variance += double(dev[e]) * double(dev[e]);
                        }
                        const auto share = float(variance > 0 ? covariance / variance : 0.0);
                        for (float &v : dev) v *= share;
                    }
                    FillAnchorForce(*set, std::max(count / 512, 1u));
                    return set;
                });
                if (index >= 0) {
                    out.SpringIndex = index;
                    const auto &set = *surface.ContactSprings[uint32_t(index)].Owned;
                    out.SpringRate = set.Springs.Width > 0 ? step / set.Springs.Width : 0.f;
                    out.FootprintLength = float(footprint_length);
                    // Sweep at the faster surface rate and offset the slower surface by the relative displacement.
                    const float slow_step = std::min(out.Tracks.front().Finish.Step, out.Tracks.back().Finish.Step);
                    out.SlowSide = out.Tracks.front().Finish.Step <= out.Tracks.back().Finish.Step ? 0u : 1u;
                    out.SlipRate = set.Springs.Width <= 0 ? 0.f : (step - slow_step) / set.Springs.Width;
                }
            }
        }
        // Generate one deterministic white turnover track per contact at the asperity spacing.
        // The cells read its samples as surface-fixed site heights through the clamped law.
        if (bed.SpotCount > 0 && peak_density > 0 && out.SpringIndex < 0) {
            const float spacing_t = float(1.0 / peak_density);
            const uint64_t key = HashParams(0x7a52b0c4d1e6f398ull, spacing_t, c.Id);
            const auto index = AdoptSurfaceTrack(surface, key, [&] {
                return std::make_shared<const RoughnessTrack>(SynthesizeTurnover(spacing_t, TrackSamples, key));
            });
            if (index >= 0) {
                const float rate = step / spacing_t;
                const float cell_width = std::max(float(cell_window) / spacing_t, 1.f);
                out.Turnover = {
                    .Index = index,
                    .Rate = rate,
                    .Window = std::min(std::max(region_width / spacing_t, 2 * rate), float(TrackSamples)),
                    .SubWindow = std::max(cell_width, 2 * rate),
                    .Step = step,
                    .Spacing = spacing_t,
                };
            }
        }
        out.StaticPenetration = float(bed.Separation * bed.HeightRms);
        if (bed.SpotCount > 0) {
            for (auto &tracks : out.Tracks) {
                auto &t = tracks.Finish;
                if (t.Index < 0 || t.Spacing <= 0) continue;
                t.SubWindow = std::max(float(cell_window) / t.Spacing, 2 * t.Rate);
            }
        }
    }
    for (auto &side : out.Sides) {
        side.Stiffness = out.Stiffness;
        side.SpotStiffness = out.SpotStiffness;
        side.StaticPenetration = out.StaticPenetration;
    }
    {
        const double dt = 1.0 / sample_rate;
        const auto stiffness_cap = [&](double mass, double alpha) {
            const double zeta = std::sqrt(std::numbers::pi) * std::tgamma(1 + 1 / (alpha + 1)) / std::tgamma(0.5 + 1 / (alpha + 1));
            return 0.5 * mass * (alpha + 1) * std::pow(MaxImpactSpeed, 1 - alpha) * std::pow(2 * zeta / (MinContactSamples * dt), alpha + 1);
        };
        const auto &bank = LiveBank(m);
        for (auto &side : out.Sides) {
            if (side.ModelEntity == null_entity) continue;
            const auto slot = FindModalObject(bank, side.ModelEntity);
            if (!slot) continue;
            // Derive contact-point effective mass from the coupled modal and rigid one-sample response.
            const auto o = *slot;
            const auto k0 = bank.ModeOffset[o], stride = bank.ModeCount[o], count = bank.TunedModeCount[o];
            const auto shape0 = bank.ShapeOffset[o];
            const auto base0 = shape0 + side.Blend.Points[0] * stride, base1 = shape0 + side.Blend.Points[1] * stride, base2 = shape0 + side.Blend.Points[2] * stride;
            const float w0 = side.Blend.Weights.x, w1 = side.Blend.Weights.y, w2 = side.Blend.Weights.z;
            double modal_response = 0;
            for (uint32_t k = 0; k < count; ++k) {
                const auto shape = BlendedShape(bank, base0, base1, base2, {w0, w1, w2}, k);
                const double normal = numeric::Dot(shape, side.Normal);
                modal_response += normal * normal * bank.QuadCompliance[k0 + k];
            }
            const double response = modal_response * surface.Coupling.load(std::memory_order_relaxed) * bank.DeflectionScale[o] * surface.SustainLevel.load(std::memory_order_relaxed) / sample_rate + dt * dt * bank.RigidInvMass[o];
            if (response <= 0) continue;
            const double mass = dt * dt / response;
            if (out.SpotCount > 0) {
                const double share = std::pow(double(out.SurfaceCompliance), 1.5);
                const double cell = double(out.SpotWeight) * out.SpotStiffness * share;
                const double cap = stiffness_cap(mass, 1.5);
                if (cell > cap && bed_height_rms > 0) {
                    side.SpotStiffness = float(cap / (double(out.SpotWeight) * share));
                    side.StaticPenetration = float(SolveBedSeparation(c.NormalForce, bed_total_spots, side.SpotStiffness, bed_height_rms, 1.5) * bed_height_rms);
                }
            } else if (const double cap = stiffness_cap(mass, 1.5); out.Stiffness > cap) {
                side.Stiffness = cap;
                side.StaticPenetration = float(StaticPenetration(c.NormalForce, cap));
            }
        }
    }
    if (out.SpringIndex >= 0) {
        const auto &set = *surface.ContactSprings[uint32_t(out.SpringIndex)].Owned;
        const double strips = strip_count(bound_area, footprint_length, double(set.StripWidth));
        for (auto &side : out.Sides) {
            side.SpringScale = float(strips);
            side.StaticPenetration = SolveSpringAnchor(set, c.NormalForce / strips);
        }
        const float anchor = out.Sides.front().StaticPenetration;
        const uint32_t stride = std::max(set.Springs.Count() / 512, 1u);
        const auto [moments, sampled] = SpringFlankSum(set, anchor, stride);
        // Scale whole-contact moments by the ribbon strip count.
        // The width they divide into does not, being their ratio over the same sampled elements.
        const double modulus = sampled > 0 ? strips * moments.Modulus / sampled : 0;
        const double bearing_stiffness = sampled > 0 ? strips * moments.Stiffness / sampled : 0;
        // The share of an excursion the interface takes, the bulk beneath it having taken the rest.
        // Add equal-force stiffnesses in series using the element-stack stiffness computed above.
        if (bearing_stiffness > 0 && bound_area > 0 && inv_modulus > 0) {
            const double bulk = 2 * std::sqrt(bound_area / std::numbers::pi) / inv_modulus;
            out.SurfaceCompliance = float(bulk / (bulk + bearing_stiffness));
        }
        const float probe = std::max(anchor * 1e-2f, 1e-9f);
        const auto stiffness_at = [&](float engagement) {
            const auto at = SpringFlankSum(set, engagement, stride);
            return at.Sampled > 0 ? strips * at.Moments.Stiffness / at.Sampled : 0.0;
        };
        out.RelaxScale = Gate.RelaxDamping ? float(0.5 * MindlinShearRatio * std::max((stiffness_at(anchor + probe) - stiffness_at(anchor - probe)) / (2 * double(probe)), 0.0)) : 0.f;
        const double flank_width = SpringBearingWidth(moments);
        // The gradient a track presents when a patch that wide averages it, per unit length.
        const auto slope_at_width = [&surface, flank_width](const ContactTrack &t) {
            if (t.Index < 0 || t.Spacing <= 0) return 0.f;
            const auto &track = *PooledTrack(surface, t.Index);
            return t.Sigma * TrackSlopeAtWidth(track, float(flank_width / t.Spacing)) / t.Spacing;
        };
        const double flank_a = slope_at_width(out.Tracks.front().Finish), flank_b = slope_at_width(out.Tracks.back().Finish);
        const double flank_slope = std::sqrt(flank_a * flank_a + flank_b * flank_b) / std::numbers::sqrt2;
        out.FlankStiffness = float(FlankJunctionStiffness(FlankSineCube(flank_slope), c.NormalForce, modulus));
        if (out.FlankStiffness > 0) {
            const auto shares = FlankSpectrumShareAt(set.Springs, anchor);
            out.FlankBins = FlankJunctionSpread(FlankSineCube(flank_slope), c.NormalForce, modulus, shares.Moment, shares.Force, out.FlankBin);
        }
        const float half_extent = float(set.Reach) * set.Springs.Width;
        vec3 axis{0};
        float speed = 0;
        for (const auto &cs : c.Sides) {
            if (const float s = numeric::Length(cs.SweepVelocity); s > speed) {
                speed = s;
                axis = cs.SweepVelocity / s;
            }
        }
        if (speed <= 0) axis = UnitOrZero(c.Slip);
        if (half_extent > 0 && numeric::Dot(axis, axis) > 0) {
            constexpr uint32_t bin_count = std::min(8u, MaxSpringBins);
            for (uint32_t i = 0; i < c.Sides.size(); ++i) {
                auto &side = out.Sides[i];
                if (side.ModelEntity == null_entity) continue;
                const auto *modes = r.try_get<const ModalModes>(side.ModelEntity);
                const auto *transform = r.try_get<const WorldTransform>(side.ModelEntity);
                if (!modes || !transform) continue;
                for (uint32_t bb = 0; bb < bin_count; ++bb) {
                    const float u = SpringBinOffset(bb, bin_count, half_extent);
                    side.SpringBinBlend[bb] = ShapeBlendAt(*modes, InverseTransformPoint(*transform, c.Point + u * axis));
                }
                const auto *dynamics = r.try_get<const ContactDynamics>(side.ModelEntity);
                const auto *motion = r.try_get<const PhysicsMotion>(side.ModelEntity);
                const vec3 pitch = numeric::Cross(side.Normal, UnitOrZero(InverseTransformDir(*transform, axis)));
                if (Gate.ContactTilt && c.NominalArea > 0 && dynamics && motion && IsAuthoritativeDynamicBody(*motion) && numeric::Dot(pitch, pitch) > 0) {
                    const vec3 pitch_axis = numeric::Normalize(pitch);
                    const float scale = UniformScaleRatio(r, side.ModelEntity, *modes);
                    const double sizing = motion->InertiaDiagonal ? 1.0 : std::pow(double(scale), 5.0);
                    const double inv = double(numeric::Dot(pitch_axis, dynamics->InverseInertia * pitch_axis)) / sizing;
                    side.InverseAngularInertia = float(std::max(inv, 0.0));
                    side.SpringHalfExtent = half_extent;
                }
            }
            out.SpringBins = bin_count;
        }
        if (numeric::Dot(axis, axis) > 0 && speed > 0 && half_extent > 0) {
            const auto &bank = LiveBank(m);
            for (uint32_t i = 0; i < c.Sides.size(); ++i) {
                auto &side = out.Sides[i];
                if (side.ModelEntity == null_entity) continue;
                if (numeric::Length(c.Sides[i].SweepVelocity) > 0.01f * speed) continue;
                const auto *modes = r.try_get<const ModalModes>(side.ModelEntity);
                const auto *transform = r.try_get<const WorldTransform>(side.ModelEntity);
                const auto slot = FindModalObject(bank, side.ModelEntity);
                if (!modes || !transform || !slot) continue;
                // The anchor engagement quantized to quarter octaves, so load wobble reuses a table.
                const double pen_bucket = QuarterOctave(double(side.StaticPenetration));
                if (!(pen_bucket > 0)) continue;
                const double scale_bucket = QuarterOctave(double(side.SpringScale));
                const uint64_t sweep_key = HashParams(
                    0x6d94a1c8f2e7b350ull, double(surface.ContactSprings[uint32_t(out.SpringIndex)].Key), double(*slot),
                    pen_bucket, double(set.Reach), scale_bucket
                );
                const auto index = AdoptSweepTable(surface, sweep_key, [&] {
                    const auto &springs = set.Springs;
                    const auto count = springs.Count();
                    const auto o = *slot;
                    const auto mode_count = bank.TunedModeCount[o];
                    const auto stride = bank.ModeCount[o], shape0 = bank.ShapeOffset[o];
                    // The mode shapes' normal projections at element pitch across the footprint.
                    const auto window = 2 * set.Reach + 1;
                    std::vector<float> phi(size_t(window) * mode_count);
                    for (uint32_t w = 0; w < window; ++w) {
                        const float u = (float(w) - float(set.Reach)) * springs.Width;
                        const auto bl = ShapeBlendAt(*modes, InverseTransformPoint(*transform, c.Point + u * axis));
                        const auto b0 = shape0 + bl.Points[0] * stride, b1 = shape0 + bl.Points[1] * stride, b2 = shape0 + bl.Points[2] * stride;
                        for (uint32_t k = 0; k < mode_count; ++k) {
                            phi[size_t(w) * mode_count + k] = numeric::Dot(BlendedShape(bank, b0, b1, b2, bl.Weights, k), side.Normal);
                        }
                    }
                    // The anchored element forces, whose stiffness the rows' conformity projections read.
                    std::vector<float> field(count);
                    const auto anchored = AnchoredElementSums(set, pen_bucket, scale_bucket, field);
                    auto table = std::make_shared<SweepTableSet>();
                    table->Modes = mode_count;
                    table->Positions = count;
                    table->ForceTotal = float(anchored.Force * double(window) / double(count));
                    table->NoiseVariance = float(anchored.Variance * double(window) / double(count));
                    table->NoiseStiffness = float(anchored.Stiffness * double(window) / double(count));
                    table->Table.assign(size_t(mode_count) * count, 0.f);
                    SweepModeDrives(field, phi, mode_count, set.Reach, table->Table);
                    // Each row zero-mean over position, so the channel is fluctuation alone.
                    for (uint32_t k = 0; k < mode_count; ++k) {
                        auto *row = &table->Table[size_t(k) * count];
                        double mean = 0;
                        for (uint32_t x = 0; x < count; ++x) mean += row[x];
                        const float mu = float(mean / count);
                        for (uint32_t x = 0; x < count; ++x) row[x] -= mu;
                    }
                    table->ModeStiffness.assign(mode_count, 0.f);
                    const double mean_stiffness = anchored.Stiffness / double(count);
                    for (uint32_t w = 0; w < window; ++w) {
                        const auto *ph = &phi[size_t(w) * mode_count];
                        for (uint32_t k = 0; k < mode_count; ++k) table->ModeStiffness[k] += float(mean_stiffness) * ph[k] * ph[k];
                    }
                    return table;
                });
                side.SweepIndex = index;
                ReadTurnoverNoise(side, surface, index, 1.f);
            }
            if (auto &side = out.Sides[out.SlowSide]; side.SweepIndex < 0 && side.ModelEntity != null_entity) {
                const double pen_bucket = QuarterOctave(double(side.StaticPenetration));
                if (pen_bucket > 0) {
                    const uint64_t noise_key = HashParams(
                        0x51c3b7e9a48d2f06ull, double(surface.ContactSprings[uint32_t(out.SpringIndex)].Key), pen_bucket, double(set.Reach)
                    );
                    const auto index = AdoptSweepTable(surface, noise_key, [&] {
                        const auto anchored = AnchoredElementSums(set, pen_bucket, 1, {});
                        const auto window = double(2 * set.Reach + 1), count = double(set.Springs.Count());
                        auto table = std::make_shared<SweepTableSet>();
                        table->NoiseVariance = float(anchored.Variance * window / count);
                        table->NoiseStiffness = float(anchored.Stiffness * window / count);
                        return table;
                    });
                    // The variance is copied out here and the render never reads the table, so the side's SweepIndex stays unset.
                    // The rowless table is built at unit scale, so the side applies its own.
                    ReadTurnoverNoise(side, surface, index, side.SpringScale);
                }
            }
        }
    }
    return out;
}

// The voice one side of a resolved contact renders, empty when the body has no bank slot or no sample points to excite.
std::optional<VoiceSet::Voice> BuildContactVoice(ModalAudio &m, const SustainedContact &c, const ResolvedContact &resolved, uint32_t side) {
    const auto &own_resolved = resolved.Sides[side];
    if (own_resolved.ModelEntity == null_entity) return {};
    const auto slot = FindModalObject(LiveBank(m), own_resolved.ModelEntity);
    if (!slot) return {};

    return VoiceSet::Voice{
        .Id = ContactVoiceId(c, side),
        .Object = *slot,
        .State = {
            .Blend = own_resolved.Blend,
            .N = own_resolved.Normal,
            .SlipDir = own_resolved.SlipDir,
            .SweepDir = own_resolved.SweepDir,
            .NormalForce = c.NormalForce,
            .Friction = c.Friction,
            .SlipSpeed = numeric::Length(c.Slip),
            .SolverFriction = numeric::Dot(c.FrictionForce, (side == 0 ? -1.f : 1.f) * UnitOrZero(c.Slip)),
            .Stiffness = float(own_resolved.Stiffness),
            .StaticPenetration = own_resolved.StaticPenetration,
            .SpotCount = resolved.SpotCount,
            .SpotWeight = resolved.SpotWeight,
            .SpotStiffness = own_resolved.SpotStiffness,
            .CellSpread = resolved.CellSpread,
            .SurfaceCompliance = resolved.SurfaceCompliance,
            .DampingFactor = resolved.DampingFactor,
            .ShearStiffness = resolved.ShearStiffness,
            .FlankStiffness = resolved.FlankStiffness,
            .FlankBins = resolved.FlankBins,
            .FlankBin = resolved.FlankBin,
            .RelaxScale = resolved.RelaxScale,
            .SpringIndex = resolved.SpringIndex,
            .SpringRate = resolved.SpringRate,
            .SlipRate = resolved.SlipRate,
            .SlowSide = resolved.SlowSide,
            .SpringScale = own_resolved.SpringScale,
            .SpringBins = resolved.SpringBins,
            .SpringBinBlend = own_resolved.SpringBinBlend,
            .SpringHalfExtent = own_resolved.SpringHalfExtent,
            .InverseAngularInertia = own_resolved.InverseAngularInertia,
            .SweepIndex = own_resolved.SweepIndex,
            .NoiseRms = own_resolved.NoiseRms,
            .NoiseStiffness = own_resolved.NoiseStiffness,
            .JunctionSpacing = resolved.JunctionSpacing,
            .JunctionTransit = resolved.JunctionTransit,
            .FootprintLength = resolved.FootprintLength,
            // Slots 0 and 1 store microscale finish; slots 2 and 3 store mesoscale relief.
            // Both voices list them in the same order, so the two read one surface at one position.
            .Tracks = {
                resolved.Tracks[0].Finish,
                resolved.Tracks[1].Finish,
                resolved.Tracks[0].Relief,
                resolved.Tracks[1].Relief,
            },
            .Turnover = resolved.Turnover,
        },
    };
}
} // namespace

/***** Core modal interface from audio/SurfaceContact.h *****/

float SurfaceRoughnessOf(const entt::registry &r, entt::entity node) {
    static constexpr ContactSurface DefaultSurface{};
    const auto *surface = node != null_entity && r.valid(node) ? r.try_get<const ContactSurface>(node) : nullptr;
    return (surface ? *surface : DefaultSurface).Roughness;
}

// A compound body's rubber foot and steel shell are separate collider nodes, so a contact on one takes that node's surface.
entt::entity ContactSurfaceNode(const entt::registry &r, entt::entity collider, entt::entity body) {
    return NearestNodeWith(r, collider, body, [&r](entt::entity e) { return r.all_of<ContactSurface>(e); });
}

void RegisterSurfaceContactHandlers(entt::registry &r) {
    RegisterSceneSetupHandler(r, [](entt::registry &r, entt::entity viewport) {
        r.emplace_or_replace<SurfaceSoundControls>(viewport);
    });
    // A surface belongs to a node.
    track<surface_changes::SurfaceEdit>(r).on<ContactSurface>(On::Create | On::Update | On::Destroy);
    // A surface with no normal map of its own inherits its material's, so a material reassignment changes the relief too.
    // A material assignment names a mesh where a surface names a node, so the two are tracked apart.
    track<surface_changes::SurfaceMaterial>(r).on<MeshMaterialAssignment>(On::Create | On::Update);
    // The relief's texel size is measured from the mesh, so an edit to the mesh restates it.
    // Tracked separately from the surface edits above so the derivation can tell which of the two it is answering.
    track<surface_changes::SurfaceGeometry>(r).on<MeshGeometryDirty>(On::Create);
    track<surface_changes::SoundControls>(r).on<SurfaceSoundControls>(On::Create | On::Update);
}

void SurfaceUpdateContacts(entt::registry &r) {
    auto &m = r.ctx().get<ModalAudio>();
    auto &surface = Surface(m);
    for (auto e : reactive<surface_changes::SoundControls>(r)) {
        const auto &controls = r.get<const SurfaceSoundControls>(e);
        surface.SustainLevel.store(controls.SustainLevel, std::memory_order_relaxed);
        surface.AccelNoiseGain.store(controls.AccelNoiseGain, std::memory_order_relaxed);
        surface.Coupling.store(controls.Coupling, std::memory_order_relaxed);
        surface.MuteGeometricDrive.store(controls.MuteGeometricDrive, std::memory_order_relaxed);
        surface.MuteFrictionDrive.store(controls.MuteFrictionDrive, std::memory_order_relaxed);
    }
    // Re-derive the mesoscale relief of any edited surface, so a sustained contact reads it without decoding a texture.
    // Refresh every node using an edited surface or mesh because relief coordinates belong to the mesh.
    // The finish key follows the surface alone.
    auto &surface_edits = reactive<surface_changes::SurfaceEdit>(r);
    auto &material_edits = reactive<surface_changes::SurfaceMaterial>(r);
    auto &mesh_edits = reactive<surface_changes::SurfaceGeometry>(r);
    if (!surface_edits.empty() || !material_edits.empty() || !mesh_edits.empty()) {
        for (const auto node : r.view<const Instance>()) {
            const auto mesh_e = r.get<const Instance>(node).Entity;
            const bool geometry_changed = mesh_edits.contains(mesh_e);
            if (!geometry_changed && !surface_edits.contains(node) && !material_edits.contains(mesh_e)) continue;
            // Intentional registry write outside Apply: derived from the node's surface and its mesh's coordinates.
            UpdateSurfaceRelief(r, node, mesh_e, geometry_changed);
        }
    }
    for (const auto node : surface_edits) {
        if (!r.valid(node)) continue;
        // Intentional registry write outside Apply: a memo derived from the surface.
        if (const auto *s = r.try_get<const ContactSurface>(node)) r.emplace_or_replace<SurfaceFinishKey>(node, FinishTrackKey(*s));
        else r.remove<SurfaceFinishKey>(node);
    }

    const auto *sustained = r.ctx().find<const PhysicsSustainedContacts>();
    if (!sustained || sustained->Step == surface.ContactStep) return;

    const std::span<const SustainedContact> active{sustained->Active};
    surface.ContactStep = sustained->Step;
    BeginSurfaceTrackFrame(surface);
    const auto &controls = SurfaceControls(r);
    const auto max_voices = controls.MaxVoices;
    auto &set = NextVoiceSet(surface);
    for (const auto &c : active) {
        const bool moving = numeric::Length(c.Slip) >= controls.MinSlipSpeed ||
            std::max(numeric::Length(c.Sides.front().SweepVelocity), numeric::Length(c.Sides.back().SweepVelocity)) >= controls.MinSweepSpeed;
        if (!moving && !(Gate.ContactSprings && c.Friction > 0)) continue;
        const std::array nodes{
            ResolveContactNodes(r, c.Sides.front().ColliderEntity, c.Sides.front().Entity),
            ResolveContactNodes(r, c.Sides.back().ColliderEntity, c.Sides.back().Entity),
        };
        // Resolving searches both bodies' sample points and adopts both surfaces' tracks, so a silent pair stops here.
        const std::array<bool, 2> sounding{IsModalSounding(r, nodes[0].Model), IsModalSounding(r, nodes[1].Model)};
        if (!sounding.front() && !sounding.back()) continue;
        // A full set can take neither side, so a contact past the cap is refused before it is resolved.
        if (set.Voices.size() >= max_voices) {
            surface.VoicesRefused += sounding.front() + sounding.back();
            continue;
        }
        // Both voices render one contact, so its elastic constants and both bodies' surfaces resolve once here.
        const auto resolved = ResolveContact(r, m, c, nodes, controls);
        for (uint32_t side = 0; side < c.Sides.size(); ++side) {
            if (!sounding[side]) continue;
            auto voice = BuildContactVoice(m, c, resolved, side);
            if (!voice) continue;
            if (set.Voices.size() >= max_voices) {
                ++surface.VoicesRefused;
                continue;
            }
            set.Voices.push_back(std::move(*voice));
        }
    }
    PublishVoiceSet(surface);
}
