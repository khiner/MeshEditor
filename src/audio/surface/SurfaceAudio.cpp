#include "SurfaceAudio.h"

#include "Reactive.h"
#include "TransformMath.h"
#include "audio/ContactScene.h"
#include "audio/ModalAudio.h"
#include "gltf/SourceTexture.h"
#include "numeric/vec2.h"
#include "physics/PhysicsContact.h"
#include "physics/PhysicsTypes.h"
#include "render/MaterialComponents.h"
#include "viewport/ViewportEvents.h"

#include <entt/entity/registry.hpp>
#include <glm/common.hpp>
#include <glm/geometric.hpp>

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
// Zero and below carry no octave and key as 0.
double QuarterOctave(double x) { return x > 0 ? std::exp2(std::round(std::log2(x) * 4) / 4) : 0.0; }

// Length over which the integrated relief leaks back to zero, mesh-local, so a node's scale does not move it.
// A sampled normal map is only approximately a gradient field, so integrating it drifts, and the leak holds that error out while passing every feature the contact filter resolves.
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
    // TODO: sample along the contact's own path, from its UV on the closest triangle and its direction in that triangle's UV gradient.
    constexpr float Slope = std::numbers::phi_v<float> - 1; // 1/phi, the least well approximated by a ratio of texel counts
    const float dir_x = 1.f / std::sqrt(1 + Slope * Slope), dir_y = Slope * dir_x;
    // A texel step covers different distances across a map whose sides differ, so the path's length comes from where it lands in texture coordinates.
    // Its direction there is what the sampled gradient projects onto.
    const vec2 step_uv{dir_x / float(image->Width), dir_y / float(image->Height)};
    const float step_uv_length = glm::length(step_uv);
    const float step_length = length_per_uv * step_uv_length;
    const vec2 travel = step_uv / step_uv_length;
    const float leak = std::exp(-step_length / ReliefLeakLength);

    std::vector<float> heights(TrackSamples);
    float x = 0, y = 0, height = 0;
    for (uint32_t i = 0; i < TrackSamples; ++i) {
        // A tangent-space normal is the surface gradient, n proportional to (-dh/du, -dh/dv, 1), with X and Y scaled as the core material property scales them.
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

/***** What a contact reads out of the scene *****/

namespace {
// The model's reactive tracking, kept apart from the collision path's.
namespace surface_changes {
struct SurfaceEdit {};
struct SurfaceMaterial {};
struct SurfaceGeometry {};
struct SoundControls {};
} // namespace surface_changes

// Sample spacing of the track synthesized from a surface's roughness parameters, m.
// The surface states its shortest wavelength and the track resolves that, so the sampling follows the band the surface is described over.
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

// Each body of a contact holds its own voice, so the pair's id numbers the two.
uint64_t ContactVoiceId(const SustainedContact &c, uint32_t side) { return c.Id * 2 + side; }

// The track a resolved pool slot holds.
const std::shared_ptr<const RoughnessTrack> &PooledTrack(const SurfaceAudioState &s, int32_t index) { return s.SurfaceTracks[uint32_t(index)].Owned; }

// Each element keeps the tallest crest across the workers' gathers.
void MergeCrests(std::vector<float> &into, std::span<const float> part) {
    if (part.empty()) return;
    if (into.empty()) into.assign(part.size(), -std::numeric_limits<float>::max());
    for (size_t e = 0; e < into.size() && e < part.size(); ++e) into[e] = std::max(into[e], part[e]);
}

SamplePointBlend NearestSamplePoints(const std::vector<vec3> &positions, vec3 local_point) {
    if (positions.size() < 2) return {};
    const auto dist2 = [&](uint32_t i) { const auto d = positions[i] - local_point; return glm::dot(d, d); };
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

// Where an excitation at `local_point` reads an object's mode shapes: the nearest point of the sample surface, in barycentric weights over the triangle holding it, so the field is continuous as the contact travels.
// A model with no sample surface blends its two nearest sample points.
SamplePointBlend ShapeBlendAt(const ModalModes &modes, vec3 local_point) {
    if (modes.Indices.size() < 3) return NearestSamplePoints(modes.Positions, local_point);

    SamplePointBlend best;
    float best_distance2 = std::numeric_limits<float>::max();
    for (size_t i = 0; i + 2 < modes.Indices.size(); i += 3) {
        const std::array tri{modes.Indices[i], modes.Indices[i + 1], modes.Indices[i + 2]};
        const auto hit = ClosestPointOnTriangle(local_point, modes.Positions[tri[0]], modes.Positions[tri[1]], modes.Positions[tri[2]]);
        const vec3 offset = hit.Position - local_point;
        if (const float distance2 = glm::dot(offset, offset); distance2 < best_distance2) {
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
    const float step = glm::length(side.SweepVelocity) / sample_rate;

    SideTracks out;
    // `size` converts the track's stored lengths to meters: 1 for the finish, whose lengths are already absolute, and the node's world scale for the relief, whose lengths are in the mesh's own coordinates.
    const auto make_track = [&](int32_t index, float sigma, float size) {
        const auto &track = *PooledTrack(surface, index);
        const float spacing = track.Spacing * size;
        const float rate = step / spacing;
        // The filter width each track is read over belongs to the contact, and is filled in once both sides have resolved, since their surface gradient sets it.
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
        // A measured profile has its own height scale, and a synthesized track is unit height, scaled here by the surface's roughness.
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
    // A measured band's own heights, unit root-mean-square, and the distance between them in world lengths.
    // The four numbers above are read off those heights rather than authored, and the field holds the heights themselves.
    // A relief map's lengths are its mesh's, so its spacing is the node's scale away from the track's.
    std::shared_ptr<const RoughnessTrack> Profile;
    float Spacing{0};
    uint64_t Key{0}; // Content key of that trace, which its numbers do not identify.
    uint32_t Side{0}; // Which surface carries this band, so its crests can be gathered apart.
};

// Fixed slots per side: [side * 2] the finish, [side * 2 + 1] the mesoscale, so the key and the quarter-track shift both read the side from the slot.
// A zero sigma is absent.
using FinishSpecs = std::array<FinishSpec, 4>;

// The roughness standing inside every bearing contact, the band below each finish's stated cutoff.
// A self-affine finish keeps a share of its variance beyond any wavenumber (Pastewka et al. 2013 Eqs. B2 and B19), and that share continued past the cutoff is the level a contact bears on.
// Waviness holds none of it, its band stopping where the finish's begins.
SubCutoffRoughness CompositeSubCutoff(const FinishSpecs &specs, float spacing) {
    std::array<std::pair<double, double>, 2> bands{}; // Height coefficient and exponent.
    uint32_t band_count = 0;
    double read_width = 0;
    for (uint32_t i = 0; i < specs.size(); i += 2) {
        const auto &spec = specs[i];
        // The synthesis cuts at the sampling where a finish states a finer band than the field can carry, so the resolved band ends there and this one starts.
        const double cutoff = std::max(double(spec.Cutoff), 2 * double(spacing));
        const auto band = UnresolvedBand(spec.Sigma, spec.Correlation, cutoff, spec.Slope);
        if (band.Amplitude <= 0) continue;
        bands[band_count++] = {band.Amplitude, band.Hurst};
        read_width = read_width > 0 ? std::min(read_width, band.MaxWidth) : band.MaxWidth;
    }
    // Two finishes bear inside one contact, so their bands add in variance there.
    // Bands of different exponents do not compose into one power law, so the pair takes the exponent they weight at the width they are read at.
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
        // The node holding the model this side excites, whose frame everything below is in.
        // Null when the side has nothing to excite.
        entt::entity ModelEntity{entt::null};
        SamplePointBlend Blend{};
        vec3 Normal{0}; // Unit contact normal, directed into this body.
        vec3 SlipDir{0}; // Unit slip direction, which the frictional force acts along.
        // Direction each surface's geometric force drives this body, one per contact side, signed so the two bodies oppose.
        std::array<vec3, 2> SweepDir{};
        // The force law this side's voice renders: the pair's, unless the quadratised exchange's stiffness cap softened it against this body's contact-point inertia.
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
        // Half the footprint's extent along the sweep, m, and the body's rigid response to the bins' moment about its pitch axis, kg^-1 m^-2.
        // Zero inertia leaves the face flat.
        float SpringHalfExtent{0};
        float InverseAngularInertia{0};
    };
    std::array<Side, 2> Sides;
    std::array<SideTracks, 2> Tracks; // Each body's finish and relief, adopted once for the pair.
    ContactTrack Turnover{}; // The pair's asperity site population under the quadratised exchange.
    double Stiffness{0}; // k, N/m^(3/2). Zero for a contact bearing on a bed of asperities.
    float StaticPenetration{0}; // The approach the contact carries its load at, m. Signed for a bed.
    float ConformalLength{0}; // u0 = h_rms/Alpha, m. Positive for a contact two faces fix the area of.
    float DampingFactor{0}; // Hunt-Crossley dissipation per unit approach speed, dimensionless.
    uint32_t SpotCount{0}; // Cells of the asperity bed rendered along the track. Positive selects the bed.
    float SpotWeight{0}; // Asperities one cell stands for.
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

// The quadratised exchange resolves a free collision against its stiffest cell over at least this many samples, van Walstijn et al. 2024's empirically sufficient minimum.
constexpr double MinContactSamples{6};
// The expected maximum closing rate the stiffness cap is sized at, their simulation value, m/s.
constexpr double MaxImpactSpeed{0.5};

ResolvedContact ResolveContact(const entt::registry &r, ModalAudio &m, const SustainedContact &c, const std::array<ContactNodes, 2> &nodes, const SurfaceSoundControls &controls) {
    auto &surface = Surface(m);
    ResolvedContact out;
    std::array<double, 2> curvature{};
    for (uint32_t i = 0; i < c.Sides.size(); ++i) {
        auto &side = out.Sides[i];
        // Each body's curvature is read on the surface it touches, at the point it touches, as Hertz theory asks for.
        // It belongs to the pair's stiffness, so a side with nothing to excite still contributes it.
        // A body with no mesh reads as flat.
        curvature[i] = SurfaceCurvature(r, nodes[i].Geometry, c.Point).value_or(0.0);
        // The blend indexes sample points, so a model whose shapes and positions disagree, or a body already gone, leaves this side unresolved.
        const auto *modes = r.valid(nodes[i].Model) ? r.try_get<const ModalModes>(nodes[i].Model) : nullptr;
        const auto *transform = modes ? r.try_get<const WorldTransform>(nodes[i].Model) : nullptr;
        if (!transform || modes->Positions.empty() || modes->Shapes.size() != modes->Positions.size()) continue;

        // Physics reports the contact in world space oriented toward the second body.
        // Mode shapes are defined in the frame of the node holding the model, which everything below moves into.
        side.ModelEntity = nodes[i].Model;
        const auto &wt = *transform;
        const vec3 local_point = InverseTransformPoint(wt, c.Point);
        // The first body is pressed and dragged the other way.
        const float toward = i == 0 ? -1.f : 1.f;
        side.Normal = InverseTransformDir(wt, toward * c.Normal);
        // A contact below the slip threshold takes a deterministic resting tangent basis rather than the slip noise direction, so the interface junction holds pressed modes on a stable axis (CONTACT_SPRINGS).
        // The junction's second axis is the render's normal cross tangent.
        vec3 slip_world = UnitOrZero(c.Slip);
        if (Gate.ContactSprings && glm::length(c.Slip) < controls.MinSlipSpeed) {
            const vec3 axis = std::abs(c.Normal.x) < 0.5f ? vec3{1, 0, 0} : vec3{0, 1, 0};
            slip_world = glm::normalize(glm::cross(c.Normal, axis));
        }
        side.SlipDir = InverseTransformDir(wt, toward * slip_world);
        // Both surfaces' sweep directions land in this body's frame, since its mode shapes are what they project onto.
        // A body is driven along the direction it travels over its own surface and against the one it travels over the other's, so one tangential force drives the two bodies of a contact apart.
        for (uint32_t j = 0; j < c.Sides.size(); ++j) {
            const float own = j == i ? 1.f : -1.f;
            side.SweepDir[j] = InverseTransformDir(wt, own * UnitOrZero(c.Sides[j].SweepVelocity));
        }
        side.Blend = ShapeBlendAt(*modes, local_point);
    }
    // 1/E* and kappa1 + kappa2 are both sums over the two bodies, so the stiffness and patch belong to the contact.
    const double inv_modulus = InvEffectiveModulus(MaterialOf(r, nodes[0].Surface, nodes[0].Model), MaterialOf(r, nodes[1].Surface, nodes[1].Model));
    // Hunt and Crossley relate dissipation to restitution by c_d = (3/2)*(1 - e)/v_i.
    // The factor is dimensionless: the kernel reads it as the constant at a 1 m/s approach, and under ETA_LANDING divides it by each engagement's own landing speed (van Walstijn et al. 2024, Eqs 38-39).
    out.DampingFactor = 1.5f * std::max(1.f - c.Restitution, 0.f) * controls.ContactDamping;
    const float sample_rate = LiveBank(m).SampleRate;
    for (uint32_t i = 0; i < c.Sides.size(); ++i) out.Tracks[i] = ResolveSideTracks(r, m, c.Sides[i], nodes[i].Surface, sample_rate);

    // The surface gradient a track presents, per unit length along the surface, from its height scale and spacing.
    const auto slope_rms = [&surface](const ContactTrack &t) {
        if (t.Index < 0 || t.Spacing <= 0) return 0.f;
        return t.Sigma * PooledTrack(surface, t.Index)->SlopeRms / t.Spacing;
    };
    // Every contact bears on the asperities standing proud inside the region it is confined to, whether the load grew that region or the geometry fixed it, so one force law covers both and only the region differs.
    static constexpr ContactSurface DefaultSurface{};
    const auto *surface_a = r.try_get<const ContactSurface>(nodes[0].Surface);
    const auto *surface_b = r.try_get<const ContactSurface>(nodes[1].Surface);
    const auto &sa = surface_a ? *surface_a : DefaultSurface;
    const auto &sb = surface_b ? *surface_b : DefaultSurface;
    // Whichever surface is rougher dominates the composite spectrum, so its slope sets the Hurst exponent.
    const double hurst = HurstExponent(sa.Roughness >= sb.Roughness ? sa.SpectralSlope : sb.SpectralSlope);
    // The gap between two rough surfaces holds both their heights, so their variances add.
    const double roughness = std::sqrt(double(sa.Roughness) * sa.Roughness + double(sb.Roughness) * sb.Roughness);
    const auto coeffs = PerssonSeparationCoefficients(hurst, RoughnessBandRatio);
    out.ConformalLength = coeffs.Alpha > 0 ? float(roughness / coeffs.Alpha) : 0.f;
    // The patch shear spring the frictional force acts through: the interfacial stiffness F/u0 at u0 = 0.4 h_rms (Pastewka et al. 2013) scaled to shear by Mindlin's ratio.
    // A smooth pair reads zero, which binds the junction to the contact point rigidly.
    out.ShearStiffness = roughness > 0 ? float(MindlinShearRatio * c.NormalForce / (0.4 * roughness)) : 0.f;
    const double combined = CombinedCurvature(curvature.front(), curvature.back());
    // The smooth Hertz spring a contact falls back to where its surfaces carry no roughness at all.
    out.Stiffness = ContactStiffness(inv_modulus, combined);
    out.StaticPenetration = float(StaticPenetration(c.NormalForce, out.Stiffness));
    // The width of the contact filter along the surface belongs to the pair, so the tracks are the same whichever voice reads them.
    // The contact is confined to the polygon its geometry shares, and inside that to the patch its load presses out, which roughness spreads wider than Hertz predicts for smooth surfaces.
    const double hertz_radius = ContactPatchRadius(c.NormalForce, inv_modulus, combined);
    const double rough_radius = ConformalPatchWidth(combined, roughness, coeffs);
    const double patch_radius = std::max(hertz_radius, rough_radius);
    const double patch_area = std::numbers::pi * patch_radius * patch_radius;
    const double bound_area = c.NominalArea > 0 ? std::min(double(c.NominalArea), patch_area) : patch_area;
    // One surface's mesoscale gradient.
    // A relief map is a realization of the same departure from flat the waviness parameters describe statistically, so a surface with one is measured from it and the parameters stand in only where it does not.
    const auto mesoscale_slope = [](const ContactSurface &s, double relief_gradient) {
        if (relief_gradient > 0) return relief_gradient;
        return s.Waviness > 0 && s.WavinessLength > 0 ? double(s.Waviness) / s.WavinessLength : 0.0;
    };
    // Every track is read over the region confining the contact, as the circle of equal area.
    // The bearing asperities stand scattered across that whole region, so a datum read narrower than it would let the contact descend into a single valley the real face would bridge (Klein and Hamet 2004, Andersson and Kropp 2008).
    // Geometry below the contact's extent belongs in the force law rather than in the datum.
    const auto equal_area_width = [](double area) { return 2.f * std::sqrt(float(area) / std::numbers::pi_v<float>); };
    // How many ribbons of `strip_width` a contact's bearing area holds side by side.
    // The elements tile one ribbon running the footprint's length, so a contact holds this many of them and the load each one bears is the contact's own divided by the count.
    const auto strip_count = [](double area, double length, double strip_width) {
        return strip_width > 0 && length > 0 ? std::max(area / (length * strip_width), 1.0) : 1.0;
    };
    // The length the footprint walks along the slide.
    // Two faces meeting clip a polygon whose extent along the track the solver measures, which is what the walk spans.
    // A point or an edge clips no polygon and keeps the circle.
    const double footprint_length = c.NominalExtent > 0 ? double(c.NominalExtent) : double(equal_area_width(bound_area));
    // The asperities a contact can bear on stand across the region it is confined to, and which of them bear load turns over as it slides, which is the whole of the fluctuation.
    // Waviness holds most of a nominally flat face apart, so the same area law applied at the waviness scale gives the region left over, which is Persson's magnification applied one scale at a time.
    // Both surfaces' mesoscale stands in the one gap, so their gradients add in quadrature as their finishes do.
    const double meso_a = mesoscale_slope(sa, slope_rms(out.Tracks.front().Relief));
    const double meso_b = mesoscale_slope(sb, slope_rms(out.Tracks.back().Relief));
    const double waviness_gradient = std::sqrt(meso_a * meso_a + meso_b * meso_b);
    const double eligible_area = waviness_gradient > 0 ? RealContactArea(c.NormalForce, bound_area, inv_modulus, waviness_gradient) : bound_area;
    const float eligible_width = equal_area_width(eligible_area);
    // The region is at least as wide as the manifold's own extent along the slide.
    // A face bearing on a thin strip bears load across the strip's whole length while the shared polygon's area collapses, and the datum follows the envelope of that whole length.
    const float region_width = std::max({equal_area_width(bound_area), eligible_width, c.NominalExtent});
    // A box this wide first nulls at one over its width, so covering two samples' travel puts that null on Nyquist and keeps the track from folding back on itself.
    // At most it covers the whole track.
    const auto size_window = [&surface](ContactTrack &t, float width) {
        if (t.Index < 0 || t.Spacing <= 0) return;
        const auto samples = float(PooledTrack(surface, t.Index)->Heights.size());
        t.Window = std::min(std::max(width / t.Spacing, 2 * t.Rate), samples);
    };
    for (auto &tracks : out.Tracks) {
        size_window(tracks.Finish, region_width);
        size_window(tracks.Relief, region_width);
    }

    // Two faces bear on the asperities standing proud across the area they share, so the contact is a bed of Hertz spots.
    // Persson's exponential law is the mean of that bed and holds where the patch spans many roll-off lengths, which ours does not (Pastewka et al. 2013).
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
        // Rice's formula gives how densely peaks stand along the profile, and the patch holds that many.
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
        // The width a cell's datum is measured over, the rule being BedCellWindow's: the contact's bearing share of the sweep, floored at the pair's half-variance width and capped at the stretch the cell's own asperity population spans.
        // Below the floor the window mean follows the asperity population itself rather than the envelope, and that population belongs to the force law's spread and sites.
        // The floor is the bearing share alone, never the region, which is the reachable stretch the cells sample for turnover and would flatten a datum averaged over it.
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
        // The turnover shot process's geometry: junctions stand a Rice peak spacing apart along the sweep, and one bears while the paired tips overlap.
        // That overlap is the Hertz patch diameter 2 sqrt(R delta), at the bed's asperity tip radius and the mean compression of the summits bearing at its separation, the Gaussian tail's inverse Mills residual.
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
        // The element/spring container (CONTACT_SPRINGS): the pair's composite finish tiled into elements of a quarter correlation length, each carrying the non-linear spring its crest-referenced bearing curve implies (Andersson and Kropp 2008, section 2.6).
        // The set is keyed by the pair's spectra and the contact's curvature bucket, so the contacts of one pair class share one build, and the bed's sites, envelope swap, and recruitment stay idle.
        // Under SLIDING_REGISTRATION each side's crests are gathered apart and the slower one is read back at the slip it stands behind, at the cost of a second field per strip.
        if (Gate.ContactSprings) {
            FinishSpecs specs{};
            uint32_t rough_count = 0;
            float spacing = std::numeric_limits<float>::max();
            for (uint32_t side = 0; side < 2; ++side) {
                const auto &s = side ? sb : sa;
                const auto &finish = out.Tracks[side].Finish;
                // A measured finish supplies the field its own heights and the band it holds them over, where a parametric one supplies the parameters a field is drawn from.
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
                // The mesoscale joins the crests as its own band, so bearing concentrates on its crests inside the window and the face-wide envelope bridges the valleys (Klein and Hamet 2004).
                // A side with a relief map takes its mesoscale from the map's heights, and the waviness parameters stand in only where it does not.
                // The mesoscale band stops where the roughness band starts, the separation ISO 4287 makes, without which the mesoscale spectrum runs down over the finish's band and counts it twice.
                const auto &relief = out.Tracks[side].Relief;
                if (relief.Index >= 0) {
                    const auto &track = PooledTrack(surface, relief.Index);
                    // A relief track's lengths are its mesh's, and the contact already sized both by the node's scale, so the band it holds scales with them.
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
            // A measured finish's trace is the field's sweep: the field spans the length the heights were measured over and samples them where they were sampled.
            // The finest-sampled trace takes the axis, a parametric band drawn beside it stops at that sampling, and the unresolved band covers the roughness below.
            // The sweep follows a finish alone, a mesoscale band being drawn over the ribbon rather than sampled into it.
            const RoughnessTrack *sweep = nullptr;
            for (uint32_t side = 0; side < 2; ++side) {
                const auto &spec = specs[side * 2];
                if (spec.Profile && (!sweep || spec.Profile->Spacing < sweep->Spacing)) sweep = spec.Profile.get();
            }
            if (sweep) spacing = sweep->Spacing;
            if (rough_count > 0 && spacing > 0) {
                // The gap parabola quantized to quarter octaves, so one set serves a contact whose curvature wobbles within the bucket.
                // A face's bearing window is its own footprint extent: the springs sweep the track, so which elements a slide engages turns over positionally and the face-wide envelope is the datum the body follows.
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
                // The elements tile a ribbon of the interface: it runs the footprint's length along the sweep, and across the sweep it samples a strip narrow enough to synthesize.
                // The width the contact bears over is that ribbon's area over its length, held as the count of strips the width holds, so how finely the strip is sampled stays a resolution choice.
                // A strip spans four correlation lengths, which holds the surface's height variance to within about a percent (Pastewka et al. 2013 Eq. B3), and a contact bearing over less than one strip samples its own width instead.
                // One strip's load is too small to bear on a statistical population (Pastewka and Robbins 2016 Eq. 5, with the sphere's radius replaced by the asperity's), so a ribbon gathers several independent realizations of the same surface side by side.
                constexpr double strip_correlations = 4.0;
                constexpr uint32_t strip_realizations = 8;
                // The width the polygon holds across the track, its area over the length the footprint walks.
                const double bearing_width = footprint_length > 0 ? bound_area / footprint_length : 0.0;
                const double strip_reach = strip_correlations * min_correlation;
                const double strip_width = bearing_width > 0 ? std::min(bearing_width, strip_reach) : strip_reach;
                const auto rows = SmoothSize(std::max(uint32_t(std::llround(strip_width / double(spacing))), 8u));
                // The load the datum seats at, quantized as the curvature and the footprint are.
                // The height a population of crests bears a load at moves with that load, so a set built for one load does not serve another and the key says so.
                // The elements tile one ribbon, so its load is the contact's divided by the ribbon count, which is the share the anchor solves at.
                const double ribbon_load =
                    double(c.NormalForce) / strip_count(bound_area, footprint_length, double(strip_realizations) * rows * double(spacing));
                const double load_bucket = c.NormalForce > 0 ? QuarterOctave(ribbon_load) : 0.0;
                // The sweep axis spans the distance a slide covers before the surface repeats, which is a length rather than a sample count, so refining the band holds the ground the field covers.
                // A measured finish spans its own trace, so its surface repeats over the length it was measured on.
                const auto columns = sweep ? uint32_t(sweep->Heights.size()) : SmoothSize(std::max(uint32_t(std::llround(double(SurfaceRepeatLength) / double(spacing))), 4 * element_columns));
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
                // The load seats the surface and everything else builds it, so the load joins the set's key last and the load-free prefix keys the built-surface cache.
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
                    // A contact as wide as the field its elements were gathered on gets no excitation from it: the reach covers every element, so the support envelope stands at the same height wherever the contact is.
                    // The excitation comes from the band between the footprint and the array, so the array spans several footprints, eight of them holding seven eighths of its power.
                    // It costs one crest per element rather than a field per footprint, the elements past the field's length repeating the field's with only their long bands differing.
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
                    // The sum of two independent draws from one spectrum is a draw from that spectrum with the variances added, so two bands describing the same surface synthesize once at the combined height.
                    std::array<FinishSpec, 4> drawn{};
                    uint32_t drawn_count = 0;
                    for (const auto &spec : specs) {
                        if (spec.Sigma <= 0) continue;
                        // A measured band is its own realization rather than a draw from a spectrum, so it joins no other and is drawn alone.
                        auto *const same = spec.Profile ? drawn.begin() + drawn_count : std::ranges::find_if(drawn.begin(), drawn.begin() + drawn_count, [&spec](const FinishSpec &d) {
                            return !d.Profile && (!Gate.SlidingRegistration || d.Side == spec.Side) &&
                                d.Correlation == spec.Correlation && d.Slope == spec.Slope && d.Cutoff == spec.Cutoff;
                        });
                        if (same == drawn.begin() + drawn_count) drawn[drawn_count++] = spec;
                        else same->Sigma = float(std::sqrt(double(same->Sigma) * same->Sigma + double(spec.Sigma) * spec.Sigma));
                    }
                    // A band whose corner stands above the strip is the same relief under every strip of one ribbon, which lie side by side within one of its wavelengths, and it varies by a few percent of its height across a single element.
                    // It is therefore drawn once over the whole ribbon on a grid resolving its own cutoff, and read as one lift per element.
                    const double ribbon_width = double(strip_realizations) * rows * spacing;
                    // What each shared band will draw: decided here, since the sigmas the shared bands leave key the strip build below, and drawn after it, since the strips do not carry the footprint's span.
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
                        // Two sides held apart are two surfaces, so each draws its own realization of a spectrum they share.
                        // Merged sides carry one draw at the combined height, which is that same pair of surfaces summed.
                        const uint32_t realization = Gate.SlidingRegistration ? drawn[i].Side : 0;
                        // Only a band the footprint cannot average away is worth drawing over the whole spanned array.
                        // The variation a band gives a face falls as its correlation length over the face's width, so a band many patches finer than the contact covers the field's own length instead.
                        // The threshold is the same count of patches the span itself is sized by, and it is the difference between a field of a few megabytes and one of twenty gigabytes.
                        const double band_length = double(drawn[i].Correlation) * FootprintSpans >= reach_width ? span_length : double(SurfaceRepeatLength);
                        // A measured band is held at the sampling it was measured at, and spans the stretch of its own heights the elements stand over.
                        if (const auto &profile = drawn[i].Profile; profile && drawn[i].Spacing > 0) {
                            const auto span = uint32_t(std::min(double(profile->Heights.size()), std::ceil(band_length / drawn[i].Spacing)));
                            const auto across = SmoothSize(uint32_t(std::llround(wide / double(drawn[i].Spacing))));
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
                                .Columns = SmoothSize(uint32_t(std::llround(band_length / coarse))),
                                .Rows = SmoothSize(uint32_t(std::llround(wide / coarse))),
                                .Realization = realization,
                                .Side = drawn[i].Side,
                            });
                        }
                        drawn[i].Sigma = 0;
                    }
                    // The strips' synthesis, gather, and spring tables are free of the footprint bucket, so a new footprint reuses already built strips.
                    // The key includes the per-strip sigmas the shared bands left, which is what the workers sum.
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
                    // The realizations are independent draws and gathering only appends, so they run at once and merge after.
                    // Each worker holds a field and the patch it sums from, so the count follows a memory budget rather than the core count.
                    const auto build_strips = [&] {
                        constexpr size_t WorkerBudget = 1ull << 30;
                        // A field, the patch it sums from, and the other side's own field where the two are gathered apart.
                        const size_t worker_bytes = (Gate.SlidingRegistration ? 3 : 2) * sizeof(float) * size_t(columns) * rows;
                        const uint32_t affordable = uint32_t(WorkerBudget / worker_bytes);
                        const uint32_t workers = std::min<uint32_t>(
                            {strip_realizations, std::max(1u, std::thread::hardware_concurrency() - 1), std::max(1u, affordable)}
                        );
                        // One worker is the least a ribbon can be built with, so a worker too big for the budget is admitted and reported rather than refused.
                        if (affordable < strip_realizations) {
                            std::fprintf(
                                stderr, "SPRINGBUDGET workers=%u of %u realizations, %.2f GiB each over a %.2f GiB budget, field %ux%u\n",
                                workers, strip_realizations, double(worker_bytes) / double(1ull << 30), double(WorkerBudget) / double(1ull << 30), columns, rows
                            );
                        }
                        // A transform takes the cores the realizations leave, so the two split the available threads rather than multiplying them.
                        SetTransformThreads(std::max(1u, std::max(1u, std::thread::hardware_concurrency() - 1) / workers));
                        // A ribbon summing exactly one synthesized band is that band's own patch scaled in place, so no summed buffer is allocated for it.
                        uint32_t active = 0, only = 0;
                        for (uint32_t i = 0; i < drawn_count; ++i) {
                            if (drawn[i].Sigma > 0) {
                                ++active;
                                only = i;
                            }
                        }
                        const bool direct = !Gate.SlidingRegistration && active == 1 && !drawn[only].Profile;
                        // The curves come from the population law: the strips still synthesize their crests, but marking and gathering drop out and one closed curve family serves every element.
                        // A measured band is its own realization rather than a law, so it keeps the realized summit build.
                        bool ensemble = active > 0;
                        for (uint32_t i = 0; i < drawn_count; ++i) {
                            if (drawn[i].Sigma > 0 && drawn[i].Profile) ensemble = false;
                        }
                        std::vector<ElementSummits> gathered(workers);
                        // The composite's crests under the ensemble build, one vector per worker, merged by the same tallest-of-each rule the summits' crests take.
                        std::vector<std::vector<float>> crest_gather(workers);
                        // Each side's crests, gathered from its own field before the two are composed, and merged across workers the same way.
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
                                    // The other side's own field, kept apart only long enough to gather its crests, then composed into the first.
                                    std::vector<float> apart(Gate.SlidingRegistration ? size_t(columns) * rows : 0, 0.f);
                                    for (uint32_t rep = next++; rep < strip_realizations; rep = next++) {
                                        if (direct) {
                                            auto patch = SynthesizeRoughnessPatch(drawn[only].Correlation, drawn[only].Slope, drawn[only].Cutoff, spacing, columns, rows, rep);
                                            // One zero-started add per sample, so the scaled patch keeps the sum's own signed zeros.
                                            for (float &h : patch.Heights) h = 0.f + drawn[only].Sigma * h;
                                            gather(w, patch.Heights);
                                            continue;
                                        }
                                        std::ranges::fill(heights, 0.f);
                                        std::ranges::fill(apart, 0.f);
                                        for (uint32_t i = 0; i < drawn_count; ++i) {
                                            if (drawn[i].Sigma <= 0) continue;
                                            auto &into = Gate.SlidingRegistration && drawn[i].Side == 1 ? apart : heights;
                                            // Each side held apart continues the realizations past the strips the first side takes, so the two carry independent surfaces of the spectrum they share.
                                            const uint32_t realization = Gate.SlidingRegistration ? rep + drawn[i].Side * strip_realizations : rep;
                                            const auto patch = drawn[i].Profile ? SynthesizeProfileField(drawn[i].Profile->Heights, drawn[i].Spacing, rows, realization) : SynthesizeRoughnessPatch(drawn[i].Correlation, drawn[i].Slope, drawn[i].Cutoff, spacing, columns, rows, realization);
                                            // A trace that is not the one the sweep follows spans its own length at its own sampling, so it is read across this field cyclically.
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
                                        // The long bands lift the crests over the whole spanned array rather than this strip, standing the same under every strip of one ribbon, and lifting an element's crest and its summits together leaves that element's curve untouched.
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
                            // A population with no crest to anchor to keeps the realized build, so a degenerate field falls back rather than going silent.
                            // The two builds carry different bearing statistics, so the fallback is reported.
                            if (built->Springs.Count() == 0) {
                                std::fprintf(stderr, "SPRINGENSEMBLE found no crest population, realized build instead, field %ux%u\n", columns, rows);
                                run_gather(false);
                            }
                        }
                        if (built->Springs.Count() == 0) {
                            // Merged in worker order, so the population is the same however the strips fell to the workers and a build stays deterministic.
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
                    // The array past the field's length repeats the field's elements, so the curves repeat with them and only the crests are laid out over the whole span, each with the long bands read where that element stands.
                    auto span_crests = [&](std::vector<float> &crest, std::span<const float> band) {
                        if (crest.empty()) return;
                        const auto tile = uint32_t(crest.size());
                        std::vector<float> spanned(span_count);
                        for (uint32_t e = 0; e < span_count; ++e) spanned[e] = crest[e % tile] + band[e];
                        crest = std::move(spanned);
                    };
                    // The composite is both surfaces, so it holds both sides' bands.
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
                    // A contact reaching half the cyclic array reaches all of it, so its envelope is a maximum over every element and stands at the same height wherever the contact is.
                    // The height excitation is then dead and only the friction channels sound, so a footprint wider than the field the elements span is reported.
                    if (set->Reach >= half && count > 2) {
                        std::fprintf(
                            stderr, "SPRINGREACH reach=%u of %u elements over a %g m field, footprint %g m, so the datum does not move\n",
                            set->Reach, count, double(count) * double(set->Springs.Width), reach_width
                        );
                    }
                    return set;
                };
                const auto index = AdoptContactSprings(surface, key, [&] {
                    // The ribbon's synthesis, gather, and spring tables carry no load, so a new load bucket seats a copy of an already built surface where one is held.
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
                    // The datum is the excitation the body takes from the surface, so it is reported beside the surface it came from: spread about its mean, and the largest step one element of advance takes.
                    // A footprint averages away everything shorter than itself, so a step near the spread itself means the seat is turning over on a few crests rather than bearing on a population.
                    // What a contact re-registers is the support that side's crests offer over its whole reach rather than one element's crest, taken about its mean, which is where the load seats the contact.
                    // It is then held to the share of the pair's support it explains, the least-squares projection of the composite's support onto it, which is one for a pair with its roughness on a single side and near a half for two like finishes.
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
                    // The array is swept at the faster surface's rate, so the slower one falls behind it by the difference and reads its own crests back there.
                    const float slow_step = std::min(out.Tracks.front().Finish.Step, out.Tracks.back().Finish.Step);
                    out.SlowSide = out.Tracks.front().Finish.Step <= out.Tracks.back().Finish.Step ? 0u : 1u;
                    out.SlipRate = set.Springs.Width <= 0 ? 0.f : (step - slow_step) / set.Springs.Width;
                }
            }
        }
        // The turnover noise track: the bed's asperity site population, white at the pair's asperity spacing, one deterministic realization per contact.
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
        // The bed bears the reported load on average, so the separation it sits at is what the asperity heights and the load leave, negative wherever only the peaks bear.
        out.StaticPenetration = float(bed.Separation * bed.HeightRms);
        // Each cell reads the finish at its own position along the region, over the contact filter the bearing width divided among the cells gives, so a cell takes the local track there while their spread placement keeps the whole contact standing on the region's tallest of them.
        // A cell's read also covers at least its own travel within a sample, so a fast sweep cannot hop the track sample to sample.
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
    // Under the quadratised exchange the contact stiffness is capped so a free collision against the stiffest cell resolves over at least MinContactSamples samples (van Walstijn et al. 2024 Eqs 14, 36-37).
    // An uncapped stiff cell makes one- and two-sample events whose ultrasonic energy an audio-rate render can only fold into the band.
    // The cap reads each body's contact-point inertia from the scheme's step response, so it caps per side, and the bed's anchor is re-solved so the capped law still bears the load exactly.
    {
        const double dt = 1.0 / sample_rate;
        // The contact time of a free mass-barrier collision at the expected maximum closing rate, held to at least MinContactSamples: Eq 37 with Eq 14's zeta in closed form.
        const auto stiffness_cap = [&](double mass, double alpha) {
            const double zeta = std::sqrt(std::numbers::pi) * std::tgamma(1 + 1 / (alpha + 1)) / std::tgamma(0.5 + 1 / (alpha + 1));
            return 0.5 * mass * (alpha + 1) * std::pow(MaxImpactSpeed, 1 - alpha) * std::pow(2 * zeta / (MinContactSamples * dt), alpha + 1);
        };
        const auto &bank = LiveBank(m);
        for (auto &side : out.Sides) {
            if (side.ModelEntity == null_entity) continue;
            const auto slot = FindModalObject(bank, side.ModelEntity);
            if (!slot) continue;
            // The inertia the exchange meets at the contact point: the coupled step's one-sample response, modal and rigid together, inverted back to a mass.
            const auto o = *slot;
            const auto k0 = bank.ModeOffset[o], stride = bank.ModeCount[o], count = bank.TunedModeCount[o];
            const auto shape0 = bank.ShapeOffset[o];
            const auto base0 = shape0 + side.Blend.Points[0] * stride, base1 = shape0 + side.Blend.Points[1] * stride, base2 = shape0 + side.Blend.Points[2] * stride;
            const float w0 = side.Blend.Weights.x, w1 = side.Blend.Weights.y, w2 = side.Blend.Weights.z;
            double modal_response = 0;
            for (uint32_t k = 0; k < count; ++k) {
                const auto shape = BlendedShape(bank, base0, base1, base2, {w0, w1, w2}, k);
                const double normal = glm::dot(shape, side.Normal);
                modal_response += normal * normal * bank.QuadCompliance[k0 + k];
            }
            const double response = modal_response * surface.Coupling.load(std::memory_order_relaxed) * bank.DeflectionScale[o] * surface.SustainLevel.load(std::memory_order_relaxed) / sample_rate + dt * dt * bank.RigidInvMass[o];
            if (response <= 0) continue;
            const double mass = dt * dt / response;
            if (out.SpotCount > 0) {
                // The cell is the event, and its force follows the bed's share of the relative excursion, so its stiffness against that excursion takes the share at the law's own power.
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
    // The springs hold the pair's asperity constant and every side reads the one shared contact force, so the anchor solves at the share of the load one ribbon bears.
    // The stiffness cap above stays a bed concern: spring voices exchange through the implicit quad solve, whose per-channel passivity cap bounds the coupling.
    if (out.SpringIndex >= 0) {
        const auto &set = *surface.ContactSprings[uint32_t(out.SpringIndex)].Owned;
        // The elements tile a ribbon running the footprint's length along the sweep and one synthesized strip across it, so the springs scale by the strips a contact's bearing area holds.
        // A strip away from the sweep line bears at the strip's own gap, which is exact for a face and the equal-area convention for a curved body.
        const double strips = strip_count(bound_area, footprint_length, double(set.StripWidth));
        for (auto &side : out.Sides) {
            side.SpringScale = float(strips);
            side.StaticPenetration = SolveSpringAnchor(set, c.NormalForce / strips);
        }
        // The oblique-flank micro-slip spring the normal channel acts through.
        // Two rough surfaces touch where their composite gap peaks, which is where they lie locally parallel, so the interface there is tilted by one surface's own slope and its variance is half the composite's.
        // A normal approach of amplitude a therefore indents each bearing flank by a cos(theta) and slides it by a sin(theta) against that flank's own Coulomb cone, dissipating with no gross tangential motion.
        // Each flank's Dahl loop area is cubic in its slide, and summing that over the bearing population collapses to one element on the normal channel at cone mu N and this stiffness (Mindlin and Deresiewicz 1953).
        const float anchor = out.Sides.front().StaticPenetration;
        const uint32_t stride = std::max(set.Springs.Count() / 512, 1u);
        const auto [moments, sampled] = SpringFlankSum(set, anchor, stride);
        // Both moments belong to the whole contact, so they carry the ribbon's strip count.
        // The width they divide into does not, being their ratio over the same sampled elements.
        const double modulus = sampled > 0 ? strips * moments.Modulus / sampled : 0;
        const double bearing_stiffness = sampled > 0 ? strips * moments.Stiffness / sampled : 0;
        // The share of an excursion the interface takes, the bulk beneath it having taken the rest.
        // The two carry the same force and so add in series, and this contact's interface is the element stack, whose stiffness the walk above already has.
        if (bearing_stiffness > 0 && bound_area > 0 && inv_modulus > 0) {
            const double bulk = 2 * std::sqrt(bound_area / std::numbers::pi) / inv_modulus;
            out.SurfaceCompliance = float(bulk / (bulk + bearing_stiffness));
        }
        // The relaxation channel's coefficient, (G*/E*)/2 times the contact's second force derivative.
        // A receding contact drops annuli that carry their stored shear away, and d2F/dd2 gives how much contact one step of approach drops (Popov, Popov and Pohrt 2015, their shape factor 1/c = (d2F/dd2)/(2 E*), the moduli cancelling against G*/E*).
        // Taken as a central difference of the same bearing stiffness the flank moment walks.
        const float probe = std::max(anchor * 1e-2f, 1e-9f);
        const auto stiffness_at = [&](float engagement) {
            const auto at = SpringFlankSum(set, engagement, stride);
            return at.Sampled > 0 ? strips * at.Moments.Stiffness / at.Sampled : 0.0;
        };
        // RELAX_DAMPING turns the channel on.
        // The mechanism reads exactly zero wherever the normal and tangential motion share one phase, which every single-mode contact does, and the central difference costs two more footprint passes.
        out.RelaxScale = Gate.RelaxDamping ? float(0.5 * MindlinShearRatio * std::max((stiffness_at(anchor + probe) - stiffness_at(anchor - probe)) / (2 * double(probe)), 0.0)) : 0.f;
        // The scale the tilt is read at, which is the bearing contact's width rather than the finest wavelength the track resolves.
        // Pastewka et al. 2013 Appendix B splits a contact's shape from the roughness inside it at q = pi/r0 for a contact of radius r0 (their Eqs. B6 and B18): components longer than the patch tilt it, components shorter act as roughness within it.
        // The elements supply that width from the contacts they are built from, each answering dF/dd = 2 a E* at the depth its own contact takes, so no asperity radius enters.
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
        // The junction takes the bearing population's spread of breakaway strengths rather than the single strength one element averages them to.
        // The population is resolved from an authored spectrum, so the density Li, Botto, Xu and Gola 2020 and Zhan and Huang 2018 must fit or calibrate is one this contact computes, and each bin takes its own share of the same normal force and flank moment through the same law.
        if (out.FlankStiffness > 0) {
            const auto shares = FlankSpectrumShareAt(set.Springs, anchor);
            out.FlankBins = FlankJunctionSpread(FlankSineCube(flank_slope), c.NormalForce, modulus, shares.Moment, shares.Force, out.FlankBin);
        }
        // The distributed-excitation bins: positions along the sweep axis across the footprint, each side reading its own mode shapes there.
        // A resting pair with no slip has no axis and takes no bins, and a sphere's bins nearly coincide, so its zero-sum channel collapses to the point drive a small footprint is.
        const float half_extent = float(set.Reach) * set.Springs.Width;
        vec3 axis{0};
        float speed = 0;
        for (const auto &cs : c.Sides) {
            if (const float s = glm::length(cs.SweepVelocity); s > speed) {
                speed = s;
                axis = cs.SweepVelocity / s;
            }
        }
        if (speed <= 0) axis = UnitOrZero(c.Slip);
        if (half_extent > 0 && glm::dot(axis, axis) > 0) {
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
                // The body's rigid response to the bins' moment about its centre of mass, taken about the pitch axis the normal and the sweep span.
                // A body the world holds does not rock, so it keeps a flat face, and CONTACT_TILT turns the channel on.
                // Only a contact two faces fix the area of takes it: rotating a curved body takes its gap profile along unchanged, which is rolling rather than rocking, so its footprint sees no tilt and Hess and Soom's flat block does not describe it.
                const auto *dynamics = r.try_get<const ContactDynamics>(side.ModelEntity);
                const auto *motion = r.try_get<const PhysicsMotion>(side.ModelEntity);
                const vec3 pitch = glm::cross(side.Normal, UnitOrZero(InverseTransformDir(*transform, axis)));
                if (Gate.ContactTilt && c.NominalArea > 0 && dynamics && motion && IsAuthoritativeDynamicBody(*motion) && glm::dot(pitch, pitch) > 0) {
                    const vec3 pitch_axis = glm::normalize(pitch);
                    // The inertia is stated at the baked size, and a scaled node's grows as the fifth power, its mass with the volume and its arms with the length squared.
                    const float scale = UniformScaleRatio(r, side.ModelEntity, *modes);
                    const double sizing = motion->InertiaDiagonal ? 1.0 : std::pow(double(scale), 5.0);
                    const double inv = double(glm::dot(pitch_axis, dynamics->InverseInertia * pitch_axis)) / sizing;
                    side.InverseAngularInertia = float(std::max(inv, 0.0));
                    side.SpringHalfExtent = half_extent;
                }
            }
            out.SpringBins = bin_count;
        }
        // The moving-load sweep table, for a side whose footprint is fixed on its surface: the anchored element forces swept through that side's mode shapes, tabulated over the spring position.
        // Mode m at frequency w is driven by the force field's spatial component at wavenumber w over speed, so falling roughness spectra put more power on each resonance as speed rises, the coincidence the bins' footprint-slice boxcar filters out.
        if (glm::dot(axis, axis) > 0 && speed > 0 && half_extent > 0) {
            const auto &bank = LiveBank(m);
            for (uint32_t i = 0; i < c.Sides.size(); ++i) {
                auto &side = out.Sides[i];
                if (side.ModelEntity == null_entity) continue;
                // A side whose own contact point sweeps its surface has no fixed footprint to tabulate over, so only the near-stationary side takes the channel.
                if (glm::length(c.Sides[i].SweepVelocity) > 0.01f * speed) continue;
                const auto *modes = r.try_get<const ModalModes>(side.ModelEntity);
                const auto *transform = r.try_get<const WorldTransform>(side.ModelEntity);
                const auto slot = FindModalObject(bank, side.ModelEntity);
                if (!modes || !transform || !slot) continue;
                // The anchor engagement quantized to quarter octaves, so load wobble reuses a table.
                const double pen_bucket = QuarterOctave(double(side.StaticPenetration));
                if (!(pen_bucket > 0)) continue;
                // The side's spring scale quantized the same way, since an unquantized scale drifting under the stiffness cap re-keys the table at every resolve and each adoption rebases the voice.
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
                            phi[size_t(w) * mode_count + k] = glm::dot(BlendedShape(bank, b0, b1, b2, bl.Weights, k), side.Normal);
                        }
                    }
                    // The anchored element forces, whose stiffness the rows' conformity projections read.
                    std::vector<float> field(count);
                    const auto anchored = AnchoredElementSums(set, pen_bucket, scale_bucket, field);
                    auto table = std::make_shared<SweepTableSet>();
                    table->Modes = mode_count;
                    table->Positions = count;
                    // The mean footprint force of this same field, so the read's load scaling matches what the table was built at whatever load adopted it.
                    table->ForceTotal = float(anchored.Force * double(window) / double(count));
                    // The footprint's share of the slip-turnover variance, scaled like the mean force above and left before the side's spring scale, which enters the variance linearly as a count of independent strips.
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
                    // The mean element stiffness through each mode's squared shape over the footprint, which is what loads the mode's response while the drive sweeps.
                    table->ModeStiffness.assign(mode_count, 0.f);
                    const double mean_stiffness = anchored.Stiffness / double(count);
                    for (uint32_t w = 0; w < window; ++w) {
                        const auto *ph = &phi[size_t(w) * mode_count];
                        for (uint32_t k = 0; k < mode_count; ++k) table->ModeStiffness[k] += float(mean_stiffness) * ph[k] * ph[k];
                    }
                    return table;
                });
                side.SweepIndex = index;
                // The table already carries this side's spring scale, having been built at it.
                ReadTurnoverNoise(side, surface, index, 1.f);
            }
            // A rolling pair, whose contact point sweeps both surfaces, has no fixed footprint to tabulate mode drives over, while its bearing population still refreshes as the footprint advances.
            // The slow side therefore takes the footprint's turnover variance from a rowless table, the same anchored variance sum with no positions, which both the sweep read and the load gate skip.
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
            .SlipSpeed = glm::length(c.Slip),
            // The solver's tangential force on this side along its slip direction, acting on body 1 with each side's SlipDir signed toward the other, so a positive value is the drag the sliding surface exerts.
            .SolverFriction = glm::dot(c.FrictionForce, (side == 0 ? -1.f : 1.f) * UnitOrZero(c.Slip)),
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
            // Slots 0 and 1 carry each side's microscale finish, slots 2 and 3 its mesoscale relief.
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

/***** What the core modal path calls (see audio/SurfaceContact.h) *****/

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
    // The surface is the node's and the coordinates the relief is measured over are its mesh's, so an edit to either restates every node standing on that mesh.
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

    // Contacts that persist drive sustained voices, republished as a whole set so a contact ends by its absence.
    // A step the simulation did not take publishes nothing and the audio thread ages out the set it holds, so a scene whose playhead has stopped falls silent even with bodies resting on one another.
    const auto *sustained = r.ctx().find<const PhysicsSustainedContacts>();
    if (!sustained || sustained->Step == surface.ContactStep) return;

    const std::span<const SustainedContact> active{sustained->Active};
    surface.ContactStep = sustained->Step;
    BeginSurfaceTrackFrame(surface);
    const auto &controls = SurfaceControls(r);
    const auto max_voices = controls.MaxVoices;
    auto &set = NextVoiceSet(surface);
    for (const auto &c : active) {
        // A contact whose surfaces are not moving over one another generates nothing, while a loaded frictional interface still holds and damps the modes pressed against it.
        // Under CONTACT_SPRINGS it therefore keeps a voice for its junction (Akay 2002 Sec. IV).
        const bool moving = glm::length(c.Slip) >= controls.MinSlipSpeed ||
            std::max(glm::length(c.Sides.front().SweepVelocity), glm::length(c.Sides.back().SweepVelocity)) >= controls.MinSweepSpeed;
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
