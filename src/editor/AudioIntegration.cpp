#include "editor/AudioIntegration.h"

#include "Job.h"
#include "Reactive.h"
#include "action/ActionApply.h"
#include "action/Audio.h"
#include "action/Errors.h"
#include "audio/AudioDevice.h"
#include "audio/AudioQueries.h"
#include "audio/AudioSystem.h"
#include "audio/ContactScene.h"
#include "audio/FftAnalysis.h"
#include "audio/ModalAudio.h"
#include "audio/ModalModelFile.h"
#include "audio/ModalSolve.h"
#include "audio/ModalWarmStart.h"
#include "audio/RealImpact.h"
#include "editor/AudioExcitation.h"
#include "editor/AudioJobs.h"
#include "mesh/MeshStore.h"
#include "physics/PhysicsContact.h"
#include "physics/PhysicsTypes.h"
#include "scene/Entity.h"
#include "selection/SelectionBitset.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"

#include <FastFEM/SolveMonitor.h>
#include <atomic>
#include <iostream>
#include <numbers>
#include <string_view>
#include <unordered_set>

namespace fs = std::filesystem;

using std::ranges::iota_view, std::ranges::to;
using std::views::transform;

void AssignVertexSample(
    entt::registry &r, entt::entity e,
    std::span<const uint32_t> mesh_vertices, fs::path path, std::vector<float> &&frames
) {
    if (mesh_vertices.empty() || path.empty()) return;
    auto &vs = r.get_or_emplace<VertexSamples>(e);
    if (auto *playback = r.try_get<SamplePlayback>(e)) playback->Stop();
    r.ctx().get<AudioSamples>().ByPath.try_emplace(path, std::move(frames));

    bool vs_changed = false;
    for (const uint32_t mv : mesh_vertices) {
        auto &assigned = vs.PathByVertex[mv];
        if (assigned == path) continue;
        assigned = path;
        vs_changed = true;
    }
    if (vs_changed) r.patch<VertexSamples>(e);
    if (!r.all_of<SoundVerticesModel>(e)) r.emplace<SoundVerticesModel>(e, SoundVerticesModel::Samples);
}

void RemoveVertexSamples(
    entt::registry &r, entt::entity e,
    std::span<const uint32_t> mesh_vertices
) {
    auto *vs = r.try_get<VertexSamples>(e);
    if (!vs || mesh_vertices.empty()) return;
    if (auto *playback = r.try_get<SamplePlayback>(e)) playback->Stop();
    size_t removed = 0;
    for (const uint32_t mv : mesh_vertices) removed += vs->PathByVertex.erase(mv);
    if (removed) r.patch<VertexSamples>(e);
    if (vs->PathByVertex.empty()) {
        if (r.all_of<ModalModes>(e)) r.remove<VertexSamples>(e);
        else RemoveAudioComponents(r, e);
    }
}

void SetVertexSamples(
    entt::registry &r, entt::entity e,
    std::span<const uint32_t> mesh_vertices, std::span<LoadedSample> samples
) {
    for (size_t i = 0; i < samples.size() && i < mesh_vertices.size(); ++i) {
        AssignVertexSample(r, e, {&mesh_vertices[i], 1}, std::move(samples[i].first), std::move(samples[i].second));
    }
}

namespace {
/***** Modal synthesis bank *****/

float ModalOutGain(const entt::registry &r, entt::entity e, float scale) {
    const auto *gain = r.try_get<const ModalGain>(e);
    return ModalControls(r).ModalLevel * (gain ? gain->Value : 1.f) * std::pow(scale, -2.f);
}

// Rewrite one slot's output level from the object's current gain and size, leaving the resonator coefficients untouched.
void SetModalOutGain(const entt::registry &r, ModalBank &b, uint32_t slot, entt::entity e) {
    const auto &modes = r.get<const ModalModes>(e);
    std::atomic_ref{b.OutGain[slot]}.store(ModalOutGain(r, e, UniformScaleRatio(r, e, modes)), std::memory_order_relaxed);
}

// Returns displaced air volume in cubic metres for recoil-filter corner calculation.
// World scale converts node-local mesh volume, and mass divided by density supplies volume for open meshes.
double DisplacedVolume(const entt::registry &r, entt::entity e, double mass, const AcousticMaterialProperties *props) {
    const auto *bvh = AssetOf<MeshBvh>(r, e);
    const auto *world = r.try_get<const WorldTransform>(e);
    const double world_scale = world ? double(MeanScale(world->S)) : 1.0;
    const double enclosed = bvh && bvh->EnclosedVolume ? *bvh->EnclosedVolume * world_scale * world_scale * world_scale : 0.0;
    if (enclosed > 0) return enclosed;
    return props && props->Density > 0 && mass > 0 ? mass / props->Density : 0.0;
}

// Returns the equivalent-sphere radius used by the recoil filters.
double VolumeEquivalentRadius(double volume) { return std::cbrt(3.0 * volume / (4.0 * std::numbers::pi)); }

// Recompute an object's resonator coefficients and output gain.
// Frequencies scale with the fundamental target and inversely with object size.
// Uniform scaling gives d' = alpha/2 + (d - alpha/2)/scale^2 and T60 = ln(1000)/d'.
// (T60 == 0 is the undamped sentinel and stays 0, muting the mode.)
void RetuneModalObject(const entt::registry &r, ModalBank &b, uint32_t slot, entt::entity e) {
    const auto &modes = r.get<const ModalModes>(e);
    const auto mode_count = modes.Freqs.size();
    if (mode_count == 0) return;

    const float scale = UniformScaleRatio(r, e, modes);
    const auto *tuning = r.try_get<const ModalTuning>(e);
    const float fundamental = tuning ? tuning->FundamentalFreq : 0.f;
    const float freq_ratio = (fundamental > 0 && modes.Freqs.front() > 0 ? fundamental / modes.Freqs.front() : 1.f) / scale;
    const float t60_scale = tuning ? tuning->T60Scale : 1.f;
    std::optional<double> alpha;
    if (const auto *mat = r.try_get<const AcousticMaterial>(e)) alpha = mat->Properties.Alpha;

    const auto *dynamics = r.try_get<const ContactDynamics>(e);
    const auto *motion = r.try_get<const PhysicsMotion>(e);
    const bool sized = motion && IsAuthoritativeDynamicBody(*motion);
    const double mass = dynamics ? dynamics->Mass * (sized ? 1.0 : double(scale) * double(scale) * double(scale)) : 0.0;
    b.RigidInvMass[slot] = mass > 0 ? float(1.0 / mass) : 0.f;
    // The displaced volume behind the body's sustained acceleration noise.
    // Only dynamic rigid bodies transfer recoil to the air model.
    const auto *mat = r.try_get<const AcousticMaterial>(e);
    const double volume = DisplacedVolume(r, e, mass, mat ? &mat->Properties : nullptr);
    const auto recoil = sized && mass > 0 ? RecoilObjectFilter(VolumeEquivalentRadius(volume), volume, b.SampleRate) : RecoilFilter{};
    b.RadiatorB0[slot] = recoil.RadB0;
    b.AirB0[slot] = recoil.AirB0;
    b.AirB1[slot] = recoil.AirB1;
    b.AirB2[slot] = recoil.AirB2;
    b.RecoilA1[slot] = recoil.A1;
    b.RecoilA2[slot] = recoil.A2;

    std::vector<float> freqs(mode_count), t60s(mode_count);
    for (size_t k = 0; k < mode_count; ++k) {
        freqs[k] = modes.Freqs[k] * freq_ratio;
        const float t60 = modes.T60s[k];
        if (t60 <= 0) {
            t60s[k] = 0;
            continue;
        }
        float d = Ln1000 / t60;
        if (alpha) d = float(*alpha / 2) + (d - float(*alpha / 2)) / (scale * scale);
        t60s[k] = t60_scale * Ln1000 / std::max(d, 1e-9f);
    }
    TuneModalObject(b, slot, freqs, t60s, scale);
    std::atomic_ref{b.OutGain[slot]}.store(ModalOutGain(r, e, scale), std::memory_order_relaxed);
}

// Builds a replacement bank from every modal sound object and installs it atomically for the audio thread.
void RebuildModalBank(entt::registry &r) {
    auto &m = r.ctx().get<ModalAudio>();

    ModalBank next;
    next.SampleRate = float(DeviceSampleRate(r));
    for (auto e : r.view<const ModalModes, const SoundVertices>()) {
        const auto &modes = r.get<const ModalModes>(e);
        if (modes.Freqs.empty()) continue;
        const auto slot = AddModalObject(next, e, modes);
        RetuneModalObject(r, next, slot, e);
    }
    InstallModalBank(m, next);
}

} // namespace

/***** Free functions for sound object control *****/

void Stop(entt::registry &r, entt::entity e) {
    if (auto *playback = r.try_get<SamplePlayback>(e)) playback->Stop();
    if (r.all_of<ModalModes>(e)) {
        auto &m = r.ctx().get<ModalAudio>();
        if (auto slot = FindModalObject(LiveBank(m), e)) EnqueueModalEvent(m, {.Kind = ModalEventKind::Silence, .Object = *slot});
    }
}

void SetModel(entt::registry &r, entt::entity e, SoundVerticesModel model) {
    Stop(r, e);

    const bool is_sample = model == SoundVerticesModel::Samples && r.all_of<VertexSamples>(e);
    const bool is_modal = model == SoundVerticesModel::Modal && r.all_of<ModalModes>(e);
    if (!is_sample && !is_modal) return;

    r.emplace_or_replace<SoundVerticesModel>(e, model);
}

namespace {
// Mean curvature (1/m) of the solid sphere with this mass and density. 0 for an immovable body.
double SphereEquivalentCurvature(double density, double inv_mass) { return std::cbrt(4.0 * std::numbers::pi / 3.0 * density * inv_mass); }

// A strike from a physics collision. Its presence marks `force` as the true contact impulse, not a nominal level.
struct PhysicsStrike {
    vec3 Direction; // node-local contact direction
    vec3 Point; // world-space contact point, which the struck body's curvature is read at
    entt::entity GeometryEntity;
    entt::entity SurfaceEntity;
    Impactor Impactor; // striking body's impactor
    float NominalArea; // area the two faces share, m^2, zero where the touch is a point or an edge
    float CombinedRoughness; // the pair's rms asperity heights in quadrature, m
    // Sample point nearest the manifold's load-weighted centre.
    // Compute collision duration from the body's response at the manifold resultant center.
    uint32_t ResultantIndex;
};

void TriggerModalStrike(entt::registry &r, entt::entity e, uint32_t excitable_index, float force, float contact_speed, std::optional<PhysicsStrike> physics = std::nullopt) {
    auto &m = r.ctx().get<ModalAudio>();
    const auto &bank = LiveBank(m);
    const auto slot = FindModalObject(bank, e);
    if (!slot) return;

    const auto &modes = r.get<const ModalModes>(e);
    if (excitable_index >= std::min(modes.Vertices.size(), modes.Positions.size())) return;
    const vec3 dir = physics ? numeric::Normalize(physics->Direction) : ExciteDirection(r, e, modes.Vertices[excitable_index]);

    const auto *cd = r.try_get<const ContactDynamics>(e);
    const auto *mat = r.try_get<const AcousticMaterial>(e);
    // A short default contact with no click applies when the material or contact dynamics are missing.
    double tau = 1e-4; // seconds
    float click_amp = 0;
    ClickFilter click{};
    if (cd && mat) {
        Impactor imp;
        if (physics) {
            imp = physics->Impactor;
        } else {
            const auto *device = r.ctx().find<AudioDeviceResource>();
            const auto *striker_ptr = device ? r.try_get<const Striker>(device->Viewport) : nullptr;
            imp = StrikerImpactor(striker_ptr ? *striker_ptr : Striker{});
        }
        const vec3 strike_point = physics ? physics->Point : TransformPoint(r.get<const WorldTransform>(e), modes.Positions[excitable_index]);
        const double curvature = SurfaceCurvature(r, physics ? physics->GeometryEntity : e, strike_point).value_or(0.0);
        // The elastic constants belong to the surface that was struck, as they do for a contact that persists.
        // Use the target node surface for external mallet impacts.
        const auto &elastic = MaterialOf(r, physics ? physics->SurfaceEntity : e, e);
        // Use rounded-tip point-contact geometry at the target position.
        const float scale_ratio = UniformScaleRatio(r, e, modes);
        // A mallet's tip is polished, so a manual strike reads the struck surface's finish alone.
        const double roughness = physics ? physics->CombinedRoughness : SurfaceRoughnessOf(r, e);
        tau = EstimateContactTime(*cd, physics ? physics->ResultantIndex : excitable_index, dir, contact_speed, elastic, curvature, physics ? physics->NominalArea : 0.f, imp, scale_ratio, roughness);
        // The click is the recoil radiator driven by this strike's force pulse, with the body's inertia in the loop (see RecoilClickFilter).
        // Use the sample-surface disc radius when displaced volume is zero.
        const double volume = DisplacedVolume(r, e, cd->Mass, &mat->Properties);
        const double radius = volume > 0 ? VolumeEquivalentRadius(volume) : double(bank.RadiantRadius[*slot] * scale_ratio);
        click = RecoilClickFilter(radius, volume, cd->Mass, bank.SampleRate);
        const double impulse = physics ? double(force) : ReducedContactMass(*cd, excitable_index, dir, imp) * std::abs(double(contact_speed));
        click_amp = float(impulse * bank.SampleRate);
    }
    const auto step = float(1.0 / (tau * bank.SampleRate));
    EnqueueModalEvent(
        m,
        {
            .Kind = ModalEventKind::Impact,
            .Object = *slot,
            .ExPos = excitable_index,
            .Jx = dir.x * force,
            .Jy = dir.y * force,
            .Jz = dir.z * force,
            .PulseStep = step,
            .PulseGamma = 2 * step,
            .AccelAmp = click_amp,
            .ClickB0 = click.B0,
            .ClickA1 = click.A1,
            .ClickA2 = click.A2,
        }
    );
}

// Survives the frame-end clear, so the audio handler picks up world-transform changes made after it already ran, on the following frame.
struct ModalScaleTracker {
    entt::storage_for_t<entt::reactive> Storage;
    void Bind(entt::registry &r) {
        Storage.bind(r);
        Storage.on_update<WorldTransform>();
    }
};

namespace audio_changes {
struct VertexForce {};
struct ModalModes {};
struct ModalGain {};
struct ModalTuning {};
struct ModalSoundControls {};
struct RecordingStart {};
struct SoundVerticesDerivation {};
struct ContactReportingDerivation {};
struct ContactDynamicsDerivation {};
struct ModelRescaleEdit {};
struct AudioConfig {};
struct AudioMix {};
} // namespace audio_changes

} // namespace

bool IsSolving(const entt::registry &r, entt::entity e) {
    return std::ranges::any_of(r.ctx().get<const ModalSolveJobs>().Jobs, [e](const auto &job) { return job->Entity == e; });
}

namespace {

// The cancelled job thread exits at its next checkpoint, and its result is discarded on arrival.
void CancelModalSolves(entt::registry &r, entt::entity e) {
    for (auto &job : r.ctx().get<ModalSolveJobs>().Jobs) {
        if (job->Entity == e) job->Work.RequestCancel();
    }
}

/***** Modal model derivation *****/

// The material properties a model's modes derive at, folding the spec's one-mass rule into the density.
// A dynamic rigid body's authored mass is the body's one mass, so the modes derive at the density that makes the solve's mass meet it.
// Scaling stiffness with mass preserves specific stiffness E/rho and modal frequencies.
// Compare masses at the baked model size recorded by the solve.
AcousticMaterialProperties EffectiveModalMaterial(AcousticMaterialProperties props, const ModalEigenSummary &summary, double solve_mass, const PhysicsMotion *motion) {
    if (!motion || !IsAuthoritativeDynamicBody(*motion) || solve_mass <= 0 || summary.SolvedMaterial.Density <= 0 || props.Density <= 0) return props;
    const double rho_eff = summary.SolvedMaterial.Density * double(motion->Mass.value_or(DefaultMass)) / solve_mass;
    props.YoungModulus *= rho_eff / props.Density;
    props.Density = rho_eff;
    return props;
}

AcousticMaterialProperties EffectiveModalMaterial(const entt::registry &r, entt::entity e, const ModalEigenSummary &summary) {
    const auto *mat = r.try_get<const AcousticMaterial>(e);
    const auto *mp = r.try_get<const MassProperties>(e);
    return EffectiveModalMaterial(mat ? mat->Properties : summary.SolvedMaterial, summary, mp ? mp->Mass : 0.0, r.try_get<const PhysicsMotion>(e));
}

// Re-derives modes with modal::RescaleModes while preserving a fundamental frequency pinned at solve time.
// Returns empty when the material edit changes Poisson ratio.
std::optional<ModalModes> RescaledModes(const ModalEigenSummary &summary, const ModalModes &modes, const AcousticMaterialProperties &props, const ModalSolveSettings &settings) {
    std::optional<float> fundamental;
    if (!modes.Freqs.empty() && modes.OriginalFundamentalFreq > 0 && modes.Freqs.front() != modes.OriginalFundamentalFreq) fundamental = modes.Freqs.front();
    auto config = settings.Solve.Modal;
    config.FundamentalFreq = fundamental;
    return modal::RescaleModes(summary, modes, props, config);
}

// A synth tuning still at its default (fundamental == the old model's lowest mode) follows the new model, while a user-set tuning stays pinned.
// Intentional registry writes outside Apply: the model and tuning are derived state.
void ReplaceModalModes(entt::registry &r, entt::entity e, ModalModes new_modes) {
    const auto *tuning = r.try_get<const ModalTuning>(e);
    const auto *old_modes = r.try_get<const ModalModes>(e);
    if (tuning && old_modes && !old_modes->Freqs.empty() && !new_modes.Freqs.empty() &&
        tuning->FundamentalFreq == old_modes->Freqs.front() && tuning->FundamentalFreq != new_modes.Freqs.front()) {
        r.replace<ModalTuning>(e, ModalTuning{new_modes.Freqs.front(), tuning->T60Scale});
    }
    r.emplace_or_replace<ModalModes>(e, std::move(new_modes));
}

// Re-derive the entity's modal model for its effective material, from the current acoustic material and the body's one mass.
// Poisson-ratio changes require a new solve.
void RescaleModalObject(entt::registry &r, entt::entity e) {
    const auto &modes = r.get<const ModalModes>(e);
    const auto &summary = r.get<const ModalEigenSummary>(e);
    const auto *settings = r.try_get<const ModalSolveSettings>(e);
    auto rescaled = RescaledModes(summary, modes, EffectiveModalMaterial(r, e, summary), settings ? *settings : ModalSolveSettings{});
    if (rescaled && *rescaled != modes) ReplaceModalModes(r, e, std::move(*rescaled));
}

/***** Modal solve inputs *****/

void HashCombine(size_t &seed, const auto &...values) {
    const auto combine = [&](const auto &value) {
        seed ^= std::hash<std::remove_cvref_t<decltype(value)>>{}(value) + 0x9e3779b97f4a7c15 + (seed << 6) + (seed >> 2);
    };
    (combine(values), ...);
}

size_t HashOperatorInputs(const std::vector<vec3> &positions, const std::vector<uint32_t> &triangle_indices, const ModalSolveSettings &settings) {
    const auto bytes = [](const auto &v) { return std::string_view{reinterpret_cast<const char *>(v.data()), v.size() * sizeof(v[0])}; };
    const std::hash<std::string_view> hash;
    size_t seed = hash(bytes(positions));
    HashCombine(seed, hash(bytes(triangle_indices)), settings.Discretization);
    if (settings.Discretization == fastfem::Discretization::Tet10) {
        const auto &tet = settings.Solve.Tetrahedralization;
        HashCombine(seed, settings.Solve.SurfaceSimplificationRatio, tet.Refinement);
        if (tet.Refinement == fastfem::TetRefinement::QualityAndResolution) HashCombine(seed, settings.Solve.Resolution);
        for (const auto &hole : tet.Holes) HashCombine(seed, hole.x, hole.y, hole.z);
    } else {
        const auto &finite = settings.Solve.FiniteCell;
        HashCombine(seed, settings.Solve.Resolution);
        HashCombine(seed, finite.CutDepth, finite.FictitiousScale, finite.PaddingCells, finite.GridOffsetCells.x, finite.GridOffsetCells.y, finite.GridOffsetCells.z);
    }
    return seed;
}

size_t HashModalConfig(const fastfem::SolverConfig &config) {
    size_t seed{};
    HashCombine(seed, config.MinModeFreq, config.MaxModeFreq, config.NumModes, config.NumFemModes, config.Tolerance, config.MaxRestarts, config.FundamentalFreq);
    return seed;
}

// Returns existing excitation vertices when copying or unique evenly spaced mesh vertices otherwise.
std::vector<uint32_t> DesiredSolveVertices(const entt::registry &r, entt::entity e, const ModalSolveSettings &settings, uint32_t num_vertices) {
    if (settings.CopySoundVertices && r.all_of<SoundVertices>(e)) {
        const auto vertices = r.ctx().get<const MeshStore>().GetSoundVertices(r.get<const SoundVertices>(e).Vertices);
        return {vertices.begin(), vertices.end()};
    }
    const uint32_t ex_count = std::clamp(settings.NumVertices, 1u, num_vertices);
    return iota_view{0u, ex_count} | transform([&](uint32_t i) { return i * num_vertices / ex_count; }) | to<std::vector<uint32_t>>();
}

// One triangle per distinct triple of sample points, dropping any triple that repeats a point.
// Preserve the first observed winding for consistent surface orientation.
std::vector<uint32_t> UniqueSampleTriangles(std::span<const std::array<uint32_t, 3>> triangles) {
    struct Candidate {
        std::array<uint32_t, 3> Key, Winding;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(triangles.size());
    for (const auto &winding : triangles) {
        if (winding[0] == winding[1] || winding[1] == winding[2] || winding[2] == winding[0]) continue;
        auto key = winding;
        std::ranges::sort(key);
        candidates.emplace_back(key, winding);
    }
    std::ranges::sort(candidates, {}, &Candidate::Key);
    const auto duplicates = std::ranges::unique(candidates, {}, &Candidate::Key);
    candidates.erase(duplicates.begin(), duplicates.end());

    std::vector<uint32_t> out;
    out.reserve(candidates.size() * 3);
    for (const auto &c : candidates) out.insert(out.end(), c.Winding.begin(), c.Winding.end());
    return out;
}

// Returns mesh triangles collapsed onto nearest excitation vertices by edge distance.
// A mesh triangle contributes when its corners map to three distinct excitation vertices.
// Empty when the excitation vertices are too few or too clustered to span the surface.
std::vector<uint32_t> SampleSurfaceTriangles(std::span<const uint32_t> triangle_indices, uint32_t vertex_count, std::span<const uint32_t> excitation_vertices) {
    if (excitation_vertices.size() < 3 || triangle_indices.size() < 3) return {};

    // Vertex adjacency over the triangles' edges, in compressed rows. Each corner of a triangle neighbours the other two.
    std::vector<uint32_t> row_start(vertex_count + 1, 0);
    for (const auto v : triangle_indices) row_start[v + 1] += 2;
    for (uint32_t v = 0; v < vertex_count; ++v) row_start[v + 1] += row_start[v];
    std::vector<uint32_t> neighbors(row_start.back());
    auto fill = row_start;
    for (size_t i = 0; i + 2 < triangle_indices.size(); i += 3) {
        const std::array tri{triangle_indices[i], triangle_indices[i + 1], triangle_indices[i + 2]};
        for (uint32_t k = 0; k < 3; ++k) {
            neighbors[fill[tri[k]]++] = tri[(k + 1) % 3];
            neighbors[fill[tri[k]]++] = tri[(k + 2) % 3];
        }
    }

    static constexpr uint32_t Unlabelled{~0u};
    std::vector<uint32_t> label(vertex_count, Unlabelled), queue;
    queue.reserve(vertex_count);
    for (uint32_t s = 0; s < excitation_vertices.size(); ++s) {
        if (const auto v = excitation_vertices[s]; v < vertex_count && label[v] == Unlabelled) {
            label[v] = s;
            queue.push_back(v);
        }
    }
    for (size_t head = 0; head < queue.size(); ++head) {
        const auto v = queue[head];
        for (auto i = row_start[v]; i < row_start[v + 1]; ++i) {
            if (const auto n = neighbors[i]; label[n] == Unlabelled) {
                label[n] = label[v];
                queue.push_back(n);
            }
        }
    }

    std::vector<std::array<uint32_t, 3>> collapsed;
    for (size_t i = 0; i + 2 < triangle_indices.size(); i += 3) {
        const std::array winding{label[triangle_indices[i]], label[triangle_indices[i + 1]], label[triangle_indices[i + 2]]};
        // Skip shell components without an excitation vertex.
        if (std::ranges::contains(winding, Unlabelled)) continue;
        collapsed.push_back(winding);
    }
    return UniqueSampleTriangles(collapsed);
}

// The excitation vertices a solve's sample points came from, one per sample point, in sample point order.
// Vertices whose positions reached the same tet point share a sample point, and the first of them supplies it.
std::vector<uint32_t> CompactExcitationVertices(std::span<const uint32_t> vertices, std::span<const uint32_t> sample_point_of) {
    std::vector<uint32_t> out;
    out.reserve(sample_point_of.empty() ? 0 : sample_point_of.back() + 1);
    for (uint32_t i = 0; i < vertices.size() && i < sample_point_of.size(); ++i) {
        if (sample_point_of[i] == out.size()) out.push_back(vertices[i]);
    }
    return out;
}

std::vector<uint32_t> RelabelSampleTriangles(std::span<const uint32_t> triangles, std::span<const uint32_t> sample_point_of) {
    if (sample_point_of.empty()) return {};
    std::vector<std::array<uint32_t, 3>> relabelled;
    relabelled.reserve(triangles.size() / 3);
    for (size_t i = 0; i + 2 < triangles.size(); i += 3) {
        relabelled.push_back({sample_point_of[triangles[i]], sample_point_of[triangles[i + 1]], sample_point_of[triangles[i + 2]]});
    }
    return UniqueSampleTriangles(relabelled);
}

} // namespace

SolveInputs BuildSolveInputs(const entt::registry &r, entt::entity e, entt::entity mesh_entity, const ModalSolveSettings &settings) {
    const auto &mesh = GetMesh(r, mesh_entity);
    const uint32_t num_vertices = mesh.VertexCount();
    const vec3 node_scale = r.get<const WorldTransform>(e).S;
    std::vector<vec3> positions(num_vertices);
    for (uint32_t i = 0; i < num_vertices; ++i) positions[i] = mesh.GetPosition(Mesh::VH{i}) * node_scale;
    auto triangle_indices = mesh.CreateTriangleIndices();
    const auto operator_hash = HashOperatorInputs(positions, triangle_indices, settings);
    return {
        std::move(positions), std::move(triangle_indices), DesiredSolveVertices(r, e, settings, num_vertices), settings.Solve,
        settings.Discretization, node_scale, operator_hash, HashModalConfig(settings.Solve.Modal)
    };
}

// True when the baked model no longer matches the current solve inputs.
bool ModalModelStale(const entt::registry &r, entt::entity e, const SolveInputs &inputs, const AcousticMaterial &material) {
    const auto *summary = r.try_get<const ModalEigenSummary>(e);
    if (!summary) return true;
    if (summary->OperatorHash != inputs.OperatorHash || summary->ModalConfigHash != inputs.ModalConfigHash) return true;
    // Compare against the prior solve request because coincident tet points produce one sample point.
    if (inputs.Vertices != summary->SolvedVertices) return true;
    return material.Properties.PoissonRatio != summary->SolvedMaterial.PoissonRatio;
}

// Launch an async solve unless one is already running or the baked model matches the inputs.
void LaunchModalSolve(entt::registry &r, entt::entity viewport, entt::entity e, const ModalSolveSettings &settings, const AcousticMaterial &material) {
    if (!r.valid(e) || IsSolving(r, e)) return;
    const auto *inst = r.try_get<const Instance>(e);
    if (!inst || !TryGetMesh(r, inst->Entity)) return;
    auto inputs = BuildSolveInputs(r, e, inst->Entity, settings);
    if (r.all_of<ModalModes>(e) && !ModalModelStale(r, e, inputs, material)) return;

    if (!inputs.Config.Modal.FundamentalFreq) {
        if (const auto path = ActiveSamplePath(r, e)) {
            const auto &frames = GetSampleFrames(r, *path);
            if (!frames.empty()) {
                const auto sr = DeviceSampleRate(r);
                inputs.Config.Modal.FundamentalFreq = EstimateFundamentalFrequency(ComputeFft(frames, sr), sr);
            }
        }
    }
    auto excite_positions = inputs.Vertices | transform([&](uint32_t v) { return inputs.Positions[v]; }) | to<std::vector<vec3>>();
    fastfem::ModeBasis warm_basis;
    if (const auto &warm = r.ctx().get<const ModalWarmStart>(); inputs.Discretization == fastfem::Discretization::Tet10 && warm.Basis && warm.OperatorHash == inputs.OperatorHash) warm_basis = warm.Basis;
    auto work = [inputs = std::move(inputs), material_props = material.Properties, excite_positions = std::move(excite_positions), warm_basis = std::move(warm_basis)](fastfem::SolveMonitor &monitor) mutable -> ModalGenerationResult {
        // Capture sample-surface triangulation before simplification.
        auto sample_triangles = SampleSurfaceTriangles(inputs.TriangleIndices, uint32_t(inputs.Positions.size()), inputs.Vertices);
        auto result = modal::SolveSurfaceModes(
            inputs.Positions, inputs.TriangleIndices, material_props, excite_positions, inputs.NodeScale,
            inputs.Discretization, inputs.Config,
            {.SeedBasis = warm_basis ? &warm_basis : nullptr, .KeepBasis = inputs.Discretization == fastfem::Discretization::Tet10}, &monitor
        );
        if (!result) {
            std::cerr << "Modal solve failed: " << result.error() << ".\n";
            return {};
        }
        // Remap vertices and the sample surface to deduplicated tet sample points.
        result->Modes.Vertices = CompactExcitationVertices(inputs.Vertices, result->SamplePointOfExcitation);
        result->Modes.Indices = RelabelSampleTriangles(sample_triangles, result->SamplePointOfExcitation);
        result->Modes.BakedScale = inputs.NodeScale;
        result->Summary.SolvedVertices = std::move(inputs.Vertices);
        result->Summary.OperatorHash = inputs.OperatorHash;
        result->Summary.ModalConfigHash = inputs.ModalConfigHash;
        monitor.Stage.store(fastfem::SolveStage::Finalizing, std::memory_order_relaxed);
        auto model_path = result->Modes.Freqs.empty() ? fs::path{} : SaveModalModelFile({std::move(result->Modes), result->Mass, std::move(result->Tetrahedra), std::move(result->Summary)});
        monitor.Stage.store(fastfem::SolveStage::Complete, std::memory_order_relaxed);
        return {std::move(model_path), {inputs.OperatorHash, std::move(result->Basis)}};
    };
    // Intentional registry-ctx write outside Apply: transient background-job bookkeeping.
    r.ctx().get<ModalSolveJobs>().Jobs.push_back(std::make_shared<ModalSolveJob>(e, viewport, Job<ModalGenerationResult, fastfem::SolveMonitor>{GetName(r, e), std::move(work)}));
}

bool HasPendingModalSolves(const entt::registry &r) { return !r.ctx().get<const ModalSolveJobs>().Jobs.empty(); }

void RegisterAudioComponentHandlers(entt::registry &r) {
    RegisterSceneClearHandler(r, [](entt::registry &r) {
        // Clear bank slots before entity IDs can be reused by the next scene.
        auto &m = r.ctx().get<ModalAudio>();
        ModalBank empty;
        InstallModalBank(m, empty);
        r.ctx().get<AudioSamples>().ByPath.clear();
        // Clear warm-start data associated with the removed scene.
        r.ctx().get<ModalWarmStart>() = {};
        // In-flight solves target entities from the cleared scene. Their results are discarded on arrival.
        for (auto &job : r.ctx().get<ModalSolveJobs>().Jobs) job->Work.RequestCancel();
    });

    // Create audio context slots once because ProcessAudio reads the registry context concurrently.
    r.ctx().emplace<AudioSamples>();
    r.ctx().emplace<ModalWarmStart>();
    r.ctx().emplace<ModalSolveJobs>();

    track<audio_changes::VertexForce>(r).on<::VertexForce>(On::Create | On::Update | On::Destroy);
    track<audio_changes::ModalModes>(r).on<::ModalModes>(On::Create | On::Update | On::Destroy);
    track<audio_changes::ModalGain>(r).on<ModalGain>(On::Update);
    track<audio_changes::ModalTuning>(r).on<ModalTuning>(On::Update);
    track<audio_changes::ModalSoundControls>(r).on<ModalSoundControls>(On::Create | On::Update);
    track<audio_changes::RecordingStart>(r).on<Recording>(On::Create | On::Update);
    r.ctx().emplace<ModalScaleTracker>().Bind(r);
    track<audio_changes::SoundVerticesDerivation>(r)
        .on<VertexSamples>(On::Create | On::Update | On::Destroy)
        .on<::ModalModes>(On::Create | On::Update | On::Destroy)
        .on<SoundVerticesModel>(On::Create | On::Update | On::Destroy);
    // Refresh body-dependent sound tags after body or hierarchy changes.
    track<audio_changes::ContactReportingDerivation>(r).on<PhysicsBodyHandle>(On::Create | On::Destroy).on<SceneNode>(On::Update | On::Destroy);
    track<audio_changes::ContactDynamicsDerivation>(r)
        .on<MassProperties>(On::Create | On::Update | On::Destroy)
        .on<::ModalModes>(On::Create | On::Update | On::Destroy);
    track<audio_changes::ModelRescaleEdit>(r)
        .on<AcousticMaterial>(On::Create | On::Update)
        .on<PhysicsMotion>(On::Create | On::Update | On::Destroy);
    track<audio_changes::AudioConfig>(r).on<AudioOutputConfig>(On::Create | On::Update);
    track<audio_changes::AudioMix>(r).on<AudioOutputMix>(On::Create | On::Update);
    RegisterSurfaceContactHandlers(r);

    RegisterComponentEventHandler(r, [](entt::registry &r) {
        // Apply completed modal solves.
        auto &solve_jobs = r.ctx().get<ModalSolveJobs>().Jobs;
        for (auto it = solve_jobs.begin(); it != solve_jobs.end();) {
            auto &job = **it;
            auto result = job.Work.Poll();
            if (!result) {
                ++it;
                continue;
            }
            if (!job.Work.Cancelled()) {
                // Intentional registry-ctx write outside Apply: the warm-start slot is a derived memo, not scene input.
                if (result->WarmStart.Basis) r.ctx().get<ModalWarmStart>() = std::move(result->WarmStart);
                if (result->ModelPath.empty()) std::cerr << "Modal model computation failed.\n";
                else if (r.valid(job.Entity) && r.all_of<ModalSolveSettings>(job.Entity)) action::ApplyNow(r, job.Viewport, action::audio::ApplyModalModel{job.Entity, std::move(result->ModelPath)});
            }
            it = solve_jobs.erase(it);
        }
        for (auto e : reactive<audio_changes::ModelRescaleEdit>(r)) {
            if (!r.valid(e) || !r.all_of<ModalEigenSummary, ::ModalModes>(e)) continue;
            RescaleModalObject(r, e);
        }
        if (!reactive<audio_changes::SoundVerticesDerivation>(r).empty()) {
            std::unordered_set<fs::path> used_samples;
            for (const auto &[_, samples] : r.view<const VertexSamples>().each()) {
                for (const auto &[__, path] : samples.PathByVertex) used_samples.insert(path);
            }
            auto &samples = r.ctx().get<AudioSamples>().ByPath;
            for (const auto &path : used_samples) {
                if (samples.contains(path)) continue;
                if (const auto source = RealImpact::SampleGroupFromKey(path)) {
                    auto group = RealImpact::LoadSamples(source->first, source->second);
                    if (group) {
                        for (auto &[key, frames] : *group) samples.try_emplace(std::move(key), std::move(frames));
                    } else {
                        r.ctx().get<action::Errors>().Messages.push_back(std::move(group.error()));
                        samples.try_emplace(path);
                    }
                } else {
                    samples.emplace(path, LoadAudioFrames(path.string(), DeviceSampleRate(r)));
                }
            }
            std::erase_if(samples, [&](const auto &entry) { return !used_samples.contains(entry.first); });
        }
        // Rebuild SoundVertices from VertexSamples/ModalModes, selected by SoundVerticesModel.
        // Runs before any handler that reads SoundVertices.
        bool reporting_stale = !reactive<audio_changes::ContactReportingDerivation>(r).empty();
        for (auto e : reactive<audio_changes::SoundVerticesDerivation>(r)) {
            const auto *model = r.try_get<const SoundVerticesModel>(e);
            std::vector<uint32_t> new_vertices;
            if (model) {
                if (*model == SoundVerticesModel::Samples) {
                    if (const auto *vs = r.try_get<const VertexSamples>(e)) {
                        new_vertices = vs->PathByVertex | std::views::keys | to<std::vector>();
                    }
                } else if (const auto *modes = r.try_get<const ::ModalModes>(e)) {
                    new_vertices = modes->Vertices;
                }
            }
            reporting_stale = true;
            if (new_vertices.empty()) {
                r.remove<SoundVertices>(e);
                continue;
            }
            auto &meshes = r.ctx().get<MeshStore>();
            if (auto *sv = r.try_get<SoundVertices>(e)) {
                if (!std::ranges::equal(meshes.GetSoundVertices(sv->Vertices), new_vertices)) {
                    meshes.ReleaseSoundVertices(sv->Vertices);
                    r.replace<SoundVertices>(e, SoundVertices{meshes.AllocateSoundVertices(new_vertices)});
                }
            } else {
                r.emplace<SoundVertices>(e, SoundVertices{meshes.AllocateSoundVertices(new_vertices)});
            }
            // Ensure MeshActiveElement is valid for the new vertex set.
            const auto mesh_entity = r.get<const Instance>(e).Entity;
            const auto &sv = r.get<const SoundVertices>(e);
            if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity)) {
                const auto vertices = meshes.GetSoundVertices(sv.Vertices);
                if (!FindSoundVertexIndex(vertices, active->Handle)) r.emplace_or_replace<MeshActiveElement>(mesh_entity, vertices.front());
            }
        }
        // A body reports contacts when anything under it can sound.
        // Stop traversal at nested rigid bodies so each node maps to one body.
        // Intentional registry write outside Apply: derived from the sound models under each body.
        if (reporting_stale) {
            const auto sounds = [&r](this auto &self, entt::entity node) -> bool {
                if (IsModalSounding(r, node)) return true;
                for (auto child : Children{&r, node}) {
                    if (!r.all_of<PhysicsBodyHandle>(child) && self(child)) return true;
                }
                return false;
            };
            for (const auto body : r.view<const PhysicsBodyHandle>()) {
                if (sounds(body)) r.emplace_or_replace<ReportContacts>(body);
                else r.remove<ReportContacts>(body);
            }
        }
        // Refresh contact dynamics before the strike loop below reads them.
        for (auto e : reactive<audio_changes::ContactDynamicsDerivation>(r)) UpdateContactDynamics(r, e);
        // A created or replaced VertexForce is a strike. Contact pulses are one-shot.
        for (auto e : reactive<audio_changes::VertexForce>(r)) {
            if (!r.all_of<SoundVerticesModel>(e)) continue;
            const auto *vf = r.try_get<::VertexForce>(e);
            if (!vf || vf->Force <= 0) continue;
            const auto &excitable = r.get<const SoundVertices>(e);
            if (auto vi = FindSoundVertexIndex(r.ctx().get<const MeshStore>().GetSoundVertices(excitable.Vertices), vf->Vertex)) {
                r.emplace_or_replace<MeshActiveElement>(r.get<const Instance>(e).Entity, vf->Vertex);
                const auto model = r.get<SoundVerticesModel>(e);
                if (model == SoundVerticesModel::Modal && r.all_of<ModalModes>(e)) {
                    TriggerModalStrike(r, e, *vi, vf->Force, vf->ContactSpeed);
                } else if (model == SoundVerticesModel::Samples && r.all_of<VertexSamples>(e)) {
                    r.get_or_emplace<SamplePlayback>(e).Play();
                }
            }
        }
        // Start a new recording with an impact at the active vertex.
        for (auto e : reactive<audio_changes::RecordingStart>(r)) {
            if (!r.all_of<ModalModes, SoundVertices, Recording>(e)) continue;
            if (r.get<const Recording>(e).Frame == 0) TriggerModalStrike(r, e, GetActiveVertexIndex(r, e), 1.f, 1.f);
        }
        // Reconcile the live output device: a config change re-inits (and may change the negotiated rate), a mix change just applies level/on-off.
        bool device_rate_changed = false;
        if (auto *res = r.ctx().find<AudioDeviceResource>()) {
            if (auto &config_tracker = reactive<audio_changes::AudioConfig>(r); !config_tracker.empty()) {
                const uint32_t prev_rate = res->SampleRate;
                for (auto e : config_tracker) {
                    if (r.all_of<AudioOutputConfig, AudioOutputMix>(e)) ReconcileAudioDevice(*res, r.get<const AudioOutputConfig>(e), r.get<const AudioOutputMix>(e));
                }
                device_rate_changed = res->SampleRate != prev_rate;
                // A reopened device publishes a new scheduling group, which the render threads have to be re-placed into.
                r.ctx().get<ModalAudio>().RenderPool.SetWorkgroup(res->RenderWorkgroup);
            }
            for (auto e : reactive<audio_changes::AudioMix>(r)) {
                if (r.all_of<AudioOutputMix>(e)) ApplyAudioMix(*res, r.get<const AudioOutputMix>(e));
            }
        }

        auto &modal_tracker = reactive<audio_changes::ModalModes>(r);
        if (!modal_tracker.empty() || device_rate_changed) {
            // Ensure every modal object has tuning, gain, solve settings, and acoustic material.
            // Intentional registry writes outside Apply: derived defaults for a new model.
            for (auto e : modal_tracker) {
                const auto *modes = r.try_get<const ::ModalModes>(e);
                if (!modes) continue;
                if (!r.all_of<ModalTuning>(e)) r.emplace<ModalTuning>(e, modes->Freqs.empty() ? 0.f : modes->Freqs.front(), 1.f);
                if (!r.all_of<ModalGain>(e)) r.emplace<ModalGain>(e);
                if (!r.all_of<ModalSolveSettings>(e)) {
                    r.emplace<ModalSolveSettings>(e, modes->Vertices.empty() ? ModalSolveSettings{} : ModalSolveSettings{.NumVertices = uint32_t(modes->Vertices.size())});
                }
                if (!r.all_of<AcousticMaterial>(e)) r.emplace<AcousticMaterial>(e, materials::acoustic::All.front());
            }
            // Retune and reshape same-layout model replacements in place.
            // Rebuild the bank for structural layout changes.
            auto &m = r.ctx().get<ModalAudio>();
            bool rebuild = false;
            for (auto e : modal_tracker) {
                if (rebuild) break;
                const auto *modes = r.valid(e) ? r.try_get<const ::ModalModes>(e) : nullptr;
                const bool active = modes && !modes->Freqs.empty() && r.all_of<SoundVertices>(e);
                const auto slot = FindModalObject(LiveBank(m), e);
                if (!active && !slot) continue;
                if (active && slot && SetModalObjectShapes(LiveBank(m), *slot, *modes)) RetuneModalObject(r, LiveBank(m), *slot, e);
                else rebuild = true;
            }
            if (rebuild) {
                RebuildModalBank(r);
            } else if (device_rate_changed) {
                // A rate change keeps the layout but rebakes every coefficient.
                LiveBank(m).SampleRate = float(DeviceSampleRate(r));
                for (uint32_t slot = 0; slot < uint32_t(LiveBank(m).Entities.size()); ++slot) RetuneModalObject(r, LiveBank(m), slot, LiveBank(m).Entities[slot]);
            }
        }
        {
            auto &m = r.ctx().get<ModalAudio>();
            auto &bank = LiveBank(m);
            for (auto e : reactive<audio_changes::ModalGain>(r)) {
                if (auto slot = FindModalObject(bank, e)) SetModalOutGain(r, bank, *slot, e);
            }
            for (auto e : reactive<audio_changes::ModalTuning>(r)) {
                if (auto slot = FindModalObject(bank, e)) RetuneModalObject(r, bank, *slot, e);
            }
            for (auto e : reactive<audio_changes::ModalSoundControls>(r)) {
                const auto &controls = r.get<const ModalSoundControls>(e);
                m.ClickGain.store(controls.ClickGain, std::memory_order_relaxed);
                m.MaxImpacts.store(controls.MaxImpacts, std::memory_order_relaxed);
                // Sized from the main thread, since a render call cannot spawn threads.
                const auto *device = r.ctx().find<const AudioDeviceResource>();
                m.RenderPool.SetSize(controls.RenderThreads);
                m.RenderPool.SetWorkgroup(device ? device->RenderWorkgroup : nullptr);
                for (uint32_t slot = 0; slot < uint32_t(bank.Entities.size()); ++slot) SetModalOutGain(r, bank, slot, bank.Entities[slot]);
            }
            // Retune objects whose node was rescaled.
            auto &scale_tracker = r.ctx().get<ModalScaleTracker>();
            for (auto e : scale_tracker.Storage) {
                if (auto slot = FindModalObject(bank, e)) RetuneModalObject(r, bank, *slot, e);
            }
            scale_tracker.Storage.clear();
            // Publish camera-derived object attenuation because the audio thread cannot access the registry.
            if (const auto *res = r.ctx().find<AudioDeviceResource>()) UpdateListenerGains(r, bank, res->Viewport);
        }
    });
}

static void UpdateAudioContacts(entt::registry &r) {
    // Displayed-frame collisions strike the objects they hit, once per contact point.
    if (auto *contacts = r.ctx().find<PhysicsContactImpacts>(); contacts && !contacts->Events.empty()) {
        const auto &controls = ModalControls(r);
        for (const auto &c : contacts->Events) {
            if (c.Speed < controls.MinContactSpeed) continue;
            const auto own = ResolveContactNodes(r, c.ColliderEntity, c.Entity);
            if (!IsModalSounding(r, own.Model)) continue;
            const auto &modes = r.get<const ModalModes>(own.Model);
            if (modes.Positions.empty()) continue;
            // Bring the world-space contact into the node-local frame the modes are defined in.
            const auto &wt = r.get<const WorldTransform>(own.Model);
            const vec3 local_point = InverseTransformPoint(wt, c.Point);
            const vec3 local_dir = InverseTransformDir(wt, c.Direction);
            const auto sample_point = NearestSamplePoint(modes.Positions, local_point);
            // Apply the audibility floor to modal excitation rather than impact momentum.
            if (PeakModalDrive(modes, sample_point, UnitOrZero(local_dir) * c.Impulse) < controls.MinContactExcitation) continue;
            const auto other = ResolveContactNodes(r, c.OtherColliderEntity, c.Other);
            // The other body is the impactor: its stiffness, mass, and curvature shape the contact time.
            // Derive impactor material and curvature from the contacted surface.
            const auto &other_props = MaterialOf(r, other.Surface, other.Model);
            // A body with no mesh is treated as a solid sphere of its mass.
            const auto other_curvature = SurfaceCurvature(r, other.Geometry, c.Point);
            const Impactor impactor{
                .Material = other_props,
                .Curvature = other_curvature.value_or(SphereEquivalentCurvature(other_props.Density, c.OtherInvMass)),
                .InvMass = c.OtherInvMass,
            };
            const auto resultant_point = NearestSamplePoint(modes.Positions, InverseTransformPoint(wt, c.ResultantPoint));
            const float own_rq = SurfaceRoughnessOf(r, own.Surface), other_rq = SurfaceRoughnessOf(r, other.Surface);
            const float pair_roughness = std::sqrt(own_rq * own_rq + other_rq * other_rq);
            TriggerModalStrike(r, own.Model, sample_point, c.Impulse, c.Speed, PhysicsStrike{local_dir, c.Point, own.Geometry, own.Surface, impactor, c.NominalArea, pair_roughness, resultant_point});
        }
        contacts->Events.clear();
    }
    // Recompute edited surfaces before publishing persistent contacts for this step.
    SurfaceUpdateContacts(r);
}

void InitAudioSystem(entt::registry &r) {
    // A second call would connect every tracker twice.
    if (r.ctx().contains<ModalAudio>()) return;
    r.ctx().emplace<ModalAudio>();
    r.ctx().emplace<MonitorLimiter>();
    RegisterAudioComponentHandlers(r);
    RegisterComponentEventHandler(r, UpdateAudioContacts, ComponentEventPhase::AfterPose);
}

void DeinitAudioSystem(entt::registry &r) { r.ctx().erase<ModalAudio>(); }

void RemoveAudioComponents(entt::registry &r, entt::entity e) {
    CancelModalSolves(r, e);
    r.remove<ScaleLocked, SoundVertices, Recording, SoundVerticesModel, ModalModes, ModalGain, ModalTuning, MassProperties, ContactDynamics, ModalEigenSummary, VertexSamples, SamplePlayback, ModalSolveSettings, RealImpactActiveMicrophone, RealImpactVertices>(e);
}

void ApplyModalModel(entt::registry &r, entt::entity e, const fs::path &relative_path) {
    if (!r.valid(e) || !r.all_of<Instance>(e)) {
        std::cerr << std::format("Modal model target entity is gone, skipping {}.\n", relative_path.string());
        return;
    }
    auto data = LoadModalModelFile(relative_path);
    if (!data) {
        std::cerr << std::format("Failed to load modal model file {}.\n", (ModalModelsDir() / relative_path).string());
        return;
    }
    const auto mesh_entity = r.get<const Instance>(e).Entity;
    // Apply the current node material with density inferred from the solved body mass.
    const auto *mat = r.try_get<const AcousticMaterial>(e);
    if (const auto props = EffectiveModalMaterial(mat ? mat->Properties : data->Summary.SolvedMaterial, data->Summary, data->Mass.Mass, r.try_get<const PhysicsMotion>(e));
        props != data->Summary.SolvedMaterial) {
        const auto *settings = r.try_get<const ModalSolveSettings>(e);
        if (auto rescaled = RescaledModes(data->Summary, data->Modes, props, settings ? *settings : ModalSolveSettings{})) data->Modes = std::move(*rescaled);
    }
    r.emplace_or_replace<MassProperties>(e, data->Mass);
    ReplaceModalModes(r, e, std::move(data->Modes));
    r.emplace_or_replace<ModalEigenSummary>(e, std::move(data->Summary));
    auto &meshes = r.ctx().get<MeshStore>();
    if (const auto *existing = r.try_get<const TetBuffers>(mesh_entity)) meshes.ReleaseTets(*existing);
    r.emplace_or_replace<TetBuffers>(mesh_entity, meshes.AllocateTets(data->Tets.Positions, data->Tets.EdgeIndices));
    SetModel(r, e, SoundVerticesModel::Modal);
}
