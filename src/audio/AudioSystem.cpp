#include "AudioSystem.h"
#include "AudioDevice.h"
#include "ContactScene.h"
#include "Fft.h"
#include "FileDialog.h"
#include "Job.h"
#include "ModalAudio.h"
#include "Reactive.h"
#include "action/ActionApply.h"
#include "action/Audio.h"
#include "audio/WavWriter.h"
#include "mesh/MeshStore.h"
#include "physics/PhysicsContact.h"
#include "physics/PhysicsTypes.h"
#include "selection/SelectionBitset.h"
#include "ui/FieldEdit.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCamera.h"
#include "viewport/ViewportEvents.h"
#include <FastFEM/SolveMonitor.h>
#include <atomic>

#include "ModalModelFile.h"
#include "ModalSolve.h"
#include "ModalWarmStart.h"
#include "implot.h"
#include "imspinner.h"

#include "ui/HelpMarker.h" // depends on imgui
#include "ui/PresetCombo.h"

#include <iostream>
#include <numbers>
#include <string_view>

namespace fs = std::filesystem;

// Ranges cover the acoustic material presets with headroom (see materials::acoustic::All).
// Limit Poisson ratio below 0.5 to keep Lame's lambda finite.
// Set the beta floor above the logarithmic slider's zero epsilon.
template<> struct FieldLimits<&AcousticMaterial::Properties, &AcousticMaterialProperties::Density> : Within<1., 25000.> {};
template<> struct FieldLimits<&AcousticMaterial::Properties, &AcousticMaterialProperties::YoungModulus> : Within<1e5, 1e12> {};
template<> struct FieldLimits<&AcousticMaterial::Properties, &AcousticMaterialProperties::PoissonRatio> : Within<0., 0.49> {};
template<> struct FieldLimits<&AcousticMaterial::Properties, &AcousticMaterialProperties::Alpha> : Within<0., 200.> {};
template<> struct FieldLimits<&AcousticMaterial::Properties, &AcousticMaterialProperties::Beta> : Within<1e-9, 1e-4> {};
using SurfaceSolveConfig = fastfem::SurfaceSolveConfig;
using ModalSolverConfig = fastfem::SolverConfig;
using TetConfig = fastfem::TetrahedralizationConfig;
using FiniteCellConfig = fastfem::FiniteCellConfig;
template<> struct FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Resolution> : Within<1., 256.> {};
template<> struct FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::SurfaceSimplificationRatio> : Within<0.25, 1.> {};
template<> struct FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::NumModes> : Within<1., 512.> {};
template<> struct FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::NumFemModes> : Within<1., 512.> {};
template<> struct FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::MinModeFreq> : Within<20., 20000.> {};
template<> struct FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::MaxModeFreq> : Within<20., 20000.> {};
template<> struct FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::Tolerance> : Within<1e-12, 1e-3> {};
template<> struct FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::MaxRestarts> : Within<1., 1000.> {};
template<> struct FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::FiniteCell, &FiniteCellConfig::CutDepth> : Within<0., 8.> {};
template<> struct FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::FiniteCell, &FiniteCellConfig::FictitiousScale> : Within<1e-12, 1e-2> {};
template<> struct FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::FiniteCell, &FiniteCellConfig::PaddingCells> : Within<0., 2.> {};

// Striker capsule dimensions, in meters.
template<> struct FieldLimits<&Striker::TipRadius> : Within<0.0005, 0.1> {};
template<> struct FieldLimits<&Striker::Length> : Within<0.001, 1.> {};

// Modal synthesis controls.
template<> struct FieldLimits<&ModalGain::Value> : Within<0., 2.> {};
template<> struct FieldLimits<&ModalTuning::FundamentalFreq> : Within<20., 16000.> {};
template<> struct FieldLimits<&ModalTuning::T60Scale> : Within<0.1, 10.> {};
template<> struct FieldLimits<&ModalSoundControls::ModalLevel> : Within<0., 1.> {};
template<> struct FieldLimits<&ModalSoundControls::ClickGain> : Within<0., 10.> {};
template<> struct FieldLimits<&ModalSoundControls::SampleGain> : Within<0., 4.> {};
template<> struct FieldLimits<&ModalSoundControls::RenderThreads> : Within<1., 16.> {};
template<> struct FieldLimits<&ModalSoundControls::MaxImpacts> : Within<1., 4096.> {};
template<> struct FieldLimits<&ModalSoundControls::MinContactExcitation> : Within<0., 1e-3> {};
template<> struct FieldLimits<&ModalSoundControls::MinContactSpeed> : Within<0., 5.> {};

using std::ranges::iota_view, std::ranges::nth_element, std::ranges::max_element, std::ranges::to;
using std::views::transform;

uint32_t DeviceSampleRate(const entt::registry &r) {
    const auto *res = r.ctx().find<AudioDeviceResource>();
    if (res && res->SampleRate) return res->SampleRate;
    // AUDIO_SAMPLE_RATE overrides the 48 kHz default when no device provides a rate.
    static const uint32_t fallback = [] {
        const char *env = std::getenv("AUDIO_SAMPLE_RATE");
        const auto rate = env ? std::atoi(env) : 0;
        return rate > 0 ? uint32_t(rate) : 48'000u;
    }();
    return fallback;
}

namespace {
// Per-sound-object component. Maps mesh vertex handles to sample keys in the scene-level AudioSamples store.
// Only vertices that have a sample appear in the map.
struct VertexSamples {
    std::map<uint32_t, fs::path> PathByVertex;
    uint32_t Frame{0};
    bool Stopped{true};

    std::optional<fs::path> FindPath(uint32_t mesh_vertex) const {
        auto it = PathByVertex.find(mesh_vertex);
        return it != PathByVertex.end() ? std::optional{it->second} : std::nullopt;
    }
    void Stop() { Stopped = true; }
    void Play() {
        Frame = 0;
        Stopped = false;
    }
};

// Stores one refcounted sample per path for all vertices and sound objects in a scene.
struct AudioSamples {
    struct Entry {
        std::vector<float> Frames;
        uint32_t RefCount{0};
    };
    std::unordered_map<fs::path, Entry> ByPath;
};

const std::vector<float> &GetSampleFrames(const entt::registry &r, entt::entity viewport, const fs::path &path) {
    static const std::vector<float> EmptyFrames{};
    if (path.empty()) return EmptyFrames;
    const auto *store = r.try_get<const AudioSamples>(viewport);
    if (!store) return EmptyFrames;
    const auto it = store->ByPath.find(path);
    return it != store->ByPath.end() ? it->second.Frames : EmptyFrames;
}

// Inserts frames if `path` is new, otherwise reuses existing frames. Bumps refcount either way.
void AcquireSample(entt::registry &r, entt::entity viewport, const fs::path &path, std::vector<float> &&frames) {
    if (path.empty()) return;
    auto &store = r.get_or_emplace<AudioSamples>(viewport);
    auto [it, inserted] = store.ByPath.try_emplace(path);
    if (inserted) it->second.Frames = std::move(frames);
    ++it->second.RefCount;
}

// Decrements refcount, erasing the entry (and the component if empty) when it hits 0.
void ReleaseSample(entt::registry &r, entt::entity viewport, const fs::path &path) {
    if (path.empty()) return;
    auto *store = r.try_get<AudioSamples>(viewport);
    if (!store) return;
    const auto it = store->ByPath.find(path);
    if (it == store->ByPath.end()) return;
    if (--it->second.RefCount == 0) store->ByPath.erase(it);
    if (store->ByPath.empty()) r.remove<AudioSamples>(viewport);
}
} // namespace

void AssignVertexSample(
    entt::registry &r, entt::entity viewport, entt::entity e,
    std::span<const uint32_t> mesh_vertices, fs::path path, std::vector<float> &&frames
) {
    if (mesh_vertices.empty() || path.empty()) return;
    auto &vs = r.get_or_emplace<VertexSamples>(e);
    vs.Stop();

    // The first new path consumes frames.
    // Later calls reuse the stored path and ignore the moved-from vector while increasing its reference count.
    bool vs_changed = false;
    for (const uint32_t mv : mesh_vertices) {
        auto [it, inserted] = vs.PathByVertex.try_emplace(mv, path);
        if (!inserted) {
            if (it->second == path) continue;
            ReleaseSample(r, viewport, it->second);
            it->second = path;
        }
        AcquireSample(r, viewport, path, std::move(frames)); // NOLINT(bugprone-use-after-move) only the first new-path call reads frames
        vs_changed = true;
    }
    if (vs_changed) r.patch<VertexSamples>(e, [](auto &) {});
    if (!r.all_of<SoundVerticesModel>(e)) r.emplace<SoundVerticesModel>(e, SoundVerticesModel::Samples);
}

void RemoveVertexSamples(
    entt::registry &r, entt::entity viewport, entt::entity e,
    std::span<const uint32_t> mesh_vertices
) {
    auto *vs = r.try_get<VertexSamples>(e);
    if (!vs || mesh_vertices.empty()) return;
    vs->Stop();
    bool vs_changed = false;
    for (const uint32_t mv : mesh_vertices) {
        const auto it = vs->PathByVertex.find(mv);
        if (it == vs->PathByVertex.end()) continue;
        ReleaseSample(r, viewport, it->second);
        vs->PathByVertex.erase(it);
        vs_changed = true;
    }
    if (vs_changed) r.patch<VertexSamples>(e, [](auto &) {});
    if (vs->PathByVertex.empty()) {
        if (r.all_of<ModalModes>(e)) r.remove<VertexSamples>(e);
        else RemoveAudioComponents(r, e);
    }
}

void SetVertexSamples(
    entt::registry &r, entt::entity viewport, entt::entity e,
    std::span<const uint32_t> mesh_vertices, std::vector<LoadedSample> &&samples
) {
    for (size_t i = 0; i < samples.size() && i < mesh_vertices.size(); ++i) {
        AssignVertexSample(r, viewport, e, {&mesh_vertices[i], 1}, std::move(samples[i].first), std::move(samples[i].second));
    }
}

namespace {
// Returns the active SoundVertices index derived from the mesh entity's MeshActiveElement.
// Returns zero when no active element is set.
uint32_t GetActiveVertexIndex(const entt::registry &r, entt::entity instance_entity) {
    const auto &excitable = r.get<const SoundVertices>(instance_entity);
    const auto mesh_entity = r.get<const Instance>(instance_entity).Entity;
    if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity)) {
        if (auto vi = FindSoundVertexIndex(r.ctx().get<const MeshStore>().GetSoundVertices(excitable.Vertices), active->Handle)) return *vi;
    }
    return 0;
}

// Returns the sample-store path assigned to the instance's active mesh vertex, if any.
std::optional<fs::path> ActiveSamplePath(const entt::registry &r, entt::entity instance_entity) {
    const auto *samples = r.try_get<const VertexSamples>(instance_entity);
    if (!samples) return std::nullopt;
    const auto mesh_entity = r.get<const Instance>(instance_entity).Entity;
    const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
    return active ? samples->FindPath(active->Handle) : std::nullopt;
}

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

// Updates listener attenuation using inverse distance beyond ListenerDistance and a constant level within it.
void UpdateListenerGains(const entt::registry &r, ModalBank &b, entt::entity viewport) {
    const auto *camera = r.valid(viewport) ? r.try_get<const ViewCamera>(viewport) : nullptr;
    if (!camera) return;
    const auto listener_pos = camera->Position();
    for (uint32_t slot = 0; slot < uint32_t(b.Entities.size()); ++slot) {
        const auto e = b.Entities[slot];
        const auto *world = r.valid(e) ? r.try_get<const WorldTransform>(e) : nullptr;
        const float distance = world ? numeric::Distance(listener_pos, world->P) : ListenerDistance;
        std::atomic_ref{b.ListenerGain[slot]}.store(ListenerDistance / std::max(distance, ListenerDistance), std::memory_order_relaxed);
    }
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
    if (auto *samples = r.try_get<VertexSamples>(e)) samples->Stop();
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
// Strike impact angle relative to the surface.
// Center strikes along surface normal, rim tilts impulse 90 degrees into the tangent plane. UI-only.
vec2 ImpulseAngle{0, 0};

// Unit surface normal at a mesh vertex.
vec3 VertexNormal(const Mesh &mesh, uint32_t vertex) { return numeric::Normalize(mesh.GetNormal(Mesh::VH{vertex})); }

// Tilts a unit normal using a joystick position in the unit disk.
vec3 TiltAlongNormal(vec3 n, vec2 joy) {
    const float r = numeric::Length(joy);
    if (r < 1e-6f) return n;
    // Orthonormal tangent basis from the normal (Duff et al. 2017).
    const float s = n.z >= 0 ? 1.f : -1.f;
    const float a = -1.f / (s + n.z);
    const float b = n.x * n.y * a;
    const vec3 t{1.f + s * n.x * n.x * a, s * b, -s * n.x};
    const vec3 bt{b, s + n.y * n.y * a, -n.y};
    const float theta = std::min(r, 1.f) * 1.57079633f; // radius maps to [0, pi/2]
    return std::cos(theta) * n + std::sin(theta) * (joy.x * t + joy.y * bt) / r;
}

// Strike direction: the excited vertex's normal, tilted by the current impact angle.
vec3 ExciteDirection(const entt::registry &r, entt::entity e, uint32_t vertex) {
    const auto n = VertexNormal(GetMesh(r, r.get<const Instance>(e).Entity), vertex);
    return TiltAlongNormal(n, ImpulseAngle);
}

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

/***** Impact spectrum analysis *****/

// The spectrum of one windowed segment, over the bins its sample count resolves.
struct FFTData {
    std::vector<std::complex<float>> Bins;
    size_t NumReal;
};

constexpr void ApplyCosineWindow(float *w, uint32_t n, const float *coeff, uint32_t ncoeff) {
    if (n == 1) {
        w[0] = 1.0;
        return;
    }

    const uint32_t wlength = n;
    for (uint32_t i = 0; i < n; ++i) {
        float wi = 0.0;
        for (uint32_t j = 0; j < ncoeff; ++j) wi += coeff[j] * __cospi(float(2 * i * j) / float(wlength));
        w[i] = wi;
    }
}

constexpr std::vector<float> CreateBlackmanHarris(uint32_t n) {
    std::vector<float> window(n);
    static constexpr float coeff[4] = {0.35875, -0.48829, 0.14128, -0.01168};
    ApplyCosineWindow(window.data(), n, coeff, sizeof(coeff) / sizeof(float));
    return window;
}

constexpr std::vector<float> ApplyWindow(const std::vector<float> &window, const float *data) {
    std::vector<float> windowed(window.size());
    for (uint32_t i = 0; i < window.size(); ++i) windowed[i] = window[i] * data[i];
    return windowed;
}

std::optional<float> EstimateFundamentalFrequency(const FFTData &fft, uint32_t sample_rate) {
    const size_t n_bins = fft.Bins.size();

    std::vector<float> mag_db(n_bins);
    for (size_t i = 0; i < n_bins; ++i) {
        mag_db[i] = 10.f * std::log10f(std::max(std::norm(fft.Bins[i]), 1e-20f));
    }

    // Noise floor from upper half median
    std::vector<float> upper(mag_db.begin() + n_bins / 2, mag_db.end());
    nth_element(upper, upper.begin() + upper.size() / 2);
    const float threshold = upper[upper.size() / 2] + 15.f;

    constexpr size_t W{15}; // Prominence window
    const size_t min_bin = 50 * fft.NumReal / sample_rate;
    for (size_t i = std::max(min_bin, W); i < n_bins - W; ++i) {
        if (mag_db[i] <= mag_db[i - 1] || mag_db[i] <= mag_db[i + 1] || mag_db[i] < threshold) continue;

        constexpr float ProminenceThresholdDb{10.f};
        // Prominence check: peak must be above the local mean by ProminenceThresholdDb
        float local_sum = 0;
        for (size_t j = i - W; j <= i + W; ++j) local_sum += mag_db[j];
        const float local_mean = local_sum / (2 * W + 1);
        if (mag_db[i] - local_mean >= ProminenceThresholdDb) return i * sample_rate / fft.NumReal;
    }
    return std::nullopt;
}

// Capture a short audio segment shortly after the impact for FFT.
FFTData ComputeFft(const std::vector<float> &frames, uint32_t sample_rate) {
    constexpr uint32_t FftStartFrame = 30;
    const uint32_t FftEndFrame = sample_rate / 16;
    const auto window = CreateBlackmanHarris(FftEndFrame - FftStartFrame);
    const auto windowed = ApplyWindow(window, frames.data() + FftStartFrame);
    return {fft::RealToComplex(windowed), windowed.size()};
}

/***** Modal solve jobs *****/

struct ModalGenerationResult {
    std::filesystem::path ModelPath; // Result file, relative to ModalModelsDir(). Empty when the solve failed or was cancelled
    ModalWarmStart WarmStart;
};

// An in-flight modal solve, at most one per sound entity.
struct ModalSolveJob {
    entt::entity Entity, Viewport;
    Job<ModalGenerationResult, fastfem::SolveMonitor> Work;
};
struct ModalSolveJobs {
    std::vector<std::shared_ptr<ModalSolveJob>> Jobs;
};

bool IsSolving(const entt::registry &r, entt::entity e) {
    return std::ranges::any_of(r.ctx().get<const ModalSolveJobs>().Jobs, [e](const auto &job) { return job->Entity == e; });
}

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

struct SolveInputs {
    std::vector<vec3> Positions; // Mesh positions at the node's world scale (SI meters)
    std::vector<uint32_t> TriangleIndices;
    std::vector<uint32_t> Vertices; // Excitation vertices
    fastfem::SurfaceSolveConfig Config;
    fastfem::Discretization Discretization;
    vec3 NodeScale;
    size_t OperatorHash, ModalConfigHash;
};

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
            const auto &frames = GetSampleFrames(r, viewport, *path);
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
} // namespace

void RegisterAudioComponentHandlers(entt::registry &r) {
    RegisterSceneClearHandler(r, [](entt::registry &r) {
        // Clear bank slots before entity IDs can be reused by the next scene.
        auto &m = r.ctx().get<ModalAudio>();
        ModalBank empty;
        InstallModalBank(m, empty);
        // Clear warm-start data associated with the removed scene.
        r.ctx().get<ModalWarmStart>() = {};
        // In-flight solves target entities from the cleared scene. Their results are discarded on arrival.
        for (auto &job : r.ctx().get<ModalSolveJobs>().Jobs) job->Work.RequestCancel();
    });

    // Create modal solve context slots once because ProcessAudio reads the registry context concurrently.
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
    // Reapply body-dependent sound tags when an ancestor body is added or removed.
    track<audio_changes::ContactReportingDerivation>(r).on<PhysicsBodyHandle>(On::Create | On::Destroy);
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
                    r.patch<VertexSamples>(e, [](auto &s) { s.Play(); });
                }
            }
        }
        // Start a new recording with an impact at the active vertex.
        for (auto e : reactive<audio_changes::RecordingStart>(r)) {
            if (!r.all_of<ModalModes, SoundVertices, Recording>(e)) continue;
            if (r.get<const Recording>(e).Frame == 0) TriggerModalStrike(r, e, GetActiveVertexIndex(r, e), 1.f, 1.f);
        }
        // Last step's collisions strike the objects they hit, once per contact point.
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

namespace {
// The device-to-main-thread ring overwrites old audio instead of blocking the callback.
struct MasterCapture {
    std::vector<float> Ring;
    std::atomic<uint64_t> Written{0}, Read{0};
};

} // namespace

uint32_t BeginAudioCapture(entt::registry &r) {
    const auto *res = r.ctx().find<AudioDeviceResource>();
    const auto rate = res && res->SampleRate ? res->SampleRate : 0u;
    if (rate == 0) return 0;
    auto &capture = r.ctx().emplace<MasterCapture>();
    capture.Ring.assign(size_t(rate) * 2, 0.f);
    capture.Written.store(0, std::memory_order_relaxed);
    capture.Read.store(0, std::memory_order_relaxed);
    return rate;
}

void EndAudioCapture(entt::registry &r) { r.ctx().erase<MasterCapture>(); }

void InitAudioSystem(entt::registry &r) {
    // A second call would connect every tracker twice.
    if (r.ctx().contains<ModalAudio>()) return;
    r.ctx().emplace<ModalAudio>();
    r.ctx().emplace<MonitorLimiter>();
    RegisterAudioComponentHandlers(r);
}

void DeinitAudioSystem(entt::registry &r) { r.ctx().erase<ModalAudio>(); }

void RenderAudioOffline(entt::registry &r, entt::entity viewport, std::vector<float> &out, uint32_t frame_count) {
    const auto first = out.size();
    out.resize(first + frame_count);
    ProcessAudio(r, viewport, out.data() + first, frame_count);
}

void DrainAudioCapture(entt::registry &r, std::vector<float> &out) {
    auto *capture = r.ctx().find<MasterCapture>();
    if (!capture || capture->Ring.empty()) return;
    const auto written = capture->Written.load(std::memory_order_acquire);
    auto read = capture->Read.load(std::memory_order_relaxed);
    // Resume from the oldest retained frame after an overwrite.
    const auto capacity = uint64_t(capture->Ring.size());
    if (written - read > capacity) read = written - capacity;
    for (auto pos = size_t(read % capacity); read < written; ++read) {
        out.push_back(capture->Ring[pos]);
        if (++pos == capacity) pos = 0;
    }
    capture->Read.store(read, std::memory_order_relaxed);
}

// The pressure a full-scale device sample represents, Pa: the monitor level, 120 dB SPL, the threshold of pain.
// Maps the monitor pressure ceiling to full scale.
constexpr float FullScalePressure{20.f};

// Uses instant attack and 100 ms release while preserving relative levels below the peak envelope.
void MonitorFrames(entt::registry &r, std::span<float> frames, MonitorLimiter &limiter) {
    const float release = std::exp(-1.f / (0.1f * float(DeviceSampleRate(r))));
    auto envelope = limiter.Envelope;
    for (auto &frame : frames) {
        const float x = frame / FullScalePressure;
        envelope = std::max(std::abs(x), envelope * release);
        frame = envelope > 1.f ? x / envelope : x;
    }
    limiter.Envelope = envelope;
}

void ProcessAudio(entt::registry &r, entt::entity viewport, float *output, uint32_t frame_count, bool monitor) {
    std::fill_n(output, frame_count, 0.f);
    auto &m = r.ctx().get<ModalAudio>();
    // The mix is pressure at the view camera.
    // The device path runs this on the audio thread, which cannot read the registry, so its gains are written by the frame handler.
    // Offline rendering uses its viewport camera on the main thread.
    if (!monitor) UpdateListenerGains(r, LiveBank(m), viewport);
    RenderModal(m, output, frame_count);

    const auto *controls = r.try_get<const ModalSoundControls>(viewport);
    // Recorded samples are normalized, so they enter the pressure mix at the monitor calibration and a full-scale sample plays at full scale.
    const float sample_gain = (controls ? controls->SampleGain : ModalSoundControls{}.SampleGain) * FullScalePressure;
    for (const auto [entity, model] : r.view<SoundVerticesModel>().each()) {
        if (model == SoundVerticesModel::Samples) {
            auto *samples = r.try_get<VertexSamples>(entity);
            if (!samples || samples->Stopped) continue;
            const auto path = ActiveSamplePath(r, entity);
            if (!path) continue;
            const auto &impact_samples = GetSampleFrames(r, viewport, *path);
            for (uint32_t i = 0; i < frame_count; ++i) {
                output[i] += (samples->Frame < impact_samples.size() ? impact_samples[samples->Frame++] : 0.0f) * sample_gain;
            }
        } else if (model == SoundVerticesModel::Modal) {
            if (auto *recording = r.try_get<Recording>(entity)) {
                for (uint32_t i = 0; i < frame_count && !recording->Complete(); ++i) recording->Record(output[i]);
            }
        }
    }

    if (auto *capture = r.ctx().find<MasterCapture>(); capture && !capture->Ring.empty()) {
        const auto capacity = uint64_t(capture->Ring.size());
        auto written = capture->Written.load(std::memory_order_relaxed);
        auto pos = size_t(written % capacity);
        for (uint32_t i = 0; i < frame_count; ++i, ++written) {
            capture->Ring[pos] = output[i];
            if (++pos == capacity) pos = 0;
        }
        capture->Written.store(written, std::memory_order_release);
    }

    // The monitor stage, after the capture tap: device units at the monitor level.
    if (monitor) MonitorFrames(r, {output, frame_count}, r.ctx().get<MonitorLimiter>());
}

using namespace ImGui;

/***** Sound object *****/

namespace {
constexpr ImVec2 ChartSize{-1, 160};

// If `normalize_max` is set, normalize the data to this maximum value.
void WriteWav(const std::vector<float> &frames, const fs::path &file_path, uint32_t sample_rate, std::optional<float> normalize_max = {}) {
    WavWriter writer{file_path, sample_rate};
    if (!writer.IsOpen()) throw std::runtime_error(std::format("Failed to open wav file {}", file_path.string()));
    const float mult = normalize_max ? *normalize_max / *max_element(frames) : 1.0f;
    const auto frames_normed = frames | transform([mult](float f) { return f * mult; }) | to<std::vector>();
    writer.Write(frames_normed);
}

void PlotFrames(const std::vector<float> &frames, std::string_view label = "Waveform", std::optional<uint> highlight_frame = {}) {
    if (ImPlot::BeginPlot(label.data(), ChartSize)) {
        ImPlot::SetupAxes("Frame", "Amplitude");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, frames.size(), ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1, ImGuiCond_Always);
        if (highlight_frame) {
            ImPlot::PlotInfLines("##Highlight", &*highlight_frame, 1, {ImPlotProp_LineColor, ImGui::GetStyleColorVec4(ImGuiCol_PlotLinesHovered)});
        }
        ImPlot::PlotLine("", frames.data(), frames.size());
        ImPlot::EndPlot();
    }
}

void PlotMagnitudeSpectrum(const std::vector<float> &frames, uint32_t sample_rate, std::string_view label = "Magnitude spectrum", std::optional<float> highlight_freq = {}) {
    static const std::vector<float> *frames_ptr{&frames};
    static FFTData fft{ComputeFft(frames, sample_rate)};
    if (&frames != frames_ptr) {
        fft = ComputeFft(frames, sample_rate);
        frames_ptr = &frames;
    }
    if (ImPlot::BeginPlot(label.data(), ChartSize)) {
        static constexpr float MinDb = -200;
        const uint32_t N = fft.NumReal, N2 = N / 2;
        const auto fs_n = float(sample_rate) / float(N);
        static std::vector<float> frequency(N2), magnitude(N2);
        frequency.resize(N2);
        magnitude.resize(N2);

        for (uint32_t i = 0; i < N2; i++) {
            frequency[i] = fs_n * float(i);
            magnitude[i] = 20.0f * log10f(std::abs(fft.Bins[i]) / float(N2));
        }

        ImPlot::SetupAxes("Frequency (Hz)", "Magnitude (dB)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, float(sample_rate) / 2, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, MinDb, 0, ImGuiCond_Always);
        if (highlight_freq) {
            ImPlot::PlotInfLines("##Highlight", &(*highlight_freq), 1, {ImPlotProp_LineColor, ImGui::GetStyleColorVec4(ImGuiCol_PlotLinesHovered)});
        }
        ImPlot::PlotShaded("", frequency.data(), magnitude.data(), N2, MinDb, {ImPlotProp_FillColor, ImGui::GetStyleColorVec4(ImGuiCol_PlotHistogramHovered)});
        ImPlot::EndPlot();
    }
}

// Returns the index of the hovered mode, if any.
std::optional<size_t> PlotModeData(
    const std::vector<float> &data, std::string_view label, std::string_view x_label, std::string_view y_label,
    std::optional<size_t> highlight_index = {}, std::optional<float> max_value_opt = {}
) {
    std::optional<size_t> hovered_index;
    if (ImPlot::BeginPlot(label.data(), ChartSize)) {
        static constexpr double BarSize = 0.9;
        const float max_value = max_value_opt.value_or(*std::max_element(data.begin(), data.end()));
        ImPlot::SetupAxes(x_label.data(), y_label.data());
        ImPlot::SetupAxesLimits(-0.5f, data.size() - 0.5f, 0, max_value, ImPlotCond_Always);
        if (ImPlot::IsPlotHovered()) {
            if (auto i = std::lround(ImPlot::GetPlotMousePos().x); i >= 0 && i < std::ssize(data)) hovered_index = i;
        }
        if (!highlight_index) {
            ImPlot::PlotBars("", data.data(), data.size(), BarSize);
        } else {
            for (size_t i = 0; i < data.size(); ++i) {
                ImPlot::PlotBars(i == *highlight_index ? "##0" : "", &data[i], 1, BarSize, i);
            }
        }
        ImPlot::EndPlot();
    }

    return hovered_index;
}

bool DrawModalModelActions(
    entt::registry &r, entt::entity viewport, entt::entity e, entt::entity mesh_entity,
    const ModalSolveSettings &settings, const AcousticMaterial &material
) {
    const bool present = r.all_of<ModalModes>(e);
    const bool solving = IsSolving(r, e);
    bool has_action = false;
    if (r.all_of<SoundVerticesModel>(e)) {
        if (Button(present ? "Delete sound object ●" : "Delete sound object")) {
            action::Emit(action::audio::DeleteSoundObject{});
            return true;
        }
        has_action = true;
    }
    if (!present) {
        if (has_action) SameLine();
        if (solving) BeginDisabled();
        if (Button("Create modal model ○")) {
            action::Emit(action::audio::ConfigureModalModel{settings, material});
            LaunchModalSolve(r, viewport, e, settings, material);
        }
        if (solving) EndDisabled();
    }
    if (present && !solving && r.all_of<ModalSolveSettings, AcousticMaterial>(e) &&
        ModalModelStale(r, e, BuildSolveInputs(r, e, mesh_entity, settings), material)) {
        if (has_action) SameLine();
        if (Button("Update modal model")) LaunchModalSolve(r, viewport, e, settings, material);
    }
    return false;
}

void DrawModalModelSettings(
    entt::registry &r, entt::entity e, entt::entity mesh_entity,
    const ModalSolveSettings &settings, const AcousticMaterial &material
) {
    const ContactSurface default_surface = WithPreset({}, surfaces::acoustic::Default);
    const auto &surface = r.all_of<ContactSurface>(e) ? r.get<const ContactSurface>(e) : default_surface;
    ui::Edit fs{r, e, ui::Replace{settings}};

    SeparatorText("Material properties");
    ui::PresetCombo("Presets", material.Name, materials::acoustic::All, [&](const auto &choice) {
        action::Emit(action::Replace<AcousticMaterial>{.Entity = e, .Value = choice});
    });
    using Props = AcousticMaterialProperties;
    ui::Edit fm{r, e, ui::Replace{material}};
    fm.Slider<&AcousticMaterial::Properties, &Props::Density>("Density (kg/m^3)", "%.0f");
    fm.Slider<&AcousticMaterial::Properties, &Props::YoungModulus>("Young's modulus (Pa)", "%.3g", ImGuiSliderFlags_Logarithmic);
    fm.Slider<&AcousticMaterial::Properties, &Props::PoissonRatio>("Poisson's ratio", "%.2f");
    fm.Run<&AcousticMaterial::Properties>([](Props &props) {
        using AlphaLimits = FieldLimits<&AcousticMaterial::Properties, &Props::Alpha>;
        using BetaLimits = FieldLimits<&AcousticMaterial::Properties, &Props::Beta>;
        double coefficients[]{props.Alpha, props.Beta * 1e6};
        if (!InputScalarN("Rayleigh damping alpha / beta (1/s, µs)", ImGuiDataType_Double, coefficients, 2, nullptr, nullptr, "%.3g")) return false;
        props.Alpha = std::clamp(coefficients[0], AlphaLimits::Min, AlphaLimits::Max);
        props.Beta = std::clamp(coefficients[1] * 1e-6, BetaLimits::Min, BetaLimits::Max);
        return true;
    });
    MeshEditor::HelpMarker("Mass-proportional alpha primarily damps low frequencies. Stiffness-proportional beta primarily damps high frequencies.");

    DrawContactSurfaceControls(r, e, surface, material);

    SeparatorText("Modes");
    fs.Slider<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::NumModes>("Retained modes");
    fs.Slider<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::NumFemModes>("FEM eigenpairs");
    MeshEditor::HelpMarker("The eigensolver computes this many eigenpairs before filtering the retained frequency band and mode count.");
    fs.Run<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal>([](ModalSolverConfig &config) {
        return DragFloatRange2("Frequency band (Hz)", &config.MinModeFreq, &config.MaxModeFreq, 1.f, 20.f, 20000.f, "%.0f", "%.0f");
    });
    fs.Slider<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::Tolerance>("Residual tolerance", "%.1e", ImGuiSliderFlags_Logarithmic);
    fs.Slider<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::MaxRestarts>("Restart limit");
    {
        const bool had_fundamental = settings.Solve.Modal.FundamentalFreq.has_value();
        bool enabled = had_fundamental;
        if (Checkbox("Fundamental override", &enabled)) {
            fs.Set<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::FundamentalFreq>(enabled ? std::optional{440.f} : std::nullopt);
        }
        if (had_fundamental) {
            SameLine();
            fs.Run<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal, &ModalSolverConfig::FundamentalFreq>([](std::optional<float> &fundamental) {
                return SliderFloat("##fundamental", &*fundamental, 20.f, 20000.f, "%.0f Hz", ImGuiSliderFlags_Logarithmic);
            });
        }
        MeshEditor::HelpMarker("Overrides the output fundamental. Without an override, an active impact recording supplies its estimated fundamental when available.");
    }

    SeparatorText("Discretization");
    int discretization = int(settings.Discretization);
    bool discretization_changed = RadioButton("Tet10", &discretization, int(fastfem::Discretization::Tet10));
    SameLine();
    discretization_changed |= RadioButton("Finite cell", &discretization, int(fastfem::Discretization::FiniteCell));
    if (discretization_changed) fs.Set<&ModalSolveSettings::Discretization>(fastfem::Discretization(discretization));

    if (settings.Discretization == fastfem::Discretization::Tet10) {
        SeparatorText("Tet10 mesh");
        fs.Enum<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Tetrahedralization, &TetConfig::Refinement>("Refinement", "None\0Quality\0Quality + resolution\0");
        MeshEditor::HelpMarker("None: basic tetrahedralization and repair. Quality: improve element shapes. Quality + resolution: also refine to the target resolution.");
        fs.Run<&ModalSolveSettings::Solve, &SurfaceSolveConfig::SurfaceSimplificationRatio>([](float &ratio) {
            using Limits = FieldLimits<&ModalSolveSettings::Solve, &SurfaceSolveConfig::SurfaceSimplificationRatio>;
            float percent = ratio * 100;
            if (!SliderFloat("Surface detail", &percent, float(Limits::Min * 100), float(Limits::Max * 100), "%.1f%%", ImGuiSliderFlags_AlwaysClamp)) return false;
            ratio = percent / 100;
            return true;
        },
                                                                                            /*delta_capable=*/true);
        MeshEditor::HelpMarker("Target percentage of surface triangles to retain before refinement. 100% keeps the original mesh.");
        if (settings.Solve.Tetrahedralization.Refinement == fastfem::TetRefinement::QualityAndResolution) {
            fs.Slider<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Resolution>("Target resolution", nullptr, ImGuiSliderFlags_AlwaysClamp);
            MeshEditor::HelpMarker("Target divisions along the object's longest scaled axis. Higher values request finer tetrahedra. Refines the surface triangles and tetrahedron volume together.");
        }
        fs.Run<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Tetrahedralization, &TetConfig::Holes>([](std::vector<fastfem::DVec3> &holes) {
            bool changed = false;
            for (size_t i = 0; i < holes.size(); ++i) {
                PushID(int(i));
                changed |= InputScalarN("Hole seed", ImGuiDataType_Double, &holes[i].x, 3);
                SameLine();
                if (Button("Remove")) {
                    holes.erase(holes.begin() + i);
                    changed = true;
                    PopID();
                    break;
                }
                PopID();
            }
            if (Button("Add hole seed")) {
                holes.emplace_back();
                changed = true;
            }
            return changed;
        });
        MeshEditor::HelpMarker("Tetrahedralization excludes the connected tetrahedral region containing each point, bounded by the input surface. Coordinates use scaled local space (m).");
    } else {
        SeparatorText("Finite-cell grid");
        fs.Slider<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Resolution>("Target resolution", nullptr, ImGuiSliderFlags_AlwaysClamp);
        MeshEditor::HelpMarker("Target divisions along the object's longest scaled axis. The other axes use the same target spacing.");
        fs.Slider<&ModalSolveSettings::Solve, &SurfaceSolveConfig::FiniteCell, &FiniteCellConfig::CutDepth>("Cut depth");
        fs.Slider<&ModalSolveSettings::Solve, &SurfaceSolveConfig::FiniteCell, &FiniteCellConfig::FictitiousScale>("Fictitious scale", "%.1e", ImGuiSliderFlags_Logarithmic);
        fs.Slider<&ModalSolveSettings::Solve, &SurfaceSolveConfig::FiniteCell, &FiniteCellConfig::PaddingCells>("Padding (cells)", "%.2f");
        fs.Run<&ModalSolveSettings::Solve, &SurfaceSolveConfig::FiniteCell, &FiniteCellConfig::GridOffsetCells>([](fastfem::DVec3 &offset) {
            return InputScalarN("Grid offset (cells)", ImGuiDataType_Double, &offset.x, 3);
        });
    }

    SeparatorText("Excitation vertices");
    const uint32_t num_vertices = GetMesh(r, mesh_entity).VertexCount();
    const bool has_excitable = r.all_of<SoundVertices>(e);
    const bool reuse = has_excitable && settings.CopySoundVertices;
    if (has_excitable) {
        PushID("ExcitationVertices");
        int source = reuse ? 0 : 1;
        bool source_changed = RadioButton("Reuse existing", &source, 0);
        SameLine();
        source_changed |= RadioButton("Evenly spaced", &source, 1);
        PopID();
        if (source_changed) fs.Set<&ModalSolveSettings::CopySoundVertices>(source == 0);
        MeshEditor::HelpMarker("Reuse existing: Solve at the object's current excitation vertices.\nEvenly spaced: Solve at new positions spread over the mesh.");
    }
    if (reuse) BeginDisabled();
    const uint32_t min_vertices = 1, max_vertices = num_vertices;
    // Reuse shows the actual count of existing excitation vertices, else the editable target count.
    if (uint32_t v = reuse ? r.get<const SoundVertices>(e).Vertices.Count : std::clamp(settings.NumVertices, 1u, num_vertices);
        SliderScalar("Count", ImGuiDataType_U32, &v, &min_vertices, &max_vertices))
        fs.Set<&ModalSolveSettings::NumVertices>(v);
    if (reuse) EndDisabled();
}

// Returns the active vertex in Excite mode or selected vertices in Edit mode.
std::vector<uint32_t> GetSampleOpVertices(const entt::registry &r, entt::entity viewport, entt::entity sound_entity) {
    if (!r.valid(sound_entity)) return {};
    const auto *inst = r.try_get<const Instance>(sound_entity);
    if (!inst) return {};
    const auto mesh_entity = inst->Entity;
    const auto mesh = TryGetMesh(r, mesh_entity);
    if (!mesh) return {};

    const auto mode = r.get<const Interaction>(viewport).Mode;
    if (mode == InteractionMode::Excite) {
        if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity)) return {active->Handle};
        return {};
    }
    if (mode != InteractionMode::Edit || !r.all_of<MeshElementSelection>(mesh_entity)) return {};

    const auto bits = r.ctx().get<const MeshStore>().GetSelectionBits(mesh->GetStoreId(), Element::Vertex);
    std::vector<uint32_t> vertices;
    selection::ForEachSelected(bits, mesh->VertexCount(), [&](uint32_t vertex) { vertices.push_back(vertex); });
    return vertices;
}

// Circular pad returning a position in the unit disk (center = zero). Drag to set, right-click recenters.
bool ImpulseJoystick(vec2 &pos) {
    constexpr float radius{32.f};
    const auto p0 = GetCursorScreenPos();
    InvisibleButton("impulse", {radius * 2, radius * 2}, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const ImVec2 center{p0.x + radius, p0.y + radius};
    bool changed = false;
    if (IsItemActive() && IsMouseDown(ImGuiMouseButton_Left)) {
        const auto m = GetIO().MousePos;
        pos = {(m.x - center.x) / radius, -(m.y - center.y) / radius};
        if (const float len = numeric::Length(pos); len > 1.f) pos /= len;
        changed = true;
    } else if (IsItemClicked(ImGuiMouseButton_Right)) {
        pos = {0, 0};
        changed = true;
    }
    auto &dl = *GetWindowDrawList();
    dl.AddCircleFilled(center, radius, GetColorU32(ImGuiCol_FrameBg));
    dl.AddCircle(center, radius, GetColorU32(ImGuiCol_Border));
    dl.AddCircleFilled({center.x + pos.x * radius, center.y - pos.y * radius}, 4.f, GetColorU32(IsItemActive() ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab));
    return changed;
}
} // namespace

void DrawObjectAudioControls(entt::registry &r, entt::entity viewport, entt::entity e, entt::entity mesh_entity) {
    if (e == entt::null || mesh_entity == entt::null) return;

    const ModalSolveSettings default_settings;
    const auto &settings = r.all_of<ModalSolveSettings>(e) ? r.get<const ModalSolveSettings>(e) : default_settings;
    const auto &material = r.all_of<AcousticMaterial>(e) ? r.get<const AcousticMaterial>(e) : materials::acoustic::All.front();
    if (DrawModalModelActions(r, viewport, e, mesh_entity, settings, material)) return;

    // Sample ops (Add/Replace/Remove) are only available in Edit / Excite mode.
    const auto mode = r.get<const Interaction>(viewport).Mode;
    const bool sample_ops_available = mode == InteractionMode::Edit || mode == InteractionMode::Excite;
    const auto op_vertices = sample_ops_available ? GetSampleOpVertices(r, viewport, e) : std::vector<uint32_t>{};

    const bool has_model = r.all_of<SoundVerticesModel>(e);
    if (!has_model) DrawModalModelSettings(r, e, mesh_entity, settings, material);

    const auto *samples = r.try_get<const VertexSamples>(e);
    const auto *modal_modes = r.try_get<const ModalModes>(e);
    const auto *excitable = r.try_get<const SoundVertices>(e);
    auto model = has_model ? r.get<SoundVerticesModel>(e) : SoundVerticesModel::Samples;
    const auto *recording = r.try_get<const Recording>(e);
    const uint32_t active_vi = excitable ? GetActiveVertexIndex(r, e) : 0;
    const auto sample_rate = DeviceSampleRate(r); // for the spectrum plots below

    if (samples && modal_modes) {
        PushID("SelectAudioModel");
        auto edit_model = int(model);
        bool model_changed = RadioButton("Recordings", &edit_model, int(SoundVerticesModel::Samples));
        SameLine();
        model_changed |= RadioButton("Modal", &edit_model, int(SoundVerticesModel::Modal));
        PopID();
        if (model_changed) {
            model = SoundVerticesModel(edit_model);
            action::Emit(action::audio::SetModel{model});
        }
    }

    if (has_model && excitable) {
        const auto excitable_vertices = r.ctx().get<const MeshStore>().GetSoundVertices(excitable->Vertices);
        const auto active_vertex = excitable_vertices[active_vi];
        const bool can_excite =
            (model == SoundVerticesModel::Samples) ||
            (model == SoundVerticesModel::Modal && (!recording || recording->Complete()));
        if (!can_excite) BeginDisabled();
        Button("Excite");
        if (IsItemActivated()) action::Emit(action::audio::StartExcite{active_vertex});
        else if (IsItemDeactivated()) action::Emit(action::audio::StopExcite{});
        if (!can_excite) EndDisabled();
        SameLine();
        if (BeginCombo("Vertex", std::to_string(active_vertex).c_str())) {
            for (uint32_t vi = 0; vi < excitable_vertices.size(); ++vi) {
                if (const auto vertex = excitable_vertices[vi]; Selectable(std::to_string(vertex).c_str(), vi == active_vi))
                    action::Emit(action::audio::SetExciteVertex{vi, vertex});
            }
            EndCombo();
        }

        if (model == SoundVerticesModel::Modal) {
            TextUnformatted("Impact angle");
            SameLine();
            MeshEditor::HelpMarker("Strike direction relative to the surface.\nCenter hits perpendicular to the surface. Edge hits tangent to the surface.\nRight-click to recenter.");
            ImpulseJoystick(ImpulseAngle);
        }
    }

    // Sample ops + waveform (rendered when in Samples mode or when no model exists yet).
    if (model == SoundVerticesModel::Samples) {
        if (has_model) SeparatorText("Sound samples");
        if (sample_ops_available) {
            std::vector<uint32_t> op_with_sample;
            if (samples) {
                for (const uint32_t mv : op_vertices) {
                    if (samples->PathByVertex.contains(mv)) op_with_sample.push_back(mv);
                }
            }
            const auto n = op_vertices.size(), with_sample = op_with_sample.size();
            if (n == 0) BeginDisabled();
            if (const auto assign_label = n > 1 ? std::format("Assign sample to {} vertices…", n) : std::string{with_sample ? "Replace sample…" : "Assign sample…"};
                Button(assign_label.c_str())) {
                FileDialog::ShowOpen("wav;mp3;flac;ogg;opus", [verts = op_vertices](const fs::path &path) mutable {
                    action::Emit(action::audio::AssignVertexSamples{std::move(verts), path});
                });
            }
            if (n == 0) EndDisabled();
            if (with_sample > 0) {
                SameLine();
                if (const auto remove_label = with_sample > 1 ? std::format("Remove {} samples", with_sample) : std::string{"Remove sample"};
                    Button(remove_label.c_str())) {
                    action::Emit(action::audio::RemoveVertexSamples{std::move(op_with_sample)});
                    return;
                }
            }
        }
        if (const auto path = ActiveSamplePath(r, e)) {
            const auto &frames = GetSampleFrames(r, viewport, *path);
            if (!frames.empty()) {
                PlotFrames(frames, "Waveform", samples->Stopped ? std::optional<uint>{} : std::optional{samples->Frame});
                PlotMagnitudeSpectrum(frames, sample_rate, "Spectrum");
            }
        }
    }

    if (!has_model) return;

    DrawModalModelSettings(r, e, mesh_entity, settings, material);

    if (model != SoundVerticesModel::Modal) return;

    if (!excitable || !modal_modes) return;

    static std::optional<size_t> hovered_mode_index;
    const auto &modes = *modal_modes;
    if (recording && recording->Complete()) {
        const auto &frames = recording->Frames;
        PlotFrames(frames, "Modal impact waveform");
        const auto highlight_freq = hovered_mode_index ? std::optional{modes.Freqs[*hovered_mode_index]} : std::nullopt;
        PlotMagnitudeSpectrum(frames, sample_rate, "Modal impact spectrum", highlight_freq);
    }

    if (CollapsingHeader("Modal data charts")) {
        std::optional<size_t> new_hovered_index;
        if (auto hovered = PlotModeData(modes.Freqs, "Mode frequencies", "", "Frequency (Hz)", hovered_mode_index)) new_hovered_index = hovered;
        if (auto hovered = PlotModeData(modes.T60s, "Mode T60s", "", "T60 decay time (s)", hovered_mode_index)) new_hovered_index = hovered;
        const auto active_gains = [&]() -> std::vector<float> {
            if (active_vi >= modes.Shapes.size()) return {};
            const auto j = TiltAlongNormal(VertexNormal(GetMesh(r, mesh_entity), r.ctx().get<const MeshStore>().GetSoundVertices(excitable->Vertices)[active_vi]), ImpulseAngle);
            return modes.Shapes[active_vi] | transform([&](const vec3 &s) { return std::abs(numeric::Dot(s, j)); }) | to<std::vector<float>>();
        }();
        if (!active_gains.empty()) {
            if (auto hovered = PlotModeData(active_gains, "Mode gains", "Mode index", "Gain", hovered_mode_index)) new_hovered_index = hovered;
        }
        if (hovered_mode_index = new_hovered_index; hovered_mode_index && *hovered_mode_index < modes.Freqs.size()) {
            const auto index = *hovered_mode_index;
            Text(
                "Mode %lu: Freq (scaled) %.2f Hz, Freq (FEM) %.2f, T60 %.2f s, Gain %.4f", index,
                modes.Freqs[index],
                modes.Freqs[index] * modes.OriginalFundamentalFreq / modes.Freqs[0],
                modes.T60s[index],
                index < active_gains.size() ? active_gains[index] : 0.f
            );
        }
    }

    if (CollapsingHeader("Synthesis")) {
        ui::Edit fe{r, e};
        fe.Slider<&ModalGain::Value>("Gain");
        fe.Drag<&ModalTuning::FundamentalFreq>("Fundamental (Hz)", 1.f, "%.1f");
        fe.Slider<&ModalTuning::T60Scale>("T60 scale");
    }

    const bool is_recording = recording && !recording->Complete();
    if (is_recording) BeginDisabled();
    static constexpr uint32_t RecordFrames = 208'592; // Same length as RealImpact recordings.
    if (Button("Record strike")) action::Emit(action::audio::StartRecording{RecordFrames});
    if (is_recording) EndDisabled();

    if (samples && recording && recording->Complete()) {
        // const auto &modal_fft = ..., &impact_fft = ...;
        // uint32_t ModeCount() const { return modes.Freqs.size(); }
        // const uint32_t n_test_modes = std::min(ModeCount(), 10u);
        // Uncomment to cache `n_test_modes` peak frequencies for display in the spectrum plot.
        // RMSE is abyssmal in most cases...
        // const float rmse = RMSE(GetPeakFrequencies(modal_fft, n_test_modes), GetPeakFrequencies(impact_fft, n_test_modes));
        // Text("RMSE of top %d mode frequencies: %f", n_test_modes, rmse);
        SameLine();
        if (Button("Save wav files")) {
            const auto name = GetName(r, e);
            static const auto WavOutDir = fs::path{".."} / "audio_samples";
            const auto sr = DeviceSampleRate(r);
            WriteWav(recording->Frames, WavOutDir / std::format("{}-modal", name), sr);
            if (const auto path = ActiveSamplePath(r, e)) {
                WriteWav(GetSampleFrames(r, viewport, *path), WavOutDir / std::format("{}-impact", name), sr);
            }
        }
    }
}

void RemoveAudioComponents(entt::registry &r, entt::entity e) {
    CancelModalSolves(r, e);
    r.remove<ScaleLocked, SoundVertices, Recording, SoundVerticesModel, ModalModes, ModalGain, ModalTuning, MassProperties, ContactDynamics, ModalEigenSummary, VertexSamples, ModalSolveSettings, RealImpactActiveMicrophone, RealImpactVertices>(e);
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

void DrawGlobalSynthControls(entt::registry &r, entt::entity viewport) {
    ui::Edit f{r, viewport};
    if (!r.view<const ModalModes>().empty() && CollapsingHeader("Modal synthesis", ImGuiTreeNodeFlags_DefaultOpen)) {
        f.Slider<&ModalSoundControls::RenderThreads>("Render threads");
        MeshEditor::HelpMarker("Objects render independently, so a scene of many ringing ones scales on this.");
        f.Slider<&ModalSoundControls::MaxImpacts>("Max impacts");
        MeshEditor::HelpMarker("Cap on simultaneous in-flight contact pulses.");
        f.Slider<&ModalSoundControls::ModalLevel>("Modal gain");
        MeshEditor::HelpMarker("Gain on every modal object's resonator output.");
        f.Slider<&ModalSoundControls::ClickGain>("Click");
        MeshEditor::HelpMarker("Level of the rigid-body acceleration-noise click.");
        f.Slider<&ModalSoundControls::MinContactExcitation>("Min contact excitation", "%.3g", ImGuiSliderFlags_Logarithmic);
        f.Slider<&ModalSoundControls::MinContactSpeed>("Min contact speed", "%.3f");
        MeshEditor::HelpMarker("A physics collision sounds only above both floors: the loudest mode its impulse starts ringing, and its approach speed.");

        DrawSurfaceSynthControls(r, viewport);

        SeparatorText("Striker");
        const auto &striker = r.get<const Striker>(viewport);
        // Replace the entire striker because its material contains a string.
        ui::PresetCombo("Material", striker.Material.Name, materials::acoustic::All, [&](const auto &choice) {
            action::Emit(action::Replace<Striker>{.Entity = viewport, .Value = {choice, striker.TipRadius, striker.Length}});
        });
        f.Slider<&Striker::TipRadius>("Tip radius (m)", "%.4f");
        f.Slider<&Striker::Length>("Length (m)", "%.3f");
        Text("Mass: %.3g kg", StrikerMass(striker));
        MeshEditor::HelpMarker("The mallet that strikes objects. A harder material or lighter capsule brightens the contact, and the tip radius sets its curvature.");
    }
    if (!r.view<const VertexSamples>().empty() && CollapsingHeader("Samples", ImGuiTreeNodeFlags_DefaultOpen)) {
        f.Slider<&ModalSoundControls::SampleGain>("Sample gain");
        MeshEditor::HelpMarker("Level of impact-sample playback.");
    }
}

void DrawAudioDebug(const entt::registry &r) {
    const auto &m = r.ctx().get<const ModalAudio>();
    const auto &bank = *m.Live;

    SeparatorText("Device");
    if (const auto *device = r.ctx().find<AudioDeviceResource>(); device && device->Initialized) {
        Text("%s at %u Hz", device->DeviceName.empty() ? "System default" : device->DeviceName.c_str(), device->SampleRate);
    } else {
        TextUnformatted("No output device");
    }

    SeparatorText("Modal bank");
    Text("Objects: %zu, modes: %zu", bank.Entities.size(), bank.CoeffRe.size());
    Text("Impacts: %u / %u", m.ActiveImpacts.load(std::memory_order_relaxed), m.MaxImpacts.load(std::memory_order_relaxed));
    Text("Energy: %.3g J, peak %.3g J", m.ModalEnergy.load(std::memory_order_relaxed), m.PeakModalEnergy.load(std::memory_order_relaxed));
    MeshEditor::HelpMarker("Mechanical energy standing in the mode banks. A passive scene loses it between strikes, so a peak that climbs while nothing strikes is a channel feeding the modes rather than damping them.");

    SeparatorText("Render");
    const auto share = m.RenderShare.load(std::memory_order_relaxed);
    Text("Block: %.2f ms (%.0f%% of budget)", m.RenderSeconds.load(std::memory_order_relaxed) * 1e3f, share * 100);
    Text("Worst: %.0f%%", m.PeakRenderShare.load(std::memory_order_relaxed) * 100);
    MeshEditor::HelpMarker("What a block costs against the time it has. Past 100% the device underruns.");

    SeparatorText("Event queue");
    const auto queued = m.EventWrite.load(std::memory_order_relaxed) - m.EventRead.load(std::memory_order_acquire);
    Text("Queued: %u / %u", queued, ModalAudio::EventCapacity);
    Text("Dropped: %llu", m.EventsDropped);
    MeshEditor::HelpMarker("Events the queue had no room for, each one a strike or contact report the bank never saw.");

    DrawSurfaceContactDebug(r);
}

std::string_view SolveStageLabel(fastfem::SolveStage stage) {
    using enum fastfem::SolveStage;
    switch (stage) {
        case PreparingSurface: return "Preparing surface";
        case GeneratingTetrahedra: return "Generating tetrahedra";
        case BuildingFiniteCellGrid: return "Building finite-cell grid";
        case ComputingMassProperties: return "Computing mass properties";
        case BuildingTopology: return "Building topology";
        case AssemblingOperators: return "Assembling operators";
        case Factorizing: return "Factorizing";
        case SolvingEigenproblem: return "Solving eigenproblem";
        case SamplingModes: return "Sampling modes";
        case Finalizing: return "Finalizing modal model";
        case Complete: return "Complete";
    }
}

void DrawModalJobsOverlay(entt::registry &r) {
    const auto &jobs = r.ctx().get<const ModalSolveJobs>().Jobs;
    if (jobs.empty()) return;

    constexpr float Pad{12.f};
    const auto anchor = GetWindowPos() + ImVec2{Pad, GetWindowSize().y - Pad};
    SetNextWindowPos(anchor, ImGuiCond_Always, {0.f, 1.f});
    SetNextWindowBgAlpha(0.85f);
    // The viewport window zeroes its window padding, so restore normal padding for the overlay.
    // Vertical padding matches the item spacing, so each row sits evenly between the window edges and its bar.
    PushStyleVar(ImGuiStyleVar_WindowPadding, {10.f, 6.f});
    PushStyleVar(ImGuiStyleVar_WindowRounding, 6.f);
    PushStyleVar(ImGuiStyleVar_ItemSpacing, {GetStyle().ItemSpacing.x, 6.f});
    PushStyleVar(ImGuiStyleVar_FramePadding, {8.f, 3.f});
    constexpr ImGuiWindowFlags OverlayFlags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove;
    if (Begin("Modal solve jobs", nullptr, OverlayFlags)) {
        // Keep the overlay above the focused viewport window.
        BringWindowToDisplayFront(GetCurrentWindow());
        for (const auto &job : jobs) {
            auto &monitor = *job->Work.Monitor;
            PushID(job.get());
            BeginGroup();
            AlignTextToFramePadding();
            ImSpinner::SpinnerRotateSegmentsPulsar("##spinner", GetTextLineHeight() * 0.5f, 2.f, GetColorU32(ImGuiCol_Text), 1.1f, 3, 3);
            SameLine();
            const auto stage = SolveStageLabel(monitor.Stage.load(std::memory_order_relaxed));
            Text("%s: %.*s", job->Work.Title.c_str(), int(stage.size()), stage.data());
            SameLine(0.f, GetStyle().ItemSpacing.x * 3.f);
            const bool cancelled = job->Work.Cancelled();
            if (cancelled) BeginDisabled();
            if (Button("Cancel")) job->Work.RequestCancel();
            if (cancelled) EndDisabled();
            EndGroup();
            const float progress = monitor.Progress.load(std::memory_order_relaxed);
            ProgressBar(progress > 0.f ? progress : -float(GetTime()), {GetItemRectSize().x, 3.f}, "");
            PopID();
        }
    }
    End();
    PopStyleVar(4);
}
