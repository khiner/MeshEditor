#include "SurfaceAudio.h"

#include "TransformMath.h"
#include "action/Action.h"
#include "audio/ContactScene.h"
#include "audio/ModalAudio.h"
#include "audio/ModalModes.h"
#include "scene/WorldTransform.h"
#include "ui/FieldEdit.h"
#include "ui/PresetCombo.h"

#include <entt/entity/registry.hpp>

#include <atomic>
#include <bit>
#include <optional>

#include "ui/HelpMarker.h" // depends on imgui

// The surface-contact model's controls: an object's acoustic surface, the viewport's sustained-contact levels, and how its pools stand against demand.

// Surface finish. Ranges cover the finish presets with headroom (see surfaces::acoustic::All).
template<> struct FieldLimits<&ContactSurface::Roughness> : Within<1e-9, 1e-2> {};
template<> struct FieldLimits<&ContactSurface::CorrelationLength> : Within<1e-7, 1e-1> {};
template<> struct FieldLimits<&ContactSurface::SpectralSlope> : Within<-3., 0.> {};

template<> struct FieldLimits<&SurfaceSoundControls::MaxVoices> : Within<1., 64.> {};
template<> struct FieldLimits<&SurfaceSoundControls::SustainLevel> : Within<0., 4.> {};
template<> struct FieldLimits<&SurfaceSoundControls::AccelNoiseGain> : Within<0., 4.> {};
template<> struct FieldLimits<&SurfaceSoundControls::Coupling> : Within<0., 4.> {};
template<> struct FieldLimits<&SurfaceSoundControls::ContactDamping> : Within<0., 10.> {};
template<> struct FieldLimits<&SurfaceSoundControls::MinSlipSpeed> : Within<0., 1.> {};
template<> struct FieldLimits<&SurfaceSoundControls::MinSweepSpeed> : Within<0., 1.> {};

using namespace ImGui;

void DrawContactSurfaceControls(entt::registry &r, entt::entity e) {
    const auto *surface = r.try_get<const ContactSurface>(e);
    const auto *material = r.try_get<const AcousticMaterial>(e);
    if (!surface || !material) return;

    SeparatorText("Surface finish");
    ui::PresetCombo("Presets##finish", surface->Name, surfaces::acoustic::All, [&](const auto &choice) {
        action::Emit(action::Replace<ContactSurface>{.Entity = e, .Value = WithPreset(*surface, choice)});
    });
    ui::Edit fsurf{r, e};
    fsurf.Slider<&ContactSurface::Roughness>("Roughness (m)", "%.3g", ImGuiSliderFlags_Logarithmic);
    MeshEditor::HelpMarker("Root-mean-square asperity height. A physical length measured with a profilometer, unrelated to a render material's roughness.");
    fsurf.Slider<&ContactSurface::CorrelationLength>("Correlation length (m)", "%.3g", ImGuiSliderFlags_Logarithmic);
    MeshEditor::HelpMarker("Lateral asperity spacing. With roughness it fixes the surface gradient a contact bears on.");
    fsurf.Slider<&ContactSurface::SpectralSlope>("Spectral slope", "%.2f");
    MeshEditor::HelpMarker("Exponent of the roughness power spectrum. More negative is smoother-sounding.");

    // Derived contact constants, for a contact against a like body at the object's own weight.
    const auto &props = material->Properties;
    const auto *cd = r.try_get<const ContactDynamics>(e);
    const auto *modes = r.try_get<const ModalModes>(e);
    const auto sample_curvature = modes && !modes->Positions.empty() ? SurfaceCurvature(r, e, TransformPoint(r.get<const WorldTransform>(e), modes->Positions.front())) : std::nullopt;
    const double curvature = CombinedCurvature(sample_curvature.value_or(0.0), 0.0);
    const double inv_modulus = InvEffectiveModulus(props, props);
    const double load = (cd ? cd->Mass : 0.0) * 9.81;
    const double patch_radius = ContactPatchRadius(load, inv_modulus, curvature);
    Text("Stiffness %.3g N/m^1.5, patch radius %.3g um", ContactStiffness(inv_modulus, curvature), patch_radius * 1e6);
    Text("Contact filter %.0f Hz at 0.1 m/s", patch_radius > 0 ? 0.1 / (2 * patch_radius) : 0.0);
    MeshEditor::HelpMarker("A patch of radius a cannot resolve wavelengths below about 2a, so the cutoff tracks the sweep speed and softens under heavier load.");
}

void DrawSurfaceSynthControls(entt::registry &r, entt::entity viewport) {
    ui::Edit f{r, viewport};
    SeparatorText("Sustained contact");
    f.Slider<&SurfaceSoundControls::MaxVoices>("Max voices");
    MeshEditor::HelpMarker("Cap on simultaneous sustained-contact voices.");
    f.Slider<&SurfaceSoundControls::SustainLevel>("Level##sustain");
    MeshEditor::HelpMarker("Level of the force a persisting contact drives the modes with as the two surfaces slide over one another.");
    f.Slider<&SurfaceSoundControls::AccelNoiseGain>("Acceleration noise");
    MeshEditor::HelpMarker("Level of the acceleration noise a body's rigid recoil radiates while a contact persists. Zero disables it.");
    f.Slider<&SurfaceSoundControls::Coupling>("Coupling");
    MeshEditor::HelpMarker("How much of the object's vibration modulates the contact separation. Coupling produces micro-collisions and chatter that an open-loop force cannot.");
    f.Slider<&SurfaceSoundControls::ContactDamping>("Contact damping");
    MeshEditor::HelpMarker("Scale on the Hunt-Crossley dissipation the pair's restitution implies.");
    f.Slider<&SurfaceSoundControls::MinSlipSpeed>("Min slip speed", "%.3f");
    f.Slider<&SurfaceSoundControls::MinSweepSpeed>("Min sweep speed", "%.3f");
    MeshEditor::HelpMarker("A persisting contact sounds only once its slip or either sweep speed clears its floor. Slip is how fast the surfaces slide, and sweep is how fast the contact travels over each surface, so a rolling body has sweep without slip.");
    f.Check<&SurfaceSoundControls::MuteGeometricDrive>("Mute geometric drive");
    f.Check<&SurfaceSoundControls::MuteFrictionDrive>("Mute friction drive");
    MeshEditor::HelpMarker("Silence one modal drive row at a time, which isolates a feedback loop through the modes to a row. The body feed and every other channel stay live.");
    Text("Active voices: %u", SurfaceActiveVoices(r.ctx().get<const ModalAudio>()));
}

void DrawSurfaceContactDebug(const entt::registry &r) {
    const auto &surface = Surface(r.ctx().get<const ModalAudio>());

    SeparatorText("Sustained contact");
    Text("Voices: %u / %u", r.ctx().get<const ModalAudio>().ActiveVoices.load(std::memory_order_relaxed), SurfaceControls(r).MaxVoices);
    Text("Refused: %llu", surface.VoicesRefused);
    MeshEditor::HelpMarker("Contacts the voice cap had no room for, each a body pressing on another that stays silent.");

    SeparatorText("Surface tracks");
    size_t filled = 0, bytes = 0;
    for (const auto &slot : surface.SurfaceTracks) {
        if (!slot.Owned) continue;
        ++filled;
        bytes += (slot.Owned->Heights.size() + slot.Owned->Sum.size()) * sizeof(float);
    }
    Text("Slots: %zu / %u (%.1f MB)", filled, SurfaceAudioState::MaxSurfaceTracks, double(bytes) / (1024 * 1024));
    MeshEditor::HelpMarker("Surfaces with the same finish share one slot. A slot keeps its track after its voices end, so the pool fills to what the scene has needed and stays there.");
    Text("Named by voices: %d", std::popcount(surface.VoiceTrackMask.load(std::memory_order_relaxed)));
    MeshEditor::HelpMarker("Slots the audio thread's voices are reading. The rest are free to take over when a new track needs the space.");
    Text("Refused: %llu", surface.SurfaceTracksRefused);
    MeshEditor::HelpMarker("Requests the pool had no room for. Each is one surface's finish or relief, and without its track that surface sounds perfectly smooth.");

    SeparatorText("Contact meter");
    if (const auto rate = surface.ContactShockRate.load(std::memory_order_relaxed); rate > 0) {
        Text("Shocks: %.4g /s", rate);
        Text("Free: %.3f, bearing: %.3f", surface.ContactFreeShare.load(std::memory_order_relaxed), surface.ContactBearingShare.load(std::memory_order_relaxed));
    } else {
        TextUnformatted("Off");
    }
    MeshEditor::HelpMarker("How often a cell takes up load again after parting, the share of samples the contact stood free, and the share of cell-samples that bore. Counting them costs an atomic per cell per sample, so the meter runs only under CONTACT_REPORT.");
}
