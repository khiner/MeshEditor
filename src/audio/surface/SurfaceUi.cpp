#include "SurfaceAudio.h"

#include "action/Action.h"
#include "audio/ContactScene.h"
#include "ui/ChoiceCombo.h"
#include "ui/FieldEdit.h"

template<> inline constexpr field::SpecTable<ContactSurface> Specs<ContactSurface>{
    {&ContactSurface::Roughness, {.Min = 1e-9, .Max = 1e-2, .Unit = FieldUnit::Meters, .Scale = FieldScale::Logarithmic, .Label = "Roughness", .Format = "%.3g", .Help = "Root-mean-square asperity height. Independent of render material roughness."}},
    {&ContactSurface::CorrelationLength, {.Min = 1e-7, .Max = 1e-1, .Unit = FieldUnit::Meters, .Scale = FieldScale::Logarithmic, .Label = "Correlation length", .Format = "%.3g", .Help = "Lateral asperity spacing. Sets the surface slope at a contact together with roughness."}},
    {&ContactSurface::SpectralSlope, {.Min = -3, .Max = 0, .Help = "Exponent of the one-dimensional roughness power spectrum. More negative values sound smoother."}},
    {&ContactSurface::ShortWavelength, {.Min = 1e-8, .Max = 1e-2, .Unit = FieldUnit::Meters, .Scale = FieldScale::Logarithmic, .Label = "Short wavelength", .Format = "%.3g", .Help = "Shortest wavelength in the synthesized roughness spectrum"}},
    {&ContactSurface::Waviness, {.Min = 0, .Max = 1e-2, .Unit = FieldUnit::Meters, .Scale = FieldScale::Logarithmic, .Label = "Waviness", .Format = "%.3g", .Help = "Root-mean-square height of long-wavelength deviation from the nominal shape. Zero disables it."}},
    {&ContactSurface::WavinessLength, {.Min = 1e-5, .Max = 1, .Unit = FieldUnit::Meters, .Scale = FieldScale::Logarithmic, .Label = "Waviness length", .Format = "%.3g", .Help = "Lateral scale of the waviness"}},
};

template<> inline constexpr field::SpecTable<SurfaceSoundControls> Specs<SurfaceSoundControls>{
    {&SurfaceSoundControls::MaxVoices, {.Min = 1, .Max = 64, .Help = "Maximum number of sustained-contact voices playing at once"}},
    {&SurfaceSoundControls::SustainLevel, {.Min = 0, .Max = 4, .Label = "Sustain level", .Help = "Level of the modal drive force from surfaces sliding in persistent contact"}},
    {&SurfaceSoundControls::AccelNoiseGain, {.Min = 0, .Max = 4, .Label = "Acceleration noise", .Help = "Level of acceleration noise from rigid-body recoil during persistent contact. Zero disables it."}},
    {&SurfaceSoundControls::Coupling, {.Min = 0, .Max = 4, .Help = "Object vibration fed back into the contact separation. Higher values add micro-collisions and chatter."}},
    {&SurfaceSoundControls::ContactDamping, {.Min = 0, .Max = 10, .Help = "Scale on the Hunt-Crossley dissipation implied by the pair's restitution"}},
    {&SurfaceSoundControls::MinSlipSpeed, {.Min = 0, .Max = 1, .Unit = FieldUnit::MetersPerSecond, .Label = "Min slip speed", .Help = "Sliding speed between the surfaces above which a persisting contact sounds"}},
    {&SurfaceSoundControls::MinSweepSpeed, {.Min = 0, .Max = 1, .Unit = FieldUnit::MetersPerSecond, .Label = "Min sweep speed", .Help = "Speed of the contact across each surface above which a persisting contact sounds. Rolling has sweep without slip."}},
    {&SurfaceSoundControls::MuteGeometricDrive, {.Help = "Silence the geometric modal drive row to isolate feedback through the modes. Other channels stay active."}},
    {&SurfaceSoundControls::MuteFrictionDrive, {.Help = "Silence the friction modal drive row to isolate feedback through the modes. Other channels stay active."}},
};

using namespace ImGui;

void DrawContactSurfaceControls(state::Registry &r, state::Entity e, const ContactSurface &surface, const AcousticMaterial &material) {
    SeparatorText("Surface finish");
    ui::ChoiceCombo("Presets##finish", surface.Name, surfaces::acoustic::All | std::views::transform(&ContactSurfacePreset::Name), std::identity{}, [&](const std::string &name) {
        action::Emit(action::audio::SetSurfacePreset{name});
    });
    ui::DrawFields<ContactSurface, &ContactSurface::Name, &ContactSurface::Profile, &ContactSurface::SampleSpacing, &ContactSurface::NormalTexture>(ui::PatchEdit{surface});

    // Derive contact constants against an identical body under this object's weight.
    const auto &props = material.Properties;
    const auto *cd = r.try_get<const ContactDynamics>(e);
    const auto *modes = r.try_get<const ModalModes>(e);
    const auto sample_curvature = modes && !modes->Positions.empty() ? SurfaceCurvature(r, e, TransformPoint(*WorldTransformOf(r, e), modes->Positions.front())) : std::nullopt;
    const double curvature = CombinedCurvature(sample_curvature.value_or(0.0), 0.0);
    const double inv_modulus = InvEffectiveModulus(props, props);
    const double load = (cd ? cd->Mass : 0.0) * 9.81;
    const double patch_radius = ContactPatchRadius(load, inv_modulus, curvature);
    Text("Stiffness %.3g N/m^1.5, patch radius %.3g um", ContactStiffness(inv_modulus, curvature), patch_radius * 1e6);
    Text("Contact filter %.0f Hz at 0.1 m/s", patch_radius > 0 ? 0.1 / (2 * patch_radius) : 0.0);
    ui::Describe("Contact filter", "Contact patch cutoff frequency, the sweep speed over twice the patch radius. Heavier load widens the patch and lowers the cutoff.");
}

void DrawSurfaceSynthControls(state::Registry &r) {
    ui::Edit f{r, state::Document};
    SeparatorText("Sustained contact");
    ui::DrawFields<SurfaceSoundControls>(f);
    Text("Active voices: %u", SurfaceActiveVoices(r.Resources.get<const ModalAudio>()));
}

void DrawSurfaceContactDebug(const state::Registry &r) {
    const auto &surface = Surface(r.Resources.get<const ModalAudio>());

    SeparatorText("Sustained contact");
    Text("Voices: %u / %u", r.Resources.get<const ModalAudio>().ActiveVoices.load(std::memory_order_relaxed), SurfaceControls(r).MaxVoices);
    Text("Refused: %llu", surface.VoicesRefused);
    ui::Describe("Refused voices", "Contacts left silent because the voice cap was full");

    SeparatorText("Surface tracks");
    size_t filled = 0, bytes = 0;
    for (const auto &slot : surface.SurfaceTracks) {
        if (!slot.Owned) continue;
        ++filled;
        bytes += (slot.Owned->Heights.size() + slot.Owned->Sum.size()) * sizeof(float);
    }
    Text("Slots: %zu / %u (%.1f MB)", filled, SurfaceAudioState::MaxSurfaceTracks, double(bytes) / (1024 * 1024));
    ui::Describe("Surface track slots", "Surfaces with the same finish share one slot. Slots keep their tracks after their voices end.");
    Text("Named by voices: %d", std::popcount(surface.VoiceTrackMask.load(std::memory_order_relaxed)));
    ui::Describe("Slots named by voices", "Slots that active voices are reading. New tracks can reuse the other slots.");
    Text("Refused: %llu", surface.SurfaceTracksRefused);
    ui::Describe("Refused surface tracks", "Surface finish or relief tracks refused by a full pool. A surface without its track sounds perfectly smooth.");

    SeparatorText("Contact meter");
    static constexpr const char *MeterHelp{"Shocks per second as parted cells take up load again, the share of samples with the contact free, and the share of cell-samples bearing load. Requires the CONTACT_REPORT environment variable."};
    if (const auto rate = surface.ContactShockRate.load(std::memory_order_relaxed); rate > 0) {
        Text("Shocks: %.4g /s", rate);
        ui::Describe("Contact shocks", MeterHelp);
        Text("Free: %.3f, bearing: %.3f", surface.ContactFreeShare.load(std::memory_order_relaxed), surface.ContactBearingShare.load(std::memory_order_relaxed));
        ui::Describe("Contact shares", MeterHelp);
    } else {
        TextUnformatted("Off");
        ui::Describe("Contact meter", MeterHelp);
    }
}
