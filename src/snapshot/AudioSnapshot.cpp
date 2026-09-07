#include "audio/AcousticMaterial.h"
#include "audio/AudioSamples.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/ModalEigenSummary.h"
#include "audio/ModalModes.h"
#include "audio/SoundVertices.h"
#include "physics/PhysicsContact.h"
#include "snapshot/SnapshotRegistration.h"
#ifdef SURFACE_AUDIO
#include "audio/surface/SurfaceAudio.h"
#endif

namespace snapshot::detail {
template<> inline constexpr bool ForceFieldwise<AudioOutputMix> = true;
template<> inline constexpr bool ForceFieldwise<ModalSolveSettings> = true;

void RegisterAudio(Tables &tables) {
    Persistent<
        AudioOutputConfig, AudioOutputMix, Striker, ModalSoundControls, ContactSurface, SurfaceSoundControls,
        AcousticMaterial, VertexSamples, SoundVerticesModel, ModalModes, ModalGain, ModalTuning, ModalSolveSettings,
        MassProperties, ModalEigenSummary>(tables);
#ifdef SURFACE_AUDIO
    Derived<SurfaceRelief, SurfaceFinishKey>(tables);
#endif
    Derived<SoundVertices, SamplePlayback, ContactDynamics, ReportContacts>(tables);
}
} // namespace snapshot::detail
