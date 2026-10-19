#include "audio/AudioSamples.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/ModalEigenSummary.h"
#include "audio/ModalModes.h"
#include "audio/RealImpactComponents.h"
#include "audio/SoundVertices.h"
#include "physics/PhysicsContact.h"
#include "snapshot/SnapshotRegistration.h"
#ifdef SURFACE_AUDIO
#include "audio/surface/SurfaceAudio.h"
#endif

namespace snapshot::detail {

void RegisterAudio(Tables &tables) {
    Classify(
        tables,
        TypeList<
            AudioOutputConfig, AudioOutputMix, Striker, ModalSoundControls, ContactSurface, SurfaceSoundControls,
            AcousticMaterial, VertexSamples, SoundVerticesModel, ModalModes, ModalGain, ModalTuning, ModalSolveSettings,
            MassProperties, ModalEigenSummary, RealImpactVertices, RealImpactMicrophone, RealImpactActiveMicrophone>{},
        TypeList<
#ifdef SURFACE_AUDIO
            SurfaceRelief, SurfaceFinishKey,
#endif
            SoundVertices, SamplePlayback, ContactDynamics, ReportContacts>{}
    );
    // The output device belongs to the workspace.
    tables.Snapshots[state::Type<AudioOutputConfig>()].History = false;
}
} // namespace snapshot::detail
