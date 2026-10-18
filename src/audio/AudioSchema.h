#pragma once

#include "state/Slot.h"

struct AcousticMaterial;
struct AudioDeviceResource;
struct MasterCapture;
struct VertexSamples;
struct AudioSamples;
struct SamplePlayback;
struct Recording;
struct AudioOutputConfig;
struct AudioOutputMix;
struct MonitorLimiter;
struct ModalSoundControls;
enum class SoundVerticesModel;
struct ModalSolveSettings;
struct ContactDynamics;
struct Striker;
struct ContactSurface;
struct SurfaceSoundControls;
struct SurfaceFinishKey;
struct ModalAudio;
struct ModalEigenSummary;
struct ModalModes;
struct ModalGain;
struct ModalTuning;
struct ModalWarmStart;
struct RealImpactActiveMicrophone;
struct RealImpactMicrophone;
struct RealImpactVertices;
struct SoundVertexChanges;
struct SoundVertices;
struct VertexForce;
namespace fastfem {
struct MassProperties;
} // namespace fastfem

using AudioSchema = state::Schema<state::AudioBase, state::AudioCount, AcousticMaterial, AudioDeviceResource, MasterCapture, VertexSamples, AudioSamples, SamplePlayback, Recording, AudioOutputConfig, AudioOutputMix, MonitorLimiter, ModalSoundControls, SoundVerticesModel, ModalSolveSettings, fastfem::MassProperties, ContactDynamics, Striker, ContactSurface, SurfaceSoundControls, SurfaceFinishKey, ModalAudio, ModalEigenSummary, ModalModes, ModalGain, ModalTuning, ModalWarmStart, RealImpactActiveMicrophone, RealImpactMicrophone, RealImpactVertices, SoundVertexChanges, SoundVertices, VertexForce>;
template<typename T>
    requires AudioSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = AudioSchema::Of<T>();
