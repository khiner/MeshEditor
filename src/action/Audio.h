#pragma once

#include "numeric/dvec3.h"

#include "action/Core.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/RealImpactComponents.h"

#include <filesystem>

namespace action::audio {
using numeric::dvec3;

struct SetModel {
    SoundVerticesModel Model;
};
struct SetExciteVertex {
    uint32_t VertexIndex;
    uint32_t MeshVertex;
};
struct StartExcite {
    uint32_t Vertex;
};
struct StopExcite {};
struct DeleteSoundObject {};
struct StartRecording {
    uint32_t FrameCount;
};
struct ApplyModalModel {
    state::Entity SoundEntity;
    std::filesystem::path Path;
};
// Retain the vertices chosen before the asynchronous file dialog opens. Load frames only when applied.
struct AssignVertexSamples {
    std::unique_ptr<std::vector<uint32_t>> MeshVertices;
    std::filesystem::path Path;
};
// Loads microphone samples into the target sound entity and activates the microphone.
struct ActivateRealImpactMicrophone {
    state::Entity TargetSoundEntity, MicrophoneEntity;
};
struct RemoveVertexSamples {
    std::vector<uint32_t> MeshVertices;
};
// Applies a RealImpact impulse at a mesh vertex.
struct ApplyExciteImpact {
    state::Entity InstanceEntity;
    uint32_t VertexIndex;
};
struct ClearExciteImpacts {};

// Create missing settings from the same defaults shown by the controls.
struct EnsureModalSettings {};
struct SetMaterialPreset {
    state::Entity Entity;
    std::string Name;
    bool Striker{false};
};
struct SetSurfacePreset {
    state::Entity Entity;
    std::string Name;
};
// Selects the output device at its default sample rate.
struct SetOutputDevice {
    std::string DeviceName;
};

using Action = std::variant<
    SetModel, SetExciteVertex,
    StartExcite, StopExcite, DeleteSoundObject, StartRecording,
    EnsureModalSettings, ApplyModalModel,
    AssignVertexSamples, RemoveVertexSamples, ActivateRealImpactMicrophone,
    ApplyExciteImpact, ClearExciteImpacts,
    SetMaterialPreset, SetSurfacePreset, SetOutputDevice,
    PatchFields<ModalSolveSettings, float, 2>, PatchFields<AcousticMaterial, double, 2>,
    PatchFields<ModalSolveSettings, bool>, PatchFields<ModalSolveSettings, uint32_t>,
    PatchFields<ModalSolveSettings, float>, PatchFields<ModalSolveSettings, double>,
    PatchFields<ModalSolveSettings, std::optional<float>>, PatchFields<ModalSolveSettings, dvec3>,
    PatchFields<ModalSolveSettings, std::vector<dvec3>>,
    PatchFields<ModalSolveSettings, fastfem::Discretization>, PatchFields<ModalSolveSettings, fastfem::TetRefinement>,
    PatchFields<AcousticMaterial, double>, PatchFields<ContactSurface, float>>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
} // namespace action::audio
