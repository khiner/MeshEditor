#pragma once

#include "action/Core.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/RealImpactComponents.h"

namespace action::audio {
using numeric::dvec3;

struct SetModel {
    SoundVerticesModel Model;
    state::Entity Entity{state::Null};
};
struct SetExciteVertex {
    uint32_t VertexIndex;
    uint32_t MeshVertex;
    state::Entity Entity{state::Null};
};
struct StartExcite {
    uint32_t Vertex;
    state::Entity Entity{state::Null};
};
struct StopExcite {
    state::Entity Entity{state::Null};
};
struct DeleteSoundObject {
    state::Entity Entity{state::Null};
};
struct StartRecording {
    uint32_t FrameCount;
    state::Entity Entity{state::Null};
};
struct ApplyModalModel {
    state::Entity SoundEntity;
    std::filesystem::path Path;
};
// Retain selected vertices across the asynchronous file dialog.
// Load frames during Apply.
struct AssignVertexSamples {
    std::unique_ptr<std::vector<uint32_t>> MeshVertices;
    std::filesystem::path Path;
    state::Entity Entity{state::Null};
};
struct ActivateRealImpactMicrophone {
    state::Entity TargetSoundEntity;
    state::Entity Entity{state::Null};
};
struct RemoveVertexSamples {
    std::vector<uint32_t> MeshVertices;
    state::Entity Entity{state::Null};
};
struct ApplyExciteImpact {
    state::Entity InstanceEntity;
    uint32_t VertexIndex;
};
struct ClearExciteImpacts {};

// Create missing settings from the same defaults shown by the controls.
struct EnsureModalSettings {
    state::Entity Entity{state::Null};
};
struct SetMaterialPreset {
    std::string Name;
    bool Striker{false};
    state::Entity Entity{state::Null};
};
struct SetSurfacePreset {
    std::string Name;
    state::Entity Entity{state::Null};
};
// Selects the output device, at the device default rate when SampleRate is zero.
struct SetOutputDevice {
    std::string DeviceName;
    uint32_t SampleRate{0};
};

using Action = std::variant<
    SetModel, SetExciteVertex,
    StartExcite, StopExcite, DeleteSoundObject, StartRecording,
    EnsureModalSettings, ApplyModalModel,
    AssignVertexSamples, RemoveVertexSamples, ActivateRealImpactMicrophone,
    ApplyExciteImpact, ClearExciteImpacts,
    SetMaterialPreset, SetSurfacePreset, SetOutputDevice,
    PatchFields<ModalSolveSettings, bool>, PatchFields<ModalSolveSettings, uint32_t>,
    PatchFields<ModalSolveSettings, float>, PatchFields<ModalSolveSettings, double>,
    PatchFields<ModalSolveSettings, std::optional<float>>, PatchFields<ModalSolveSettings, dvec3>,
    PatchFields<ModalSolveSettings, std::vector<dvec3>>,
    PatchFields<ModalSolveSettings, fastfem::Discretization>, PatchFields<ModalSolveSettings, fastfem::TetRefinement>,
    PatchFields<AcousticMaterial, double>, PatchFields<ContactSurface, float>>;

void Apply(state::Registry &, const Action &);
} // namespace action::audio
