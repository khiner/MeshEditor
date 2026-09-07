#pragma once

#include "Variant.h"
#include "action/Core.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/RealImpactComponents.h"

#include <filesystem>

namespace action::audio {
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
// Applies a completed modal solve from a path relative to the modal results directory.
struct ApplyModalModel {
    entt::entity SoundEntity;
    std::filesystem::path Path;
};
// Retain the vertices chosen before the asynchronous file dialog opens. Load frames only when applied.
struct AssignVertexSamples {
    std::unique_ptr<std::vector<uint32_t>> MeshVertices;
    std::filesystem::path Path;
};
// Loads microphone samples into the target sound entity and activates the microphone.
struct ActivateRealImpactMicrophone {
    entt::entity TargetSoundEntity, MicrophoneEntity;
};
struct RemoveVertexSamples {
    std::vector<uint32_t> MeshVertices;
};
// Applies a RealImpact impulse at a mesh vertex.
struct ApplyExciteImpact {
    entt::entity InstanceEntity;
    uint32_t VertexIndex;
};
struct ClearExciteImpacts {};

// Create missing settings from the same defaults shown by the controls.
struct EnsureModalSettings {};
struct SetMaterialPreset {
    entt::entity Entity;
    std::string Name;
    bool Striker{false};
};
struct SetSurfacePreset {
    entt::entity Entity;
    std::string Name;
};

using Actions = std::variant<
    SetModel, SetExciteVertex,
    StartExcite, StopExcite, DeleteSoundObject, StartRecording,
    EnsureModalSettings, ApplyModalModel,
    AssignVertexSamples, RemoveVertexSamples, ActivateRealImpactMicrophone,
    ApplyExciteImpact, ClearExciteImpacts>;

using Action = MergedVariantT<
    Actions, Replace<RealImpactActiveMicrophone>, Replace<AudioOutputConfig>, Replace<AudioOutputMix>, SetMaterialPreset, SetSurfacePreset,
    PatchFields<ModalSolveSettings, float, 2>, PatchFields<AcousticMaterial, double, 2>,
    PatchFields<ModalSolveSettings, bool>, PatchFields<ModalSolveSettings, uint32_t>,
    PatchFields<ModalSolveSettings, float>, PatchFields<ModalSolveSettings, double>,
    PatchFields<ModalSolveSettings, std::optional<float>>, PatchFields<ModalSolveSettings, fastfem::DVec3>,
    PatchFields<ModalSolveSettings, std::vector<fastfem::DVec3>>,
    PatchFields<ModalSolveSettings, fastfem::Discretization>, PatchFields<ModalSolveSettings, fastfem::TetRefinement>,
    PatchFields<AcousticMaterial, double>, PatchFields<ContactSurface, float>>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::audio
