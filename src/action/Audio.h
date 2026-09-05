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
struct ConfigureModalModel {
    ModalSolveSettings Settings;
    AcousticMaterial Material;
};
// Applies a completed modal solve from a path relative to the modal results directory.
struct ApplyModalModel {
    entt::entity SoundEntity;
    std::filesystem::path Path;
};
// Loads frames from Path during application.
struct AssignVertexSamples {
    std::vector<uint32_t> MeshVertices;
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

using Actions = std::variant<
    SetModel, SetExciteVertex,
    StartExcite, StopExcite, DeleteSoundObject, StartRecording,
    ConfigureModalModel, ApplyModalModel,
    AssignVertexSamples, RemoveVertexSamples, ActivateRealImpactMicrophone,
    ApplyExciteImpact, ClearExciteImpacts>;

using Action = MergedVariantT<
    Actions, Replace<RealImpactActiveMicrophone>, Replace<AudioOutputConfig>, Replace<AudioOutputMix>, Replace<Striker>, Replace<ModalSolveSettings>, Replace<AcousticMaterial>, Replace<ContactSurface>>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::audio
