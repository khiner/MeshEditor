#pragma once

#include "Variant.h"
#include "action/Core.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/RealImpactComponents.h"

#include <filesystem>

namespace action::audio {
struct SetModel {
    SoundVerticesModel Model;
};
struct SetExciteVertex {
    uint32_t VertexIndex; // Indexes `SoundVertices::Vertices`
    uint32_t MeshVertex; // Mesh handle stored at `SoundVertices::Vertices[VertexIndex]`
};
struct StartExcite {
    uint32_t Vertex;
};
struct StopExcite {};
struct DeleteSoundObject {};
struct StartRecording {
    uint32_t FrameCount;
};
// Give the active sound entity default modal solve settings, and its mesh entity a default
// acoustic material when missing.
struct SetupModalModel {};
// Apply a completed modal solve from its result file.
// `Path` is relative to the modal results dir (see audio/ModalModelFile.h).
struct ApplyModalModel {
    entt::entity SoundEntity;
    std::filesystem::path Path;
};
// Frames are loaded from Path when applied, not stored.
struct AssignVertexSamples {
    std::vector<uint32_t> MeshVertices;
    std::filesystem::path Path;
};
// Load samples for the target sound entity from `MicrophoneEntity` and mark that mic active.
struct ActivateRealImpactMicrophone {
    entt::entity TargetSoundEntity, MicrophoneEntity;
};
struct RemoveVertexSamples {
    std::vector<uint32_t> MeshVertices;
};
// Apply an impulse at a mesh vertex (RealImpact excitation).
struct ApplyExciteImpact {
    entt::entity InstanceEntity;
    uint32_t VertexIndex;
};
struct ClearExciteImpacts {};

using Actions = std::variant<
    SetModel, SetExciteVertex,
    StartExcite, StopExcite, DeleteSoundObject, StartRecording,
    SetupModalModel, ApplyModalModel,
    AssignVertexSamples, RemoveVertexSamples, ActivateRealImpactMicrophone,
    ApplyExciteImpact, ClearExciteImpacts>;

using Action = MergedVariantT<Actions, Replace<RealImpactActiveMicrophone>, Replace<AudioOutputConfig>, Replace<AudioOutputMix>, Replace<Striker>, Replace<AcousticMaterial>>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::audio
