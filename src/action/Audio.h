#pragma once

#include "Variant.h"
#include "action/Core.h"
#include "audio/AudioTypes.h"
#include "audio/ModalModes.h"
#include "audio/RealImpactComponents.h"
#include "mesh/TetMeshData.h"

#include <filesystem>

namespace action::audio {
struct SetModel {
    SoundVerticesModel Model;
};
struct SetExciteVertex {
    uint32_t VertexIndex; // Indexes `SoundVertices::Vertices`
    uint32_t MeshVertex; // Mesh handle stored at `SoundVertices::Vertices[VertexIndex]`
};
struct SetActiveElementFromDsp {
    uint32_t Vertex;
};
struct StartExcite {
    uint32_t Vertex;
};
struct StopExcite {};
struct DeleteSoundObject {};
struct StartRecording {
    uint32_t FrameCount;
};
struct OpenModalForm {
    std::unique_ptr<ModalModelCreateInfo> Info;
};
struct CancelModalForm {};
struct SubmitModalForm {};
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
struct SetModalFormMaterial {
    std::unique_ptr<AcousticMaterial> Material;
};
// Apply an impulse at a mesh vertex (RealImpact excitation).
struct ApplyExciteImpact {
    entt::entity InstanceEntity;
    uint32_t VertexIndex;
};
struct ClearExciteImpacts {};

using Actions = std::variant<
    SetModel, SetExciteVertex, SetActiveElementFromDsp,
    StartExcite, StopExcite, DeleteSoundObject, StartRecording,
    OpenModalForm, CancelModalForm, SubmitModalForm,
    AssignVertexSamples, RemoveVertexSamples, ActivateRealImpactMicrophone, SetModalFormMaterial,
    ApplyExciteImpact, ClearExciteImpacts>;

using Action = MergedVariantT<Actions, Replace<RealImpactActiveMicrophone>>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::audio
