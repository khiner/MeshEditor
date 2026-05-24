#pragma once

#include "Tets.h"
#include "Variant.h"
#include "action/Core.h"
#include "audio/AcousticMaterial.h"
#include "audio/AudioTypes.h"
#include "audio/ModalModes.h"
#include "audio/RealImpactComponents.h"

#include <filesystem>
#include <memory>

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
struct AcceptModalGenerationResult {
    struct Data {
        ModalModes Modes;
        TetMeshData Tets;
    };
    std::unique_ptr<Data> D;
};
struct AssignVertexSamples {
    struct Data {
        std::vector<uint32_t> MeshVertices;
        std::filesystem::path Path;
        std::vector<float> Frames;
    };
    std::unique_ptr<Data> D;
};
struct SetVertexSamples {
    entt::entity SoundEntity;
    std::vector<uint32_t> MeshVertices;
    std::vector<std::pair<std::filesystem::path, std::vector<float>>> Samples;
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
    OpenModalForm, CancelModalForm, SubmitModalForm, AcceptModalGenerationResult,
    AssignVertexSamples, SetVertexSamples, RemoveVertexSamples, ActivateRealImpactMicrophone, SetModalFormMaterial,
    ApplyExciteImpact, ClearExciteImpacts>;

using Action = MergedVariantT<Actions, Replace<RealImpactActiveMicrophone>>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::audio
