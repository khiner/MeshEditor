#pragma once

#include "Tets.h"
#include "Variant.h"
#include "action/Core.h"
#include "audio/AcousticMaterial.h"
#include "audio/AudioTypes.h"
#include "audio/ModalModes.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

namespace action::audio {
// All actions here target the active sound entity (FindActiveEntity(R)) and its mesh
// (GetActiveMeshEntity()), except `SetVertexSamples` which carries an explicit SoundEntity
// because it's emitted from a microphone branch that targets a discovered sound object.
struct SetModel {
    SoundVerticesModel Model;
};
// `VertexIndex` indexes `SoundVertices::Vertices`; `MeshVertex` is the mesh handle stored there.
struct SetExciteVertex {
    uint32_t VertexIndex;
    uint32_t MeshVertex;
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

using Actions = std::variant<
    SetModel, SetExciteVertex, SetActiveElementFromDsp,
    StartExcite, StopExcite, DeleteSoundObject, StartRecording,
    OpenModalForm, CancelModalForm, SubmitModalForm, AcceptModalGenerationResult,
    AssignVertexSamples, SetVertexSamples, RemoveVertexSamples, ActivateRealImpactMicrophone, SetModalFormMaterial>;

using Action = MergedVariantT<Actions, action::Core>;
} // namespace action::audio
