#pragma once

#include "Tets.h"
#include "Variant.h"
#include "action/Update.h"
#include "audio/AcousticMaterial.h"
#include "audio/AudioTypes.h"
#include "audio/ModalModes.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

namespace action::audio {
struct SetModel {
    entt::entity Entity;
    SoundVerticesModel Model;
};
// `VertexIndex` indexes `SoundVertices::Vertices`; `MeshVertex` is the mesh handle stored there.
struct SetExciteVertex {
    entt::entity Entity;
    entt::entity MeshEntity;
    uint32_t VertexIndex;
    uint32_t MeshVertex;
};
struct SetActiveElementFromDsp {
    entt::entity MeshEntity;
    uint32_t Vertex;
};
struct StartExcite {
    entt::entity Entity;
    uint32_t Vertex;
};
struct StopExcite {
    entt::entity Entity;
};
struct DeleteSoundObject {
    entt::entity Entity;
};
struct StartRecording {
    entt::entity Entity;
    uint32_t FrameCount;
};
struct OpenModalForm {
    entt::entity Entity;
    std::unique_ptr<ModalModelCreateInfo> Info;
};
struct CancelModalForm {
    entt::entity Entity;
};
struct SubmitModalForm {
    entt::entity Entity;
    entt::entity MeshEntity;
};
struct AcceptModalGenerationResult {
    struct Data {
        ModalModes Modes;
        TetMeshData Tets;
    };
    entt::entity Entity;
    entt::entity MeshEntity;
    std::unique_ptr<Data> D;
};
struct AssignVertexSamples {
    struct Data {
        std::vector<uint32_t> MeshVertices;
        std::filesystem::path Path;
        std::vector<float> Frames;
    };
    entt::entity SceneEntity;
    entt::entity SoundEntity;
    std::unique_ptr<Data> D;
};
struct SetVertexSamples {
    entt::entity SceneEntity;
    entt::entity SoundEntity;
    std::vector<uint32_t> MeshVertices;
    std::vector<std::pair<std::filesystem::path, std::vector<float>>> Samples;
};
struct RemoveVertexSamples {
    entt::entity SceneEntity;
    entt::entity SoundEntity;
    std::vector<uint32_t> MeshVertices;
};
struct SetModalFormMaterial {
    entt::entity Entity;
    std::unique_ptr<AcousticMaterial> Material;
};

using Actions = std::variant<
    SetModel, SetExciteVertex, SetActiveElementFromDsp,
    StartExcite, StopExcite, DeleteSoundObject, StartRecording,
    OpenModalForm, CancelModalForm, SubmitModalForm, AcceptModalGenerationResult,
    AssignVertexSamples, SetVertexSamples, RemoveVertexSamples, SetModalFormMaterial>;

using Action = MergedVariantT<
    Actions, std::variant<action::Update<bool>, action::Update<uint32_t>, action::Update<double>>>;
} // namespace action::audio
