#pragma once

#include "AcousticMaterial.h"
#include "AudioBuffer.h"
#include "CreateSvgResource.h"
#include "ModalSoundObject.h"

#include "entt_fwd.h"

#include <filesystem>
#include <memory>
#include <vector>

struct Scene;
struct FaustDSP;
struct FaustGenerator;

enum class SoundObjectModel {
    // Play back recordings of impacts on the object at provided listener points/vertices.
    Samples,
    // Model a rigid body's response to an impact using modal analysis/synthesis:
    // - transform the object's geometry into a tetrahedral volume mesh
    // - use FEM to estimate the mass/sg/damping matrices from the mesh
    // - estimate the dominant modes (frequencies/amplitudes/T60s) using an eigensolver
    // - simulate the object's response to an impact by exciting the modes associated with the impacted vertex
    Modal,
};

// If an entity has this component, user has opened the modal model create/edit pane.
struct ModalModelCreateInfo {
    AcousticMaterial Material{materials::acoustic::All.front()};
    uint32_t NumExcitableVertices{10};
    bool CopyExcitable{true}; // Only used if excitable component is already present.
    bool QualityTets{false};
};

struct AcousticScene {
    AcousticScene(entt::registry &, CreateSvgResource);
    ~AcousticScene();

    void RenderControls(Scene &);

    void LoadRealImpact(const std::filesystem::path &directory, Scene &);
    void Process(AudioBuffer) const;

private:
    void Draw(entt::entity, entt::entity mesh_entity);

    void SetVertex(entt::entity, uint);
    void SetVertexForce(entt::entity, float);
    void SetImpactFrames(entt::entity, std::vector<std::vector<float>> &&impact_frames, std::vector<uint> &&vertex_indices);
    void SetImpactFrames(entt::entity, std::vector<std::vector<float>> &&impact_frames);
    void SetModel(entt::entity, SoundObjectModel);
    void Stop(entt::entity);

    void OnCreateExcitedVertex(const entt::registry &, entt::entity);
    void OnDestroyExcitedVertex(const entt::registry &, entt::entity);

    ModalSoundObject CreateModalSoundObject(entt::entity, entt::entity mesh_entity, const ModalModelCreateInfo &) const;

    entt::registry &R;
    CreateSvgResource CreateSvg;
    std::unique_ptr<FaustDSP> Dsp;
    std::unique_ptr<FaustGenerator> FaustGenerator;
};
