#pragma once

#include "AcousticMaterial.h"
#include "AudioBuffer.h"
#include "CreateSvgResource.h"
#include "ModalSoundObject.h"

#include <entt/entity/fwd.hpp>

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
    bool UseSampleVertices{true}; // Only used in Samples mode
    bool QualityTets{false};
};

struct AcousticScene {
    AcousticScene(entt::registry &, CreateSvgResource);
    ~AcousticScene();

    void RenderControls(Scene &);

    void LoadRealImpact(const std::filesystem::path &directory, Scene &);
    void ProduceAudio(AudioBuffer) const;

private:
    void Draw(entt::entity);

    void ProduceAudio(entt::entity, AudioBuffer &) const;

    void SetVertex(entt::entity, uint);
    void SetVertexForce(entt::entity, float);
    void SetImpactFrames(entt::entity, std::vector<std::vector<float>> &&impact_frames, std::vector<uint> &&vertex_indices);
    void SetImpactFrames(entt::entity, std::vector<std::vector<float>> &&impact_frames);
    void SetModel(entt::entity, SoundObjectModel);
    void Stop(entt::entity);

    void OnCreateExcitedVertex(entt::registry &, entt::entity);
    void OnDestroyExcitedVertex(entt::registry &, entt::entity);

    ModalSoundObject CreateModalSoundObject(entt::entity, const ModalModelCreateInfo &) const;

    entt::registry &R;
    CreateSvgResource CreateSvg;
    std::unique_ptr<FaustDSP> Dsp;
    std::unique_ptr<FaustGenerator> FaustGenerator;
};
