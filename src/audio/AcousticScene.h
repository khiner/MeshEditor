#pragma once

#include "AudioBuffer.h"
#include "CreateSvgResource.h"

#include <entt/entity/fwd.hpp>

#include <filesystem>
#include <memory>

struct AcousticMaterial;
struct Scene;
struct SoundObject;
struct FaustDSP;
struct FaustGenerator;

struct AcousticScene {
    AcousticScene(entt::registry &, CreateSvgResource);
    ~AcousticScene();

    void LoadRealImpact(const std::filesystem::path &directory, Scene &) const;

    void RenderControls(Scene &);

    void ProduceAudio(AudioBuffer) const;

private:
    SoundObject &AddSoundObject(entt::entity, AcousticMaterial) const;

    entt::registry &R;
    CreateSvgResource CreateSvg;
    std::unique_ptr<FaustDSP> Dsp;
    std::unique_ptr<FaustGenerator> FaustGenerator;
};
