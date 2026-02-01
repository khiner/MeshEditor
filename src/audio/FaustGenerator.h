#pragma once

#include "entt_fwd.h"

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

using OnFaustCodeChanged = std::function<void(std::string_view)>;

// Observes the registry and maintains a single Faust DSP program with all sound objects.
struct FaustGenerator {
    FaustGenerator(entt::registry &, OnFaustCodeChanged);
    ~FaustGenerator();

    struct ModalDsp {
        std::string Name, Definition, Eval;
    };

private:
    void OnCreateModalSoundObject(const entt::registry &, entt::entity);
    void OnDestroyModalSoundObject(const entt::registry &, entt::entity);

    entt::registry &R;
    OnFaustCodeChanged OnCodeChanged;

    std::unordered_map<entt::entity, ModalDsp> ModalDspByEntity;
};
