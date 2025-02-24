#pragma once

#include <entt/entity/fwd.hpp>

#include <functional>
#include <string_view>

using OnFaustCodeChanged = std::function<void(std::string_view)>;

// Observes the registry and maintains a single Faust DSP program with all sound objects.
struct FaustGenerator {
    FaustGenerator(entt::registry &, OnFaustCodeChanged);
    ~FaustGenerator();

    void OnCreateModalSoundObject(entt::registry &, entt::entity);
    void OnDestroyModalSoundObject(entt::registry &, entt::entity);

    entt::registry &R;
    OnFaustCodeChanged OnCodeChanged;
};
