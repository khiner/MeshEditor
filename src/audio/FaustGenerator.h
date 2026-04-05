#pragma once

#include "entt_fwd.h"

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

static constexpr std::string_view ExciteIndexParamName{"Excite index"};
static constexpr std::string_view GateParamName{"Gate"};

using OnFaustCodeChanged = std::function<void(std::string_view)>;

// Observes the registry and maintains a single Faust DSP program with all sound objects.
struct FaustGenerator {
    FaustGenerator(entt::registry &, OnFaustCodeChanged);
    ~FaustGenerator();

    struct ModalDsp {
        std::string Name, Definition, Eval;
    };

private:
    void OnCreateModalModes(const entt::registry &, entt::entity);
    void OnDestroyModalModes(const entt::registry &, entt::entity);

    entt::registry &R;
    OnFaustCodeChanged OnCodeChanged;

    std::unordered_map<entt::entity, ModalDsp> ModalDspByEntity;
};
