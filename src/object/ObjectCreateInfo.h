#pragma once

#include "gpu/Transform.h"
#include "state/Entity.h"

#include <string>

struct MeshObjectCreateInfo {
    std::string Name{};
    Transform Transform{};

    enum class SelectBehavior {
        Exclusive,
        Additive,
        None,
    };
    SelectBehavior Select{SelectBehavior::Exclusive};
    bool Visible{true};
    state::Entity Scene{state::Null};
    bool BeginTranslate{true};
    state::Entity Graph{state::Null};
};

struct ObjectCreateInfo {
    std::string Name{};
    Transform Transform{};
    MeshObjectCreateInfo::SelectBehavior Select{MeshObjectCreateInfo::SelectBehavior::Exclusive};
    state::Entity Scene{state::Null};
    bool BeginTranslate{true};
    state::Entity Graph{state::Null};
};
