#pragma once

#include "gpu/Transform.h"

#include <string>

struct MeshInstanceCreateInfo {
    std::string Name{};
    Transform Transform{};

    enum class SelectBehavior {
        Exclusive,
        Additive,
        None,
    };
    SelectBehavior Select{SelectBehavior::Exclusive};
    bool Visible{true};
};

struct ObjectCreateInfo {
    std::string Name{};
    Transform Transform{};
    MeshInstanceCreateInfo::SelectBehavior Select{MeshInstanceCreateInfo::SelectBehavior::Exclusive};
};
