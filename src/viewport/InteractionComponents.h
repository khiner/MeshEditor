#pragma once

#include "gpu/InteractionMode.h"

#include <numbers>
#include <set>

struct Interaction {
    InteractionMode Mode{InteractionMode::Object};
};

struct EditMode {
    Element Value{Element::Vertex};
};

struct EnabledInteractionModes {
    std::set<InteractionMode> Value{InteractionMode::Object, InteractionMode::Edit, InteractionMode::Pose};
};

struct OrbitToActive {
    bool Value{false};
    bool operator==(const OrbitToActive &) const = default;
};

struct ShadeSmoothAngle {
    float Value{std::numbers::pi_v<float> / 6.f};
};
template<> inline constexpr field::SpecTable<ShadeSmoothAngle> Specs<ShadeSmoothAngle>{{&ShadeSmoothAngle::Value, {.Min = 0, .Max = std::numbers::pi, .Unit = FieldUnit::Radians, .Help = "Largest angle between adjacent face normals that shades smooth"}}};
