#pragma once

#include <map>
#include <string>

// Defaults to aluminum.
struct MaterialProperties {
    double YoungModulus, PoissonRatio, Density;
};

inline const std::map<std::string, MaterialProperties> MaterialPresets = {
    {"Bell", {105e9f, 0.33f, 8600.0f}},
    {"Copper", {110e9f, 0.33f, 8900.0f}},
    {"Aluminum", {70e9f, 0.35f, 2700.0f}},
    {"Steel", {200e9f, 0.3f, 8000.0f}},
    {"Glass", {70e9f, 0.2f, 2500.0f}},
    {"Wood", {10e9f, 0.3f, 500.0f}},
};
