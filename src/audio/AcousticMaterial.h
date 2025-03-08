#pragma once

#include "AcousticMaterialProperties.h"

#include <string>
#include <vector>

struct AcousticMaterial {
    std::string Name;
    AcousticMaterialProperties Properties;
};

/*
From Table 4 in the [Kleinpat paper](https://graphics.stanford.edu/projects/kleinpat/assets/mfpat_opt.pdf).
(The same set of materials is used in [RealImpact](https://arxiv.org/pdf/2306.09944.pdf).)

| Materials     | ρ    | E      | ν   | α  | β    |
|---------------|------|--------|-----|----|------|
| Ceramic       | 2700 | 7.2E10 | 0.19| 6  | 1E-7 |
| Glass         | 2600 | 6.2E10 | 0.20| 1  | 1E-7 |
| Wood          | 750  | 1.1E10 | 0.25| 60 | 2E-6 |
| Plastic       | 1070 | 1.4E9  | 0.35| 30 | 1E-6 |
| Iron          | 8000 | 2.1E11 | 0.28| 5  | 1E-7 |
| Polycarbonate | 1190 | 2.4E9  | 0.37| 0.5| 4E-7 |
| Steel         | 7850 | 2.0E11 | 0.29| 5  | 3E-8 |
*/

namespace materials {
namespace acoustic {
constexpr AcousticMaterial
    Ceramic{"Ceramic", {2700, 7.2E10, 0.19, 6, 1E-7}},
    Glass{"Glass", {2600, 6.2E10, 0.20, 1, 1E-7}},
    Wood{"Wood", {750, 1.1E10, 0.25, 60, 2E-6}},
    Plastic{"Plastic", {1070, 1.4E9, 0.35, 30, 1E-6}},
    Iron{"Iron", {8000, 2.1E11, 0.28, 5, 1E-7}},
    Polycarbonate{"Polycarbonate", {1190, 2.4E9, 0.37, 0.5, 4E-7}},
    Steel{"Steel", {7850, 2.0E11, 0.29, 5, 3E-8}};

const std::vector<AcousticMaterial> All{Ceramic, Glass, Wood, Plastic, Iron, Polycarbonate, Steel};
} // namespace acoustic
} // namespace materials
