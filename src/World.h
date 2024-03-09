#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

static const glm::mat4 I{1};

struct World {
    const glm::vec3 Origin{0, 0, 0}, Up{0, 1, 0};
};
