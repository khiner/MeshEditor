#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

struct Ray {
    const glm::vec3 Origin, Direction;

    glm::vec3 operator()(float t) const { return Origin + Direction * t; }

    Ray WorldToLocal(const glm::mat4 &model) const;
};
