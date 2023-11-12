#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

struct Ray {
    Ray WorldToLocal(const glm::mat4 &model) const;

    glm::vec3 Origin;
    glm::vec3 Direction;
};
