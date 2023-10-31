#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

struct Vertex2D {
    glm::vec2 Position;
    glm::vec4 Color;
};

struct Vertex3D {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec4 Color;
};
