#include "Rect.h"

using glm::vec2, glm::vec3;

Rect::Rect(vec2 half_extents) : Geometry() {
    const auto &he = half_extents;
    const std::vector<Point> positions{
        {-he.x, -he.y, 0},
        {he.x, -he.y, 0},
        {he.x, he.y, 0},
        {-he.x, he.y, 0},
    };

    std::vector<VH> face;
    face.reserve(positions.size());
    for (const auto &position : positions) face.push_back(Mesh.add_vertex(position));

    AddFace(face);
}