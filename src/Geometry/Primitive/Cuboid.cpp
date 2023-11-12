#include "Cuboid.h"

Cuboid::Cuboid(glm::vec3 half_extents) : Geometry() {
    const auto &he = half_extents;
    const std::vector<Point> positions{
        {-he.x, -he.y, -he.z},
        {he.x, -he.y, -he.z},
        {he.x, he.y, -he.z},
        {-he.x, he.y, -he.z},
        {-he.x, -he.y, he.z},
        {he.x, -he.y, he.z},
        {he.x, he.y, he.z},
        {-he.x, he.y, he.z},
    };

    std::vector<VH> vhs;
    vhs.reserve(positions.size());
    for (const auto &position : positions) vhs.push_back(Mesh.add_vertex(position));

    const std::vector<std::vector<VH>> faces{
        {vhs[0], vhs[3], vhs[2], vhs[1]}, // front
        {vhs[4], vhs[5], vhs[6], vhs[7]}, // back
        {vhs[0], vhs[1], vhs[5], vhs[4]}, // bottom
        {vhs[3], vhs[7], vhs[6], vhs[2]}, // top
        {vhs[0], vhs[4], vhs[7], vhs[3]}, // left
        {vhs[1], vhs[2], vhs[6], vhs[5]} // right
    };

    for (const auto &face : faces) AddFace(face);
}
