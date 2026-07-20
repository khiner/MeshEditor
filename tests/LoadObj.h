#pragma once

#include "numeric/vec3.h"

#include "tiny_obj_loader.h"

#include <array>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

// Triangle surface mesh with exact-duplicate vertices welded (scan exports repeat vertices per face).
struct SurfaceMesh {
    std::vector<vec3> Positions;
    std::vector<uint32_t> TriangleIndices;
};

inline std::optional<SurfaceMesh> LoadObj(const std::filesystem::path &path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.string().c_str())) return {};

    SurfaceMesh mesh;
    std::map<std::array<float, 3>, uint32_t> welded;
    std::vector<uint32_t> remap(attrib.vertices.size() / 3);
    for (size_t i = 0; i < remap.size(); ++i) {
        const std::array<float, 3> p{attrib.vertices[i * 3], attrib.vertices[i * 3 + 1], attrib.vertices[i * 3 + 2]};
        const auto [it, inserted] = welded.try_emplace(p, uint32_t(mesh.Positions.size()));
        if (inserted) mesh.Positions.emplace_back(p[0], p[1], p[2]);
        remap[i] = it->second;
    }
    // tinyobj triangulates as it reads, so every face is already three corners.
    for (const auto &shape : shapes) {
        for (const auto &index : shape.mesh.indices) mesh.TriangleIndices.push_back(remap[size_t(index.vertex_index)]);
    }
    return mesh;
}
