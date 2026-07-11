#pragma once

#include "numeric/vec3.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// Triangle surface mesh with exact-duplicate vertices welded (scan exports repeat vertices per face).
struct SurfaceMesh {
    std::vector<vec3> Positions;
    std::vector<uint32_t> TriangleIndices;
};

inline std::optional<SurfaceMesh> LoadObj(const std::filesystem::path &path) {
    std::ifstream file{path};
    if (!file) return {};
    SurfaceMesh mesh;
    std::map<std::array<float, 3>, uint32_t> welded;
    std::vector<uint32_t> remap;
    std::string line, token;
    while (std::getline(file, line)) {
        std::istringstream ss{line};
        ss >> token;
        if (token == "v") {
            std::array<float, 3> p;
            ss >> p[0] >> p[1] >> p[2];
            const auto [it, inserted] = welded.try_emplace(p, uint32_t(mesh.Positions.size()));
            if (inserted) mesh.Positions.emplace_back(p[0], p[1], p[2]);
            remap.push_back(it->second);
        } else if (token == "f") {
            std::vector<uint32_t> corners;
            while (ss >> token) corners.push_back(remap[std::stoi(token.substr(0, token.find('/'))) - 1]);
            for (size_t i = 2; i < corners.size(); ++i) mesh.TriangleIndices.insert(mesh.TriangleIndices.end(), {corners[0], corners[i - 1], corners[i]});
        }
    }
    return mesh;
}
