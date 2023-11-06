#include "Geometry.h"

#include <glm/geometric.hpp>

bool Geometry::Load(const fs::path &file_path) {
    OpenMesh::IO::Options read_options; // No options used yet, but keeping this here for future use.
    if (!OpenMesh::IO::read_mesh(Mesh, file_path.string(), read_options)) {
        std::cerr << "Error loading mesh: " << file_path << std::endl;
        return false;
    }
    return true;
}

void Geometry::Save(const fs::path &file_path) const {
    if (file_path.extension() != ".obj") throw std::runtime_error("Unsupported file type: " + file_path.string());

    if (!OpenMesh::IO::write_mesh(Mesh, file_path.string())) {
        std::cerr << "Error writing mesh to file: " << file_path << std::endl;
    }
}

// [{min_x, min_y, min_z}, {max_x, max_y, max_z}]
std::pair<glm::vec3, glm::vec3> Geometry::ComputeBounds() const {
    static const float min_float = std::numeric_limits<float>::lowest();
    static const float max_float = std::numeric_limits<float>::max();

    glm::vec3 min(max_float), max(min_float);
    for (const auto &vh : Mesh.vertices()) {
        const auto &p = Mesh.point(vh);
        min.x = std::min(min.x, p[0]);
        min.y = std::min(min.y, p[1]);
        min.z = std::min(min.z, p[2]);
        max.x = std::max(max.x, p[0]);
        max.y = std::max(max.y, p[1]);
        max.z = std::max(max.z, p[2]);
    }

    return {min, max};
}

uint Geometry::FindVertextNearestTo(const glm::vec3 point) const {
    uint nearest_index = 0;
    float nearest_distance = std::numeric_limits<float>::max();
    for (uint i = 0; i < NumPositions(); i++) {
        const float distance = glm::distance(point, GetPosition(i));
        if (distance < nearest_distance) {
            nearest_distance = distance;
            nearest_index = i;
        }
    }
    return nearest_index;
}

std::vector<uint> Geometry::GenerateTriangleIndices() const {
    auto triangulated_mesh = Mesh; // `triangulate` is in-place, so we need to make a copy.
    triangulated_mesh.triangulate();
    std::vector<uint> indices;
    for (const auto &fh : triangulated_mesh.faces()) {
        auto v_it = triangulated_mesh.cfv_iter(fh);
        indices.insert(indices.end(), {uint(v_it->idx()), uint((++v_it)->idx()), uint((++v_it)->idx())});
    }
    return indices;
}

std::vector<uint> Geometry::GenerateTriangulatedFaceIndices() const {
    std::vector<uint> indices;
    uint index = 0;
    for (const auto &fh : Mesh.faces()) {
        auto valence = Mesh.valence(fh);
        for (uint i = 0; i < valence - 2; ++i) {
            indices.insert(indices.end(), {index, index + i + 1, index + i + 2});
        }
        index += valence;
    }
    return indices;
}
