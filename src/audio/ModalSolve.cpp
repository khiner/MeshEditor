#include "ModalSolve.h"

#include <algorithm>
#include <cstdint>

namespace {
ModalModes Extend(fastfem::ModalModes modes) {
    ModalModes result;
    static_cast<fastfem::ModalModes &>(result) = std::move(modes);
    return result;
}

ModalEigenSummary Extend(fastfem::ModalEigenSummary summary) {
    ModalEigenSummary result;
    static_cast<fastfem::ModalEigenSummary &>(result) = std::move(summary);
    return result;
}

TetMeshData Import(const fastfem::TetMesh &mesh, vec3 scale) {
    const vec3 inverse_scale{1 / scale.x, 1 / scale.y, 1 / scale.z};
    TetMeshData result;
    result.Positions.reserve(mesh.Points.size());
    for (const auto &point : mesh.Points) result.Positions.emplace_back(vec3{float(point.x), float(point.y), float(point.z)} * inverse_scale);

    std::vector<uint64_t> edges;
    edges.reserve(mesh.Tets.size() * 6);
    for (const auto &tetrahedron : mesh.Tets) {
        for (uint32_t first = 0; first < 4; ++first) {
            for (uint32_t second = first + 1; second < 4; ++second) {
                const auto [lower, upper] = std::minmax(tetrahedron[first], tetrahedron[second]);
                edges.push_back(uint64_t(lower) << 32 | upper);
            }
        }
    }
    std::ranges::sort(edges);
    edges.erase(std::ranges::unique(edges).begin(), edges.end());
    result.EdgeIndices.reserve(edges.size() * 2);
    for (const auto edge : edges) {
        result.EdgeIndices.push_back(uint32_t(edge >> 32));
        result.EdgeIndices.push_back(uint32_t(edge));
    }
    return result;
}
} // namespace

std::expected<modal::SurfaceSolveResult, std::string> modal::SolveSurfaceModes(
    std::span<const vec3> positions, std::span<const uint32_t> triangle_indices,
    const AcousticMaterialProperties &material, std::span<const vec3> excitation_positions,
    vec3 baked_scale, fastfem::Discretization discretization, fastfem::SurfaceSolveConfig config,
    fastfem::SolveReuse reuse, fastfem::SolveMonitor *monitor
) {
    auto result = fastfem::Surface2Modes(positions, triangle_indices, material, excitation_positions, baked_scale, discretization, std::move(config), reuse, monitor);
    if (!result) return std::unexpected(result.error());
    return SurfaceSolveResult{
        .Modes = Extend(std::move(result->Modes)),
        .Mass = std::move(result->Mass),
        .Summary = Extend(std::move(result->Summary)),
        .Tetrahedra = Import(result->Tetrahedra, baked_scale),
        .Basis = result->Basis,
        .SamplePointOfExcitation = result->SamplePointOfExcitation,
    };
}

std::optional<ModalModes> modal::RescaleModes(
    const ModalEigenSummary &summary, const ModalModes &current,
    const AcousticMaterialProperties &material, fastfem::SolverConfig config
) {
    auto result = fastfem::RescaleModes(summary, current, material, config);
    if (!result) return {};
    auto modes = Extend(std::move(*result));
    modes.Vertices = current.Vertices;
    modes.Indices = current.Indices;
    return modes;
}
