#include "mesh/Tets.h"

#include "meshoptimizer.h"
#include "tetgen.h"

#include <filesystem>
#include <format>

namespace {
std::string TetGenErrorMessage(int code) {
    switch (code) {
        case 1: return "out of memory";
        case 2: return "internal tetgen error";
        case 3: return "the input surface mesh contains self-intersections";
        case 4: return "a very small input feature size was detected";
        case 5: return "two very close input facets were detected";
        case 10: return "an input error was detected";
        case 200: return "the input boundary contains Steiner points";
        default: return std::format("tetgen error code {}", code);
    }
}
} // namespace

std::expected<std::unique_ptr<tetgenio>, std::string> GenerateTets(std::vector<vec3> positions, std::vector<uint32_t> triangle_indices, TetGenOptions options) {
    static constexpr int TriVerts = 3;
    if (options.SimplifyRatio < 1) {
        // Quadric edge-collapse to the target triangle count, then drop unreferenced vertices.
        const auto target_indices = std::max<size_t>(size_t(triangle_indices.size() * options.SimplifyRatio) / 3 * 3, 12);
        std::vector<uint32_t> simplified(triangle_indices.size());
        simplified.resize(meshopt_simplify(simplified.data(), triangle_indices.data(), triangle_indices.size(), &positions[0].x, positions.size(), sizeof(vec3), target_indices, 0.05f, 0, nullptr));
        positions.resize(meshopt_optimizeVertexFetch(positions.data(), simplified.data(), simplified.size(), positions.data(), positions.size(), sizeof(vec3)));
        triangle_indices = std::move(simplified);
    }

    tetgenio in;
    in.numberofpoints = positions.size();
    in.pointlist = new REAL[in.numberofpoints * TriVerts];
    for (uint32_t i = 0; i < uint32_t(in.numberofpoints); ++i) {
        const auto &p = positions[i];
        in.pointlist[i * TriVerts] = p.x;
        in.pointlist[i * TriVerts + 1] = p.y;
        in.pointlist[i * TriVerts + 2] = p.z;
    }
    in.numberoffacets = triangle_indices.size() / TriVerts;
    in.facetlist = new tetgenio::facet[in.numberoffacets];

    for (uint32_t i = 0; i < uint32_t(in.numberoffacets); ++i) {
        auto &f = in.facetlist[i];
        tetgenio::init(&f);
        f.numberofpolygons = 1;
        f.polygonlist = new tetgenio::polygon[f.numberofpolygons];
        f.polygonlist[0].numberofvertices = TriVerts;
        f.polygonlist[0].vertexlist = new int[TriVerts];
        for (int j = 0; j < TriVerts; ++j) {
            f.polygonlist[0].vertexlist[j] = triangle_indices[i * TriVerts + j];
        }
    }

    // (-Q) Quiet: no terminal output except errors.
    auto flags = std::format("peQ{}{}", options.PreserveSurface ? "Y" : "", options.Quality ? "q" : "");
    tetgenbehavior behavior;
    behavior.parse_commandline(flags.data());
    // On a self-intersecting input, tetgen writes diagnostic files next to `outfilename`.
    const auto diagnostic_base = (std::filesystem::temp_directory_path() / "tetgen").string();
    std::strncpy(behavior.outfilename, diagnostic_base.c_str(), sizeof(behavior.outfilename) - 1);
    auto result = std::make_unique<tetgenio>();
    try {
        tetrahedralize(&behavior, &in, result.get());
    } catch (int code) { // tetgen throws an int on failure, e.g. a self-intersecting or non-watertight surface.
        return std::unexpected(TetGenErrorMessage(code));
    }
    return result;
}

TetMeshData BuildTetMeshData(const tetgenio &tets, vec3 scale) {
    static_assert(sizeof(int) == sizeof(uint32_t));
    const vec3 inv_scale{1.f / scale.x, 1.f / scale.y, 1.f / scale.z};
    TetMeshData out;
    out.Positions.reserve(tets.numberofpoints);
    for (int i = 0; i < tets.numberofpoints; ++i) {
        out.Positions.emplace_back(
            float(tets.pointlist[i * 3] * inv_scale.x),
            float(tets.pointlist[i * 3 + 1] * inv_scale.y),
            float(tets.pointlist[i * 3 + 2] * inv_scale.z)
        );
    }
    out.EdgeIndices.assign(reinterpret_cast<const uint32_t *>(tets.edgelist), reinterpret_cast<const uint32_t *>(tets.edgelist) + tets.numberofedges * 2);
    return out;
}
