#include "Tets.h"
#include "mesh/Mesh.h"

#include "tetgen.h"

#include <format>

std::unique_ptr<tetgenio> GenerateTets(const Mesh &mesh, vec3 scale, TetGenOptions options) {
    static constexpr int TriVerts = 3;
    tetgenio in;
    const auto triangle_indices = mesh.CreateTriangleIndices();
    in.numberofpoints = mesh.VertexCount();
    in.pointlist = new REAL[in.numberofpoints * TriVerts];
    for (uint i = 0; i < uint(in.numberofpoints); ++i) {
        const auto &p = mesh.GetPosition(Mesh::VH{i});
        in.pointlist[i * TriVerts] = p.x * scale.x;
        in.pointlist[i * TriVerts + 1] = p.y * scale.y;
        in.pointlist[i * TriVerts + 2] = p.z * scale.z;
    }
    in.numberoffacets = triangle_indices.size() / TriVerts;
    in.facetlist = new tetgenio::facet[in.numberoffacets];

    for (uint i = 0; i < uint(in.numberoffacets); ++i) {
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

    auto result = std::make_unique<tetgenio>();
    const auto flags = std::format("pe{}{}", options.PreserveSurface ? "Y" : "", options.Quality ? "q" : "");
    tetrahedralize(const_cast<char *>(flags.c_str()), &in, result.get());
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
