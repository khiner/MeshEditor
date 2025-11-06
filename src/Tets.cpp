#include "Tets.h"
#include "mesh/Mesh.h"

#include "tetgen.h"

#include <format>

std::unique_ptr<tetgenio> GenerateTets(const Mesh &mesh, vec3 scale, TetGenOptions options) {
    static constexpr int TriVerts = 3;
    tetgenio in;
    const float *vertices = mesh.GetPositionData();
    const auto triangle_indices = mesh.CreateTriangleIndices();
    in.numberofpoints = mesh.VertexCount();
    in.pointlist = new REAL[in.numberofpoints * TriVerts];
    for (uint i = 0; i < uint(in.numberofpoints); ++i) {
        in.pointlist[i * TriVerts] = vertices[i * TriVerts] * scale.x;
        in.pointlist[i * TriVerts + 1] = vertices[i * TriVerts + 1] * scale.y;
        in.pointlist[i * TriVerts + 2] = vertices[i * TriVerts + 2] * scale.z;
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
    const auto flags = std::format("p{}{}", options.PreserveSurface ? "Y" : "", options.Quality ? "q" : "");
    tetrahedralize(const_cast<char *>(flags.c_str()), &in, result.get());
    return result;
}

/*
Mesh CreateTetsMesh(const tetgenio &tets) const {
    std::vector<vec3> vertices;
    std::vector<std::vector<uint>> faces;
    for (uint i = 0; i < uint(tets.numberofpoints); ++i) {
        vertices.emplace_back(tets.pointlist[i * 3], tets.pointlist[i * 3 + 1], tets.pointlist[i * 3 + 2]);
    }
    for (uint i = 0; i < uint(tets.numberoftrifaces); ++i) {
        const auto &tri_indices = tets.trifacelist;
        const uint tri_i = i * 3;
        const uint a = tri_indices[tri_i + 2], b = tri_indices[tri_i + 1], c = tri_indices[tri_i];
        faces.push_back({a, b, c});
    }

    return {std::move(vertices), std::move(faces)};
}
*/
