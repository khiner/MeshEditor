#include "Tets.h"

#include <format>

#include "tetgen.h"

std::string TetGenOptions::CreateFlags() const { return std::format("p{}{}", PreserveSurface ? "Y" : "", Quality ? "q" : ""); }

Tets::Tets(std::unique_ptr<tetgenio> tet_gen) : TetGen(std::move(tet_gen)) {}
Tets::Tets(const Mesh &mesh, TetGenOptions options) : Tets(CreateTets(mesh, options)) {}
Tets::Tets(Tets &&other) = default;
Tets::~Tets() = default;

Tets &Tets::operator=(Tets &&other) noexcept {
    if (this != &other) {
        TetGen = std::move(other.TetGen);
        // Move or reassign other member variables if necessary
    }
    return *this;
}

Tets Tets::CreateTets(const Mesh &mesh, TetGenOptions options) {
    static constexpr int TriVerts = 3;
    tetgenio in;
    const float *vertices = mesh.GetPositionData();
    const auto triangle_indices = mesh.CreateTriangleIndices();
    in.numberofpoints = mesh.GetVertexCount();
    in.pointlist = new REAL[in.numberofpoints * TriVerts];
    for (uint i = 0; i < uint(in.numberofpoints); ++i) {
        in.pointlist[i * TriVerts] = vertices[i * TriVerts];
        in.pointlist[i * TriVerts + 1] = vertices[i * TriVerts + 1];
        in.pointlist[i * TriVerts + 2] = vertices[i * TriVerts + 2];
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
    const std::string flags_str = options.CreateFlags();
    const char *flags = flags_str.c_str();
    tetrahedralize(const_cast<char *>(flags), &in, result.get());
    return {std::move(result)};
}

vec3 Tets::GetVertexPosition(uint vertex) const {
    return {TetGen->pointlist[vertex * 3], TetGen->pointlist[vertex * 3 + 1], TetGen->pointlist[vertex * 3 + 2]};
}

Mesh Tets::CreateMesh() const {
    std::vector<vec3> vertices;
    std::vector<std::vector<uint>> faces;
    for (uint i = 0; i < uint(TetGen->numberofpoints); ++i) {
        vertices.emplace_back(TetGen->pointlist[i * 3], TetGen->pointlist[i * 3 + 1], TetGen->pointlist[i * 3 + 2]);
    }
    for (uint i = 0; i < uint(TetGen->numberoftrifaces); ++i) {
        const auto &tri_indices = TetGen->trifacelist;
        const uint tri_i = i * 3;
        const uint a = tri_indices[tri_i + 2], b = tri_indices[tri_i + 1], c = tri_indices[tri_i];
        faces.push_back({a, b, c});
    }

    return {std::move(vertices), std::move(faces)};
}
