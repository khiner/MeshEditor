#include "MeshRender.h"

#include "Mesh.h"
#include "WorldMatrix.h"

#include <entt/entity/registry.hpp>

#include <numeric>
#include <ranges>

using std::views::iota, std::ranges::to;

using namespace he;

namespace MeshRender {

std::vector<uint> CreateFaceIndices(const Mesh &mesh) { return mesh.CreateTriangulatedFaceIndices(); }
std::vector<uint> CreateEdgeIndices(const Mesh &mesh) { return mesh.CreateEdgeIndices(); }

std::vector<uint> CreateVertexIndices(const Mesh &mesh) { return iota(0u, mesh.VertexCount()) | to<std::vector<uint>>(); }

std::vector<uint32_t> CreateFaceElementIds(const Mesh &mesh) {
    std::vector<uint32_t> ids;
    ids.reserve(mesh.FaceCount() * 3);
    for (const auto fh : mesh.faces()) {
        ids.insert(ids.end(), mesh.GetValence(fh), *fh + 1);
    }
    return ids;
}

std::vector<uint> CreateNormalIndices(const Mesh &mesh, Element element) {
    if (element == Element::None || element == Element::Edge) return {};
    const auto n = element == Element::Face ? mesh.FaceCount() : mesh.VertexCount();
    return iota(0u, n * 2) | to<std::vector<uint>>();
}

std::vector<Vertex3D> CreateFaceVertices(const Mesh &mesh, bool smooth_shading) {
    std::vector<Vertex3D> vertices;
    vertices.reserve(mesh.FaceCount() * 3);
    for (const auto fh : mesh.faces()) {
        for (const auto vh : mesh.fv_range(fh)) {
            vertices.emplace_back(mesh.GetPosition(vh), smooth_shading ? mesh.GetNormal(vh) : mesh.GetNormal(fh));
        }
    }
    return vertices;
}

std::vector<Vertex3D> CreateEdgeVertices(const Mesh &mesh) {
    std::vector<Vertex3D> vertices;
    vertices.reserve(mesh.EdgeCount() * 2);
    for (const auto eh : mesh.edges()) {
        const auto heh = mesh.GetHalfedge(eh, 0);
        const auto v_from = mesh.GetFromVertex(heh);
        const auto v_to = mesh.GetToVertex(heh);
        vertices.emplace_back(mesh.GetPosition(v_from), mesh.GetNormal(v_from));
        vertices.emplace_back(mesh.GetPosition(v_to), mesh.GetNormal(v_to));
    }
    return vertices;
}

std::vector<Vertex3D> CreateVertexPoints(const Mesh &mesh) {
    std::vector<Vertex3D> vertices;
    vertices.reserve(mesh.VertexCount());
    for (const auto vh : mesh.vertices()) {
        vertices.emplace_back(mesh.GetPosition(vh), mesh.GetNormal(vh));
    }
    return vertices;
}

std::vector<Vertex3D> CreateNormalVertices(const Mesh &mesh, Element element) {
    std::vector<Vertex3D> vertices;
    if (element == Element::Vertex) {
        vertices.reserve(mesh.VertexCount() * 2);
        for (const auto vh : mesh.vertices()) {
            const auto vn = mesh.GetNormal(vh);
            const auto &voh_range = mesh.voh_range(vh);
            const float total_edge_length = std::reduce(voh_range.begin(), voh_range.end(), 0.f, [&](float total, const auto &heh) {
                return total + mesh.CalcEdgeLength(heh);
            });
            const float avg_edge_length = total_edge_length / mesh.GetValence(vh);
            const auto p = mesh.GetPosition(vh);
            vertices.emplace_back(p, vn);
            vertices.emplace_back(p + NormalIndicatorLengthScale * avg_edge_length * vn, vn);
        }
    } else if (element == Element::Face) {
        vertices.reserve(mesh.FaceCount() * 2);
        for (const auto fh : mesh.faces()) {
            const auto fn = mesh.GetNormal(fh);
            const auto p = mesh.CalcFaceCentroid(fh);
            vertices.emplace_back(p, fn);
            vertices.emplace_back(p + NormalIndicatorLengthScale * std::sqrt(mesh.CalcFaceArea(fh)) * fn, fn);
        }
    }
    return vertices;
}

BBox ComputeBoundingBox(const Mesh &mesh) {
    BBox bbox;
    for (const auto vh : mesh.vertices()) {
        const auto p = mesh.GetPosition(vh);
        bbox.Min = glm::min(bbox.Min, p);
        bbox.Max = glm::max(bbox.Max, p);
    }
    return bbox;
}

} // namespace MeshRender

// Returns `std::nullopt` if the entity is not Visible (and thus does not have a RenderInstance).
std::optional<uint32_t> GetModelBufferIndex(const entt::registry &r, entt::entity e) {
    if (e == entt::null || !r.all_of<Visible>(e)) return std::nullopt;
    return r.get<RenderInstance>(e).BufferIndex;
}

void UpdateModelBuffer(entt::registry &r, entt::entity e, const WorldMatrix &m) {
    if (const auto i = GetModelBufferIndex(r, e)) {
        r.get<ModelsBuffer>(r.get<MeshInstance>(e).MeshEntity).Buffer.Update(as_bytes(m), *i * sizeof(WorldMatrix));
    }
}
