#include "MeshRender.h"

#include "WorldMatrix.h"

#include <entt/entity/registry.hpp>

#include <numeric>
#include <ranges>

using std::ranges::all_of;

using namespace he;

namespace MeshRender {

std::vector<BBox> CreateFaceBoundingBoxes(const Mesh &mesh) {
    std::vector<BBox> boxes;
    boxes.reserve(mesh.FaceCount());
    for (auto fh : mesh.faces()) {
        BBox box;
        for (auto vh : mesh.fv_range(fh)) {
            const auto p = mesh.GetPosition(vh);
            box.Min = glm::min(box.Min, p);
            box.Max = glm::max(box.Max, p);
        }
        boxes.emplace_back(std::move(box));
    }
    return boxes;
}

std::vector<uint> CreateFaceIndices(const Mesh &mesh) {
    return mesh.CreateTriangulatedFaceIndices();
}

std::vector<uint> CreateEdgeIndices(const Mesh &mesh) {
    return mesh.CreateEdgeIndices();
}

std::vector<uint> CreateVertexIndices(const Mesh &mesh) {
    std::vector<uint> indices;
    indices.reserve(mesh.VertexCount());
    for (uint i = 0; i < mesh.VertexCount(); ++i) {
        indices.push_back(i);
    }
    return indices;
}

std::vector<uint32_t> CreateFaceElementIds(const Mesh &mesh) {
    std::vector<uint32_t> ids;
    ids.reserve(mesh.FaceCount() * 3);
    for (const auto fh : mesh.faces()) {
        const uint32_t id = *fh + 1;
        for (uint32_t i = 0; i < mesh.GetValence(fh); ++i) {
            ids.push_back(id);
        }
    }
    return ids;
}

std::vector<uint> CreateNormalIndices(const Mesh &mesh, Element element) {
    if (element == Element::None || element == Element::Edge) return {};

    const auto n = element == Element::Face ? mesh.FaceCount() : mesh.VertexCount();
    std::vector<uint> indices;
    indices.reserve(n * 2);
    for (uint i = 0; i < n; ++i) {
        indices.emplace_back(i * 2);
        indices.emplace_back(i * 2 + 1);
    }
    return indices;
}

namespace {
// Used as an intermediate for creating render vertices
struct VerticesHandle {
    AnyHandle Parent; // A vertex can belong to itself, an edge, or a face.
    std::vector<VH> VHs;
};
} // namespace

std::vector<Vertex3D> CreateFaceVertices(
    const Mesh &mesh,
    bool smooth_shading
) {
    std::vector<Vertex3D> vertices;
    vertices.reserve(mesh.FaceCount() * 3); // Lower bound assuming all faces are triangles

    for (const auto fh : mesh.faces()) {
        for (const auto vh : mesh.fv_range(fh)) {
            vertices.emplace_back(mesh.GetPosition(vh), smooth_shading ? mesh.GetNormal(vh) : mesh.GetNormal(fh));
        }
    }

    return vertices;
}

std::vector<Vertex3D> CreateEdgeVertices(
    const Mesh &mesh
) {
    std::vector<Vertex3D> vertices;
    vertices.reserve(mesh.EdgeCount() * 2);

    for (const auto eh : mesh.edges()) {
        const auto heh = mesh.GetHalfedge(eh, 0);
        const auto v_from = mesh.GetFromVertex(heh);
        const auto v_to = mesh.GetToVertex(heh);
        for (const auto vh : {v_from, v_to}) {
            vertices.emplace_back(mesh.GetPosition(vh), mesh.GetNormal(vh));
        }
    }

    return vertices;
}

std::vector<Vertex3D> CreateVertexPoints(
    const Mesh &mesh
) {
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
        // Line for each vertex normal, with length scaled by the average edge length.
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
        // Line for each face normal, with length scaled by the face area.
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
