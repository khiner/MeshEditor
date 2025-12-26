#include "MeshRender.h"

#include "BVH.h"
#include "WorldMatrix.h"

#include <entt/entity/registry.hpp>

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

MeshRenderBuffers CreateBvhBuffers(const BVH &bvh, vec4 color) {
    std::vector<BBox> boxes = bvh.CreateInternalBoxes();
    std::vector<Vertex3D> vertices;
    vertices.reserve(boxes.size() * 8);
    std::vector<uint> indices;
    indices.reserve(boxes.size() * BBox::EdgeIndices.size());
    for (uint i = 0; i < boxes.size(); ++i) {
        const auto &box = boxes[i];
        for (auto &corner : box.Corners()) vertices.emplace_back(corner, vec3{}, color);

        const uint index_offset = i * 8;
        for (const auto &index : BBox::EdgeIndices) indices.emplace_back(index_offset + index);
    }
    return {std::move(vertices), std::move(indices)};
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
    bool smooth_shading,
    const std::unordered_set<VH> &highlighted_vertices,
    const std::unordered_set<FH> &selected_faces,
    const std::unordered_set<FH> &active_faces
) {
    std::vector<Vertex3D> vertices;
    vertices.reserve(mesh.FaceCount() * 3); // Lower bound assuming all faces are triangles

    const auto mesh_color = mesh.GetColor();
    for (const auto fh : mesh.faces()) {
        const bool is_active = active_faces.contains(fh);
        const bool is_selected = selected_faces.contains(fh);
        for (const auto vh : mesh.fv_range(fh)) {
            const auto color = is_active          ? ActiveColor :
                is_selected                       ? SelectedColor :
                highlighted_vertices.contains(vh) ? HighlightedColor :
                                                    mesh_color;
            vertices.emplace_back(mesh.GetPosition(vh), smooth_shading ? mesh.GetNormal(vh) : mesh.GetNormal(fh), color);
        }
    }

    return vertices;
}

std::vector<Vertex3D> CreateEdgeVertices(
    const Mesh &mesh,
    Element edit_mode,
    const std::unordered_set<VH> &selected_vertices,
    const std::unordered_set<EH> &selected_edges,
    const std::unordered_set<VH> &active_vertices,
    const std::unordered_set<EH> &active_edges
) {
    std::vector<Vertex3D> vertices;
    vertices.reserve(mesh.EdgeCount() * 2);

    for (const auto eh : mesh.edges()) {
        const bool edge_active = active_edges.contains(eh);
        const bool edge_selected = selected_edges.contains(eh);
        const auto heh = mesh.GetHalfedge(eh, 0);
        const auto v_from = mesh.GetFromVertex(heh);
        const auto v_to = mesh.GetToVertex(heh);
        for (const auto vh : {v_from, v_to}) {
            const bool is_active =
                edge_active                  ? true :
                edit_mode == Element::Vertex ? active_vertices.contains(vh) :
                                               active_vertices.contains(v_from) && active_vertices.contains(v_to);
            const bool is_selected =
                edge_selected                ? true :
                edit_mode == Element::Vertex ? selected_vertices.contains(vh) :
                                               selected_vertices.contains(v_from) && selected_vertices.contains(v_to);
            const auto color = is_active ? ActiveColor : is_selected ? SelectedColor :
                                                                       EdgeColor;
            vertices.emplace_back(mesh.GetPosition(vh), mesh.GetNormal(vh), color);
        }
    }

    return vertices;
}

std::vector<Vertex3D> CreateVertexPoints(
    const Mesh &mesh,
    Element edit_mode,
    const std::unordered_set<VH> &selected_vertices,
    const std::unordered_set<VH> &active_vertices
) {
    std::vector<Vertex3D> vertices;
    vertices.reserve(mesh.VertexCount());
    for (const auto vh : mesh.vertices()) {
        const bool is_active = edit_mode == Element::Vertex && active_vertices.contains(vh);
        const bool is_selected = edit_mode == Element::Vertex && selected_vertices.contains(vh);
        const auto color = is_active ? ActiveColor : is_selected ? SelectedColor :
                                                                   UnselectedVertexEditColor;
        vertices.emplace_back(mesh.GetPosition(vh), mesh.GetNormal(vh), color);
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
            vertices.emplace_back(p, vn, VertexNormalIndicatorColor);
            vertices.emplace_back(p + NormalIndicatorLengthScale * avg_edge_length * vn, vn, VertexNormalIndicatorColor);
        }
    } else if (element == Element::Face) {
        // Line for each face normal, with length scaled by the face area.
        vertices.reserve(mesh.FaceCount() * 2);
        for (const auto fh : mesh.faces()) {
            const auto fn = mesh.GetNormal(fh);
            const auto p = mesh.CalcFaceCentroid(fh);
            vertices.emplace_back(p, fn, FaceNormalIndicatorColor);
            vertices.emplace_back(p + NormalIndicatorLengthScale * std::sqrt(mesh.CalcFaceArea(fh)) * fn, fn, FaceNormalIndicatorColor);
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
