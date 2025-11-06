#include "MeshRender.h"

#include "BVH.h"

#include <algorithm>
#include <ranges>

using namespace he;

namespace MeshRender {

std::vector<BBox> CreateFaceBoundingBoxes(const PolyMesh &polymesh) {
    std::vector<BBox> boxes;
    boxes.reserve(polymesh.FaceCount());
    for (auto fh : polymesh.faces()) {
        BBox box;
        for (auto vh : polymesh.fv_range(fh)) {
            const auto p = polymesh.GetPosition(vh);
            box.Min = glm::min(box.Min, p);
            box.Max = glm::max(box.Max, p);
        }
        boxes.emplace_back(std::move(box));
    }
    return boxes;
}

RenderBuffers CreateBvhBuffers(const BVH &bvh, vec4 color) {
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

std::vector<uint> CreateIndices(const PolyMesh &polymesh, Element element) {
    switch (element) {
        case Element::Vertex: return polymesh.CreateTriangleIndices();
        case Element::Edge: return polymesh.CreateEdgeIndices();
        case Element::Face: return polymesh.CreateTriangulatedFaceIndices();
        case Element::None: return {};
    }
}

std::vector<uint> CreateNormalIndices(const PolyMesh &polymesh, Element element) {
    if (element == Element::None || element == Element::Edge) return {};

    const auto n = element == Element::Face ? polymesh.FaceCount() : polymesh.VertexCount();
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

std::vector<Vertex3D> CreateVertices(
    const PolyMesh &polymesh,
    Element render_element,
    const AnyHandle &selected,
    const std::unordered_set<AnyHandle, AnyHandleHash> &highlighted
) {
    std::vector<VerticesHandle> handles;
    if (render_element == Element::Vertex) {
        handles.reserve(polymesh.VertexCount());
        for (const auto vh : polymesh.vertices()) handles.emplace_back(vh, std::vector<VH>{vh});
    } else if (render_element == Element::Edge) {
        handles.reserve(polymesh.EdgeCount() * 2);
        for (const auto eh : polymesh.edges()) {
            const auto heh = polymesh.GetHalfedge(eh, 0);
            handles.emplace_back(eh, std::vector<VH>{polymesh.GetFromVertex(heh), polymesh.GetToVertex(heh)});
        }
    } else if (render_element == Element::Face) {
        handles.reserve(polymesh.FaceCount() * 3); // Lower bound assuming all faces are triangles.
        for (const auto fh : polymesh.faces()) {
            for (const auto vh : polymesh.fv_range(fh)) handles.emplace_back(fh, std::vector<VH>{vh});
        }
    }

    std::vector<Vertex3D> vertices;
    for (const auto &handle : handles) {
        const auto parent = handle.Parent;
        const auto normal = render_element == Element::Vertex || render_element == Element::Edge ?
            polymesh.GetNormal(handle.VHs[0]) :
            polymesh.GetNormal(FH(handle.Parent));
        for (const auto vh : handle.VHs) {
            const bool is_selected =
                selected == vh || selected == parent ||
                // Note: If we want to support `highlighted` having handle types (not just the selection `highlight`),
                // we need to update the methods to accept sets of `AnyHandle` instead of just one.
                (render_element == Element::Vertex && (polymesh.VertexBelongsToFace(parent, selected) || polymesh.VertexBelongsToEdge(parent, selected))) ||
                (render_element == Element::Edge && polymesh.EdgeBelongsToFace(parent, selected)) ||
                (render_element == Element::Face && polymesh.VertexBelongsToFaceEdge(vh, parent, selected));
            const bool is_highlighted = highlighted.contains(vh) || highlighted.contains(parent);
            const auto color = is_selected        ? SelectedColor :
                is_highlighted                    ? HighlightedColor :
                render_element == Element::Vertex ? VertexColor :
                render_element == Element::Edge   ? EdgeColor :
                                                    polymesh.GetColor(FH(parent));
            vertices.emplace_back(polymesh.GetPosition(vh), normal, color);
        }
    }

    return vertices;
}

std::vector<Vertex3D> CreateNormalVertices(const PolyMesh &polymesh, Element element) {
    std::vector<Vertex3D> vertices;
    if (element == Element::Vertex) {
        // Line for each vertex normal, with length scaled by the average edge length.
        vertices.reserve(polymesh.VertexCount() * 2);
        for (const auto vh : polymesh.vertices()) {
            const auto vn = polymesh.GetNormal(vh);
            const auto &voh_range = polymesh.voh_range(vh);
            const float total_edge_length = std::reduce(voh_range.begin(), voh_range.end(), 0.f, [&](float total, const auto &heh) {
                return total + polymesh.CalcEdgeLength(heh);
            });
            const float avg_edge_length = total_edge_length / polymesh.GetValence(vh);
            const auto p = polymesh.GetPosition(vh);
            vertices.emplace_back(p, vn, VertexNormalIndicatorColor);
            vertices.emplace_back(p + NormalIndicatorLengthScale * avg_edge_length * vn, vn, VertexNormalIndicatorColor);
        }
    } else if (element == Element::Face) {
        // Line for each face normal, with length scaled by the face area.
        vertices.reserve(polymesh.FaceCount() * 2);
        for (const auto fh : polymesh.faces()) {
            const auto fn = polymesh.GetNormal(fh);
            const auto p = polymesh.CalcFaceCentroid(fh);
            vertices.emplace_back(p, fn, FaceNormalIndicatorColor);
            vertices.emplace_back(p + NormalIndicatorLengthScale * std::sqrt(polymesh.CalcFaceArea(fh)) * fn, fn, FaceNormalIndicatorColor);
        }
    }
    return vertices;
}

BBox ComputeBoundingBox(const PolyMesh &polymesh) {
    BBox bbox;
    for (const auto vh : polymesh.vertices()) {
        const auto p = polymesh.GetPosition(vh);
        bbox.Min = glm::min(bbox.Min, p);
        bbox.Max = glm::max(bbox.Max, p);
    }
    return bbox;
}

} // namespace MeshRender
