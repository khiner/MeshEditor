#include "mesh/Neighborhood.h"

#include "SortUnique.h"
#include "gpu/NeighborhoodPushConstants.h"
#include "mesh/EdgeKey.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"

namespace {
NeighborhoodPushConstants Constants(const MeshStore &meshes, mtl::ComputeChain &chain, uint32_t id, std::span<const uint32_t> elements) {
    const auto &a = meshes.Arenas();
    return {
        .Connectivity = meshes.GetConnectivityRef(id),
        .VertexSlot = a.Vertices.Buffer.Slot,
        .FaceNormalSlot = a.BaseFaceNormals.Buffer.Slot,
        .VertexNormalSlot = a.BaseVertexNormals.Buffer.Slot,
        .VertexBlocksSlot = a.Vertices.Blocks.Buffer.Slot,
        .ScratchSlot = chain.Scratch.Buffer.Slot,
        .Elements = chain.Scratch.Allocate(elements).Offset,
        .ElementCount = uint32_t(elements.size()),
        .FacePrimitives = a.FacePrimitives.Ref(meshes.Get(id).FacePrimitives),
    };
}
template<typename T> Gathered<T> Gather(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> elements, MeshPass count, MeshPass fill) {
    constexpr uint32_t Words = sizeof(T) / sizeof(uint32_t);
    Gathered<T> gathered;
    gathered.First.assign(elements.size() + 1u, 0u);
    if (elements.empty()) return gathered;
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto pc = Constants(meshes, chain, id, elements);
    pc.Counts = chain.Scratch.Allocate(uint32_t(elements.size())).Offset;
    chain.Threads(pipelines[count], pc, pc.ElementCount);
    chain.Submit();
    const auto counts = chain.Scratch.Get({pc.Counts, pc.ElementCount});
    for (uint32_t i = 0u; i < elements.size(); ++i) gathered.First[i + 1u] = gathered.First[i] + counts[i];
    const uint64_t total = gathered.First.back();
    chain.Scratch.Buffer.Update(std::as_bytes(std::span{gathered.First}.first(elements.size())), uint64_t(pc.Counts) * sizeof(uint32_t));
    pc.Items = TakeScratch(chain, Words * total).Offset;
    chain.Threads(pipelines[fill], pc, pc.ElementCount);
    chain.Submit();
    gathered.Items = ReadScratch<T>(chain, pc.Items, total);
    return gathered;
}
} // namespace

Gathered<FaceCorner> FaceCorners(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> faces) {
    return Gather<FaceCorner>(meshes, pipelines, id, faces, MeshPass::NeighborhoodFaceCount, MeshPass::NeighborhoodFaceFill);
}
Gathered<VertexEdge> VertexEdges(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> vertices) {
    return Gather<VertexEdge>(meshes, pipelines, id, vertices, MeshPass::NeighborhoodVertexEdgeCount, MeshPass::NeighborhoodVertexEdgeFill);
}
Gathered<VertexCorner> VertexCorners(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> vertices) {
    return Gather<VertexCorner>(meshes, pipelines, id, vertices, MeshPass::NeighborhoodVertexCornerCount, MeshPass::NeighborhoodVertexCornerFill);
}

namespace {
template<typename T> std::vector<T> Direct(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> elements, MeshPass pass, uint32_t NeighborhoodPushConstants::*run) {
    constexpr uint32_t Words = sizeof(T) / sizeof(uint32_t);
    if (elements.empty()) return {};
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto pc = Constants(meshes, chain, id, elements);
    pc.*run = chain.Scratch.Allocate(Words * uint32_t(elements.size())).Offset;
    chain.Threads(pipelines[pass], pc, pc.ElementCount);
    chain.Submit();
    return ReadScratch<T>(chain, pc.*run, elements.size());
}
} // namespace

std::vector<CornerFacts> Corners(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> corners) {
    return Direct<CornerFacts>(meshes, pipelines, id, corners, MeshPass::NeighborhoodCorners, &NeighborhoodPushConstants::Items);
}
std::vector<vec3> VertexPositions(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> vertices) {
    return Direct<vec3>(meshes, pipelines, id, vertices, MeshPass::NeighborhoodPositions, &NeighborhoodPushConstants::Positions);
}
std::vector<vec3> FaceNormals(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> faces) {
    return Direct<vec3>(meshes, pipelines, id, faces, MeshPass::NeighborhoodFaceNormals, &NeighborhoodPushConstants::Positions);
}
std::vector<vec3> VertexNormals(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> vertices) {
    return Direct<vec3>(meshes, pipelines, id, vertices, MeshPass::NeighborhoodVertexNormals, &NeighborhoodPushConstants::Positions);
}

std::vector<uint32_t> FacePrimitives(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> faces) {
    return Direct<uint32_t>(meshes, pipelines, id, faces, MeshPass::NeighborhoodFacePrimitives, &NeighborhoodPushConstants::Items);
}

float VertexExtent(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> vertices) {
    if (vertices.empty()) return 0.f;
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto pc = Constants(meshes, chain, id, vertices);
    pc.Positions = chain.Scratch.Allocate(1u).Offset;
    chain.Groups(pipelines[MeshPass::NeighborhoodExtent], pc, 1u);
    chain.Submit();
    return std::bit_cast<float>(chain.Scratch.Get({pc.Positions, 1u})[0]);
}

std::vector<uint32_t> NearestVertices(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const vec3> points, float radius) {
    std::vector<uint32_t> nearest(points.size(), InvalidOffset);
    if (points.empty()) return nearest;
    const auto &a = meshes.Arenas();
    const auto leaves = a.VertexAggregates.Buffer.GetSpan<SelectionAggregate>();
    const auto all = meshes.GetBlockList(id, MeshStore::ElementDomain::Vertex).Blocks;
    const float radius_squared = radius * radius;
    std::vector<uint32_t> first{0u}, blocks;
    for (const auto &point : points) {
        const auto within = [&](const AABB &bounds) {
            const auto delta = Max(Max(bounds.Min - point, point - bounds.Max), vec3{});
            return Dot(delta, delta) <= radius_squared;
        };
        if (!a.SelectionTree.VisitBounds(3u * id, leaves, within, [&](uint32_t block) { blocks.push_back(block); })) blocks.insert(blocks.end(), all.begin(), all.end());
        first.push_back(uint32_t(blocks.size()));
    }
    if (blocks.empty()) return nearest;
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto pc = Constants(meshes, chain, id, blocks);
    pc.ElementCount = uint32_t(points.size());
    pc.Counts = chain.Scratch.Allocate(first).Offset;
    pc.Positions = chain.Upload(std::as_bytes(points)).Offset;
    pc.Items = chain.Scratch.Allocate(2u * pc.ElementCount).Offset;
    chain.Groups(pipelines[MeshPass::NeighborhoodNearest], pc, pc.ElementCount);
    chain.Submit();
    const auto words = chain.Scratch.Get({pc.Items, 2u * pc.ElementCount});
    for (uint32_t i = 0u; i < points.size(); ++i) {
        if (std::bit_cast<float>(words[2u * i + 1u]) < radius_squared) nearest[i] = words[2u * i];
    }
    return nearest;
}

std::vector<uint32_t> NearestVertexOrdinals(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const vec3> points) {
    const Mesh mesh{meshes, id};
    auto ordinals = NearestVertices(meshes, pipelines, id, points, std::numeric_limits<float>::infinity());
    for (auto &ordinal : ordinals) ordinal = mesh.VertexOrdinal(Mesh::VH{ordinal});
    return ordinals;
}

std::vector<uint32_t> TriangleVertices(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id) {
    std::vector<uint32_t> corners;
    for (const auto triangle : meshes.TriangleView(id)) corners.insert(corners.end(), {triangle.x, triangle.y, triangle.z});
    std::vector<uint32_t> vertices;
    vertices.reserve(corners.size());
    for (const auto &corner : Corners(meshes, pipelines, id, corners)) vertices.push_back(corner.Vertex);
    return vertices;
}

std::vector<uint32_t> OrdinalTriangles(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id) {
    auto triangles = TriangleVertices(meshes, pipelines, id);
    const Mesh mesh{meshes, id};
    std::vector<uint32_t> ordinals;
    uint32_t ordinal = 0u;
    for (const auto v : mesh.vertices()) {
        if (*v >= ordinals.size()) ordinals.resize(*v + 1u, InvalidOffset);
        ordinals[*v] = ordinal++;
    }
    for (auto &vertex : triangles) vertex = ordinals[vertex];
    return triangles;
}

std::vector<vec3> OrdinalPositions(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id) {
    return VertexPositions(meshes, pipelines, id, HandleList(Mesh{meshes, id}.vertices()));
}

FaceLoops ReadFaceLoops(const MeshStore &meshes, const MeshPipelines &pipelines, const Mesh &mesh) {
    const auto id = mesh.GetStoreId();
    auto faces = HandleList(mesh.faces());
    const auto vertex_handles = HandleList(mesh.vertices());
    auto corners = FaceCorners(meshes, pipelines, id, faces);
    const auto gathered = VertexPositions(meshes, pipelines, id, vertex_handles);
    std::vector<vec3> positions(vertex_handles.empty() ? 0u : vertex_handles.back() + 1u);
    for (uint32_t i = 0u; i < vertex_handles.size(); ++i) positions[vertex_handles[i]] = gathered[i];
    return {std::move(faces), std::move(corners), std::move(positions)};
}

LocalMesh::LocalMesh(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> vertices) {
    std::vector<uint32_t> set{vertices.begin(), vertices.end()};
    SortUnique(set);
    for (uint32_t i = 0u; i < set.size(); ++i) VertexIndex.emplace(set[i], i);
    VertexEdgesOf = VertexEdges(meshes, pipelines, id, set);
    VertexCornersOf = VertexCorners(meshes, pipelines, id, set);
    std::vector<uint32_t> faces;
    for (const auto &corner : VertexCornersOf.Items) {
        if (corner.Face != InvalidOffset) faces.push_back(corner.Face);
    }
    SortUnique(faces);
    for (uint32_t i = 0u; i < faces.size(); ++i) FaceIndex.emplace(faces[i], i);
    FaceCornersOf = FaceCorners(meshes, pipelines, id, faces);
    std::vector<uint32_t> positioned{set};
    for (uint32_t f = 0u; f < faces.size(); ++f) {
        const auto corners = FaceCornersOf.Of(f);
        for (size_t k = 0u; k < corners.size(); ++k) {
            const auto &corner = corners[k], &previous = corners[(k + corners.size() - 1u) % corners.size()], &next = corners[(k + 1u) % corners.size()];
            CornerOf.emplace(corner.Halfedge, CornerFacts{corner.Vertex, previous.Vertex, corner.Edge, faces[f], next.Halfedge, previous.Halfedge, corner.Opposite});
            positioned.push_back(corner.Vertex);
        }
    }
    std::vector<uint32_t> firsts;
    for (uint32_t i = 0u; i < set.size(); ++i) {
        for (const auto &edge : VertexEdgesOf.Of(i)) {
            SidesOf.emplace(EdgeKey(set[i], edge.Other), edge.Sides);
            positioned.push_back(edge.Other);
            if (!CornerOf.contains(edge.Halfedge)) firsts.push_back(edge.Halfedge);
        }
    }
    SortUnique(firsts);
    const auto facts = ::Corners(meshes, pipelines, id, firsts);
    for (uint32_t i = 0u; i < firsts.size(); ++i) CornerOf.emplace(firsts[i], facts[i]);
    SortUnique(positioned);
    const auto positions = VertexPositions(meshes, pipelines, id, positioned);
    for (uint32_t i = 0u; i < positioned.size(); ++i) PositionOf.emplace(positioned[i], positions[i]);
}

uint32_t LocalMesh::Sides(uint32_t a, uint32_t b) const {
    const auto found = SidesOf.find(EdgeKey(a, b));
    return found == SidesOf.end() ? 0u : found->second;
}
