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
void RecordCount(mtl::ComputeChain &chain, const MeshPipelines &pipelines, NeighborhoodPushConstants &pc, MeshPass count) {
    pc.Counts = chain.Scratch.Allocate(pc.ElementCount).Offset;
    chain.Threads(pipelines[count], pc, pc.ElementCount);
}
// Scans the submitted counts into each element's first item, which the fill pass reads in place of its count, and records the fill.
template<typename T> std::vector<uint32_t> RecordFill(mtl::ComputeChain &chain, const MeshPipelines &pipelines, NeighborhoodPushConstants &pc, MeshPass fill) {
    constexpr uint32_t Words = sizeof(T) / sizeof(uint32_t);
    std::vector<uint32_t> first(pc.ElementCount + 1u, 0u);
    const auto counts = chain.Scratch.Get({pc.Counts, pc.ElementCount});
    for (uint32_t i = 0u; i < pc.ElementCount; ++i) first[i + 1u] = first[i] + counts[i];
    chain.Scratch.Buffer.Update(std::as_bytes(std::span{first}.first(pc.ElementCount)), uint64_t(pc.Counts) * sizeof(uint32_t));
    pc.Items = TakeScratch(chain, uint64_t(Words) * first.back()).Offset;
    chain.Threads(pipelines[fill], pc, pc.ElementCount);
    return first;
}
template<typename T> void RecordDirect(mtl::ComputeChain &chain, const MeshPipelines &pipelines, NeighborhoodPushConstants &pc, MeshPass pass, uint32_t NeighborhoodPushConstants::*run) {
    constexpr uint32_t Words = sizeof(T) / sizeof(uint32_t);
    pc.*run = chain.Scratch.Allocate(Words * pc.ElementCount).Offset;
    chain.Threads(pipelines[pass], pc, pc.ElementCount);
}
template<typename T> std::vector<T> Direct(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> elements, MeshPass pass, uint32_t NeighborhoodPushConstants::*run) {
    if (elements.empty()) return {};
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto pc = Constants(meshes, chain, id, elements);
    RecordDirect<T>(chain, pipelines, pc, pass, run);
    chain.Submit();
    return ReadScratch<T>(chain, pc.*run, elements.size());
}

// The elements of several records in record order, each beside its record's index, with each record's first element.
struct Listing {
    std::vector<uint32_t> Elements, Records, Starts{0u};

    void Add(std::span<const uint32_t> handles) {
        Elements.insert(Elements.end(), handles.begin(), handles.end());
        Records.insert(Records.end(), handles.size(), uint32_t(Starts.size() - 1u));
        Starts.push_back(uint32_t(Elements.size()));
    }
    uint32_t First(uint32_t record) const { return Starts[record]; }
    std::span<const uint32_t> Of(uint32_t record) const { return std::span{Elements}.subspan(Starts[record], Starts[record + 1u] - Starts[record]); }
};
// The scratch words a listing takes for one record's elements, with `words` of results per element.
uint64_t ListingWords(uint64_t elements, uint32_t words) { return (2u + words) * elements + sizeof(NeighborhoodRecord) / sizeof(uint32_t); }
// The constants of a pass over the listed elements of the records, each element resolving through its own record.
NeighborhoodPushConstants Constants(const MeshStore &meshes, mtl::ComputeChain &chain, std::span<const uint32_t> ids, const Listing &listing) {
    const auto &a = meshes.Arenas();
    std::vector<NeighborhoodRecord> records;
    records.reserve(ids.size());
    for (const auto id : ids) records.push_back({meshes.GetElementTables(id), a.FacePrimitives.Ref(meshes.Get(id).FacePrimitives)});
    auto pc = Constants(meshes, chain, ids.front(), listing.Elements);
    pc.Records = chain.Scratch.Allocate(listing.Records).Offset;
    pc.RecordTable = chain.Upload(std::as_bytes(std::span{records})).Offset;
    return pc;
}
// A lane's gather of one record's listed elements, the first of them in the listing the gather shares with the other pending lanes, and its elements' first items once the host scanned the counts.
template<typename T> struct GatherState {
    uint32_t Record;
    std::vector<uint32_t> Elements;
    uint32_t First{};
    std::vector<uint32_t> Starts{};
    uint32_t Items{};
    std::function<void(Gathered<T>)> Then;
};
// Records the count pass over the record's elements with the other pending lanes' gathers through the same passes, scans the counts on the host once it completes, records the fill pass, and calls `then` with the record's items once that completes.
template<typename T> void RecordGather(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, std::span<const uint32_t> elements, MeshPass count, MeshPass fill, std::function<void(Gathered<T>)> then) {
    if (elements.empty()) return then({.First = {0u}});
    auto state = std::make_shared<GatherState<T>>(GatherState<T>{.Record = id, .Elements = {elements.begin(), elements.end()}, .Then = std::move(then)});
    chain.Lane(id, [&] {
        chain.Gather(&pipelines[count], std::vector{state}, ListingWords(elements.size(), 1u), [&meshes, &pipelines, &chain, fill, count](std::vector<std::shared_ptr<GatherState<T>>> states) {
            Listing listing;
            std::vector<uint32_t> ids;
            for (const auto &gathered : states) {
                gathered->First = uint32_t(listing.Elements.size());
                ids.push_back(gathered->Record);
                listing.Add(gathered->Elements);
            }
            auto pc = Constants(meshes, chain, ids, listing);
            RecordCount(chain, pipelines, pc, count);
            chain.AfterSubmit([&pipelines, &chain, pc, fill, states = std::move(states)] mutable {
                const auto first = RecordFill<T>(chain, pipelines, pc, fill);
                for (const auto &gathered : states) {
                    gathered->Starts.assign(first.begin() + gathered->First, first.begin() + gathered->First + gathered->Elements.size() + 1u);
                    gathered->Items = pc.Items;
                }
            });
        });
    });
    chain.AfterSubmit([&chain, state] {
        chain.AfterSubmit([&chain, state] {
            constexpr uint32_t Words = sizeof(T) / sizeof(uint32_t);
            const auto base = state->Starts.front();
            Gathered<T> gathered{.Items = ReadScratch<T>(chain, state->Items + Words * base, state->Starts.back() - base)};
            for (const auto start : state->Starts) gathered.First.push_back(start - base);
            state->Then(std::move(gathered));
        });
    });
}
// The record's gather on a chain of its own.
template<typename T> Gathered<T> Gather(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> elements, MeshPass count, MeshPass fill) {
    return mtl::Drained<Gathered<T>>(meshes.BufferContext(), [&](mtl::ComputeChain &chain, auto then) { RecordGather<T>(meshes, pipelines, chain, id, elements, count, fill, std::move(then)); });
}
// Visits the records in chunks within the scratch budget, `words_of(i)` giving record i's demand, each chunk on a chain reserving its words.
void ForEachChunk(const MeshStore &meshes, uint32_t count, auto &&words_of, auto &&visit) {
    const auto split = ChunkByScratch(count, ScratchWordBudget, [&](uint32_t i) { return uint32_t(std::min<uint64_t>(words_of(i), UINT32_MAX)); });
    for (const auto chunk : split.Chunks) {
        if (chunk.Count == 0u) continue;
        uint64_t words = 1u;
        for (uint32_t i = chunk.Offset; i < chunk.Offset + chunk.Count; ++i) words += words_of(i);
        mtl::ComputeChain chain{meshes.BufferContext(), words};
        visit(chain, chunk.Offset, chunk.Count);
    }
}
// Each record's results of a direct pass over its listed elements, one submit per chunk.
template<typename T> std::vector<std::vector<T>> Direct(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const RecordElements> records, MeshPass pass, uint32_t NeighborhoodPushConstants::*run) {
    constexpr uint32_t Words = sizeof(T) / sizeof(uint32_t);
    std::vector<std::vector<T>> values(records.size());
    ForEachChunk(meshes, uint32_t(records.size()), [&](uint32_t i) { return ListingWords(records[i].Elements.size(), Words); }, [&](mtl::ComputeChain &chain, uint32_t first, uint32_t count) {
        Listing listing;
        std::vector<uint32_t> ids;
        for (const auto &[id, elements] : records.subspan(first, count)) {
            listing.Add(elements);
            ids.push_back(id);
        }
        if (listing.Elements.empty()) return;
        auto pc = Constants(meshes, chain, ids, listing);
        RecordDirect<T>(chain, pipelines, pc, pass, run);
        chain.Submit();
        for (uint32_t k = 0u; k < count; ++k) values[first + k] = ReadScratch<T>(chain, pc.*run + Words * listing.First(k), listing.Of(k).size());
    });
    return values;
}
} // namespace

Gathered<FaceCorner> FaceCorners(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> faces) {
    return Gather<FaceCorner>(meshes, pipelines, id, faces, MeshPass::NeighborhoodFaceCount, MeshPass::NeighborhoodFaceFill);
}
Gathered<VertexEdge> VertexEdges(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> vertices) {
    return Gather<VertexEdge>(meshes, pipelines, id, vertices, MeshPass::NeighborhoodVertexEdgeCount, MeshPass::NeighborhoodVertexEdgeFill);
}
void RecordFaceCorners(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, std::span<const uint32_t> faces, std::function<void(Gathered<FaceCorner>)> then) {
    RecordGather<FaceCorner>(meshes, pipelines, chain, id, faces, MeshPass::NeighborhoodFaceCount, MeshPass::NeighborhoodFaceFill, std::move(then));
}
void RecordVertexEdges(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, std::span<const uint32_t> vertices, std::function<void(Gathered<VertexEdge>)> then) {
    RecordGather<VertexEdge>(meshes, pipelines, chain, id, vertices, MeshPass::NeighborhoodVertexEdgeCount, MeshPass::NeighborhoodVertexEdgeFill, std::move(then));
}
std::vector<vec3> VertexPositions(const MeshStore &meshes, uint32_t id, std::span<const uint32_t> vertices) {
    const Mesh mesh{meshes, id};
    const auto &arena = meshes.Arenas().Vertices;
    return vertices | std::views::transform([&](uint32_t vertex) { return arena.Get({mesh.GetConnectivity().VertexAddress(vertex).Value(), 1u})[0].Position; }) | std::ranges::to<std::vector>();
}
std::vector<vec3> FaceNormals(const MeshStore &meshes, uint32_t id, std::span<const uint32_t> faces) {
    const Mesh mesh{meshes, id};
    const auto normals = meshes.Arenas().BaseFaceNormals.Buffer.GetSpan<vec3>();
    return faces | std::views::transform([&](uint32_t face) { return normals[mesh.GetConnectivity().FaceAddress(face).Value()]; }) | std::ranges::to<std::vector>();
}
std::vector<vec3> VertexNormals(const MeshStore &meshes, uint32_t id, std::span<const uint32_t> vertices) {
    const Mesh mesh{meshes, id};
    const auto normals = meshes.Arenas().BaseVertexNormals.Buffer.GetSpan<vec3>();
    return vertices | std::views::transform([&](uint32_t vertex) { return normals[mesh.GetConnectivity().VertexAddress(vertex).Value()]; }) | std::ranges::to<std::vector>();
}

std::vector<uint32_t> FacePrimitives(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, std::span<const uint32_t> faces) {
    return Direct<uint32_t>(meshes, pipelines, id, faces, MeshPass::NeighborhoodFacePrimitives, &NeighborhoodPushConstants::Items);
}
std::vector<std::vector<uint32_t>> FacePrimitives(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const RecordElements> faces) {
    return Direct<uint32_t>(meshes, pipelines, faces, MeshPass::NeighborhoodFacePrimitives, &NeighborhoodPushConstants::Items);
}
std::vector<std::vector<vec3>> VertexPositions(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const RecordElements> vertices) {
    return Direct<vec3>(meshes, pipelines, vertices, MeshPass::NeighborhoodPositions, &NeighborhoodPushConstants::Positions);
}
std::vector<std::vector<vec3>> VertexNormals(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const RecordElements> vertices) {
    return Direct<vec3>(meshes, pipelines, vertices, MeshPass::NeighborhoodVertexNormals, &NeighborhoodPushConstants::Positions);
}
std::vector<std::vector<vec3>> FaceNormals(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const RecordElements> faces) {
    return Direct<vec3>(meshes, pipelines, faces, MeshPass::NeighborhoodFaceNormals, &NeighborhoodPushConstants::Positions);
}
std::vector<std::vector<uint32_t>> FaceValences(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const RecordElements> faces) {
    return Direct<uint32_t>(meshes, pipelines, faces, MeshPass::NeighborhoodFaceCount, &NeighborhoodPushConstants::Counts);
}
std::vector<std::vector<EdgeEnds>> EdgeEndpoints(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const RecordElements> edges) {
    return Direct<EdgeEnds>(meshes, pipelines, edges, MeshPass::NeighborhoodEdgeEnds, &NeighborhoodPushConstants::Items);
}
std::vector<std::vector<CornerFacts>> Corners(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const RecordElements> corners) {
    return Direct<CornerFacts>(meshes, pipelines, corners, MeshPass::NeighborhoodCorners, &NeighborhoodPushConstants::Items);
}

std::vector<float> VertexExtents(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const uint32_t> ids) {
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto extents = chain.Scratch.Allocate(uint32_t(ids.size()));
    std::ranges::fill(chain.Scratch.GetMutable(extents), 0u);
    const auto jobs = std::views::iota(0u, uint32_t(ids.size())) | std::views::transform([&](uint32_t i) {
                          return VertexExtentJob{
                              .Connectivity = meshes.GetConnectivityRef(ids[i]),
                              .Blocks = BlocksOf(meshes, ids[i], MeshStore::ElementDomain::Vertex),
                              .Extent = {chain.Scratch.Buffer.Slot, extents.Offset + i},
                              .VertexSlot = meshes.Arenas().Vertices.Buffer.Slot,
                          };
                      }) |
        std::ranges::to<std::vector>();
    mtl::GatherPass(chain, pipelines[MeshPass::NeighborhoodExtent], std::span{jobs}, [](const VertexExtentJob &job) { return job.Blocks.Count * MeshElementBlockSize; });
    chain.Submit();
    return chain.Scratch.Get(extents) | std::views::transform([](uint32_t bits) { return std::bit_cast<float>(bits); }) | std::ranges::to<std::vector>();
}

namespace {
// Records the pass of each query point's nearest live vertex of its record closer than its query's radius, and calls `then` with them after the submit that completes it.
void RecordNearest(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, std::span<const RecordPoints> queries, std::span<const float> radii, std::function<void(std::vector<std::vector<uint32_t>>)> then) {
    std::vector<std::vector<uint32_t>> nearest(queries.size());
    const auto &a = meshes.Arenas();
    const auto leaves = a.VertexAggregates.Buffer.GetSpan<SelectionAggregate>();
    // Each point's candidate blocks, whose vertices the pass reads by arena address, so points of several records share one pass.
    std::vector<uint32_t> first{0u}, blocks;
    std::vector<vec3> points;
    std::vector<float> squared;
    for (uint32_t q = 0u; q < queries.size(); ++q) {
        const auto [id, query_points] = queries[q];
        const float radius_squared = radii[q] * radii[q];
        nearest[q].assign(query_points.size(), InvalidOffset);
        if (query_points.empty()) continue;
        const auto all = meshes.GetBlockList(id, MeshStore::ElementDomain::Vertex).Blocks;
        for (const auto &point : query_points) {
            const auto within = [&](const AABB &bounds) {
                const auto delta = Max(Max(bounds.Min - point, point - bounds.Max), vec3{});
                return Dot(delta, delta) <= radius_squared;
            };
            if (!a.SelectionTree.VisitBounds(3u * id, leaves, within, [&](uint32_t block) { blocks.push_back(block); })) blocks.insert(blocks.end(), all.begin(), all.end());
            first.push_back(uint32_t(blocks.size()));
            squared.push_back(radius_squared);
        }
        points.insert(points.end(), query_points.begin(), query_points.end());
    }
    if (blocks.empty()) return then(std::move(nearest));
    auto pc = Constants(meshes, chain, queries.front().Id, blocks);
    pc.ElementCount = uint32_t(points.size());
    pc.Counts = chain.Scratch.Allocate(first).Offset;
    pc.Positions = chain.Upload(std::as_bytes(std::span{points})).Offset;
    pc.Items = chain.Scratch.Allocate(2u * pc.ElementCount).Offset;
    chain.Groups(pipelines[MeshPass::NeighborhoodNearest], pc, pc.ElementCount);
    chain.AfterSubmit([&chain, items = pc.Items, count = pc.ElementCount, squared = std::move(squared), nearest = std::move(nearest), then = std::move(then)] mutable {
        const auto words = chain.Scratch.Get({items, 2u * count});
        uint32_t i = 0u;
        for (auto &query_nearest : nearest) {
            for (auto &vertex : query_nearest) {
                if (std::bit_cast<float>(words[2u * i + 1u]) < squared[i]) vertex = words[2u * i];
                ++i;
            }
        }
        then(std::move(nearest));
    });
}
// A lane's query of its record's vertices nearest its points.
struct NearestQuery {
    uint32_t Record;
    std::vector<vec3> Points;
    float Radius;
    std::function<void(std::vector<uint32_t>)> Then;
};
} // namespace

void RecordNearestVertices(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, std::span<const vec3> points, float radius, std::function<void(std::vector<uint32_t>)> then) {
    chain.Lane(id, [&] {
        chain.Gather(&pipelines[MeshPass::NeighborhoodNearest], std::vector{NearestQuery{id, {points.begin(), points.end()}, radius, std::move(then)}}, 0u, [&meshes, &pipelines, &chain](std::vector<NearestQuery> gathered) {
            const auto queries = gathered | std::views::transform([](const auto &query) { return RecordPoints{query.Record, query.Points}; }) | std::ranges::to<std::vector>();
            const auto radii = gathered | std::views::transform(&NearestQuery::Radius) | std::ranges::to<std::vector>();
            RecordNearest(meshes, pipelines, chain, queries, radii, [gathered = std::move(gathered)](std::vector<std::vector<uint32_t>> nearest) {
                for (size_t q = 0u; q < gathered.size(); ++q) gathered[q].Then(std::move(nearest[q]));
            });
        });
    });
}

std::vector<std::vector<uint32_t>> NearestVertexOrdinals(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const RecordPoints> queries) {
    auto ordinals = mtl::Drained<std::vector<std::vector<uint32_t>>>(meshes.BufferContext(), [&](mtl::ComputeChain &chain, auto then) { RecordNearest(meshes, pipelines, chain, queries, std::vector<float>(queries.size(), std::numeric_limits<float>::infinity()), std::move(then)); });
    for (uint32_t q = 0u; q < queries.size(); ++q) {
        const Mesh mesh{meshes, queries[q].Id};
        for (auto &ordinal : ordinals[q]) ordinal = mesh.VertexOrdinal(Mesh::VH{ordinal});
    }
    return ordinals;
}

std::vector<std::vector<uint32_t>> TriangleVertices(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const uint32_t> ids) {
    std::vector<std::vector<uint32_t>> corners(ids.size());
    for (uint32_t k = 0u; k < ids.size(); ++k) {
        for (const auto triangle : meshes.TriangleView(ids[k])) corners[k].insert(corners[k].end(), {triangle.x, triangle.y, triangle.z});
    }
    std::vector<RecordElements> records;
    records.reserve(ids.size());
    for (uint32_t k = 0u; k < ids.size(); ++k) records.push_back({ids[k], corners[k]});
    const auto facts = Direct<CornerFacts>(meshes, pipelines, records, MeshPass::NeighborhoodCorners, &NeighborhoodPushConstants::Items);
    std::vector<std::vector<uint32_t>> vertices(ids.size());
    for (uint32_t k = 0u; k < ids.size(); ++k) {
        vertices[k].reserve(facts[k].size());
        for (const auto &corner : facts[k]) vertices[k].push_back(corner.Vertex);
    }
    return vertices;
}

std::vector<std::vector<uint32_t>> OrdinalTriangles(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const uint32_t> ids) {
    auto triangles = TriangleVertices(meshes, pipelines, ids);
    for (uint32_t k = 0u; k < ids.size(); ++k) {
        std::vector<uint32_t> ordinals;
        uint32_t ordinal = 0u;
        for (const auto v : Mesh{meshes, ids[k]}.vertices()) {
            if (*v >= ordinals.size()) ordinals.resize(*v + 1u, InvalidOffset);
            ordinals[*v] = ordinal++;
        }
        for (auto &vertex : triangles[k]) vertex = ordinals[vertex];
    }
    return triangles;
}

std::vector<std::vector<vec3>> OrdinalPositions(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const uint32_t> ids) {
    std::vector<std::vector<uint32_t>> vertices;
    std::vector<RecordElements> records;
    vertices.reserve(ids.size());
    records.reserve(ids.size());
    for (const auto id : ids) records.push_back({id, vertices.emplace_back(HandleList(Mesh{meshes, id}.vertices()))});
    return Direct<vec3>(meshes, pipelines, records, MeshPass::NeighborhoodPositions, &NeighborhoodPushConstants::Positions);
}

void VisitFaceLoops(const MeshStore &meshes, const MeshPipelines &pipelines, std::span<const uint32_t> ids, const std::function<void(uint32_t, FaceLoops &&)> &visit) {
    constexpr uint32_t CornerWords = sizeof(FaceCorner) / sizeof(uint32_t), PositionWords = sizeof(vec3) / sizeof(uint32_t);
    const auto &a = meshes.Arenas();
    // Each face with its count, the face corner items, and each vertex with its position.
    const auto words = [&](uint32_t i) {
        const auto &record = meshes.Get(ids[i]);
        return ListingWords(a.FaceTriangles.Count(record.FaceData), 1u) + uint64_t(CornerWords) * a.FaceCorners.Count(record.FaceCorners) + ListingWords(a.Vertices.Count(record.Vertices), PositionWords);
    };
    ForEachChunk(meshes, uint32_t(ids.size()), words, [&](mtl::ComputeChain &chain, uint32_t first, uint32_t count) {
        const auto chunk_ids = ids.subspan(first, count);
        Listing faces, vertices;
        for (const auto id : chunk_ids) {
            const Mesh mesh{meshes, id};
            faces.Add(HandleList(mesh.faces()));
            vertices.Add(HandleList(mesh.vertices()));
        }
        auto corners = Constants(meshes, chain, chunk_ids, faces), positions = Constants(meshes, chain, chunk_ids, vertices);
        RecordCount(chain, pipelines, corners, MeshPass::NeighborhoodFaceCount);
        chain.Submit();
        std::vector<uint32_t> item_first;
        chain.Concurrent([&] {
            item_first = RecordFill<FaceCorner>(chain, pipelines, corners, MeshPass::NeighborhoodFaceFill);
            RecordDirect<vec3>(chain, pipelines, positions, MeshPass::NeighborhoodPositions, &NeighborhoodPushConstants::Positions);
        });
        chain.Submit();
        for (uint32_t k = 0u; k < count; ++k) {
            const auto record_faces = faces.Of(k), record_vertices = vertices.Of(k);
            const auto base = item_first[faces.First(k)];
            Gathered<FaceCorner> loops{.First = std::vector<uint32_t>(record_faces.size() + 1u)};
            for (uint32_t f = 0u; f <= record_faces.size(); ++f) loops.First[f] = item_first[faces.First(k) + f] - base;
            loops.Items = ReadScratch<FaceCorner>(chain, corners.Items + CornerWords * base, loops.First.back());
            const auto gathered = ReadScratch<vec3>(chain, positions.Positions + PositionWords * vertices.First(k), record_vertices.size());
            std::vector<vec3> by_handle(record_vertices.empty() ? 0u : record_vertices.back() + 1u);
            for (uint32_t v = 0u; v < record_vertices.size(); ++v) by_handle[record_vertices[v]] = gathered[v];
            visit(first + k, {{record_faces.begin(), record_faces.end()}, std::move(loops), std::move(by_handle)});
        }
    });
}

FaceLoops ReadFaceLoops(const MeshStore &meshes, const MeshPipelines &pipelines, const Mesh &mesh) {
    FaceLoops loops;
    VisitFaceLoops(meshes, pipelines, std::array{mesh.GetStoreId()}, [&](uint32_t, FaceLoops &&read) { loops = std::move(read); });
    return loops;
}

namespace {
// The faces of the fans' corners, sorted.
std::vector<uint32_t> FanFaces(const Gathered<VertexCorner> &fans) {
    std::vector<uint32_t> faces;
    for (const auto &corner : fans.Items) {
        if (corner.Face != InvalidOffset) faces.push_back(corner.Face);
    }
    SortUnique(faces);
    return faces;
}
// The first halfedges of the edges at the set's vertices that no face corner holds, sorted.
std::vector<uint32_t> UncorneredHalfedges(std::span<const uint32_t> set, const Gathered<VertexEdge> &edges, const Gathered<FaceCorner> &corners) {
    auto cornered = corners.Items | std::views::transform(&FaceCorner::Halfedge) | std::ranges::to<std::vector>();
    SortUnique(cornered);
    std::vector<uint32_t> firsts;
    for (uint32_t i = 0u; i < set.size(); ++i) {
        for (const auto &edge : edges.Of(i)) {
            if (!std::ranges::binary_search(cornered, edge.Halfedge)) firsts.push_back(edge.Halfedge);
        }
    }
    SortUnique(firsts);
    return firsts;
}
// Records the facts of the listed corners in the record's lane of the chain and calls `then` with them after the submit that completes them.
void RecordCorners(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, std::span<const uint32_t> corners, std::function<void(std::vector<CornerFacts>)> then) {
    if (corners.empty()) return then({});
    auto pc = Constants(meshes, chain, id, corners);
    chain.Lane(id, [&] { RecordDirect<CornerFacts>(chain, pipelines, pc, MeshPass::NeighborhoodCorners, &NeighborhoodPushConstants::Items); });
    chain.AfterSubmit([&chain, items = pc.Items, count = corners.size(), then = std::move(then)] { then(ReadScratch<CornerFacts>(chain, items, count)); });
}
} // namespace

void RecordLocalMesh(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, std::span<const uint32_t> vertices, std::function<void(LocalMesh)> then) {
    std::vector<uint32_t> set{vertices.begin(), vertices.end()};
    SortUnique(set);
    RecordGather<VertexEdge>(meshes, pipelines, chain, id, set, MeshPass::NeighborhoodVertexEdgeCount, MeshPass::NeighborhoodVertexEdgeFill, [&meshes, &pipelines, &chain, id, set, then = std::move(then)](Gathered<VertexEdge> edges) mutable {
        RecordGather<VertexCorner>(meshes, pipelines, chain, id, set, MeshPass::NeighborhoodVertexCornerCount, MeshPass::NeighborhoodVertexCornerFill, [&meshes, &pipelines, &chain, id, set = std::move(set), edges = std::move(edges), then = std::move(then)](Gathered<VertexCorner> fans) mutable {
            auto faces = FanFaces(fans);
            RecordFaceCorners(meshes, pipelines, chain, id, faces, [&meshes, &pipelines, &chain, id, set = std::move(set), edges = std::move(edges), fans = std::move(fans), faces, then = std::move(then)](Gathered<FaceCorner> corners) mutable {
                const auto firsts = UncorneredHalfedges(set, edges, corners);
                RecordCorners(meshes, pipelines, chain, id, firsts, [&meshes, id, set = std::move(set), edges = std::move(edges), fans = std::move(fans), faces = std::move(faces), corners = std::move(corners), firsts, then = std::move(then)](std::vector<CornerFacts> facts) mutable {
                    then(LocalMesh{meshes, id, set, std::move(edges), std::move(fans), faces, std::move(corners), firsts, facts});
                });
            });
        });
    });
}

LocalMesh::LocalMesh(const MeshStore &meshes, uint32_t id, std::span<const uint32_t> set, Gathered<VertexEdge> edges, Gathered<VertexCorner> fans, std::span<const uint32_t> faces, Gathered<FaceCorner> corners, std::span<const uint32_t> firsts, std::span<const CornerFacts> facts)
    : VertexEdgesOf{std::move(edges)}, VertexCornersOf{std::move(fans)}, FaceCornersOf{std::move(corners)} {
    for (uint32_t i = 0u; i < set.size(); ++i) VertexIndex.emplace(set[i], i);
    for (uint32_t i = 0u; i < faces.size(); ++i) FaceIndex.emplace(faces[i], i);
    std::vector<uint32_t> positioned{set.begin(), set.end()};
    for (uint32_t f = 0u; f < faces.size(); ++f) {
        const auto face_corners = FaceCornersOf.Of(f);
        for (size_t k = 0u; k < face_corners.size(); ++k) {
            const auto &corner = face_corners[k], &previous = face_corners[(k + face_corners.size() - 1u) % face_corners.size()], &next = face_corners[(k + 1u) % face_corners.size()];
            CornerOf.emplace(corner.Halfedge, CornerFacts{corner.Vertex, previous.Vertex, corner.Edge, faces[f], next.Halfedge, previous.Halfedge, corner.Opposite});
            positioned.push_back(corner.Vertex);
        }
    }
    for (uint32_t i = 0u; i < set.size(); ++i) {
        for (const auto &edge : VertexEdgesOf.Of(i)) {
            SidesOf.emplace(EdgeKey(set[i], edge.Other), edge.Sides);
            positioned.push_back(edge.Other);
        }
    }
    for (uint32_t i = 0u; i < firsts.size(); ++i) CornerOf.emplace(firsts[i], facts[i]);
    SortUnique(positioned);
    const auto positions = VertexPositions(meshes, id, positioned);
    for (uint32_t i = 0u; i < positioned.size(); ++i) PositionOf.emplace(positioned[i], positions[i]);
}

uint32_t LocalMesh::Sides(uint32_t a, uint32_t b) const {
    const auto found = SidesOf.find(EdgeKey(a, b));
    return found == SidesOf.end() ? 0u : found->second;
}
