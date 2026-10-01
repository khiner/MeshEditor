#include "mesh/MeshClosure.h"

#include "gpu/MeshClosurePushConstants.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "state/Scene.h"

namespace {
uint32_t Groups(uint64_t count) { return uint32_t((count + 255u) / 256u); }

uint32_t IncidenceWord(mtl::ComputeChain &chain) {
    const auto word = chain.Scratch.Allocate(1u);
    chain.Scratch.GetMutable(word)[0] = 0u;
    return word.Offset;
}

// Fan corners are the corners at a vertex, so every corner belongs to exactly one fan.
struct HostIncidence {
    const MeshStore &Meshes;
    const MeshArenas &A;
    uint32_t Corners;
    uint32_t Fan(uint32_t v) const {
        const auto fans = A.VertexCorners.Buffer.GetSpan<uvec2>();
        return v < fans.size() ? fans[v].y : 0u;
    }
    // Adds both endpoints of an edge, whose first halfedge starts at its previous corner in a face or at its pair on a line.
    void AddEndpoints(uint32_t e, std::vector<uint32_t> &vertices) const {
        const auto h = A.EdgeHalfedges.Buffer.GetSpan<uint32_t>()[e];
        if (h == InvalidOffset) return;
        const auto face = A.HalfedgeFaces.Buffer.GetSpan<uint32_t>()[h];
        const auto ranges = A.FaceRanges.Buffer.GetSpan<uvec2>();
        const auto corners = A.FaceCorners.Buffer.GetSpan<uint32_t>();
        const auto previous = face == InvalidOffset ? A.OppositeHalfedges.Buffer.GetSpan<uint32_t>()[h] :
            h == ranges[face].x ? ranges[face].y - 1u : h - 1u;
        vertices.push_back(corners[h]);
        vertices.push_back(corners[previous]);
    }
    void AddLoopVertices(uint32_t f, std::vector<uint32_t> &vertices) const {
        const auto range = A.FaceRanges.Buffer.GetSpan<uvec2>()[f];
        const auto corners = A.FaceCorners.Buffer.GetSpan<uint32_t>();
        for (auto h = range.x; h < range.y; ++h) vertices.push_back(corners[h]);
    }
    // Adds the faces or edges of the vertex's fan corners.
    void AddFan(Element element, uint32_t v, std::vector<uint32_t> &handles) const {
        const auto fans = A.VertexCorners.Buffer.GetSpan<uvec2>();
        if (v >= fans.size()) return;
        const auto owners = (element == Element::Face ? A.HalfedgeFaces : A.HalfedgeEdges).Buffer.GetSpan<uint32_t>();
        for (const auto item : A.VertexFans.Items.Get({fans[v].x, fans[v].y}))
            if (owners[item.x] != InvalidOffset) handles.push_back(owners[item.x]);
    }
    // The fan corners of the vertices, counted once per listing.
    uint32_t Sum(std::span<const uint32_t> vertices) const {
        uint64_t sum = 0u;
        for (const auto v : vertices) sum += Fan(v);
        return uint32_t(std::min<uint64_t>(sum, Corners));
    }
    // Adds a seed element's vertices, which are a vertex, a face's loop vertices or a face-owned edge's endpoints, and returns its incidence.
    uint32_t Add(Element element, uint32_t handle, std::vector<uint32_t> &vertices) const {
        const auto first = vertices.size();
        if (element == Element::Face) {
            AddLoopVertices(handle, vertices);
            return uint32_t(vertices.size() - first);
        }
        if (element == Element::Vertex) vertices.push_back(handle);
        else AddEndpoints(handle, vertices);
        return Sum(std::span{vertices}.subspan(first));
    }
    // The fan corners of the distinct vertices, which it leaves ascending and unique.
    uint32_t Fans(std::vector<uint32_t> &vertices) const {
        std::ranges::sort(vertices);
        vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
        return Sum(vertices);
    }
};

uint32_t DomainBlocks(const MeshArenas &a, const MeshStore::Record &record, uint32_t d) {
    const auto blocks = [](const auto &arena, ElementSetRef set) { return set ? arena.Set(set).BlockCount : 0u; };
    return d == 0u ? blocks(a.Vertices, record.Vertices) : d == 1u ? blocks(a.FaceCorners, record.FaceCorners) :
        d == 2u ? blocks(a.FaceTriangles, record.FaceData) : blocks(a.EdgeHalfedges, record.EdgeData);
}
uint32_t DomainCapacity(const MeshArenas &a, uint32_t d) {
    return d == 0u ? a.Vertices.Capacity() : d == 1u ? a.FaceCorners.Capacity() : d == 2u ? a.FaceTriangles.Capacity() : a.EdgeHalfedges.Capacity();
}

MeshClosurePushConstants ClosureConstants(const MeshStore &meshes, uint32_t id) {
    const auto &a = meshes.Arenas();
    return {
        .Connectivity = meshes.GetConnectivityRef(id), .CornerSlot = a.FaceCorners.Buffer.Slot,
        .FaceCount = a.FaceTriangles.Count(meshes.Get(id).FaceData),
    };
}

// Allocates the level's output work for the domains `bounds` names and records the expansion of `input`.
MeshClosure EncodeClosure(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, uint32_t domain, const ClosureSeed &input,
                          std::array<uint32_t, 4> bounds, const ClosureSeed &retained) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    MeshClosure closure;
    closure.Bounds = bounds;
    closure.Elements[domain] = input.Work;
    if (!input.Count && !retained.Count) return closure;
    auto pc = ClosureConstants(meshes, id);
    std::vector<ElementWork> sorted;
    for (uint32_t d = 0u; d < 4u; ++d) {
        if (d == domain) continue;
        closure.Elements[d] = pc.Work[d] = AllocateElementWork(chain.Scratch, DomainCapacity(a, d), std::min(bounds[d], DomainBlocks(a, record, d)));
        sorted.push_back(pc.Work[d]);
    }
    pc.Input = input.Work;
    pc.InputDomain = domain;
    pc.InputBound = input.Count;
    pc.Retained = retained.Work;
    pc.RetainedBound = retained.Count;
    chain.Groups(GetMeshPipelines(r)[MeshPass::MeshClosureExpand], pc, std::max(Groups(input.Count), Groups(retained.Count)));
    EncodeSortElementWork(r, chain, sorted);
    return closure;
}
} // namespace

MeshClosure EncodeFaceClosure(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, const ClosureSeed &faces, const ClosureSeed &retained) {
    const uint64_t corners = faces.Incidence;
    const auto bound = [](uint64_t n) { return uint32_t(std::min<uint64_t>(n, UINT32_MAX)); };
    auto closure = EncodeClosure(r, chain, id, 2u, faces, {bound(corners + retained.Count), bound(corners), faces.Count, bound(corners)}, retained);
    const auto &meshes = r.Context.get<const MeshStore>();
    closure.VertexFans = uint32_t(std::min<uint64_t>(uint64_t(faces.LoopFans) + retained.Incidence, meshes.Arenas().FaceCorners.Count(meshes.Get(id).FaceCorners)));
    return closure;
}

MeshClosure EncodeVertexClosure(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, const ClosureSeed &vertices) {
    const auto bound = [](uint64_t n) { return uint32_t(std::min<uint64_t>(n, UINT32_MAX)); };
    const uint64_t fans = vertices.Incidence;
    return EncodeClosure(r, chain, id, 0u, vertices, {vertices.Count, bound(2u * fans), bound(fans), bound(2u * fans)}, {});
}

MeshClosure EncodeEdgeClosure(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, const ClosureSeed &edges, const ClosureSeed &retained) {
    const auto bound = [](uint64_t n) { return uint32_t(std::min<uint64_t>(n, UINT32_MAX)); };
    const uint64_t corners = 2ull * edges.Count;
    auto closure = EncodeClosure(r, chain, id, 3u, edges, {bound(corners + retained.Count), bound(corners), 0u, edges.Count}, retained);
    const auto &meshes = r.Context.get<const MeshStore>();
    closure.VertexFans = uint32_t(std::min<uint64_t>(uint64_t(edges.Incidence) + retained.Incidence, meshes.Arenas().FaceCorners.Count(meshes.Get(id).FaceCorners)));
    return closure;
}

namespace {
// The incidence word and domain of a face or edge closure element.
std::pair<uint32_t, uint32_t> IncidenceIndex(Element element) {
    if (element != Element::Face && element != Element::Edge) throw std::invalid_argument("Closure incidence is recorded for faces or edges.");
    return element == Element::Face ? std::pair{0u, 2u} : std::pair{1u, 3u};
}
} // namespace

void MeshClosure::EncodeIncidence(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, Element element) {
    const auto [index, d] = IncidenceIndex(element);
    const auto word = IncidenceWords[index] = IncidenceWord(chain);
    if (!Bounds[d]) return;
    auto pc = ClosureConstants(r.Context.get<const MeshStore>(), id);
    pc.Input = Elements[d];
    pc.InputDomain = d;
    pc.InputBound = Bounds[d];
    pc.Incidence = {chain.Scratch.Buffer.Slot, word};
    chain.Groups(GetMeshPipelines(r)[MeshPass::MeshClosureCount], pc, Groups(Bounds[d]));
}

void MeshClosure::Finish(const mtl::ComputeChain &chain) {
    for (uint32_t d = 0u; d < 4u; ++d) {
        Counts[d] = ElementWorkCount(chain.Scratch, Elements[d]);
        if (Counts[d] > Bounds[d]) throw std::logic_error("Mesh closure exceeds its host bound.");
    }
    for (uint32_t i = 0u; i < 2u; ++i)
        if (IncidenceWords[i] != InvalidOffset) Incidences[i] = chain.Scratch.Get({IncidenceWords[i], 1u})[0];
}

ClosureSeed MeshClosure::Seed(Element element) const {
    if (element == Element::Vertex) return {Elements[0], Bounds[0], VertexFans};
    const auto [index, d] = IncidenceIndex(element);
    if (IncidenceWords[index] == InvalidOffset) throw std::logic_error("A closure seed requires its recorded incidence.");
    return {Elements[d], Counts[d], Incidences[index]};
}

ClosureSeed EncodeEdgeVertices(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, const ClosureSeed &edges) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const auto count = uint32_t(std::min<uint64_t>(2ull * edges.Count, a.Vertices.Count(record.Vertices)));
    auto pc = ClosureConstants(meshes, id);
    pc.Work[0] = AllocateElementWork(chain.Scratch, a.Vertices.Capacity(), std::min(count, DomainBlocks(a, record, 0u)));
    if (!edges.Count) return {pc.Work[0]};
    pc.Input = edges.Work;
    pc.InputDomain = 3u;
    pc.InputBound = edges.Count;
    chain.Groups(GetMeshPipelines(r)[MeshPass::MeshClosureEdgeVertices], pc, Groups(edges.Count));
    EncodeSortElementWork(r, chain, std::span{&pc.Work[0], 1u});
    return {.Work = pc.Work[0], .Count = count, .Incidence = edges.Incidence, .Vertices = edges.Vertices, .All = edges.All};
}

ClosureSeed EncodeSelectionSeed(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, Element element, bool select_all) {
    if (element != Element::Vertex && element != Element::Edge && element != Element::Face) {
        throw std::invalid_argument("A closure seed requires vertices, edges or faces.");
    }
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const HostIncidence incidence{meshes, a, a.FaceCorners.Count(record.FaceCorners)};
    ClosureSeed result;
    const auto gather = [&](const auto &arena, ElementSetRef set) {
        const auto selection = meshes.GetSelectedElements(id, element);
        // A selection of every live element bounds as the whole mesh, with no host enumeration.
        if (select_all || selection.Count() == arena.Count(set)) {
            result.Count = arena.Count(set);
            result.Incidence = result.LoopFans = incidence.Corners;
            result.All = true;
            return PrepareElementMembershipWork(chain.Scratch, arena, set);
        }
        uint64_t sum = 0u;
        selection.ForEach([&](uint32_t handle) { sum += incidence.Add(element, handle, result.Vertices); });
        result.Count = selection.Count();
        result.Incidence = uint32_t(std::min<uint64_t>(sum, incidence.Corners));
        const auto fans = incidence.Fans(result.Vertices);
        if (element == Element::Face) result.LoopFans = fans;
        return PrepareSelectedMembershipWork(chain.Scratch, arena, set, selection, meshes.GetSelectionSlot(element));
    };
    const auto job = element == Element::Vertex ? gather(a.Vertices, record.Vertices) :
        element == Element::Face ? gather(a.FaceTriangles, record.FaceData) : gather(a.EdgeHalfedges, record.EdgeData);
    result.Work = job.Work;
    if (!result.Count) return result;
    EncodeElementMembershipWork(r, chain, std::span{&job, 1u});
    EncodeSortElementWork(r, chain, std::span{&result.Work, 1u});
    return result;
}

ClosureSeed ListSeed(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, Element element, std::span<const uint32_t> handles) {
    if (element != Element::Vertex && element != Element::Edge && element != Element::Face) {
        throw std::invalid_argument("A topology handle list requires vertices, edges or faces.");
    }
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const HostIncidence incidence{meshes, a, a.FaceCorners.Count(record.FaceCorners)};
    ClosureSeed seed;
    uint64_t sum = 0u;
    for (const auto handle : handles) {
        if (!meshes.IsLiveElement(id, element, handle)) throw std::invalid_argument("Topology handle list contains a foreign or retired element.");
        sum += incidence.Add(element, handle, seed.Vertices);
    }
    const auto d = element == Element::Vertex ? 0u : element == Element::Face ? 2u : 3u;
    seed.Work = SeedElementWorkHandles(chain.Scratch, DomainCapacity(a, d), handles, DomainBlocks(a, record, d));
    seed.Count = ElementWorkCount(chain.Scratch, seed.Work);
    seed.Incidence = uint32_t(std::min<uint64_t>(sum, incidence.Corners));
    const auto fans = incidence.Fans(seed.Vertices);
    if (element == Element::Face) seed.LoopFans = fans;
    return seed;
}

ClosureSeed FaceSeed(const state::Scene &r, uint32_t id, const BufferArena<uint32_t> &storage, ElementWork faces) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const HostIncidence incidence{meshes, a, a.FaceCorners.Count(record.FaceCorners)};
    ClosureSeed seed{.Work = faces, .Count = ElementWorkCount(storage, faces)};
    // Every face of a mesh holds each of its corners once.
    if (seed.Count == a.FaceTriangles.Count(record.FaceData)) {
        seed.Incidence = seed.LoopFans = incidence.Corners;
        seed.All = true;
        return seed;
    }
    uint64_t corners = 0u;
    ForEachWorkElement(storage, faces, [&](uint32_t face) { corners += incidence.Add(Element::Face, face, seed.Vertices); });
    seed.Incidence = uint32_t(std::min<uint64_t>(corners, incidence.Corners));
    seed.LoopFans = incidence.Fans(seed.Vertices);
    return seed;
}

ClosureSeed AroundVertices(const state::Scene &r, uint32_t id, Element element, ElementWork work, const ClosureSeed &vertices) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const HostIncidence incidence{meshes, a, a.FaceCorners.Count(record.FaceCorners)};
    ClosureSeed seed{.Work = work, .All = vertices.All};
    if (vertices.All) {
        seed.Count = element == Element::Face ? a.FaceTriangles.Count(record.FaceData) : a.EdgeHalfedges.Count(record.EdgeData);
        seed.Incidence = seed.LoopFans = incidence.Corners;
        return seed;
    }
    std::vector<uint32_t> around;
    for (const auto v : vertices.Vertices) incidence.AddFan(element, v, around);
    std::ranges::sort(around);
    around.erase(std::unique(around.begin(), around.end()), around.end());
    uint64_t corners = 0u;
    for (const auto handle : around) corners += incidence.Add(element, handle, seed.Vertices);
    seed.Count = uint32_t(around.size());
    seed.Incidence = uint32_t(std::min<uint64_t>(corners, incidence.Corners));
    const auto fans = incidence.Fans(seed.Vertices);
    if (element == Element::Face) seed.LoopFans = fans;
    return seed;
}

FaceTriangles EncodeFaceTriangles(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, const ClosureSeed &faces) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    FaceTriangles result;
    result.TotalWord = IncidenceWord(chain);
    // A face with n loop corners derives n - 2 triangles.
    const auto bound = faces.Incidence;
    result.Triangles = AllocateElementWork(chain.Scratch, a.Triangles.Capacity(),
        std::min(bound, record.TriangleData ? a.Triangles.Set(record.TriangleData).BlockCount : 0u));
    if (!faces.Count) return result;
    const FaceTrianglePushConstants pc{
        .Faces = faces.Work, .Triangles = result.Triangles,
        .Error = {chain.Scratch.Buffer.Slot, 0u}, .Total = {chain.Scratch.Buffer.Slot, result.TotalWord},
        .FaceBound = faces.Count, .FaceOwner = record.FaceData.Index, .TriangleOwner = record.TriangleData.Index,
        .FaceBlocksSlot = a.FaceTriangles.Blocks.Buffer.Slot, .TriangleBlocksSlot = a.Triangles.Blocks.Buffer.Slot,
        .FaceRangesSlot = a.FaceRanges.Buffer.Slot, .FaceTrianglesSlot = a.FaceTriangles.Buffer.Slot, .TrianglesSlot = a.Triangles.Buffer.Slot,
        .FaceCapacity = std::min({a.FaceTriangles.Capacity(), a.FaceTriangles.Buffer.Count<uint32_t>(), a.FaceRanges.Buffer.Count<uvec2>()}),
        .TriangleCapacity = std::min(a.Triangles.Capacity(), a.Triangles.Buffer.Count<uvec3>()), .CornerCapacity = a.FaceCorners.Capacity(),
    };
    chain.Groups(GetMeshPipelines(r)[MeshPass::MeshClosureTriangles], pc, faces.Count, 32u);
    EncodeSortElementWork(r, chain, std::span{&result.Triangles, 1u});
    return result;
}

void FaceTriangles::Finish(const mtl::ComputeChain &chain) {
    Count = ElementWorkCount(chain.Scratch, Triangles);
    if (Count != chain.Scratch.Get({TotalWord, 1u})[0]) throw std::invalid_argument("Face triangle ranges overlap.");
}
