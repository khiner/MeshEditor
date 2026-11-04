#include "mesh/AttributeFields.h"

#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"
#include "metal/Dispatch.h"
#include "state/Registry.h"

namespace {
struct Field {
    mtl::ComputeChain Chain;
    AttributeFieldPushConstants Pc{};
    uint32_t Threads{}, HandleBound{};

    Field(const MeshStore &meshes, uint32_t id, AttributeDomain domain) : Chain{meshes.BufferContext()} {
        const auto element_domain = ElementDomainOf(domain);
        const auto &a = meshes.Arenas();
        Pc.Blocks = BlocksOf(meshes, id, element_domain);
        Pc.Connectivity = meshes.GetConnectivityRef(id);
        Pc.VertexSlot = a.Vertices.Buffer.Slot;
        Pc.FaceTriangleStartSlot = a.FaceTriangles.Buffer.Slot;
        Pc.TriangleSlot = a.Triangles.Buffer.Slot;
        Pc.Domain = ElementOf(domain);
        Threads = Pc.Blocks.Count * MeshElementBlockSize;
        HandleBound = WithDomain(a, element_domain, [&](const auto &arena) { return arena.HandleBound(DomainSet(meshes.Get(id), element_domain)); });
    }
    SlotOffset Words(uint32_t count) { return {Chain.Scratch.Buffer.Slot, Chain.Scratch.Allocate(count).Offset}; }
};

void Label(const MeshPipelines &pipelines, Field &field, FieldAdjacency adjacency, std::span<const uint32_t> listed = {}, bool parity = false) {
    auto &pc = field.Pc;
    pc.Adjacency = adjacency;
    pc.Parity = parity ? 1u : 0u;
    pc.Labels = field.Words(field.HandleBound);
    pc.Flags = field.Words(field.HandleBound);
    pc.Changed = field.Words(1u);
    auto &chain = field.Chain;
    if (!listed.empty()) {
        pc.Restrict = field.Words(field.HandleBound);
        pc.List = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(listed).Offset};
        pc.ListCount = uint32_t(listed.size());
    }
    chain.Threads(pipelines[MeshPass::FieldLabelInit], pc, field.Threads);
    if (!listed.empty()) chain.Threads(pipelines[MeshPass::FieldMarkList], pc, pc.ListCount);
    RunUntilUnchanged(chain, pipelines[MeshPass::FieldLink], pc, field.Threads, pc.Changed.Offset);
}
std::optional<uint32_t> WriteField(state::Registry &r, uint32_t id, AttributeDomain domain, AttributeType type, std::string_view name, auto &&record) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto output = meshes.AddAttribute(id, domain, type, name);
    if (!output) return {};
    Field field{meshes, id, domain};
    field.Pc.Output = meshes.AttributeRef(*output);
    if (!field.Threads) return *output;
    record(field, GetMeshPipelines(r));
    field.Chain.Submit();
    return *output;
}
std::optional<uint32_t> Components(state::Registry &r, uint32_t id, uint32_t seed, FieldAdjacency adjacency, std::string_view name) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    return WriteField(r, id, meshes.GetAttribute(seed).Domain, AttributeType::Bool, name, [&](Field &field, const MeshPipelines &pipelines) {
        field.Pc.Seed = meshes.AttributeRef(seed);
        Label(pipelines, field, adjacency);
        field.Chain.Threads(pipelines[MeshPass::FieldMarkSeeds], field.Pc, field.Threads);
        field.Chain.Threads(pipelines[MeshPass::FieldComponents], field.Pc, field.Threads);
    });
}
} // namespace

std::optional<uint32_t> CurvatureField(state::Registry &r, uint32_t id, CurvatureKind kind, std::string_view name) {
    return WriteField(r, id, AttributeDomain::Vertex, AttributeType::Float, name, [&](Field &field, const MeshPipelines &pipelines) {
        field.Pc.Mean = kind == CurvatureKind::Mean ? 1u : 0u;
        field.Chain.Threads(pipelines[MeshPass::FieldCurvature], field.Pc, field.Threads);
    });
}

std::optional<uint32_t> BoundaryField(state::Registry &r, uint32_t id, AttributeDomain domain, std::string_view name) {
    return WriteField(r, id, domain, AttributeType::Bool, name, [](Field &field, const MeshPipelines &pipelines) {
        field.Chain.Threads(pipelines[MeshPass::FieldBoundary], field.Pc, field.Threads);
    });
}

std::optional<uint32_t> PlaneSideField(state::Registry &r, uint32_t id, AttributeDomain domain, vec3 point, vec3 normal, PlaneSideKind side, float tolerance, std::string_view name) {
    if (!(Dot(normal, normal) > 0.f)) return {};
    return WriteField(r, id, domain, AttributeType::Bool, name, [&](Field &field, const MeshPipelines &pipelines) {
        const auto n = Normalize(normal);
        field.Pc.PlaneNormal = n;
        field.Pc.PlaneOffset = Dot(n, point);
        field.Pc.Tolerance = std::max(tolerance, 0.f);
        field.Pc.Side = side;
        field.Chain.Threads(pipelines[MeshPass::FieldPlaneSide], field.Pc, field.Threads);
    });
}

std::optional<uint32_t> LinkedField(state::Registry &r, uint32_t id, uint32_t seed, std::string_view name) { return Components(r, id, seed, FieldAdjacency::Domain, name); }

std::vector<uint32_t> ComponentLabels(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, Element domain, std::span<const uint32_t> handles, FieldAdjacency adjacency, bool parity) {
    if (handles.empty()) return {};
    Field field{meshes, id, AttributeDomainOf(domain)};
    Label(pipelines, field, adjacency, handles, parity);
    const auto labels = field.Chain.Scratch.Get({field.Pc.Labels.Offset, field.HandleBound});
    std::vector<uint32_t> result(handles.size());
    for (uint32_t i = 0u; i < handles.size(); ++i) result[i] = labels[handles[i]];
    return result;
}
std::optional<uint32_t> EdgeLoopField(state::Registry &r, uint32_t id, uint32_t seed, std::string_view name) { return Components(r, id, seed, FieldAdjacency::EdgeLoop, name); }
std::optional<uint32_t> EdgeRingField(state::Registry &r, uint32_t id, uint32_t seed, std::string_view name) { return Components(r, id, seed, FieldAdjacency::EdgeRing, name); }

std::optional<uint32_t> AdjacentField(state::Registry &r, uint32_t id, uint32_t seed, bool more, uint32_t steps, std::string_view name) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    return WriteField(r, id, meshes.GetAttribute(seed).Domain, AttributeType::Bool, name, [&](Field &field, const MeshPipelines &pipelines) {
        auto &pc = field.Pc;
        pc.Adjacency = FieldAdjacency::Domain;
        pc.Complement = more ? 0u : 1u;
        pc.Seed = meshes.AttributeRef(seed);
        pc.Labels = field.Words(field.HandleBound);
        pc.Flags = field.Words(field.HandleBound);
        auto &chain = field.Chain;
        chain.Threads(pipelines[MeshPass::FieldGrowInit], pc, field.Threads);
        for (uint32_t step = 0u; step < steps; ++step) {
            chain.Threads(pipelines[MeshPass::FieldGrow], pc, field.Threads);
            std::swap(pc.Labels, pc.Flags);
        }
        chain.Threads(pipelines[MeshPass::FieldGrowOutput], pc, field.Threads);
    });
}
