#include "mesh/AttributeFields.h"

#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"
#include "state/Registry.h"

namespace {
struct Field {
    mtl::ComputeChain &Chain;
    NeighborFieldJob Pc{};
    uint32_t Threads{}, HandleBound{};

    Field(const MeshStore &meshes, mtl::ComputeChain &chain, uint32_t id, AttributeDomain domain) : Chain{chain} {
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
// Records `pass` over `threads` threads of the job, with the jobs the other pending lanes gather for it at the same step.
void RecordNeighborField(const MeshPipelines &pipelines, mtl::ComputeChain &chain, MeshPass pass, const NeighborFieldJob &job, uint32_t threads) {
    if (threads) mtl::GatherPass(chain, pipelines[pass], job, threads);
}
// The field writing `output` over the live elements of its domain.
Field OutputField(const MeshStore &meshes, mtl::ComputeChain &chain, uint32_t output) {
    const auto &attribute = meshes.GetAttribute(output);
    Field field{meshes, chain, attribute.Record, attribute.Domain};
    field.Pc.Output = meshes.AttributeRef(output);
    return field;
}

AttributeFieldJob OutputJob(const MeshStore &meshes, uint32_t output) {
    const auto &attribute = meshes.GetAttribute(output);
    return {
        .Blocks = BlocksOf(meshes, attribute.Record, ElementDomainOf(attribute.Domain)),
        .Connectivity = meshes.GetConnectivityRef(attribute.Record),
        .VertexSlot = meshes.Arenas().Vertices.Buffer.Slot,
        .Domain = ElementOf(attribute.Domain),
        .Output = meshes.AttributeRef(output),
    };
}
// Records `pass` writing the job's output, a tile per block of its domain, with the jobs the other pending lanes gather for it at the same step.
void RecordFieldJob(const MeshPipelines &pipelines, mtl::ComputeChain &chain, MeshPass pass, const auto &job) {
    if (job.Blocks.Count) mtl::GatherPass(chain, pipelines[pass], job, job.Blocks.Count * MeshElementBlockSize);
}
} // namespace

void RecordDomainConversion(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t source, uint32_t output, DomainReduction reduction) {
    const auto &from = meshes.GetAttribute(source), &to = meshes.GetAttribute(output);
    const DomainConversionJob job{
        .Blocks = BlocksOf(meshes, to.Record, ElementDomainOf(to.Domain)),
        .Connectivity = meshes.GetConnectivityRef(to.Record),
        .Source = meshes.AttributeRef(source),
        .Output = meshes.AttributeRef(output),
        .Type = from.Type,
        .From = from.Domain,
        .To = to.Domain,
        .Reduction = reduction,
    };
    RecordFieldJob(pipelines, chain, MeshPass::FieldConvertDomain, job);
}

void RecordCurvature(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t output, CurvatureKind kind) {
    auto field = OutputField(meshes, chain, output);
    field.Pc.Mean = kind == CurvatureKind::Mean ? 1u : 0u;
    RecordNeighborField(pipelines, chain, MeshPass::FieldCurvature, field.Pc, field.Threads);
}

void RecordBoundary(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t output) {
    RecordFieldJob(pipelines, chain, MeshPass::FieldBoundary, OutputJob(meshes, output));
}

void RecordAdjacent(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t seed, uint32_t output, bool more, uint32_t steps) {
    auto field = OutputField(meshes, chain, output);
    if (!field.Threads) return;
    auto &pc = field.Pc;
    pc.Adjacency = FieldAdjacency::Domain;
    pc.Complement = more ? 0u : 1u;
    pc.Seed = meshes.AttributeRef(seed);
    pc.Labels = field.Words(field.HandleBound);
    pc.Flags = field.Words(field.HandleBound);
    RecordNeighborField(pipelines, chain, MeshPass::FieldGrowInit, pc, field.Threads);
    for (uint32_t step = 0u; step < steps; ++step) {
        RecordNeighborField(pipelines, chain, MeshPass::FieldGrow, pc, field.Threads);
        std::swap(pc.Labels, pc.Flags);
    }
    RecordNeighborField(pipelines, chain, MeshPass::FieldGrowOutput, pc, field.Threads);
}

void RecordComponents(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t seed, uint32_t output, FieldAdjacency adjacency, std::function<void()> then) {
    auto field = OutputField(meshes, chain, output);
    if (!field.Threads) return then();
    auto &pc = field.Pc;
    pc.Seed = meshes.AttributeRef(seed);
    pc.Adjacency = adjacency;
    pc.Labels = field.Words(field.HandleBound);
    pc.Flags = field.Words(field.HandleBound);
    pc.Changed = field.Words(1u);
    const auto id = meshes.GetAttribute(output).Record;
    chain.Lane(id, [&] { RecordNeighborField(pipelines, chain, MeshPass::FieldLabelInit, pc, field.Threads); });
    RecordUntilUnchanged(chain, id, pc.Changed.Offset, [&pipelines, &chain, pc, threads = field.Threads] { RecordNeighborField(pipelines, chain, MeshPass::FieldLink, pc, threads); }, [&pipelines, &chain, id, pc, threads = field.Threads, then = std::move(then)] {
        chain.Lane(id, [&] {
            RecordNeighborField(pipelines, chain, MeshPass::FieldMarkSeeds, pc, threads);
            RecordNeighborField(pipelines, chain, MeshPass::FieldComponents, pc, threads);
        });
        then(); });
}

void RecordPlaneSide(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t output, vec3 point, vec3 normal, PlaneSideKind side, float tolerance) {
    auto job = OutputJob(meshes, output);
    job.PlaneNormal = Normalize(normal);
    job.PlaneOffset = Dot(job.PlaneNormal, point);
    job.Tolerance = std::max(tolerance, 0.f);
    job.Side = side;
    RecordFieldJob(pipelines, chain, MeshPass::FieldPlaneSide, job);
}

std::optional<uint32_t> PlaneSideField(state::Registry &r, uint32_t id, AttributeDomain domain, vec3 point, vec3 normal, PlaneSideKind side, float tolerance, std::string_view name) {
    if (!(Dot(normal, normal) > 0.f)) return {};
    auto &meshes = r.Resources.get<MeshStore>();
    const auto output = meshes.AddAttribute(id, domain, AttributeType::Bool, name);
    if (!output) return {};
    mtl::ComputeChain chain{meshes.BufferContext()};
    RecordPlaneSide(meshes, GetMeshPipelines(r), chain, *output, point, normal, side, tolerance);
    chain.Submit();
    return *output;
}

void RecordComponentLabels(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, Element domain, std::span<const uint32_t> handles, FieldAdjacency adjacency, bool parity, std::function<void(std::vector<uint32_t>)> then) {
    if (handles.empty()) return then({});
    Field field{meshes, chain, id, AttributeDomainOf(domain)};
    auto &pc = field.Pc;
    pc.Adjacency = adjacency;
    pc.Parity = parity ? 1u : 0u;
    pc.Labels = field.Words(field.HandleBound);
    pc.Flags = field.Words(field.HandleBound);
    pc.Changed = field.Words(1u);
    pc.Restrict = field.Words(field.HandleBound);
    pc.List = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(handles).Offset};
    pc.ListCount = uint32_t(handles.size());
    chain.Lane(id, [&] {
        RecordNeighborField(pipelines, chain, MeshPass::FieldLabelInit, pc, field.Threads);
        RecordNeighborField(pipelines, chain, MeshPass::FieldMarkList, pc, pc.ListCount);
    });
    // With parity, distance rounds follow the settled labels, and each label takes the parity of its distance.
    const auto read = [&chain, pc, bound = field.HandleBound, handles = std::vector<uint32_t>{handles.begin(), handles.end()}, then = std::move(then)] {
        const auto labels = chain.Scratch.Get({pc.Labels.Offset, bound}), distances = chain.Scratch.Get({pc.Flags.Offset, bound});
        then(handles | std::views::transform([&](uint32_t handle) { return pc.Parity ? labels[handle] << 1u | (distances[handle] & 1u) : labels[handle]; }) | std::ranges::to<std::vector>());
    };
    RecordUntilUnchanged(chain, id, pc.Changed.Offset, [&pipelines, &chain, pc, threads = field.Threads] { RecordNeighborField(pipelines, chain, MeshPass::FieldLink, pc, threads); }, [&pipelines, &chain, id, pc, threads = field.Threads, read] {
        if (!pc.Parity) return read();
        RecordUntilUnchanged(chain, id, pc.Changed.Offset, [&pipelines, &chain, pc, threads] { RecordNeighborField(pipelines, chain, MeshPass::FieldDistance, pc, threads); }, read); });
}
std::vector<uint32_t> ComponentLabels(const MeshStore &meshes, const MeshPipelines &pipelines, uint32_t id, Element domain, std::span<const uint32_t> handles, FieldAdjacency adjacency, bool parity) {
    return mtl::Drained<std::vector<uint32_t>>(meshes.BufferContext(), [&](mtl::ComputeChain &chain, auto then) { RecordComponentLabels(meshes, pipelines, chain, id, domain, handles, adjacency, parity, std::move(then)); });
}
