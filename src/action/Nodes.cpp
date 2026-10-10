#include "action/Nodes.h"
#include "Variant.h"
#include "action/Errors.h"
#include "action/Mesh.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "object/PendingSync.h"
#include "procedure/Procedure.h"
#include "state/Scene.h"

namespace action::nodes {
namespace {
using namespace procedure;

state::Entity CreateNode(state::Scene &r, uint16_t kind, vec2 position) {
    const auto entity = r.create();
    VisitKind(kind, [&]<typename K> { r.emplace<K>(entity); });
    r.emplace<NodeLayout>(entity, position, vec2{});
    return entity;
}

// An Output node's geometry input also takes instances, which its mesh entity publishes.
constexpr bool Links(Socket from, Socket to, uint16_t to_kind) {
    if (from.Type == SocketType::Instances) return to.Type == SocketType::Instances || to_kind == KindOf<Output>;
    const bool typed = from.Type == to.Type || (from.Type == SocketType::Attribute && to.Type == SocketType::Selection);
    return typed && (from.Type == SocketType::Geometry || (from.Types & to.Types));
}

// The kind of the node's parameter component, or KindCount for an entity without one.
uint16_t KindOfNode(const state::Scene &r, state::Entity node) {
    uint16_t kind = KindCount;
    if (r.valid(node)) ForEachKind([&]<typename K> {
        if (r.all_of<K>(node)) kind = KindOf<K>;
    });
    return kind;
}

// The record's selection masks at each block it owns, vertices then edges then faces.
SetInlineSelection SelectionOf(const MeshStore &meshes, uint32_t id, state::Entity node) {
    SetInlineSelection selection{.Node = node};
    using D = MeshStore::ElementDomain;
    for (const auto [domain, element] : {std::pair{D::Vertex, Element::Vertex}, std::pair{D::Edge, Element::Edge}, std::pair{D::Face, Element::Face}}) {
        const auto bits = meshes.GetSelectedElements(id, element).Bits;
        for (const auto block : meshes.GetBlockList(id, domain).Blocks) {
            auto &words = selection.Blocks.emplace_back();
            std::ranges::copy(bits.subspan(block * MeshElementBlockWords, MeshElementBlockWords), words.begin());
        }
        if (domain == D::Vertex) selection.Vertices = uint32_t(selection.Blocks.size());
        if (domain == D::Edge) selection.Edges = uint32_t(selection.Blocks.size()) - selection.Vertices;
    }
    return selection;
}

// Writes the name into the node's NamedAttribute or StoreNamedAttribute parameters.
template<typename K> bool Rename(state::Scene &r, state::Entity node, std::string_view name) {
    if (!r.valid(node) || !r.all_of<K>(node)) return false;
    r.patch<K>(node, [&](K &p) {
        p.Name = {};
        std::ranges::copy(name, p.Name.begin());
    });
    return true;
}
} // namespace

void Apply(state::Scene &r, state::Entity, const Action &action) {
    // The mesh entity's procedure, or null after failing the action.
    const auto procedure_of = [&](state::Entity mesh_entity) {
        const auto *procedure = r.try_get<const GeometryProcedure>(mesh_entity);
        if (!procedure) Fail(r, "The mesh has no geometry procedure");
        return procedure;
    };
    const auto set_input = [&](state::Entity mesh_entity, uint32_t slot, SocketLink link) {
        r.patch<GeometryProcedure>(mesh_entity, [&](GeometryProcedure &procedure) { procedure.Inputs[slot] = link; });
    };
    std::visit(
        overloaded{
            [&](const AddProcedure &a) {
                if (!HasMesh(r, a.MeshEntity) || r.all_of<GeometryProcedure>(a.MeshEntity)) return Fail(r, "A geometry procedure needs a mesh without one");
                const auto source = CreateNode(r, KindOf<Source>, {0.f, 0.f}), output = CreateNode(r, KindOf<Output>, {240.f, 0.f});
                r.emplace<GeometryProcedure>(a.MeshEntity, std::vector<NodeRecord>{{source, KindOf<Source>}, {output, KindOf<Output>}}, std::vector<SocketLink>{{0u, 0u}}, 1u);
            },
            [&](const AddNode &a) {
                if (!procedure_of(a.MeshEntity)) return;
                if (a.Kind >= KindCount) return Fail(r, "Unknown node kind");
                const auto node = CreateNode(r, a.Kind, a.Position);
                r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &procedure) {
                    procedure.Nodes.push_back({node, a.Kind});
                    procedure.Inputs.resize(procedure.Inputs.size() + InputCount[a.Kind]);
                });
            },
            [&](const Link &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (a.FromNode >= procedure->Nodes.size() || a.ToNode >= procedure->Nodes.size()) return Fail(r, "A link names a missing node");
                const auto from = KindSockets(procedure->Nodes[a.FromNode].Kind), to = KindSockets(procedure->Nodes[a.ToNode].Kind);
                if (a.FromSocket >= from.size() || from[a.FromSocket].Direction != SocketDirection::Out ||
                    a.ToSocket >= to.size() || to[a.ToSocket].Direction != SocketDirection::In) return Fail(r, "A link joins an output socket to an input socket");
                if (!Links(from[a.FromSocket], to[a.ToSocket], procedure->Nodes[a.ToNode].Kind)) return Fail(r, "Linked sockets differ in type");
                if (Upstream(*procedure, InputFirsts(*procedure), a.FromNode, [](uint32_t) { return true; })[a.ToNode]) return Fail(r, "A link cannot close a cycle");
                set_input(a.MeshEntity, InputFirst(*procedure, a.ToNode) + a.ToSocket, {a.FromNode, a.FromSocket});
            },
            [&](const Unlink &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (a.ToNode >= procedure->Nodes.size() || a.ToSocket >= InputCount[procedure->Nodes[a.ToNode].Kind]) return Fail(r, "An unlink names a missing input");
                set_input(a.MeshEntity, InputFirst(*procedure, a.ToNode) + a.ToSocket, {});
            },
            [&](const SetOutput &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (a.Node >= procedure->Nodes.size() || procedure->Nodes[a.Node].Kind != KindOf<Output>) return Fail(r, "The result must be an Output node");
                r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &edited) { edited.OutputNode = a.Node; });
            },
            [&](const SetAttributeName &a) {
                if (a.Name.empty() || a.Name.size() >= AttributeName{}.size() || a.Name.front() == '.') return Fail(r, "An attribute name needs 1 to 63 characters and no leading '.'");
                if (!Rename<procedure::NamedAttribute>(r, a.Node, a.Name) && !Rename<StoreNamedAttribute>(r, a.Node, a.Name)) Fail(r, "Only attribute name nodes take a name");
            },
            [&](const SetInlineSelection &a) {
                const auto kind = KindOfNode(r, a.Node);
                if (kind == KindCount || !std::ranges::contains(KindSockets(kind), SocketType::Selection, &Socket::Type)) return Fail(r, "Only a node with a selection input takes an inline selection");
                if (uint64_t(a.Vertices) + a.Edges > a.Blocks.size()) return Fail(r, "An inline selection's domain block counts exceed its blocks");
                const auto vertices = a.Blocks.begin() + a.Vertices, edges = vertices + a.Edges;
                r.emplace_or_replace<InlineSelection>(a.Node, std::vector(a.Blocks.begin(), vertices), std::vector(vertices, edges), std::vector(edges, a.Blocks.end()));
            },
            [&](const ApplyProcedure &a) {
                const auto *evaluated = procedure_of(a.MeshEntity) ? r.try_get<const EvaluatedMesh>(a.MeshEntity) : nullptr;
                if (!evaluated) return;
                const auto output = evaluated->StoreId;
                // The output record becomes the source, so the evaluated mesh leaving after it keeps the record.
                r.remove<MeshHandle>(a.MeshEntity);
                r.emplace<MeshHandle>(a.MeshEntity, output);
                // The published instances stay as plain objects with the mesh entity they instance, so their lists empty before they leave.
                if (r.all_of<ProcedureInstances>(a.MeshEntity)) r.replace<ProcedureInstances>(a.MeshEntity);
                if (r.all_of<ProcedureOwnedMesh>(a.MeshEntity)) r.replace<ProcedureOwnedMesh>(a.MeshEntity, state::Null);
                r.remove<GeometryProcedure, ProcedureInstances, ProcedureOwnedMesh, EvaluatedMesh, ResyncState, CookPlan, PrimitiveShape>(a.MeshEntity);
            },
        },
        action
    );
}

void CopyProcedure(state::Scene &r, state::Entity from, state::Entity to) {
    auto procedure = r.get<const GeometryProcedure>(from);
    for (auto &[entity, kind] : procedure.Nodes) {
        const auto copy = r.create();
        VisitKind(kind, [&]<typename K> {
            const auto parameters = r.get<const K>(entity);
            r.emplace<K>(copy, parameters);
        });
        const auto layout = r.get<const NodeLayout>(entity);
        r.emplace<NodeLayout>(copy, layout);
        if (const auto *selection = r.try_get<const InlineSelection>(entity)) {
            auto copied = *selection;
            r.emplace<InlineSelection>(copy, std::move(copied));
        }
        if (r.all_of<PublishSelection>(entity)) r.emplace<PublishSelection>(copy);
        entity = copy;
    }
    r.emplace<GeometryProcedure>(to, std::move(procedure));
}

state::Entity AppendNode(state::Scene &r, state::Entity viewport, state::Entity mesh_entity, uint16_t kind) {
    const auto &procedure = r.get<const GeometryProcedure>(mesh_entity);
    const auto output = procedure.OutputNode;
    if (output == NoNode) {
        Fail(r, "The geometry procedure has no Output node");
        return state::Null;
    }
    const auto read = procedure.Inputs[InputFirst(procedure, output)];
    const auto node = uint32_t(procedure.Nodes.size());
    // The node takes the Output node's place, and the Output node moves past it.
    const auto position = r.get<const NodeLayout>(procedure.Nodes[output].Entity).Position;
    r.patch<NodeLayout>(procedure.Nodes[output].Entity, [](NodeLayout &layout) { layout.Position.x += 240.f; });
    Apply(r, viewport, AddNode{mesh_entity, kind, position});
    const auto entity = r.get<const GeometryProcedure>(mesh_entity).Nodes.back().Entity;
    Apply(r, viewport, SelectionOf(r.Context.get<const MeshStore>(), *DrawnStoreId(r, mesh_entity), entity));
    if (read.FromNode != NoNode) Apply(r, viewport, Link{mesh_entity, read.FromNode, read.FromSocket, node, 0u});
    Apply(r, viewport, Link{mesh_entity, node, InputCount[kind], output, 0u});
    r.emplace<PublishSelection>(entity);
    return entity;
}

void ForEachResultNode(const state::Scene &r, state::Entity viewport, const std::function<void(state::Entity, state::Entity)> &fn) {
    for (const auto mesh_entity : mesh::SelectedEditMeshes(r, viewport)) {
        const auto *procedure = r.try_get<const GeometryProcedure>(mesh_entity);
        if (!procedure || procedure->OutputNode == NoNode) continue;
        if (const auto read = procedure->Inputs[InputFirst(*procedure, procedure->OutputNode)]; read.FromNode != NoNode) fn(mesh_entity, procedure->Nodes[read.FromNode].Entity);
    }
}
} // namespace action::nodes
