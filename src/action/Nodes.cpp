#include "action/Nodes.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "procedure/Procedure.h"

namespace action::nodes {
namespace {
using namespace procedure;

constexpr uint32_t OutputGap{12u}; // The cells a new node leaves before the Output node.

state::Entity CreateNode(state::Registry &r, uint16_t kind, uvec2 position) {
    const auto entity = r.create();
    VisitKind(kind, [&]<typename K> { r.emplace<K>(entity); });
    r.emplace<NodeLayout>(entity, position);
    return entity;
}

state::Entity CopyNode(state::Registry &r, state::Entity node, uint16_t kind) {
    const auto copy = r.create();
    VisitKind(kind, [&]<typename K> {
        const auto parameters = r.get<const K>(node);
        r.emplace<K>(copy, parameters);
    });
    const auto layout = r.get<const NodeLayout>(node);
    r.emplace<NodeLayout>(copy, layout);
    if (const auto *selection = r.try_get<const InlineSelection>(node)) {
        auto copied = *selection;
        r.emplace<InlineSelection>(copy, std::move(copied));
    }
    if (r.all_of<PublishSelection>(node)) r.emplace<PublishSelection>(copy);
    return copy;
}

// Whether the list names distinct nodes of the procedure in ascending order.
bool ListsNodes(const GeometryProcedure &procedure, std::span<const uint32_t> nodes) {
    return std::ranges::is_sorted(nodes) && std::ranges::adjacent_find(nodes) == nodes.end() && (nodes.empty() || nodes.back() < procedure.Nodes.size());
}

void DeselectNodes(state::Registry &r, const GeometryProcedure &procedure) {
    for (const auto &node : procedure.Nodes) {
        if (r.all_of<NodeSelected>(node.Entity)) r.remove<NodeSelected>(node.Entity);
    }
}

// The kind of the node's parameter component, or KindCount for an entity without one.
uint16_t KindOfNode(const state::Registry &r, state::Entity node) {
    uint16_t kind = KindCount;
    if (r.valid(node)) {
        ForEachKind([&]<typename K> {
            if (r.all_of<K>(node)) kind = KindOf<K>;
        });
    }
    return kind;
}

template<typename K> bool Rename(state::Registry &r, state::Entity node, std::string_view name) {
    if (!r.valid(node) || !r.all_of<K>(node)) return false;
    r.patch<K>(node, [&](K &p) {
        p.Name = {};
        std::ranges::copy(name, p.Name.begin());
    });
    return true;
}
} // namespace

void Apply(state::Registry &r, state::Entity, const Action &action) {
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
                const auto source = CreateNode(r, KindOf<Source>, {0u, 0u}), output = CreateNode(r, KindOf<Output>, {OutputGap, 0u});
                r.emplace<GeometryProcedure>(a.MeshEntity, std::vector<NodeRecord>{{source, KindOf<Source>}, {output, KindOf<Output>}}, std::vector<SocketLink>{{0u, 0u}}, 1u);
            },
            [&](const AddNode &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (a.Kind >= KindCount) return Fail(r, "Unknown node kind");
                const auto node = CreateNode(r, a.Kind, a.Position);
                DeselectNodes(r, *procedure);
                r.emplace<NodeSelected>(node);
                r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &procedure) {
                    procedure.ActiveNode = uint32_t(procedure.Nodes.size());
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
                // Transfer the evaluated record to the source before removing its MeshHandle.
                r.remove<MeshHandle>(a.MeshEntity);
                r.emplace<MeshHandle>(a.MeshEntity, output);
                // Detach published instance lists before destroying the procedure to retain the objects.
                if (r.all_of<ProcedureInstances>(a.MeshEntity)) r.replace<ProcedureInstances>(a.MeshEntity);
                if (r.all_of<ProcedureOwnedMesh>(a.MeshEntity)) r.replace<ProcedureOwnedMesh>(a.MeshEntity, state::Null);
                r.remove<GeometryProcedure, ProcedureInstances, ProcedureOwnedMesh, EvaluatedMesh, ResyncState, CookPlan, PrimitiveShape>(a.MeshEntity);
            },
            [&](const DuplicateNodes &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (a.Nodes.empty() || !ListsNodes(*procedure, a.Nodes)) return Fail(r, "A duplicate lists distinct nodes of the procedure in ascending order");
                const auto first = uint32_t(procedure->Nodes.size());
                std::vector<uint32_t> copy_of(first, NoNode);
                for (uint32_t i = 0u; i < a.Nodes.size(); ++i) copy_of[a.Nodes[i]] = first + i;
                std::vector<NodeRecord> copies;
                std::vector<SocketLink> inputs;
                for (const auto n : a.Nodes) {
                    const auto [entity, kind] = procedure->Nodes[n];
                    const auto copy = CopyNode(r, entity, kind);
                    copies.push_back({copy, kind});
                    const auto slot = InputFirst(*procedure, n);
                    for (uint32_t i = 0u; i < InputCount[kind]; ++i) {
                        const auto link = procedure->Inputs[slot + i];
                        inputs.push_back(link.FromNode != NoNode && copy_of[link.FromNode] != NoNode ? SocketLink{copy_of[link.FromNode], link.FromSocket} : SocketLink{});
                    }
                }
                const auto active = procedure->ActiveNode != NoNode && copy_of[procedure->ActiveNode] != NoNode ? copy_of[procedure->ActiveNode] : first + uint32_t(a.Nodes.size()) - 1u;
                DeselectNodes(r, *procedure);
                for (const auto &copy : copies) r.emplace<NodeSelected>(copy.Entity);
                r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &edited) {
                    edited.Nodes.append_range(copies);
                    edited.Inputs.append_range(inputs);
                    edited.ActiveNode = active;
                });
            },
            [&](const DeleteNodes &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (a.Nodes.empty() || !ListsNodes(*procedure, a.Nodes)) return Fail(r, "A delete lists distinct nodes of the procedure in ascending order");
                const auto input_first = InputFirsts(*procedure);
                const auto removed = [&](uint32_t n) { return std::ranges::binary_search(a.Nodes, n); };
                // Resolve links through removed nodes to surviving outputs.
                const auto through = [&](this const auto &self, SocketLink link) -> SocketLink {
                    if (link.FromNode == NoNode || !removed(link.FromNode)) return link;
                    const auto kind = procedure->Nodes[link.FromNode].Kind;
                    const auto sockets = KindSockets(kind);
                    SocketLink source{};
                    uint32_t sources = 0u;
                    for (uint16_t i = 0u; i < InputCount[kind]; ++i) {
                        const auto read = sockets[i].Type == sockets[link.FromSocket].Type ? self(procedure->Inputs[input_first[link.FromNode] + i]) : SocketLink{};
                        if (read.FromNode == NoNode) continue;
                        source = read;
                        ++sources;
                    }
                    return sources == 1u ? source : SocketLink{};
                };
                std::vector<std::pair<uint32_t, SocketLink>> relinks;
                for (uint32_t n = 0u; n < procedure->Nodes.size(); ++n) {
                    const auto kind = procedure->Nodes[n].Kind;
                    for (uint32_t i = 0u, slot = input_first[n]; i < InputCount[kind]; ++i, ++slot) {
                        const auto link = procedure->Inputs[slot];
                        if (removed(n) || link.FromNode == NoNode || !removed(link.FromNode)) continue;
                        if (const auto source = through(link); source.FromNode != NoNode && Links(KindSockets(procedure->Nodes[source.FromNode].Kind)[source.FromSocket], KindSockets(kind)[i], kind)) relinks.emplace_back(slot, source);
                    }
                }
                std::vector<state::Entity> entities;
                entities.reserve(a.Nodes.size());
                for (const auto n : a.Nodes) entities.push_back(procedure->Nodes[n].Entity);
                if (!relinks.empty()) {
                    r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &edited) {
                        for (const auto &[slot, source] : relinks) edited.Inputs[slot] = source;
                    });
                }
                // Destroying a node removes it from the procedure and disconnects its consumers.
                for (const auto entity : entities) r.destroy(entity);
            },
            [&](const InsertNode &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                const auto count = uint32_t(procedure->Nodes.size());
                if (a.Node >= count || a.ToNode >= count || a.ToSocket >= InputCount[procedure->Nodes[a.ToNode].Kind]) return Fail(r, "An insert names a missing node or input");
                const auto input_first = InputFirsts(*procedure);
                if (!Inserts(*procedure, input_first, a.Node, a.ToNode, a.ToSocket)) return Fail(r, "A node inserts on a link through geometry sockets that link with its ends, without closing a cycle");
                // The input's node and the nodes reading it, transitively.
                std::vector<bool> moved(count);
                const auto reads_moved = [&](uint32_t n) {
                    return std::ranges::any_of(procedure->Inputs.begin() + input_first[n], procedure->Inputs.begin() + input_first[n + 1u], [&](SocketLink link) { return link.FromNode != NoNode && moved[link.FromNode]; });
                };
                moved[a.ToNode] = a.Shift != 0u;
                for (bool grew = moved[a.ToNode]; grew;) {
                    grew = false;
                    for (uint32_t n = 0u; n < count; ++n) {
                        if (!moved[n] && reads_moved(n)) moved[n] = grew = true;
                    }
                }
                for (uint32_t n = 0u; n < count; ++n) {
                    if (moved[n]) r.patch<NodeLayout>(procedure->Nodes[n].Entity, [&](NodeLayout &layout) { layout.Position.x += a.Shift; });
                }
                const auto kind = procedure->Nodes[a.Node].Kind;
                const auto input = input_first[a.Node] + *MainSocket(kind, SocketDirection::In), to_input = input_first[a.ToNode] + a.ToSocket;
                r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &edited) {
                    if (edited.Inputs[input].FromNode == NoNode) edited.Inputs[input] = edited.Inputs[to_input];
                    edited.Inputs[to_input] = {a.Node, *MainSocket(kind, SocketDirection::Out)};
                });
            },
            [&](const SelectNodes &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (!ListsNodes(*procedure, a.Nodes)) return Fail(r, "A node selection lists distinct nodes of the procedure in ascending order");
                if (a.Active != NoNode && !std::ranges::binary_search(a.Nodes, a.Active)) return Fail(r, "The active node must be selected");
                for (uint32_t n = 0u; n < procedure->Nodes.size(); ++n) {
                    const auto entity = procedure->Nodes[n].Entity;
                    const bool selected = std::ranges::binary_search(a.Nodes, n);
                    if (selected && !r.all_of<NodeSelected>(entity)) r.emplace<NodeSelected>(entity);
                    else if (!selected && r.all_of<NodeSelected>(entity)) r.remove<NodeSelected>(entity);
                }
                if (procedure->ActiveNode != a.Active) r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &edited) { edited.ActiveNode = a.Active; });
            },
        },
        action
    );
}

SetInlineSelection CaptureSelection(const state::Registry &r, state::Entity mesh_entity, state::Entity node) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto id = *DrawnStoreId(r, mesh_entity);
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

void CopyProcedure(state::Registry &r, state::Entity from, state::Entity to) {
    auto procedure = r.get<const GeometryProcedure>(from);
    for (auto &[entity, kind] : procedure.Nodes) {
        const auto copy = CopyNode(r, entity, kind);
        if (r.all_of<NodeSelected>(entity)) r.emplace<NodeSelected>(copy);
        entity = copy;
    }
    r.emplace<GeometryProcedure>(to, std::move(procedure));
}

state::Entity EditedMesh(const state::Registry &r) {
    const auto active = FindActiveEntity(r);
    if (active == state::Null) return state::Null;
    return HasMesh(r, active) ? active : GetMeshEntity(r, active);
}

state::Entity AppendNode(state::Registry &r, state::Entity viewport, state::Entity mesh_entity, uint16_t kind) {
    const auto &procedure = r.get<const GeometryProcedure>(mesh_entity);
    const auto output = procedure.OutputNode;
    if (output == NoNode) {
        Fail(r, "The geometry procedure has no Output node");
        return state::Null;
    }
    const auto read = procedure.Inputs[InputFirst(procedure, output)];
    const auto node = uint32_t(procedure.Nodes.size());
    const auto position = r.get<const NodeLayout>(procedure.Nodes[output].Entity).Position;
    r.patch<NodeLayout>(procedure.Nodes[output].Entity, [](NodeLayout &layout) { layout.Position.x += OutputGap; });
    Apply(r, viewport, AddNode{mesh_entity, kind, position});
    const auto entity = r.get<const GeometryProcedure>(mesh_entity).Nodes.back().Entity;
    Apply(r, viewport, CaptureSelection(r, mesh_entity, entity));
    if (read.FromNode != NoNode) Apply(r, viewport, Link{mesh_entity, read.FromNode, read.FromSocket, node, 0u});
    Apply(r, viewport, Link{mesh_entity, node, InputCount[kind], output, 0u});
    r.emplace<PublishSelection>(entity);
    return entity;
}

void ForEachResultNode(const state::Registry &r, state::Entity viewport, const std::function<void(state::Entity, state::Entity)> &fn) {
    for (const auto mesh_entity : mesh::SelectedEditMeshes(r, viewport)) {
        const auto *procedure = r.try_get<const GeometryProcedure>(mesh_entity);
        if (!procedure || procedure->OutputNode == NoNode) continue;
        if (const auto read = procedure->Inputs[InputFirst(*procedure, procedure->OutputNode)]; read.FromNode != NoNode) fn(mesh_entity, procedure->Nodes[read.FromNode].Entity);
    }
}
} // namespace action::nodes

namespace action {
state::Entity UpdateTraits<procedure::NodeLayout>::Active(const state::Registry &r) {
    const auto *procedure = r.try_get<const procedure::GeometryProcedure>(nodes::EditedMesh(r));
    return procedure && procedure->ActiveNode != procedure::NoNode ? procedure->Nodes[procedure->ActiveNode].Entity : state::Null;
}
void UpdateTraits<procedure::NodeLayout>::ForEachSelected(state::Registry &r, state::Entity, const std::function<void(state::Entity)> &fn) {
    const auto *procedure = r.try_get<const procedure::GeometryProcedure>(nodes::EditedMesh(r));
    if (!procedure) return;
    for (const auto &node : procedure->Nodes) {
        if (r.all_of<procedure::NodeSelected>(node.Entity)) fn(node.Entity);
    }
}
} // namespace action
