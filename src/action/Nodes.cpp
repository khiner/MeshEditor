#include "action/Nodes.h"
#include "Variant.h"
#include "action/Errors.h"
#include "mesh/Mesh.h"
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

constexpr bool Links(Socket from, Socket to) {
    const bool typed = from.Type == to.Type || (from.Type == SocketType::Attribute && to.Type == SocketType::Selection);
    return typed && (from.Type == SocketType::Geometry || (from.Types & to.Types));
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
                if (!Links(from[a.FromSocket], to[a.ToSocket])) return Fail(r, "Linked sockets differ in type");
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
        },
        action
    );
}
} // namespace action::nodes
