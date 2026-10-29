#include "SortUnique.h"
#include "action/Bone.h"
#include "action/Build.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "action/Object.h"
#include "armature/ArmatureComponents.h"
#include "armature/BoneConstraint.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "physics/PhysicsTypes.h"
#include "procedure/Procedure.h"
#include "render/GpuBufferOps.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/HierarchyInstance.h"
#include "scene/SceneGraph.h"

namespace procedure {
namespace {
void ForEachReference(const state::Registry &r, NetworkItem owner, auto &&visit) {
    const auto add = [&](ReferenceKind kind, uint32_t index = 0u) { visit(SceneReference{owner, kind, index}); };
    const auto e = owner.Entity;
    if (owner.Kind == ItemKind::Material) {
        const auto materials = GetMaterials(r);
        if (owner.Index >= materials.size()) return;
        const auto &material = materials[owner.Index];
        const auto fields = [&](this const auto &self, const auto &value) -> void {
            using T = std::remove_cvref_t<decltype(value)>;
            if constexpr (std::same_as<T, TextureInfo>) add(ReferenceKind::Texture, uint32_t(reinterpret_cast<const std::byte *>(&value.Slot) - reinterpret_cast<const std::byte *>(&material)));
            else if constexpr (field::Walkable<T>) field::ForEach(value, [&](const auto &member, auto) { self(member); });
        };
        fields(material);
        return;
    }
    if (owner.Kind != ItemKind::Entity || !r.valid(e)) return;
    if (r.all_of<ObjectKind>(e) && !r.any_of<BoneIndex, SubElementOf>(e)) add(ReferenceKind::Parent);
    if (const auto *instance = r.try_get<const Instance>(e); instance && r.all_of<GeometryProcedure>(instance->Entity)) add(ReferenceKind::Geometry);
    if (r.all_of<HierarchyInstance>(e)) add(ReferenceKind::InstanceSource);
    if (r.all_of<ArmatureModifier>(e)) add(ReferenceKind::Skin);
    if (const auto *constraints = r.try_get<const BoneConstraints>(e)) {
        for (uint32_t i = 0u; i < constraints->Stack.size(); ++i) add(ReferenceKind::Constraint, i);
    }
    if (const auto *collider = r.try_get<const ColliderShape>(e); collider && IsMeshBackedShape(collider->Shape)) add(ReferenceKind::ColliderMesh);
    if (r.all_of<ColliderMaterial>(e)) {
        add(ReferenceKind::PhysicsMaterial);
        add(ReferenceKind::CollisionFilter);
    }
    if (r.all_of<PhysicsJoint>(e)) {
        add(ReferenceKind::JointBody);
        add(ReferenceKind::JointDefinition);
    }
    if (const auto *trigger = r.try_get<const TriggerNodes>(e)) {
        add(ReferenceKind::TriggerFilter);
        for (uint32_t i = 0u; i < trigger->Nodes.size(); ++i) add(ReferenceKind::TriggerMember, i);
    }
    if (r.all_of<GeometryProcedure>(e) && HasMesh(r, e)) {
        const auto &meshes = r.Resources.get<const MeshStore>();
        const auto &record = meshes.Get(GetMesh(r, e).GetStoreId());
        for (uint32_t i = 0u; i < record.PrimitiveMaterials.Count; ++i) add(ReferenceKind::Material, i);
    }
}

// The component required on the target of an entity reference.
decltype(auto) ReferenceTarget(ReferenceKind kind, auto &&visit) {
    using enum ReferenceKind;
    switch (kind) {
        case Parent:
        case InstanceSource: return visit.template operator()<ObjectKind>();
        case Geometry:
        case ColliderMesh: return visit.template operator()<GeometryProcedure>();
        case Skin: return visit.template operator()<ArmatureObject>();
        case PhysicsMaterial: return visit.template operator()<::PhysicsMaterial>();
        case CollisionFilter:
        case TriggerFilter: return visit.template operator()<::CollisionFilter>();
        case JointDefinition: return visit.template operator()<PhysicsJointDef>();
        case Constraint:
        case JointBody:
        case TriggerMember: return visit.template operator()<Transform>();
        case Material:
        case Texture: std::unreachable();
    }
    std::unreachable();
}

bool Accepts(const state::Registry &r, SceneReference reference, NetworkItem source) {
    using enum ReferenceKind;
    if (source == NetworkItem{}) return reference.Kind != Geometry && reference.Kind != Skin && reference.Kind != ColliderMesh && reference.Kind != Material && reference.Kind != InstanceSource;
    if (reference.Kind == Material) return source.Kind == ItemKind::Material && source.Index < GetMaterials(r).size();
    if (reference.Kind == Texture) {
        const auto *textures = r.try_get<const MaterializedTextures>(state::Document);
        return source.Kind == ItemKind::Texture && textures && source.Index < textures->Items.size();
    }
    const auto e = source.Entity;
    if (source.Kind != ItemKind::Entity || !ReferenceTarget(reference.Kind, [&]<typename T> { return r.all_of<T>(e); })) return false;
    if (reference.Kind == InstanceSource) return !HierarchyParentCycle(r, e, reference.Owner.Entity);
    if (reference.Kind == Parent) return !r.any_of<BoneIndex, SubElementOf>(e);
    if (reference.Kind == Skin) return r.get<const ArmatureObject>(e).Entity == r.get<const ArmatureModifier>(reference.Owner.Entity).ArmatureEntity;
    return true;
}

bool HasReference(const state::Registry &r, SceneReference reference) {
    bool found = false;
    ForEachReference(r, reference.Owner, [&](auto candidate) { found |= candidate == reference; });
    return found;
}
bool CanExpose(const state::Registry &r, std::span<const NetworkItem> members, const SubnetPort &port) {
    const auto item = std::visit(overloaded{
                                     [](SocketLink link) { return NetworkItem{link.FromNode}; },
                                     [](PromotedField field) { return NetworkItem{field.Entity}; },
                                     [](SceneReference reference) { return reference.Owner; },
                                     [](NetworkItem item) { return item; },
                                 },
                                 port.Binding);
    if (std::ranges::contains(members, item)) return true;
    for (const auto member : members) {
        if (member.Kind != ItemKind::Entity) continue;
        if (const auto *ports = r.try_get<const SubnetInterface>(member.Entity)) {
            if (std::ranges::any_of(ports->Ports, [&](const auto &inner) { return port.Binding == inner.Binding; })) return true;
        }
    }
    return false;
}
} // namespace
std::vector<SceneReference> SceneReferences(const state::Registry &r, NetworkItem owner) {
    std::vector<SceneReference> references;
    ForEachReference(r, owner, [&](auto reference) { references.push_back(reference); });
    return references;
}

NetworkItem ReferenceValue(const state::Registry &r, SceneReference reference) {
    if (!HasReference(r, reference)) return {};
    const auto e = reference.Owner.Entity;
    const auto index = reference.Index;
    using enum ReferenceKind;
    switch (reference.Kind) {
        case InstanceSource: return {r.get<const HierarchyInstance>(e).Source};
        case Parent: return {ParentOrNull(r, e)};
        case Geometry: return {r.get<const Instance>(e).Entity};
        case Skin: return {r.get<const ArmatureModifier>(e).ArmatureObjectEntity};
        case Constraint: return {r.get<const BoneConstraints>(e).Stack[index].TargetEntity};
        case ColliderMesh: return {r.get<const ColliderShape>(e).MeshEntity};
        case PhysicsMaterial: return {r.get<const ::ColliderMaterial>(e).PhysicsMaterialEntity};
        case CollisionFilter: return {r.get<const ::ColliderMaterial>(e).CollisionFilterEntity};
        case JointBody: return {r.get<const PhysicsJoint>(e).ConnectedNode};
        case JointDefinition: return {r.get<const PhysicsJoint>(e).JointDefEntity};
        case TriggerFilter: return {r.get<const TriggerNodes>(e).CollisionFilterEntity};
        case TriggerMember: return {r.get<const TriggerNodes>(e).Nodes[index]};
        case Material: {
            const auto &meshes = r.Resources.get<const MeshStore>();
            return {state::Document, meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(GetMesh(r, e).GetStoreId()).PrimitiveMaterials)[index], ItemKind::Material};
        }
        case Texture: {
            uint32_t slot;
            const auto materials = GetMaterials(r);
            std::memcpy(&slot, reinterpret_cast<const std::byte *>(&materials[reference.Owner.Index]) + index, sizeof(slot));
            if (const auto *textures = r.try_get<const MaterializedTextures>(state::Document)) {
                for (uint32_t i = 0; i < textures->Items.size(); ++i) {
                    if (textures->Items[i].SamplerSlot == slot) return {state::Document, i, ItemKind::Texture};
                }
            }
            return {};
        }
    }
    std::unreachable();
}

std::string ReferenceLabel(SceneReference reference) {
    constexpr std::array labels{"Parent", "Geometry", "Skin armature", "Constraint target", "Collider geometry", "Physics material", "Collision filter", "Joint body", "Joint definition", "Trigger filter", "Trigger member", "Material slot", "Texture", "Instance source"};
    auto label = std::string{labels[uint8_t(reference.Kind)]};
    if (reference.Kind == ReferenceKind::Constraint || reference.Kind == ReferenceKind::TriggerMember || reference.Kind == ReferenceKind::Material || reference.Kind == ReferenceKind::Texture) label += " " + std::to_string(reference.Index);
    return label;
}

bool ReferenceAccepts(const state::Registry &r, SceneReference reference, NetworkItem source) {
    return HasReference(r, reference) && Accepts(r, reference, source);
}

std::vector<NetworkItem> ReferenceCandidates(const state::Registry &r, SceneReference reference) {
    std::vector<NetworkItem> candidates;
    if (!HasReference(r, reference)) return candidates;
    if (reference.Kind == ReferenceKind::Material) {
        for (uint32_t i = 0; i < GetMaterials(r).size(); ++i) candidates.push_back({state::Document, i, ItemKind::Material});
    } else if (reference.Kind == ReferenceKind::Texture) {
        if (const auto *textures = r.try_get<const MaterializedTextures>(state::Document))
            for (uint32_t i = 0; i < textures->Items.size(); ++i) candidates.push_back({state::Document, i, ItemKind::Texture});
    } else ReferenceTarget(reference.Kind, [&]<typename T> {
        for (const auto e : r.view<const T>())
            if (Accepts(r, reference, {e})) candidates.push_back({e});
    });
    std::ranges::sort(candidates);
    return candidates;
}
} // namespace procedure

namespace action::nodes {
using namespace procedure;
void Apply(state::Registry &r, const AssignReference &a) {
    if (!ReferenceAccepts(r, a.Reference, a.Source)) return Fail(r, "The source does not match this scene input");
    const auto [e, source] = HierarchyReferenceTarget(r, a.Reference.Owner.Entity, a.Source.Entity, a.Reference.Kind == ReferenceKind::Parent || a.Reference.Kind == ReferenceKind::InstanceSource);
    const auto index = a.Reference.Index;
    using enum ReferenceKind;
    switch (a.Reference.Kind) {
        case InstanceSource: BindHierarchyInstance(r, e, source); return;
        case Parent:
            if (source == state::Null) object::Apply(r, object::ClearParent{e});
            else object::Apply(r, object::SetParent{e, source});
            break;
        case Geometry:
            if (!ObjectReferencesAcyclic(r, {}, state::Null, state::Null, std::pair{e, source})) return Fail(r, "The geometry assignment would create a dependency cycle");
            r.replace<Instance>(e, source);
            break;
        case Skin: r.patch<ArmatureModifier>(e, [&](auto &modifier) { modifier.ArmatureObjectEntity = source; }); break;
        case Constraint: bone::Apply(r, bone::SetConstraintTarget{index, source, e}); break;
        case ColliderMesh: r.patch<ColliderShape>(e, [&](auto &shape) { shape.MeshEntity = source; }); break;
        case PhysicsMaterial: action::Apply(r, Core{UpdateOf<&ColliderMaterial::PhysicsMaterialEntity>(e, source)}); break;
        case CollisionFilter: action::Apply(r, Core{UpdateOf<&ColliderMaterial::CollisionFilterEntity>(e, source)}); break;
        case JointBody: action::Apply(r, Core{UpdateOf<&PhysicsJoint::ConnectedNode>(e, source)}); break;
        case JointDefinition: action::Apply(r, Core{UpdateOf<&PhysicsJoint::JointDefEntity>(e, source)}); break;
        case TriggerFilter: action::Apply(r, Core{UpdateOf<&TriggerNodes::CollisionFilterEntity>(e, source)}); break;
        case TriggerMember: r.patch<TriggerNodes>(e, [&](auto &trigger) { trigger.Nodes[index] = source; }); break;
        case Material: object::Apply(r, object::SetMaterialAssignment{index, a.Source.Index, e}); break;
        case Texture: {
            const auto *textures = r.try_get<const MaterializedTextures>(state::Document);
            const auto slot = source == state::Null ? InvalidSlot : textures->Items[a.Source.Index].SamplerSlot;
            object::Apply(r, object::UpdateMaterial<uint32_t>{a.Reference.Owner.Index, uint16_t(index), slot});
            break;
        }
    }
}

void Apply(state::Registry &r, const ExposePort &a) {
    SubnetPort port{.Name = a.Name ? *a.Name : std::string{}, .Binding = a.Binding};
    const bool valid = std::visit(overloaded{
                                      [](SocketLink) { return false; },
                                      [&](PromotedField field) {
                                          port.Type = SocketType::Value;
                                          return !port.Name.empty() && bool(action::ReadUpdate(r, field.Entity, field.Component, field.Offset));
                                      },
                                      [&](SceneReference reference) {
                                          port.Type = SocketType::Reference;
                                          if (port.Name.empty()) port.Name = ReferenceLabel(reference);
                                          return HasReference(r, reference);
                                      },
                                      [&](NetworkItem item) {
                                          port.Type = SocketType::Reference;
                                          port.Direction = SocketDirection::Out;
                                          if (port.Name.empty()) port.Name = item.Kind == ItemKind::Entity ? GetName(r, item.Entity) : item.Kind == ItemKind::Material ? "Material" :
                                                                                                                                                                         "Texture";
                                          return true;
                                      },
                                  },
                                  a.Binding);
    if (!valid || !r.all_of<NetworkSubnet>(a.Subnet) || !CanExpose(r, NetworkItems(r, a.Subnet), port)) return Fail(r, "Expose an internal field or reference through each enclosing subnet");
    const auto &interface = r.get<const SubnetInterface>(a.Subnet);
    if (std::ranges::any_of(interface.Ports, [&](const auto &existing) { return port.Binding == existing.Binding; })) return;
    if (interface.NextId >= UINT16_MAX) return Fail(r, "The subnet has no available port identities");
    r.patch<SubnetInterface>(a.Subnet, [&](auto &interface) {
        port.Id = uint16_t(interface.NextId++);
        interface.Ports.push_back(std::move(port));
    });
}
} // namespace action::nodes

namespace procedure {
void ReconcileNetworkPorts(state::Registry &r) {
    const auto subnets = SortedEntities(r.view<const NetworkSubnet>());
    const auto order = DependencyOrder(uint32_t(subnets.size()), [&](auto edge) {
        for (uint32_t child = 0; child < subnets.size(); ++child) {
            const auto parent = r.get<const NetworkSubnet>(subnets[child]).Parent;
            const auto at = std::ranges::lower_bound(subnets, parent);
            if (at != subnets.end() && *at == parent) edge(child, uint32_t(at - subnets.begin()));
        }
    });
    for (const auto index : order) {
        const auto subnet = subnets[index];
        const auto members = NetworkItems(r, subnet);
        const auto valid = [&](const SubnetPort &port) {
            if (const auto *reference = std::get_if<SceneReference>(&port.Binding); reference && !HasReference(r, *reference)) return false;
            if (const auto *field = std::get_if<PromotedField>(&port.Binding); field && !action::ReadUpdate(r, field->Entity, field->Component, field->Offset)) return false;
            return CanExpose(r, members, port);
        };
        if (!std::ranges::all_of(r.get<const SubnetInterface>(subnet).Ports, valid))
            r.patch<SubnetInterface>(subnet, [&](auto &interface) { std::erase_if(interface.Ports, [&](const auto &port) { return !valid(port); }); });
    }
}
void ExposeNetworkReferences(state::Registry &r, state::Entity subnet) {
    const auto internal = NetworkLeaves(r, subnet);
    const auto all = NetworkLeaves(r, NetworkRoot(r, subnet));
    const auto expose = [&](this const auto &self, state::Entity group, NetworkItem item, std::optional<SceneReference> reference) -> void {
        const auto child = NetworkRepresentative(r, group, item);
        if (child != item && r.all_of<NetworkSubnet>(child.Entity)) self(child.Entity, item, reference);
        action::nodes::Apply(r, action::nodes::ExposePort{group, reference ? PortBinding{*reference} : PortBinding{item}, {}});
    };
    for (const auto owner : all) {
        const bool owner_inside = std::ranges::binary_search(internal, owner);
        for (const auto reference : SceneReferences(r, owner)) {
            const auto source = ReferenceValue(r, reference);
            if (source == NetworkItem{}) continue;
            const bool source_inside = std::ranges::binary_search(internal, source);
            if (owner_inside && !source_inside) expose(subnet, owner, reference);
            if (!owner_inside && source_inside) expose(subnet, source, {});
        }
    }
}
} // namespace procedure
