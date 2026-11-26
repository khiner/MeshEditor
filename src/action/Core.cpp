#include "action/Core.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "animation/AnimationData.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/ModalModes.h"
#include "gpu/ViewportTheme.h"
#include "physics/PhysicsTypes.h"
#include "procedure/NodeTheme.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
#include "scene/CameraLens.h"
#include "snapshot/SnapshotRoles.h"
#include "timeline/Timeline.h"
#include "timeline/TimelineTheme.h"
#include "ui/InterfaceTheme.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportInteractionState.h"

namespace action {
namespace {
// The operations an Update runs on one updatable component type.
struct UpdateEntry {
    state::TypeKey Key;
    state::TypeId Type;
    std::string_view Name;
    bool Placement; // A write to the component addresses the entity that places its hierarchy instance.
    const FieldRange &(*FieldAt)(uint16_t offset);
    bool (*Has)(const state::Registry &, state::Entity);
    state::Entity (*Active)(const state::Registry &);
    void (*ForEachSelected)(state::Registry &, const std::function<void(state::Entity)> &);
    void (*Read)(const state::Registry &, state::Entity, uint16_t offset, void *dst, size_t size);
    void (*Write)(state::Registry &, state::Entity, uint16_t offset, const void *src, size_t size);
    const FieldSpec &(*Bounds)(const state::Registry &, state::Entity, uint16_t offset);
    std::optional<Core> (*ReadUpdate)(const state::Registry &, state::Entity, uint16_t offset);
    // Returns whether the entity value can be assigned, and fails when it cannot.
    // Null for a component that accepts any entity.
    bool (*AcceptsReference)(state::Registry &, const UpdateEntry &, const Update<state::Entity> &);
};

template<typename C>
std::optional<Core> ReadComponentUpdate(const state::Registry &r, state::Entity entity, uint16_t offset) {
    std::optional<Core> result;
    const auto *component = r.try_get<const C>(entity);
    if (!component || snapshot::SnapshotTable()[state::Type<C>()].Kind == snapshot::Role::Workspace) return result;
    const auto visit = [&](this const auto &self, const auto &value) -> void {
        using V = std::remove_cvref_t<decltype(value)>;
        const auto at = reinterpret_cast<const std::byte *>(&value) - reinterpret_cast<const std::byte *>(component);
        if constexpr (std::constructible_from<Core, Update<UpdateValue<V>>>) {
            if (at == offset) result = Update<UpdateValue<V>>{entity, state::Key<C>(), offset, ToUpdateValue(value)};
        } else if constexpr (field::Walkable<V>) {
            field::ForEach(value, [&](const auto &member, auto) { self(member); });
        }
    };
    visit(*component);
    return result;
}

bool AcceptsObjectReference(state::Registry &r, const UpdateEntry &entry, const Update<state::Entity> &a) {
    std::vector<state::Entity> nodes;
    ForEachTarget(a.Target, [&] { return entry.Active(r); }, [&](auto &&fn) { entry.ForEachSelected(r, fn); }, [&](state::Entity e) { if (entry.Has(r, e)) nodes.push_back(e); });
    if (procedure::ObjectReferencesAcyclic(r, nodes, a.Value)) return true;
    Fail(r, "Object references would create a geometry dependency cycle");
    return false;
}
bool AcceptsPrototype(state::Registry &r, const UpdateEntry &entry, const Update<state::Entity> &a) {
    if (a.Value != state::Null && !r.all_of<ObjectKind>(a.Value)) {
        Fail(r, "Choose an object hierarchy as the prototype");
        return false;
    }
    bool cycle = false;
    const auto uses = procedure::DefinitionUses(r);
    ForEachTarget(a.Target, [&] { return entry.Active(r); }, [&](auto &&fn) { entry.ForEachSelected(r, fn); }, [&](state::Entity node) {
        const auto *owner = r.try_get<const procedure::ProcedureNode>(node);
        if (!owner) return;
        const auto meshes = procedure::MeshesUsing(r, uses, owner->Network);
        for (const auto [object, instance] : r.view<const Instance>().each())
            if (std::ranges::contains(meshes, instance.Entity) && HierarchyParentCycle(r, a.Value, object)) cycle = true; });
    if (cycle) Fail(r, "The prototype would recursively instance its producer");
    return !cycle;
}

template<typename C> constexpr UpdateEntry EntryFor() {
    using Traits = UpdateTraits<C>;
    return {
        .Key = state::Key<C>(),
        .Type = state::Type<C>(),
        .Name = state::LeafName<C>(),
        .Placement = std::same_as<C, Transform> || std::same_as<C, PosedLocal> || std::same_as<C, Visibility>,
        .FieldAt = &FieldAt<C>,
        .Has = &Traits::Has,
        .Active = &Traits::Active,
        .ForEachSelected = &Traits::ForEachSelected,
        .Read = &Traits::Read,
        .Write = &Traits::Write,
        .Bounds = &Traits::Bounds,
        .ReadUpdate = &ReadComponentUpdate<C>,
        .AcceptsReference = std::same_as<C, procedure::ObjectInfo> ? &AcceptsObjectReference : std::same_as<C, procedure::InstanceOnPoints> ? &AcceptsPrototype :
                                                                                                                                              nullptr,
    };
}
constexpr auto Entries = []<typename... Cs>(TypeList<Cs...>) { return std::array{EntryFor<Cs>()...}; }(UpdatableComponents{});

// The entry of the updatable component with `key`, or null for a key outside the list.
const UpdateEntry *EntryOf(state::TypeKey key) {
    const auto found = std::ranges::find(Entries, key, &UpdateEntry::Key);
    return found == Entries.end() ? nullptr : &*found;
}

// Writes the field of the component on each entity the target resolves to.
template<typename Field>
void ApplyUpdate(state::Registry &r, const Update<Field> &a) {
    const auto *entry = EntryOf(a.ComponentType);
    if (!entry) return;
    if constexpr (std::same_as<Field, state::Entity>) {
        if (entry->AcceptsReference && !entry->AcceptsReference(r, *entry, a)) return;
    }
    const auto write = [&](state::Entity e, Field value) {
        if constexpr (std::same_as<Field, state::Entity>) std::tie(e, value) = HierarchyReferenceTarget(r, e, value, entry->Placement);
        else e = HierarchyEditTarget(r, e, entry->Placement);
        if constexpr (DeltaField<Field> && !RotationField<Field>) value = ClampField(value, entry->Bounds(r, e, a.Offset));
        entry->Write(r, e, a.Offset, &value, sizeof(Field));
    };
    if constexpr (DeltaField<Field>) {
        if (std::holds_alternative<OnSelectedDelta>(a.Target)) {
            // Offset each selected target by the active target's change from its drag start.
            const auto active = entry->Active(r);
            if (active == state::Null) return;
            const auto start = [&](state::Entity e) {
                return FieldGestureStart<Field>(r, e, entry->Type, a.Offset, [&](Field &v) { entry->Read(r, e, a.Offset, &v, sizeof(Field)); });
            };
            const auto active_start = start(active);
            entry->ForEachSelected(r, [&](state::Entity e) {
                if (e == active) {
                    write(e, a.Value);
                } else if constexpr (std::integral<Field>) {
                    // Accumulate in a wider signed type so an unsigned field can't wrap on a downward delta.
                    const auto value = int64_t(start(e)) + int64_t(a.Value) - int64_t(active_start);
                    write(e, Field(std::clamp<int64_t>(value, std::numeric_limits<Field>::min(), std::numeric_limits<Field>::max())));
                } else if constexpr (std::integral<Limit<Field>>) {
                    using L = Limit<Field>;
                    auto value = start(e);
                    for (int c = 0; c < Components<Field>; ++c) {
                        value[c] = L(std::clamp<int64_t>(int64_t(value[c]) + int64_t(a.Value[c]) - int64_t(active_start[c]), std::numeric_limits<L>::min(), std::numeric_limits<L>::max()));
                    }
                    write(e, value);
                } else if constexpr (RotationField<Field>) {
                    // Rotate each target by the active target's rotation since the drag start.
                    write(e, Normalize(a.Value * Conjugate(active_start) * start(e)));
                } else {
                    write(e, start(e) + (a.Value - active_start));
                }
            });
            return;
        }
    }
    ForEachTarget(
        a.Target,
        [&] { return entry->Active(r); },
        [&](auto &&fn) { entry->ForEachSelected(r, fn); },
        [&](state::Entity e) {
            if (entry->Has(r, e)) write(e, a.Value);
        }
    );
}
} // namespace

ComponentField UpdatedField(state::TypeKey key, uint16_t offset) {
    const auto *entry = EntryOf(key);
    return entry ? ComponentField{std::string{entry->Name}, entry->FieldAt(offset)} : ComponentField{{}, NoField};
}

std::optional<Core> ReadUpdate(const state::Registry &r, state::Entity entity, state::TypeKey key, uint16_t offset) {
    const auto *entry = EntryOf(key);
    return entry ? entry->ReadUpdate(r, entity, offset) : std::nullopt;
}

void TrackFieldRemovals(state::DirtySet &set) {
    for (const auto &entry : Entries) set.Track(entry.Type, state::Event::Destroy);
}

state::Entity ActiveUpdateTarget(const state::Registry &r, state::TypeKey key) {
    const auto *entry = EntryOf(key);
    return entry ? entry->Active(r) : state::Null;
}

void Apply(state::Registry &r, const Core &action) {
    std::visit(
        overloaded{
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate(r, a); },
            [&](const DestroyEntity &a) { r.destroy(a.Entity); },
        },
        action
    );
}

bool IsRecordable(const Core &action) {
    return std::visit(
        overloaded{
            []<typename Field>(const Update<Field> &a) {
                const auto *entry = EntryOf(a.ComponentType);
                return entry && snapshot::SnapshotTable()[entry->Type].Kind != snapshot::Role::Workspace;
            },
            [](const DestroyEntity &) { return true; },
        },
        action
    );
}
} // namespace action
