#pragma once
#include "PathSerialize.h"
#include "TypeList.h"
#include "numeric/Serialize.h"
#include "snapshot/NativeSize.h"
#include "snapshot/SnapshotRoles.h"
#include "snapshot/Workspace.h"
#include "state/Registry.h"

namespace snapshot::detail {
using Comparator = bool (*)(const void *, const void *);
struct Tables {
    SnapshotEntries Snapshots{};
    std::array<Comparator, state::SchemaSize> Comparators{};
    std::array<bool, state::SchemaSize> Classified{};
};
// Non-default-constructible serialized types specialize this emplacer.
template<typename C>
inline constexpr void (*CustomEmplace)(state::Registry &, state::Entity, std::span<const std::byte>) = nullptr;

// Domains with mixed canonical/cache objects copy only canonical fields.
template<typename C> C CopyNative(const C &value) { return value; }
template<typename C> void PrepareNative(C &) {}

template<typename C> struct ComponentPage : NativePage {
    union Slot {
        C Value;
        Slot() {}
        ~Slot() {}
    };
    std::array<Slot, state::Table::PageCount> Values;
    ~ComponentPage() {
        for (auto bits = Mask; bits; bits &= bits - 1u) std::destroy_at(&Values[std::countr_zero(bits)].Value);
    }
};

template<typename C>
void EmplaceTag(state::Registry &r, state::Entity e, std::span<const std::byte>) {
    r.emplace_or_replace<C>(e);
}

template<typename C>
void SerializeThunk(const void *component, std::vector<std::byte> &out) {
    const auto start = out.size();
    // Grow only the appended record.
    // Vector capacity handles allocation growth.
    zpp::bits::out archive{out, zpp::bits::exact_enlarger{}};
    archive.reset(start);
    const auto result = archive(*static_cast<const C *>(component));
    out.resize(zpp::bits::failure(result) ? start : archive.position());
}

template<typename C>
void SerializeBytes(const void *component, std::vector<std::byte> &out) {
    const auto *bytes = static_cast<const std::byte *>(component);
    out.insert(out.end(), bytes, bytes + sizeof(C));
}

// The numeric types serialize their components in memory order.
template<typename T>
concept NumericComponents = std::same_as<T, numeric::vec2> || std::same_as<T, numeric::vec3> || std::same_as<T, numeric::vec4> ||
    std::same_as<T, numeric::uvec2> || std::same_as<T, numeric::uvec3> || std::same_as<T, numeric::uvec4> || std::same_as<T, numeric::dvec3> ||
    std::same_as<T, numeric::quat> || std::same_as<T, numeric::mat3> || std::same_as<T, numeric::mat4>;

// A padding-free trivially copyable type, whose object representation is its zpp encoding.
template<typename T> struct PaddingFreeTrait {
    static constexpr bool Value = [] {
        if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T> || NumericComponents<T>) return true;
        else if constexpr (std::is_aggregate_v<T> && std::is_trivially_copyable_v<T> && !std::is_empty_v<T>) {
            return zpp::bits::access::visit_members_types<T>([]<typename... M>() {
                return std::bool_constant<(... && PaddingFreeTrait<std::remove_cvref_t<M>>::Value) && (size_t{} + ... + sizeof(M)) == sizeof(T)>{};
            })();
        } else {
            return false;
        }
    }();
};
template<typename T, size_t N> struct PaddingFreeTrait<std::array<T, N>> {
    static constexpr bool Value = PaddingFreeTrait<T>::Value;
};
template<typename C>
constexpr bool EncodesAsBytes = PaddingFreeTrait<C>::Value;

// Decodes a whole record in the entry's encoding.
template<typename C> bool Decode(std::span<const std::byte> bytes, C &value) {
    if constexpr (std::is_empty_v<C>) {
        return bytes.empty();
    } else if constexpr (EncodesAsBytes<C>) {
        if (bytes.size() != sizeof(C)) return false;
        std::memcpy(&value, bytes.data(), sizeof(C));
        return true;
    } else {
        zpp::bits::in in{bytes};
        return zpp::bits::success(in(value)) && in.position() == bytes.size();
    }
}

template<typename C>
void EmplaceDecoded(state::Registry &r, state::Entity e, std::span<const std::byte> bytes) {
    if constexpr (CustomEmplace<C> != nullptr) CustomEmplace<C>(r, e, bytes);
    else if (C value{}; Decode(bytes, value)) r.emplace_or_replace<C>(e, std::move(value));
}

// Checks the byte encoding against zpp on a value whose bytes count upward, so reordered or resized members differ.
template<typename C>
bool BytesMatchSerialization() {
    alignas(C) std::byte storage[sizeof(C)];
    for (size_t i = 0; i < sizeof(C); ++i) storage[i] = std::byte(i + 1);
    std::vector<std::byte> encoded;
    SerializeThunk<C>(storage, encoded);
    return encoded.size() == sizeof(C) && std::memcmp(encoded.data(), storage, sizeof(C)) == 0;
}

// Serialized values compare through their encoding.
// Byte-encoded and derived aggregates compare by bytes.
template<typename C, bool Recorded>
bool ValuesEqual(const void *a, const void *b) {
    if constexpr (std::is_empty_v<C>) {
        return true;
    } else if constexpr (Recorded && !EncodesAsBytes<C>) {
        std::vector<std::byte> ba, bb;
        SerializeThunk<C>(a, ba);
        SerializeThunk<C>(b, bb);
        return ba == bb;
    } else {
        return std::memcmp(a, b, sizeof(C)) == 0;
    }
}

// Returns nullptr for derived types without a valid comparator.
template<typename C, bool Recorded>
constexpr Comparator MakeComparator() {
    if constexpr (std::is_empty_v<C> || Recorded || (std::is_trivially_copyable_v<C> && std::is_aggregate_v<C>)) return &ValuesEqual<C, Recorded>;
    else return nullptr;
}

// Selects the Tag, Bytes or Serialized encoding from the component traits.
template<typename C>
snapshot::SnapshotEntry MakeEntry() {
    using snapshot::Encoding;
    if constexpr (std::is_empty_v<C>) return {Encoding::Tag, nullptr, &EmplaceTag<C>};
    else if constexpr (EncodesAsBytes<C>) {
        assert(BytesMatchSerialization<C>() && "a byte-encoded component must serialize as its object representation");
        return {Encoding::Bytes, &SerializeBytes<C>, &EmplaceDecoded<C>, sizeof(C)};
    } else {
        return {Encoding::Serialized, &SerializeThunk<C>, &EmplaceDecoded<C>};
    }
}

// Encodes a whole record in the entry's encoding.
template<typename C> void Encode(const C &value, std::vector<std::byte> &out) {
    if constexpr (!std::is_empty_v<C>) {
        if constexpr (EncodesAsBytes<C>) SerializeBytes<C>(&value, out);
        else SerializeThunk<C>(&value, out);
    }
}

struct WorkspaceOrder {
    bool operator()(state::Entity a, state::Entity b) const {
        return std::pair{state::Index(a), state::Generation(a)} < std::pair{state::Index(b), state::Generation(b)};
    }
};
template<typename C> using RetainedValues = std::map<state::Entity, C, WorkspaceOrder>;
template<typename C> RetainedValues<C> &Retained(state::Registry &r) {
    return r.Resources.get<Retention>().Slots[state::Type<C>()].template Emplace<RetainedValues<C>>();
}
template<typename C> void Retain(state::Registry &r, state::Entity e, C value) {
    CaptureWorkspace(r, state::Type<C>(), e);
    Retained<C>(r).insert_or_assign(e, std::move(value));
}
template<typename C> void RetainDying(state::Registry &r, state::Entity e) {
    if (!r.Resources.get<const Retention>().Installing) Retain(r, e, CopyNative(r.get<C>(e)));
}
template<typename C> void VisitWorkspacePage(const state::Registry &r, uint32_t page, auto &&fn) {
    const auto &table = r.storage<C>();
    const auto *retained = static_cast<const RetainedValues<C> *>(r.Resources.get<const Retention>().Slots[state::Type<C>()].Value);
    for (auto bits = table.mask(page); bits; bits &= bits - 1u) {
        const auto live = table.entity(page, uint32_t(std::countr_zero(bits)));
        fn(live, *table.template at<C>(live));
    }
    if (retained) {
        for (auto it = retained->lower_bound(state::MakeEntity(page * state::Table::PageCount, 0)); it != retained->end() && state::Index(it->first) / state::Table::PageCount == page; ++it) {
            if (!table.contains(it->first)) fn(it->first, it->second);
        }
    }
}
template<typename C> struct WorkspacePage : WorkspacePageHeader {
    WorkspacePage() : WorkspacePageHeader{state::Type<C>()} {}
    std::vector<std::pair<state::Entity, C>> Values;
};
template<typename C, Role ComponentRole>
void Add(Tables &tables) {
    const auto id = state::Type<C>();
    if (std::exchange(tables.Classified[id], true)) throw std::logic_error("Duplicate snapshot component classification");
    constexpr bool Recorded = ComponentRole != Role::Derived;
    tables.Comparators[id] = MakeComparator<C, Recorded>();
    if constexpr (Recorded) {
        auto entry = MakeEntry<C>();
        entry.Name = state::TypeName<C>();
        entry.Key = state::Key<C>();
        entry.Equal = MakeComparator<C, Recorded>();
        if constexpr (ComponentRole == Role::Persistent) {
            entry.Rebind = [](state::Registry &r, state::Entity previous, state::Entity current) {
                auto value = CopyNative(r.get<C>(previous));
                r.remove<C>(previous);
                if (current != state::Null) {
                    PrepareNative(value);
                    r.emplace_or_replace<C>(current, std::move(value));
                }
            };
            entry.CopyPage = [](const state::Table &table, uint32_t page) {
                auto value = std::make_unique<ComponentPage<C>>();
                auto bytes = uint64_t(sizeof(ComponentPage<C>));
                for (auto bits = table.mask(page); bits; bits &= bits - 1u) {
                    const auto slot = uint32_t(std::countr_zero(bits));
                    const auto &copy = *std::construct_at(&value->Values[slot].Value, CopyNative(*table.at<C>(table.entity(page, slot))));
                    value->Mask |= 1u << slot;
                    bytes += NativeExtra(copy);
                }
                auto *header = static_cast<NativePage *>(value.release());
                return store::Blob{reinterpret_cast<std::byte *>(header), sizeof(ComponentPage<C>), [](void *p) { delete static_cast<ComponentPage<C> *>(static_cast<NativePage *>(p)); }, bytes};
            };
            entry.PageValue = [](const store::Blob &value, uint32_t slot) -> const void * {
                return &static_cast<const ComponentPage<C> *>(reinterpret_cast<const NativePage *>(value.Data))->Values[slot].Value;
            };
            entry.MovePageValue = [](state::Registry &r, state::Entity e, store::Blob &value, uint32_t slot) {
                auto &native = static_cast<ComponentPage<C> *>(reinterpret_cast<NativePage *>(value.Data))->Values[slot].Value;
                PrepareNative(native);
                r.emplace_or_replace<C>(e, std::move(native));
            };
        } else {
            entry.CopyWorkspacePage = [](const state::Registry &r, uint32_t page) {
                std::unique_ptr<WorkspacePage<C>> values;
                uint64_t bytes = sizeof(WorkspacePage<C>);
                VisitWorkspacePage<C>(r, page, [&](state::Entity e, const C &v) {
                    if (!values) values = std::make_unique<WorkspacePage<C>>();
                    const auto &copy = values->Values.emplace_back(e, CopyNative(v)).second;
                    bytes += NativeExtra(copy);
                });
                if (!values) return store::Blob{};
                std::ranges::sort(values->Values, WorkspaceOrder{}, &std::pair<state::Entity, C>::first);
                bytes += values->Values.capacity() * sizeof(std::pair<state::Entity, C>);
                auto *header = static_cast<WorkspacePageHeader *>(values.release());
                return store::Blob{reinterpret_cast<std::byte *>(header), sizeof(WorkspacePage<C>), [](void *p) { delete static_cast<WorkspacePage<C> *>(static_cast<WorkspacePageHeader *>(p)); }, bytes};
            };
            entry.ReadWorkspacePage = [](const state::Registry &r, uint32_t page, std::vector<std::byte> &out) {
                std::vector<std::pair<state::Entity, const C *>> values;
                VisitWorkspacePage<C>(r, page, [&](state::Entity e, const C &v) { values.emplace_back(e, &v); });
                std::ranges::sort(values, WorkspaceOrder{}, &std::pair<state::Entity, const C *>::first);
                out.clear();
                if (values.empty()) return;
                zpp::bits::out archive{out};
                archive(uint32_t(values.size())).or_throw();
                for (const auto &[e, value] : values) archive(e, *value).or_throw();
            };
            entry.EncodeWorkspacePage = [](const store::Blob &blob, std::vector<std::byte> &out) {
                out.clear();
                zpp::bits::out{out}(static_cast<const WorkspacePage<C> *>(reinterpret_cast<const WorkspacePageHeader *>(blob.Data))->Values).or_throw();
            };
            entry.ApplyWorkspacePage = [](state::Registry &r, uint32_t page, store::Blob &blob) {
                std::vector<std::pair<state::Entity, C>> decoded;
                if (blob.Data && !blob.Destroy) zpp::bits::in{blob.View()}(decoded).or_throw();
                auto &values = blob.Destroy ? static_cast<WorkspacePage<C> *>(reinterpret_cast<WorkspacePageHeader *>(blob.Data))->Values : decoded;
                auto &retained = Retained<C>(r);
                auto it = retained.lower_bound(state::MakeEntity(page * state::Table::PageCount, 0));
                while (it != retained.end() && state::Index(it->first) / state::Table::PageCount == page) it = retained.erase(it);
                const auto &table = r.storage<C>();
                for (auto bits = table.mask(page); bits; bits &= bits - 1u) {
                    const auto e = table.entity(page, uint32_t(std::countr_zero(bits)));
                    if (!std::ranges::binary_search(values, e, WorkspaceOrder{}, &std::pair<state::Entity, C>::first)) r.remove<C>(e);
                }
                for (auto &[e, value] : values) {
                    if (r.all_of<C>(e)) r.replace_if_changed(e, value);
                    else retained.insert_or_assign(e, std::move(value));
                }
            };
            entry.ClearRetained = [](state::Registry &r) {
                auto &values = Retained<C>(r);
                for (const auto &[e, value] : values) CaptureWorkspace(r, state::Type<C>(), e);
                values.clear();
            };
            entry.Connect = [](state::Registry &r) { r.on_destroy<C, &RetainDying<C>>(); };
            entry.Return = [](state::Registry &r, state::Entity e, bool emplace) {
                auto &values = Retained<C>(r);
                const auto found = values.find(e);
                if (found == values.end()) return;
                CaptureWorkspace(r, state::Type<C>(), e);
                auto retained = values.extract(found);
                if (retained && emplace) r.emplace_or_replace<C>(e, std::move(retained.mapped()));
            };
            entry.AppendRecords = [](const state::Registry &r, std::vector<WorkspaceRecord> &records) {
                const auto append = [&](state::Entity e, const C &value) { Encode(value, records.emplace_back(e, std::vector<std::byte>{}).second); };
                for (const auto e : r.view<const C>()) append(e, r.get<const C>(e));
                if (const auto *values = static_cast<const RetainedValues<C> *>(r.Resources.get<const Retention>().Slots[state::Type<C>()].Value)) {
                    for (const auto &[e, value] : *values) append(e, value);
                }
            };
            entry.Place = [](state::Registry &r, state::Entity e, std::span<const std::byte> bytes) {
                C value{};
                if (!Decode(bytes, value)) return;
                // An identity without this value may become its kind of entity through history.
                if (r.all_of<C>(e)) r.replace_if_changed(e, value);
                else Retain(r, e, std::move(value));
            };
        }
        tables.Snapshots[id] = entry;
    }
    tables.Snapshots[id].Kind = ComponentRole;
}
// Classifies one domain's Persistent, Workspace, and Derived components.
template<typename... Ps, typename... Ws, typename... Ds>
void Classify(Tables &tables, TypeList<Ps...>, TypeList<Ws...>, TypeList<Ds...>) {
    (Add<Ps, Role::Persistent>(tables), ...);
    (Add<Ws, Role::Workspace>(tables), ...);
    (Add<Ds, Role::Derived>(tables), ...);
}

void RegisterAudio(Tables &);
void RegisterPhysics(Tables &);
void RegisterArmature(Tables &);
void RegisterMesh(Tables &);
void RegisterViewport(Tables &);
void RegisterAssets(Tables &);
void RegisterScene(Tables &);
} // namespace snapshot::detail
