#pragma once
#include "PathSerialize.h"
#include "numeric/Serialize.h"
#include "snapshot/NativeSize.h"
#include "snapshot/SnapshotRoles.h"
#include "state/Scene.h"
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace snapshot::detail {
using Comparator = bool (*)(const void *, const void *);
struct Tables {
    SnapshotEntries Snapshots{};
    std::array<Comparator, state::SchemaSize> Comparators{};
    std::array<bool, state::SchemaSize> Classified{};
};
// Non-default-constructible serialized types specialize this emplacer.
template<typename C>
inline constexpr void (*CustomEmplace)(state::Scene &, state::Entity, std::span<const std::byte>) = nullptr;

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
void EmplaceTag(state::Scene &r, state::Entity e, std::span<const std::byte>) {
    r.emplace_or_replace<C>(e);
}

template<typename C>
void SerializeThunk(const void *component, std::vector<std::byte> &out) {
    const auto start = out.size();
    // Grow only the appended record. Vector capacity handles allocation growth.
    zpp::bits::out archive{out, zpp::bits::exact_enlarger{}};
    archive.reset(start);
    // zpp aggregate reflection mis-encodes large const aggregates, while the output archive only reads this reference.
    const auto result = archive(const_cast<C &>(*static_cast<const C *>(component)));
    out.resize(zpp::bits::failure(result) ? start : archive.position());
}

template<typename C>
void EmplaceSerialized(state::Scene &r, state::Entity e, std::span<const std::byte> bytes) {
    if constexpr (CustomEmplace<C> != nullptr) CustomEmplace<C>(r, e, bytes);
    else {
        C value;
        if (zpp::bits::failure(zpp::bits::in{bytes}(value))) return;
        r.emplace_or_replace<C>(e, std::move(value));
    }
}

template<typename C>
void SerializeBytes(const void *component, std::vector<std::byte> &out) {
    const auto *bytes = static_cast<const std::byte *>(component);
    out.insert(out.end(), bytes, bytes + sizeof(C));
}

template<typename C>
void EmplaceBytes(state::Scene &r, state::Entity e, std::span<const std::byte> bytes) {
    if constexpr (CustomEmplace<C> != nullptr) CustomEmplace<C>(r, e, bytes);
    else {
        if (bytes.size() != sizeof(C)) return;
        C value;
        std::memcpy(&value, bytes.data(), sizeof(C));
        r.emplace_or_replace<C>(e, std::move(value));
    }
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
        } else return false;
    }();
};
template<typename T, size_t N> struct PaddingFreeTrait<std::array<T, N>> {
    static constexpr bool Value = PaddingFreeTrait<T>::Value;
};
template<typename C>
constexpr bool EncodesAsBytes = PaddingFreeTrait<C>::Value;

// Checks the byte encoding against zpp on a value whose bytes count upward, so reordered or resized members differ.
template<typename C>
bool BytesMatchSerialization() {
    alignas(C) std::byte storage[sizeof(C)];
    for (size_t i = 0; i < sizeof(C); ++i) storage[i] = std::byte(i + 1);
    std::vector<std::byte> encoded;
    SerializeThunk<C>(storage, encoded);
    return encoded.size() == sizeof(C) && std::memcmp(encoded.data(), storage, sizeof(C)) == 0;
}

// Serialized values compare through their encoding. Byte-encoded and derived aggregates compare by bytes.
template<typename C, bool Persistent>
bool ValuesEqual(const void *a, const void *b) {
    if constexpr (std::is_empty_v<C>) {
        return true;
    } else if constexpr (Persistent && !EncodesAsBytes<C>) {
        std::vector<std::byte> ba, bb;
        SerializeThunk<C>(a, ba);
        SerializeThunk<C>(b, bb);
        return ba == bb;
    } else {
        return std::memcmp(a, b, sizeof(C)) == 0;
    }
}

// Returns nullptr for derived types without a valid comparator.
template<typename C, bool Persistent>
constexpr Comparator MakeComparator() {
    if constexpr (std::is_empty_v<C> || Persistent || (std::is_trivially_copyable_v<C> && std::is_aggregate_v<C>)) return &ValuesEqual<C, Persistent>;
    else return nullptr;
}

// Selects the Tag, Bytes or Serialized encoding from the component traits.
template<typename C>
snapshot::SnapshotEntry MakeEntry() {
    using snapshot::Encoding;
    if constexpr (std::is_empty_v<C>) return {Encoding::Tag, nullptr, &EmplaceTag<C>};
    else if constexpr (EncodesAsBytes<C>) {
        assert(BytesMatchSerialization<C>() && "a byte-encoded component must serialize as its object representation");
        return {Encoding::Bytes, &SerializeBytes<C>, &EmplaceBytes<C>, sizeof(C)};
    } else return {Encoding::Serialized, &SerializeThunk<C>, &EmplaceSerialized<C>};
}

template<typename C, bool Persistent>
void Add(Tables &tables) {
    const auto id = state::Type<C>();
    if (std::exchange(tables.Classified[id], true)) throw std::logic_error("Duplicate snapshot component classification");
    tables.Comparators[id] = MakeComparator<C, Persistent>();
    if constexpr (Persistent) {
        auto entry = MakeEntry<C>();
        entry.Name = state::TypeName<C>();
        entry.Rebind = [](state::Scene &r, state::Entity previous, state::Entity current) {
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
                auto &copy = *std::construct_at(&value->Values[slot].Value, CopyNative(*table.at<C>(table.entity(page, slot))));
                value->Mask |= 1u << slot;
                bytes += NativeExtra(copy);
            }
            auto *header = static_cast<NativePage *>(value.release());
            return store::Blob{reinterpret_cast<std::byte *>(header), sizeof(ComponentPage<C>), [](void *p) { delete static_cast<ComponentPage<C> *>(static_cast<NativePage *>(p)); }, bytes};
        };
        entry.PageValue = [](const store::Blob &value, uint32_t slot) -> const void * {
            return &static_cast<const ComponentPage<C> *>(reinterpret_cast<const NativePage *>(value.Data))->Values[slot].Value;
        };
        entry.MovePageValue = [](state::Scene &r, state::Entity e, store::Blob &value, uint32_t slot) {
            auto &native = static_cast<ComponentPage<C> *>(reinterpret_cast<NativePage *>(value.Data))->Values[slot].Value;
            PrepareNative(native);
            r.emplace_or_replace<C>(e, std::move(native));
        };
        entry.Equal = MakeComparator<C, Persistent>();
        tables.Snapshots[id] = entry;
    }
}
template<typename... Cs>
void Persistent(Tables &tables) { (Add<Cs, true>(tables), ...); }
template<typename... Cs>
void Derived(Tables &tables) { (Add<Cs, false>(tables), ...); }

void RegisterAudio(Tables &);
void RegisterPhysics(Tables &);
void RegisterArmature(Tables &);
void RegisterMesh(Tables &);
void RegisterViewport(Tables &);
void RegisterAssets(Tables &);
void RegisterScene(Tables &);
} // namespace snapshot::detail
