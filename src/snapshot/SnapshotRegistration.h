#pragma once
#include "PathSerialize.h"
#include "numeric/Serialize.h"
#include "snapshot/SnapshotRoles.h"
#include <cstring>
#include <entt/entity/registry.hpp>
#include <stdexcept>

namespace snapshot::detail {
using Comparator = bool (*)(const void *, const void *);
struct Tables {
    std::unordered_map<entt::id_type, SnapshotEntry> Snapshots;
    std::unordered_map<entt::id_type, Comparator> Comparators;
};
// Non-default-constructible serialized types specialize this emplacer.
template<typename C>
inline constexpr void (*CustomEmplace)(entt::registry &, entt::entity, std::span<const std::byte>) = nullptr;

// Bone transforms are derived from RestLocal and ArmaturePose.
template<typename C>
inline constexpr bool (*SkipEntityFor)(const entt::registry &, entt::entity) = nullptr;

// Aligned storage supports non-default-constructible implicit-lifetime types.
template<typename C>
void EmplaceTrivial(entt::registry &r, entt::entity e, std::span<const std::byte> bytes) {
    if constexpr (std::is_empty_v<C>) {
        r.emplace_or_replace<C>(e);
    } else {
        alignas(C) std::byte storage[sizeof(C)];
        std::memcpy(storage, bytes.data(), sizeof(C));
        r.emplace_or_replace<C>(e, *std::launder(reinterpret_cast<const C *>(storage)));
    }
}

template<typename C>
void SerializeThunk(const void *component, std::vector<std::byte> &out) {
    thread_local std::vector<std::byte> buffer;
    buffer.clear();
    zpp::bits::out archive{buffer};
    // zpp aggregate reflection mis-encodes large const aggregates, while the output archive only reads this reference.
    if (zpp::bits::failure(archive(const_cast<C &>(*static_cast<const C *>(component))))) return;
    out.insert(out.end(), buffer.begin(), buffer.begin() + archive.position());
}

template<typename C>
void EmplaceSerialized(entt::registry &r, entt::entity e, std::span<const std::byte> bytes) {
    if constexpr (CustomEmplace<C> != nullptr) CustomEmplace<C>(r, e, bytes);
    else {
        C value;
        if (zpp::bits::failure(zpp::bits::in{bytes}(value))) return;
        r.emplace_or_replace<C>(e, std::move(value));
    }
}

template<typename C> inline constexpr bool ForceFieldwise = false;
template<typename C, bool Persistent>
inline constexpr bool NeedsFieldwise = CustomEmplace<C> != nullptr || ForceFieldwise<C> || (Persistent && !std::is_trivially_copyable_v<C>);
// Padding-only cases require explicit ForceFieldwise specializations because they cannot be detected statically.
template<typename> inline constexpr bool IsVariantOrOptional = false;
template<typename... Ts> inline constexpr bool IsVariantOrOptional<std::variant<Ts...>> = true;
template<typename T> inline constexpr bool IsVariantOrOptional<std::optional<T>> = true;

template<typename C>
consteval bool HoldsVariantOrOptional() {
    if constexpr (IsVariantOrOptional<C>) return true;
    else if constexpr (std::is_trivially_copyable_v<C> && std::is_aggregate_v<C>) // non-aggregates (e.g. ViewCamera) aren't reflectable and use CustomEmplace
        return zpp::bits::visit_members_types<C>([]<typename... Ms>() { return (IsVariantOrOptional<std::remove_cvref_t<Ms>> || ...); });
    else return false;
}
template<typename C, bool Persistent>
bool ValuesEqual(const void *a, const void *b) {
    if constexpr (std::is_empty_v<C>) {
        return true;
    } else if constexpr (NeedsFieldwise<C, Persistent>) {
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
    if constexpr (std::is_empty_v<C> || NeedsFieldwise<C, Persistent> || std::is_trivially_copyable_v<C>) return &ValuesEqual<C, Persistent>;
    else return nullptr;
}

// Selects Tag, Bytes, or Serialized encoding from the component traits and overrides.
template<typename C>
snapshot::SnapshotEntry MakeEntry() {
    using snapshot::Encoding;
    if constexpr (std::is_empty_v<C>) return {Encoding::Tag, 0, nullptr, &EmplaceTrivial<C>, SkipEntityFor<C>};
    else if constexpr (NeedsFieldwise<C, true>) return {Encoding::Serialized, 0, &SerializeThunk<C>, &EmplaceSerialized<C>, SkipEntityFor<C>};
    else return {Encoding::Bytes, sizeof(C), nullptr, &EmplaceTrivial<C>, SkipEntityFor<C>};
}

template<typename C, bool Persistent>
void Add(Tables &tables) {
    const auto id = entt::type_hash<C>::value();
    if (!tables.Comparators.emplace(id, MakeComparator<C, Persistent>()).second) throw std::logic_error("Duplicate snapshot component classification");
    if constexpr (Persistent) {
        static_assert(!HoldsVariantOrOptional<C>() || NeedsFieldwise<C, true>, "A persistent variant/optional needs field-wise serialization");
        tables.Snapshots.emplace(id, MakeEntry<C>());
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
