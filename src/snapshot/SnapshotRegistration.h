#pragma once
#include "PathSerialize.h"
#include "numeric/Serialize.h"
#include "snapshot/NativeSize.h"
#include "snapshot/SnapshotRoles.h"
#include "state/Scene.h"
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

// Aligned storage supports non-default-constructible implicit-lifetime types.
template<typename C>
void EmplaceTrivial(state::Scene &r, state::Entity e, std::span<const std::byte> bytes) {
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
    const auto start = out.size();
    // Grow only the appended record; vector capacity handles allocation growth.
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
    if constexpr (std::is_empty_v<C>) return {Encoding::Tag, 0, nullptr, &EmplaceTrivial<C>};
    else if constexpr (NeedsFieldwise<C, true>) return {Encoding::Serialized, 0, &SerializeThunk<C>, &EmplaceSerialized<C>};
    else return {Encoding::Bytes, sizeof(C), nullptr, &EmplaceTrivial<C>};
}

template<typename C, bool Persistent>
void Add(Tables &tables) {
    const auto id = state::Type<C>();
    if (std::exchange(tables.Classified[id], true)) throw std::logic_error("Duplicate snapshot component classification");
    tables.Comparators[id] = MakeComparator<C, Persistent>();
    if constexpr (Persistent) {
        static_assert(!HoldsVariantOrOptional<C>() || NeedsFieldwise<C, true>, "A persistent variant/optional needs field-wise serialization");
        auto entry = MakeEntry<C>();
        entry.Name = state::TypeName<C>();
        entry.Copy = [](const void *p) {
            auto value = std::make_unique<C>(CopyNative(*static_cast<const C *>(p)));
            const auto bytes = sizeof(C) + NativeExtra(*value);
            return store::Blob{reinterpret_cast<std::byte *>(value.release()), sizeof(C), [](void *v) { delete static_cast<C *>(v); }, bytes};
        };
        entry.Move = [](state::Scene &r, state::Entity e, store::Blob value) {
            auto &native = *reinterpret_cast<C *>(value.Data);
            PrepareNative(native);
            r.emplace_or_replace<C>(e, std::move(native));
            store::FreeBlob(value);
        };
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
