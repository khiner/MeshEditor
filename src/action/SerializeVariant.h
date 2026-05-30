#pragma once

#include <cstdint>
#include <variant>

// Serialize a variant as [uint32 index][active alternative]
namespace action {
template<typename Archive, typename Variant>
auto SerializeVariant(Archive &&archive, const Variant &v) {
    if (auto errc = archive(uint32_t(v.index())); failure(errc)) return errc;
    return std::visit([&](const auto &alt) { return archive(alt); }, v);
}

template<typename Archive, typename Variant>
auto DeserializeVariant(Archive &&archive, Variant &v) {
    uint32_t tag;
    auto errc = archive(tag);
    if (failure(errc)) return errc;

    // Construct the alternative named by `tag`, then deserialize into it.
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((Is == tag ? (void)(v.template emplace<Is>(), errc = archive(std::get<Is>(v))) : void()), ...);
    }(std::make_index_sequence<std::variant_size_v<Variant>>{});
    return errc;
}
} // namespace action
