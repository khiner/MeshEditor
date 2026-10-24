#pragma once
#include "numeric/Serialize.h"

#include "viewport/ViewCameraOps.h"

// Excludes inactive variant storage and transient animation state from the serialized representation.
constexpr auto serialize(auto &archive, const ViewCamera &v) { return archive(v.Data, v.Target, v.Distance, v.Orientation); }
constexpr auto serialize(auto &archive, ViewCamera &v) { return archive(v.Data, v.Target, v.Distance, v.Orientation); }

// The look-through camera encodes beside the active view, and a saved view as a presence flag and its camera, read into a copy of the active view.
template<typename S>
    requires std::same_as<std::remove_const_t<S>, ViewCameraState>
constexpr zpp::bits::errc serialize(auto &archive, S &state) {
    bool saved = state.LookThroughSaved.has_value();
    if (const auto result = archive(state.Active, state.LookThrough, saved); zpp::bits::failure(result)) return result;
    if constexpr (std::remove_reference_t<decltype(archive)>::kind() == zpp::bits::kind::in) {
        if (saved) state.LookThroughSaved.emplace(state.Active);
        else state.LookThroughSaved.reset();
    }
    return saved ? archive(*state.LookThroughSaved) : zpp::bits::errc{};
}
