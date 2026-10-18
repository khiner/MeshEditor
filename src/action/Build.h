#pragma once

#include "action/Updatable.h"

namespace action {
// Ms... walks from the component down to the leaf field being written.
template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateOf(Target target, detail::last_field<Ms...> v) {
    using C = detail::first_class<Ms...>;
    using F = detail::last_field<Ms...>;
    static_assert(Updatable<C>, "Add the component to action::UpdatableComponents");
    static_assert(std::is_trivially_copyable_v<F>, "Update<T> is for trivially-copyable fields only");
    return {std::move(target), state::Key<C>(), detail::FieldOffset<Ms...>(), std::move(v)};
}

template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateActive(detail::last_field<Ms...> v) { return UpdateOf<Ms...>(OnActive{}, std::move(v)); }

// Update a field within the active PrimitiveShape alternative.
template<typename C, typename F> Update<F> UpdateOfShape(Target target, F C::*member, F value) {
    static_assert(std::is_trivially_copyable_v<F>, "Update<T> is for trivially-copyable fields only");
    return {std::move(target), state::Key<PrimitiveShape>(), uint16_t(detail::MemPtrOffset(member)), std::move(value)};
}
} // namespace action
