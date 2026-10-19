#pragma once

#include "state/Slot.h"

#include <cassert>

namespace state {
struct Registry;
// A resource holding scene state declares `void Reset(T &, Registry &)` beside its type, which returns it to its post-init state.
// Reset touches only its own domain and engine-lifetime resources, so resources reset in any order.
// An engine-lifetime resource, such as the bindless set whose allocation is order-independent, declares none.
// Fixed resource slots are selected at compile time.
// Definitions stay in the owning TU.
struct Resources {
    struct Slot {
        void *Value{};
        void (*Destroy)(void *){};
        void (*Reset)(void *, Registry &){};
        ~Slot() { Release(); }
        void Release() {
            if (Value) Destroy(std::exchange(Value, nullptr));
        }
    };
    std::array<Slot, SchemaSize> Slots;
    template<typename T> T *find() const { return static_cast<T *>(Slots[Type<T>()].Value); }
    template<typename T> T &get() const {
        auto *p = find<T>();
        assert(p);
        return *p;
    }
    template<typename T, typename... A> T &emplace(A &&...args) {
        auto &s = Slots[Type<T>()];
        if (!s.Value) {
            s.Value = static_cast<void *>(new T(std::forward<A>(args)...));
            s.Destroy = [](void *p) { delete static_cast<T *>(p); };
            if constexpr (requires(T &t, Registry &r) { Reset(t, r); }) s.Reset = [](void *p, Registry &r) { Reset(*static_cast<T *>(p), r); };
        }
        return *static_cast<T *>(s.Value);
    }
    template<typename T> void erase() { Slots[Type<T>()].Release(); }
    void Clear() {
        for (auto &slot : Slots) slot.Release();
    }
};
} // namespace state
