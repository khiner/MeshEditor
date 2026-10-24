#pragma once

#include "state/Slot.h"

#include <cassert>

namespace state {
struct Registry;
// The reset hook of a resource type that declares `void Reset(T &, Registry &)`, and null for one that declares none.
template<typename T> inline constexpr void (*ResetOf)(void *, Registry &) = nullptr;
template<typename T>
    requires requires(T &t, Registry &r) { Reset(t, r); }
inline constexpr void (*ResetOf<T>)(void *, Registry &) = [](void *p, Registry &r) { Reset(*static_cast<T *>(p), r); };

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
        // Creates the value when the slot is empty and returns it.
        template<typename T, typename... A> T &Emplace(A &&...args) {
            if (!Value) {
                Value = static_cast<void *>(new T(std::forward<A>(args)...));
                Destroy = [](void *p) { delete static_cast<T *>(p); };
                Reset = ResetOf<T>;
            }
            return *static_cast<T *>(Value);
        }
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
    template<typename T, typename... A> T &emplace(A &&...args) { return Slots[Type<T>()].template Emplace<T>(std::forward<A>(args)...); }
    template<typename T> void erase() { Slots[Type<T>()].Release(); }
    void Clear() {
        for (auto &slot : Slots) slot.Release();
    }
};
} // namespace state
