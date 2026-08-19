#pragma once

#include "metal/MetalCpp.h"

#include <string>

namespace mtl {
// Owns the +1 reference returned by metal-cpp `new*` methods.
template<typename T> struct Owned {
    Owned() = default;
    explicit Owned(T *ptr) : Ptr(ptr) {}
    Owned(const Owned &) = delete;
    Owned &operator=(const Owned &) = delete;
    Owned(Owned &&other) noexcept : Ptr(std::exchange(other.Ptr, nullptr)) {}
    Owned &operator=(Owned &&other) noexcept {
        if (this != &other) {
            if (Ptr) Ptr->release();
            Ptr = std::exchange(other.Ptr, nullptr);
        }
        return *this;
    }
    ~Owned() {
        if (Ptr) Ptr->release();
    }

    T *operator*() const { return Ptr; }
    T *operator->() const { return Ptr; }
    explicit operator bool() const { return Ptr != nullptr; }

private:
    T *Ptr{nullptr};
};

NS::String *Str(std::string_view);

// Metal requires anything reached through an argument buffer to be resident, so allocations register
// here once rather than per encoder.
struct Context {
    Context();
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    Context(Context &&) noexcept;
    Context &operator=(Context &&) = delete;

    void AddResident(MTL::Allocation *) const;
    void RemoveResident(MTL::Allocation *) const;
    void CommitResidency() const;

    Owned<MTL::Device> Device;
    Owned<MTL::CommandQueue> Queue;
    Owned<MTL::ResidencySet> Residency;

private:
    mutable bool ResidencyDirty{false};
};
} // namespace mtl
