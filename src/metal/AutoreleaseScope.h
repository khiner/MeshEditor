#pragma once

#include <Foundation/NSAutoreleasePool.hpp>

namespace mtl {
// Declare before native locals so they are released before the pool drains.
struct AutoreleaseScope {
    AutoreleaseScope() : Pool(NS::AutoreleasePool::alloc()->init()) {}
    ~AutoreleaseScope() { Pool->release(); }
    AutoreleaseScope(const AutoreleaseScope &) = delete;
    AutoreleaseScope &operator=(const AutoreleaseScope &) = delete;

    template<typename... T> static void Release(T &...owners) {
        const AutoreleaseScope pool;
        ((owners = T{}), ...);
    }

private:
    NS::AutoreleasePool *Pool;
};
} // namespace mtl
