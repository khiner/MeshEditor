#include "mesh/ElementArena.h"
#include "RunSuites.h"
#include "metal/MetalCpp.h"

using namespace boost::ut;

int main() {
    "an insertion fills an interior hole before growing the table"_test = [] {
        const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        mtl::Context ctx;
        mtl::BindlessSet slots{ctx};
        mtl::BufferContext buffers{ctx, slots};
        ElementArena<uint32_t> arena{buffers, SlotType::Buffer};
        auto set = arena.Create(3u * MeshElementBlockSize);
        std::vector<uint32_t> middle(MeshElementBlockSize);
        std::iota(middle.begin(), middle.end(), MeshElementBlockSize);
        arena.Erase(set, middle);
        expect(arena.Set(set).BlockCount == 3u);
        expect(arena.Entries(set)[1] == InvalidOffset);
        const auto capacity = arena.Set(set).Capacity;

        const auto inserted = arena.Insert(set, MeshElementBlockSize, nullptr).Handles;
        expect(inserted.First == MeshElementBlockSize);
        expect(arena.Set(set).BlockCount == 3u);
        expect(arena.Set(set).Capacity == capacity);
        expect(arena.Entries(set)[1] != InvalidOffset);
        expect(arena.Count(set) == 3u * MeshElementBlockSize);
        expect(std::ranges::all_of(middle, [&](uint32_t h) { return arena.Contains(set, ElementHandle{h}); }));

        // A run longer than every hole appends.
        arena.Erase(set, middle);
        const auto appended = arena.Insert(set, 2u * MeshElementBlockSize, nullptr).Handles;
        expect(appended.First == 3u * MeshElementBlockSize);
        expect(arena.Set(set).BlockCount == 5u);
        expect(arena.Entries(set)[1] == InvalidOffset);
    };
    return RunSuites();
}
