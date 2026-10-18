#pragma once

#include "state/Entity.h"
#include "state/Slot.h"

#include <cassert>
#include <vector>

namespace state {
struct PageMask {
    uint32_t Page{}, Mask{};
};

// Paged sparse set of raw values.
// Each page contains its slot owners and then the values.
// Small pages keep unrelated sparse components inexpensive and addresses stable.
struct Table {
    static constexpr uint32_t PageCount = 32;
    struct Page {
        uint32_t Mask{};
        std::array<Entity, PageCount> Owners;
    };
    uint32_t Size{}, Align{alignof(Page)}, ValueOffset{sizeof(Page)}, Count{};
    void (*Destroy)(void *){};
    std::string_view Name; // The bound value type's name.
    std::vector<uint64_t> Occupied; // One bit per allocated page.
    std::vector<Page *> Pages;
    Table() = default;
    Table(const Table &) = delete;
    Table &operator=(const Table &) = delete;
    ~Table() { clear(); }
    // Describes the value type.
    // An unbound table contains no value bytes.
    // A table binds to one type, so two types declaring the same slot fail here.
    template<typename C> void Bind() {
        Name = TypeName<C>();
        Size = sizeof(C);
        Align = uint32_t(std::max(alignof(C), alignof(Page)));
        ValueOffset = (uint32_t(sizeof(Page)) + Align - 1) / Align * Align;
        if constexpr (!std::is_trivially_destructible_v<C>) Destroy = [](void *p) { std::destroy_at(static_cast<C *>(p)); };
    }
    uint32_t pages() const { return uint32_t(Pages.size()); }
    uint32_t mask(uint32_t page) const { return page < Pages.size() && Pages[page] ? Pages[page]->Mask : 0; }
    uint64_t occupied(uint32_t word) const { return word < Occupied.size() ? Occupied[word] : 0; }
    Entity entity(uint32_t page, uint32_t slot) const { return Pages[page]->Owners[slot]; }
    Entity entity_at(uint32_t index) const {
        const auto p = index / PageCount;
        return p < Pages.size() && Pages[p] ? Pages[p]->Owners[index % PageCount] : Null;
    }
    bool contains(Entity e) const { return e != Null && entity_at(Index(e)) == e; }
    std::byte *slot(Entity e, size_t stride) const {
        const auto index = Index(e), p = index / PageCount, i = index % PageCount;
        auto *page = e != Null && p < Pages.size() ? Pages[p] : nullptr;
        return page && page->Owners[i] == e ? reinterpret_cast<std::byte *>(page) + ValueOffset + i * stride : nullptr;
    }
    void *value(Entity e) const { return slot(e, Size); }
    template<typename C> C *at(Entity e) const {
        auto *p = slot(e, sizeof(C)); // NOLINT(misc-const-correctness)
        return p ? std::launder(reinterpret_cast<C *>(p)) : nullptr;
    }
    size_t size() const { return Count; }
    bool empty() const { return Count == 0; }
    // Links the slot and returns its uninitialized value storage.
    void *insert(Entity);
    // Destroys the value and unlinks the slot.
    void erase(Entity);
    void erase(uint32_t page, uint32_t mask);
    void clear();

private:
    void Release(uint32_t page);
};
} // namespace state
