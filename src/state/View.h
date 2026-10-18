#pragma once

#include "state/Table.h"

#include <bit>
#include <ranges>

namespace state {
template<typename... C> struct ExcludeList {};
template<typename... C> inline constexpr ExcludeList<C...> Exclude{};
template<typename R, typename Ex, typename... C> struct View;

// Entities holding every listed component and none of the excluded ones, joined page by page on the table masks.
template<typename R, typename... X, typename... C>
struct View<R, ExcludeList<X...>, C...> : std::ranges::view_interface<View<R, ExcludeList<X...>, C...>> {
    using Driver = std::tuple_element_t<0, std::tuple<C...>>;
    // Walks the live slots in entity index order.
    // Slots removed ahead of the cursor are skipped.
    struct Iterator {
        using value_type = Entity;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        static constexpr uint32_t End = UINT32_MAX;
        View Source{};
        uint32_t Page{}, Bits{};
        // Move to the first page at or after Page with live slots.
        void Seek() {
            while (Page < Source.pages()) {
                const auto word = Source.occupied(Page / 64) & (~uint64_t{0} << Page % 64);
                if (!word) {
                    Page = (Page / 64 + 1) * 64;
                    continue;
                }
                Page = Page / 64 * 64 + std::countr_zero(word);
                if ((Bits = Source.mask(Page))) return;
                ++Page;
            }
            Page = End;
            Bits = 0;
        }
        Entity operator*() const { return Source.entity(Page, std::countr_zero(Bits)); }
        Iterator &operator++() {
            Bits = (Bits & (Bits - 1)) & Source.mask(Page);
            if (!Bits) {
                ++Page;
                Seek();
            }
            return *this;
        }
        Iterator operator++(int) {
            auto old = *this;
            ++*this;
            return old;
        }
        bool operator==(const Iterator &) const = default;
    };
    R *Owner{};
    View() = default;
    explicit View(R &r) : Owner(&r) {}
    bool operator==(const View &o) const { return Owner == o.Owner; }
    template<typename T> const Table &table() const { return std::as_const(*Owner).template storage<T>(); }
    uint32_t pages() const { return std::min({table<C>().pages()...}); }
    uint64_t occupied(uint32_t word) const { return (table<C>().occupied(word) & ...); }
    uint32_t mask(uint32_t page) const {
        return (table<C>().mask(page) & ...) & ~(0u | ... | table<X>().mask(page));
    }
    Entity entity(uint32_t page, uint32_t slot) const { return table<Driver>().entity(page, slot); }
    bool contains(Entity e) const {
        return table<Driver>().contains(e) && (mask(Index(e) / Table::PageCount) >> (Index(e) % Table::PageCount) & 1);
    }
    Iterator begin() const {
        Iterator it{*this};
        it.Seek();
        return it;
    }
    Iterator end() const { return {*this, Iterator::End, 0}; }
    // A single component without exclusions answers from its table.
    static constexpr bool Direct = sizeof...(C) == 1 && sizeof...(X) == 0;
    bool empty() const {
        if constexpr (Direct) return table<Driver>().empty();
        else return begin() == end();
    }
    size_t size() const {
        if constexpr (Direct) return table<Driver>().size();
        else {
            size_t n = 0;
            for (auto it = begin(); it != end(); it.Page++, it.Seek()) n += std::popcount(it.Bits);
            return n;
        }
    }
    template<typename T> static auto Value(R &owner, Entity e) {
        if constexpr (std::is_empty_v<T> || std::same_as<std::remove_const_t<T>, Entity>) return std::tuple<>{};
        else if constexpr (std::is_const_v<T> || std::is_const_v<R>) return std::tie(owner.template get<T>(e));
        else return std::tie(owner.template edit<T>(e));
    }
    auto each() const {
        return View{*this} | std::views::transform([owner = Owner](Entity e) { return std::tuple_cat(std::tuple{e}, Value<C>(*owner, e)...); });
    }
};

} // namespace state

namespace std::ranges {
template<typename R, typename Ex, typename... C> inline constexpr bool enable_borrowed_range<state::View<R, Ex, C...>> = true;
} // namespace std::ranges
