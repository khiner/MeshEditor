#pragma once
#include "gpu/MeshElementBlock.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <iterator>
#include <span>
#include <stdexcept>
#include <utility>

// Borrow a canonical domain without packing its payload for CPU consumers.
// Iteration follows live bits.
// A packed soup or a dense set resolves an ordinal by offset.
// A sparse set resolves an ordinal through its ascending block list and inclusive live-count prefix.
template<typename T> struct ElementView {
    std::span<const T> Values;
    std::span<const MeshElementBlock> Blocks;
    std::span<const uint32_t> Order, Prefix;
    uint32_t Base{}, Count{};
    ElementView() = default;
    ElementView(std::span<const T> values) : Values(values), Count(uint32_t(values.size())) {}
    ElementView(std::span<const T> values, uint32_t base, uint32_t count) : Values(values), Base(base), Count(count) {}
    ElementView(std::span<const T> values, std::span<const MeshElementBlock> blocks, std::span<const uint32_t> order, std::span<const uint32_t> prefix, uint32_t count)
        : Values(values), Blocks(blocks), Order(order), Prefix(prefix), Count(count) {}
    uint32_t size() const { return Count; }
    bool empty() const { return !Count; }
    uint32_t Handle(uint32_t ordinal) const {
        assert(ordinal < Count);
        if (Order.empty()) return Base + ordinal;
        const auto at = uint32_t(std::ranges::upper_bound(Prefix, ordinal) - Prefix.begin());
        if (at >= Order.size()) throw std::logic_error("Live element count disagrees with membership.");
        ordinal -= at ? Prefix[at - 1u] : 0u;
        const auto &live = Blocks[Order[at]].Live;
        for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) {
            auto bits = live[w];
            const uint32_t n = std::popcount(bits);
            if (ordinal >= n) {
                ordinal -= n;
                continue;
            }
            while (ordinal--) bits &= bits - 1u;
            return Order[at] * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits));
        }
        throw std::logic_error("Live element count disagrees with membership.");
    }
    const T &operator[](uint32_t ordinal) const { return Values[Handle(ordinal)]; }
    struct Iterator {
        const ElementView *View;
        uint32_t At{}, Word{}, Bits{}, Ordinal{};
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        void Seek() {
            while (!Bits && At < View->Order.size()) {
                if (Word == MeshElementBlockWords) {
                    ++At;
                    Word = 0u;
                }
                if (At < View->Order.size()) Bits = View->Blocks[View->Order[At]].Live[Word++];
            }
        }
        uint32_t Handle() const {
            return View->Order.empty() ? View->Base + Ordinal :
                                         View->Order[At] * MeshElementBlockSize + (Word - 1u) * 32u + uint32_t(std::countr_zero(Bits));
        }
        const T &operator*() const {
            return View->Values[Handle()];
        }
        Iterator &operator++() {
            ++Ordinal;
            if (!View->Order.empty()) {
                Bits &= Bits - 1u;
                Seek();
            }
            return *this;
        }
        Iterator operator++(int) {
            auto before = *this;
            ++*this;
            return before;
        }
        bool operator==(const Iterator &other) const { return Ordinal == other.Ordinal; }
    };
    Iterator begin() const {
        Iterator i{this};
        if (!Order.empty()) i.Seek();
        return i;
    }
    Iterator end() const { return {this, 0u, 0u, 0u, Count}; }
};
