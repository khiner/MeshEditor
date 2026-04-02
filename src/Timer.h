#pragma once

#include <chrono>
#include <print>
#include <string_view>
#include <vector>

struct Timer {
    static inline bool Enabled{true};

    Timer(std::string_view name) : Name{name}, InitialDepth{Depth++}, Seq{uint32_t(Pending.size())} {
        Pending.emplace_back(InitialDepth, Name, 0.f);
    }
    ~Timer() {
        --Depth;
        Pending[Seq].Ms = std::chrono::duration<float, std::milli>{std::chrono::steady_clock::now() - Start}.count();
        if (InitialDepth == 0) {
            if (Enabled) {
                for (const auto &e : Pending) std::println("{:>{}}{}: ms={:.3f}", "", e.Depth * 2, e.Name, e.Ms);
            }
            Pending.clear();
        }
    }

private:
    std::string_view Name;
    std::chrono::steady_clock::time_point Start{std::chrono::steady_clock::now()};
    uint32_t InitialDepth, Seq;

    struct Entry {
        uint32_t Depth;
        std::string_view Name;
        float Ms;
    };
    static inline thread_local uint32_t Depth = 0;
    static inline thread_local std::vector<Entry> Pending;
};
