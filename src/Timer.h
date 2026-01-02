#pragma once

#include <chrono>
#include <print>
#include <string_view>

struct Timer {
    std::string_view Name;
    std::chrono::steady_clock::time_point Start{std::chrono::steady_clock::now()};

    Timer(const std::string_view name) : Name{name} {}
    ~Timer() {
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - Start).count();
        std::println("{}: ms={:.3f}", Name, ms);
    }
};
