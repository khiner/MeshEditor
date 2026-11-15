#pragma once

#include <algorithm>
#include <chrono>
#include <vector>

// The wall time `run` takes, in milliseconds.
double Milliseconds(auto &&run) {
    const auto start = std::chrono::steady_clock::now();
    run();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
inline double Median(std::vector<double> values) {
    std::ranges::sort(values);
    return values[values.size() / 2];
}
