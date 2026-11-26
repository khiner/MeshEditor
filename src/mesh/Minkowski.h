#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace state {
struct Registry;
}
enum class MinkowskiOperation : uint8_t { Sum,
                                          Difference };
// Two records whose Minkowski result is wanted, the second in the first's space.
struct MinkowskiTask {
    uint32_t First, Second;
    MinkowskiOperation Operation;
};
// Each task's new record holding the Minkowski result of two closed surfaces, or of two regions on parallel planes with the second's through the origin.
std::vector<std::optional<uint32_t>> MinkowskiRecords(state::Registry &, std::span<const MinkowskiTask>);
