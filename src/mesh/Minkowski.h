#pragma once
#include <cstdint>
#include <optional>

namespace state {
struct Registry;
}
struct Mesh;
enum class MinkowskiOperation : uint8_t { Sum,
                                          Difference };
// A new record holding the Minkowski result of two closed surfaces, or of two regions on parallel planes with the second's through the origin.
// The second mesh is in the first's space.
std::optional<uint32_t> MinkowskiRecord(state::Registry &, const Mesh &first, const Mesh &second, MinkowskiOperation);
