#pragma once

#include <cstdint>

// Bone IDs are stable and never reused.
using BoneId = uint32_t;
inline constexpr BoneId InvalidBoneId{0};
