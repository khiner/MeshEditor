#pragma once
#include "gpu/Types.h"
GPU_CONSTANT uint32_t RayEnvironmentSize = 64u;
GPU_CONSTANT uint32_t RayEnvironmentRows = 6u * RayEnvironmentSize;
GPU_CONSTANT uint32_t RayEnvironmentStride = RayEnvironmentSize + 1u;
GPU_CONSTANT uint32_t RayEnvironmentMarginal = RayEnvironmentRows * RayEnvironmentStride;
GPU_CONSTANT uint32_t RayEnvironmentCount = RayEnvironmentMarginal + RayEnvironmentRows + 1u;
