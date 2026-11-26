#pragma once

// A table holds its job count, then the first tile of each job in ascending order.
// Returns the job of tile `group` and sets `tile` to the tile's index within the job.
inline uint TileJob(device const uint *firsts, uint group, thread uint &tile) {
    uint low = 0u, high = firsts[0];
    while (high - low > 1u) {
        const uint middle = (low + high) / 2u;
        if (firsts[1u + middle] <= group) low = middle;
        else high = middle;
    }
    tile = group - firsts[1u + low];
    return low;
}
