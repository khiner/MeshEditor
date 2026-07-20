// Workgroup-shared AABB fold over the 256 lanes.
// Each lane seeds its slot with its partial min/max, the empty constants when it covers nothing.
// The fold leaves the workgroup's bounds in SharedMin[0]/SharedMax[0].
// An empty workgroup leaves Min > Max, the AABB empty state.
const vec3 AabbEmptyMin = vec3(3.402823466e38);
const vec3 AabbEmptyMax = vec3(-3.402823466e38);

shared vec3 SharedMin[256];
shared vec3 SharedMax[256];

void FoldSharedAabb(uint tid, vec3 lo, vec3 hi) {
    SharedMin[tid] = lo;
    SharedMax[tid] = hi;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            SharedMin[tid] = min(SharedMin[tid], SharedMin[tid + stride]);
            SharedMax[tid] = max(SharedMax[tid], SharedMax[tid + stride]);
        }
        barrier();
    }
}
