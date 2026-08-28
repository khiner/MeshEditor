#ifndef CONNECTIVITYREAD_MSL
#define CONNECTIVITYREAD_MSL

inline uint ConnectivityWordCount(uint halfedge_count) { return (halfedge_count + 31u) / 32u; }

inline device const uint *ConnectivityOpposites(device const uint *connectivity, uint vertex_count) { return connectivity + vertex_count; }

inline device const uint *ConnectivityFaceStarts(
    device const uint *connectivity, uint vertex_count, uint halfedge_count
) { return ConnectivityOpposites(connectivity, vertex_count) + halfedge_count + 3u * ConnectivityWordCount(halfedge_count); }

inline uint2 ConnectivityFaceHalfedges(
    device const uint *connectivity, uint vertex_count, uint halfedge_count, uint face_count, bool face_starts, uint face
) {
    if (!face_starts) return uint2(face * 3u, metal::min(face * 3u + 3u, halfedge_count));
    device const uint *starts = ConnectivityFaceStarts(connectivity, vertex_count, halfedge_count);
    return uint2(starts[face], face + 1u < face_count ? starts[face + 1u] : halfedge_count);
}

inline uint ConnectivityHalfedgeFace(
    device const uint *connectivity, uint vertex_count, uint halfedge_count, uint face_count, bool face_starts, uint halfedge
) {
    if (!face_starts) return halfedge / 3u;
    device const uint *starts = ConnectivityFaceStarts(connectivity, vertex_count, halfedge_count);
    uint lo = 0u, hi = face_count;
    while (lo + 1u < hi) {
        const uint mid = (lo + hi) >> 1u;
        if (starts[mid] <= halfedge) lo = mid;
        else hi = mid;
    }
    return lo;
}

#endif
