#ifndef CONNECTIVITYREAD_MSL
#define CONNECTIVITYREAD_MSL

inline uint ConnectivityWordCount(uint halfedge_count) { return (halfedge_count + 31u) / 32u; }

// The halfedge before `h` in its triangle's loop, whose corner is `h`'s from-vertex.
inline uint ConnectivityPrevious(uint h) {
    const uint first = h - h % 3u;
    return first + (h - first + 2u) % 3u;
}

// A mesh's connectivity run: outgoing halfedges, opposites, each halfedge's edge, an n-gon mesh's face starts, then each edge's first halfedge.
struct ConnectivityView {
    device const uint *Run;
    uint VertexCount, HalfedgeCount, FaceCount;
    bool FaceStarts;

    device const uint *Opposites() const { return Run + VertexCount; }
    device const uint *HalfedgeToEdge() const { return Opposites() + HalfedgeCount; }
    device const uint *Starts() const { return HalfedgeToEdge() + HalfedgeCount; }
    device const uint *Edges() const { return Starts() + (FaceStarts ? FaceCount : 0u); }

    uint Opposite(uint h) const { return Opposites()[h]; }
    uint Edge(uint h) const { return HalfedgeToEdge()[h]; }
    uint EdgeHalfedge(uint e) const { return Edges()[e]; }
    bool EdgeFirst(uint h) const { return EdgeHalfedge(Edge(h)) == h; }

    uint2 FaceHalfedges(uint f) const {
        if (!FaceStarts) return uint2(f * 3u, metal::min(f * 3u + 3u, HalfedgeCount));
        return uint2(Starts()[f], f + 1u < FaceCount ? Starts()[f + 1u] : HalfedgeCount);
    }
    uint HalfedgeFace(uint h) const {
        if (!FaceStarts) return h / 3u;
        uint lo = 0u, hi = FaceCount;
        while (lo + 1u < hi) {
            const uint mid = (lo + hi) >> 1u;
            if (Starts()[mid] <= h) lo = mid;
            else hi = mid;
        }
        return lo;
    }
    uint Previous(uint h) const {
        if (!FaceStarts) return ConnectivityPrevious(h);
        const uint2 range = FaceHalfedges(HalfedgeFace(h));
        return h == range.x ? range.y - 1u : h - 1u;
    }
};

#endif
