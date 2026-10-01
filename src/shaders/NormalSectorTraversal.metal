#ifndef NORMAL_SECTOR_TRAVERSAL_MSL
#define NORMAL_SECTOR_TRAVERSAL_MSL

#include "ConnectivityRead.metal"

// Classification and normal derivation must traverse exactly the same smooth
// component, in the same order. The caller handles the root before this walk.
struct NormalSectorTraversal {
    ConnectivityView Conn;
    device const uchar *FaceSharpness;
    device const uchar *EdgeSharpness;
    bool Flat(uint h) const { return FaceSharpness[Conn.HalfedgeFace(h)] != 0u; }
    bool Sharp(uint h) const { return EdgeSharpness[Conn.Edge(h)] != 0u; }

    template<typename Visit>
    void VisitOthers(uint root, uint limit, Visit visit) const {
        const uint face = Conn.HalfedgeFace(root);
        bool closed = false;
        uint h = root;
        for (uint i = 0u; i < limit; ++i) {
            const uint out = Conn.Next(h);
            if (Sharp(out)) break;
            const uint opposite = Conn.Opposite(out);
            if (opposite == InvalidOffset) break;
            const uint next_face = Conn.HalfedgeFace(opposite);
            if (next_face == InvalidOffset) break;
            if (next_face == face) { closed = true; break; }
            if (Flat(opposite)) break;
            visit(opposite);
            h = opposite;
        }
        if (!closed) {
            h = root;
            for (uint i = 0u; i < limit; ++i) {
                if (Sharp(h)) break;
                const uint opposite = Conn.Opposite(h);
                if (opposite == InvalidOffset) break;
                const uint next_face = Conn.HalfedgeFace(opposite);
                if (next_face == InvalidOffset || next_face == face || Flat(opposite)) break;
                h = Conn.Previous(opposite);
                visit(h);
            }
        }
    }
};

#endif
