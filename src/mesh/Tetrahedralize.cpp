// Tetrahedral mesh generator.
// `Tetrahedralize.h` says what this implements.
//
// A tetrahedron (a, b, c, d) is stored with Orient3D(a, b, c, d) < 0.
// The output flips each tet into TetMesh's positive convention.

#include "mesh/Tetrahedralize.h"

#include "numeric/Predicates.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <unordered_map>
#include <vector>

#include <glm/geometric.hpp>

namespace tetra {
namespace {

using u32 = uint32_t;
using u64 = uint64_t;

constexpr int None = -1;

// The point standing for infinity.
// Every hull tetrahedron holds it as its fourth vertex, and no predicate is ever handed it.
// It occupies point slot 0, so an input point i lives at i + 1.
constexpr int DummyPoint = 0;

//=== Handles ===
// A `Triface` is a (tet, ver) pair naming one of the twelve directed edges of a tetrahedron.
// The version's upper two bits are the edge within an oriented face, its lower two the face.
// A `Facet` is the same idea for a triangle, six versions over three edges and two orientations.
// Both pack into a single int for storage: the index shifted past the version.

struct Triface {
    int tet{None};
    int ver{0};
};

struct Facet {
    int sh{None};
    int shver{0};
};

// Lookup tables driving the handle algebra.
// `esymtbl`, the pivots and `snextpivot` are constants; the rest are the products `inittables` computes from them.
constexpr int esymtbl[12]{9, 6, 11, 4, 3, 7, 1, 5, 10, 0, 8, 2};
constexpr int orgpivot[12]{3, 3, 1, 1, 2, 0, 0, 2, 1, 2, 3, 0};
constexpr int destpivot[12]{2, 0, 0, 2, 1, 2, 3, 0, 3, 3, 1, 1};
constexpr int apexpivot[12]{1, 2, 3, 0, 3, 3, 1, 1, 2, 0, 0, 2};
constexpr int oppopivot[12]{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
constexpr int ver2edge[12]{0, 1, 2, 3, 3, 5, 1, 5, 4, 0, 4, 2};
constexpr int edge2ver[6]{0, 1, 2, 3, 8, 5};
// The pair of faces meeting at each of those six edges: [c,d], [a,d], [a,b], [b,c], [b,d], [a,c].
constexpr std::array<std::array<int, 2>, 6> DihedralFaces{{{0, 1}, {1, 2}, {2, 3}, {0, 3}, {2, 0}, {1, 3}}};
constexpr int epivot[12]{4, 5, 2, 11, 4, 5, 2, 11, 4, 5, 2, 11};
constexpr int snextpivot[6]{2, 5, 4, 1, 0, 3};
constexpr int sorgpivot[6]{0, 1, 1, 2, 2, 0};
constexpr int sdestpivot[6]{1, 0, 2, 1, 0, 2};
constexpr int sapexpivot[6]{2, 2, 0, 0, 1, 1};

struct Tables {
    int bondtbl[12][12]{}, fsymtbl[12][12]{}, facepivot2[12][12]{};
    int facepivot1[12]{}, enexttbl[12]{}, eprevtbl[12]{};
    int enextesymtbl[12]{}, eprevesymtbl[12]{}, eorgoppotbl[12]{}, edestoppotbl[12]{};
    int tsbondtbl[12][6]{}, stbondtbl[12][6]{}, tspivottbl[12][6]{}, stpivottbl[12][6]{};

    constexpr Tables() {
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                bondtbl[i][j] = (j & 3) + (((i & 12) + (j & 12)) % 12);
                fsymtbl[i][j] = (j + 12 - (i & 12)) % 12;
            }
        }
        for (int i = 0; i < 12; ++i) facepivot1[i] = esymtbl[i] & 3;
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) facepivot2[i][j] = fsymtbl[esymtbl[i]][j];
        }
        for (int i = 0; i < 12; ++i) {
            enexttbl[i] = (i + 4) % 12;
            eprevtbl[i] = (i + 8) % 12;
        }
        for (int i = 0; i < 12; ++i) {
            enextesymtbl[i] = esymtbl[enexttbl[i]];
            eprevesymtbl[i] = esymtbl[eprevtbl[i]];
        }
        for (int i = 0; i < 12; ++i) {
            eorgoppotbl[i] = eprevtbl[esymtbl[enexttbl[i]]];
            edestoppotbl[i] = enexttbl[esymtbl[eprevtbl[i]]];
        }
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 6; ++j) {
                const int soffset = (j & 1) == 0 ? (6 - ((i & 12) >> 1)) % 6 : (i & 12) >> 1;
                const int toffset = (j & 1) == 0 ? (12 - ((j & 6) << 1)) % 12 : (j & 6) << 1;
                tsbondtbl[i][j] = (j & 1) + (((j & 6) + soffset) % 6);
                stbondtbl[i][j] = (i & 3) + (((i & 12) + toffset) % 12);
                const int poffset = (j & 1) == 0 ? (i & 12) >> 1 : (6 - ((i & 12) >> 1)) % 6;
                const int qoffset = (j & 1) == 0 ? (j & 6) << 1 : (12 - ((j & 6) << 1)) % 12;
                tspivottbl[i][j] = (j & 1) + (((j & 6) + poffset) % 6);
                stpivottbl[i][j] = (i & 3) + (((i & 12) + qoffset) % 12);
            }
        }
    }
};
constexpr Tables Tbl{};

inline int Encode(const Triface &t) { return t.tet < 0 ? None : (t.tet << 4) | t.ver; }
inline int Encode2(int tet, int ver) { return tet < 0 ? None : (tet << 4) | ver; }
inline void Decode(int p, Triface &t) {
    if (p < 0) {
        t.tet = None;
        t.ver = 0;
    } else {
        t.ver = p & 15;
        t.tet = p >> 4;
    }
}
inline int SEncode(const Facet &s) { return s.sh < 0 ? None : (s.sh << 3) | s.shver; }
inline int SEncode2(int sh, int shver) { return sh < 0 ? None : (sh << 3) | shver; }
inline void SDecode(int p, Facet &s) {
    if (p < 0) {
        s.sh = None;
        s.shver = 0;
    } else {
        s.shver = p & 7;
        s.sh = p >> 3;
    }
}

//=== Records ===

// Four neighbours indexed by face, four vertices, the six subsegments on its edges, the four subfaces on its faces, and the packed marker word.
struct Tet {
    int N[4]{None, None, None, None};
    int V[4]{None, None, None, None};
    int Seg[6]{None, None, None, None, None, None};
    int Sub[4]{None, None, None, None};
    int Marker{0};
};

// A shellface, used for both subfaces and subsegments.
// Three neighbouring subfaces and three subsegments, one per edge, three vertices, and the two tetrahedra either side.
struct Shell {
    int S[3]{None, None, None};
    int V[3]{None, None, None};
    int Seg[3]{None, None, None};
    int T[2]{None, None};
    int Mark{0};
    int Flags{0};
    int FacetIndex{0};
    double AreaBound{0};
    // 0 for a subface, 1 for a subsegment.
    // One pool holds both, so the record says which it is.
    int Kind{0};
};

enum VertType {
    UnusedVertex,
    DuplicatedVertex,
    RidgeVertex,
    FacetVertex,
    VolVertex,
    FreeSegVertex,
    FreeFacetVertex,
    FreeVolVertex,
    NRegularVertex,
    DeadVertex
};

struct Point {
    dvec3 Pos{};
    // point2tet, point2sh and point2ppt: an encoded handle each, and the parent this was split from.
    int Tet{None}, Sh{None}, Parent{None};
    // Low byte holds the infect and marktest bits, and the type sits above it.
    int Flags{0};
    double InsRadius{0};
    // The target edge length at this vertex.
    double Mtr{0};
};

// Thrown to abandon the build.
// Code 2 is an internal error, 3 a self-intersecting surface, 4 a feature below the resolvable size, 10 a degenerate point set.
struct TetError {
    int Code;
};

// Triangle-triangle intersection verdicts.
enum InterResult {
    Disjoint,
    Intersect,
    ShareVert,
    ShareEdge,
    ShareFace,
    TouchEdge,
    TouchFace,
    AcrossVert,
    AcrossEdge,
    AcrossFace,
    SelfIntersect
};
enum LocateResult {
    LocUnknown,
    Outside,
    InTetrahedron,
    OnFace,
    OnEdge,
    OnVertex,
    EncVertex,
    EncSegment,
    EncSubface,
    NearVertex,
    NonRegular,
    InStar,
    BadElement,
    NullCavity,
    SharpCorner,
    FencedIn,
    NonCoplanar,
    SelfEncroach
};
// Which of the three faces about the current edge a walk crosses to reach the next tet.
enum WalkMove {
    OrgMove,
    DestMove,
    ApexMove
};

struct InsertFlags {
    int iloc{0};
    int bowywat{0}, lawson{0};
    int splitbdflag{0}, validflag{0}, respectbdflag{0};
    int rejflag{0}, chkencflag{0}, cdtflag{0};
    int assignmeshsize{0};
    int sloc{0}, sbowywat{0};
    int collect_inial_cavity_flag{0};
    int ignore_near_vertex{0};
    int check_insert_radius{0};
    int refineflag{0};
    Triface refinetet{};
    Facet refinesh{};
    int smlenflag{0};
    double smlen{0};
    int parentpt{None};
};

// How a Steiner point placed on the boundary is inserted: Bowyer-Watson in the tets and again in the
// subfaces, flipped back to Delaunay afterwards, and leaving the boundary itself unsplit.
InsertFlags boundary_split_flags(int iloc) {
    return {
        .iloc = iloc,
        .bowywat = 1,
        .lawson = 2,
        .splitbdflag = 0,
        .validflag = 1,
        .respectbdflag = 1,
        .rejflag = 0,
        .chkencflag = 0,
        .sloc = OnEdge,
        .sbowywat = 1,
    };
}

struct FlipConstraints {
    int enqflag{0};
    int chkencflag{0};
    int unflip{0};
    int collectnewtets{0};
    int collectencsegflag{0};
    int noflip_in_surface{0};
    int remove_ndelaunay_edge{0};
    double bak_tetprism_vol{0};
    double tetprism_vol_sum{0};
    int remove_large_angle{0};
    double cosdihed_in{0};
    double cosdihed_out{0};
    double max_asp_out{0};
    int checkflipeligibility{0};
    int seg[2]{None, None};
    int fac[3]{None, None, None};
    int remvert{None};
};

// Options driving vertex smoothing.
struct OptParameters {
    int max_min_volume{0};
    int min_max_aspectratio{0};
    int min_max_dihedangle{0};
    double initval{0}, imprval{0};
    int numofsearchdirs{10};
    double searchstep{0.01};
    int maxiter{-1};
    int smthiter{0};
};

// A queued element, carrying the vertices it had when it was queued so a stale entry can be recognised on pop.
struct BadFace {
    Triface tt{};
    Facet ss{};
    double key{0};
    double cent[6]{};
    int forg{None}, fdest{None}, fapex{None}, foppo{None}, noppo{None};
    int nextitem{None};
};

//=== Geometric predicates and constructions ===
// These take coordinates rather than point handles, because the coplanar triangle-edge test invents a lift point that belongs to no mesh.

constexpr double PI = 3.14159265358979323846264338327950288419716939937510582;

// The determinant is expanded about the z terms.
// The approximate value is returned whenever it clears the first error bound, and the rest goes to the refinement stages.
// Callers read the magnitude as a volume, not only the sign, so the expansion order is part of the result.
// Every sign comes from the error bound, the refinement stages and the exact fallback.
// None comes from a fixed magnitude below which a determinant is treated as noise.
// Such a filter is sized from the input bounding box, and not every point asked about is inside it.
// A point far outside would be decided on a value that can be pure cancellation.
inline double orient3d(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    const double adx = a.x - d.x, ady = a.y - d.y, adz = a.z - d.z;
    const double bdx = b.x - d.x, bdy = b.y - d.y, bdz = b.z - d.z;
    const double cdx = c.x - d.x, cdy = c.y - d.y, cdz = c.z - d.z;

    const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
    const double cdxady = cdx * ady, adxcdy = adx * cdy;
    const double adxbdy = adx * bdy, bdxady = bdx * ady;

    const double det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);

    constexpr double Epsilon = 0x1p-53;
    constexpr double ErrBoundA = (7.0 + 56.0 * Epsilon) * Epsilon;
    const double permanent = (std::abs(bdxcdy) + std::abs(cdxbdy)) * std::abs(adz) +
        (std::abs(cdxady) + std::abs(adxcdy)) * std::abs(bdz) +
        (std::abs(adxbdy) + std::abs(bdxady)) * std::abs(cdz);
    const double errbound = ErrBoundA * permanent;
    if (det > errbound || -det > errbound) return det;

    return geom::Orient3DRefined(a, b, c, d, permanent);
}
inline double insphere(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e) { return geom::InSphere(a, b, c, d, e); }
// Written as a running sum of products, which is the form the results are compared in.
inline double dot(const dvec3 &a, const dvec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double distance(const dvec3 &a, const dvec3 &b) {
    const double x = b.x - a.x, y = b.y - a.y, z = b.z - a.z;
    return std::sqrt(x * x + y * y + z * z);
}
inline dvec3 cross(const dvec3 &a, const dvec3 &b) { return glm::cross(a, b); }

// The normal of [pa, pb, pc].
// With pivot set it picks the two shortest edge vectors by Burdakov's rule, which keeps the cross product accurate on a thin triangle.
// The average edge length comes back through lav.
inline void facenormal(const dvec3 &pa, const dvec3 &pb, const dvec3 &pc, dvec3 &n, int pivot, double *lav) {
    const dvec3 v1 = pb - pa, v2 = pa - pc, v3 = pc - pb;
    const dvec3 *pv1 = &v1, *pv2 = &v2;
    if (pivot > 0) {
        const double l1 = dot(v1, v1), l2 = dot(v2, v2), l3 = dot(v3, v3);
        if (l1 < l2) {
            if (l2 < l3) {
                pv1 = &v1;
                pv2 = &v2;
            } else {
                pv1 = &v3;
                pv2 = &v1;
            }
        } else {
            if (l1 < l3) {
                pv1 = &v1;
                pv2 = &v2;
            } else {
                pv1 = &v2;
                pv2 = &v3;
            }
        }
        if (lav) *lav = (std::sqrt(l1) + std::sqrt(l2) + std::sqrt(l3)) / 3.0;
    }
    n = -cross(*pv1, *pv2);
}

inline double triarea(const dvec3 &pa, const dvec3 &pb, const dvec3 &pc) {
    return 0.5 * glm::length(cross(pb - pa, pc - pa));
}

inline double orient3dfast(const dvec3 &pa, const dvec3 &pb, const dvec3 &pc, const dvec3 &pd) {
    const dvec3 ad = pa - pd, bd = pb - pd, cd = pc - pd;
    return ad.x * (bd.y * cd.z - bd.z * cd.y) + bd.x * (cd.y * ad.z - cd.z * ad.y) + cd.x * (ad.y * bd.z - ad.z * bd.y);
}

// The angle between the normals of (pa, pb, pc1) and (pa, pb, pc2), between 0 and 2 pi.
inline double facedihedral(const dvec3 &pa, const dvec3 &pb, const dvec3 &pc1, const dvec3 &pc2) {
    dvec3 n1, n2;
    facenormal(pa, pb, pc1, n1, 1, nullptr);
    facenormal(pa, pb, pc2, n2, 1, nullptr);
    const double n1len = glm::length(n1), n2len = glm::length(n2);
    double costheta = dot(n1, n2) / (n1len * n2len);
    if (costheta > 1.0) costheta = 1.0;
    else if (costheta < -1.0) costheta = -1.0;
    double theta = std::acos(costheta);
    const double ori = orient3d(pa, pb, pc1, pc2);
    if (ori > 0.0) theta = 2 * PI - theta;
    return theta;
}

// The angle at o in (o, p1, p2).
// With a normal supplied, the reflex angle is returned when p1 and p2 turn the other way about it.
inline double interiorangle(const dvec3 &o, const dvec3 &p1, const dvec3 &p2, const dvec3 *n) {
    const dvec3 v1 = p1 - o, v2 = p2 - o;
    const double l1 = glm::length(v1), l2 = glm::length(v2);
    double cosAngle = dot(v1, v2) / (l1 * l2);
    if (cosAngle > 1.0) cosAngle = 1.0;
    else if (cosAngle < -1.0) cosAngle = -1.0;
    double theta = std::acos(cosAngle);
    if (n != nullptr) {
        if (dot(cross(v1, v2), *n) < 0) theta = 2 * PI - theta;
    }
    return theta;
}

inline void projpt2face(const dvec3 &p, const dvec3 &f1, const dvec3 &f2, const dvec3 &f3, dvec3 &prj) {
    dvec3 fnormal;
    facenormal(f1, f2, f3, fnormal, 1, nullptr);
    fnormal /= glm::length(fnormal);
    prj = p - dot(fnormal, p - f1) * fnormal;
}

// An LU factorisation with partial pivoting, used to solve the small dense systems the circumcentre constructions set up.
inline bool lu_decmp(double lu[4][4], int n, int *ps, double *d, int N) {
    double scales[4];
    *d = 1.0;
    for (int i = N; i < n + N; ++i) {
        double biggest = 0.0;
        for (int j = N; j < n + N; ++j) biggest = std::max(biggest, std::abs(lu[i][j]));
        if (biggest != 0.0) scales[i] = 1.0 / biggest;
        else {
            scales[i] = 0.0;
            return false;
        }
        ps[i] = i;
    }
    for (int k = N; k < n + N - 1; ++k) {
        double biggest = 0.0;
        int pivotindex = k;
        for (int i = k; i < n + N; ++i) {
            const double size = std::abs(lu[ps[i]][k]) * scales[ps[i]];
            if (size > biggest) {
                biggest = size;
                pivotindex = i;
            }
        }
        if (biggest == 0.0) return false;
        if (pivotindex != k) {
            std::swap(ps[k], ps[pivotindex]);
            *d = -*d;
        }
        const int pivotrow = ps[k];
        const double pivot = lu[pivotrow][k];
        for (int i = k + 1; i < n + N; ++i) {
            const int therow = ps[i];
            const double mult = lu[therow][k] / pivot;
            lu[therow][k] = mult;
            if (mult != 0.0) {
                for (int j = k + 1; j < n + N; ++j) lu[therow][j] -= mult * lu[pivotrow][j];
            }
        }
    }
    return lu[ps[n + N - 1]][n + N - 1] != 0.0;
}

inline void lu_solve(double lu[4][4], int n, int *ps, double *b, int N) {
    // The right-hand side is read through the pivot sequence, so the solution comes back in the original row order.
    // The result accumulates in X because b still holds entries not yet read.
    double X[4]{};
    for (int i = N; i < n + N; ++i) {
        double dot = 0.0;
        for (int j = N; j < i + N; ++j) dot += lu[ps[i]][j] * X[j];
        X[i] = b[ps[i]] - dot;
    }
    for (int i = n + N - 1; i >= N; --i) {
        double dot = 0.0;
        for (int j = i + 1; j < n + N; ++j) dot += lu[ps[i]][j] * X[j];
        X[i] = (X[i] - dot) / lu[ps[i]][i];
    }
    for (int i = N; i < n + N; ++i) b[i] = X[i];
}

// With pd omitted it returns the circumcircle of the triangle, lying in the triangle's own plane.
// The four inward face normals of a tet, from its already-factorised edge matrix.
// Each of the first three solves for one unit right-hand side, and the fourth closes the sum.
inline void face_normals(double A[4][4], int *indx, dvec3 *N) {
    for (int j = 0; j < 3; ++j) {
        double n[4]{};
        n[j] = 1.0; // Positive points inward.
        lu_solve(A, 3, indx, n, 0);
        N[j] = dvec3{n[0], n[1], n[2]};
    }
    N[3] = -N[0] - N[1] - N[2];
}

// The three vectors as the rows of the 3x3 system the LU routines factorise.
inline void set_rows(double A[4][4], const dvec3 &r0, const dvec3 &r1, const dvec3 &r2) {
    A[0][0] = r0.x;
    A[0][1] = r0.y;
    A[0][2] = r0.z;
    A[1][0] = r1.x;
    A[1][1] = r1.y;
    A[1][2] = r1.z;
    A[2][0] = r2.x;
    A[2][1] = r2.y;
    A[2][2] = r2.z;
}

inline bool circumsphere(const dvec3 &pa, const dvec3 &pb, const dvec3 &pc, const dvec3 *pd, dvec3 *cent, double *radius) {
    double A[4][4]{}, rhs[4]{};
    int indx[4]{};
    double D;
    const dvec3 r0 = pb - pa, r1 = pc - pa;
    const dvec3 r2 = pd != nullptr ? *pd - pa : cross(r0, r1);
    set_rows(A, r0, r1, r2);
    rhs[0] = 0.5 * dot(r0, r0);
    rhs[1] = 0.5 * dot(r1, r1);
    rhs[2] = pd != nullptr ? 0.5 * dot(r2, r2) : 0.0;
    if (!lu_decmp(A, 3, indx, &D, 0)) {
        if (radius != nullptr) *radius = 0.0;
        return false;
    }
    lu_solve(A, 3, indx, rhs, 0);
    if (cent != nullptr) *cent = pa + dvec3{rhs[0], rhs[1], rhs[2]};
    if (radius != nullptr) *radius = std::sqrt(rhs[0] * rhs[0] + rhs[1] * rhs[1] + rhs[2] * rhs[2]);
    return true;
}

// Where the line through (e1, e2) meets the plane of (pa, pb, pc).
// Both determinants are exact, because the ratio of the two places a Steiner point.
inline void planelineint(const dvec3 &pa, const dvec3 &pb, const dvec3 &pc, const dvec3 &e1, const dvec3 &e2, dvec3 &ip, double *u) {
    const dvec3 vuv = e2 - e1;
    // The coordinates transposed into the rows of the 4x4 system.
    // The line's direction is the fourth column, with the heights it carries.
    const dvec3 A{pa.x, pb.x, pc.x}, B{pa.y, pb.y, pc.y}, C{pa.z, pb.z, pc.z}, D{1, 1, 1}, O{0, 0, 0};
    const double det = geom::Orient4DExactCofactor(A, B, C, D, O, -vuv.x, -vuv.y, -vuv.z, 0, 0);
    if (det != 0.0) {
        *u = geom::Orient3DExactCofactor(pa, pb, pc, e1) / det;
        // Per component, which is the form the intersection points are compared in.
        for (int i = 0; i < 3; ++i) ip[i] = e1[i] + (*u) * vuv[i];
    } else {
        *u = 0.0;
        ip = dvec3{0, 0, 0};
    }
}

// The shortest segment [P, Q] between the lines (A, B) and (C, D).
inline int linelineint(const dvec3 &A, const dvec3 &B, const dvec3 &C, const dvec3 &D, dvec3 &P, dvec3 &Q, double *tp, double *tq, double eps_bound) {
    const dvec3 vab = B - A, vcd = D - C, vca = A - C;
    const double vab_vab = dot(vab, vab), vcd_vcd = dot(vcd, vcd), vab_vcd = dot(vab, vcd);
    const double det = vab_vab * vcd_vcd - vab_vcd * vab_vcd;
    const double eps = det / (std::abs(vab_vab * vcd_vcd) + std::abs(vab_vcd * vab_vcd));
    if (eps < eps_bound) return 0;
    const double vca_vab = dot(vca, vab), vca_vcd = dot(vca, vcd);
    *tp = (vcd_vcd * (-vca_vab) + vab_vcd * vca_vcd) / det;
    *tq = (vab_vcd * (-vca_vab) + vab_vab * vca_vcd) / det;
    P = A + (*tp) * vab;
    Q = C + (*tq) * vcd;
    return 1;
}

// The orientation of five points lifted to the given heights.
// The expression order is fixed, since callers compare one result against another.
// The approximate value is returned whenever it clears the first error bound, and the rest goes to the refinement stages.
// The caller compares a sum of these against zero, so a term the filter cannot resolve decides the comparison.
inline double orient4d(const dvec3 &pa, const dvec3 &pb, const dvec3 &pc, const dvec3 &pd, const dvec3 &pe, double aheight, double bheight, double cheight, double dheight, double eheight) {
    const double aex = pa.x - pe.x, bex = pb.x - pe.x, cex = pc.x - pe.x, dex = pd.x - pe.x;
    const double aey = pa.y - pe.y, bey = pb.y - pe.y, cey = pc.y - pe.y, dey = pd.y - pe.y;
    const double aez = pa.z - pe.z, bez = pb.z - pe.z, cez = pc.z - pe.z, dez = pd.z - pe.z;
    const double aeheight = aheight - eheight, beheight = bheight - eheight, ceheight = cheight - eheight, deheight = dheight - eheight;

    const double aexbey = aex * bey, bexaey = bex * aey, ab = aexbey - bexaey;
    const double bexcey = bex * cey, cexbey = cex * bey, bc = bexcey - cexbey;
    const double cexdey = cex * dey, dexcey = dex * cey, cd = cexdey - dexcey;
    const double dexaey = dex * aey, aexdey = aex * dey, da = dexaey - aexdey;
    const double aexcey = aex * cey, cexaey = cex * aey, ac = aexcey - cexaey;
    const double bexdey = bex * dey, dexbey = dex * bey, bd = bexdey - dexbey;

    const double abc = aez * bc - bez * ac + cez * ab;
    const double bcd = bez * cd - cez * bd + dez * bc;
    const double cda = cez * da + dez * ac + aez * cd;
    const double dab = dez * ab + aez * bd + bez * da;

    const double det = (deheight * abc - ceheight * dab) + (beheight * cda - aeheight * bcd);

    constexpr double Epsilon = 0x1p-53;
    constexpr double ErrBoundA = (16.0 + 224.0 * Epsilon) * Epsilon;
    const double aezplus = std::abs(aez), bezplus = std::abs(bez), cezplus = std::abs(cez), dezplus = std::abs(dez);
    const double permanent =
        ((std::abs(cexdey) + std::abs(dexcey)) * bezplus + (std::abs(dexbey) + std::abs(bexdey)) * cezplus + (std::abs(bexcey) + std::abs(cexbey)) * dezplus) * std::abs(aeheight) +
        ((std::abs(dexaey) + std::abs(aexdey)) * cezplus + (std::abs(aexcey) + std::abs(cexaey)) * dezplus + (std::abs(cexdey) + std::abs(dexcey)) * aezplus) * std::abs(beheight) +
        ((std::abs(aexbey) + std::abs(bexaey)) * dezplus + (std::abs(bexdey) + std::abs(dexbey)) * aezplus + (std::abs(dexaey) + std::abs(aexdey)) * bezplus) * std::abs(ceheight) +
        ((std::abs(bexcey) + std::abs(cexbey)) * aezplus + (std::abs(cexaey) + std::abs(aexcey)) * bezplus + (std::abs(aexbey) + std::abs(bexaey)) * cezplus) * std::abs(deheight);
    const double errbound = ErrBoundA * permanent;
    if (det > errbound || -det > errbound) return det;

    return geom::Orient4DRefined(pa, pb, pc, pd, pe, aheight, bheight, cheight, dheight, eheight, permanent);
}

// 24 times the 4d volume of the prism between the tet and its lift onto the paraboloid.
// It decomposes into four 4d orientations.
// The magnitudes are summed, so a term of either sign adds to the volume.
inline double tetprismvol(const dvec3 &p0, const dvec3 &p1, const dvec3 &p2, const dvec3 &p3) {
    const double w4 = dot(p0, p0), w5 = dot(p1, p1), w6 = dot(p2, p2), w7 = dot(p3, p3);
    const double vol0 = orient4d(p1, p2, p0, p3, p3, w5, w6, w4, 0, w7);
    const double vol1 = orient4d(p3, p2, p2, p0, p1, 0, w6, 0, 0, 0);
    const double vol2 = orient4d(p0, p2, p3, p0, p1, w4, w6, 0, 0, 0);
    const double vol3 = orient4d(p2, p1, p0, p3, p1, w6, w5, w4, 0, 0);
    return std::abs(vol0) + std::abs(vol1) + std::abs(vol2) + std::abs(vol3);
}

// Is pd inside the circumcircle of (pa, pb, pc), all four coplanar?
inline double incircle3d(const dvec3 &pa, const dvec3 &pb, const dvec3 &pc, const dvec3 &pd) {
    double area2[2];
    dvec3 nn1, nn2, cc;
    double sign, r, d;
    facenormal(pa, pb, pc, nn1, 1, nullptr);
    area2[0] = dot(nn1, nn1);
    facenormal(pb, pa, pd, nn2, 1, nullptr);
    area2[1] = dot(nn2, nn2);
    if (area2[0] > area2[1]) {
        circumsphere(pa, pb, pc, nullptr, &cc, &r);
        d = distance(cc, pd);
    } else {
        if (area2[1] > 0) {
            circumsphere(pb, pa, pd, nullptr, &cc, &r);
            d = distance(cc, pc);
        } else {
            // The four points are collinear.
            // This case only happens on the boundary.
            return 0;
        }
    }
    sign = d - r;
    if (std::abs(sign) / r < 1e-8) sign = 0;
    return sign;
}

// One verdict of a triangle-edge test: what was met, and the two vertices naming where.
// Slot 0 takes the first verdict and slot 1 the second, each writing its own pair of position slots.
inline void set_inter(int *types, int *pos, int slot, int type, int p0, int p1) {
    types[slot] = type;
    pos[slot * 2] = p0;
    pos[slot * 2 + 1] = p1;
}

// The coplanar branch of the triangle-edge test.
// R lies strictly above the plane of (A, B, C); with none supplied one is built from the face normal.
// Returns 0 when disjoint, 1 when intersecting and level is 0, and 4 with types and pos filled otherwise.
inline int tri_edge_2d(const dvec3 &A, const dvec3 &B, const dvec3 &C, const dvec3 &P, const dvec3 &Q, const dvec3 *Rin, int level, int *types, int *pos) {
    dvec3 abovept;
    const dvec3 *R = Rin;
    if (R == nullptr) {
        dvec3 n;
        facenormal(A, B, C, n, 1, nullptr);
        const double nlen = std::sqrt(dot(n, n));
        if (nlen == 0) return 0; // The triangle is degenerate: a line-line test would be needed.
        n /= nlen;
        const double len = (distance(A, B) + distance(B, C) + distance(C, A)) / 3.0;
        abovept = A + len * n;
        R = &abovept;
    }

    const double sA = orient3d(P, Q, *R, A);
    const double sB = orient3d(P, Q, *R, B);
    const double sC = orient3d(P, Q, *R, C);

    const dvec3 *U[3], *V[3];
    int pu[3], pv[3], z1;
    const auto setU = [&](const dvec3 &u0, const dvec3 &u1, const dvec3 &u2, int a0, int a1, int a2) {
        U[0] = &u0;
        U[1] = &u1;
        U[2] = &u2;
        pu[0] = a0;
        pu[1] = a1;
        pu[2] = a2;
    };
    const auto setV = [&](const dvec3 &v0, const dvec3 &v1, const dvec3 &v2, int a0, int a1, int a2) {
        V[0] = &v0;
        V[1] = &v1;
        V[2] = &v2;
        pv[0] = a0;
        pv[1] = a1;
        pv[2] = a2;
    };

    if (sA < 0) {
        if (sB < 0) {
            if (sC < 0) return 0; // (---)
            setU(A, B, C, 0, 1, 2);
            setV(P, Q, *R, 0, 1, 2);
            z1 = sC > 0 ? 0 : 1; // (--+) or (--0)
        } else {
            if (sB > 0) {
                if (sC < 0) { // (-+-)
                    setU(C, A, B, 2, 0, 1);
                    setV(P, Q, *R, 0, 1, 2);
                    z1 = 0;
                } else if (sC > 0) { // (-++)
                    setU(B, C, A, 1, 2, 0);
                    setV(Q, P, *R, 1, 0, 2);
                    z1 = 0;
                } else { // (-+0)
                    setU(C, A, B, 2, 0, 1);
                    setV(P, Q, *R, 0, 1, 2);
                    z1 = 2;
                }
            } else {
                if (sC < 0) { // (-0-)
                    setU(C, A, B, 2, 0, 1);
                    setV(P, Q, *R, 0, 1, 2);
                    z1 = 1;
                } else if (sC > 0) { // (-0+)
                    setU(B, C, A, 1, 2, 0);
                    setV(Q, P, *R, 1, 0, 2);
                    z1 = 2;
                } else { // (-00)
                    setU(B, C, A, 1, 2, 0);
                    setV(Q, P, *R, 1, 0, 2);
                    z1 = 3;
                }
            }
        }
    } else if (sA > 0) {
        if (sB < 0) {
            if (sC < 0) { // (+--)
                setU(B, C, A, 1, 2, 0);
                setV(P, Q, *R, 0, 1, 2);
                z1 = 0;
            } else if (sC > 0) { // (+-+)
                setU(C, A, B, 2, 0, 1);
                setV(Q, P, *R, 1, 0, 2);
                z1 = 0;
            } else { // (+-0)
                setU(C, A, B, 2, 0, 1);
                setV(Q, P, *R, 1, 0, 2);
                z1 = 2;
            }
        } else if (sB > 0) {
            if (sC < 0) { // (++-)
                setU(A, B, C, 0, 1, 2);
                setV(Q, P, *R, 1, 0, 2);
                z1 = 0;
            } else if (sC > 0) {
                return 0; // (+++)
            } else { // (++0)
                setU(A, B, C, 0, 1, 2);
                setV(Q, P, *R, 1, 0, 2);
                z1 = 1;
            }
        } else { // (+0#)
            if (sC < 0) { // (+0-)
                setU(B, C, A, 1, 2, 0);
                setV(P, Q, *R, 0, 1, 2);
                z1 = 2;
            } else if (sC > 0) { // (+0+)
                setU(C, A, B, 2, 0, 1);
                setV(Q, P, *R, 1, 0, 2);
                z1 = 1;
            } else { // (+00)
                setU(B, C, A, 1, 2, 0);
                setV(P, Q, *R, 0, 1, 2);
                z1 = 3;
            }
        }
    } else {
        if (sB < 0) {
            if (sC < 0) { // (0--)
                setU(B, C, A, 1, 2, 0);
                setV(P, Q, *R, 0, 1, 2);
                z1 = 1;
            } else if (sC > 0) { // (0-+)
                setU(A, B, C, 0, 1, 2);
                setV(P, Q, *R, 0, 1, 2);
                z1 = 2;
            } else { // (0-0)
                setU(C, A, B, 2, 0, 1);
                setV(Q, P, *R, 1, 0, 2);
                z1 = 3;
            }
        } else if (sB > 0) {
            if (sC < 0) { // (0+-)
                setU(A, B, C, 0, 1, 2);
                setV(Q, P, *R, 1, 0, 2);
                z1 = 2;
            } else if (sC > 0) { // (0++)
                setU(B, C, A, 1, 2, 0);
                setV(Q, P, *R, 1, 0, 2);
                z1 = 1;
            } else { // (0+0)
                setU(C, A, B, 2, 0, 1);
                setV(P, Q, *R, 0, 1, 2);
                z1 = 3;
            }
        } else { // (00#)
            if (sC < 0) { // (00-)
                setU(A, B, C, 0, 1, 2);
                setV(Q, P, *R, 1, 0, 2);
                z1 = 3;
            } else if (sC > 0) { // (00+)
                setU(A, B, C, 0, 1, 2);
                setV(P, Q, *R, 0, 1, 2);
                z1 = 3;
            } else { // (000), only reachable when ABC is degenerate
                setU(A, B, C, 0, 1, 2);
                setV(P, Q, *R, 0, 1, 2);
                z1 = 4;
            }
        }
    }

    const double s1 = orient3d(*U[0], *U[2], *R, *V[1]); // A, C, R, Q
    const double s2 = orient3d(*U[1], *U[2], *R, *V[0]); // B, C, R, P
    if (s1 > 0) return 0;
    if (s2 < 0) return 0;
    if (level == 0) return 1;

    if (z1 == 1) {
        if (s1 == 0) { // C = Q
            set_inter(types, pos, 0, ShareVert, pu[2], pv[1]);
            types[1] = Disjoint;
        } else if (s2 == 0) { // C = P
            set_inter(types, pos, 0, ShareVert, pu[2], pv[0]);
            types[1] = Disjoint;
        } else { // C in [P, Q]
            set_inter(types, pos, 0, AcrossVert, pu[2], pv[0]);
            types[1] = Disjoint;
        }
        return 4;
    }

    const double s3 = orient3d(*U[0], *U[2], *R, *V[0]); // A, C, R, P
    const double s4 = orient3d(*U[1], *U[2], *R, *V[1]); // B, C, R, Q

    if (z1 == 0) {
        if (s1 < 0) {
            if (s3 > 0) {
                if (s4 > 0) { // [P, Q] overlaps [k, l]
                    set_inter(types, pos, 0, AcrossEdge, pu[2], pv[0]);
                    set_inter(types, pos, 1, TouchFace, 3, pv[1]);
                } else if (s4 == 0) { // Q = l
                    set_inter(types, pos, 0, AcrossEdge, pu[2], pv[0]);
                    set_inter(types, pos, 1, TouchEdge, pu[1], pv[1]);
                } else { // [P, Q] contains [k, l]
                    set_inter(types, pos, 0, AcrossEdge, pu[2], pv[0]);
                    set_inter(types, pos, 1, AcrossEdge, pu[1], pv[0]);
                }
            } else if (s3 == 0) {
                if (s4 > 0) { // P = k
                    set_inter(types, pos, 0, TouchEdge, pu[2], pv[0]);
                    set_inter(types, pos, 1, TouchFace, 3, pv[1]);
                } else if (s4 == 0) { // [P, Q] = [k, l]
                    set_inter(types, pos, 0, TouchEdge, pu[2], pv[0]);
                    set_inter(types, pos, 1, TouchEdge, pu[1], pv[1]);
                } else { // P = k, [P, Q] contains [k, l]
                    set_inter(types, pos, 0, TouchEdge, pu[2], pv[0]);
                    set_inter(types, pos, 1, AcrossEdge, pu[1], pv[0]);
                }
            } else { // s3 < 0
                if (s2 > 0) {
                    if (s4 > 0) { // [P, Q] in [k, l]
                        set_inter(types, pos, 0, TouchFace, 3, pv[0]);
                        set_inter(types, pos, 1, TouchFace, 3, pv[1]);
                    } else if (s4 == 0) { // Q = l
                        set_inter(types, pos, 0, TouchFace, 3, pv[0]);
                        set_inter(types, pos, 1, TouchEdge, pu[1], pv[1]);
                    } else { // [P, Q] overlaps [k, l]
                        set_inter(types, pos, 0, TouchFace, 3, pv[0]);
                        set_inter(types, pos, 1, AcrossEdge, pu[1], pv[0]);
                    }
                } else { // P = l
                    set_inter(types, pos, 0, TouchEdge, pu[1], pv[0]);
                    types[1] = Disjoint;
                }
            }
        } else { // s1 == 0, Q = k
            set_inter(types, pos, 0, TouchEdge, pu[2], pv[1]);
            types[1] = Disjoint;
        }
    } else if (z1 == 2) {
        if (s1 < 0) {
            if (s3 > 0) {
                if (s4 > 0) { // [P, Q] overlaps [A, l]
                    set_inter(types, pos, 0, AcrossVert, pu[0], pv[0]);
                    set_inter(types, pos, 1, TouchFace, 3, pv[1]);
                } else if (s4 == 0) { // Q = l
                    set_inter(types, pos, 0, AcrossVert, pu[0], pv[0]);
                    set_inter(types, pos, 1, TouchEdge, pu[1], pv[1]);
                } else { // [P, Q] contains [A, l]
                    set_inter(types, pos, 0, AcrossVert, pu[0], pv[0]);
                    set_inter(types, pos, 1, AcrossEdge, pu[1], pv[0]);
                }
            } else if (s3 == 0) {
                if (s4 > 0) { // P = A
                    set_inter(types, pos, 0, ShareVert, pu[0], pv[0]);
                    set_inter(types, pos, 1, TouchFace, 3, pv[1]);
                } else if (s4 == 0) { // [P, Q] = [A, l]
                    set_inter(types, pos, 0, ShareVert, pu[0], pv[0]);
                    set_inter(types, pos, 1, TouchEdge, pu[1], pv[1]);
                } else { // Q = l
                    set_inter(types, pos, 0, ShareVert, pu[0], pv[0]);
                    set_inter(types, pos, 1, AcrossEdge, pu[1], pv[0]);
                }
            } else { // s3 < 0
                if (s2 > 0) {
                    // The second verdict overwrites types[0], pos[0] and pos[1] rather than filling the [1] and [2],[3] slots.
                    if (s4 > 0) { // [P, Q] in [A, l]
                        set_inter(types, pos, 0, TouchFace, 3, pv[0]);
                        set_inter(types, pos, 0, TouchFace, 3, pv[1]);
                    } else if (s4 == 0) { // Q = l
                        set_inter(types, pos, 0, TouchFace, 3, pv[0]);
                        set_inter(types, pos, 0, TouchEdge, pu[1], pv[1]);
                    } else { // [P, Q] overlaps [A, l]
                        set_inter(types, pos, 0, TouchFace, 3, pv[0]);
                        set_inter(types, pos, 0, AcrossEdge, pu[1], pv[0]);
                    }
                } else { // P = l
                    set_inter(types, pos, 0, TouchEdge, pu[1], pv[0]);
                    types[1] = Disjoint;
                }
            }
        } else { // s1 == 0, Q = A
            set_inter(types, pos, 0, ShareVert, pu[0], pv[1]);
            types[1] = Disjoint;
        }
    } else if (z1 == 3) {
        if (s1 < 0) {
            if (s3 > 0) {
                if (s4 > 0) { // [P, Q] overlaps [A, B]
                    set_inter(types, pos, 0, AcrossVert, pu[0], pv[0]);
                    set_inter(types, pos, 1, TouchEdge, pu[0], pv[1]);
                } else if (s4 == 0) { // Q = B
                    set_inter(types, pos, 0, AcrossVert, pu[0], pv[0]);
                    set_inter(types, pos, 1, ShareVert, pu[1], pv[1]);
                } else { // [P, Q] contains [A, B]
                    set_inter(types, pos, 0, AcrossVert, pu[0], pv[0]);
                    set_inter(types, pos, 1, AcrossVert, pu[1], pv[0]);
                }
            } else if (s3 == 0) {
                if (s4 > 0) { // P = A
                    set_inter(types, pos, 0, ShareVert, pu[0], pv[0]);
                    set_inter(types, pos, 1, TouchEdge, pu[0], pv[1]);
                } else if (s4 == 0) { // [P, Q] = [A, B]
                    set_inter(types, pos, 0, ShareEdge, pu[0], pv[0]);
                    types[1] = Disjoint;
                } else { // P = A
                    set_inter(types, pos, 0, ShareVert, pu[0], pv[0]);
                    set_inter(types, pos, 1, AcrossVert, pu[1], pv[0]);
                }
            } else { // s3 < 0
                if (s2 > 0) {
                    if (s4 > 0) { // [P, Q] in [A, B]
                        set_inter(types, pos, 0, TouchEdge, pu[0], pv[0]);
                        set_inter(types, pos, 1, TouchEdge, pu[0], pv[1]);
                    } else if (s4 == 0) { // Q = B
                        set_inter(types, pos, 0, TouchEdge, pu[0], pv[0]);
                        set_inter(types, pos, 1, ShareVert, pu[1], pv[1]);
                    } else { // [P, Q] overlaps [A, B]
                        set_inter(types, pos, 0, TouchEdge, pu[0], pv[0]);
                        set_inter(types, pos, 1, AcrossVert, pu[1], pv[0]);
                    }
                } else { // P = B
                    set_inter(types, pos, 0, ShareVert, pu[1], pv[0]);
                    types[1] = Disjoint;
                }
            }
        } else { // s1 == 0, Q = A
            set_inter(types, pos, 0, ShareVert, pu[0], pv[1]);
            types[1] = Disjoint;
        }
    }
    return 4;
}

// The non-coplanar branch, given the two orientations of P and Q against the plane of (A, B, C).
inline int tri_edge_tail(const dvec3 &A, const dvec3 &B, const dvec3 &C, const dvec3 &P, const dvec3 &Q, const dvec3 *R, double sP, double sQ, int level, int *types, int *pos) {
    const dvec3 *U[3], *V[3];
    int pu[3], pv[3], z1;
    const auto set = [&](const dvec3 &u0, const dvec3 &u1, const dvec3 &u2, int a0, int a1, int a2, const dvec3 &v0, const dvec3 &v1, int b0, int b1) {
        U[0] = &u0;
        U[1] = &u1;
        U[2] = &u2;
        pu[0] = a0;
        pu[1] = a1;
        pu[2] = a2;
        V[0] = &v0;
        V[1] = &v1;
        V[2] = R;
        pv[0] = b0;
        pv[1] = b1;
        pv[2] = 2;
    };

    if (sP < 0) {
        if (sQ < 0) return 0; // (--)
        set(A, B, C, 0, 1, 2, P, Q, 0, 1);
        z1 = sQ > 0 ? 0 : 1; // (-+) or (-0)
    } else if (sP > 0) {
        if (sQ < 0) { // (+-)
            set(A, B, C, 0, 1, 2, Q, P, 1, 0);
            z1 = 0;
        } else if (sQ > 0) {
            return 0; // (++)
        } else { // (+0)
            set(B, A, C, 1, 0, 2, P, Q, 0, 1);
            z1 = 1;
        }
    } else { // sP == 0
        if (sQ < 0) { // (0-)
            set(A, B, C, 0, 1, 2, Q, P, 1, 0);
            z1 = 1;
        } else if (sQ > 0) { // (0+)
            set(B, A, C, 1, 0, 2, Q, P, 1, 0);
            z1 = 1;
        } else { // (00): all five points are coplanar
            return tri_edge_2d(A, B, C, P, Q, R, level, types, pos);
        }
    }

    const double s1 = orient3d(*U[0], *U[1], *V[0], *V[1]);
    if (s1 < 0) return 0;
    const double s2 = orient3d(*U[1], *U[2], *V[0], *V[1]);
    if (s2 < 0) return 0;
    const double s3 = orient3d(*U[2], *U[0], *V[0], *V[1]);
    if (s3 < 0) return 0;
    if (level == 0) return 1;

    types[1] = Disjoint; // No second intersection point.

    if (z1 == 0) {
        if (s1 > 0) {
            if (s2 > 0) {
                if (s3 > 0) { // [P, Q] passes the interior of [A, B, C]
                    set_inter(types, pos, 0, AcrossFace, 3, 0);
                } else { // [P, Q] meets [C, A]
                    set_inter(types, pos, 0, AcrossEdge, pu[2], 0);
                }
            } else {
                if (s3 > 0) { // [P, Q] meets [B, C]
                    set_inter(types, pos, 0, AcrossEdge, pu[1], 0);
                } else { // [P, Q] passes C
                    set_inter(types, pos, 0, AcrossVert, pu[2], 0);
                }
            }
        } else {
            if (s2 > 0) {
                if (s3 > 0) { // [P, Q] meets [A, B]
                    set_inter(types, pos, 0, AcrossEdge, pu[0], 0);
                } else { // [P, Q] passes A
                    set_inter(types, pos, 0, AcrossVert, pu[0], 0);
                }
            } else {
                if (s3 > 0) { // [P, Q] passes B
                    set_inter(types, pos, 0, AcrossVert, pu[1], 0);
                }
            }
        }
    } else { // z1 == 1
        if (s1 > 0) {
            if (s2 > 0) {
                if (s3 > 0) { // Q lies in [A, B, C]
                    set_inter(types, pos, 0, TouchFace, 0, pv[1]);
                } else { // Q lies on [C, A]
                    set_inter(types, pos, 0, TouchEdge, pu[2], pv[1]);
                }
            } else {
                if (s3 > 0) { // Q lies on [B, C]
                    set_inter(types, pos, 0, TouchEdge, pu[1], pv[1]);
                } else { // Q = C
                    set_inter(types, pos, 0, ShareVert, pu[2], pv[1]);
                }
            }
        } else {
            if (s2 > 0) {
                if (s3 > 0) { // Q lies on [A, B]
                    set_inter(types, pos, 0, TouchEdge, pu[0], pv[1]);
                } else { // Q = A
                    set_inter(types, pos, 0, ShareVert, pu[0], pv[1]);
                }
            } else {
                if (s3 > 0) { // Q = B
                    set_inter(types, pos, 0, ShareVert, pu[1], pv[1]);
                }
            }
        }
    }
    return 2;
}

inline int tri_edge_test(const dvec3 &A, const dvec3 &B, const dvec3 &C, const dvec3 &P, const dvec3 &Q, const dvec3 *R, int level, int *types, int *pos) {
    return tri_edge_tail(A, B, C, P, Q, R, orient3d(A, B, C, P), orient3d(A, B, C, Q), level, types, pos);
}

//=== Mesh ===
// The pools, the counters and every routine that works on them.
// Subfaces and subsegments share one pool, since their record is the same eleven slots.
// A `Kind` field tells them apart, and two free lists keep the allocations separate.

struct Mesh {
    //=== Pools ===
    // A dead tetrahedron and a dead shellface both have V[0] == None.
    // A freed slot is handed straight back out by the next allocation.
    std::vector<Point> Pts;
    std::vector<Tet> Tets;
    std::vector<Shell> Shells;
    std::vector<int> DeadTets, DeadSubfaces, DeadSubsegs, DeadPoints;
    long TetItems{0}, SubfaceItems{0}, SubsegItems{0}, PointItems{0};

    //=== Behaviour ===
    // The run path is fixed, apart from the quality pass and the volume bound.
    bool Quality{false};
    double Epsilon{1.0e-8};
    double MinRatio{2.0};
    // Zero leaves tet size unconstrained.
    double MaxVolume{0};
    double MinDihedral{3.5}, CosMinDihedral{0};
    double OptMaxDihedral{177.0}, CosOptMaxDihedral{0};
    double OptMaxAspRatio{1000.0}, OptMaxEdgeRatio{100.0};
    double SmoothAlpha{0.3};
    int SmoothCriterion{3}, SmoothMaxIter{7};
    int OptMaxFlipLevel{3}, OptIterations{3};
    int SupSteinerLevel{2}, AddSteinerAlgo{1};
    int UnflipQueueLimit{1000};
    int HilbertOrder{52}, HilbertLimit{8}, BrioThreshold{64};
    long TetrahedraPerBlock{8188};
    double BrioRatio{0.125};
    int FlipLinkLevel{-1}, FlipStarSize{-1}, FlipLinkLevelInc{1};
    double FacetSepAngTol{179.9}, CollinearAngTol{179.9}, FacetSmallAngTol{15.0};
    double CosFacetSepAngTol{0}, CosSmallAngTol{0};
    // Set for a piecewise-linear input, and cleared while a cavity's own Delaunay triangulation is built.
    int Plc{1};

    //=== State carried across the passes ===
    long HullSize{0};
    int CheckSubsegFlag{0}, CheckSubfaceFlag{0}, CheckConstraints{0};
    int BoundaryRecoveryFlag{0};
    int NonConvex{0};
    int UseInsertRadius{0};
    long InSegments{0};
    long DupVerts{0}, UnuVerts{0};
    long StSegRefCount{0}, StFacRefCount{0}, StVolRefCount{0};
    long Flip14Count{0}, Flip26Count{0}, FlipN2nCount{0};
    long Flip23Count{0}, Flip32Count{0}, Flip44Count{0}, Flip41Count{0}, Flip31Count{0}, Flip22Count{0};
    unsigned long RandomSeed{1};
    // Draws the sequence libc's rand() produces, held per mesh so one build's draws stand alone.
    std::minstd_rand0 Rng;
    double LongEst{0}, MinEdgeLength{0};
    dvec3 BoxMin{0}, BoxMax{0};
    long SteinerLeft{-1};
    // Set when recovery gives up on a facet it cannot recover, which means self-intersecting input.
    bool SkippedFacet{false};
    // How many segments and subfaces the initial Delaunay triangulation was missing.
    long MissingEdgeCount{0}, MissingFaceCount{0};

    Triface RecentTet{};
    Facet RecentSh{};

    //=== Working arrays, reused across calls ===
    // Their lifetime is part of the algorithm: one routine fills a list and a later routine drains it.
    std::vector<Triface> CaveOldTetList, CaveBdryList, CaveTetList;
    std::vector<Facet> CaveTetShList, CaveTetSegList;
    std::vector<int> CaveTetVertList;
    std::vector<Facet> CaveShList, CaveShBdList, CaveSegShList;
    std::vector<Facet> CaveEncSegList, CaveEncShList;
    std::vector<int> SubSegStack, SubFaceStack, SubVertStack;

    // A stack of faces that may need flipping, each carrying the tet handle it was pushed with.
    // The face is marked in that tet, so a duplicate push is refused.
    std::vector<BadFace> FlipStack;

    // Queues of bad-quality elements, drained by refinement and by mesh improvement.
    std::vector<BadFace> BadSubsegs, BadSubfacs;
    // 64-bucket priority queues over encroached subfaces and bad tets.
    // Each bucket is a singly linked list of slots in the store, chained through BadFace::nextitem.
    // A freed slot is handed straight back out by the next enqueue.
    std::vector<BadFace> BadTetStore;
    std::vector<int> DeadBadTets;
    long BadTetItems{0};
    int BtQueueFront[64], BtQueueTail[64];
    int BtNextNonEmptyQ[64]{}, BtFirstNonEmptyQ{-1}, BtRecentQ{-1};
    std::vector<BadFace> UnsplitBadTets;

    // The facet and segment maps refinement reads.
    std::vector<int> IdxSegmentFacetList, SegmentFacetList;
    std::vector<int> IdxRidgeVertexFacetList, RidgeVertexFacetList;
    std::vector<int> SegmentEndpointsList;
    std::vector<int> IdxSegmentRidgeVertexList, SegmentRidgeVertexList;

    // Input, kept for the surface mesh.
    std::vector<std::array<int, 3>> InTris;
    int NumInputPoints{0};

    //=== Allocation ===

    void maketetrahedron(Triface *newtet) {
        if (!DeadTets.empty()) {
            newtet->tet = DeadTets.back();
            DeadTets.pop_back();
            Tets[newtet->tet] = Tet{};
        } else {
            newtet->tet = int(Tets.size());
            Tets.emplace_back();
        }
        ++TetItems;
        newtet->ver = 11;
    }
    void maketetrahedron2(Triface *newtet, int pa, int pb, int pc, int pd) {
        maketetrahedron(newtet);
        Tet &e = Tets[newtet->tet];
        e.V[0] = pa;
        e.V[1] = pb;
        e.V[2] = pc;
        e.V[3] = pd;
    }
    void tetrahedrondealloc(int t) {
        Tets[t].V[0] = None;
        DeadTets.push_back(t);
        --TetItems;
    }

    void makeshellface(bool isseg, Facet *newface) {
        std::vector<int> &dead = isseg ? DeadSubsegs : DeadSubfaces;
        if (!dead.empty()) {
            newface->sh = dead.back();
            dead.pop_back();
            Shells[newface->sh] = Shell{};
        } else {
            newface->sh = int(Shells.size());
            Shells.emplace_back();
        }
        Shells[newface->sh].Kind = isseg ? 1 : 0;
        (isseg ? SubsegItems : SubfaceItems) += 1;
        newface->shver = 0;
    }
    void shellfacedealloc(bool isseg, int sh) {
        Shells[sh].V[0] = None;
        (isseg ? DeadSubsegs : DeadSubfaces).push_back(sh);
        (isseg ? SubsegItems : SubfaceItems) -= 1;
    }
    void makesubface(Facet *s) { makeshellface(false, s); }
    void makesubseg(Facet *s) { makeshellface(true, s); }
    void subfacedealloc(int sh) { shellfacedealloc(false, sh); }
    void subsegdealloc(int sh) { shellfacedealloc(true, sh); }

    int makepoint(VertType vtype) {
        int p;
        if (!DeadPoints.empty()) {
            p = DeadPoints.back();
            DeadPoints.pop_back();
            Pts[p] = Point{};
        } else {
            p = int(Pts.size());
            Pts.emplace_back();
        }
        ++PointItems;
        Pts[p].Flags = int(vtype) << 8;
        return p;
    }
    void pointdealloc(int p) {
        Pts[p].Flags = int(DeadVertex) << 8;
        DeadPoints.push_back(p);
        --PointItems;
    }

    //=== Primitives for tetrahedra ===

    void bond(const Triface &t1, const Triface &t2) {
        Tets[t1.tet].N[t1.ver & 3] = Encode2(t2.tet, Tbl.bondtbl[t1.ver][t2.ver]);
        Tets[t2.tet].N[t2.ver & 3] = Encode2(t1.tet, Tbl.bondtbl[t2.ver][t1.ver]);
    }

    static void enext(const Triface &t1, Triface &t2) { t2 = {t1.tet, Tbl.enexttbl[t1.ver]}; }
    static void enextself(Triface &t) { t.ver = Tbl.enexttbl[t.ver]; }
    static void eprev(const Triface &t1, Triface &t2) { t2 = {t1.tet, Tbl.eprevtbl[t1.ver]}; }
    static void eprevself(Triface &t) { t.ver = Tbl.eprevtbl[t.ver]; }
    static void esym(const Triface &t1, Triface &t2) { t2 = {t1.tet, esymtbl[t1.ver]}; }
    static void esymself(Triface &t) { t.ver = esymtbl[t.ver]; }
    static void enextesym(const Triface &t1, Triface &t2) { t2 = {t1.tet, Tbl.enextesymtbl[t1.ver]}; }
    static void enextesymself(Triface &t) { t.ver = Tbl.enextesymtbl[t.ver]; }
    static void eprevesym(const Triface &t1, Triface &t2) { t2 = {t1.tet, Tbl.eprevesymtbl[t1.ver]}; }
    static void eprevesymself(Triface &t) { t.ver = Tbl.eprevesymtbl[t.ver]; }
    static void eorgoppo(const Triface &t1, Triface &t2) { t2 = {t1.tet, Tbl.eorgoppotbl[t1.ver]}; }
    static void eorgoppoself(Triface &t) { t.ver = Tbl.eorgoppotbl[t.ver]; }
    static void edestoppo(const Triface &t1, Triface &t2) { t2 = {t1.tet, Tbl.edestoppotbl[t1.ver]}; }
    static void edestoppoself(Triface &t) { t.ver = Tbl.edestoppotbl[t.ver]; }

    void fsym(const Triface &t1, Triface &t2) const {
        Decode(Tets[t1.tet].N[t1.ver & 3], t2);
        t2.ver = Tbl.fsymtbl[t1.ver][t2.ver];
    }
    void fsymself(Triface &t) const {
        const int v = t.ver;
        Decode(Tets[t.tet].N[t.ver & 3], t);
        t.ver = Tbl.fsymtbl[v][t.ver];
    }
    void fnext(const Triface &t1, Triface &t2) const {
        Decode(Tets[t1.tet].N[Tbl.facepivot1[t1.ver]], t2);
        t2.ver = Tbl.facepivot2[t1.ver][t2.ver];
    }
    void fnextself(Triface &t) const {
        const int v = t.ver;
        Decode(Tets[t.tet].N[Tbl.facepivot1[t.ver]], t);
        t.ver = Tbl.facepivot2[v][t.ver];
    }

    int org(const Triface &t) const { return Tets[t.tet].V[orgpivot[t.ver]]; }
    int dest(const Triface &t) const { return Tets[t.tet].V[destpivot[t.ver]]; }
    int apex(const Triface &t) const { return Tets[t.tet].V[apexpivot[t.ver]]; }
    int oppo(const Triface &t) const { return Tets[t.tet].V[oppopivot[t.ver]]; }
    void setorg(const Triface &t, int p) { Tets[t.tet].V[orgpivot[t.ver]] = p; }
    void setdest(const Triface &t, int p) { Tets[t.tet].V[destpivot[t.ver]] = p; }
    void setapex(const Triface &t, int p) { Tets[t.tet].V[apexpivot[t.ver]] = p; }
    void setoppo(const Triface &t, int p) { Tets[t.tet].V[oppopivot[t.ver]] = p; }
    void setvertices(const Triface &t, int a, int b, int c, int d) {
        Tet &e = Tets[t.tet];
        e.V[orgpivot[t.ver]] = a;
        e.V[destpivot[t.ver]] = b;
        e.V[apexpivot[t.ver]] = c;
        e.V[oppopivot[t.ver]] = d;
    }

    void infect(const Triface &t) { Tets[t.tet].Marker |= 1; }
    void uninfect(const Triface &t) { Tets[t.tet].Marker &= ~1; }
    bool infected(const Triface &t) const { return (Tets[t.tet].Marker & 1) != 0; }
    void marktest(const Triface &t) { Tets[t.tet].Marker |= 2; }
    void unmarktest(const Triface &t) { Tets[t.tet].Marker &= ~2; }
    bool marktested(const Triface &t) const { return (Tets[t.tet].Marker & 2) != 0; }
    void markface(const Triface &t) { Tets[t.tet].Marker |= 4 << (t.ver & 3); }
    void unmarkface(const Triface &t) { Tets[t.tet].Marker &= ~(4 << (t.ver & 3)); }
    bool facemarked(const Triface &t) const { return (Tets[t.tet].Marker & (4 << (t.ver & 3))) != 0; }
    void marktest2(const Triface &t) { Tets[t.tet].Marker |= 4096; }
    void unmarktest2(const Triface &t) { Tets[t.tet].Marker &= ~4096; }
    bool marktest2ed(const Triface &t) const { return (Tets[t.tet].Marker & 4096) != 0; }
    int elemcounter(const Triface &t) const { return Tets[t.tet].Marker >> 16; }
    void setelemcounter(const Triface &t, int value) {
        int &c = Tets[t.tet].Marker;
        c = (c & 65535) | (value << 16);
    }
    void increaseelemcounter(const Triface &t) { setelemcounter(t, elemcounter(t) + 1); }
    void decreaseelemcounter(const Triface &t) { setelemcounter(t, elemcounter(t) - 1); }

    bool ishulltet(const Triface &t) const { return Tets[t.tet].V[3] == DummyPoint; }
    bool isdeadtet(const Triface &t) const { return t.tet < 0 || Tets[t.tet].V[0] == None; }

    //=== Primitives for subfaces and subsegments ===

    void sdissolve(const Facet &s) { Shells[s.sh].S[s.shver >> 1] = None; }
    void sbond(const Facet &s1, const Facet &s2) {
        Shells[s1.sh].S[s1.shver >> 1] = SEncode(s2);
        Shells[s2.sh].S[s2.shver >> 1] = SEncode(s1);
    }
    void sbond1(const Facet &s1, const Facet &s2) { Shells[s1.sh].S[s1.shver >> 1] = SEncode(s2); }
    void spivot(const Facet &s1, Facet &s2) const { SDecode(Shells[s1.sh].S[s1.shver >> 1], s2); }
    void spivotself(Facet &s) const { SDecode(Shells[s.sh].S[s.shver >> 1], s); }

    int sorg(const Facet &s) const { return Shells[s.sh].V[sorgpivot[s.shver]]; }
    int sdest(const Facet &s) const { return Shells[s.sh].V[sdestpivot[s.shver]]; }
    int sapex(const Facet &s) const { return Shells[s.sh].V[sapexpivot[s.shver]]; }
    void setsorg(const Facet &s, int p) { Shells[s.sh].V[sorgpivot[s.shver]] = p; }
    void setsdest(const Facet &s, int p) { Shells[s.sh].V[sdestpivot[s.shver]] = p; }
    void setsapex(const Facet &s, int p) { Shells[s.sh].V[sapexpivot[s.shver]] = p; }
    void setshvertices(const Facet &s, int pa, int pb, int pc) {
        setsorg(s, pa);
        setsdest(s, pb);
        setsapex(s, pc);
    }

    static void sesym(const Facet &s1, Facet &s2) { s2 = {s1.sh, s1.shver ^ 1}; }
    static void sesymself(Facet &s) { s.shver ^= 1; }
    static void senext(const Facet &s1, Facet &s2) { s2 = {s1.sh, snextpivot[s1.shver]}; }
    static void senextself(Facet &s) { s.shver = snextpivot[s.shver]; }
    static void senext2(const Facet &s1, Facet &s2) { s2 = {s1.sh, snextpivot[snextpivot[s1.shver]]}; }
    static void senext2self(Facet &s) { s.shver = snextpivot[snextpivot[s.shver]]; }

    double areabound(const Facet &s) const { return Shells[s.sh].AreaBound; }
    void setareabound(const Facet &s, double v) { Shells[s.sh].AreaBound = v; }
    int shellmark(const Facet &s) const { return Shells[s.sh].Mark; }
    void setshellmark(const Facet &s, int v) { Shells[s.sh].Mark = v; }
    void sinfect(const Facet &s) { Shells[s.sh].Flags |= 1; }
    void suninfect(const Facet &s) { Shells[s.sh].Flags &= ~1; }
    bool sinfected(const Facet &s) const { return (Shells[s.sh].Flags & 1) != 0; }
    void smarktest(const Facet &s) { Shells[s.sh].Flags |= 2; }
    void sunmarktest(const Facet &s) { Shells[s.sh].Flags &= ~2; }
    bool smarktested(const Facet &s) const { return (Shells[s.sh].Flags & 2) != 0; }
    void smarktest2(const Facet &s) { Shells[s.sh].Flags |= 4; }
    bool smarktest2ed(const Facet &s) const { return (Shells[s.sh].Flags & 4) != 0; }
    void smarktest3(const Facet &s) { Shells[s.sh].Flags |= 8; }
    bool smarktest3ed(const Facet &s) const { return (Shells[s.sh].Flags & 8) != 0; }
    void setfacetindex(const Facet &s, int v) { Shells[s.sh].FacetIndex = v; }
    int getfacetindex(const Facet &s) const { return Shells[s.sh].FacetIndex; }
    bool isdeadsh(const Facet &s) const { return s.sh < 0 || Shells[s.sh].V[0] == None; }

    //=== Primitives between tetrahedra and subfaces ===

    void tsbond(const Triface &t, const Facet &s) {
        Tets[t.tet].Sub[t.ver & 3] = SEncode2(s.sh, Tbl.tsbondtbl[t.ver][s.shver]);
        Shells[s.sh].T[s.shver & 1] = Encode2(t.tet, Tbl.stbondtbl[t.ver][s.shver]);
    }
    void tspivot(const Triface &t, Facet &s) const {
        SDecode(Tets[t.tet].Sub[t.ver & 3], s);
        if (s.sh >= 0) s.shver = Tbl.tspivottbl[t.ver][s.shver];
    }
    bool issubface(const Triface &t) const { return Tets[t.tet].Sub[t.ver & 3] != None; }
    void stpivot(const Facet &s, Triface &t) const {
        Decode(Shells[s.sh].T[s.shver & 1], t);
        if (t.tet >= 0) t.ver = Tbl.stpivottbl[t.ver][s.shver];
    }
    bool isshtet(const Facet &s) const { return Shells[s.sh].T[s.shver & 1] != None; }
    void tsdissolve(const Triface &t) { Tets[t.tet].Sub[t.ver & 3] = None; }
    void stdissolve(const Facet &s) {
        Shells[s.sh].T[0] = None;
        Shells[s.sh].T[1] = None;
    }

    //=== Primitives between subfaces and segments ===

    void ssbond(const Facet &s, const Facet &edge) {
        Shells[s.sh].Seg[s.shver >> 1] = SEncode(edge);
        Shells[edge.sh].S[0] = SEncode(s);
    }
    void ssdissolve(const Facet &s) { Shells[s.sh].Seg[s.shver >> 1] = None; }
    void sspivot(const Facet &s, Facet &edge) const { SDecode(Shells[s.sh].Seg[s.shver >> 1], edge); }
    bool isshsubseg(const Facet &s) const { return Shells[s.sh].Seg[s.shver >> 1] != None; }

    //=== Primitives between tetrahedra and segments ===

    void tssbond1(const Triface &t, const Facet &s) { Tets[t.tet].Seg[ver2edge[t.ver]] = SEncode(s); }
    void sstbond1(const Facet &s, const Triface &t) { Shells[s.sh].T[0] = Encode(t); }
    void tssdissolve1(const Triface &t) { Tets[t.tet].Seg[ver2edge[t.ver]] = None; }
    void sstdissolve1(const Facet &s) { Shells[s.sh].T[0] = None; }
    void tsspivot1(const Triface &t, Facet &s) const { SDecode(Tets[t.tet].Seg[ver2edge[t.ver]], s); }
    bool issubseg(const Triface &t) const { return Tets[t.tet].Seg[ver2edge[t.ver]] != None; }
    void sstpivot1(const Facet &s, Triface &t) const { Decode(Shells[s.sh].T[0], t); }

    // Hang `s` on every tet around the edge, starting at `t`.
    void mark_seg_ring(const Facet &s, const Triface &t) {
        Triface spintet = t;
        do {
            tssbond1(spintet, s);
            fnextself(spintet);
        } while (spintet.tet != t.tet);
    }
    // The same, with the segment pointed back at `t`.
    void bond_seg_ring(const Facet &s, const Triface &t) {
        sstbond1(s, t);
        mark_seg_ring(s, t);
    }
    // Take `s` off every tet around it.
    void dissolve_seg_ring(const Facet &s) {
        Triface neightet;
        sstpivot1(s, neightet);
        Triface spintet = neightet;
        do {
            tssdissolve1(spintet);
            fnextself(spintet);
        } while (spintet.tet != neightet.tet);
    }

    //=== Primitives for points ===

    const dvec3 &P(int p) const { return Pts[p].Pos; }
    VertType pointtype(int pt) const { return VertType(Pts[pt].Flags >> 8); }
    void setpointtype(int pt, VertType v) { Pts[pt].Flags = (int(v) << 8) | (Pts[pt].Flags & 255); }
    void pinfect(int pt) { Pts[pt].Flags |= 1; }
    void puninfect(int pt) { Pts[pt].Flags &= ~1; }
    bool pinfected(int pt) const { return (Pts[pt].Flags & 1) != 0; }
    int point2tet(int pt) const { return Pts[pt].Tet; }
    void setpoint2tet(int pt, int enc) {
        if (pt != None) Pts[pt].Tet = enc;
    }
    int point2sh(int pt) const { return Pts[pt].Sh; }
    void setpoint2sh(int pt, int enc) { Pts[pt].Sh = enc; }
    int point2ppt(int pt) const { return Pts[pt].Parent; }
    void setpoint2ppt(int pt, int p) { Pts[pt].Parent = p; }
    double getpointinsradius(int pt) const { return Pts[pt].InsRadius; }
    void setpointinsradius(int pt, double v) { Pts[pt].InsRadius = v; }

    // Retire a Steiner point that was of type `vt`: mark it unused and give back the budget it took.
    void release_steiner(int steinerpt, VertType vt) {
        if (pointtype(steinerpt) != UnusedVertex) {
            setpointtype(steinerpt, UnusedVertex);
            ++UnuVerts;
        }
        if (vt != VolVertex) {
            if (vt == FreeSegVertex) --StSegRefCount;
            else if (vt == FreeFacetVertex) --StFacRefCount;
            else if (vt == FreeVolVertex) --StVolRefCount;
            if (SteinerLeft > 0) ++SteinerLeft;
        }
    }

    //=== Flips ===

    // Mark and stack a face that may need flipping.
    // The mark refuses a duplicate push and is cleared when the face is popped.
    void flippush(Triface *flipface) {
        if (!facemarked(*flipface)) {
            markface(*flipface);
            BadFace bf;
            bf.tt = *flipface;
            FlipStack.push_back(bf);
        }
    }

    // Arrays flipnm takes for the star of a link edge.
    // They are addressed by slot, so a nested call growing the table cannot invalidate an outer call's pointer.
    // A slot keeps its buffer between uses and grows only when a larger star asks for it, so a run of flips reuses the same memory.
    std::vector<std::vector<Triface>> FlipArrays;
    std::vector<int> FlipArraysFree;
    int flip_array_new(int n) {
        int slot;
        if (!FlipArraysFree.empty()) {
            slot = FlipArraysFree.back();
            FlipArraysFree.pop_back();
        } else {
            slot = int(FlipArrays.size());
            FlipArrays.emplace_back();
        }
        if (int(FlipArrays[slot].size()) < n) FlipArrays[slot].resize(size_t(n));
        return slot;
    }
    void flip_array_delete(int slot) { FlipArraysFree.push_back(slot); }

    // Ready a tet the flip is about to rebuild: version 11, no marks, and nothing attached.
    void reuse_tet(Triface &t) {
        t.ver = 11;
        Tet &e = Tets[t.tet];
        e.Marker = 0;
        for (int k = 0; k < 6; ++k) e.Seg[k] = None;
        for (int k = 0; k < 4; ++k) e.Sub[k] = None;
    }
    // Hang the segment `s` on the tet edge `t`, both ways.
    void bond_seg(const Triface &t, const Facet &s) {
        tssbond1(t, s);
        sstbond1(s, t);
    }
    // Move the subface carried by `from` onto `to`, turned to face the other way.
    void rebond_subface(const Triface &from, const Triface &to, FlipConstraints *fc) {
        if (!issubface(from)) return;
        Facet checksh;
        tspivot(from, checksh);
        sesymself(checksh);
        tsbond(to, checksh);
        if (fc->chkencflag & 2) enqueuesubface(BadSubfacs, &checksh);
    }

    // Replace the face [a,b,c] shared by [a,b,c,d] and [b,a,c,e] with the edge [d,e].
    // fliptets[0] and [1] are those two tets on entry.
    // The three new tets [e,d,a,b], [e,d,b,c] and [e,d,c,a] come back in [0], [1] and [2].
    void flip23(Triface *fliptets, int hullflag, FlipConstraints *fc) {
        Triface topcastets[3], botcastets[3];
        Triface newface, casface;
        int dummyflag = 0; // in {-1, 0, 1, 2}

        if (hullflag > 0) {
            if (oppo(fliptets[1]) == DummyPoint) {
                std::swap(fliptets[0], fliptets[1]);
                dummyflag = -1; // d is the dummy point
            } else if (org(fliptets[0]) == DummyPoint) {
                dummyflag = 1; // a is the dummy point
                enextself(fliptets[0]);
                eprevself(fliptets[1]);
            } else if (dest(fliptets[0]) == DummyPoint) {
                dummyflag = 2; // b is the dummy point
                eprevself(fliptets[0]);
                enextself(fliptets[1]);
            } else {
                dummyflag = 0; // either c or d may be the dummy point
            }
        }

        const int pa = org(fliptets[0]), pb = dest(fliptets[0]), pc = apex(fliptets[0]);
        const int pd = oppo(fliptets[0]), pe = oppo(fliptets[1]);
        ++Flip23Count;

        for (int i = 0; i < 3; ++i) {
            fnext(fliptets[0], topcastets[i]);
            enextself(fliptets[0]);
        }
        for (int i = 0; i < 3; ++i) {
            fnext(fliptets[1], botcastets[i]);
            eprevself(fliptets[1]);
        }

        reuse_tet(fliptets[0]);
        reuse_tet(fliptets[1]);
        maketetrahedron(&fliptets[2]);

        if (hullflag > 0) {
            if (pd != DummyPoint) {
                setvertices(fliptets[0], pe, pd, pa, pb);
                setvertices(fliptets[1], pe, pd, pb, pc);
                if (pc != DummyPoint) {
                    setvertices(fliptets[2], pe, pd, pc, pa);
                } else {
                    setvertices(fliptets[2], pd, pe, pa, pc);
                    esymself(fliptets[2]);
                }
            } else {
                setvertices(fliptets[0], pa, pb, pe, pd);
                setvertices(fliptets[1], pb, pc, pe, pd);
                setvertices(fliptets[2], pc, pa, pe, pd);
                for (int i = 0; i < 3; ++i) {
                    eprevesymself(fliptets[i]);
                    enextself(fliptets[i]);
                }
                HullSize += 2; // one hull tet went, three came
            }
        } else {
            setvertices(fliptets[0], pe, pd, pa, pb);
            setvertices(fliptets[1], pe, pd, pb, pc);
            setvertices(fliptets[2], pe, pd, pc, pa);
        }

        if (fc->remove_ndelaunay_edge) {
            double volneg[2]{}, volpos[3]{};
            if (pd != DummyPoint) {
                if (pc != DummyPoint) {
                    volpos[0] = tetprismvol(P(pe), P(pd), P(pa), P(pb));
                    volpos[1] = tetprismvol(P(pe), P(pd), P(pb), P(pc));
                    volpos[2] = tetprismvol(P(pe), P(pd), P(pc), P(pa));
                    volneg[0] = tetprismvol(P(pa), P(pb), P(pc), P(pd));
                    volneg[1] = tetprismvol(P(pb), P(pa), P(pc), P(pe));
                } else {
                    volpos[0] = tetprismvol(P(pe), P(pd), P(pa), P(pb));
                }
            } else {
                volneg[1] = tetprismvol(P(pb), P(pa), P(pc), P(pe));
            }
            fc->tetprism_vol_sum += volpos[0] + volpos[1] + volpos[2] - volneg[0] - volneg[1];
        }

        for (int i = 0; i < 3; ++i) {
            esym(fliptets[i], newface);
            bond(newface, fliptets[(i + 1) % 3]);
        }
        for (int i = 0; i < 3; ++i) {
            eorgoppo(fliptets[i], newface);
            bond(newface, topcastets[i]);
        }
        for (int i = 0; i < 3; ++i) {
            edestoppo(fliptets[i], newface);
            bond(newface, botcastets[i]);
        }

        if (CheckSubsegFlag) {
            Facet checkseg;
            for (int i = 0; i < 3; ++i) { // the middle three edges [a,b], [b,c], [c,a]
                if (issubseg(topcastets[i])) {
                    tsspivot1(topcastets[i], checkseg);
                    eorgoppo(fliptets[i], newface);
                    bond_seg(newface, checkseg);
                    if (fc->chkencflag & 1) enqueuesubface(BadSubsegs, &checkseg);
                }
            }
            for (int i = 0; i < 3; ++i) { // the top three edges [d,a], [d,b], [d,c]
                eprev(topcastets[i], casface);
                if (issubseg(casface)) {
                    tsspivot1(casface, checkseg);
                    enext(fliptets[i], newface);
                    bond_seg(newface, checkseg);
                    esym(fliptets[(i + 2) % 3], newface);
                    eprevself(newface);
                    bond_seg(newface, checkseg);
                    if (fc->chkencflag & 1) enqueuesubface(BadSubsegs, &checkseg);
                }
            }
            for (int i = 0; i < 3; ++i) { // the bottom three edges [a,e], [b,e], [c,e]
                enext(botcastets[i], casface);
                if (issubseg(casface)) {
                    tsspivot1(casface, checkseg);
                    eprev(fliptets[i], newface);
                    bond_seg(newface, checkseg);
                    esym(fliptets[(i + 2) % 3], newface);
                    enextself(newface);
                    bond_seg(newface, checkseg);
                    if (fc->chkencflag & 1) enqueuesubface(BadSubsegs, &checkseg);
                }
            }
        }

        if (CheckSubfaceFlag) {
            for (int i = 0; i < 3; ++i) {
                eorgoppo(fliptets[i], newface);
                rebond_subface(topcastets[i], newface, fc);
            }
            for (int i = 0; i < 3; ++i) {
                edestoppo(fliptets[i], newface);
                rebond_subface(botcastets[i], newface, fc);
            }
        }

        if (fc->chkencflag & 4) {
            for (int i = 0; i < 3; ++i) enqueuetetrahedron(&fliptets[i]);
        }

        setpoint2tet(pa, Encode2(fliptets[0].tet, 0));
        setpoint2tet(pb, Encode2(fliptets[0].tet, 0));
        setpoint2tet(pc, Encode2(fliptets[1].tet, 0));
        setpoint2tet(pd, Encode2(fliptets[0].tet, 0));
        setpoint2tet(pe, Encode2(fliptets[0].tet, 0));

        if (hullflag > 0 && dummyflag != 0) {
            // Put the points back where flipnm expects them.
            if (dummyflag == -1) {
                for (int i = 0; i < 3; ++i) esymself(fliptets[i]);
                std::swap(fliptets[1], fliptets[2]);
            } else if (dummyflag == 1) {
                newface = fliptets[0];
                fliptets[0] = fliptets[2];
                fliptets[2] = fliptets[1];
                fliptets[1] = newface;
            } else {
                newface = fliptets[0];
                fliptets[0] = fliptets[1];
                fliptets[1] = fliptets[2];
                fliptets[2] = newface;
            }
        }

        if (fc->enqflag > 0) {
            for (int i = 0; i < 3; ++i) {
                eprevesym(fliptets[i], newface);
                flippush(&newface);
            }
            if (fc->enqflag > 1) {
                for (int i = 0; i < 3; ++i) {
                    enextesym(fliptets[i], newface);
                    flippush(&newface);
                }
            }
        }
        RecentTet = fliptets[0];
    }

    // Replace the edge [e,d] shared by [e,d,a,b], [e,d,b,c], [e,d,c,a] with the face [a,b,c].
    // The two new tets [a,b,c,d] and [b,a,c,e] come back in [0] and [1].
    void flip32(Triface *fliptets, int hullflag, FlipConstraints *fc) {
        Triface topcastets[3], botcastets[3];
        Triface newface, casface;
        Facet flipshs[3];
        Facet checkseg;
        int dummyflag = 0;
        int spivotidx = -1, scount = 0;

        if (hullflag > 0) {
            if (org(fliptets[0]) == DummyPoint) {
                for (int i = 0; i < 3; ++i) esymself(fliptets[i]);
                std::swap(fliptets[1], fliptets[2]);
                dummyflag = -1; // e is the dummy point
            } else if (apex(fliptets[0]) == DummyPoint) {
                dummyflag = 1; // a is the dummy point
                newface = fliptets[0];
                fliptets[0] = fliptets[1];
                fliptets[1] = fliptets[2];
                fliptets[2] = newface;
            } else if (apex(fliptets[1]) == DummyPoint) {
                dummyflag = 2; // b is the dummy point
                newface = fliptets[0];
                fliptets[0] = fliptets[2];
                fliptets[2] = fliptets[1];
                fliptets[1] = newface;
            } else {
                dummyflag = 0; // either c or d may be the dummy point
            }
        }

        const int pa = apex(fliptets[0]), pb = apex(fliptets[1]), pc = apex(fliptets[2]);
        const int pd = dest(fliptets[0]), pe = org(fliptets[0]);
        ++Flip32Count;

        for (int i = 0; i < 3; ++i) {
            eorgoppo(fliptets[i], casface);
            fsym(casface, topcastets[i]);
        }
        for (int i = 0; i < 3; ++i) {
            edestoppo(fliptets[i], casface);
            fsym(casface, botcastets[i]);
        }

        if (CheckSubfaceFlag) {
            for (int i = 0; i < 3; ++i) {
                tspivot(fliptets[i], flipshs[i]);
                if (flipshs[i].sh != None) {
                    stdissolve(flipshs[i]);
                    ++scount;
                } else {
                    spivotidx = i;
                }
            }
        }

        reuse_tet(fliptets[0]);
        reuse_tet(fliptets[1]);
        tetrahedrondealloc(fliptets[2].tet);

        if (hullflag > 0) {
            if (pc != DummyPoint) {
                if (pd == DummyPoint) HullSize -= 2; // three hull tets went, one came
                setvertices(fliptets[0], pa, pb, pc, pd);
                setvertices(fliptets[1], pb, pa, pc, pe);
            } else {
                setvertices(fliptets[0], pb, pa, pd, pc);
                setvertices(fliptets[1], pa, pb, pe, pc);
                esymself(fliptets[0]);
                esymself(fliptets[1]);
            }
        } else {
            setvertices(fliptets[0], pa, pb, pc, pd);
            setvertices(fliptets[1], pb, pa, pc, pe);
        }

        if (fc->remove_ndelaunay_edge) {
            double volneg[3]{}, volpos[2]{};
            if (pc != DummyPoint) {
                if (pd != DummyPoint) {
                    volneg[0] = tetprismvol(P(pe), P(pd), P(pa), P(pb));
                    volneg[1] = tetprismvol(P(pe), P(pd), P(pb), P(pc));
                    volneg[2] = tetprismvol(P(pe), P(pd), P(pc), P(pa));
                    volpos[0] = tetprismvol(P(pa), P(pb), P(pc), P(pd));
                    volpos[1] = tetprismvol(P(pb), P(pa), P(pc), P(pe));
                } else {
                    volpos[1] = tetprismvol(P(pb), P(pa), P(pc), P(pe));
                }
            } else {
                volneg[0] = tetprismvol(P(pe), P(pd), P(pa), P(pb));
            }
            fc->tetprism_vol_sum += volpos[0] + volpos[1] - volneg[0] - volneg[1] - volneg[2];
        }

        bond(fliptets[0], fliptets[1]);
        for (int i = 0; i < 3; ++i) {
            esym(fliptets[0], newface);
            bond(newface, topcastets[i]);
            enextself(fliptets[0]);
        }
        for (int i = 0; i < 3; ++i) {
            esym(fliptets[1], newface);
            bond(newface, botcastets[i]);
            eprevself(fliptets[1]);
        }

        if (CheckSubsegFlag) {
            for (int i = 0; i < 3; ++i) { // edges a->b, b->c, c->a
                if (issubseg(topcastets[i])) {
                    tsspivot1(topcastets[i], checkseg);
                    bond_seg(fliptets[0], checkseg);
                    bond_seg(fliptets[1], checkseg);
                    if (fc->chkencflag & 1) enqueuesubface(BadSubsegs, &checkseg);
                }
                enextself(fliptets[0]);
                eprevself(fliptets[1]);
            }
            for (int i = 0; i < 3; ++i) { // edges b->d, c->d, a->d
                esym(fliptets[0], newface);
                eprevself(newface);
                enext(topcastets[i], casface);
                if (issubseg(casface)) {
                    tsspivot1(casface, checkseg);
                    bond_seg(newface, checkseg);
                    if (fc->chkencflag & 1) enqueuesubface(BadSubsegs, &checkseg);
                }
                enextself(fliptets[0]);
            }
            for (int i = 0; i < 3; ++i) { // edges b<-e, c<-e, a<-e
                esym(fliptets[1], newface);
                enextself(newface);
                eprev(botcastets[i], casface);
                if (issubseg(casface)) {
                    tsspivot1(casface, checkseg);
                    bond_seg(newface, checkseg);
                    if (fc->chkencflag & 1) enqueuesubface(BadSubsegs, &checkseg);
                }
                eprevself(fliptets[1]);
            }
        }

        if (CheckSubfaceFlag) {
            Facet checksh;
            for (int i = 0; i < 3; ++i) { // at edges [b,a], [c,b], [a,c]
                esym(fliptets[0], newface);
                rebond_subface(topcastets[i], newface, fc);
                enextself(fliptets[0]);
            }
            for (int i = 0; i < 3; ++i) { // at edges [a,b], [b,c], [c,a]
                esym(fliptets[1], newface);
                rebond_subface(botcastets[i], newface, fc);
                eprevself(fliptets[1]);
            }

            if (scount > 0) {
                // Two interior subfaces meet the vanishing edge: turn them over with a 2-2 flip and hang the results back on the two new tets.
                Facet flipfaces[2];
                flipfaces[0] = flipshs[(spivotidx + 1) % 3];
                flipfaces[1] = flipshs[(spivotidx + 2) % 3];
                sesymself(flipfaces[1]);
                flip22(flipfaces, 0, fc->chkencflag);
                topcastets[0] = fliptets[0];
                botcastets[0] = fliptets[1];
                for (int i = 0; i < ((spivotidx + 1) % 3); ++i) {
                    enextself(topcastets[0]);
                    eprevself(botcastets[0]);
                }
                esymself(topcastets[0]);
                sesymself(flipfaces[0]);
                tspivot(topcastets[0], checksh);
                if (checksh.sh == None) {
                    tsbond(topcastets[0], flipfaces[0]);
                    fsymself(topcastets[0]);
                    sesymself(flipfaces[0]);
                    tsbond(topcastets[0], flipfaces[0]);
                } else {
                    bail(2); // an invalid 2-to-2 flip
                }
                esymself(botcastets[0]);
                sesymself(flipfaces[1]);
                tspivot(botcastets[0], checksh);
                if (checksh.sh == None) {
                    tsbond(botcastets[0], flipfaces[1]);
                    fsymself(botcastets[0]);
                    sesymself(flipfaces[1]);
                    tsbond(botcastets[0], flipfaces[1]);
                } else {
                    bail(2); // an invalid 2-to-2 flip
                }
            }
        }

        if (fc->chkencflag & 4) {
            for (int i = 0; i < 2; ++i) enqueuetetrahedron(&fliptets[i]);
        }

        setpoint2tet(pa, Encode2(fliptets[0].tet, 0));
        setpoint2tet(pb, Encode2(fliptets[0].tet, 0));
        setpoint2tet(pc, Encode2(fliptets[0].tet, 0));
        setpoint2tet(pd, Encode2(fliptets[0].tet, 0));
        setpoint2tet(pe, Encode2(fliptets[1].tet, 0));

        if (hullflag > 0 && dummyflag != 0) {
            if (dummyflag == -1) {
                std::swap(fliptets[0], fliptets[1]);
            } else if (dummyflag == 1) {
                eprevself(fliptets[0]);
                enextself(fliptets[1]);
            } else {
                enextself(fliptets[0]);
                eprevself(fliptets[1]);
            }
        }

        if (fc->enqflag > 0) {
            enextesym(fliptets[0], newface);
            flippush(&newface);
            eprevesym(fliptets[1], newface);
            flippush(&newface);
            if (fc->enqflag > 1) {
                eprevesym(fliptets[0], newface);
                flippush(&newface);
                enextesym(fliptets[1], newface);
                flippush(&newface);
                esym(fliptets[0], newface);
                flippush(&newface);
                esym(fliptets[1], newface);
                flippush(&newface);
            }
        }
        RecentTet = fliptets[0];
    }

    // Remove the vertex p at the centre of [p,d,a,b], [p,d,b,c], [p,d,c,a] and [a,b,c,p].
    // [a,b,c,d] is left in fliptets[0].
    void flip41(Triface *fliptets, int hullflag, FlipConstraints *fc) {
        Triface topcastets[3], botcastet;
        Triface newface, neightet;
        Facet flipshs[4];
        int dummyflag = 0;
        int spivotidx = -1, scount = 0;

        const int pa = org(fliptets[3]), pb = dest(fliptets[3]), pc = apex(fliptets[3]);
        const int pd = dest(fliptets[0]), pp = org(fliptets[0]);
        ++Flip41Count;

        for (int i = 0; i < 3; ++i) {
            enext(fliptets[i], topcastets[i]);
            fnextself(topcastets[i]);
            enextself(topcastets[i]);
        }
        fsym(fliptets[3], botcastet);

        if (CheckSubfaceFlag) {
            for (int i = 0; i < 3; ++i) {
                fnext(fliptets[3], newface);
                tspivot(newface, flipshs[i]);
                if (flipshs[i].sh != None) {
                    spivotidx = i;
                    ++scount;
                }
                enextself(fliptets[3]);
            }
            if (scount > 0) {
                if (scount < 3) {
                    fsym(topcastets[spivotidx], neightet);
                    for (int i = 0; i < 3; ++i) {
                        esym(neightet, newface);
                        tspivot(newface, flipshs[i]);
                        eprevself(neightet);
                    }
                } else {
                    spivotidx = 3;
                }
            }
        }

        reuse_tet(fliptets[0]);
        for (int i = 1; i < 4; ++i) tetrahedrondealloc(fliptets[i].tet);

        if (pp != DummyPoint) {
            setpointtype(pp, UnusedVertex);
            ++UnuVerts;
        }

        if (hullflag > 0) {
            if (pa == DummyPoint) {
                setvertices(fliptets[0], pc, pb, pd, pa);
                esymself(fliptets[0]);
                eprevself(fliptets[0]);
                dummyflag = 1;
            } else if (pb == DummyPoint) {
                setvertices(fliptets[0], pa, pc, pd, pb);
                esymself(fliptets[0]);
                enextself(fliptets[0]);
                dummyflag = 2;
            } else if (pc == DummyPoint) {
                setvertices(fliptets[0], pb, pa, pd, pc);
                esymself(fliptets[0]);
                dummyflag = 3;
            } else if (pd == DummyPoint) {
                setvertices(fliptets[0], pa, pb, pc, pd);
                dummyflag = 4;
            } else {
                setvertices(fliptets[0], pa, pb, pc, pd);
                dummyflag = pp == DummyPoint ? -1 : 0;
            }
            if (dummyflag > 0) HullSize -= 2;
            else if (dummyflag < 0) HullSize -= 4;
        } else {
            setvertices(fliptets[0], pa, pb, pc, pd);
        }

        if (fc->remove_ndelaunay_edge) {
            double volneg[4]{}, volpos[1]{};
            if (dummyflag > 0) {
                if (pa == DummyPoint) volneg[1] = tetprismvol(P(pp), P(pd), P(pb), P(pc));
                else if (pb == DummyPoint) volneg[2] = tetprismvol(P(pp), P(pd), P(pc), P(pa));
                else if (pc == DummyPoint) volneg[0] = tetprismvol(P(pp), P(pd), P(pa), P(pb));
                else volneg[3] = tetprismvol(P(pa), P(pb), P(pc), P(pp));
            } else if (dummyflag < 0) {
                volpos[0] = tetprismvol(P(pa), P(pb), P(pc), P(pd));
            } else {
                volneg[0] = tetprismvol(P(pp), P(pd), P(pa), P(pb));
                volneg[1] = tetprismvol(P(pp), P(pd), P(pb), P(pc));
                volneg[2] = tetprismvol(P(pp), P(pd), P(pc), P(pa));
                volneg[3] = tetprismvol(P(pa), P(pb), P(pc), P(pp));
                volpos[0] = tetprismvol(P(pa), P(pb), P(pc), P(pd));
            }
            fc->tetprism_vol_sum += volpos[0] - volneg[0] - volneg[1] - volneg[2] - volneg[3];
        }

        for (int i = 0; i < 3; ++i) {
            esym(fliptets[0], newface);
            bond(newface, topcastets[i]);
            enextself(fliptets[0]);
        }
        bond(fliptets[0], botcastet);

        if (CheckSubsegFlag) {
            Facet checkseg;
            for (int i = 0; i < 3; ++i) {
                eprev(topcastets[i], newface);
                if (issubseg(newface)) {
                    tsspivot1(newface, checkseg);
                    esym(fliptets[0], newface);
                    enextself(newface);
                    bond_seg(newface, checkseg);
                    if (fc->chkencflag & 1) enqueuesubface(BadSubsegs, &checkseg);
                }
                enextself(fliptets[0]);
            }
            for (int i = 0; i < 3; ++i) {
                if (issubseg(topcastets[i])) {
                    tsspivot1(topcastets[i], checkseg);
                    bond_seg(fliptets[0], checkseg);
                    if (fc->chkencflag & 1) enqueuesubface(BadSubsegs, &checkseg);
                }
                enextself(fliptets[0]);
            }
        }

        if (CheckSubfaceFlag) {
            Facet checksh;
            for (int i = 0; i < 3; ++i) {
                esym(fliptets[0], newface);
                rebond_subface(topcastets[i], newface, fc);
                enextself(fliptets[0]);
            }
            rebond_subface(botcastet, fliptets[0], fc);

            if (spivotidx >= 0) {
                // Three subfaces meet at p: take p out of the surface with a 3-1 flip.
                for (int i = 0; i < 3; ++i) senext2self(flipshs[i]);
                flip31(flipshs, 0);
                for (int i = 0; i < 3; ++i) subfacedealloc(flipshs[i].sh);
                if (spivotidx < 3) {
                    tsbond(topcastets[spivotidx], flipshs[3]);
                    fsym(topcastets[spivotidx], newface);
                    sesym(flipshs[3], checksh);
                    tsbond(newface, checksh);
                } else {
                    tsbond(fliptets[0], flipshs[3]);
                    fsym(fliptets[0], newface);
                    sesym(flipshs[3], checksh);
                    tsbond(newface, checksh);
                }
            }
        }

        if (fc->chkencflag & 4) enqueuetetrahedron(&fliptets[0]);

        setpoint2tet(pa, Encode2(fliptets[0].tet, 0));
        setpoint2tet(pb, Encode2(fliptets[0].tet, 0));
        setpoint2tet(pc, Encode2(fliptets[0].tet, 0));
        setpoint2tet(pd, Encode2(fliptets[0].tet, 0));

        if (fc->enqflag > 0) {
            flippush(&fliptets[0]);
            if (fc->enqflag > 1) {
                for (int i = 0; i < 3; ++i) {
                    esym(fliptets[0], newface);
                    flippush(&newface);
                    enextself(fliptets[0]);
                }
            }
        }
        RecentTet = fliptets[0];
    }

    // The link level the flip search climbs to when no explicit one is set.
    int AutoFlipLinkLevel{1};

    static void sort_2pts(int p1, int p2, int ppt[2]) {
        if (p1 < p2) {
            ppt[0] = p1;
            ppt[1] = p2;
        } else {
            ppt[0] = p2;
            ppt[1] = p1;
        }
    }
    static void sort_3pts(int i1, int i2, int i3, int ppt[3]) {
        if (i1 < i2) {
            if (i1 < i3) {
                ppt[0] = i1;
                if (i2 < i3) {
                    ppt[1] = i2;
                    ppt[2] = i3;
                } else {
                    ppt[1] = i3;
                    ppt[2] = i2;
                }
            } else {
                ppt[0] = i3;
                ppt[1] = i1;
                ppt[2] = i2;
            }
        } else {
            if (i2 < i3) {
                ppt[0] = i2;
                if (i1 < i3) {
                    ppt[1] = i1;
                    ppt[2] = i3;
                } else {
                    ppt[1] = i3;
                    ppt[2] = i1;
                }
            } else {
                ppt[0] = i3;
                ppt[1] = i2;
                ppt[2] = i1;
            }
        }
    }

    double CosCollinearAngTol{0};

    bool is_collinear_at(int mid, int left, int right) const {
        const dvec3 v1 = P(left) - P(mid), v2 = P(right) - P(mid);
        const double L1 = glm::length(v1), L2 = glm::length(v2);
        return dot(v1, v2) / (L1 * L2) < CosCollinearAngTol;
    }

    bool is_segment(int p1, int p2) const {
        if (pointtype(p1) == RidgeVertex) {
            if (pointtype(p2) == RidgeVertex) {
                for (int i = IdxSegmentRidgeVertexList[p1]; i < IdxSegmentRidgeVertexList[p1 + 1]; ++i) {
                    if (SegmentRidgeVertexList[i] == p2) return true;
                }
            } else if (pointtype(p2) == FreeSegVertex) {
                Facet parentseg;
                SDecode(point2sh(p2), parentseg);
                const int segidx = getfacetindex(parentseg);
                if (SegmentEndpointsList[segidx * 2] == p1 || SegmentEndpointsList[segidx * 2 + 1] == p1) return true;
            }
        } else if (pointtype(p1) == FreeSegVertex) {
            if (pointtype(p2) == RidgeVertex) {
                Facet parentseg;
                SDecode(point2sh(p1), parentseg);
                const int segidx = getfacetindex(parentseg);
                if (SegmentEndpointsList[segidx * 2] == p2 || SegmentEndpointsList[segidx * 2 + 1] == p2) return true;
            } else if (pointtype(p2) == FreeSegVertex) {
                Facet parentseg1, parentseg2;
                SDecode(point2sh(p1), parentseg1);
                SDecode(point2sh(p2), parentseg2);
                if (getfacetindex(parentseg1) == getfacetindex(parentseg2)) return true;
            }
        }
        return false;
    }

    // Refuse a 2-3 flip that would make a face whose three vertices sit on one segment or on two nearly collinear ones.
    bool valid_constrained_f23(Triface &checktet, int pd, int pe) {
        Triface spintet;
        Facet checkseg1, checkseg2;
        for (int k = 0; k < 3; ++k) {
            const int checkpt = org(checktet);
            esym(checktet, spintet);
            enextself(spintet); // [x, d], x = a, b, c
            tsspivot1(spintet, checkseg1);
            bool isseg = checkseg1.sh != None;
            if (!isseg && BoundaryRecoveryFlag) isseg = is_segment(checkpt, pd);
            if (isseg) {
                fsym(checktet, spintet);
                esymself(spintet);
                eprevself(spintet);
                tsspivot1(spintet, checkseg2);
                isseg = checkseg2.sh != None;
                if (!isseg && BoundaryRecoveryFlag) isseg = is_segment(checkpt, pe);
                if (isseg) {
                    if (pointtype(checkpt) == FreeSegVertex) return false;
                    if (checkpt != DummyPoint && pe != DummyPoint && pd != DummyPoint) {
                        if (is_collinear_at(checkpt, pe, pd)) return false;
                    }
                }
            }
            enextself(checktet);
        }
        return true;
    }

    // The same rule for the one face a 3-2 flip creates.
    bool valid_constrained_f32(Triface *abtets, int pa, int pb) {
        (void)pa;
        (void)pb;
        Triface spintet;
        Facet checksegs[3];
        for (int k = 0; k < 3; ++k) {
            enext(abtets[k], spintet);
            esymself(spintet);
            eprevself(spintet); // [c,d], [d,e], [e,c]
            tsspivot1(spintet, checksegs[k]);
            // A temporary segment, the kind recoversubfaces plants, does not count.
            if (checksegs[k].sh != None && smarktest2ed(checksegs[k])) checksegs[k].sh = None;
        }
        for (int k = 0; k < 3; ++k) {
            const int chkpt = apex(abtets[k]);
            const int leftpt = apex(abtets[(k + 2) % 3]);
            const int rightpt = apex(abtets[(k + 1) % 3]);
            bool isseg = checksegs[k].sh != None;
            if (!isseg && BoundaryRecoveryFlag) isseg = is_segment(chkpt, rightpt);
            if (isseg) {
                isseg = checksegs[(k + 2) % 3].sh != None;
                if (!isseg && BoundaryRecoveryFlag) isseg = is_segment(chkpt, leftpt);
                if (isseg) {
                    if (pointtype(chkpt) == FreeSegVertex) return false;
                    if (chkpt != DummyPoint && leftpt != DummyPoint && rightpt != DummyPoint) {
                        if (is_collinear_at(chkpt, leftpt, rightpt)) return false;
                    }
                }
            }
        }
        return true;
    }

    // The call back boundary recovery and mesh improvement install, deciding whether one elementary flip may be taken.
    int checkflipeligibility(int fliptype, int pa, int pb, int pc, int pd, int pe, int level, int edgepivot, FlipConstraints *fc) {
        int tmppts[3];
        int types[2], poss[4];
        int rejflag = 0;

        if (fc->seg[0] != None) {
            if (fliptype == 1) {
                // A 2-3 flip: [a,b,c] becomes [e,d,a], [e,d,b], [e,d,c].
                tmppts[0] = pa;
                tmppts[1] = pb;
                tmppts[2] = pc;
                for (int i = 0; i < 3 && !rejflag; ++i) {
                    if (tmppts[i] == DummyPoint) continue;
                    const int intflag = tri_edge_test(P(pe), P(pd), P(tmppts[i]), P(fc->seg[0]), P(fc->seg[1]), nullptr, 1, types, poss);
                    if (intflag == 2) {
                        const int dir = types[0];
                        if (dir == AcrossFace) rejflag = 1;
                        else if (dir == AcrossEdge) {
                            if (poss[0] == 0) rejflag = 1; // the new edge [e,d] cuts the segment
                        } else if (dir == AcrossVert || dir == TouchEdge || dir == TouchFace) rejflag = 1;
                    } else if (intflag == 4) {
                        const int dir = types[0];
                        if (dir == AcrossEdge) {
                            if (poss[0] == 0) rejflag = 1;
                        } else if (dir == AcrossFace) {
                            bail(2);
                        } else if (dir == AcrossVert || dir == TouchEdge || dir == TouchFace) rejflag = 1;
                    }
                }
            } else if (fliptype == 2) {
                // A 3-2 flip: [e,d,a], [e,d,b], [e,d,c] become [a,b,c].
                if (pc != DummyPoint) {
                    const int intflag = tri_edge_test(P(pa), P(pb), P(pc), P(fc->seg[0]), P(fc->seg[1]), nullptr, 1, types, poss);
                    if (intflag == 2) {
                        if (types[0] == AcrossFace) rejflag = 1;
                    } else if (intflag == 4) {
                        if (types[0] == AcrossEdge) rejflag = 1;
                    }
                }
            }
        }

        if (fc->fac[0] != None && !rejflag) {
            if (fliptype == 1) {
                const int intflag = tri_edge_test(P(fc->fac[0]), P(fc->fac[1]), P(fc->fac[2]), P(pe), P(pd), nullptr, 1, types, poss);
                if (intflag == 2) {
                    if (types[0] == AcrossFace || types[0] == AcrossEdge) rejflag = 1;
                } else if (intflag == 4) {
                    for (int i = 0; i < 2 && !rejflag; ++i) {
                        if (types[i] == AcrossFace || types[i] == AcrossEdge) rejflag = 1;
                    }
                }
            }
        }

        if (fc->remvert != None && !rejflag) {
            if (fliptype == 1) {
                if (pd == fc->remvert || pe == fc->remvert) rejflag = 1;
            }
        }

        if (fc->remove_large_angle && !rejflag) {
            BadFace bf;
            // Both tets a flip produces have to improve, and the pair's worst values are what the flip reports.
            const auto accept_pair = [&](int a1, int b1, int c1, int d1, int a2, int b2, int c2, int d2) {
                if (!improves_dihedral(a1, b1, c1, d1, fc->cosdihed_in, &bf)) return false;
                const double cosmaxd = bf.cent[0], asp = bf.key;
                if (!improves_dihedral(a2, b2, c2, d2, fc->cosdihed_in, &bf)) return false;
                fc->cosdihed_out = std::min({fc->cosdihed_out, cosmaxd, bf.cent[0]});
                fc->max_asp_out = std::max({fc->max_asp_out, asp, bf.key});
                return true;
            };
            if (fliptype == 1) {
                // Only [e,d,b,c] and [e,d,c,a] need checking: [e,d,a,b] is flipped again below.
                if (pc != DummyPoint && pe != DummyPoint && pd != DummyPoint) {
                    if (!accept_pair(pe, pd, pb, pc, pe, pd, pc, pa)) rejflag = 1;
                }
            } else if (fliptype == 2 && pa != DummyPoint && pb != DummyPoint && pc != DummyPoint) {
                if (level == 0) {
                    if (!accept_pair(pa, pb, pc, pd, pb, pa, pc, pe)) rejflag = 1;
                } else if (edgepivot == 1 ? !improves_dihedral(pb, pa, pc, pe, fc->cosdihed_in, &bf) : !improves_dihedral(pa, pb, pc, pd, fc->cosdihed_in, &bf)) {
                    rejflag = 1;
                } else {
                    fc->cosdihed_out = std::min(fc->cosdihed_out, bf.cent[0]);
                    fc->max_asp_out = std::max(fc->max_asp_out, bf.key);
                }
            }
        }
        return rejflag;
    }

    // The two tets about [a,b] that go back into the star once a nested edge removal has finished, as [a,b,e,c] and [a,b,c,d].
    // The pivot decides which way each one is turned.
    void make_abtets_pair(const Triface *tmpabtets, int edgepivot, Triface *fliptets) {
        fliptets[0] = tmpabtets[1];
        if (edgepivot == 1) enextself(fliptets[0]);
        else eprevself(fliptets[0]);
        esymself(fliptets[0]); // [a,b,e,c]
        fliptets[1] = tmpabtets[0];
        esymself(fliptets[1]);
        if (edgepivot == 1) eprevself(fliptets[1]);
        else enextself(fliptets[1]); // [a,b,c,d]
    }

    // Remove the edge [a,b] through a sequence of elementary flips.
    // abtets holds the n tets around it in a counterclockwise cycle about a->b.
    // The return is the size of the current star, 2 when the edge is gone.
    int flipnm(Triface *abtets, int n, int level, int abedgepivot, FlipConstraints *fc) {
        Triface fliptets[3], spintet, flipedge;
        int hullflag, hulledgeflag;
        int reducflag, rejflag;
        int reflexlinkedgecount;
        int edgepivot;
        int n1, nn;

        const int pa = org(abtets[0]), pb = dest(abtets[0]);
        int pc, pd, pe, pf;
        double ori;

        if (n > 3) {
            reflexlinkedgecount = 0;
            for (int i = 0; i < n; ++i) {
                // Let the face of abtets[i] be [a,b,c].
                if (CheckSubfaceFlag && issubface(abtets[i])) continue;
                // Leave a face alone when it belongs to two stars at once.
                if (elemcounter(abtets[i]) > 1 || elemcounter(abtets[(i - 1 + n) % n]) > 1) continue;

                pc = apex(abtets[i]);
                pd = apex(abtets[(i + 1) % n]);
                pe = apex(abtets[(i - 1 + n) % n]);
                if (pd == DummyPoint || pe == DummyPoint) continue; // [a,b,c] is a hull face

                reducflag = 0;
                hullflag = pc == DummyPoint;
                hulledgeflag = 0;
                ori = 1.0;
                if (hullflag == 0) {
                    ori = orient3d(P(pb), P(pc), P(pd), P(pe)); // is [b,c] locally convex?
                    if (ori > 0) {
                        ori = orient3d(P(pc), P(pa), P(pd), P(pe)); // is [c,a] locally convex?
                        if (ori > 0) {
                            ori = orient3d(P(pa), P(pb), P(pd), P(pe)); // is [a,b] convex or flat?
                            if (ori > 0) {
                                reducflag = 1; // a 2-3 flip: [a,b,c] becomes [e,d]
                            } else if (ori == 0) {
                                if (n == 4) {
                                    // The flat tet goes straight out with a 3-2 flip.
                                    reducflag = 1;
                                    pf = apex(abtets[(i + 2) % n]);
                                    hulledgeflag = pf == DummyPoint;
                                }
                            }
                        }
                    }
                    if (!reducflag) ++reflexlinkedgecount;
                } else {
                    // c is the dummy point.
                    if (n == 4) {
                        pf = apex(abtets[(i + 2) % n]);
                        ori = orient3d(P(pd), P(pe), P(pf), P(pa));
                        if (ori < 0) {
                            ori = orient3d(P(pe), P(pd), P(pf), P(pb));
                            if (ori < 0) {
                                reducflag = 1; // a 4-4 flip: [a,b] becomes [e,d]
                                ori = 0; // signal it as the coplanar case
                                hulledgeflag = 1;
                            }
                        }
                    }
                }

                if (reducflag && NonConvex && hulledgeflag) {
                    // The hull edge [e,d] is about to be created: make sure it does not exist.
                    if (getedge(pe, pd, &spintet)) reducflag = 0;
                }
                if (reducflag) {
                    Triface checktet = abtets[i];
                    if (!valid_constrained_f23(checktet, pd, pe)) reducflag = 0;
                }

                if (reducflag) {
                    rejflag = 0;
                    if (fc->checkflipeligibility) {
                        rejflag = checkflipeligibility(1, pa, pb, pc, pd, pe, level, abedgepivot, fc);
                    }
                    if (!rejflag) {
                        fliptets[0] = abtets[i];
                        fsym(fliptets[0], fliptets[1]); // abtets[i-1]
                        flip23(fliptets, hullflag, fc);

                        // Shrink abtets while keeping its order.
                        // The tet [a,b,e,d] stays in the star and takes the [i-1] slot; everything above [i] shifts down.
                        edestoppoself(fliptets[0]); // [a,b,e,d]
                        increaseelemcounter(fliptets[0]);
                        abtets[(i - 1 + n) % n] = fliptets[0];
                        for (int j = i; j < n - 1; ++j) abtets[j] = abtets[j + 1];
                        // The freed last entry remembers the vertex c and the position of this flip, which is what lets flipnm_post undo it.
                        abtets[n - 1].tet = pc;
                        abtets[n - 1].ver = (1 << 4) | (i << 6);

                        if (fc->collectnewtets) {
                            for (int j = 1; j < 3; ++j) CaveTetList.push_back(fliptets[j]);
                        }

                        nn = flipnm(abtets, n - 1, level, abedgepivot, fc);

                        if (nn == 2) return nn;
                        if (fc->unflip || ori == 0) {
                            // Undo the 2-3 flip.
                            // `ori == 0` means it made a degenerate tet, which has to come out whether or not the caller asked for an unflip.
                            fliptets[0] = abtets[(i - 1 + (n - 1)) % (n - 1)]; // [a,b,e,d]
                            edestoppoself(fliptets[0]); // [e,d,a,b]
                            fnext(fliptets[0], fliptets[1]); // [e,d,b,c]
                            fnext(fliptets[1], fliptets[2]); // [e,d,c,a]
                            flip32(fliptets, hullflag, fc);
                            for (int j = 0; j < 2; ++j) increaseelemcounter(fliptets[j]);
                            for (int j = n - 2; j >= i; --j) abtets[j + 1] = abtets[j];
                            esym(fliptets[1], abtets[(i - 1 + n) % n]); // [a,b,e,c]
                            abtets[i] = fliptets[0]; // [a,b,c,d]
                            ++nn;
                            if (fc->collectnewtets) CaveTetList.resize(CaveTetList.size() - 2);
                        }
                        if (!fc->unflip) return nn;
                        // With unflip set the mesh is back where it started: keep searching.
                    }
                }
            }

            if (reflexlinkedgecount > 0) {
                // There are reflex edges in the link of [a,b]: try to flip one away.
                if ((FlipLinkLevel < 0 && level < AutoFlipLinkLevel) || (FlipLinkLevel >= 0 && level < FlipLinkLevel)) {
                    for (int i = 0; i < n; ++i) {
                        if (elemcounter(abtets[i]) > 1 || elemcounter(abtets[(i - 1 + n) % n]) > 1) continue;
                        pc = apex(abtets[i]);
                        if (pc == DummyPoint) continue; // [a,b] is a hull edge
                        pd = apex(abtets[(i + 1) % n]);
                        pe = apex(abtets[(i - 1 + n) % n]);
                        if (pd == DummyPoint || pe == DummyPoint) continue;

                        edgepivot = 0;
                        ori = orient3d(P(pb), P(pc), P(pd), P(pe)); // is [b,c] convex or flat?
                        if (ori <= 0) {
                            enext(abtets[i], flipedge); // [b,c,a,d]
                            edgepivot = 1;
                        }
                        if (!edgepivot) {
                            ori = orient3d(P(pc), P(pa), P(pd), P(pe)); // is [c,a] convex or flat?
                            if (ori <= 0) {
                                eprev(abtets[i], flipedge); // [c,a,b,d]
                                edgepivot = 2;
                            }
                        }
                        if (!edgepivot) continue;

                        if (CheckSubsegFlag && issubseg(flipedge)) {
                            if (fc->collectencsegflag) {
                                Facet checkseg;
                                tsspivot1(flipedge, checkseg);
                                if (!sinfected(checkseg)) {
                                    sinfect(checkseg);
                                    CaveEncSegList.push_back(checkseg);
                                }
                            }
                            continue;
                        }

                        esymself(flipedge);
                        int subface_count = 0;
                        n1 = 0;
                        int starsum = 0;
                        spintet = flipedge;
                        while (true) {
                            if (issubface(spintet)) ++subface_count;
                            ++n1;
                            starsum += elemcounter(spintet);
                            fnextself(spintet);
                            if (spintet.tet == flipedge.tet) break;
                        }
                        if (n1 < 3) bail(2); // only reachable with inverted elements present
                        if (starsum > 2) continue; // the two stars overlap
                        if (fc->noflip_in_surface && subface_count > 0) continue;
                        if (FlipStarSize > 0 && n1 > FlipStarSize) continue;

                        const int arrayslot = flip_array_new(n1);
                        Triface *tmpabtets = FlipArrays[arrayslot].data();
                        spintet = flipedge;
                        for (int j = 0; j < n1; ++j) {
                            tmpabtets[j] = spintet;
                            increaseelemcounter(tmpabtets[j]);
                            fnextself(spintet);
                        }

                        nn = flipnm(tmpabtets, n1, level + 1, edgepivot, fc);

                        if (nn == 2) {
                            // The link edge went, so the star of [a,b] is one smaller.
                            if (edgepivot == 1) {
                                spintet = tmpabtets[0]; // [d,a,e,b]
                                enextself(spintet);
                                esymself(spintet);
                                enextself(spintet); // [a,b,e,d]
                            } else {
                                spintet = tmpabtets[1]; // [b,d,e,a]
                                eprevself(spintet);
                                esymself(spintet);
                                eprevself(spintet); // [a,b,e,d]
                            }
                            increaseelemcounter(spintet);
                            abtets[(i - 1 + n) % n] = spintet;
                            for (int j = i; j < n - 1; ++j) abtets[j] = abtets[j + 1];
                            // Record the flip so it can be undone: star array, pivot, position and star size pack into the last entry.
                            abtets[n - 1].tet = arrayslot;
                            abtets[n - 1].ver = edgepivot | (1 << 5) | (i << 6) | (n1 << 19);
                            tmpabtets[0].tet = pc;
                            tmpabtets[0].ver = 1 << 5;

                            nn = flipnm(abtets, n - 1, level, abedgepivot, fc);

                            if (nn == 2) return nn;
                            if (fc->unflip) {
                                // Put the link edge back. abtets[(i-1)%(n-1)] is [a,b,e,d], the tet the link flip created, and it is still in the star.
                                if (edgepivot == 1) {
                                    tmpabtets[0] = abtets[((i - 1) + (n - 1)) % (n - 1)]; // [a,b,e,d]
                                    eprevself(tmpabtets[0]);
                                    esymself(tmpabtets[0]);
                                    eprevself(tmpabtets[0]); // [d,a,e,b]
                                    fsym(tmpabtets[0], tmpabtets[1]); // [a,d,e,c]
                                } else {
                                    tmpabtets[1] = abtets[((i - 1) + (n - 1)) % (n - 1)]; // [a,b,e,d]
                                    enextself(tmpabtets[1]);
                                    esymself(tmpabtets[1]);
                                    enextself(tmpabtets[1]); // [b,d,e,a]
                                    fsym(tmpabtets[1], tmpabtets[0]); // [d,b,e,c]
                                }
                                flipnm_post(tmpabtets, n1, 2, edgepivot, fc);
                                for (int j = n - 2; j >= i; --j) abtets[j + 1] = abtets[j];
                                make_abtets_pair(tmpabtets, edgepivot, fliptets);
                                for (int j = 0; j < 2; ++j) increaseelemcounter(fliptets[j]);
                                abtets[(i - 1 + n) % n] = fliptets[0];
                                abtets[i] = fliptets[1];
                                ++nn;
                                flip_array_delete(arrayslot);
                            }
                            if (!fc->unflip) return nn;
                        } else {
                            if (!fc->unflip) flipnm_post(tmpabtets, n1, nn, edgepivot, fc);
                            for (int j = 0; j < nn; ++j) decreaseelemcounter(tmpabtets[j]);
                            flip_array_delete(arrayslot);
                        }
                    }
                }
            }
        } else {
            // n == 3: try a 3-2 flip.
            // Rearrange so that any dummy apex sits at e.
            hullflag = 0;
            if (apex(abtets[0]) == DummyPoint) {
                pc = apex(abtets[1]);
                pd = apex(abtets[2]);
                pe = apex(abtets[0]);
                hullflag = 1;
            } else if (apex(abtets[1]) == DummyPoint) {
                pc = apex(abtets[2]);
                pd = apex(abtets[0]);
                pe = apex(abtets[1]);
                hullflag = 2;
            } else {
                pc = apex(abtets[0]);
                pd = apex(abtets[1]);
                pe = apex(abtets[2]);
                hullflag = apex(abtets[2]) == DummyPoint ? 3 : 0;
            }

            reducflag = 0;
            rejflag = 0;
            if (hullflag == 0) {
                // Both new tets [d,c,e,a] and [c,d,e,b] must be valid.
                ori = orient3d(P(pd), P(pc), P(pe), P(pa));
                if (ori < 0) {
                    ori = orient3d(P(pc), P(pd), P(pe), P(pb));
                    if (ori < 0) reducflag = 1;
                }
            } else {
                // [a,b] is a hull edge, which happens in the middle of a 4-4 flip.
                if (!NonConvex) {
                    ori = orient3d(P(pa), P(pb), P(pc), P(pd));
                    if (ori == 0) reducflag = 1;
                } else {
                    reducflag = 1;
                }
                if (reducflag == 1) {
                    // Find an interior apex of [c,d] to test against.
                    // The one whose tet has the biggest volume decides the sign as far from zero as possible.
                    int searchpt = None, chkpt;
                    double bigvol = 0.0, ori1, ori2;
                    fliptets[0] = abtets[hullflag % 3]; // [a,b,c,d]
                    eorgoppoself(fliptets[0]); // [d,c,b,a]
                    spintet = fliptets[0];
                    while (true) {
                        fnextself(spintet);
                        chkpt = oppo(spintet);
                        if (chkpt == pb) break;
                        if (chkpt != DummyPoint && apex(spintet) != DummyPoint) {
                            ori = -orient3d(P(pd), P(pc), P(apex(spintet)), P(chkpt));
                            if (ori > bigvol) {
                                bigvol = ori;
                                searchpt = chkpt;
                            }
                        }
                    }
                    if (searchpt != None) {
                        ori1 = orient3d(P(pd), P(pc), P(searchpt), P(pa));
                        ori2 = orient3d(P(pd), P(pc), P(searchpt), P(pb));
                        if (ori1 * ori2 >= 0.0) reducflag = 0;
                        else {
                            ori1 = orient3d(P(pa), P(pb), P(searchpt), P(pc));
                            ori2 = orient3d(P(pa), P(pb), P(searchpt), P(pd));
                            if (ori1 * ori2 >= 0.0) reducflag = 0;
                        }
                    } else {
                        reducflag = 0;
                    }
                }
            }

            if (reducflag) {
                if (CheckSubfaceFlag) {
                    // The edge may be flipped only when 0 or 2 subfaces meet it.
                    // With two, the 3-2 flip carries a 2-2 flip in the surface mesh with it.
                    nn = 0;
                    edgepivot = -1;
                    for (int j = 0; j < 3; ++j) {
                        if (issubface(abtets[j])) ++nn;
                        else edgepivot = j;
                    }
                    if (nn == 1) {
                        // One subface only, which happens during recovery while the neighbour is not yet back.
                        // Leave the edge alone until the neighbour is back.
                        rejflag = 1;
                    } else if (nn == 2) {
                        eorgoppo(abtets[(edgepivot + 1) % 3], spintet); // [q,p,b,a]
                        if (issubface(spintet)) rejflag = 1;
                        else {
                            esymself(spintet);
                            if (issubface(spintet)) rejflag = 1;
                        }
                    }
                }
                if (!rejflag && !valid_constrained_f32(abtets, pa, pb)) rejflag = 1;
                if (!rejflag && fc->checkflipeligibility) {
                    // The eligibility check names the face to be flipped as [a,b,c] and the new edge as [e,d], so a and b swap places here.
                    rejflag = checkflipeligibility(2, pc, pd, pe, pb, pa, level, abedgepivot, fc);
                }
                if (!rejflag) {
                    flip32(abtets, hullflag, fc);
                    if (fc->remove_ndelaunay_edge && level == 0) {
                        // This is the edge we set out to remove: keep the flip only if the objective actually improved.
                        if (fc->tetprism_vol_sum >= 0.0 || std::abs(fc->tetprism_vol_sum) < fc->bak_tetprism_vol) {
                            flip23(abtets, hullflag, fc);
                            for (int j = 0; j < 3; ++j) increaseelemcounter(abtets[j]);
                            return 3;
                        }
                    }
                    if (fc->collectnewtets) {
                        if (level == 0) {
                            for (int j = 0; j < 2; ++j) CaveTetList.push_back(abtets[j]);
                        } else {
                            // Only one of the two is outside the reduced star of the parent edge.
                            CaveTetList.push_back(abedgepivot == 1 ? abtets[1] : abtets[0]);
                        }
                    }
                    return 2;
                }
            }
        }
        return n;
    }

    // Walk back over the flips flipnm recorded, freeing their arrays.
    // With unflip set each one is undone, so the mesh returns to its original state.
    int flipnm_post(Triface *abtets, int n, int nn, int abedgepivot, FlipConstraints *fc) {
        Triface fliptets[3];
        int fliptype, edgepivot, t, n1;

        if (nn == 2) {
            if (fc->unflip) {
                flip23(abtets, 1, fc);
                if (fc->collectnewtets) {
                    CaveTetList.resize(CaveTetList.size() - (abedgepivot == 0 ? 2 : 1));
                }
            }
            ++nn; // the initial star size was 3
        }

        for (int i = nn; i < n; ++i) {
            fliptype = (abtets[i].ver >> 4) & 3;
            if (fliptype == 1) {
                t = abtets[i].ver >> 6;
                if (fc->unflip) {
                    fliptets[0] = abtets[((t - 1) + i) % i]; // [a,b,e,d]
                    eprevself(fliptets[0]);
                    esymself(fliptets[0]);
                    enextself(fliptets[0]); // [e,d,a,b]
                    fnext(fliptets[0], fliptets[1]); // [e,d,b,c]
                    fnext(fliptets[1], fliptets[2]); // [e,d,c,a]
                    flip32(fliptets, 1, fc);
                    for (int j = i - 1; j >= t; --j) abtets[j + 1] = abtets[j];
                    esym(fliptets[1], abtets[((t - 1) + (i + 1)) % (i + 1)]); // [a,b,e,c]
                    abtets[t] = fliptets[0]; // [a,b,c,d]
                    if (fc->collectnewtets) CaveTetList.resize(CaveTetList.size() - 2);
                }
            } else if (fliptype == 2) {
                const int arrayslot = abtets[i].tet;
                Triface *tmpabtets = FlipArrays[arrayslot].data();
                n1 = (abtets[i].ver >> 19) & 8191;
                edgepivot = abtets[i].ver & 3;
                t = (abtets[i].ver >> 6) & 8191;
                if (fc->unflip) {
                    if (edgepivot == 1) {
                        tmpabtets[0] = abtets[(t - 1 + i) % i]; // [a,b,e,d]
                        eprevself(tmpabtets[0]);
                        esymself(tmpabtets[0]);
                        eprevself(tmpabtets[0]); // [d,a,e,b]
                        fsym(tmpabtets[0], tmpabtets[1]); // [a,d,e,c]
                    } else {
                        tmpabtets[1] = abtets[(t - 1 + i) % i]; // [a,b,e,d]
                        enextself(tmpabtets[1]);
                        esymself(tmpabtets[1]);
                        enextself(tmpabtets[1]); // [b,d,e,a]
                        fsym(tmpabtets[1], tmpabtets[0]); // [d,b,e,c]
                    }
                    flipnm_post(tmpabtets, n1, 2, edgepivot, fc);
                    for (int j = i - 1; j >= t; --j) abtets[j + 1] = abtets[j];
                    make_abtets_pair(tmpabtets, edgepivot, fliptets);
                    abtets[((t - 1) + (i + 1)) % (i + 1)] = fliptets[0];
                    abtets[t] = fliptets[1];
                } else {
                    flipnm_post(tmpabtets, n1, 2, edgepivot, fc);
                }
                flip_array_delete(arrayslot);
            }
        }
        return 1;
    }

    void point2tetorg(int pa, Triface &searchtet) const {
        Decode(point2tet(pa), searchtet);
        const Tet &e = Tets[searchtet.tet];
        searchtet.ver = e.V[0] == pa ? 11 : (e.V[1] == pa ? 3 : (e.V[2] == pa ? 7 : 0));
    }

    void flipshpush(Facet *flipedge) {
        FlipStack.push_back({.ss = *flipedge, .forg = sorg(*flipedge), .fdest = sdest(*flipedge)});
    }

    // Note what the boundary edge `bdedge` is attached to: the subface on its far side, the last
    // subface before the ring comes back to it, and any segment it carries.
    void take_bdedge(const Facet &bdedge, Facet &outface, Facet &inface, Facet &bdseg) const {
        spivot(bdedge, outface);
        inface = outface;
        sspivot(bdedge, bdseg);
        if (outface.sh != None && isshsubseg(bdedge)) {
            Facet checkface;
            spivot(inface, checkface);
            while (checkface.sh != bdedge.sh) {
                inface = checkface;
                spivot(inface, checkface);
            }
        }
    }
    // Hang `bdedge` back where `take_bdedge` found it.
    void put_bdedge(Facet &bdedge, const Facet &outface, const Facet &inface, Facet &bdseg) {
        if (bdseg.sh != None) {
            bdseg.shver = 0;
            if (sorg(bdedge) != sorg(bdseg)) sesymself(bdedge);
        }
        sbond1(bdedge, outface);
        sbond1(inface, bdedge);
    }

    // Turn the edge [a,b] shared by the subfaces [a,b,c] and [b,a,d] into the edge [c,d].
    void flip22(Facet *flipfaces, int flipflag, int chkencflag) {
        Facet bdedges[4], outfaces[4], infaces[4], bdsegs[4];

        const int pa = sorg(flipfaces[0]), pb = sdest(flipfaces[0]);
        const int pc = sapex(flipfaces[0]), pd = sapex(flipfaces[1]);
        if (sorg(flipfaces[1]) != pb) sesymself(flipfaces[1]);
        ++Flip22Count;

        senext(flipfaces[0], bdedges[0]);
        senext2(flipfaces[0], bdedges[1]);
        senext(flipfaces[1], bdedges[2]);
        senext2(flipfaces[1], bdedges[3]);

        for (int i = 0; i < 4; ++i) take_bdedge(bdedges[i], outfaces[i], infaces[i], bdsegs[i]);

        setshvertices(flipfaces[0], pc, pd, pb);
        setshvertices(flipfaces[1], pd, pc, pa);

        if (pointtype(pa) == FreeFacetVertex) setpoint2sh(pa, SEncode(flipfaces[1]));
        if (pointtype(pb) == FreeFacetVertex) setpoint2sh(pb, SEncode(flipfaces[0]));
        if (pointtype(pc) == FreeFacetVertex) setpoint2sh(pc, SEncode(flipfaces[0]));
        if (pointtype(pd) == FreeFacetVertex) setpoint2sh(pd, SEncode(flipfaces[0]));

        for (int i = 0; i < 4; ++i) {
            const int k = (3 + i) % 4;
            if (outfaces[k].sh != None) put_bdedge(bdedges[i], outfaces[k], infaces[k], bdsegs[k]);
            else sdissolve(bdedges[i]);
            if (bdsegs[k].sh != None) {
                ssbond(bdedges[i], bdsegs[k]);
                if (chkencflag & 1) enqueuesubface(BadSubsegs, &bdsegs[k]);
            } else {
                ssdissolve(bdedges[i]);
            }
        }

        if (chkencflag & 2) {
            for (int i = 0; i < 2; ++i) enqueuesubface(BadSubfacs, &flipfaces[i]);
        }
        RecentSh = flipfaces[0];
        if (flipflag) {
            for (int i = 0; i < 4; ++i) flipshpush(&bdedges[i]);
        }
    }

    // Replace the three subfaces [p,a,b], [p,b,c], [p,c,a] with the one subface [a,b,c], returned in flipfaces[3].
    // The three old ones stay allocated for the caller.
    void flip31(Facet *flipfaces, int flipflag) {
        Facet bdedges[3], outfaces[3], infaces[3], bdsegs[3];
        const int pa = sdest(flipfaces[0]), pb = sdest(flipfaces[1]), pc = sdest(flipfaces[2]);
        ++Flip31Count;

        for (int i = 0; i < 3; ++i) {
            senext(flipfaces[i], bdedges[i]);
            take_bdedge(bdedges[i], outfaces[i], infaces[i], bdsegs[i]);
        }

        makesubface(&flipfaces[3]);
        setshvertices(flipfaces[3], pa, pb, pc);
        setshellmark(flipfaces[3], shellmark(flipfaces[0]));
        if (CheckConstraints) setareabound(flipfaces[3], areabound(flipfaces[0]));
        if (UseInsertRadius) setfacetindex(flipfaces[3], getfacetindex(flipfaces[0]));

        if (pointtype(pa) == FreeFacetVertex) setpoint2sh(pa, SEncode(flipfaces[3]));
        if (pointtype(pb) == FreeFacetVertex) setpoint2sh(pb, SEncode(flipfaces[3]));
        if (pointtype(pc) == FreeFacetVertex) setpoint2sh(pc, SEncode(flipfaces[3]));

        bdedges[0] = flipfaces[3];
        senext(flipfaces[3], bdedges[1]);
        senext2(flipfaces[3], bdedges[2]);

        for (int i = 0; i < 3; ++i) {
            if (outfaces[i].sh != None) put_bdedge(bdedges[i], outfaces[i], infaces[i], bdsegs[i]);
            if (bdsegs[i].sh != None) ssbond(bdedges[i], bdsegs[i]);
        }
        RecentSh = flipfaces[3];
        if (flipflag) {
            for (int i = 0; i < 3; ++i) flipshpush(&bdedges[i]);
        }
    }

    // Flip the stacked surface edges that are not locally Delaunay.
    long lawsonflip() {
        Facet flipfaces[2];
        long flipcount = 0;
        while (!FlipStack.empty()) {
            const BadFace popface = FlipStack.back();
            FlipStack.pop_back();
            flipfaces[0] = popface.ss;
            const int pa = popface.forg, pb = popface.fdest;
            if (isdeadsh(flipfaces[0])) continue;
            if (sorg(flipfaces[0]) != pa || sdest(flipfaces[0]) != pb) continue;
            if (isshsubseg(flipfaces[0])) continue;
            spivot(flipfaces[0], flipfaces[1]);
            if (flipfaces[1].sh == None) continue; // a hull edge
            const int pc = sapex(flipfaces[0]), pd = sapex(flipfaces[1]);
            if (incircle3d(P(pa), P(pb), P(pc), P(pd)) < 0) {
                flip22(flipfaces, 1, 0);
                ++flipcount;
            }
        }
        return flipcount;
    }

    // Try to take the edge out.
    // Returns 2 when it is gone, otherwise the current size of its star.
    int removeedgebyflips(Triface *flipedge, FlipConstraints *fc) {
        Triface spintet;
        if (CheckSubsegFlag && issubseg(*flipedge)) {
            if (fc->collectencsegflag) {
                Facet checkseg;
                tsspivot1(*flipedge, checkseg);
                if (!sinfected(checkseg)) {
                    sinfect(checkseg);
                    CaveEncSegList.push_back(checkseg);
                }
            }
            return 0;
        }

        int subface_count = 0, n = 0;
        spintet = *flipedge;
        while (true) {
            if (issubface(spintet)) ++subface_count;
            ++n;
            fnextself(spintet);
            if (spintet.tet == flipedge->tet) break;
        }
        if (n < 3) bail(2); // only reachable when the mesh holds inverted tets
        if (fc->noflip_in_surface && subface_count > 0) return 0;
        if (FlipStarSize > 0 && n > FlipStarSize) return 0;

        const int arrayslot = flip_array_new(n);
        Triface *abtets = FlipArrays[arrayslot].data();
        spintet = *flipedge;
        for (int i = 0; i < n; ++i) {
            abtets[i] = spintet;
            setelemcounter(abtets[i], 1);
            fnextself(spintet);
        }

        const int nn = flipnm(abtets, n, 0, 0, fc);

        if (nn > 2) {
            for (int i = 0; i < nn; ++i) setelemcounter(abtets[i], 0);
            *flipedge = abtets[0];
        }
        const int bakunflip = fc->unflip;
        fc->unflip = 0;
        flipnm_post(abtets, n, nn, 0, fc);
        fc->unflip = bakunflip;
        flip_array_delete(arrayslot);
        return nn;
    }

    // Take a face out, by a 2-3 flip when it is convex and otherwise by removing one of its edges.
    int removefacebyflips(Triface *flipface, FlipConstraints *fc) {
        Triface fliptets[3], flipedge;
        int reducflag = 0;

        fliptets[0] = *flipface;
        fsym(*flipface, fliptets[1]);
        const int pa = org(fliptets[0]), pb = dest(fliptets[0]), pc = apex(fliptets[0]);
        const int pd = oppo(fliptets[0]), pe = oppo(fliptets[1]);

        double ori = orient3d(P(pa), P(pb), P(pd), P(pe));
        if (ori > 0) {
            ori = orient3d(P(pb), P(pc), P(pd), P(pe));
            if (ori > 0) {
                ori = orient3d(P(pc), P(pa), P(pd), P(pe));
                if (ori > 0) reducflag = 1;
                else eprev(*flipface, flipedge); // [c,a]
            } else {
                enext(*flipface, flipedge); // [b,c]
            }
        } else {
            flipedge = *flipface; // [a,b]
        }

        if (reducflag) {
            Triface checkface = fliptets[0];
            if (!valid_constrained_f23(checkface, pd, pe)) return 0;
            flip23(fliptets, 0, fc);
            return 1;
        }
        return removeedgebyflips(&flipedge, fc) == 2 ? 1 : 0;
    }

    // The two segments meeting at a Steiner point that sits on a segment, turned so the point is the
    // destination of the left one and the origin of the right one.
    void segments_at(int steinerpt, Facet &leftseg, Facet &rightseg) {
        SDecode(point2sh(steinerpt), leftseg);
        leftseg.shver = 0;
        if (sdest(leftseg) == steinerpt) {
            senext(leftseg, rightseg);
            spivotself(rightseg);
            rightseg.shver = 0;
        } else {
            rightseg = leftseg;
            senext2(rightseg, leftseg);
            spivotself(leftseg);
            leftseg.shver = 0;
        }
    }

    // Cross to the face opposite searchtet's edge, collecting a subface found on the way.
    // False when that subface stops the walk, which it does unless the whole star is wanted.
    bool cross_link_face(const Triface &searchtet, Triface &neightet, int fullstar, std::vector<Facet> *shlist) {
        esym(searchtet, neightet);
        if (!issubface(neightet)) return true;
        if (shlist != nullptr) {
            Facet checksh;
            tspivot(neightet, checksh);
            if (!sinfected(checksh)) {
                sinfect(checksh);
                shlist->push_back(checksh);
            }
        }
        return fullstar != 0;
    }

    // Collect the tets, the link vertices and the subfaces around a vertex.
    // With fullstar clear the walk stops at subfaces.
    int getvertexstar(int fullstar, int searchpt, std::vector<Triface> *tetlist, std::vector<int> *vertlist, std::vector<Facet> *shlist) {
        Triface searchtet, neightet;
        point2tetorg(searchpt, searchtet);
        enextesymself(searchtet); // the link face of the vertex
        infect(searchtet);
        tetlist->push_back(searchtet);
        if (vertlist != nullptr) {
            const int j = searchtet.ver & 3;
            for (int i = 1; i < 4; ++i) {
                const int pt = Tets[searchtet.tet].V[(j + i) % 4];
                pinfect(pt);
                vertlist->push_back(pt);
            }
        }

        if (cross_link_face(searchtet, neightet, fullstar, shlist)) {
            fsymself(neightet);
            esymself(neightet);
            infect(neightet);
            tetlist->push_back(neightet);
            if (vertlist != nullptr) {
                const int pt = apex(neightet);
                pinfect(pt);
                vertlist->push_back(pt);
            }
        }

        for (size_t i = 0; i < tetlist->size(); ++i) {
            searchtet = (*tetlist)[i];
            for (int j = 0; j < 2; ++j) {
                enextself(searchtet);
                if (cross_link_face(searchtet, neightet, fullstar, shlist)) {
                    fsymself(neightet);
                    if (!infected(neightet)) {
                        esymself(neightet);
                        infect(neightet);
                        tetlist->push_back(neightet);
                        if (vertlist != nullptr) {
                            const int pt = apex(neightet);
                            if (!pinfected(pt)) {
                                pinfect(pt);
                                vertlist->push_back(pt);
                            }
                        }
                    }
                }
            }
        }

        for (auto &t : *tetlist) uninfect(t);
        if (vertlist != nullptr) {
            for (int pt : *vertlist) puninfect(pt);
        }
        if (shlist != nullptr) {
            for (auto &s : *shlist) suninfect(s);
        }
        return int(tetlist->size());
    }

    // Find a tet holding the edge (e1, e2), searching the link faces of e1 when the direction walk does not land on it.
    int getedge(int e1, int e2, Triface *tedge) {
        Triface searchtet, neightet;
        if (e1 == None || e2 == None) return 0;
        if (pointtype(e1) == UnusedVertex || pointtype(e2) == UnusedVertex) return 0;

        if (!isdeadtet(*tedge)) {
            if (org(*tedge) == e1) {
                if (dest(*tedge) == e2) return 1;
            } else if (org(*tedge) == e2) {
                if (dest(*tedge) == e1) {
                    esymself(*tedge);
                    return 1;
                }
            }
        }

        point2tetorg(e1, *tedge);
        finddirection(tedge, e2);
        if (dest(*tedge) == e2) return 1;
        point2tetorg(e2, *tedge);
        finddirection(tedge, e1);
        if (dest(*tedge) == e1) {
            esymself(*tedge);
            return 1;
        }

        point2tetorg(e1, searchtet);
        enextesymself(searchtet);
        std::vector<Triface> &tetlist = CaveBdryList;

        for (int i = 0; i < 3; ++i) {
            if (apex(searchtet) == e2) {
                eorgoppo(searchtet, *tedge);
                return 1;
            }
            enextself(searchtet);
        }
        fnext(searchtet, neightet);
        esymself(neightet);
        if (apex(neightet) == e2) {
            eorgoppo(neightet, *tedge);
            return 1;
        }

        infect(searchtet);
        tetlist.push_back(searchtet);
        infect(neightet);
        tetlist.push_back(neightet);

        int done = 0;
        for (size_t i = 0; i < tetlist.size() && !done; ++i) {
            searchtet = tetlist[i];
            for (int j = 0; j < 2 && !done; ++j) {
                enextself(searchtet);
                fnext(searchtet, neightet);
                if (!infected(neightet)) {
                    esymself(neightet);
                    if (apex(neightet) == e2) {
                        eorgoppo(neightet, *tedge);
                        done = 1;
                    } else {
                        infect(neightet);
                        tetlist.push_back(neightet);
                    }
                }
            }
        }
        for (auto &t : tetlist) uninfect(t);
        tetlist.clear();
        return done;
    }

    //=== Point insertion ===

    std::vector<BadFace> EncSegList, EncShList;
    std::vector<int> CaveOldTetOnly;

    // A random number in [0, choices).
    unsigned long randomnation(unsigned int choices) {
        if (choices == 0) return 0;
        if (choices >= 714025l) {
            unsigned long newrandom = (RandomSeed * 1366l + 150889l) % 714025l;
            RandomSeed = (newrandom * 1366l + 150889l) % 714025l;
            newrandom = newrandom * (choices / 714025l) + RandomSeed;
            return newrandom >= choices ? newrandom - choices : newrandom;
        }
        RandomSeed = (RandomSeed * 1366l + 150889l) % 714025l;
        return RandomSeed % choices;
    }

    // Put the cavity back and clear the working lists.
    void insertpoint_abort(Facet *splitseg, InsertFlags *ivf) {
        for (auto &t : CaveOldTetList) {
            uninfect(t);
            unmarktest(t);
        }
        for (auto &t : CaveBdryList) unmarktest(t);
        CaveTetList.clear();
        CaveBdryList.clear();
        CaveOldTetList.clear();
        CaveTetSegList.clear();
        CaveTetShList.clear();
        if (ivf->splitbdflag) {
            if (splitseg != nullptr && splitseg->sh != None) sunmarktest(*splitseg);
            for (auto &s : CaveShList) sunmarktest(s);
            CaveShList.clear();
            CaveSegShList.clear();
        }
    }

    // The Bowyer-Watson insertion, with all the cavity validation and boundary bookkeeping the constrained phases need.
    // The three faces of a tet's link, turned to the pivot convention and put on the cavity boundary.
    void push_link_faces(Triface &t) {
        Triface link;
        for (int j = 0; j < 3; ++j) {
            esym(t, link);
            link.ver = epivot[link.ver];
            CaveBdryList.push_back(link);
            enextself(t);
        }
    }

    // Cut a tet out of the cavity, putting it and the three faces of its link on the boundary.
    void cut_from_cavity(Triface &t, int &cutcount) {
        uninfect(t);
        unmarktest(t);
        ++cutcount;
        t.ver = epivot[t.ver];
        CaveBdryList.push_back(t);
        push_link_faces(t);
    }

    // The tets p conflicts with, and on a segment or subface the sub-cavity in the surface mesh alongside them.
    // False when p needs no insertion, being already a vertex or sitting on an encroached subface.
    bool build_initial_cavity(int loc, Triface *searchtet, Facet *splitsh, Facet *splitseg, InsertFlags *ivf) {
        Triface neightet, spintet;
        if (loc == Outside || loc == InTetrahedron) {
            ++Flip14Count;
            for (int i = 0; i < 4; ++i) {
                Decode(Tets[searchtet->tet].N[i], neightet);
                neightet.ver = epivot[neightet.ver];
                CaveBdryList.push_back(neightet);
            }
            infect(*searchtet);
            CaveOldTetList.push_back(*searchtet);
        } else if (loc == OnFace) {
            ++Flip26Count;
            int j = searchtet->ver & 3;
            for (int i = 1; i < 4; ++i) {
                Decode(Tets[searchtet->tet].N[(j + i) % 4], neightet);
                neightet.ver = epivot[neightet.ver];
                CaveBdryList.push_back(neightet);
            }
            Decode(Tets[searchtet->tet].N[j], spintet);
            j = spintet.ver & 3;
            for (int i = 1; i < 4; ++i) {
                Decode(Tets[spintet.tet].N[(j + i) % 4], neightet);
                neightet.ver = epivot[neightet.ver];
                CaveBdryList.push_back(neightet);
            }
            infect(spintet);
            CaveOldTetList.push_back(spintet);
            infect(*searchtet);
            CaveOldTetList.push_back(*searchtet);
            if (ivf->splitbdflag && splitsh != nullptr && splitsh->sh != None) {
                smarktest(*splitsh);
                CaveShList.push_back(*splitsh);
            }
        } else if (loc == OnEdge) {
            ++FlipN2nCount;
            spintet = *searchtet;
            while (true) {
                eorgoppo(spintet, neightet);
                Decode(Tets[neightet.tet].N[neightet.ver & 3], neightet);
                neightet.ver = epivot[neightet.ver];
                CaveBdryList.push_back(neightet);
                edestoppo(spintet, neightet);
                Decode(Tets[neightet.tet].N[neightet.ver & 3], neightet);
                neightet.ver = epivot[neightet.ver];
                CaveBdryList.push_back(neightet);
                infect(spintet);
                CaveOldTetList.push_back(spintet);
                fnextself(spintet);
                if (spintet.tet == searchtet->tet) break;
            }
            if (ivf->splitbdflag) {
                if (splitseg != nullptr && splitseg->sh != None) {
                    smarktest(*splitseg);
                    splitseg->shver = 0;
                    spivot(*splitseg, *splitsh);
                }
                if (splitsh != nullptr && splitsh->sh != None) {
                    const int pa = sorg(*splitsh);
                    Facet neighsh = *splitsh;
                    while (true) {
                        if (sorg(neighsh) != pa) sesymself(neighsh);
                        smarktest(neighsh);
                        CaveShList.push_back(neighsh);
                        CaveSegShList.push_back(neighsh);
                        spivotself(neighsh);
                        if (neighsh.sh == splitsh->sh) break;
                        if (neighsh.sh == None) break;
                    }
                }
            }
        } else if (loc == InStar) {
            // The star is already in CaveOldTetList and infected.
            if (CaveBdryList.empty()) {
                for (size_t i = 0; i < CaveOldTetList.size(); ++i) {
                    const Triface cavetet = CaveOldTetList[i];
                    for (int j = 0; j < 4; ++j) {
                        Decode(Tets[cavetet.tet].N[j], neightet);
                        if (!infected(neightet)) {
                            neightet.ver = epivot[neightet.ver];
                            CaveBdryList.push_back(neightet);
                        }
                    }
                }
            }
        } else if (loc == OnVertex) {
            return false; // the point is already there
        } else if (loc == EncSubface) {
            ivf->iloc = EncSubface;
            return false;
        } else {
            bail(2); // unknown case
        }
        return true;
    }

    // True when p sits on top of an existing vertex or inside its protecting ball, which leaves that vertex as the search handle.
    bool reject_near_vertex(int insertpt, int loc, Triface *searchtet, Facet *splitseg, InsertFlags *ivf) {
        if (!(Plc || Quality) || loc == InStar) return false;
        Triface spintet;
        if (loc == Outside) {
            for (int i = 0; i < 3; ++i) CaveTetVertList.push_back(Tets[searchtet->tet].V[i]);
        } else if (loc == InTetrahedron) {
            for (int i = 0; i < 4; ++i) CaveTetVertList.push_back(Tets[searchtet->tet].V[i]);
        } else if (loc == OnFace) {
            for (int i = 0; i < 3; ++i) CaveTetVertList.push_back(Tets[searchtet->tet].V[i]);
            if (Tets[searchtet->tet].V[3] != DummyPoint) CaveTetVertList.push_back(Tets[searchtet->tet].V[3]);
            fsym(*searchtet, spintet);
            if (oppo(spintet) != DummyPoint) CaveTetVertList.push_back(oppo(spintet));
        } else if (loc == OnEdge) {
            spintet = *searchtet;
            CaveTetVertList.push_back(org(spintet));
            CaveTetVertList.push_back(dest(spintet));
            while (true) {
                if (apex(spintet) != DummyPoint) CaveTetVertList.push_back(apex(spintet));
                fnextself(spintet);
                if (spintet.tet == searchtet->tet) break;
            }
        }

        const int rejptflag = ivf->rejflag & 4;
        int nearpt = None;
        for (const int parypt : CaveTetVertList) {
            const double rd = distance(P(parypt), P(insertpt));
            if (rd < MinEdgeLength) {
                if (!create_a_shorter_edge(insertpt, parypt) && !ivf->ignore_near_vertex) {
                    nearpt = parypt;
                    loc = NearVertex;
                    break;
                }
            }
            if (ivf->check_insert_radius) {
                const double ins_radius = getpointinsradius(parypt);
                if (ins_radius > 0.0 && rd < ins_radius) {
                    if (!create_a_shorter_edge(insertpt, parypt)) {
                        nearpt = parypt;
                        loc = EncVertex;
                        break;
                    }
                }
            }
            if (rejptflag && rd < 0.5 * Pts[parypt].Mtr) {
                nearpt = parypt;
                loc = EncVertex;
                break;
            }
        }
        CaveTetVertList.clear();
        if (nearpt != None) {
            point2tetorg(nearpt, *searchtet);
            insertpoint_abort(splitseg, ivf);
            ivf->iloc = loc;
            return true;
        }
        return false;
    }

    // Grow C(p) by the Bowyer-Watson rule: take in every neighbouring tet whose circumsphere holds p.
    void grow_cavity_bw(int insertpt, InsertFlags *ivf) {
        if (!ivf->bowywat) return;
        Triface neightet, neineitet;
        bool enqflag;
        std::swap(CaveTetList, CaveBdryList);
        CaveBdryList.clear();
        for (size_t i = 0; i < CaveTetList.size(); ++i) {
            Triface cavetet = CaveTetList[i];
            if (infected(cavetet)) continue;
            enqflag = false;
            if (!marktested(cavetet)) {
                const Tet &e = Tets[cavetet.tet];
                if (e.V[3] != DummyPoint) {
                    enqflag = insphere_s(e.V[0], e.V[1], e.V[2], e.V[3], insertpt) < 0.0;
                } else {
                    // A hull face. On a convex mesh a visible one grows the hull and a coplanar one is settled by the
                    // tet behind it. On a non-convex mesh the face is a subface, so the tet behind it always settles it,
                    // and validation decides later whether the face survives.
                    bool decide_behind = NonConvex;
                    if (!NonConvex) {
                        const double ori = orient3d(P(e.V[0]), P(e.V[1]), P(e.V[2]), P(insertpt));
                        if (ori < 0) enqflag = true;
                        else decide_behind = ori == 0.0;
                    }
                    if (decide_behind) {
                        Decode(Tets[cavetet.tet].N[3], neineitet);
                        if (!infected(neineitet)) {
                            if (!marktested(neineitet)) {
                                const Tet &f = Tets[neineitet.tet];
                                enqflag = insphere_s(f.V[0], f.V[1], f.V[2], f.V[3], insertpt) < 0.0;
                            }
                        } else {
                            enqflag = true;
                        }
                    }
                }
                marktest(cavetet);
            }

            if (enqflag) {
                const int k = cavetet.ver & 3;
                for (int j = 1; j < 4; ++j) {
                    Decode(Tets[cavetet.tet].N[(j + k) % 4], neightet);
                    CaveTetList.push_back(neightet);
                }
                infect(cavetet);
                CaveOldTetList.push_back(cavetet);
            } else {
                cavetet.ver = epivot[cavetet.ver];
                CaveBdryList.push_back(cavetet);
            }
        }
        CaveTetList.clear();
    }

    // Every segment and subface of a tet in C(p), collected so the new tets can be hung on them.
    // False when p would encroach one of them, which refuses the point.
    bool collect_cavity_boundary(int insertpt, Facet *splitseg, InsertFlags *ivf) {
        if (CheckSubsegFlag) {
            Facet checkseg;
            for (const auto &cavetet : CaveOldTetList) {
                for (int j = 0; j < 6; ++j) {
                    const int enc = Tets[cavetet.tet].Seg[j];
                    if (enc != None) {
                        SDecode(enc, checkseg);
                        if (!sinfected(checkseg)) {
                            sinfect(checkseg);
                            CaveTetSegList.push_back(checkseg);
                        }
                    }
                }
            }
            for (auto &s : CaveTetSegList) suninfect(s);

            if (ivf->rejflag & 1) {
                for (const auto &paryseg1 : CaveTetSegList) {
                    const int p0 = Shells[paryseg1.sh].V[0], p1 = Shells[paryseg1.sh].V[1];
                    if (check_encroachment(p0, p1, insertpt)) {
                        EncSegList.push_back({.ss = paryseg1, .forg = sorg(paryseg1), .fdest = sdest(paryseg1)});
                    }
                }
                if (!EncSegList.empty()) {
                    insertpoint_abort(splitseg, ivf);
                    ivf->iloc = EncSegment;
                    return false;
                }
            }
        }

        if (CheckSubfaceFlag) {
            Facet checksh;
            for (const auto &cavetet : CaveOldTetList) {
                for (int j = 0; j < 4; ++j) {
                    const int enc = Tets[cavetet.tet].Sub[j];
                    if (enc != None) {
                        SDecode(enc, checksh);
                        if (!sinfected(checksh)) {
                            sinfect(checksh);
                            CaveTetShList.push_back(checksh);
                        }
                    }
                }
            }
            for (auto &s : CaveTetShList) suninfect(s);

            if (ivf->rejflag & 2) {
                dvec3 ccent;
                double radius;
                for (auto &parysh : CaveTetShList) {
                    if (get_subface_ccent(&parysh, ccent)) {
                        int encpt = insertpt;
                        if (check_enc_subface(&parysh, &encpt, ccent, &radius)) {
                            EncShList.push_back({
                                .ss = parysh,
                                .key = radius,
                                .cent = {ccent.x, ccent.y, ccent.z},
                                .forg = sorg(parysh),
                                .fdest = sdest(parysh),
                                .fapex = sapex(parysh),
                                .noppo = None,
                            });
                        }
                    }
                }
                if (!EncShList.empty()) {
                    insertpoint_abort(splitseg, ivf);
                    ivf->iloc = EncSubface;
                    return false;
                }
            }
        }
        return true;
    }

    // Grow the sub-cavity out from the subfaces p meets, without crossing a segment.
    void grow_subcavity(int insertpt, InsertFlags *ivf) {
        if (!ivf->splitbdflag) return;
        Triface neightet;
        Facet checksh, neighsh;
        for (size_t i = 0; i < CaveShList.size(); ++i) {
            checksh = CaveShList[i];
            for (int j = 0; j < 3; ++j) {
                if (!isshsubseg(checksh)) {
                    spivot(checksh, neighsh);
                    if (neighsh.sh != None && !smarktested(neighsh)) {
                        stpivot(neighsh, neightet);
                        if (neightet.tet != None && infected(neightet)) {
                            fsymself(neightet);
                            if (infected(neightet)) {
                                const double sign = incircle3d(P(sorg(neighsh)), P(sdest(neighsh)), P(sapex(neighsh)), P(insertpt));
                                if (sign < 0) {
                                    smarktest(neighsh);
                                    CaveShList.push_back(neighsh);
                                }
                            }
                        }
                    }
                }
                senextself(checksh);
            }
        }
    }

    // Cut C(p) back until it is star-shaped from p, keeping any segment or subface not being split on its boundary.
    // False when that empties the cavity, or cuts away a subface that has to be split.
    bool carve_cavity_star_shaped(int insertpt, int loc, Facet *splitsh, Facet *splitseg, InsertFlags *ivf) {
        if (!ivf->validflag) return true;
        Triface neightet, spintet, neineitet;
        Facet neighsh;
        double ori;
        bool enqflag;
        int cutcount = 0;

        if (ivf->respectbdflag) {
            for (auto &parysh : CaveTetShList) {
                stpivot(parysh, neightet);
                if (neightet.tet == None || !infected(neightet)) continue;
                fsymself(neightet);
                if (!infected(neightet)) continue;
                if (smarktested(parysh)) continue;
                if (oppo(neightet) != DummyPoint) fsymself(neightet);
                if (oppo(neightet) != DummyPoint) {
                    ori = orient3d(P(org(neightet)), P(dest(neightet)), P(apex(neightet)), P(insertpt));
                    if (ori < 0) {
                        fsymself(neightet);
                        ori = -ori;
                    }
                } else {
                    ori = 1; // a hull tet has to be cut
                }
                if (ori >= 0) {
                    cut_from_cavity(neightet, cutcount);
                }
            }

            for (auto &paryseg : CaveTetSegList) {
                if (smarktested(paryseg)) continue;
                sstpivot1(paryseg, neightet);
                spintet = neightet;
                while (true) {
                    if (!infected(spintet)) break;
                    fnextself(spintet);
                    if (spintet.tet == neightet.tet) break;
                }
                if (!infected(spintet)) continue;
                // Find a tet at this segment with neither of its two faces visible from p.
                const int pa = org(neightet), pb = dest(neightet);
                spintet = neightet;
                int found = 0;
                while (true) {
                    int pc = apex(spintet);
                    if (pc != DummyPoint) {
                        ori = orient3d(P(pa), P(pb), P(pc), P(insertpt));
                        if (ori >= 0) {
                            esym(spintet, neineitet);
                            pc = apex(neineitet);
                            if (pc != DummyPoint) {
                                ori = orient3d(P(pb), P(pa), P(pc), P(insertpt));
                                if (ori >= 0) {
                                    found = 1;
                                    break;
                                }
                            }
                        }
                    }
                    fnextself(spintet);
                    if (spintet.tet == neightet.tet) break;
                }
                if (!found) bail(2); // no such face
                neightet = spintet;
                cut_from_cavity(neightet, cutcount);
            }
        }

        for (size_t i = 0; i < CaveBdryList.size(); ++i) {
            Triface cavetet = CaveBdryList[i];
            fsym(cavetet, neightet);
            if (infected(neightet)) {
                if (apex(cavetet) != DummyPoint) {
                    if (oppo(neightet) != DummyPoint) {
                        const int pa = org(cavetet), pb = dest(cavetet), pc = apex(cavetet);
                        if (issubface(neightet)) {
                            const double volume = orient3dfast(P(pa), P(pb), P(pc), P(insertpt));
                            const double scale = distance(P(pa), P(pb)) * distance(P(pb), P(pc)) * distance(P(pc), P(pa));
                            ori = std::abs(volume) / scale < Epsilon ? 0.0 : orient3d(P(pa), P(pb), P(pc), P(insertpt));
                        } else {
                            ori = orient3d(P(pa), P(pb), P(pc), P(insertpt));
                        }
                        enqflag = ori > 0; // a coplanar face is cut too
                    } else {
                        // A hull face whose interior tet was already cut away.
                        enqflag = false;
                    }
                } else {
                    enqflag = true; // a hull edge
                }
                if (enqflag) {
                    CaveTetList.push_back(cavetet);
                } else {
                    uninfect(neightet);
                    unmarktest(neightet);
                    ++cutcount;
                    push_link_faces(neightet);
                    unmarktest(cavetet);
                }
            } else {
                unmarktest(cavetet);
            }
        }

        if (cutcount > 0) {
            CaveBdryList.clear();
            for (const auto &cavetet : CaveTetList) {
                fsym(cavetet, neightet);
                if (infected(neightet)) CaveBdryList.push_back(cavetet);
                else unmarktest(cavetet);
            }
            CaveTetList.clear();
            for (const auto &cavetet : CaveOldTetList) {
                if (infected(cavetet)) CaveTetList.push_back(cavetet);
            }
            std::swap(CaveOldTetList, CaveTetList);
            CaveTetList.clear();

            if (CaveOldTetList.empty()) {
                insertpoint_abort(splitseg, ivf);
                ivf->iloc = NullCavity;
                return false;
            }

            if (ivf->splitbdflag) {
                int cutshcount = 0;
                for (size_t i = 0; i < CaveShList.size(); ++i) {
                    Facet parysh = CaveShList[i];
                    if (!smarktested(parysh)) continue;
                    enqflag = false;
                    stpivot(parysh, neightet);
                    if (neightet.tet != None && infected(neightet)) {
                        fsymself(neightet);
                        if (infected(neightet)) enqflag = true;
                    }
                    if (!enqflag) {
                        sunmarktest(parysh);
                        CaveShList[i] = CaveShList.back();
                        CaveShList.pop_back();
                        ++cutshcount;
                        --i;
                    }
                }
                if (cutshcount > 0) {
                    int bad = 0;
                    if (loc == OnFace) {
                        if (splitsh != nullptr && splitsh->sh != None && !smarktested(*splitsh)) ++bad;
                    } else if (loc == OnEdge) {
                        if (splitseg != nullptr && splitseg->sh != None && !smarktested(*splitseg)) ++bad;
                        if (splitsh != nullptr && splitsh->sh != None) {
                            const int pa = sorg(*splitsh);
                            neighsh = *splitsh;
                            while (true) {
                                if (sorg(neighsh) != pa) sesymself(neighsh);
                                if (!smarktested(neighsh)) ++bad;
                                spivotself(neighsh);
                                if (neighsh.sh == splitsh->sh) break;
                                if (neighsh.sh == None) break;
                            }
                        }
                    }
                    if (bad > 0) {
                        insertpoint_abort(splitseg, ivf);
                        ivf->iloc = NullCavity;
                        return false;
                    }
                }
            }
        }
        return true;
    }

    // Bond every boundary segment and subface of C(p) to a tet outside it, setting aside any that lies strictly inside.
    void bond_boundary_outward(InsertFlags *ivf) {
        Triface neightet, spintet, neineitet;
        if (CheckSubsegFlag) {
            for (size_t i = 0; i < CaveTetSegList.size(); ++i) {
                Facet paryseg = CaveTetSegList[i];
                if (!smarktested(paryseg)) {
                    int j = 0, k = 0;
                    sstpivot1(paryseg, neightet);
                    spintet = neightet;
                    while (true) {
                        ++j;
                        if (!infected(spintet)) neineitet = spintet;
                        else ++k;
                        fnextself(spintet);
                        if (spintet.tet == neightet.tet) break;
                    }
                    if (k == 0) {
                        CaveTetSegList[i] = CaveTetSegList.back();
                        CaveTetSegList.pop_back();
                        --i;
                    } else if (k < j) {
                        sstbond1(paryseg, neineitet);
                    } else {
                        // The segment is inside C(p).
                        if (ivf->splitbdflag) bail(2);
                        sinfect(paryseg);
                        CaveEncSegList.push_back(paryseg);
                    }
                } else {
                    sinfect(paryseg);
                }
            }
        }

        if (CheckSubfaceFlag) {
            for (size_t i = 0; i < CaveTetShList.size(); ++i) {
                Facet parysh = CaveTetShList[i];
                if (!smarktested(parysh)) {
                    int k = 0;
                    Facet outside = parysh;
                    for (int j = 0; j < 2; ++j) {
                        stpivot(parysh, neightet);
                        if (neightet.tet == None || !infected(neightet)) outside = parysh;
                        else ++k;
                        sesymself(parysh);
                    }
                    if (k == 0) {
                        CaveTetShList[i] = CaveTetShList.back();
                        CaveTetShList.pop_back();
                        --i;
                    } else if (k == 1) {
                        CaveTetShList[i] = outside;
                    } else {
                        // The subface is inside C(p).
                        if (ivf->splitbdflag) bail(2);
                        sinfect(parysh);
                        CaveEncShList.push_back(parysh);
                    }
                } else {
                    sinfect(parysh);
                }
            }
        }
    }

    // Fill the cavity with new tets, one per boundary face, and bond them to each other around the interior faces.
    void fill_cavity(int insertpt, InsertFlags *ivf) {
        Triface neightet, spintet, oldtet, newtet, newneitet;
        for (size_t i = 0; i < CaveBdryList.size(); ++i) {
            neightet = CaveBdryList[i];
            unmarktest(neightet);
            fsym(neightet, oldtet);
            if (apex(neightet) != DummyPoint) {
                maketetrahedron(&newtet);
                setorg(newtet, dest(neightet));
                setdest(newtet, org(neightet));
                setapex(newtet, apex(neightet));
                setoppo(newtet, insertpt);
            } else {
                ++HullSize;
                maketetrahedron(&newtet);
                setorg(newtet, org(neightet));
                setdest(newtet, dest(neightet));
                setapex(newtet, insertpt);
                setoppo(newtet, DummyPoint);
                esymself(newtet);
            }
            bond(newtet, neightet);
            CaveBdryList[i] = oldtet;
        }

        RecentTet = newtet;
        setpoint2tet(insertpt, Encode2(newtet.tet, 0));
        CaveTetList.clear();

        // Bond the new tets to each other around the interior faces.
        for (size_t i = 0; i < CaveBdryList.size(); ++i) {
            oldtet = CaveBdryList[i];
            fsym(oldtet, neightet);
            fsym(neightet, newtet);
            for (int j = 0; j < 3; ++j) {
                esym(newtet, neightet);
                if (Tets[neightet.tet].N[neightet.ver & 3] == None) {
                    spintet = oldtet;
                    while (true) {
                        fnextself(spintet);
                        if (!infected(spintet)) break;
                    }
                    fsym(spintet, newneitet);
                    esymself(newneitet);
                    bond(neightet, newneitet);
                    if (ivf->lawson > 1) CaveTetList.push_back(neightet);
                }
                setpoint2tet(org(newtet), Encode2(newtet.tet, 0));
                enextself(newtet);
                enextself(oldtet);
            }
            CaveBdryList[i] = newtet;
        }
    }

    // Hang the boundary subfaces and segments on the new tets, split whichever one p landed on, and queue what that produced.
    void reattach_boundary(int insertpt, Facet *splitsh, Facet *splitseg, InsertFlags *ivf) {
        Triface neightet, spintet;
        Facet checksh, checkseg;
        if (CheckSubfaceFlag) {
            for (auto &parysh : CaveTetShList) {
                if (!sinfected(parysh)) {
                    stpivot(parysh, neightet);
                    fsym(neightet, spintet);
                    sesymself(parysh);
                    tsbond(spintet, parysh);
                }
            }
        }

        if (CheckSubsegFlag) {
            for (auto &paryseg : CaveTetSegList) {
                if (!sinfected(paryseg)) {
                    sstpivot1(paryseg, neightet);
                    mark_seg_ring(paryseg, neightet);
                }
            }
        }

        if ((splitsh != nullptr && splitsh->sh != None) || (splitseg != nullptr && splitseg->sh != None)) {
            sinsertvertex(insertpt, splitsh, splitseg, ivf->sloc, ivf->sbowywat, 0);
        }

        if (CheckSubfaceFlag) {
            if (ivf->splitbdflag) {
                for (auto &parysh : CaveShBdList) {
                    spivot(parysh, checksh); // the new subface [a, b, p]
                    if (isdeadsh(checksh)) continue;
                    stpivot(parysh, neightet);
                    spintet = neightet;
                    while (true) {
                        fnextself(spintet);
                        if (!infected(spintet)) break;
                        if (spintet.tet == neightet.tet) bail(2);
                    }
                    fsym(spintet, neightet);
                    spintet = neightet;
                    while (true) {
                        fnextself(spintet);
                        if (apex(spintet) == insertpt) break;
                    }
                    if (sorg(checksh) != org(spintet)) sesymself(checksh);
                    tsbond(spintet, checksh);
                    fsymself(spintet);
                    sesymself(checksh);
                    tsbond(spintet, checksh);
                }
            } else {
                for (auto &parysh : CaveShBdList) {
                    spivot(parysh, checksh);
                    if (!isdeadsh(checksh)) SubFaceStack.push_back(SEncode(checksh));
                }
                for (auto &parysh : CaveEncShList) {
                    if (!smarktested(parysh)) {
                        checksh = parysh;
                        suninfect(checksh);
                        stdissolve(checksh);
                        SubFaceStack.push_back(SEncode(checksh));
                    }
                }
            }
        }

        if (CheckSubsegFlag) {
            if (ivf->splitbdflag) {
                if (splitseg != nullptr) {
                    for (auto &paryseg : CaveSegShList) {
                        checkseg = paryseg;
                        checkseg.shver = 0;
                        spivot(checkseg, checksh);
                        if (checksh.sh != None) {
                            stpivot(checksh, neightet);
                        } else {
                            point2tetorg(sorg(checkseg), neightet);
                            finddirection(&neightet, sdest(checkseg));
                        }
                        if (isdeadtet(neightet)) bail(2);
                        bond_seg_ring(checkseg, neightet);
                    }
                }
            } else {
                if (splitseg != nullptr) {
                    for (auto &paryseg : CaveSegShList) SubSegStack.push_back(SEncode(paryseg));
                }
                for (auto &paryseg : CaveEncSegList) {
                    if (!smarktested(paryseg)) {
                        checkseg = paryseg;
                        suninfect(checkseg);
                        sstdissolve1(checkseg);
                        const int s = int(randomnation(unsigned(SubSegStack.size()) + 1));
                        SubSegStack.push_back(SubSegStack[s]);
                        SubSegStack[s] = SEncode(checkseg);
                    }
                }
            }
        }
    }

    // Retire what the cavity swallowed, queue the new elements for the encroachment passes, and empty the work lists.
    void finish_insertion(Facet *splitsh, Facet *splitseg, InsertFlags *ivf) {
        Triface neightet;
        Facet checksh;
        if (ivf->validflag) {
            // A vertex can end up strictly inside the cavity; it is gone from the mesh.
            for (const int pv : CaveTetVertList) {
                Triface t;
                Decode(point2tet(pv), t);
                if (infected(t)) release_steiner(pv, pointtype(pv));
            }
        }

        if (ivf->chkencflag & 1) {
            for (auto &paryseg : CaveTetSegList) {
                if (!sinfected(paryseg)) enqueuesubface(BadSubsegs, &paryseg);
            }
            if (splitseg != nullptr) {
                for (auto &paryseg : CaveSegShList) enqueuesubface(BadSubsegs, &paryseg);
            }
        }
        if (ivf->chkencflag & 2) {
            for (auto &parysh : CaveTetShList) {
                if (!sinfected(parysh)) enqueuesubface(BadSubfacs, &parysh);
            }
            for (auto &parysh : CaveShBdList) {
                spivot(parysh, checksh);
                if (!isdeadsh(checksh)) enqueuesubface(BadSubfacs, &checksh);
            }
        }
        if (ivf->chkencflag & 4) {
            for (auto &cavetet : CaveBdryList) enqueuetetrahedron(&cavetet);
        }

        for (auto &t : CaveOldTetList) {
            if (ishulltet(t)) --HullSize;
            tetrahedrondealloc(t.tet);
        }

        if ((splitsh != nullptr && splitsh->sh != None) || (splitseg != nullptr && splitseg->sh != None)) {
            for (auto &parysh : CaveShList) {
                if (CheckSubfaceFlag) {
                    stpivot(parysh, neightet);
                    if (neightet.tet != None && Tets[neightet.tet].V[0] != None) {
                        tsdissolve(neightet);
                        fsymself(neightet);
                        tsdissolve(neightet);
                    }
                }
                subfacedealloc(parysh.sh);
            }
            if (splitseg != nullptr && splitseg->sh != None) subsegdealloc(splitseg->sh);
        }

        if (ivf->lawson) {
            for (auto &t : CaveBdryList) flippush(&t);
            if (ivf->lawson > 1) {
                for (auto &t : CaveTetList) flippush(&t);
            }
        }

        CaveOldTetList.clear();
        CaveBdryList.clear();
        CaveTetList.clear();
        if (CheckSubsegFlag) {
            CaveTetSegList.clear();
            CaveEncSegList.clear();
        }
        if (CheckSubfaceFlag) {
            CaveTetShList.clear();
            CaveEncShList.clear();
        }
        if (ivf->smlenflag || ivf->validflag) CaveTetVertList.clear();
        if ((splitsh != nullptr && splitsh->sh != None) || (splitseg != nullptr && splitseg->sh != None)) {
            CaveShList.clear();
            CaveShBdList.clear();
            CaveSegShList.clear();
        }
    }

    // True when some tet around the edge at `t` is outside the cavity, so the edge survives the insertion.
    bool ring_leaves_cavity(const Triface &t) const {
        Triface spintet = t;
        do {
            if (!infected(spintet)) return true;
            fnextself(spintet);
        } while (spintet.tet != t.tet);
        return false;
    }

    int insertpoint(int insertpt, Triface *searchtet, Facet *splitsh, Facet *splitseg, InsertFlags *ivf) {
        Triface neightet;
        int loc = Outside;

        if (searchtet->tet != None) loc = ivf->iloc;
        if (loc == Outside) {
            if (searchtet->tet == None) randomsample(insertpt, searchtet);
            loc = locate(insertpt, searchtet);
        }
        ivf->iloc = loc;

        if (!build_initial_cavity(loc, searchtet, splitsh, splitseg, ivf)) return 0;
        if (ivf->collect_inial_cavity_flag) {
            for (const auto &cavetet : CaveOldTetList) CaveOldTetOnly.push_back(cavetet.tet);
            insertpoint_abort(splitseg, ivf);
            return 0;
        }
        if (reject_near_vertex(insertpt, loc, searchtet, splitseg, ivf)) return 0;
        if (ivf->assignmeshsize) Pts[insertpt].Mtr = getpointmeshsize(insertpt, searchtet, loc);
        grow_cavity_bw(insertpt, ivf);
        if (ivf->refineflag == 1 && !infected(ivf->refinetet)) {
            insertpoint_abort(splitseg, ivf);
            ivf->iloc = BadElement;
            return 0;
        }
        if (!collect_cavity_boundary(insertpt, splitseg, ivf)) return 0;
        if (ivf->iloc == Outside && ivf->refineflag) {
            insertpoint_abort(splitseg, ivf);
            return 0;
        }
        grow_subcavity(insertpt, ivf);
        if (!carve_cavity_star_shaped(insertpt, loc, splitsh, splitseg, ivf)) return 0;
        if (ivf->refineflag) {
            if ((ivf->refineflag == 1 && !infected(ivf->refinetet)) || (ivf->refineflag == 2 && !smarktested(ivf->refinesh))) {
                insertpoint_abort(splitseg, ivf);
                ivf->iloc = BadElement;
                return 0;
            }
            // Recovery asks for the crossing face (4) or crossing edge (8) to be gone: refuse the point when it would survive the insertion.
            bool bflag = false;
            if (ivf->refineflag == 4) {
                Triface adjtet;
                fsym(ivf->refinetet, adjtet);
                if (!infected(ivf->refinetet) || !infected(adjtet)) bflag = true;
            } else if (ivf->refineflag == 8) {
                bflag = ring_leaves_cavity(ivf->refinetet);
            }
            if (bflag) {
                insertpoint_abort(splitseg, ivf);
                ivf->iloc = BadElement;
                return 0;
            }
        }

        if (splitseg != nullptr && splitseg->sh != None) {
            sstpivot1(*splitseg, neightet);
            if (neightet.tet != None && ring_leaves_cavity(neightet)) {
                insertpoint_abort(splitseg, ivf);
                ivf->iloc = BadElement;
                return 0;
            }
        }

        if (ivf->cdtflag || ivf->smlenflag || ivf->validflag) {
            for (const auto &cavetet : CaveOldTetList) {
                for (int j = 0; j < 4; ++j) {
                    const int pt = Tets[cavetet.tet].V[j];
                    if (pt != DummyPoint && !pinfected(pt)) {
                        pinfect(pt);
                        CaveTetVertList.push_back(pt);
                    }
                }
            }
            for (const int pv : CaveTetVertList) puninfect(pv);
            if (ivf->smlenflag) {
                ivf->smlen = distance(P(CaveTetVertList[0]), P(insertpt));
                ivf->parentpt = CaveTetVertList[0];
                for (size_t i = 1; i < CaveTetVertList.size(); ++i) {
                    const double len = distance(P(CaveTetVertList[i]), P(insertpt));
                    if (len < ivf->smlen) {
                        ivf->smlen = len;
                        ivf->parentpt = CaveTetVertList[i];
                    }
                }
            }
        }

        if (ivf->cdtflag) {
            for (auto &t : CaveOldTetList) unmarktest(t);
            for (auto &t : CaveBdryList) unmarktest(t);
            CaveTetList.clear();
            if (CheckSubsegFlag) CaveTetSegList.clear();
            if (CheckSubfaceFlag) CaveTetShList.clear();
            ivf->iloc = InStar;
            return 1;
        }

        bond_boundary_outward(ivf);
        fill_cavity(insertpt, ivf);
        reattach_boundary(insertpt, splitsh, splitseg, ivf);
        finish_insertion(splitsh, splitseg, ivf);
        return 1;
    }

    //=== Point location and the incremental Delaunay construction ===

    int TransGC[8][3][8]{};
    int Tsb1Mod3[8]{};
    long Samples{1};

    void hilbert_init(int n) {
        const int N = n == 2 ? 4 : 8;
        const int mask = n == 2 ? 3 : 7;
        int gc[8];
        for (int i = 0; i < N; ++i) gc[i] = i ^ (i >> 1);
        for (int e = 0; e < N; ++e) {
            for (int d = 0; d < n; ++d) {
                const int f = e ^ (1 << d);
                const int travel_bit = e ^ f;
                for (int i = 0; i < N; ++i) {
                    const int k = gc[i] * (travel_bit * 2);
                    const int g = (k | (k / N)) & mask;
                    TransGC[e][d][i] = g ^ e;
                }
            }
        }
        Tsb1Mod3[0] = 0;
        for (int i = 1; i < N; ++i) {
            int v = ~i;
            v = (v ^ (v - 1)) >> 1;
            int c = 0;
            for (; v; ++c) v >>= 1;
            Tsb1Mod3[i] = c % n;
        }
    }

    int hilbert_split(int *va, int arraysize, int gc0, int gc1, double bxmin, double bxmax, double bymin, double bymax, double bzmin, double bzmax) {
        const int axis = (gc0 ^ gc1) >> 1;
        const double split = axis == 0 ? 0.5 * (bxmin + bxmax) : (axis == 1 ? 0.5 * (bymin + bymax) : 0.5 * (bzmin + bzmax));
        const int d = (gc0 & (1 << axis)) == 0 ? 1 : -1;
        const auto coord = [&](int p) { return axis == 0 ? Pts[p].Pos.x : (axis == 1 ? Pts[p].Pos.y : Pts[p].Pos.z); };
        int i = 0, j = arraysize - 1;
        if (d > 0) {
            while (true) {
                for (; i < arraysize; ++i) {
                    if (coord(va[i]) >= split) break;
                }
                for (; j >= 0; --j) {
                    if (coord(va[j]) < split) break;
                }
                if (i == j + 1) break;
                std::swap(va[i], va[j]);
            }
        } else {
            while (true) {
                for (; i < arraysize; ++i) {
                    if (coord(va[i]) <= split) break;
                }
                for (; j >= 0; --j) {
                    if (coord(va[j]) > split) break;
                }
                if (i == j + 1) break;
                std::swap(va[i], va[j]);
            }
        }
        return i;
    }

    void hilbert_sort3(int *va, int arraysize, int e, int d, double bxmin, double bxmax, double bymin, double bymax, double bzmin, double bzmax, int depth) {
        int p[9];
        const int n = 3, mask = 7;
        p[0] = 0;
        p[8] = arraysize;
        p[4] = hilbert_split(va, p[8], TransGC[e][d][3], TransGC[e][d][4], bxmin, bxmax, bymin, bymax, bzmin, bzmax);
        p[2] = hilbert_split(va, p[4], TransGC[e][d][1], TransGC[e][d][2], bxmin, bxmax, bymin, bymax, bzmin, bzmax);
        p[1] = hilbert_split(va, p[2], TransGC[e][d][0], TransGC[e][d][1], bxmin, bxmax, bymin, bymax, bzmin, bzmax);
        p[3] = hilbert_split(&va[p[2]], p[4] - p[2], TransGC[e][d][2], TransGC[e][d][3], bxmin, bxmax, bymin, bymax, bzmin, bzmax) + p[2];
        p[6] = hilbert_split(&va[p[4]], p[8] - p[4], TransGC[e][d][5], TransGC[e][d][6], bxmin, bxmax, bymin, bymax, bzmin, bzmax) + p[4];
        p[5] = hilbert_split(&va[p[4]], p[6] - p[4], TransGC[e][d][4], TransGC[e][d][5], bxmin, bxmax, bymin, bymax, bzmin, bzmax) + p[4];
        p[7] = hilbert_split(&va[p[6]], p[8] - p[6], TransGC[e][d][6], TransGC[e][d][7], bxmin, bxmax, bymin, bymax, bzmin, bzmax) + p[6];

        if (HilbertOrder > 0 && depth + 1 == HilbertOrder) return;

        for (int w = 0; w < 8; ++w) {
            if (p[w + 1] - p[w] <= HilbertLimit) continue;
            int e_w;
            if (w == 0) e_w = 0;
            else {
                const int k = 2 * ((w - 1) / 2);
                e_w = k ^ (k >> 1);
            }
            const int k = e_w;
            e_w = ((k << (d + 1)) & mask) | ((k >> (n - d - 1)) & mask);
            const int ei = e ^ e_w;
            const int d_w = w == 0 ? 0 : ((w % 2) == 0 ? Tsb1Mod3[w - 1] : Tsb1Mod3[w]);
            const int di = (d + d_w + 1) % n;
            const double x1 = TransGC[e][d][w] & 1 ? 0.5 * (bxmin + bxmax) : bxmin;
            const double x2 = TransGC[e][d][w] & 1 ? bxmax : 0.5 * (bxmin + bxmax);
            const double y1 = TransGC[e][d][w] & 2 ? 0.5 * (bymin + bymax) : bymin;
            const double y2 = TransGC[e][d][w] & 2 ? bymax : 0.5 * (bymin + bymax);
            const double z1 = TransGC[e][d][w] & 4 ? 0.5 * (bzmin + bzmax) : bzmin;
            const double z2 = TransGC[e][d][w] & 4 ? bzmax : 0.5 * (bzmin + bzmax);
            hilbert_sort3(&va[p[w]], p[w + 1] - p[w], ei, di, x1, x2, y1, y2, z1, z2, depth + 1);
        }
    }

    void brio_multiscale_sort(int *va, int arraysize, int threshold, double ratio, int *depth) {
        int middle = 0;
        if (arraysize >= threshold) {
            ++(*depth);
            middle = int(arraysize * ratio);
            brio_multiscale_sort(va, middle, threshold, ratio, depth);
        }
        hilbert_sort3(&va[middle], arraysize - middle, 0, 0, BoxMin.x, BoxMax.x, BoxMin.y, BoxMax.y, BoxMin.z, BoxMax.z, 0);
    }

    // The insphere test with symbolic perturbation of a cospherical tie, resolved on point index.
    double insphere_s(int pa, int pb, int pc, int pd, int pe) {
        double sign = insphere(P(pa), P(pb), P(pc), P(pd), P(pe));
        if (sign != 0.0) return sign;
        int pt[5]{pa, pb, pc, pd, pe};
        int swaps = 0, n = 5, count;
        do {
            count = 0;
            --n;
            for (int i = 0; i < n; ++i) {
                if (pt[i] > pt[i + 1]) {
                    std::swap(pt[i], pt[i + 1]);
                    ++count;
                }
            }
            swaps += count;
        } while (count > 0);
        double oriA = orient3d(P(pt[1]), P(pt[2]), P(pt[3]), P(pt[4]));
        if (oriA != 0.0) return (swaps % 2) != 0 ? -oriA : oriA;
        const double oriB = -orient3d(P(pt[0]), P(pt[2]), P(pt[3]), P(pt[4]));
        if (oriB == 0.0) bail(2);
        return (swaps % 2) != 0 ? -oriB : oriB;
    }

    // Pick a starting tet for point location: the closest of the current handle, the most recent tet, and a random sample of the pool.
    void randomsample(int searchpt, Triface *searchtet) {
        double searchdist;
        if (!NonConvex) {
            if (searchtet->tet == None) *searchtet = RecentTet;
            searchtet->ver = 3;
            searchdist = dot(P(searchpt) - P(org(*searchtet)), P(searchpt) - P(org(*searchtet)));
            if (RecentTet.tet != searchtet->tet && RecentTet.tet != None) {
                Triface rt = RecentTet;
                rt.ver = 3;
                const double d = dot(P(searchpt) - P(org(rt)), P(searchpt) - P(org(rt)));
                if (d < searchdist) {
                    *searchtet = rt;
                    searchdist = d;
                }
            }
        } else {
            searchdist = LongEst;
        }
        // As many samples as the fourth root of the pool size, spread so each block gets at least one.
        // A dead slot is re-drawn rather than skipped.
        while (Samples * Samples * Samples * Samples < TetItems) ++Samples;
        const long maxitems = long(Tets.size());
        const long tetblocks = (maxitems + TetrahedraPerBlock - 1) / TetrahedraPerBlock;
        const long samplesperblock = 1 + (Samples / tetblocks);
        long sampleblocks = Samples / samplesperblock;
        if (sampleblocks == 0) sampleblocks = 1;
        for (long i = 0; i < sampleblocks; ++i) {
            for (long j = 0; j < samplesperblock; ++j) {
                const long samplenum = i == tetblocks - 1 ? long(randomnation(unsigned(maxitems - i * TetrahedraPerBlock))) : long(randomnation(unsigned(TetrahedraPerBlock)));
                const long t = i * TetrahedraPerBlock + samplenum;
                if (Tets[t].V[0] != None) {
                    const double d = dot(P(searchpt) - P(Tets[t].V[0]), P(searchpt) - P(Tets[t].V[0]));
                    if (d < searchdist) {
                        searchtet->tet = int(t);
                        searchtet->ver = 11;
                        searchdist = d;
                    }
                } else if (i != tetblocks - 1) {
                    --j; // a dead slot, draw again
                }
            }
        }
    }

    // Turn searchtet to the one face searchpt lies below, which is where a walk toward it starts.
    // Every face turned away means the mesh holds an inverted tet.
    void face_toward(int searchpt, Triface *searchtet) {
        for (searchtet->ver = 0; searchtet->ver < 4; ++searchtet->ver) {
            if (orient3d(P(org(*searchtet)), P(dest(*searchtet)), P(apex(*searchtet)), P(searchpt)) < 0.0) break;
        }
        if (searchtet->ver == 4) bail(2);
    }

    // Cross the face the move names, into the neighbouring tet.
    // Returns LocUnknown to keep walking, or the reason the walk stopped.
    int step_across(Triface *searchtet, WalkMove nextmove, int chkencflag) {
        if (nextmove == OrgMove) enextesymself(*searchtet);
        else if (nextmove == DestMove) eprevesymself(*searchtet);
        else esymself(*searchtet);

        if (chkencflag && issubface(*searchtet)) return EncSubface;
        Decode(Tets[searchtet->tet].N[searchtet->ver & 3], *searchtet);
        return ishulltet(*searchtet) ? Outside : LocUnknown;
    }

    // Where searchpt lies in searchtet, once a walk has stopped and none of the three orientations is negative.
    // searchtet is turned so that the feature the point lies on is named by its face, its edge or its origin.
    int locate_feature(Triface *searchtet, double oriorg, double oridest, double oriapex) {
        if (oriorg == 0) {
            enextesymself(*searchtet); // The face opposite the origin.
            if (oridest == 0) {
                eprevself(*searchtet); // The edge oppo->apex.
                return oriapex == 0 ? OnVertex : OnEdge; // oppo coincides with the point.
            }
            if (oriapex == 0) {
                enextself(*searchtet); // The edge dest->oppo.
                return OnEdge;
            }
            return OnFace;
        }
        if (oridest == 0) {
            eprevesymself(*searchtet); // The face opposite the destination.
            if (oriapex == 0) {
                eprevself(*searchtet); // The edge oppo->org.
                return OnEdge;
            }
            return OnFace;
        }
        if (oriapex == 0) {
            esymself(*searchtet); // The face opposite the apex.
            return OnFace;
        }
        return InTetrahedron;
    }

    // The walk used while the triangulation is still convex.
    int locate_dt(int searchpt, Triface *searchtet) {
        int loc = Outside;
        if (searchtet->tet == None) searchtet->tet = RecentTet.tet;
        if (ishulltet(*searchtet)) searchtet->tet = Tets[searchtet->tet].N[3] >> 4;

        face_toward(searchpt, searchtet);

        while (true) {
            const int toppo = oppo(*searchtet);
            if (toppo == searchpt) {
                esymself(*searchtet);
                eprevself(*searchtet);
                loc = OnVertex;
                break;
            }
            const int s = int(Rng() % 3);
            for (int i = 0; i < s; ++i) enextself(*searchtet);

            // Each orientation is asked for only once the one before it has failed to settle the direction.
            WalkMove nextmove;
            const double oriorg = orient3d(P(dest(*searchtet)), P(apex(*searchtet)), P(toppo), P(searchpt));
            if (oriorg < 0) {
                nextmove = OrgMove;
            } else {
                const double oridest = orient3d(P(apex(*searchtet)), P(org(*searchtet)), P(toppo), P(searchpt));
                if (oridest < 0) {
                    nextmove = DestMove;
                } else {
                    const double oriapex = orient3d(P(org(*searchtet)), P(dest(*searchtet)), P(toppo), P(searchpt));
                    if (oriapex >= 0) {
                        loc = locate_feature(searchtet, oriorg, oridest, oriapex);
                        break;
                    }
                    nextmove = ApexMove;
                }
            }
            if (const int stop = step_across(searchtet, nextmove, 0); stop != LocUnknown) {
                loc = stop;
                break;
            }
        }
        return loc;
    }

    // The general walk, which may stop at a subface.
    int locate(int searchpt, Triface *searchtet, int chkencflag = 0) {
        WalkMove nextmove = OrgMove;
        int loc = Outside;
        if (searchtet->tet == None) searchtet->tet = RecentTet.tet;
        if (ishulltet(*searchtet)) searchtet->tet = Tets[searchtet->tet].N[3] >> 4;

        face_toward(searchpt, searchtet);
        int torg = org(*searchtet), tdest = dest(*searchtet), tapex = apex(*searchtet);

        while (true) {
            const int toppo = oppo(*searchtet);
            if (toppo == searchpt) {
                esymself(*searchtet);
                eprevself(*searchtet);
                loc = OnVertex;
                break;
            }
            const double oriorg = orient3d(P(tdest), P(tapex), P(toppo), P(searchpt));
            const double oridest = orient3d(P(tapex), P(torg), P(toppo), P(searchpt));
            const double oriapex = orient3d(P(torg), P(tdest), P(toppo), P(searchpt));

            if (oriorg < 0) {
                if (oridest < 0) {
                    if (oriapex < 0) {
                        const int s = int(randomnation(3));
                        nextmove = s == 0 ? OrgMove : (s == 1 ? DestMove : ApexMove);
                    } else {
                        nextmove = randomnation(2) ? OrgMove : DestMove;
                    }
                } else {
                    nextmove = oriapex < 0 ? (randomnation(2) ? OrgMove : ApexMove) : OrgMove;
                }
            } else {
                if (oridest < 0) {
                    nextmove = oriapex < 0 ? (randomnation(2) ? DestMove : ApexMove) : DestMove;
                } else {
                    if (oriapex < 0) {
                        nextmove = ApexMove;
                    } else {
                        loc = locate_feature(searchtet, oriorg, oridest, oriapex);
                        break;
                    }
                }
            }

            if (const int stop = step_across(searchtet, nextmove, chkencflag); stop != LocUnknown) {
                loc = stop;
                break;
            }
            torg = org(*searchtet);
            tdest = dest(*searchtet);
            tapex = apex(*searchtet);
        }
        return loc;
    }

    // The plain Bowyer-Watson insertion used to build the initial Delaunay triangulation.
    // A small cavity could be filled through an adjacency matrix and a large one by neighbour search.
    // The two produce the same mesh, and this is the latter.
    int insert_vertex_bw(int insertpt, Triface *searchtet, InsertFlags *ivf) {
        Triface cavetet, spintet, neightet, neineitet;
        Triface oldtet, newtet;
        int loc = Outside;
        bool enqflag;

        if (searchtet->tet != None) loc = ivf->iloc;
        if (loc == Outside) {
            if (searchtet->tet == None) randomsample(insertpt, searchtet);
            loc = locate_dt(insertpt, searchtet);
        }
        ivf->iloc = loc;

        if (loc == Outside || loc == InTetrahedron) {
            infect(*searchtet);
            CaveOldTetOnly.push_back(searchtet->tet);
        } else if (loc == OnFace) {
            infect(*searchtet);
            CaveOldTetOnly.push_back(searchtet->tet);
            neightet.tet = Tets[searchtet->tet].N[searchtet->ver & 3] >> 4;
            neightet.ver = 0;
            infect(neightet);
            CaveOldTetOnly.push_back(neightet.tet);
        } else if (loc == OnEdge) {
            spintet = *searchtet;
            while (true) {
                infect(spintet);
                CaveOldTetOnly.push_back(spintet.tet);
                fnextself(spintet);
                if (spintet.tet == searchtet->tet) break;
            }
        } else if (loc == OnVertex) {
            return 0;
        } else {
            CaveOldTetOnly.clear();
            return 0;
        }

        for (size_t i = 0; i < CaveOldTetOnly.size(); ++i) {
            cavetet.tet = CaveOldTetOnly[i];
            for (cavetet.ver = 0; cavetet.ver < 4; ++cavetet.ver) {
                neightet.tet = Tets[cavetet.tet].N[cavetet.ver] >> 4;
                neightet.ver = 0;
                if (infected(neightet)) continue;
                enqflag = false;
                if (!marktested(neightet)) {
                    const Tet &e = Tets[neightet.tet];
                    if (e.V[3] != DummyPoint) {
                        enqflag = insphere_s(e.V[0], e.V[1], e.V[2], e.V[3], insertpt) < 0.0;
                    } else {
                        const double ori = orient3d(P(e.V[0]), P(e.V[1]), P(e.V[2]), P(insertpt));
                        if (ori < 0) {
                            enqflag = true;
                        } else if (ori == 0.0) {
                            const Triface behind{.tet = Tets[neightet.tet].N[3] >> 4, .ver = 0};
                            const Tet &f = Tets[behind.tet];
                            enqflag = insphere_s(f.V[0], f.V[1], f.V[2], f.V[3], insertpt) < 0.0;
                        }
                    }
                    marktest(neightet);
                }
                if (enqflag) {
                    infect(neightet);
                    CaveOldTetOnly.push_back(neightet.tet);
                } else {
                    CaveBdryList.push_back(cavetet);
                }
            }
        }

        const size_t f_out = CaveBdryList.size();
        for (size_t i = 0; i < f_out; ++i) {
            oldtet = CaveBdryList[i];
            Decode(Tets[oldtet.tet].N[oldtet.ver], neightet);
            unmarktest(neightet);
            if (ishulltet(oldtet)) {
                neightet.ver = epivot[neightet.ver];
                if (apex(neightet) == DummyPoint) ++HullSize;
            }
            const int v0 = dest(neightet), v1 = org(neightet), v2 = apex(neightet);
            maketetrahedron2(&newtet, v1, v0, insertpt, v2);
            newtet.ver = 2;
            bond(newtet, neightet);
            // Each cavity vertex points at the first new tet that reaches it, whose third vertex is not yet the inserted point.
            for (const int v : {v0, v1, v2}) {
                const int enc = point2tet(v);
                if (enc == None || Tets[enc >> 4].V[2] != insertpt) setpoint2tet(v, Encode2(newtet.tet, 0));
            }
            CaveBdryList[i] = oldtet;
        }

        for (size_t i = 0; i < f_out; ++i) {
            oldtet = CaveBdryList[i];
            fsym(oldtet, neightet);
            fsym(neightet, newtet);
            for (int j = 0; j < 3; ++j) {
                esym(newtet, neightet);
                if (Tets[neightet.tet].N[neightet.ver & 3] == None) {
                    spintet = oldtet;
                    while (true) {
                        fnextself(spintet);
                        if (!infected(spintet)) break;
                    }
                    fsym(spintet, neineitet);
                    esymself(neineitet);
                    bond(neightet, neineitet);
                }
                enextself(newtet);
                enextself(oldtet);
            }
            CaveBdryList[i] = newtet;
        }

        RecentTet = CaveBdryList[Rng() % f_out];
        setpoint2tet(insertpt, Encode2(RecentTet.tet, 0));

        for (int t : CaveOldTetOnly) {
            Triface o{t, 0};
            if (ishulltet(o)) --HullSize;
            tetrahedrondealloc(t);
        }
        CaveOldTetOnly.clear();
        CaveBdryList.clear();
        return 1;
    }

    // One real tet and the four hull tets closing its faces.
    void initialdelaunay(int pa, int pb, int pc, int pd) {
        Triface firsttet, tetopa, tetopb, tetopc, tetopd, worktet, worktet1;
        maketetrahedron2(&firsttet, pa, pb, pc, pd);
        maketetrahedron2(&tetopa, pb, pc, pd, DummyPoint);
        maketetrahedron2(&tetopb, pc, pa, pd, DummyPoint);
        maketetrahedron2(&tetopc, pa, pb, pd, DummyPoint);
        maketetrahedron2(&tetopd, pb, pa, pc, DummyPoint);
        HullSize += 4;

        bond(firsttet, tetopd);
        esym(firsttet, worktet);
        bond(worktet, tetopc);
        enextesym(firsttet, worktet);
        bond(worktet, tetopa);
        eprevesym(firsttet, worktet);
        bond(worktet, tetopb);

        esym(tetopc, worktet);
        esym(tetopd, worktet1);
        bond(worktet, worktet1);
        esym(tetopa, worktet);
        eprevesym(tetopd, worktet1);
        bond(worktet, worktet1);
        esym(tetopb, worktet);
        enextesym(tetopd, worktet1);
        bond(worktet, worktet1);
        eprevesym(tetopc, worktet);
        enextesym(tetopb, worktet1);
        bond(worktet, worktet1);
        eprevesym(tetopa, worktet);
        enextesym(tetopc, worktet1);
        bond(worktet, worktet1);
        eprevesym(tetopb, worktet);
        enextesym(tetopa, worktet1);
        bond(worktet, worktet1);

        for (int p : {pa, pb, pc, pd}) {
            if (pointtype(p) == UnusedVertex) setpointtype(p, VolVertex);
            setpoint2tet(p, Encode(firsttet));
        }
        setpoint2tet(DummyPoint, Encode(tetopa));
        RecentTet = firsttet;
    }

    bool incrementaldelaunay() {
        // A uniformly random permutation, then BRIO rounds laid out along a Hilbert curve.
        std::vector<int> permutarray(NumInputPoints);
        Rng.seed(unsigned(NumInputPoints));
        for (int i = 0; i < NumInputPoints; ++i) {
            const int randindex = int(Rng() % (i + 1));
            permutarray[i] = permutarray[randindex];
            permutarray[randindex] = i + 1;
        }
        hilbert_init(3);
        int ngroup = 0;
        brio_multiscale_sort(permutarray.data(), NumInputPoints, BrioThreshold, BrioRatio, &ngroup);

        const double bboxsize = std::sqrt((BoxMax.x - BoxMin.x) * (BoxMax.x - BoxMin.x) + (BoxMax.y - BoxMin.y) * (BoxMax.y - BoxMin.y) + (BoxMax.z - BoxMin.z) * (BoxMax.z - BoxMin.z));
        const double bboxsize2 = bboxsize * bboxsize;
        const double bboxsize3 = bboxsize2 * bboxsize;

        int i = 1;
        while (distance(P(permutarray[0]), P(permutarray[i])) / bboxsize < Epsilon) {
            ++i;
            if (i == NumInputPoints - 1) return false; // all vertices are essentially one point
        }
        if (i > 1) std::swap(permutarray[i], permutarray[1]);

        i = 2;
        dvec3 v1 = P(permutarray[1]) - P(permutarray[0]);
        dvec3 v2 = P(permutarray[i]) - P(permutarray[0]);
        dvec3 n = cross(v1, v2);
        while (std::sqrt(dot(n, n)) / bboxsize2 < Epsilon) {
            ++i;
            if (i == NumInputPoints - 1) return false; // all vertices are collinear
            v2 = P(permutarray[i]) - P(permutarray[0]);
            n = cross(v1, v2);
        }
        if (i > 2) std::swap(permutarray[i], permutarray[2]);

        i = 3;
        double ori = orient3dfast(P(permutarray[0]), P(permutarray[1]), P(permutarray[2]), P(permutarray[i]));
        while (std::abs(ori) / bboxsize3 < Epsilon) {
            ++i;
            if (i == NumInputPoints) return false; // all vertices are coplanar
            ori = orient3dfast(P(permutarray[0]), P(permutarray[1]), P(permutarray[2]), P(permutarray[i]));
        }
        if (i > 3) std::swap(permutarray[i], permutarray[3]);
        if (ori > 0.0) std::swap(permutarray[0], permutarray[1]);

        initialdelaunay(permutarray[0], permutarray[1], permutarray[2], permutarray[3]);

        InsertFlags ivf{.bowywat = 1, .lawson = 0};
        Triface searchtet;
        for (int k = 4; k < NumInputPoints; ++k) {
            if (pointtype(permutarray[k]) == UnusedVertex) setpointtype(permutarray[k], VolVertex);
            searchtet.tet = RecentTet.tet;
            searchtet.ver = 0;
            ivf.iloc = Outside;
            if (!insert_vertex_bw(permutarray[k], &searchtet, &ivf)) {
                if (ivf.iloc == OnVertex) {
                    const int dup = org(searchtet);
                    setpoint2ppt(permutarray[k], dup);
                    setpointtype(permutarray[k], DuplicatedVertex);
                    ++DupVerts;
                } else if (ivf.iloc == NearVertex) {
                    bail(2); // insert_vertex_bw never reports this
                }
            }
        }
        return true;
    }

    // Walk the tets around org(searchtet) until the one facing endpt is found, reporting how the ray leaves it.
    int finddirection(Triface *searchtet, int endpt) {
        enum { HMove,
               RMove,
               LMove } nextmove = HMove;
        const int pa = org(*searchtet);
        if (Tets[searchtet->tet].V[3] == DummyPoint) {
            Decode(Tets[searchtet->tet].N[3], *searchtet);
            const Tet &e = Tets[searchtet->tet];
            searchtet->ver = e.V[0] == pa ? 11 : (e.V[1] == pa ? 3 : (e.V[2] == pa ? 7 : 0));
        }
        int pb = dest(*searchtet);
        if (pb == endpt) return AcrossVert;
        int pc = apex(*searchtet);
        if (pc == endpt) {
            eprevesymself(*searchtet);
            return AcrossVert;
        }

        while (true) {
            const int pd = oppo(*searchtet);
            if (pd == endpt) {
                esymself(*searchtet);
                enextself(*searchtet);
                return AcrossVert;
            }
            if (pd == DummyPoint) {
                // Only reachable once carving has made the mesh non-convex.
                if (NonConvex) return AcrossFace;
                bail(2);
            }

            const double hori = orient3d(P(pa), P(pb), P(pc), P(endpt));
            const double rori = orient3d(P(pb), P(pa), P(pd), P(endpt));
            const double lori = orient3d(P(pa), P(pc), P(pd), P(endpt));

            if (hori > 0) {
                if (rori > 0) {
                    if (lori > 0) {
                        const int s = int(randomnation(3));
                        nextmove = s == 0 ? HMove : (s == 1 ? RMove : LMove);
                    } else {
                        nextmove = randomnation(2) ? HMove : RMove;
                    }
                } else {
                    nextmove = lori > 0 ? (randomnation(2) ? HMove : LMove) : HMove;
                }
            } else {
                if (rori > 0) {
                    nextmove = lori > 0 ? (randomnation(2) ? RMove : LMove) : RMove;
                } else {
                    if (lori > 0) {
                        nextmove = LMove;
                    } else {
                        if (hori == 0) {
                            if (rori == 0) return AcrossVert;
                            if (lori == 0) {
                                eprevesymself(*searchtet);
                                return AcrossVert;
                            }
                            return AcrossEdge;
                        }
                        if (rori == 0) {
                            esymself(*searchtet);
                            enextself(*searchtet);
                            return lori == 0 ? AcrossVert : AcrossEdge;
                        }
                        if (lori == 0) {
                            eprevesymself(*searchtet);
                            return AcrossEdge;
                        }
                        return AcrossFace;
                    }
                }
            }

            if (nextmove == RMove) {
                fnextself(*searchtet);
            } else if (nextmove == LMove) {
                eprevself(*searchtet);
                fnextself(*searchtet);
                enextself(*searchtet);
            } else {
                fsymself(*searchtet);
                enextself(*searchtet);
            }
            if (org(*searchtet) != pa) bail(2);
            pb = dest(*searchtet);
            pc = apex(*searchtet);
        }
    }

    //=== The surface mesh ===

    // Insert a vertex into the triangulation of a facet, and split the segment it lies on when one is given.
    int sinsertvertex(int insertpt, Facet *searchsh, Facet *splitseg, int iloc, int bowywat, int rflag) {
        Facet cavesh, neighsh, newsh, casout, casin, checkseg;
        int loc = Outside;
        double sign, ori;
        newsh.sh = None;

        if (bowywat == 3) loc = InStar;

        if (splitseg != nullptr && splitseg->sh != None) {
            spivot(*splitseg, *searchsh);
            if (loc != InStar) loc = OnEdge;
        } else {
            if (loc != InStar) loc = iloc;
            if (loc == Outside) {
                if (searchsh->sh == None) *searchsh = RecentSh;
                loc = slocate(insertpt, searchsh, 1, 1, rflag);
            }
        }

        if (loc == OnFace) {
            smarktest(*searchsh);
            CaveShList.push_back(*searchsh);
        } else if (loc == OnEdge) {
            int pa;
            if (splitseg != nullptr && splitseg->sh != None) {
                splitseg->shver = 0;
                pa = sorg(*splitseg);
            } else {
                pa = sorg(*searchsh);
            }
            if (searchsh->sh != None) {
                neighsh = *searchsh;
                while (true) {
                    if (sorg(neighsh) != pa) sesymself(neighsh);
                    smarktest(neighsh);
                    CaveShList.push_back(neighsh);
                    CaveSegShList.push_back(neighsh);
                    spivotself(neighsh);
                    if (neighsh.sh == searchsh->sh) break;
                    if (neighsh.sh == None) break;
                }
            }
        } else if (loc == OnVertex) {
            return loc;
        } else if (loc == Outside) {
            // Grow the facet's convex hull to take p in.
            // dummypoint carries an above point of the facet, so every 2d orientation here is an orient3d against it.
            neighsh = *searchsh;
            while (true) {
                senext2self(neighsh);
                spivot(neighsh, casout);
                if (casout.sh == None) {
                    ori = orient3d(P(sorg(neighsh)), P(sdest(neighsh)), P(DummyPoint), P(insertpt));
                    if (ori < 0) *searchsh = neighsh;
                    else break;
                } else {
                    if (sorg(casout) != sdest(neighsh)) sesymself(casout);
                    neighsh = casout;
                }
            }
            casin.sh = None;
            int pa = sorg(*searchsh), pb = sdest(*searchsh);
            while (true) {
                makesubface(&newsh);
                setshvertices(newsh, pb, pa, insertpt);
                setshellmark(newsh, shellmark(*searchsh));
                if (CheckConstraints) setareabound(newsh, areabound(*searchsh));
                if (UseInsertRadius) setfacetindex(newsh, getfacetindex(*searchsh));
                sbond1(newsh, *searchsh);
                sbond1(*searchsh, newsh);
                if (casin.sh != None) {
                    senext(newsh, casout);
                    sbond1(casout, casin);
                    sbond1(casin, casout);
                }
                senext2(newsh, casin);
                smarktest(newsh);
                CaveShList.push_back(newsh);
                neighsh = *searchsh;
                while (true) {
                    senextself(neighsh);
                    spivot(neighsh, casout);
                    if (casout.sh == None) {
                        *searchsh = neighsh;
                        break;
                    }
                    if (sorg(casout) != sdest(neighsh)) sesymself(casout);
                    neighsh = casout;
                }
                pa = sorg(*searchsh);
                pb = sdest(*searchsh);
                ori = orient3d(P(pa), P(pb), P(DummyPoint), P(insertpt));
                if (ori >= 0) break;
            }
        }

        // Grow the Bowyer-Watson sub-cavity.
        for (size_t i = 0; i < CaveShList.size(); ++i) {
            cavesh = CaveShList[i];
            for (int j = 0; j < 3; ++j) {
                if (!isshsubseg(cavesh)) {
                    spivot(cavesh, neighsh);
                    if (neighsh.sh != None) {
                        if (!smarktested(neighsh)) {
                            if (bowywat) {
                                if (loc == InStar) {
                                    sign = 1;
                                } else if (!isshtet(neighsh)) {
                                    sign = incircle3d(P(sorg(neighsh)), P(sdest(neighsh)), P(sapex(neighsh)), P(insertpt));
                                } else {
                                    sign = 1;
                                }
                                if (sign < 0) {
                                    smarktest(neighsh);
                                    CaveShList.push_back(neighsh);
                                }
                            } else {
                                sign = 1;
                            }
                        } else {
                            sign = -1;
                        }
                    } else {
                        if (loc == Outside) {
                            sign = (sorg(cavesh) == insertpt || sdest(cavesh) == insertpt) ? -1 : 1;
                        } else {
                            sign = 1;
                        }
                    }
                } else {
                    sign = 1; // never cross a segment
                }
                if (sign >= 0) CaveShBdList.push_back(cavesh);
                senextself(cavesh);
            }
        }

        // One new subface per boundary edge of the sub-cavity.
        for (auto &parysh : CaveShBdList) {
            sspivot(parysh, checkseg);
            if ((parysh.shver & 1) != 0) sesymself(parysh);
            const int pa = sorg(parysh), pb = sdest(parysh);
            makesubface(&newsh);
            setshvertices(newsh, pa, pb, insertpt);
            setshellmark(newsh, shellmark(parysh));
            if (CheckConstraints) setareabound(newsh, areabound(parysh));
            if (UseInsertRadius) setfacetindex(newsh, getfacetindex(parysh));
            if (pointtype(pa) == FreeFacetVertex) setpoint2sh(pa, SEncode(newsh));
            if (pointtype(pb) == FreeFacetVertex) setpoint2sh(pb, SEncode(newsh));
            spivot(parysh, casout);
            if (casout.sh != None) {
                casin = casout;
                if (checkseg.sh != None) {
                    checkseg.shver = 0;
                    if (sorg(newsh) != sorg(checkseg)) {
                        sesymself(newsh);
                        sesymself(parysh);
                    }
                    spivot(casin, neighsh);
                    while (neighsh.sh != parysh.sh) {
                        casin = neighsh;
                        spivot(casin, neighsh);
                    }
                }
                sbond1(newsh, casout);
                sbond1(casin, newsh);
            }
            if (checkseg.sh != None) ssbond(newsh, checkseg);
            sbond1(parysh, newsh);
        }

        if (newsh.sh != None) RecentSh = newsh;
        if (pointtype(insertpt) == FreeFacetVertex) setpoint2sh(insertpt, SEncode(newsh));

        // Bond the new subfaces to each other.
        for (auto &parysh : CaveShBdList) {
            spivot(parysh, newsh);
            senextself(newsh); // at edge [b, p]
            spivot(newsh, neighsh);
            if (neighsh.sh == None) {
                const int pb = sdest(parysh);
                neighsh = parysh;
                while (true) {
                    senextself(neighsh);
                    spivotself(neighsh);
                    if (neighsh.sh == None) break;
                    if (!smarktested(neighsh)) break;
                    if (sdest(neighsh) != pb) sesymself(neighsh);
                }
                if (neighsh.sh != None) {
                    if (sorg(neighsh) != pb) sesymself(neighsh);
                    senext2self(neighsh);
                    sbond(newsh, neighsh);
                }
            }
            spivot(parysh, newsh);
            senext2self(newsh); // at edge [p, a]
            spivot(newsh, neighsh);
            if (neighsh.sh == None) {
                const int pa = sorg(parysh);
                neighsh = parysh;
                while (true) {
                    senext2self(neighsh);
                    spivotself(neighsh);
                    if (neighsh.sh == None) break;
                    if (!smarktested(neighsh)) break;
                    if (sorg(neighsh) != pa) sesymself(neighsh);
                }
                if (neighsh.sh != None) {
                    if (sdest(neighsh) != pa) sesymself(neighsh);
                    senextself(neighsh);
                    sbond(newsh, neighsh);
                }
            }
        }

        if (loc == OnEdge || (splitseg != nullptr && splitseg->sh != None) || !CaveSegShList.empty()) {
            // An edge is being split.
            // Any new face degenerate on that edge is squeezed out here.
            Facet aseg, bseg, aoutseg, boutseg;
            for (size_t i = 0; i < CaveSegShList.size(); ++i) {
                Facet parysh = CaveSegShList[i];
                spivot(parysh, cavesh);
                if (sapex(cavesh) != insertpt) continue;
                if (CaveSegShList.size() > 1) {
                    const size_t j = (i + 1) % CaveSegShList.size();
                    parysh = CaveSegShList[j];
                    spivot(parysh, neighsh);
                    if (sorg(neighsh) != sorg(cavesh)) sesymself(neighsh);
                    for (int k = 0; k < 2; ++k) {
                        senextself(cavesh);
                        senextself(neighsh);
                        spivot(cavesh, newsh);
                        spivot(neighsh, casout);
                        sbond1(newsh, casout);
                    }
                } else {
                    for (int k = 0; k < 2; ++k) {
                        senextself(cavesh);
                        spivot(cavesh, newsh);
                        sdissolve(newsh);
                    }
                }
                if (pointtype(insertpt) == FreeFacetVertex) setpoint2sh(insertpt, SEncode(newsh));
            }

            if (splitseg != nullptr && splitseg->sh != None) {
                if (loc != InStar) smarktest(*splitseg);
                const int pa = sorg(*splitseg), pb = sdest(*splitseg);
                makesubseg(&aseg);
                makesubseg(&bseg);
                setshvertices(aseg, pa, insertpt, None);
                setshvertices(bseg, insertpt, pb, None);
                setshellmark(aseg, shellmark(*splitseg));
                setshellmark(bseg, shellmark(*splitseg));
                if (CheckConstraints) {
                    setareabound(aseg, areabound(*splitseg));
                    setareabound(bseg, areabound(*splitseg));
                }
                if (UseInsertRadius) {
                    setfacetindex(aseg, getfacetindex(*splitseg));
                    setfacetindex(bseg, getfacetindex(*splitseg));
                }
                senext2(*splitseg, boutseg);
                spivotself(boutseg);
                if (boutseg.sh != None) {
                    senext2(aseg, aoutseg);
                    sbond(boutseg, aoutseg);
                }
                senext(*splitseg, aoutseg);
                spivotself(aoutseg);
                if (aoutseg.sh != None) {
                    senext(bseg, boutseg);
                    sbond(boutseg, aoutseg);
                }
                senext(aseg, aoutseg);
                senext2(bseg, boutseg);
                sbond(aoutseg, boutseg);

                for (auto &parysh : CaveSegShList) {
                    spivot(parysh, neighsh);
                    if (sorg(neighsh) != pa) sesymself(neighsh);
                    senext2(neighsh, newsh);
                    spivotself(newsh); // the edge [p, a]
                    ssbond(newsh, aseg);
                    senext(neighsh, newsh);
                    spivotself(newsh); // the edge [b, p]
                    ssbond(newsh, bseg);
                }

                if (pointtype(insertpt) == FreeSegVertex) setpoint2sh(insertpt, SEncode(aseg));
                if (pointtype(pa) == FreeSegVertex) setpoint2sh(pa, SEncode(aseg));
                if (pointtype(pb) == FreeSegVertex) setpoint2sh(pb, SEncode(bseg));
            }

            for (auto &parysh : CaveSegShList) {
                spivotself(parysh);
                if (sapex(parysh) == insertpt) subfacedealloc(parysh.sh);
            }
            CaveSegShList.clear();

            if (splitseg != nullptr && splitseg->sh != None) {
                CaveSegShList.push_back(aseg);
                CaveSegShList.push_back(bseg);
            }
        }
        return loc;
    }

    // Put an above point of the facet into dummypoint, and report the three points of the largest triangle it found on the way.
    bool calculateabovepoint(const std::vector<int> &facpoints, int *ppa, int *ppb, int *ppc) {
        const int pa = facpoints[0];
        int pb = None, pc = None;
        double lab = 0;
        for (size_t i = 1; i < facpoints.size(); ++i) {
            const dvec3 v = P(facpoints[i]) - P(pa);
            const double len = dot(v, v);
            if (len > lab) {
                lab = len;
                pb = facpoints[i];
            }
        }
        lab = std::sqrt(lab);
        if (lab == 0) return false; // every point of the facet is one point
        const dvec3 v1 = P(pb) - P(pa);
        double A = 0;
        for (size_t i = 1; i < facpoints.size(); ++i) {
            const dvec3 n = cross(v1, P(facpoints[i]) - P(pa));
            const double area = dot(n, n);
            if (area > A) {
                A = area;
                pc = facpoints[i];
            }
        }
        if (A == 0) return false; // every point of the facet is collinear
        dvec3 n;
        facenormal(P(pa), P(pb), P(pc), n, 1, nullptr);
        n /= std::sqrt(dot(n, n));
        Pts[DummyPoint].Pos = P(pa) + (lab / 2.0) * n;
        if (ppa != nullptr) {
            *ppa = pa;
            *ppb = pb;
            *ppc = pc;
        }
        return true;
    }

    void calculateabovepoint4(const dvec3 &pa, const dvec3 &pb, const dvec3 &pc, const dvec3 &pd) {
        dvec3 n1, n2;
        facenormal(pa, pb, pc, n1, 1, nullptr);
        const double len1 = std::sqrt(dot(n1, n1));
        facenormal(pa, pb, pd, n2, 1, nullptr);
        const double len2 = std::sqrt(dot(n2, n2));
        const dvec3 norm = len1 > len2 ? n1 / len1 : n2 / len2;
        Pts[DummyPoint].Pos = pa + distance(pa, pb) * norm;
    }

    // Locate a point in the triangulation of a facet.
    // dummypoint holds an above point, so every 2d orientation is an orient3d against it.
    // The coordinate and the point index are separate because hole seeds are located by position and belong to no vertex.
    int slocate_at(const dvec3 &q, int qidx, Facet *searchsh, int aflag, int cflag, int rflag) {
        Facet neighsh;
        int loc;
        enum {
            MoveBC,
            MoveCA
        } nextmove = MoveBC;

        int pa = sorg(*searchsh), pb = sdest(*searchsh), pc = sapex(*searchsh);
        if (!aflag) calculateabovepoint4(P(pa), P(pb), P(pc), q);

        double ori = orient3d(P(pa), P(pb), P(pc), P(DummyPoint));
        if (ori > 0) sesymself(*searchsh);
        else if (ori == 0.0) return LocUnknown;

        int i;
        for (i = 0; i < 3; ++i) {
            pa = sorg(*searchsh);
            pb = sdest(*searchsh);
            ori = orient3d(P(pa), P(pb), P(DummyPoint), q);
            if (ori > 0) break;
            senextself(*searchsh);
        }
        if (i == 3) return LocUnknown;

        pc = sapex(*searchsh);
        if (pc == qidx) {
            senext2self(*searchsh);
            return OnVertex;
        }

        while (true) {
            const double ori_bc = orient3d(P(pb), P(pc), P(DummyPoint), q);
            const double ori_ca = orient3d(P(pc), P(pa), P(DummyPoint), q);
            if (ori_bc < 0) {
                nextmove = ori_ca < 0 ? (randomnation(2) ? MoveCA : MoveBC) : MoveBC;
            } else {
                if (ori_ca < 0) {
                    nextmove = MoveCA;
                } else {
                    if (ori_bc > 0) {
                        if (ori_ca > 0) {
                            loc = OnFace;
                            break;
                        }
                        senext2self(*searchsh); // on edge [c, a]
                        loc = OnEdge;
                        break;
                    }
                    if (ori_ca > 0) {
                        senextself(*searchsh); // on edge [b, c]
                        loc = OnEdge;
                        break;
                    }
                    senext2self(*searchsh); // coincident with c
                    return OnVertex;
                }
            }

            if (nextmove == MoveBC) senextself(*searchsh);
            else senext2self(*searchsh);
            if (!cflag && isshsubseg(*searchsh)) return EncSegment;
            spivot(*searchsh, neighsh);
            if (neighsh.sh == None) return Outside; // a hull edge
            if (sorg(neighsh) != sdest(*searchsh)) sesymself(neighsh);
            *searchsh = neighsh;
            pa = sorg(*searchsh);
            pb = sdest(*searchsh);
            pc = sapex(*searchsh);
            if (pc == qidx) {
                senext2self(*searchsh);
                return OnVertex;
            }
        }

        if (rflag) {
            // Round the verdict: a sliver of area against the triangle's own counts as zero.
            dvec3 n;
            pa = sorg(*searchsh);
            pb = sdest(*searchsh);
            pc = sapex(*searchsh);
            facenormal(P(pa), P(pb), P(pc), n, 1, nullptr);
            const double area_abc = std::sqrt(dot(n, n));
            facenormal(P(pb), P(pc), q, n, 1, nullptr);
            double area_bcp = std::sqrt(dot(n, n));
            if (area_bcp / area_abc < Epsilon) area_bcp = 0;
            facenormal(P(pc), P(pa), q, n, 1, nullptr);
            double area_cap = std::sqrt(dot(n, n));
            if (area_cap / area_abc < Epsilon) area_cap = 0;
            double area_abp = 0;
            if (loc == OnFace || loc == Outside) {
                facenormal(P(pa), P(pb), q, n, 1, nullptr);
                area_abp = std::sqrt(dot(n, n));
                if (area_abp / area_abc < Epsilon) area_abp = 0;
            }
            if (area_abp == 0) {
                if (area_bcp == 0) {
                    senextself(*searchsh);
                    loc = OnVertex; // close to b
                } else {
                    loc = area_cap == 0 ? OnVertex : OnEdge; // close to a, or on [a,b]
                }
            } else if (area_bcp == 0) {
                if (area_cap == 0) {
                    senext2self(*searchsh);
                    loc = OnVertex; // close to c
                } else {
                    senextself(*searchsh);
                    loc = OnEdge; // on [b,c]
                }
            } else if (area_cap == 0) {
                senext2self(*searchsh);
                loc = OnEdge; // on [c,a]
            } else {
                loc = OnFace;
            }
        }
        return loc;
    }
    int slocate(int searchpt, Facet *searchsh, int aflag, int cflag, int rflag) {
        return slocate_at(P(searchpt), searchpt, searchsh, aflag, cflag, rflag);
    }

    // Find the segment from sorg(searchsh) to endpt in the facet's triangulation, flipping the first edge it crosses and recurring.
    int sscoutsegment(Facet *searchsh, int endpt, int insertsegflag, int reporterrorflag, int chkencflag) {
        Facet flipshs[2], neighsh;
        int dir;
        enum {
            MoveAB,
            MoveCA
        } nextmove = MoveAB;

        const int startpt = sorg(*searchsh);
        const double len = distance(P(startpt), P(endpt));
        int pb = None, pc = None;

        while (true) {
            pb = sdest(*searchsh);
            if (pb == endpt) {
                dir = ShareEdge;
                break;
            }
            pc = sapex(*searchsh);
            if (pc == endpt) {
                senext2self(*searchsh);
                sesymself(*searchsh);
                dir = ShareEdge;
                break;
            }

            const double ori_ab = std::sqrt(triarea(P(startpt), P(pb), P(endpt))) / len < Epsilon ? 0.0 : orient3d(P(startpt), P(pb), P(DummyPoint), P(endpt));
            const double ori_ca = std::sqrt(triarea(P(pc), P(startpt), P(endpt))) / len < Epsilon ? 0.0 : orient3d(P(pc), P(startpt), P(DummyPoint), P(endpt));

            if (ori_ab < 0) {
                nextmove = ori_ca < 0 ? (randomnation(2) ? MoveCA : MoveAB) : MoveAB;
            } else {
                if (ori_ca < 0) {
                    nextmove = MoveCA;
                } else {
                    if (ori_ab > 0) {
                        if (ori_ca > 0) {
                            dir = AcrossEdge; // the segment cuts edge [b, c]
                            break;
                        }
                        senext2self(*searchsh); // collinear with edge [c, a]
                        sesymself(*searchsh);
                        dir = AcrossVert;
                        break;
                    }
                    if (ori_ca > 0) {
                        dir = AcrossVert; // collinear with edge [a, b]
                        break;
                    }
                    return Disjoint; // startpt == endpt, which cannot happen
                }
            }

            if (nextmove == MoveAB) {
                if (chkencflag && isshsubseg(*searchsh)) return AcrossEdge;
                spivot(*searchsh, neighsh);
                if (neighsh.sh != None) {
                    if (sorg(neighsh) != pb) sesymself(neighsh);
                    senext(neighsh, *searchsh);
                } else {
                    // This side is outside, from rounding.
                    // Take the other one.
                    senext2(*searchsh, neighsh);
                    if (chkencflag && isshsubseg(neighsh)) {
                        *searchsh = neighsh;
                        return AcrossEdge;
                    }
                    spivotself(neighsh);
                    if (sdest(neighsh) != pc) sesymself(neighsh);
                    *searchsh = neighsh;
                }
            } else {
                senext2(*searchsh, neighsh);
                if (chkencflag && isshsubseg(neighsh)) {
                    *searchsh = neighsh;
                    return AcrossEdge;
                }
                spivotself(neighsh);
                if (neighsh.sh != None) {
                    if (sdest(neighsh) != pc) sesymself(neighsh);
                    *searchsh = neighsh;
                } else {
                    if (chkencflag && isshsubseg(*searchsh)) return AcrossEdge;
                    spivot(*searchsh, neighsh);
                    if (sorg(neighsh) != pb) sesymself(neighsh);
                    senext(neighsh, *searchsh);
                }
            }
        }

        if (dir == ShareEdge) {
            if (insertsegflag) {
                Facet newseg;
                makesubseg(&newseg);
                setshvertices(newseg, startpt, endpt, None);
                setshellmark(newseg, -1);
                ssbond(*searchsh, newseg);
                spivot(*searchsh, neighsh);
                if (neighsh.sh != None) ssbond(neighsh, newseg);
            }
            return dir;
        }
        if (dir == AcrossVert) return dir; // a vertex lies inside the segment

        // Edge [b, c] crosses the segment: flip it, unless it is itself a segment.
        senext(*searchsh, flipshs[0]);
        if (isshsubseg(flipshs[0])) {
            (void)reporterrorflag;
            return dir; // two input segments intersect
        }
        spivot(flipshs[0], flipshs[1]);
        if (sorg(flipshs[1]) != sdest(flipshs[0])) sesymself(flipshs[1]);
        flip22(flipshs, 1, 0);
        // The flip can leave an inverted triangle behind: queue whichever side turned over.
        {
            const int pa = sapex(flipshs[1]), pbb = sapex(flipshs[0]);
            const int pcc = sorg(flipshs[0]), pd = sdest(flipshs[0]);
            const double ori_ab = orient3d(P(pcc), P(pd), P(DummyPoint), P(pbb));
            const double ori_ca = orient3d(P(pd), P(pcc), P(DummyPoint), P(pa));
            if (ori_ab <= 0) flipshpush(&flipshs[0]);
            else if (ori_ca <= 0) flipshpush(&flipshs[1]);
        }
        *searchsh = flipshs[0];
        return sscoutsegment(searchsh, endpt, insertsegflag, reporterrorflag, chkencflag);
    }

    // Drop the triangles of the facet's triangulation that lie outside its segments.
    void scarveholes(const std::vector<dvec3> &holelist) {
        Facet searchsh, neighsh;
        smarktest(RecentSh);
        CaveShList.push_back(RecentSh);
        for (size_t i = 0; i < CaveShList.size(); ++i) {
            searchsh = CaveShList[i];
            searchsh.shver = 0;
            for (int j = 0; j < 3; ++j) {
                spivot(searchsh, neighsh);
                if (neighsh.sh != None) {
                    if (!smarktested(neighsh)) {
                        smarktest(neighsh);
                        CaveShList.push_back(neighsh);
                    }
                } else if (!isshsubseg(searchsh)) {
                    // A hull side no segment protects: the outside starts here.
                    if (!sinfected(searchsh)) {
                        sinfect(searchsh);
                        CaveShBdList.push_back(searchsh);
                    }
                }
                senextself(searchsh);
            }
        }

        for (const dvec3 &h : holelist) {
            searchsh = RecentSh;
            if (slocate_at(h, None, &searchsh, 1, 1, 0) != Outside) {
                sinfect(searchsh);
                CaveShBdList.push_back(searchsh);
            }
        }

        for (size_t i = 0; i < CaveShBdList.size(); ++i) {
            searchsh = CaveShBdList[i];
            searchsh.shver = 0;
            for (int j = 0; j < 3; ++j) {
                spivot(searchsh, neighsh);
                if (neighsh.sh != None) {
                    if (!isshsubseg(searchsh)) {
                        if (!sinfected(neighsh)) {
                            sinfect(neighsh);
                            CaveShBdList.push_back(neighsh);
                        }
                    } else {
                        sdissolve(neighsh); // cut a protected face loose
                    }
                }
                senextself(searchsh);
            }
        }

        for (auto &parysh : CaveShList) {
            if (sinfected(parysh)) subfacedealloc(parysh.sh);
            else sunmarktest(parysh);
        }
        CaveShList.clear();
        CaveShBdList.clear();
    }

    double MinFacetDihed{PI};

    // Iterate the live records of one shell pool.
    template<typename F> void forallsubfaces(F &&fn) {
        for (int i = 0; i < int(Shells.size()); ++i) {
            if (Shells[i].Kind == 0 && Shells[i].V[0] != None) fn(i);
        }
    }
    template<typename F> void forallsubsegs(F &&fn) {
        for (int i = 0; i < int(Shells.size()); ++i) {
            if (Shells[i].Kind == 1 && Shells[i].V[0] != None) fn(i);
        }
    }

    // The subfaces incident at each vertex, in one flat array indexed by a per-point offset table.
    void makepoint2submap(bool isseg, std::vector<int> &idx2faclist, std::vector<Facet> &facperverlist) {
        const int np = int(Pts.size());
        idx2faclist.assign(np + 1, 0);
        const auto forall = [&](auto &&fn) {
            for (int i = 0; i < int(Shells.size()); ++i) {
                if (Shells[i].Kind == (isseg ? 1 : 0) && Shells[i].V[0] != None) fn(i);
            }
        };
        forall([&](int sh) {
            ++idx2faclist[Shells[sh].V[0]];
            ++idx2faclist[Shells[sh].V[1]];
            if (Shells[sh].V[2] != None) ++idx2faclist[Shells[sh].V[2]];
        });
        int j = idx2faclist[0];
        idx2faclist[0] = 0;
        for (int i = 0; i < np; ++i) {
            const int k = idx2faclist[i + 1];
            idx2faclist[i + 1] = idx2faclist[i] + j;
            j = k;
        }
        facperverlist.assign(idx2faclist[np], Facet{});
        forall([&](int sh) {
            int v = Shells[sh].V[0];
            facperverlist[idx2faclist[v]++] = Facet{sh, 0};
            if (Shells[sh].V[2] != None) {
                v = Shells[sh].V[1];
                facperverlist[idx2faclist[v]++] = Facet{sh, 2};
                v = Shells[sh].V[2];
                facperverlist[idx2faclist[v]++] = Facet{sh, 4};
            } else {
                v = Shells[sh].V[1];
                facperverlist[idx2faclist[v]++] = Facet{sh, 1};
            }
        });
        for (int i = np - 1; i >= 0; --i) idx2faclist[i + 1] = idx2faclist[i];
        idx2faclist[0] = 0;
    }

    // Build a constrained Delaunay triangulation of one facet from its vertex set and its boundary segments.
    int triangulate(int shmark, std::vector<int> &ptlist, std::vector<std::array<int, 2>> &conlist, const std::vector<dvec3> &holelist) {
        Facet searchsh, newsh, newseg;
        int pa, pb, pc;

        if (ptlist.size() < 2) return 1; // neither a segment nor a facet
        if (ptlist.size() == 2) {
            pa = ptlist[0];
            pb = ptlist[1];
            if (distance(P(pa), P(pb)) > 0) {
                makesubseg(&newseg);
                setshvertices(newseg, pa, pb, None);
                setshellmark(newseg, -1);
            }
            if (pointtype(pa) == VolVertex) setpointtype(pa, FacetVertex);
            if (pointtype(pb) == VolVertex) setpointtype(pb, FacetVertex);
            return 1;
        }
        if (ptlist.size() == 3) {
            pa = ptlist[0];
            pb = ptlist[1];
            pc = ptlist[2];
        } else {
            if (!calculateabovepoint(ptlist, &pa, &pb, &pc)) return 0; // degenerate point set
        }

        makesubface(&newsh);
        setshvertices(newsh, pa, pb, pc);
        setshellmark(newsh, shmark);
        RecentSh = newsh;
        if (pointtype(pa) == VolVertex) setpointtype(pa, FacetVertex);
        if (pointtype(pb) == VolVertex) setpointtype(pb, FacetVertex);
        if (pointtype(pc) == VolVertex) setpointtype(pc, FacetVertex);

        if (ptlist.size() == 3) {
            for (int i = 0; i < 3; ++i) {
                makesubseg(&newseg);
                setshvertices(newseg, sorg(newsh), sdest(newsh), None);
                setshellmark(newseg, -1);
                ssbond(newsh, newseg);
                senextself(newsh);
            }
            return 1;
        }

        // Everything created here is remembered, so a failed triangulation can be undone.
        CaveEncShList.push_back(newsh);

        pinfect(pa);
        pinfect(pb);
        pinfect(pc);
        size_t i = 0;
        for (; i < ptlist.size(); ++i) {
            const int ppt = ptlist[i];
            if (pinfected(ppt)) continue;
            searchsh = RecentSh;
            const int iloc = sinsertvertex(ppt, &searchsh, nullptr, Outside, 1, 1);
            if (iloc == OnVertex) break; // the facet triangulation failed
            if (pointtype(ppt) == VolVertex) setpointtype(ppt, FacetVertex);
            for (auto &parysh : CaveShBdList) {
                spivot(parysh, searchsh);
                if (!isdeadsh(searchsh)) CaveEncShList.push_back(searchsh);
            }
            for (auto &parysh : CaveShList) subfacedealloc(parysh.sh);
            CaveShBdList.clear();
            CaveShList.clear();
            CaveSegShList.clear();
        }
        puninfect(pa);
        puninfect(pb);
        puninfect(pc);

        if (i < ptlist.size()) {
            for (auto &parysh : CaveEncShList) {
                if (!isdeadsh(parysh)) subfacedealloc(parysh.sh);
            }
            CaveEncShList.clear();
            return 0;
        }

        size_t c = 0;
        for (; c < conlist.size(); ++c) {
            const int c0 = conlist[c][0], c1 = conlist[c][1];
            searchsh = RecentSh;
            int iloc = slocate(c0, &searchsh, 1, 1, 0);
            if (iloc != OnVertex) {
                // Rounding lost it: sweep every subface of this facet for the vertex.
                bool bflag = false;
                Facet found{None, 0};
                for (int sh = 0; sh < int(Shells.size()) && !bflag; ++sh) {
                    if (Shells[sh].Kind != 0 || Shells[sh].V[0] == None) continue;
                    if (shellmark(Facet{sh, 0}) != shmark) continue;
                    Facet cand{sh, 0};
                    if (Shells[sh].V[0] == c0) cand.shver = 0;
                    else if (Shells[sh].V[1] == c0) cand.shver = 2;
                    else if (Shells[sh].V[2] == c0) cand.shver = 4;
                    else continue;
                    // Sharing the facet mark is not enough: the subface must also be coplanar.
                    const int qa = sorg(cand), qb = sdest(cand), qc = sapex(cand);
                    const double chkori = orient3d(P(qa), P(qb), P(qc), P(c1));
                    if (chkori == 0.0) {
                        found = cand;
                        bflag = true;
                    } else {
                        double len = distance(P(qa), P(qb)) + distance(P(qb), P(qc)) + distance(P(qc), P(qa));
                        len /= 3.0;
                        if (std::abs(chkori) / (len * len * len) < 1e-5) {
                            found = cand;
                            bflag = true;
                        }
                    }
                }
                searchsh = found;
            }
            if (searchsh.sh == None) break;
            if (sscoutsegment(&searchsh, c1, 1, 1, 0) != ShareEdge) break;
            sspivot(searchsh, newseg);
            CaveEncSegList.push_back(newseg);
            if (!FlipStack.empty()) lawsonflip();
        }

        if (c < conlist.size()) {
            for (auto &parysh : CaveEncShList) {
                if (!isdeadsh(parysh)) subfacedealloc(parysh.sh);
            }
            for (auto &paryseg : CaveEncSegList) {
                if (!isdeadsh(paryseg)) subsegdealloc(paryseg.sh);
            }
            CaveEncShList.clear();
            CaveEncSegList.clear();
            return 0;
        }

        scarveholes(holelist);
        CaveEncShList.clear();
        CaveEncSegList.clear();
        return 1;
    }

    // Drop duplicate segments and build the ring of subfaces around each one, ordered by dihedral angle about the segment.
    void unifysegments() {
        std::vector<int> idx2faclist;
        std::vector<Facet> facperverlist;
        makepoint2submap(false, idx2faclist, facperverlist);

        std::vector<Facet> facelink;
        std::vector<int> segs;
        forallsubsegs([&](int sh) { segs.push_back(sh); });

        for (const int segsh : segs) {
            if (Shells[segsh].V[0] == None) continue; // already dropped as a duplicate
            Facet subsegloop{segsh, 0};
            const int torg = sorg(subsegloop), tdest = sdest(subsegloop);
            facelink.clear();

            for (int k = idx2faclist[torg]; k < idx2faclist[torg + 1]; ++k) {
                Facet sface = facperverlist[k];
                if (isdeadsh(sface)) continue;
                if (sdest(sface) != tdest) {
                    senext2self(sface);
                    sesymself(sface);
                }
                if (sdest(sface) != tdest) continue;

                if (facelink.size() >= 2) {
                    size_t idx = 0;
                    for (size_t m = 0; m + 1 < facelink.size(); ++m) {
                        const double ori1 = facedihedral(P(torg), P(tdest), P(sapex(facelink[idx])), P(sapex(facelink[idx + 1])));
                        const double ori2 = facedihedral(P(torg), P(tdest), P(sapex(facelink[idx])), P(sapex(sface)));
                        if (ori1 >= ori2) break;
                        ++idx;
                    }
                    facelink.insert(facelink.begin() + long(idx) + 1, sface);
                } else {
                    facelink.push_back(sface);
                }
            }

            // Bond every subface of the ring to this segment, dropping any other segment on it.
            for (auto &f : facelink) {
                Facet testseg;
                sspivot(f, testseg);
                if (testseg.sh != None && testseg.sh != subsegloop.sh && !isdeadsh(testseg)) {
                    subsegdealloc(testseg.sh);
                }
                ssbond(f, subsegloop);
            }

            if (facelink.size() > 1) {
                for (size_t k = 0; k < facelink.size(); ++k) {
                    const Facet &f1 = facelink[k];
                    const Facet &f2 = facelink[(k + 1) % facelink.size()];
                    dvec3 n1, n2;
                    facenormal(P(torg), P(tdest), P(sapex(f1)), n1, 1, nullptr);
                    facenormal(P(torg), P(tdest), P(sapex(f2)), n2, 1, nullptr);
                    double cosang = dot(n1, n2) / (std::sqrt(dot(n1, n1)) * std::sqrt(dot(n2, n2)));
                    if (cosang > 1.0) cosang = 1.0;
                    else if (cosang < -1.0) cosang = -1.0;
                    const double ang = std::acos(cosang);
                    if (ang < MinFacetDihed) MinFacetDihed = ang;
                    sbond1(f1, f2);
                }
            }
        }
    }

    // Turn each edge given in the input into a segment.
    // A triangle-soup input supplies none, so the list this walks is empty.
    void identifyinputedges(const std::vector<std::array<int, 2>> &inedges) {
        if (inedges.empty()) return;
        std::vector<int> idx2shlist;
        std::vector<Facet> shperverlist;
        makepoint2submap(false, idx2shlist, shperverlist);

        for (const auto &e : inedges) {
            int endpts[2]{e[0], e[1]};
            if (endpts[0] == endpts[1]) continue;
            if (DupVerts > 0) {
                for (int j = 0; j < 2; ++j) {
                    if (pointtype(endpts[j]) == DuplicatedVertex) endpts[j] = point2ppt(endpts[j]);
                }
            }
            Facet newseg{None, 0}, searchsh{None, 0}, neighsh, checkseg;
            for (int j = idx2shlist[endpts[0]]; j < idx2shlist[endpts[0] + 1]; ++j) {
                if (sdest(shperverlist[j]) == endpts[1]) {
                    searchsh = shperverlist[j];
                    break;
                }
                if (sapex(shperverlist[j]) == endpts[1]) {
                    senext2(shperverlist[j], searchsh);
                    sesymself(searchsh);
                    break;
                }
            }
            if (searchsh.sh != None) {
                sspivot(searchsh, checkseg);
                if (checkseg.sh != None) {
                    newseg = checkseg;
                } else {
                    makesubseg(&newseg);
                    setshvertices(newseg, sorg(searchsh), sdest(searchsh), None);
                    ssbond(searchsh, newseg);
                    spivot(searchsh, neighsh);
                    if (neighsh.sh != None) ssbond(neighsh, newseg);
                }
            } else {
                // A dangling segment, in no facet at all.
                forallsubsegs([&](int sh) {
                    if (newseg.sh != None) return;
                    const int a = Shells[sh].V[0], b = Shells[sh].V[1];
                    if ((a == endpts[0] && b == endpts[1]) || (a == endpts[1] && b == endpts[0])) newseg = Facet{sh, 0};
                });
                if (newseg.sh == None) {
                    makesubseg(&newseg);
                    setshvertices(newseg, endpts[0], endpts[1], None);
                }
            }
            setshellmark(newseg, -2);
        }
    }

    // Triangulate every input facet, then unify the segments and mark their endpoints as ridge vertices.
    void meshsurface() {
        std::vector<int> ptlist;
        std::vector<std::array<int, 2>> conlist;
        const std::vector<dvec3> noholes;

        for (int shmark = 1; shmark <= int(InTris.size()); ++shmark) {
            const std::array<int, 3> &tri = InTris[shmark - 1];
            ptlist.clear();
            conlist.clear();
            // One polygon of three corners per input triangle, so the walk below reduces to three vertices and three sides.
            int end1 = tri[0];
            if (pointtype(end1) == DuplicatedVertex) end1 = point2ppt(end1);
            int tstart = end1;
            if (!pinfected(tstart)) {
                pinfect(tstart);
                ptlist.push_back(tstart);
            }
            for (int j = 1; j <= 3; ++j) {
                int end2 = tri[j % 3];
                if (pointtype(end2) == DuplicatedVertex) end2 = point2ppt(end2);
                if (end1 == end2) continue; // an isolated vertex of the facet
                const int tend = end2;
                if (!pinfected(tend)) {
                    pinfect(tend);
                    ptlist.push_back(tend);
                }
                conlist.push_back({tstart, tend});
                end1 = end2;
                tstart = tend;
            }
            for (int p : ptlist) puninfect(p);
            triangulate(-1, ptlist, conlist, noholes);
        }

        unifysegments();
        identifyinputedges({});

        forallsubsegs([&](int sh) {
            setpointtype(Shells[sh].V[0], RidgeVertex);
            setpointtype(Shells[sh].V[1], RidgeVertex);
        });
        InSegments = SubsegItems;
    }

    //=== Boundary recovery ===

    // Abandon the build.
    // Code 2 is an internal error, 3 a self-intersecting input surface, 4 an input feature too small to resolve.
    // The top level turns each into an error string.
    [[noreturn]] static void bail(int code) { throw TetError{code}; }

    bool issteinerpoint(int pt) const {
        const VertType t = pointtype(pt);
        return t == FreeSegVertex || t == FreeFacetVertex || t == FreeVolVertex;
    }

    // A vertex the recovery of `misseg` ran into.
    // It blocks the segment unless it is a Steiner point that the segment's own facet placed.
    bool blocks_segment(int nearpt, const Facet &misseg) const {
        if (!issteinerpoint(nearpt)) return true;
        if (pointtype(nearpt) != FreeSegVertex) bail(2);
        Facet parentseg;
        SDecode(point2sh(nearpt), parentseg);
        return getfacetindex(parentseg) != getfacetindex(misseg);
    }

    // Flip away whatever crosses the edge (startpt, endpt).
    // With fullsearch set it works along the whole run of crossings, not only the first.
    // Reports a self-intersecting input through idir.
    int recoveredgebyflips(int startpt, int endpt, Facet *sedge, Triface *searchtet, int fullsearch, int &idir) {
        FlipConstraints fc;
        idir = Disjoint;
        fc.seg[0] = startpt;
        fc.seg[1] = endpt;
        fc.checkflipeligibility = 1;

        while (true) {
            point2tetorg(startpt, *searchtet);
            int dir = finddirection(searchtet, endpt);

            if (dir == AcrossVert) {
                if (dest(*searchtet) == endpt) return 1;
                if (sedge != nullptr) {
                    // A vertex sits on the element being recovered.
                    // Whether that is a genuine self-intersection depends on what the vertex is.
                    const int nearpt = dest(*searchtet);
                    bool intersect_flag = false;
                    if (Shells[sedge->sh].V[2] == None) { // sedge is a segment
                        if (!issteinerpoint(nearpt)) {
                            intersect_flag = true;
                        } else if (pointtype(nearpt) == FreeSegVertex) {
                            const int segidx = getfacetindex(*sedge);
                            Facet parentseg;
                            SDecode(point2sh(nearpt), parentseg);
                            if (getfacetindex(parentseg) != segidx) intersect_flag = true;
                            else bail(2);
                        } else if (pointtype(nearpt) == FreeFacetVertex) {
                            intersect_flag = true;
                        } else {
                            bail(2);
                        }
                    } else { // sedge is an edge of a facet
                        if (!issteinerpoint(nearpt)) intersect_flag = true;
                        else if (pointtype(nearpt) == FreeSegVertex) intersect_flag = true;
                        else if (pointtype(nearpt) == FreeFacetVertex) intersect_flag = true;
                        else bail(2);
                    }
                    if (intersect_flag) idir = SelfIntersect;
                }
                return 0;
            }

            enextesymself(*searchtet); // the face the edge leaves through

            if (dir == AcrossFace) {
                if (CheckSubfaceFlag && issubface(*searchtet)) {
                    if (sedge) idir = SelfIntersect;
                    return 0;
                }
                if (removefacebyflips(searchtet, &fc)) continue;
            } else if (dir == AcrossEdge) {
                if (CheckSubsegFlag && issubseg(*searchtet)) {
                    if (sedge) idir = SelfIntersect;
                    return 0;
                }
                if (removeedgebyflips(searchtet, &fc) == 2) continue;
            } else {
                bail(2);
            }

            if (fullsearch) {
                // Work along the whole run of crossings rather than only the first one.
                Triface neightet, spintet;
                BadFace bakface;
                int types[2], poss[4], pos = 0;
                int success = 0;

                point2tetorg(startpt, *searchtet);
                dir = finddirection(searchtet, endpt);
                enextesymself(*searchtet);

                while (true) {
                    fsymself(*searchtet);
                    if (dir == AcrossFace) {
                        neightet = *searchtet;
                        const int j = neightet.ver & 3;
                        for (int i = j + 1; i < j + 4; ++i) {
                            neightet.ver = i % 4;
                            const int pa = org(neightet), pb = dest(neightet), pc = apex(neightet), pd = oppo(neightet);
                            if (tri_edge_test(P(pa), P(pb), P(pc), P(startpt), P(endpt), &P(pd), 1, types, poss)) {
                                dir = types[0];
                                pos = poss[0];
                                break;
                            }
                            dir = Disjoint;
                            pos = 0;
                        }
                        if (dir == Disjoint) bail(2);
                    } else if (dir == AcrossEdge) {
                        while (true) {
                            for (int i = 0; i < 2; ++i) {
                                if (i == 0) enextesym(*searchtet, neightet);
                                else eprevesym(*searchtet, neightet);
                                const int pa = org(neightet), pb = dest(neightet), pc = apex(neightet), pd = oppo(neightet);
                                if (tri_edge_test(P(pa), P(pb), P(pc), P(startpt), P(endpt), &P(pd), 1, types, poss)) {
                                    dir = types[0];
                                    pos = poss[0];
                                    break;
                                }
                                dir = Disjoint;
                                pos = 0;
                            }
                            if (dir != Disjoint) break;
                            fnextself(*searchtet);
                        }
                    } else {
                        bail(2);
                    }

                    for (int i = 0; i < pos; ++i) enextself(neightet);

                    if (dir == ShareVert) {
                        if (org(neightet) == endpt) break; // reached the far end without success
                        return 0;
                    }

                    *searchtet = neightet;
                    bakface.forg = org(*searchtet);
                    bakface.fdest = dest(*searchtet);
                    bakface.fapex = apex(*searchtet);
                    bakface.foppo = oppo(*searchtet);

                    if (dir == AcrossFace) {
                        if (CheckSubfaceFlag && issubface(*searchtet)) return 0;
                        if (removefacebyflips(searchtet, &fc)) {
                            success = 1;
                            break;
                        }
                    } else if (dir == AcrossEdge) {
                        if (CheckSubsegFlag && issubseg(*searchtet)) return 0;
                        if (removeedgebyflips(searchtet, &fc) == 2) {
                            success = 1;
                            break;
                        }
                    } else if (dir == AcrossVert) {
                        return 0;
                    } else {
                        bail(2);
                    }

                    // The flip failed, and it may have moved the face under us.
                    // Find it again.
                    if (searchtet->tet == None || org(*searchtet) != bakface.forg || dest(*searchtet) != bakface.fdest ||
                        apex(*searchtet) != bakface.fapex || oppo(*searchtet) != bakface.foppo) {
                        point2tetorg(bakface.forg, *searchtet);
                        const int dir1 = finddirection(searchtet, bakface.fdest);
                        if (dir1 == AcrossVert && dest(*searchtet) == bakface.fdest) {
                            spintet = *searchtet;
                            while (true) {
                                if (apex(spintet) == bakface.fapex) {
                                    *searchtet = spintet;
                                    break;
                                }
                                fnextself(spintet);
                                if (spintet.tet == searchtet->tet) {
                                    searchtet->tet = None;
                                    break;
                                }
                            }
                            if (searchtet->tet != None && oppo(*searchtet) != bakface.foppo) {
                                fsymself(*searchtet);
                                if (oppo(*searchtet) != bakface.foppo) searchtet->tet = None;
                            }
                        } else {
                            searchtet->tet = None;
                        }
                        if (searchtet->tet == None) {
                            success = 0;
                            break;
                        }
                    }
                }
                if (success) continue;
            }
            break;
        }
        return 0;
    }

    // Put a point inside the pocket the n tets around edge [a,b] form.
    // It is chosen along (p0, p_(n-1)) to maximise the smallest volume it makes with the pocket's outer faces, then moved inward by smoothing.
    int add_steinerpt_in_schoenhardtpoly(Triface *abtets, int n, int splitsliverflag, int chkencflag) {
        Triface worktet, faketet1, faketet2;
        InsertFlags ivf;
        OptParameters opm;

        if (splitsliverflag) {
            const int idx = int(Rng() % n);
            const int pa = org(abtets[idx]), pb = dest(abtets[idx]);
            const int pcc = apex(abtets[idx]), pdd = oppo(abtets[idx]);
            const int steinerpt = makepoint(FreeVolVertex);
            Pts[steinerpt].Pos = (P(pa) + P(pb) + P(pcc) + P(pdd)) / 4.0;
            worktet = abtets[idx];
            ivf.iloc = Outside;
            ivf.bowywat = 1;
            ivf.lawson = 2;
            ivf.rejflag = 0;
            ivf.chkencflag = chkencflag;
            ivf.validflag = 1;
            ivf.respectbdflag = 1;
            if (insertpoint(steinerpt, &worktet, nullptr, nullptr, &ivf)) {
                if (!FlipStack.empty()) recoverdelaunay();
                ++StVolRefCount;
                if (SteinerLeft > 0) --SteinerLeft;
                return 1;
            }
            pointdealloc(steinerpt);
            return 0;
        }

        const int pc = apex(abtets[0]);
        const int pd = oppo(abtets[n - 1]);

        for (int i = 0; i < n; ++i) {
            edestoppo(abtets[i], worktet); // [p_i, p_i+1, a]
            CaveTetList.push_back(worktet);
            eorgoppo(abtets[i], worktet); // [p_i+1, p_i, b]
            CaveTetList.push_back(worktet);
        }

        const int N = 100;
        const double stepi = 0.01;
        const dvec3 vcd = P(pd) - P(pc);
        double maxminvol = 0.0;
        int maxidx = 0;
        for (int it = 1; it < N; ++it) {
            // Per component, which is the form the sample points are compared in.
            dvec3 sampt;
            for (int i = 0; i < 3; ++i) sampt[i] = P(pc)[i] + (stepi * double(it)) * vcd[i];
            double minvol = 0;
            for (size_t i = 0; i < CaveTetList.size(); ++i) {
                const Triface &t = CaveTetList[i];
                const double ori = orient3d(P(dest(t)), P(org(t)), P(apex(t)), sampt);
                if (i == 0 || minvol > ori) minvol = ori;
            }
            if (it == 1 || maxminvol < minvol) {
                maxminvol = minvol;
                maxidx = it;
            }
        }
        if (maxminvol <= 0) {
            CaveTetList.clear();
            return 0;
        }
        dvec3 smtpt;
        for (int i = 0; i < 3; ++i) smtpt[i] = P(pc)[i] + (stepi * double(maxidx)) * vcd[i];

        // Two stand-in tets carry the pocket's two missing outer faces [d,c,a] and [c,d,b].
        maketetrahedron(&faketet1);
        setvertices(faketet1, pd, pc, org(abtets[0]), DummyPoint);
        CaveTetList.push_back(faketet1);
        maketetrahedron(&faketet2);
        setvertices(faketet2, pc, pd, dest(abtets[0]), DummyPoint);
        CaveTetList.push_back(faketet2);

        opm.max_min_volume = 1;
        opm.numofsearchdirs = 20;
        opm.searchstep = 0.001;
        opm.maxiter = 100;
        opm.initval = 0.0;

        int success = smoothpoint(smtpt, CaveTetList, 1, &opm);
        if (success) {
            while (opm.smthiter == 100) {
                opm.searchstep *= 10.0;
                opm.initval = opm.imprval;
                opm.smthiter = 0;
                smoothpoint(smtpt, CaveTetList, 1, &opm);
            }
        }
        tetrahedrondealloc(faketet1.tet);
        tetrahedrondealloc(faketet2.tet);
        CaveTetList.clear();
        if (!success) return 0;

        const int steinerpt = makepoint(FreeVolVertex);
        Pts[steinerpt].Pos = smtpt;
        for (int i = 0; i < n; ++i) {
            infect(abtets[i]);
            CaveOldTetList.push_back(abtets[i]);
        }
        worktet = abtets[0];
        ivf.iloc = InStar;
        ivf.chkencflag = chkencflag;
        if (insertpoint(steinerpt, &worktet, nullptr, nullptr, &ivf)) {
            ++StVolRefCount;
            if (SteinerLeft > 0) --SteinerLeft;
            return 1;
        }
        pointdealloc(steinerpt);
        return 0;
    }

    // Split a missing segment where it comes closest to whichever segment the flip search reported it fighting with.
    int add_steinerpt_in_segment(Facet *misseg, int searchlevel, int &idir) {
        Triface searchtet;
        Facet candseg;
        FlipConstraints fc;
        dvec3 Pp, Qq;
        double tp, tq;
        double smlen = 0, split = 0, split_q = 0;

        const int startpt = sorg(*misseg), endpt = sdest(*misseg);
        idir = Disjoint;
        fc.seg[0] = startpt;
        fc.seg[1] = endpt;
        fc.checkflipeligibility = 1;
        fc.collectencsegflag = 1;

        point2tetorg(startpt, searchtet);
        const int dir = finddirection(&searchtet, endpt);
        if (dir == AcrossVert) return 0;
        enextesymself(searchtet);

        const int bak_fliplinklevel = FlipLinkLevel;
        FlipLinkLevel = searchlevel;
        if (dir == AcrossFace) removefacebyflips(&searchtet, &fc);
        else if (dir == AcrossEdge) removeedgebyflips(&searchtet, &fc);

        for (auto &paryseg : CaveEncSegList) {
            suninfect(paryseg);
            const int pc0 = sorg(paryseg), pd0 = sdest(paryseg);
            tp = tq = 0;
            if (!linelineint(P(startpt), P(endpt), P(pc0), P(pd0), Pp, Qq, &tp, &tq, Epsilon)) continue;
            if (tp > 0 && tq < 1) {
                if (tp < 0.5) {
                    if (tp < Epsilon * 1e+3) tp = 0.0;
                } else if (1.0 - tp < Epsilon * 1e+3) {
                    tp = 1.0;
                }
            }
            if (tp <= 0 || tp >= 1) continue;
            if (tq > 0 && tq < 1) {
                if (tq < 0.5) {
                    if (tq < Epsilon * 1e+3) tq = 0.0;
                } else if (1.0 - tq < Epsilon * 1e+3) {
                    tq = 1.0;
                }
            }
            if (tq <= 0 || tq >= 1) continue;
            const double len = distance(Pp, Qq);
            if (split == 0 || len < smlen) {
                smlen = len;
                split = tp;
                split_q = tq;
                candseg = paryseg;
            }
        }
        CaveEncSegList.clear();
        FlipLinkLevel = bak_fliplinklevel;
        if (split == 0) return 0; // no crossing segment found

        Facet splitsh, splitseg;
        int steinerpt;
        if (AddSteinerAlgo == 1) {
            steinerpt = makepoint(FreeSegVertex);
            for (int i = 0; i < 3; ++i) Pts[steinerpt].Pos[i] = P(startpt)[i] + split * (P(endpt)[i] - P(startpt)[i]);
        } else {
            const dvec3 pp = P(startpt) + split * (P(endpt) - P(startpt));
            const int pc0 = sorg(candseg), pd0 = sdest(candseg);
            const dvec3 qq = P(pc0) + split_q * (P(pd0) - P(pc0));
            steinerpt = makepoint(FreeVolVertex);
            Pts[steinerpt].Pos = 0.5 * (pp + qq);
        }

        {
            const int pc0 = sorg(candseg), pd0 = sdest(candseg);
            if (is_collinear_at(steinerpt, pc0, pd0)) {
                // The two segments almost cross.
                // Loosen the tolerance once; if it cannot be loosened any further the input really does self-intersect.
                const double collinear_ang = interiorangle(P(steinerpt), P(pc0), P(pd0), nullptr) / PI * 180.0;
                const double new_ang_tol = collinear_ang + (collinear_ang - CollinearAngTol) / 180.0;
                if (new_ang_tol < 180.0) {
                    CollinearAngTol = new_ang_tol;
                    CosCollinearAngTol = std::cos(CollinearAngTol / 180.0 * PI);
                } else {
                    idir = SelfIntersect;
                    pointdealloc(steinerpt);
                    return 0;
                }
            }
        }

        point2tetorg(split < 0.5 ? startpt : endpt, searchtet);
        if (AddSteinerAlgo == 1) {
            splitseg = *misseg;
            spivot(*misseg, splitsh);
            setpoint2sh(steinerpt, SEncode(*misseg));
        } else {
            splitsh.sh = None;
            splitseg.sh = None;
        }

        InsertFlags ivf = boundary_split_flags(Outside);

        if (insertpoint(steinerpt, &searchtet, &splitsh, &splitseg, &ivf)) {
            if (!FlipStack.empty()) recoverdelaunay();
        } else {
            pointdealloc(steinerpt);
            return 0;
        }

        if (AddSteinerAlgo == 1) {
            SubVertStack.push_back(steinerpt);
            ++StSegRefCount;
        } else {
            SubSegStack.push_back(SEncode(*misseg));
            ++StVolRefCount;
        }
        if (SteinerLeft > 0) --SteinerLeft;
        return 1;
    }

    // Place a Steiner point that helps the edge (startpt, endpt) appear.
    // With splitsegflag clear it only tries the volume placements; with it set it may also split the segment itself.
    int add_steinerpt_to_recover_edge(int startpt, int endpt, Facet *misseg, int splitsegflag, int splitsliverflag, int &idir) {
        Triface searchtet, spintet;
        Facet splitsh;
        int types[2], poss[4];
        idir = Disjoint;

        if (misseg != nullptr) {
            startpt = sorg(*misseg);
            if (pointtype(startpt) == FreeSegVertex) {
                sesymself(*misseg);
                startpt = sorg(*misseg);
            }
            endpt = sdest(*misseg);
        }

        point2tetorg(startpt, searchtet);
        int dir = finddirection(&searchtet, endpt);

        if (dir == AcrossVert) {
            if (dest(searchtet) == endpt) {
                if (misseg != nullptr) SubSegStack.push_back(SEncode(*misseg));
                return 1;
            }
            if (misseg != nullptr) {
                if (blocks_segment(dest(searchtet), *misseg)) idir = SelfIntersect;
                else bail(2);
            }
            return 0;
        }

        enextself(searchtet);

        if (dir == AcrossFace) {
            // The segment cuts at least three faces.
            // Find the common edge of the first three.
            esymself(searchtet);
            fsym(searchtet, spintet);
            const int pd = oppo(spintet);

            if (pd == endpt) {
                if (misseg != nullptr) {
                    // A 2-3 flip recovers it unless the face is almost collinear with the segment.
                    double collinear_ang = 0.;
                    for (int k = 0; k < 3; ++k) {
                        const double ang = interiorangle(P(org(searchtet)), P(startpt), P(endpt), nullptr);
                        if (ang > collinear_ang) collinear_ang = ang;
                        enextself(searchtet);
                    }
                    collinear_ang = collinear_ang / PI * 180.0;
                    if (collinear_ang > CollinearAngTol) {
                        const double new_ang_tol = collinear_ang + (collinear_ang - CollinearAngTol) / 180.0;
                        if (new_ang_tol < 180.0) {
                            CollinearAngTol = new_ang_tol;
                            CosCollinearAngTol = std::cos(CollinearAngTol / 180.0 * PI);
                            SubSegStack.push_back(SEncode(*misseg));
                            return 1;
                        }
                        idir = SelfIntersect;
                        return 0;
                    }
                    SubSegStack.push_back(SEncode(*misseg));
                    return 1;
                }
                return 1;
            }

            if (issubface(searchtet)) {
                if (misseg != nullptr) bail(2); // a segment and a facet intersect
                if (misseg != nullptr) idir = SelfIntersect;
                return 0;
            }

            for (int i = 0; i < 3; ++i) {
                if (tri_edge_test(P(org(spintet)), P(dest(spintet)), P(pd), P(startpt), P(endpt), nullptr, 1, types, poss)) break;
                enextself(spintet);
                eprevself(searchtet);
            }
            esymself(searchtet);
        } else {
            if (issubseg(searchtet)) {
                bail(2);
                if (misseg != nullptr) idir = SelfIntersect;
                return 0;
            }
        }

        if (!splitsegflag) {
            // Only volume Steiner points are allowed here.
            spintet = searchtet;
            int n = 0, endi = -1;
            while (true) {
                if (apex(spintet) == endpt) endi = n;
                ++n;
                fnextself(spintet);
                if (spintet.tet == searchtet.tet) break;
            }
            if (endi <= 0) return 0;

            std::vector<Triface> abtets(n);
            spintet = searchtet;
            for (int i = 0; i < n; ++i) {
                abtets[i] = spintet;
                fnextself(spintet);
            }

            int success = 0;
            if (dir == AcrossFace) {
                if (add_steinerpt_in_schoenhardtpoly(abtets.data(), endi, splitsliverflag, 0)) success = 1;
            } else if (dir == AcrossEdge) {
                if (issubseg(searchtet)) bail(2);
                if (n > 4) {
                    // The plane of the two crossing edges cuts the star into two pockets.
                    if (endi > 2) {
                        if (add_steinerpt_in_schoenhardtpoly(abtets.data(), endi, splitsliverflag, 0)) ++success;
                    }
                    if (n - endi > 2) {
                        if (add_steinerpt_in_schoenhardtpoly(&abtets[endi], n - endi, splitsliverflag, 0)) ++success;
                    }
                }
            } else {
                bail(2);
            }
            if (success && misseg != nullptr) SubSegStack.push_back(SEncode(*misseg));
            return success ? 1 : 0;
        }

        if (AddSteinerAlgo > 0) {
            if (add_steinerpt_in_segment(misseg, 3, idir)) return 1;
            if (idir == SelfIntersect) return 0;
            sesymself(*misseg);
            if (add_steinerpt_in_segment(misseg, 3, idir)) return 1;
            sesymself(*misseg);
            if (idir == SelfIntersect) return 0;
        }

        // Split the segment where it enters the first crossing face.
        point2tetorg(startpt, searchtet);
        dir = finddirection(&searchtet, endpt);
        if (dir == AcrossVert) {
            if (dest(searchtet) == endpt) {
                if (misseg != nullptr) SubSegStack.push_back(SEncode(*misseg));
                return 1;
            }
            if (misseg != nullptr) idir = SelfIntersect;
            return 0;
        }

        enextself(searchtet);
        {
            const int pa = org(searchtet), pb = dest(searchtet), pd = oppo(searchtet);
            int fpt[3], ept[2];
            sort_3pts(pa, pb, pd, fpt);
            sort_2pts(startpt, endpt, ept);
            dvec3 ip;
            double u;
            planelineint(P(fpt[0]), P(fpt[1]), P(fpt[2]), P(ept[0]), P(ept[1]), ip, &u);
            if (u <= 0 || u >= 1) return 0;

            const int steinerpt = makepoint(FreeSegVertex);
            Pts[steinerpt].Pos = ip;
            setpoint2sh(steinerpt, SEncode(*misseg));
            esymself(searchtet);
            spivot(*misseg, splitsh);

            InsertFlags ivf = boundary_split_flags(Outside);
            ivf.refineflag = dir == AcrossFace ? 4 : 8;
            ivf.refinetet = searchtet;

            if (insertpoint(steinerpt, &searchtet, &splitsh, misseg, &ivf)) {
                if (!FlipStack.empty()) recoverdelaunay();
                SubVertStack.push_back(steinerpt);
                ++StSegRefCount;
                if (SteinerLeft > 0) --SteinerLeft;
                return 1;
            }
            if ((ivf.iloc == OnVertex || ivf.iloc == NearVertex) && misseg != nullptr && blocks_segment(org(searchtet), *misseg)) {
                idir = SelfIntersect;
            }
            pointdealloc(steinerpt);
        }
        return 0;
    }

    // Segments and subfaces recovery gave up on because the input intersects itself.
    std::vector<BadFace> SkippedSegmentList, SkippedFacetList;

    // Drain the segment queue, flipping each missing segment in.
    // With steinerflag set, the ones flips cannot reach take Steiner points.
    int recoversegments(std::vector<Facet> *misseglist, int fullsearch, int steinerflag) {
        Triface searchtet;
        Facet sseg;
        int idir;

        while (!SubSegStack.empty()) {
            SDecode(SubSegStack.back(), sseg);
            SubSegStack.pop_back();
            if (isdeadsh(sseg)) continue;
            sstpivot1(sseg, searchtet);
            if (searchtet.tet != None) continue; // not missing

            const int startpt = sorg(sseg), endpt = sdest(sseg);
            int success = 0;

            if (recoveredgebyflips(startpt, endpt, &sseg, &searchtet, 0, idir)) {
                success = 1;
            } else if (idir != SelfIntersect && recoveredgebyflips(endpt, startpt, &sseg, &searchtet, 0, idir)) {
                success = 1;
            }
            if (!success && fullsearch && idir != SelfIntersect) {
                if (recoveredgebyflips(startpt, endpt, &sseg, &searchtet, fullsearch, idir)) success = 1;
            }

            if (success) {
                bond_seg_ring(sseg, searchtet);
                continue;
            }

            if (idir != SelfIntersect && steinerflag > 0) {
                if (add_steinerpt_to_recover_edge(startpt, endpt, &sseg, 0, 0, idir)) success = 1;
                if (!success && idir != SelfIntersect && steinerflag > 1) {
                    if (add_steinerpt_to_recover_edge(startpt, endpt, &sseg, 1, 0, idir)) success = 1;
                }
            }
            if (success) continue;

            if (idir != SelfIntersect) {
                if (misseglist != nullptr) misseglist->push_back(sseg);
            } else {
                // The input intersects itself here.
                // Set this segment and every subface at it aside, so recovery does not try them again.
                SkippedSegmentList.push_back({.ss = sseg, .key = double(shellmark(sseg)), .forg = sorg(sseg), .fdest = sdest(sseg)});
                smarktest3(sseg);
                Facet neighsh, spinsh;
                Facet base = sseg;
                base.shver = 0;
                spivot(base, neighsh);
                spinsh = neighsh;
                while (spinsh.sh != None) {
                    SkippedFacetList.push_back({
                        .ss = spinsh,
                        .key = double(shellmark(spinsh)),
                        .forg = Shells[spinsh.sh].V[0],
                        .fdest = Shells[spinsh.sh].V[1],
                        .fapex = Shells[spinsh.sh].V[2],
                    });
                    smarktest3(spinsh);
                    spivotself(spinsh);
                    if (spinsh.sh == neighsh.sh) break;
                }
                SkippedFacet = true;
            }
        }
        return 0;
    }

    // Flip away whatever crosses the face (pa, pb, pc), whose three edges are assumed present.
    int recoverfacebyflips(int pa, int pb, int pc, Facet *searchsh, Triface *searchtet, int &dir, int *p1, int *p2) {
        Triface spintet, flipedge;
        FlipConstraints fc;
        int types[2], poss[4];
        fc.fac[0] = pa;
        fc.fac[1] = pb;
        fc.fac[2] = pc;
        fc.checkflipeligibility = 1;
        dir = Disjoint;
        int success = 0;

        for (int i = 0; i < 3 && !success; ++i) {
            while (true) {
                point2tetorg(fc.fac[i], *searchtet);
                finddirection(searchtet, fc.fac[(i + 1) % 3]);
                spintet = *searchtet;
                while (true) {
                    if (apex(spintet) == fc.fac[(i + 2) % 3]) {
                        *searchtet = spintet;
                        for (int j = i; j > 0; --j) eprevself(*searchtet);
                        dir = ShareFace;
                        success = 1;
                        break;
                    }
                    fnextself(spintet);
                    if (spintet.tet == searchtet->tet) break;
                }
                if (success) break;

                flipedge.tet = None;
                spintet = *searchtet;
                while (true) {
                    const int pd = apex(spintet), pe = oppo(spintet);
                    if (pd != DummyPoint && pe != DummyPoint) {
                        const int intflag = tri_edge_test(P(pa), P(pb), P(pc), P(pd), P(pe), nullptr, 1, types, poss);
                        if (intflag > 0) {
                            if (intflag != 2) bail(2);
                            edestoppo(spintet, flipedge); // [d,e,a,b]
                            if (searchsh != nullptr) {
                                dir = types[0];
                                if (dir == AcrossFace || dir == AcrossEdge) {
                                    if (issubseg(flipedge)) {
                                        dir = SelfIntersect;
                                        return 0;
                                    }
                                    Triface chkface = flipedge;
                                    while (true) {
                                        if (issubface(chkface)) break;
                                        fsymself(chkface);
                                        if (chkface.tet == flipedge.tet) break;
                                    }
                                    if (issubface(chkface)) {
                                        dir = SelfIntersect;
                                        return 0;
                                    }
                                } else if (dir == TouchFace) {
                                    // A Steiner point already sits in this subface.
                                    const int touchpt = poss[1] == 0 ? pd : pe;
                                    if (!issteinerpoint(touchpt)) {
                                        dir = SelfIntersect;
                                        return 0;
                                    }
                                    if (pointtype(touchpt) == FreeSegVertex || pointtype(touchpt) == FreeFacetVertex) {
                                        dir = SelfIntersect;
                                        return 0;
                                    }
                                    if (pointtype(touchpt) == FreeVolVertex) {
                                        // Split the subface on the volume point, by a 1-3 flip.
                                        Facet checksh;
                                        setpointtype(touchpt, FreeFacetVertex);
                                        sinsertvertex(touchpt, searchsh, nullptr, OnFace, 0, 0);
                                        --StVolRefCount;
                                        ++StFacRefCount;
                                        SubVertStack.push_back(touchpt);
                                        for (auto &parysh : CaveShBdList) {
                                            spivot(parysh, checksh);
                                            if (!isdeadsh(checksh)) SubFaceStack.push_back(SEncode(checksh));
                                        }
                                        for (auto &parysh : CaveShList) subfacedealloc(parysh.sh);
                                        CaveShList.clear();
                                        CaveShBdList.clear();
                                        CaveSegShList.clear();
                                        searchsh->sh = None; // it has been split
                                        return 1;
                                    }
                                    bail(2);
                                } else {
                                    bail(2);
                                }
                            }
                            break;
                        }
                    }
                    fnextself(spintet);
                    if (spintet.tet == searchtet->tet) bail(2);
                }

                *p1 = org(flipedge);
                *p2 = dest(flipedge);
                if (removeedgebyflips(&flipedge, &fc) == 2) continue;
                break;
            }
        }
        return success;
    }

    long DuplicatedFacetsCount{0};

    // Take out a temporary segment recoversubfaces planted on an edge of a subface.
    void remove_temp_segment(Facet &seg) {
        Facet neineish, neighsh;
        spivot(seg, neineish);
        ssdissolve(neineish);
        spivot(neineish, neighsh);
        if (neighsh.sh != None) ssdissolve(neighsh);
        dissolve_seg_ring(seg);
        subsegdealloc(seg.sh);
    }

    // Recover every queued subface.
    // Its three edges are made present first, each fenced with a temporary segment, and then the face is flipped in.
    // With steinerflag set, a face flips cannot reach is split.
    int recoversubfaces(std::vector<Facet> *misshlist, int steinerflag) {
        Triface searchtet, neightet;
        Facet searchsh, neighsh, bdsegs[3];
        InsertFlags ivf;
        int dir;

        while (!SubFaceStack.empty()) {
            SDecode(SubFaceStack.back(), searchsh);
            SubFaceStack.pop_back();
            if (isdeadsh(searchsh)) continue;
            if (smarktest3ed(searchsh)) continue; // set aside as self-intersecting
            stpivot(searchsh, neightet);
            if (neightet.tet != None) continue; // already recovered

            dir = Disjoint;
            int success = 0;
            int startpt = None, endpt = None;
            for (int k = 0; k < 3; ++k) bdsegs[k].sh = None;

            int i = 0;
            for (; i < 3; ++i) {
                sspivot(searchsh, bdsegs[i]);
                startpt = sorg(searchsh);
                endpt = sdest(searchsh);
                if (bdsegs[i].sh != None) {
                    sstpivot1(bdsegs[i], searchtet);
                    if (searchtet.tet != None) {
                        senextself(searchsh);
                        continue;
                    }
                    success = 0;
                    if (recoveredgebyflips(startpt, endpt, &bdsegs[i], &searchtet, 0, dir)) success = 1;
                    else if (dir != SelfIntersect && recoveredgebyflips(endpt, startpt, &bdsegs[i], &searchtet, 0, dir)) success = 1;
                    if (!success) break;
                    bond_seg_ring(bdsegs[i], searchtet);
                } else {
                    success = 0;
                    point2tetorg(startpt, searchtet);
                    finddirection(&searchtet, endpt);
                    if (dest(searchtet) == endpt) {
                        success = 1;
                    } else {
                        if (recoveredgebyflips(startpt, endpt, &searchsh, &searchtet, 0, dir)) success = 1;
                        else if (dir != SelfIntersect && recoveredgebyflips(endpt, startpt, &searchsh, &searchtet, 0, dir)) success = 1;
                    }
                    if (success && dir != SelfIntersect) {
                        // Fence the edge with a temporary segment while the face is recovered.
                        makesubseg(&bdsegs[i]);
                        setshvertices(bdsegs[i], startpt, endpt, None);
                        smarktest2(bdsegs[i]);
                        ssbond(searchsh, bdsegs[i]);
                        spivot(searchsh, neighsh);
                        if (neighsh.sh != None) ssbond(neighsh, bdsegs[i]);
                        bond_seg_ring(bdsegs[i], searchtet);
                    } else {
                        break;
                    }
                }
                senextself(searchsh);
            }

            if (i < 3) {
                // An edge is missing.
                // Take back the fences already planted.
                for (int j = i - 1; j >= 0; --j) {
                    if (bdsegs[j].sh != None && smarktest2ed(bdsegs[j])) remove_temp_segment(bdsegs[j]);
                }
            } else {
                startpt = sorg(searchsh);
                endpt = sdest(searchsh);
                const int apexpt = sapex(searchsh);
                int cross_e1 = None, cross_e2 = None;
                success = recoverfacebyflips(startpt, endpt, apexpt, &searchsh, &searchtet, dir, &cross_e1, &cross_e2);

                for (int j = 0; j < 3; ++j) {
                    if (bdsegs[j].sh != None && !isdeadsh(bdsegs[j]) && smarktest2ed(bdsegs[j])) remove_temp_segment(bdsegs[j]);
                }

                if (success) {
                    if (searchsh.sh != None) {
                        Facet chkface;
                        tspivot(searchtet, chkface);
                        if (chkface.sh == None) {
                            tsbond(searchtet, searchsh);
                            fsymself(searchtet);
                            sesymself(searchsh);
                            tsbond(searchtet, searchsh);
                        } else if (shellmark(chkface) == shellmark(searchsh)) {
                            ++DuplicatedFacetsCount;
                            smarktest3(searchsh);
                            sinfect(searchsh);
                        } else {
                            dir = SelfIntersect;
                            success = 0;
                        }
                    }
                } else if (dir != SelfIntersect && steinerflag) {
                    // Split the subface where the crossing edge meets its plane, falling back to the barycentre when that lands on a side.
                    dvec3 ip;
                    double u;
                    int fpt[3], ept[2];
                    sort_3pts(startpt, endpt, apexpt, fpt);
                    sort_2pts(cross_e1, cross_e2, ept);
                    planelineint(P(fpt[0]), P(fpt[1]), P(fpt[2]), P(ept[0]), P(ept[1]), ip, &u);

                    const int steinerpt = makepoint(FreeFacetVertex);
                    bool use_bary = !(u > 0.0 && u < 1.0);
                    if (!use_bary) {
                        Pts[steinerpt].Pos = ip;
                        use_bary = is_collinear_at(steinerpt, startpt, endpt) || is_collinear_at(steinerpt, endpt, apexpt) || is_collinear_at(steinerpt, apexpt, startpt);
                    }
                    if (use_bary) {
                        Pts[steinerpt].Pos = (P(startpt) + P(endpt) + P(apexpt)) / 3.0;
                        if (is_collinear_at(steinerpt, startpt, endpt) || is_collinear_at(steinerpt, endpt, apexpt) || is_collinear_at(steinerpt, apexpt, startpt)) {
                            bail(2);
                        }
                    }
                    setpoint2sh(steinerpt, SEncode(searchsh));

                    ivf = InsertFlags{};
                    point2tetorg(startpt, searchtet);
                    ivf.iloc = Outside;
                    ivf.bowywat = 1;
                    ivf.lawson = 2;
                    ivf.rejflag = 0;
                    ivf.chkencflag = 0;
                    ivf.sloc = OnFace;
                    ivf.sbowywat = 1;
                    ivf.splitbdflag = 0;
                    ivf.validflag = 1;
                    ivf.respectbdflag = 1;

                    if (insertpoint(steinerpt, &searchtet, &searchsh, nullptr, &ivf)) {
                        if (!FlipStack.empty()) recoverdelaunay();
                        SubVertStack.push_back(steinerpt);
                        ++StFacRefCount;
                        if (SteinerLeft > 0) --SteinerLeft;
                        success = 1;
                    } else {
                        if (ivf.iloc == NearVertex) {
                            const int chkpt = org(searchtet);
                            if (distance(P(steinerpt), P(chkpt)) < MinEdgeLength) {
                                if (!issteinerpoint(chkpt)) dir = SelfIntersect;
                            } else {
                                dir = SelfIntersect;
                            }
                        }
                        if (dir != SelfIntersect && steinerflag >= 2) bail(2);
                        pointdealloc(steinerpt);
                        success = 0;
                    }
                }
            }

            if (i < 3 && dir != SelfIntersect && steinerflag > 0) {
                // An edge of the subface is missing.
                // Split it where it first leaves the mesh.
                point2tetorg(startpt, searchtet);
                dir = finddirection(&searchtet, endpt);
                enextself(searchtet);
                const int pa = org(searchtet), pb = dest(searchtet), pd = oppo(searchtet);
                dvec3 ip;
                double u;
                int fpt[3], ept[2];
                sort_3pts(pa, pb, pd, fpt);
                sort_2pts(startpt, endpt, ept);
                planelineint(P(fpt[0]), P(fpt[1]), P(fpt[2]), P(ept[0]), P(ept[1]), ip, &u);

                const int steinerpt = makepoint(FreeFacetVertex);
                Pts[steinerpt].Pos = ip;

                Triface tmptet = searchtet;
                ivf = boundary_split_flags(locate(steinerpt, &tmptet));
                ivf.refinetet = searchtet;
                if (ivf.iloc == OnVertex) {
                    searchtet = tmptet;
                } else if (dir == AcrossFace) {
                    ivf.iloc = OnFace;
                } else if (dir == AcrossEdge) {
                    ivf.iloc = OnEdge;
                } else {
                    bail(2);
                }
                Facet misseg;
                Facet *splitseg = nullptr;
                sspivot(searchsh, misseg);
                if (misseg.sh != None) {
                    splitseg = &misseg;
                    setpointtype(steinerpt, FreeSegVertex);
                    setpoint2sh(steinerpt, SEncode(misseg));
                } else {
                    setpoint2sh(steinerpt, SEncode(searchsh));
                }
                const bool splitseg_flag = splitseg != nullptr;

                if (insertpoint(steinerpt, &searchtet, &searchsh, splitseg, &ivf)) {
                    if (!FlipStack.empty()) recoverdelaunay();
                    SubVertStack.push_back(steinerpt);
                    if (splitseg_flag) ++StSegRefCount;
                    else ++StFacRefCount;
                    if (SteinerLeft > 0) --SteinerLeft;
                    success = 1;
                } else {
                    if (ivf.iloc == NearVertex) {
                        const int chkpt = org(searchtet);
                        if (!issteinerpoint(chkpt)) dir = SelfIntersect;
                    }
                    if (dir != SelfIntersect && steinerflag >= 2) {
                        success = 0;
                        if (ivf.iloc == OnVertex || ivf.iloc == NearVertex) {
                            if (dir == AcrossEdge) {
                                int idir;
                                if (add_steinerpt_to_recover_edge(startpt, endpt, nullptr, 0, 1, idir)) {
                                    SubFaceStack.push_back(SEncode(searchsh));
                                    success = 1;
                                }
                            } else {
                                bail(2);
                            }
                        } else {
                            bail(2);
                        }
                    }
                    pointdealloc(steinerpt);
                }
            }

            if (success) continue;

            if (dir == SelfIntersect) {
                SkippedFacetList.push_back({
                    .ss = searchsh,
                    .key = double(shellmark(searchsh)),
                    .forg = Shells[searchsh.sh].V[0],
                    .fdest = Shells[searchsh.sh].V[1],
                    .fapex = Shells[searchsh.sh].V[2],
                });
                smarktest3(searchsh);
                SkippedFacet = true;
                continue;
            }

            if (steinerflag >= 2) bail(2);
            if (misshlist != nullptr) misshlist->push_back(searchsh);
        }
        return 0;
    }

    // Flip away as many of the edges at a vertex as possible, returning how many remain.
    int reduceedgesatvertex(int startpt, std::vector<int> &endptlist) {
        Triface searchtet;
        FlipConstraints fc{.checkflipeligibility = 1, .remvert = startpt};

        while (true) {
            int count = 0;
            for (size_t i = 0; i < endptlist.size(); ++i) {
                const int pendpt = endptlist[i];
                if (pendpt == DummyPoint) continue;
                int reduceflag = 0;
                int dir;
                if (NonConvex) {
                    dir = getedge(startpt, pendpt, &searchtet) ? AcrossVert : Intersect;
                } else {
                    point2tetorg(startpt, searchtet);
                    dir = finddirection(&searchtet, pendpt);
                }
                if (dir == AcrossVert) {
                    if (dest(searchtet) != pendpt) bail(2);
                    if (!issubseg(searchtet)) {
                        if (removeedgebyflips(&searchtet, &fc) == 2) reduceflag = 1;
                    }
                } else {
                    reduceflag = 1; // the edge is gone already
                }
                if (reduceflag) {
                    ++count;
                    endptlist[i] = endptlist.back();
                    endptlist.pop_back();
                    --i;
                }
            }
            if (count == 0) break;
        }
        return int(endptlist.size());
    }

    // Take a Steiner point out by a sequence of flips, when its star reduces far enough for one of the vertex-removing flips.
    int removevertexbyflips(int steinerpt) {
        Triface searchtet, spintet, neightet;
        Triface wrktets[4];
        Facet parentsh, spinsh;
        Facet leftseg, rightseg, checkseg;
        int lpt = None, rpt = None;
        FlipConstraints fc;
        int loc = LocUnknown;
        int valence, removeflag = 0;

        const VertType vt = pointtype(steinerpt);
        if (vt == FreeSegVertex) {
            segments_at(steinerpt, leftseg, rightseg);
            lpt = sorg(leftseg);
            rpt = sdest(rightseg);
            sstpivot1(leftseg, neightet);
            if (neightet.tet == None) return 0;
            sstpivot1(rightseg, neightet);
            if (neightet.tet == None) return 0;
        } else if (vt != FreeFacetVertex && vt != FreeVolVertex && vt != VolVertex) {
            return 0;
        }

        getvertexstar(1, steinerpt, &CaveTetList, &CaveTetVertList, nullptr);
        CaveTetList.clear();
        valence = CaveTetVertList.size() > 3 ? reduceedgesatvertex(steinerpt, CaveTetVertList) : int(CaveTetVertList.size());
        CaveTetVertList.clear();

        if (valence == 4) {
            // Four vertices left: p is inside their convex hull.
            point2tetorg(steinerpt, searchtet);
            loc = InTetrahedron;
            removeflag = 1;
        } else if (valence == 5) {
            if (vt == FreeSegVertex) {
                sstpivot1(leftseg, searchtet);
                if (org(searchtet) != steinerpt) esymself(searchtet);
                int i = 0;
                neightet.tet = None;
                spintet = searchtet;
                while (true) {
                    ++i;
                    if (apex(spintet) == rpt) neightet = spintet;
                    fnextself(spintet);
                    if (spintet.tet == searchtet.tet) break;
                }
                if (i == 4 && neightet.tet != None && apex(neightet) == rpt) {
                    // The segment is already back.
                    // A 6-2 flip may take p out.
                    esym(neightet, searchtet);
                    enextself(searchtet);
                    wrktets[0] = searchtet; // [p,d,a,b]
                    for (int k = 0; k < 2; ++k) fnext(wrktets[k], wrktets[k + 1]);
                    if (apex(wrktets[0]) == oppo(wrktets[2])) {
                        loc = OnFace;
                        removeflag = 1;
                    }
                }
            } else if (vt == FreeFacetVertex) {
                point2tetorg(steinerpt, searchtet);
                wrktets[0] = searchtet;
                wrktets[1] = searchtet;
                esymself(wrktets[1]);
                enextself(wrktets[1]);
                wrktets[2] = searchtet;
                eprevself(wrktets[2]);
                esymself(wrktets[2]);
                searchtet.tet = None;
                for (int i = 0; i < 3; ++i) {
                    spintet = wrktets[i];
                    valence = 0;
                    while (true) {
                        ++valence;
                        fnextself(spintet);
                        if (spintet.tet == wrktets[i].tet) break;
                    }
                    if (valence == 3) {
                        searchtet = wrktets[i];
                        break;
                    }
                }
                loc = OnFace;
                removeflag = 1;
            }
        }

        if (!removeflag && vt == FreeSegVertex) {
            // Can the edge [lpt, rpt] be recovered directly?
            // Every tet at leftseg must be adjacent to one at rightseg, and none of the results may be inverted.
            sstpivot1(leftseg, searchtet);
            if (org(searchtet) != steinerpt) esymself(searchtet);
            spintet = searchtet;
            while (true) {
                eprev(spintet, neightet);
                esymself(neightet);
                fsymself(neightet);
                if (oppo(neightet) != rpt) break;
                const int chkp1 = org(neightet), chkp2 = apex(neightet);
                if (orient3d(P(rpt), P(lpt), P(chkp1), P(chkp2)) >= 0.0) break;
                fnextself(spintet);
                if (spintet.tet == searchtet.tet) {
                    loc = OnEdge;
                    removeflag = 1;
                    break;
                }
            }
        }

        if (!removeflag && vt == FreeSegVertex) {
            if (getedge(lpt, rpt, &searchtet) && !CheckSubfaceFlag) {
                // The edge is already there: move the point into the volume instead.
                for (const Facet &seg : {leftseg, rightseg}) {
                    dissolve_seg_ring(seg);
                    sstdissolve1(seg);
                }
                spivot(rightseg, parentsh);
                sremovevertex(steinerpt, &parentsh, &rightseg, 1);
                CaveShBdList.clear();
                bond_seg_ring(rightseg, searchtet);
                setpointtype(steinerpt, FreeVolVertex);
                --StSegRefCount;
                ++StVolRefCount;
                return 1;
            }
        }

        if (!removeflag) return 0;

        if (vt == FreeSegVertex) {
            for (const Facet &seg : {leftseg, rightseg}) {
                dissolve_seg_ring(seg);
                sstdissolve1(seg);
            }
            if (CheckSubfaceFlag) {
                for (int i = 0; i < 2; ++i) {
                    checkseg = i == 0 ? leftseg : rightseg;
                    spivot(checkseg, parentsh);
                    if (parentsh.sh == None) continue;
                    spinsh = parentsh;
                    while (true) {
                        stpivot(spinsh, neightet);
                        if (neightet.tet != None) tsdissolve(neightet);
                        sesymself(spinsh);
                        stpivot(spinsh, neightet);
                        if (neightet.tet != None) tsdissolve(neightet);
                        stdissolve(spinsh);
                        spivotself(spinsh);
                        if (spinsh.sh == parentsh.sh) break;
                    }
                }
            }
        }

        std::vector<Triface> fliptets;
        if (loc == InTetrahedron) {
            fliptets.resize(4);
            fliptets[0] = searchtet; // [p,d,a,b]
            for (int i = 0; i < 2; ++i) fnext(fliptets[i], fliptets[i + 1]);
            eprev(fliptets[0], fliptets[3]);
            fnextself(fliptets[3]);
            eprevself(fliptets[3]);
            esymself(fliptets[3]); // [a,b,c,p]
            if (vt == FreeFacetVertex && !valid_41_flip_at_facet_vertex(fliptets.data())) return 0;
            flip41(fliptets.data(), 1, &fc);
        } else if (loc == OnFace) {
            fliptets.resize(6);
            fliptets[0] = searchtet; // [p,d,a,b]
            for (int i = 0; i < 2; ++i) fnext(fliptets[i], fliptets[i + 1]);
            eprev(fliptets[0], fliptets[3]);
            fnextself(fliptets[3]);
            esymself(fliptets[3]);
            eprevself(fliptets[3]); // [e,p,a,b]
            for (int i = 3; i < 5; ++i) fnext(fliptets[i], fliptets[i + 1]);
            if (vt == FreeFacetVertex) {
                int count = 0;
                for (int i = 3; i < 6; ++i) {
                    if (issubface(fliptets[i])) ++count;
                }
                if (count > 0) {
                    // The three subfaces sit in the upper half: swap the two halves round so the 3-2 flip happens there.
                    for (int i = 0; i < 3; ++i) {
                        esym(fliptets[i + 3], wrktets[i]);
                        esym(fliptets[i], fliptets[i + 3]);
                        fliptets[i] = wrktets[i];
                    }
                    std::swap(fliptets[1], fliptets[2]);
                    std::swap(fliptets[4], fliptets[5]);
                }
                if (!valid_41_flip_at_facet_vertex(fliptets.data())) return 0;
            }
            flip32(&fliptets[3], 1, &fc);
            flip41(fliptets.data(), 1, &fc);
        } else if (loc == OnEdge) {
            int n = 0;
            spintet = searchtet;
            while (true) {
                ++n;
                fnextself(spintet);
                if (spintet.tet == searchtet.tet) break;
            }
            fliptets.resize(2 * n);
            fliptets[0] = searchtet;
            for (int i = 0; i < n - 1; ++i) fnext(fliptets[i], fliptets[i + 1]);
            eprev(fliptets[0], fliptets[n]);
            fnextself(fliptets[n]);
            esymself(fliptets[n]);
            eprevself(fliptets[n]); // [e,p,p_0,p_1]
            for (int i = n; i < 2 * n - 1; ++i) fnext(fliptets[i], fliptets[i + 1]);

            // A 2n-to-n flip, run as one 2-3, then n-2 3-2s, then a 4-1.
            wrktets[0] = fliptets[0];
            eprevself(wrktets[0]);
            esymself(wrktets[0]);
            enextself(wrktets[0]); // [p_0,p_1,p,d]
            wrktets[1] = fliptets[n];
            enextself(wrktets[1]);
            esymself(wrktets[1]);
            eprevself(wrktets[1]); // [p_1,p_0,p,e]
            flip23(wrktets, 1, &fc);
            fliptets[n] = wrktets[2];
            fliptets[0] = wrktets[0];
            for (int i = 1; i < n - 1; ++i) {
                wrktets[0] = wrktets[1];
                enextself(wrktets[0]);
                esymself(wrktets[0]);
                eprevself(wrktets[0]);
                wrktets[1] = fliptets[n + i];
                enextself(wrktets[1]);
                wrktets[2] = fliptets[i];
                eprevself(wrktets[2]);
                esymself(wrktets[2]);
                flip32(wrktets, 1, &fc);
                fliptets[i] = wrktets[0];
                esymself(fliptets[i]);
            }
            wrktets[3] = wrktets[1];
            wrktets[0] = fliptets[n];
            eprevself(wrktets[0]);
            esymself(wrktets[0]);
            enextself(wrktets[0]);
            wrktets[1] = fliptets[n - 1];
            esymself(wrktets[1]);
            enextself(wrktets[1]);
            wrktets[2] = fliptets[2 * n - 1];
            enextself(wrktets[2]);
            esymself(wrktets[2]);
            enextself(wrktets[2]);
            flip41(wrktets, 1, &fc);
        }

        if (vt == FreeSegVertex) {
            const int slawson = CheckSubfaceFlag ? 0 : 1;
            spivot(rightseg, parentsh);
            sremovevertex(steinerpt, &parentsh, &rightseg, slawson);
            rightseg.shver = 0;
            point2tetorg(lpt, searchtet);
            finddirection(&searchtet, rpt);
            if (dest(searchtet) != rpt) bail(2);
            bond_seg_ring(rightseg, searchtet);
            if (CheckSubfaceFlag) {
                spivot(rightseg, parentsh);
                if (parentsh.sh != None) {
                    spinsh = parentsh;
                    while (true) {
                        if (sorg(spinsh) != lpt) sesymself(spinsh);
                        const int apexpt = sapex(spinsh);
                        spintet = searchtet;
                        while (true) {
                            if (apex(spintet) == apexpt) {
                                tsbond(spintet, spinsh);
                                sesymself(spinsh);
                                fsym(spintet, neightet);
                                tsbond(neightet, spinsh);
                                sesymself(spinsh);
                                break;
                            }
                            fnextself(spintet);
                        }
                        spivotself(spinsh);
                        if (spinsh.sh == parentsh.sh) break;
                    }
                }
            }
            CaveShBdList.clear();
        }

        release_steiner(steinerpt, vt);
        return 1;
    }

    // The check made before the closing 4-1 flip at a facet vertex.
    // One of the three tets must carry all three subfaces, or all three must carry one each.
    bool valid_41_flip_at_facet_vertex(Triface *fliptets) {
        Triface checktet, chkface;
        int i = 0;
        for (; i < 3; ++i) {
            enext(fliptets[i], checktet);
            esymself(checktet);
            int scount = 0;
            for (int k = 0; k < 3; ++k) {
                esym(checktet, chkface);
                if (issubface(chkface)) ++scount;
                enextself(checktet);
            }
            if (scount == 3) return true;
            if (scount == 2) return false;
        }
        int scount = 0;
        for (i = 0; i < 3; ++i) {
            eprev(fliptets[i], checktet);
            esymself(checktet);
            if (issubface(checktet)) ++scount;
        }
        return scount == 3;
    }

    // Walk the point toward the centre of one of its link faces for as long as the chosen objective over the whole star improves.
    int smoothpoint(dvec3 &smtpt, std::vector<Triface> &linkfacelist, int ccw, OptParameters *opm) {
        BadFace bf;
        dvec3 startpt = smtpt, bestpt = smtpt;
        double minval = 0.0;

        int numdirs = int(linkfacelist.size());
        if (numdirs > opm->numofsearchdirs) numdirs = opm->numofsearchdirs;
        opm->imprval = opm->initval;
        int iter = 0;

        while (true) {
            const double oldval = opm->imprval;
            for (int i = 0; i < numdirs; ++i) {
                const int k = int(randomnation(unsigned(linkfacelist.size() - size_t(i))));
                Triface t = linkfacelist[k];
                const dvec3 fcent = (P(org(t)) + P(dest(t)) + P(apex(t))) / 3.0;
                // Per component, which is the form the candidate positions are compared in.
                dvec3 nextpt;
                for (int j = 0; j < 3; ++j) nextpt[j] = startpt[j] + opm->searchstep * (fcent[j] - startpt[j]);
                size_t j = 0;
                for (; j < linkfacelist.size(); ++j) {
                    const Triface &f = linkfacelist[j];
                    const int pa = ccw ? org(f) : dest(f);
                    const int pb = ccw ? dest(f) : org(f);
                    const int pc = apex(f);
                    const double ori = orient3d(P(pa), P(pb), P(pc), nextpt);
                    double val;
                    if (ori < 0.0) {
                        if (opm->max_min_volume) {
                            val = -orient3dfast(P(pa), P(pb), P(pc), nextpt);
                        } else if (opm->min_max_aspectratio) {
                            get_tetqual_at(P(pa), P(pb), P(pc), nextpt, &bf);
                            val = 1.0 / bf.key;
                        } else if (opm->min_max_dihedangle) {
                            get_tetqual_at(P(pa), P(pb), P(pc), nextpt, &bf);
                            double maxcosd = bf.cent[0];
                            if (maxcosd < -1) maxcosd = -1.0;
                            val = maxcosd + 1.0;
                        } else {
                            val = 0.0;
                        }
                    } else {
                        // The new tet would be inverted, which a mesh with inverted elements can produce.
                        // Only the volume objective can still score it.
                        if (opm->max_min_volume) val = -orient3dfast(P(pa), P(pb), P(pc), nextpt);
                        else break;
                    }
                    if (val <= opm->imprval) break;
                    minval = j == 0 ? val : std::min(val, minval);
                }
                if (j == linkfacelist.size()) {
                    opm->imprval = minval;
                    bestpt = nextpt;
                }
                std::swap(linkfacelist[k], linkfacelist[linkfacelist.size() - size_t(i) - 1]);
            }

            double diff = opm->imprval - oldval;
            if (diff > 0.0 && opm->min_max_aspectratio) {
                if (diff / oldval < 1e-3) diff = 0.0;
            }
            if (diff <= 0.0) break;
            startpt = bestpt;
            ++iter;
            if (opm->maxiter > 0 && iter >= opm->maxiter) break;
        }

        if (iter > 0) {
            opm->smthiter = iter;
            smtpt = startpt;
        }
        return iter;
    }

    // Replace a Steiner point on a segment or facet by one interior point per sector it separates.
    // The boundary point then comes out by flips.
    int suppressbdrysteinerpoint(int steinerpt) {
        Facet parentsh, spinsh;
        Facet leftseg, rightseg;
        Triface searchtet, neightet;
        dvec3 v1, v2, startpt, samplept, candpt;
        double len, u, ori, minvol, smallvol;

        const VertType vt = pointtype(steinerpt);
        if (vt == FreeSegVertex) {
            segments_at(steinerpt, leftseg, rightseg);
            spivot(leftseg, parentsh);
            if (parentsh.sh != None) {
                spinsh = parentsh;
                while (true) {
                    Facet f = spinsh;
                    if (sorg(f) != sorg(parentsh)) sesymself(f);
                    CaveSegShList.push_back(f);
                    spivotself(spinsh);
                    if (spinsh.sh == None) break;
                    if (spinsh.sh == parentsh.sh) break;
                }
            }
            if (CaveSegShList.size() < 2) {
                CaveSegShList.clear();
                return 0; // a single segment, left alone
            }
        } else if (vt == FreeFacetVertex) {
            SDecode(point2sh(steinerpt), parentsh);
            for (int i = 0; i < 2; ++i) {
                CaveSegShList.push_back(parentsh);
                sesymself(parentsh);
            }
        } else {
            return 0;
        }

        const int n = int(CaveSegShList.size());
        std::vector<int> newsteiners(size_t(n), None);

        size_t i = 0;
        for (; i < CaveSegShList.size(); ++i) {
            Facet parysh = CaveSegShList[i];
            stpivot(parysh, searchtet);
            if (ishulltet(searchtet)) continue;
            setpoint2tet(steinerpt, Encode(searchtet));
            getvertexstar(0, steinerpt, &CaveTetList, nullptr, &CaveShList);

            const int pa0 = sorg(parysh), pb0 = sdest(parysh), pc0 = sapex(parysh);
            facenormal(P(pa0), P(pb0), P(pc0), v1, 1, nullptr);
            v1 /= std::sqrt(dot(v1, v1));
            if (vt == FreeSegVertex) {
                const Facet &nextsh = CaveSegShList[(i + 1) % size_t(n)];
                const int pd = sapex(nextsh);
                facenormal(P(pb0), P(pa0), P(pd), v2, 1, nullptr);
                v2 /= std::sqrt(dot(v2, v2));
                v1 = 0.5 * (v1 + v2);
            }
            len = distance(P(pa0), P(pb0));
            v2 = P(steinerpt) + len * v1;

            size_t j = 0;
            for (; j < CaveTetList.size(); ++j) {
                const Triface &t = CaveTetList[j];
                const int pa = org(t), pb = dest(t), pc = apex(t);
                if (orient3d(P(steinerpt), P(pa), P(pb), v2) < 0) continue;
                if (orient3d(P(steinerpt), P(pb), P(pc), v2) < 0) continue;
                if (orient3d(P(steinerpt), P(pc), P(pa), v2) < 0) continue;
                planelineint(P(pa), P(pb), P(pc), P(steinerpt), v2, startpt, &u);
                break;
            }
            if (j == CaveTetList.size()) break; // the ray leaves no face of the ball

            for (auto &sh : CaveShList) {
                stpivot(sh, neightet);
                CaveTetList.push_back(neightet);
            }

            int it = 0, samplesize = 100;
            v1 = P(steinerpt) - startpt;
            minvol = -1.0;
            while (it < 3) {
                for (int s = 1; s < samplesize - 1; ++s) {
                    // Per component, which is the form the sample points are compared in.
                    for (int c = 0; c < 3; ++c) samplept[c] = startpt[c] + (double(s) / double(samplesize)) * v1[c];
                    smallvol = -1;
                    size_t k = 0;
                    for (; k < CaveTetList.size(); ++k) {
                        const Triface &t = CaveTetList[k];
                        const int pa = org(t), pb = dest(t), pc = apex(t);
                        ori = orient3d(P(pb), P(pa), P(pc), samplept);
                        const double lv = (distance(P(pa), P(pb)) + distance(P(pb), P(pc)) + distance(P(pc), P(pa))) / 3.0;
                        if (std::abs(ori) / (lv * lv * lv) < 1e-8) ori = 0.0;
                        if (ori <= 0) break;
                        smallvol = smallvol == -1 ? ori : std::min(smallvol, ori);
                    }
                    if (k == CaveTetList.size()) {
                        if (minvol == -1.0 || minvol < smallvol) {
                            candpt = samplept;
                            minvol = smallvol;
                        } else {
                            // The smallest volume can only fall from here along this line.
                            break;
                        }
                    }
                }
                if (minvol > 0) break;
                samplesize *= 10;
                ++it;
            }
            if (minvol == -1.0) {
                CaveTetList.clear();
                CaveShList.clear();
                break;
            }
            newsteiners[i] = makepoint(FreeVolVertex);
            Pts[newsteiners[i]].Pos = candpt;
            CaveTetList.clear();
            CaveShList.clear();
        }

        if (i < CaveSegShList.size()) {
            // Failed to suppress the vertex.
            for (size_t j = i; j > 0; --j) {
                if (newsteiners[j - 1] != None) pointdealloc(newsteiners[j - 1]);
            }
            CaveSegShList.clear();
            return 0;
        }

        std::vector<Facet> segshlist(CaveSegShList.begin(), CaveSegShList.end());
        CaveSegShList.clear();

        size_t k = 0;
        for (; k < size_t(n); ++k) {
            Facet parysh = segshlist[k];
            stpivot(parysh, searchtet);
            if (ishulltet(searchtet)) continue;
            setpoint2tet(steinerpt, Encode(searchtet));
            getvertexstar(0, steinerpt, &CaveTetList, nullptr, &CaveShList);
            for (auto &t : CaveTetList) {
                infect(t);
                CaveOldTetList.push_back(t);
                neightet = t;
            }
            CaveTetList.clear();
            CaveShList.clear();

            InsertFlags ivf;
            searchtet = neightet;
            ivf.iloc = InStar;
            if (insertpoint(newsteiners[k], &searchtet, nullptr, nullptr, &ivf)) {
                ++StVolRefCount;
                if (SteinerLeft > 0) --SteinerLeft;
            } else {
                pointdealloc(newsteiners[k]);
                newsteiners[k] = None;
                break;
            }
        }
        if (k < size_t(n)) return 0;

        if (!removevertexbyflips(steinerpt)) return 0;

        setpointtype(steinerpt, UnusedVertex);
        ++UnuVerts;

        const int bak_fliplinklevel = FlipLinkLevel;
        FlipLinkLevel = 100000;
        for (size_t j = 0; j < size_t(n); ++j) {
            if (newsteiners[j] == None) continue;
            if (!removevertexbyflips(newsteiners[j])) {
                if (SupSteinerLevel > 0) SubVertStack.push_back(newsteiners[j]);
            }
        }
        FlipLinkLevel = bak_fliplinklevel;
        return 1;
    }

    // The two far endpoints of the input segment each subsegment belongs to.
    // The ridge-vertex adjacency the flip guards read comes with them.
    void makesegmentendpointsmap() {
        std::vector<std::array<int, 2>> segptlist;
        const int np = int(Pts.size());
        IdxSegmentRidgeVertexList.assign(size_t(np) + 2, 0);

        // A segment may be split into many subsegments.
        // Work from the one holding its origin.
        std::vector<int> segs;
        forallsubsegs([&](int sh) { segs.push_back(sh); });
        int segindex = 0;
        for (const int sh : segs) {
            Facet segloop{sh, 0}, prevseg, nextseg;
            senext2(segloop, prevseg);
            spivotself(prevseg);
            if (prevseg.sh != None) continue;
            const int eorg = sorg(segloop);
            int edest = sdest(segloop);
            setfacetindex(segloop, segindex);
            senext(segloop, nextseg);
            spivotself(nextseg);
            while (nextseg.sh != None) {
                setfacetindex(nextseg, segindex);
                nextseg.shver = 0;
                if (sorg(nextseg) != edest) sesymself(nextseg);
                edest = sdest(nextseg);
                senextself(nextseg);
                spivotself(nextseg);
            }
            segptlist.push_back({eorg, edest});
            ++segindex;
            ++IdxSegmentRidgeVertexList[eorg];
            ++IdxSegmentRidgeVertexList[edest];
        }

        SegmentEndpointsList.clear();
        for (const auto &e : segptlist) {
            SegmentEndpointsList.push_back(e[0]);
            SegmentEndpointsList.push_back(e[1]);
        }

        int j = IdxSegmentRidgeVertexList[0];
        IdxSegmentRidgeVertexList[0] = 0;
        for (int i = 0; i < np + 1; ++i) {
            const int k = IdxSegmentRidgeVertexList[i + 1];
            IdxSegmentRidgeVertexList[i + 1] = IdxSegmentRidgeVertexList[i] + j;
            j = k;
        }
        SegmentRidgeVertexList.assign(size_t(IdxSegmentRidgeVertexList[np + 1]) + 1, None);
        for (size_t i = 0; i < segptlist.size(); ++i) {
            const int eorg = SegmentEndpointsList[i * 2], edest = SegmentEndpointsList[i * 2 + 1];
            SegmentRidgeVertexList[IdxSegmentRidgeVertexList[eorg]++] = edest;
            SegmentRidgeVertexList[IdxSegmentRidgeVertexList[edest]++] = eorg;
        }
        for (int i = np; i >= 0; --i) IdxSegmentRidgeVertexList[i + 1] = IdxSegmentRidgeVertexList[i];
        IdxSegmentRidgeVertexList[0] = 0;
    }

    // Take the Steiner points recovery left on the boundary back off it, then remove or smooth the interior ones it created.
    int suppresssteinerpoints() {
        const int bak_fliplinklevel = FlipLinkLevel;
        FlipLinkLevel = 100000;

        for (size_t i = 0; i < SubVertStack.size(); ++i) {
            const int rempt = SubVertStack[i];
            if (pointtype(rempt) == UnusedVertex) continue;
            if (pointtype(rempt) == FreeSegVertex || pointtype(rempt) == FreeFacetVertex) {
                suppressbdrysteinerpoint(rempt);
            }
        }
        if (SupSteinerLevel > 0) {
            for (size_t i = 0; i < SubVertStack.size(); ++i) {
                const int rempt = SubVertStack[i];
                if (pointtype(rempt) == FreeVolVertex) removevertexbyflips(rempt);
            }
        }
        FlipLinkLevel = bak_fliplinklevel;

        if (SupSteinerLevel > 1) {
            OptParameters opm{.max_min_volume = 1, .numofsearchdirs = 20, .searchstep = 0.001, .maxiter = 30};
            int ivcount = 0;
            while (true) {
                int nt = 0;
                while (true) {
                    int count = 0;
                    ivcount = 0;
                    for (size_t i = 0; i < SubVertStack.size(); ++i) {
                        const int rempt = SubVertStack[i];
                        if (pointtype(rempt) != FreeVolVertex) continue;
                        getvertexstar(1, rempt, &CaveTetList, nullptr, nullptr);
                        for (size_t j = 0; j < CaveTetList.size(); ++j) {
                            const Tet &e = Tets[CaveTetList[j].tet];
                            const double ori = orient3dfast(P(e.V[1]), P(e.V[0]), P(e.V[2]), P(e.V[3]));
                            if (j == 0 || opm.initval > ori) opm.initval = ori;
                        }
                        if (smoothpoint(Pts[rempt].Pos, CaveTetList, 1, &opm)) ++count;
                        if (opm.imprval <= 0.0) ++ivcount;
                        CaveTetList.clear();
                    }
                    if (count == 0) break;
                    ++nt;
                    if (nt > 2) break;
                }
                if (ivcount > 0 && opm.maxiter > 0) {
                    // Inverted elements remain: try again with an unlimited, finer search.
                    opm.numofsearchdirs = 30;
                    opm.searchstep = 0.0001;
                    opm.maxiter = -1;
                    continue;
                }
                break;
            }
        }

        SubVertStack.clear();
        return 1;
    }

    // The four-tier ladder over segments, then the same over subfaces.
    // Each tier flips harder and allows more Steiner points than the last.
    void recoverboundary() {
        std::vector<Facet> misseglist, misshlist, bdrysteinerptlist;
        Triface neightet;

        BoundaryRecoveryFlag = 1;
        CosCollinearAngTol = std::cos(CollinearAngTol / 180.0 * PI);
        if (SegmentEndpointsList.empty()) makesegmentendpointsmap();

        CheckSubsegFlag = 1;

        // Queue the segments in a random order.
        {
            std::vector<int> segs;
            forallsubsegs([&](int sh) { segs.push_back(sh); });
            for (size_t i = 0; i < segs.size(); ++i) {
                const size_t s = randomnation(unsigned(i) + 1);
                // Move the s-th segment to the i-th, and put this one at the s-th.
                SubSegStack.push_back(None);
                SubSegStack[i] = SubSegStack[s];
                SubSegStack[s] = SEncode2(segs[i], 0);
            }
        }

        long ms = SubsegItems;
        int nit = 0;
        if (FlipLinkLevel < 0) AutoFlipLinkLevel = 1;

        // Tier 1: flips only, at a climbing link level.
        bool first_pass = true;
        while (true) {
            recoversegments(&misseglist, 0, 0);
            if (first_pass) {
                MissingEdgeCount = long(misseglist.size());
                first_pass = false;
            }
            if (misseglist.empty()) break;
            if (FlipLinkLevel >= 0) break;
            if (long(misseglist.size()) >= ms) {
                ++nit;
                if (nit >= 3) FlipLinkLevel = 100000; // one last unbounded round
            } else {
                ms = long(misseglist.size());
                if (nit > 0) --nit;
            }
            for (auto &s : misseglist) SubSegStack.push_back(SEncode(s));
            misseglist.clear();
            AutoFlipLinkLevel += FlipLinkLevelInc;
        }

        // Requeue what is still missing and recover again, until a pass stops shrinking the list.
        const auto retry_segments = [&](int fullsearch, int steinerflag) {
            while (!misseglist.empty()) {
                ms = long(misseglist.size());
                for (auto &s : misseglist) SubSegStack.push_back(SEncode(s));
                misseglist.clear();
                recoversegments(&misseglist, fullsearch, steinerflag);
                if (long(misseglist.size()) >= ms) break;
            }
        };

        // Tier 2: the full crossing search.
        retry_segments(1, 0);

        // Tier 3: Delaunay recovery, then Steiner points in the volume.
        if (!misseglist.empty()) {
            recoverdelaunay();
            retry_segments(0, 1);
        }

        // Tier 4: also split the segments.
        if (!misseglist.empty()) {
            recoverdelaunay();
            retry_segments(0, 2);
        }

        if (StSegRefCount > 0) {
            // Try to take the segment Steiner points back out.
            const int bak = FlipLinkLevel;
            FlipLinkLevel = 20;
            for (const int rempt : SubVertStack) {
                if (!removevertexbyflips(rempt)) bdrysteinerptlist.push_back(Facet{rempt, 0});
            }
            FlipLinkLevel = bak;
            SubVertStack.clear();
        }

        CheckSubfaceFlag = 1;

        {
            std::vector<int> shs;
            forallsubfaces([&](int sh) { shs.push_back(sh); });
            for (size_t i = 0; i < shs.size(); ++i) {
                const size_t s = randomnation(unsigned(i) + 1);
                // Move the s-th subface to the i-th, and put this one at the s-th.
                SubFaceStack.push_back(None);
                SubFaceStack[i] = SubFaceStack[s];
                SubFaceStack[s] = SEncode2(shs[i], 0);
            }
        }

        ms = SubfaceItems;
        nit = 0;
        FlipLinkLevel = -1;
        AutoFlipLinkLevel = 1;

        first_pass = true;
        while (true) {
            recoversubfaces(&misshlist, 0);
            if (first_pass) {
                MissingFaceCount = long(misshlist.size());
                first_pass = false;
            }
            if (misshlist.empty()) break;
            if (FlipLinkLevel >= 0) break;
            if (long(misshlist.size()) >= ms) {
                ++nit;
                if (nit >= 3) FlipLinkLevel = AutoFlipLinkLevel < 30 ? 30 : AutoFlipLinkLevel + 30;
            } else {
                ms = long(misshlist.size());
                if (nit > 0) --nit;
            }
            for (auto &s : misshlist) SubFaceStack.push_back(SEncode(s));
            misshlist.clear();
            AutoFlipLinkLevel += FlipLinkLevelInc;
        }

        if (!misshlist.empty()) {
            recoverdelaunay();
            while (!misshlist.empty()) {
                ms = long(misshlist.size());
                for (auto &s : misshlist) SubFaceStack.push_back(SEncode(s));
                misshlist.clear();
                recoversubfaces(&misshlist, 1);
                if (long(misshlist.size()) < ms) continue;
                break;
            }
        }

        if (!misshlist.empty()) {
            recoverdelaunay();
            while (!misshlist.empty()) {
                ms = long(misshlist.size());
                for (auto &s : misshlist) SubFaceStack.push_back(SEncode(s));
                misshlist.clear();
                recoversubfaces(&misshlist, 2);
                if (long(misshlist.size()) < ms) continue;
                break;
            }
            if (!SubSegStack.empty()) {
                for (const int enc : SubSegStack) {
                    Facet checkseg;
                    SDecode(enc, checkseg);
                    if (checkseg.sh == None || isdeadsh(checkseg)) continue;
                    sstpivot1(checkseg, neightet);
                    if (neightet.tet != None) continue;
                    misseglist.push_back(checkseg);
                }
                SubSegStack.clear();
            }
        }

        if (!misshlist.empty()) bail(2);

        if (DuplicatedFacetsCount > 0) {
            // Drop the ignored duplicates and rebuild the face ring at every remaining subface.
            Triface spintet;
            Facet sseg;
            std::vector<int> shs;
            forallsubfaces([&](int sh) { shs.push_back(sh); });
            for (const int sh : shs) {
                Facet faceloop{sh, 0};
                if (isdeadsh(faceloop)) continue;
                if (sinfected(faceloop)) {
                    subfacedealloc(faceloop.sh);
                    continue;
                }
                if (smarktest3ed(faceloop)) continue;
                faceloop.shver = 0;
                stpivot(faceloop, neightet);
                if (neightet.tet == None) bail(2);
                for (int k = 0; k < 3; ++k) {
                    sspivot(faceloop, sseg);
                    if (sseg.sh != None) ssbond(faceloop, sseg);
                    std::vector<Facet> sfaces;
                    spintet = neightet;
                    do {
                        if (issubface(spintet)) {
                            Facet f;
                            tspivot(spintet, f);
                            sfaces.push_back(f);
                        }
                        fnextself(spintet);
                    } while (spintet.tet != neightet.tet);
                    for (size_t j = 0; j + 1 < sfaces.size(); ++j) sbond1(sfaces[j], sfaces[j + 1]);
                    if (!sfaces.empty()) sbond1(sfaces.back(), sfaces[0]);
                    enextself(neightet);
                    senextself(faceloop);
                }
            }
        }

        if (StFacRefCount > 0) {
            const int bak = FlipLinkLevel;
            FlipLinkLevel = 30;
            for (const int rempt : SubVertStack) {
                if (!removevertexbyflips(rempt)) bdrysteinerptlist.push_back(Facet{rempt, 0});
            }
            FlipLinkLevel = bak;
            SubVertStack.clear();
        }

        if (!misseglist.empty()) {
            Triface adjtet;
            for (auto &checkseg : misseglist) {
                if (checkseg.sh == None || isdeadsh(checkseg)) continue;
                sstpivot1(checkseg, adjtet);
                if (adjtet.tet != None) continue;
                SubSegStack.push_back(SEncode(checkseg));
            }
            SubSegStack.clear();
        }

        BoundaryRecoveryFlag = 0;
    }

    // Flood from the hull inward across every face that is not a subface.
    // What the flood reaches is deleted, and a fresh hull closes the domain.
    void carveholes() {
        std::vector<Triface> tetarray, hullarray, newhullfacearray;
        Triface tetloop, neightet, hulltet, casface;
        Facet checksh, checkseg;

        for (int t = 0; t < int(Tets.size()); ++t) {
            if (Tets[t].V[0] == None) continue;
            tetloop = Triface{t, 11}; // the face opposite the point at infinity
            if (!ishulltet(tetloop)) continue;
            if (issubface(tetloop)) continue;
            infect(tetloop);
            hullarray.push_back(tetloop);
            Decode(Tets[tetloop.tet].N[3], neightet);
            if (!infected(neightet)) {
                infect(neightet);
                tetarray.push_back(neightet);
            }
        }

        for (size_t i = 0; i < tetarray.size(); ++i) {
            Triface parytet = tetarray[i];
            const int j = parytet.ver & 3;
            for (int k = 1; k < 4; ++k) {
                Decode(Tets[parytet.tet].N[(j + k) % 4], neightet);
                if (!infected(neightet)) {
                    if (!issubface(neightet)) {
                        infect(neightet);
                        tetarray.push_back(neightet);
                    } else if (ishulltet(neightet)) {
                        infect(neightet);
                        hullarray.push_back(neightet);
                        tspivot(neightet, checksh);
                        sinfect(checksh);
                        SubFaceStack.push_back(SEncode(checksh));
                    }
                } else if (issubface(neightet)) {
                    tspivot(neightet, checksh);
                    if (!sinfected(checksh)) {
                        sinfect(checksh);
                        SubFaceStack.push_back(SEncode(checksh));
                    }
                }
            }
        }

        for (int p = 1; p < int(Pts.size()); ++p) {
            if (pointtype(p) == UnusedVertex || pointtype(p) == DuplicatedVertex) continue;
            Decode(point2tet(p), neightet);
            if (neightet.tet != None && infected(neightet)) CaveTetVertList.push_back(p);
            if (SupSteinerLevel > 0 && p > NumInputPoints) SubVertStack.push_back(p);
        }

        if (!tetarray.empty()) {
            {
                std::vector<int> segs;
                forallsubsegs([&](int sh) { segs.push_back(sh); });
                for (const int sh : segs) {
                    Facet segloop{sh, 0};
                    sstpivot1(segloop, neightet);
                    if (neightet.tet != None && infected(neightet)) SubSegStack.push_back(SEncode(segloop));
                }
            }

            for (size_t i = 0; i < tetarray.size(); ++i) {
                const Triface parytet = tetarray[i];
                for (int j = 0; j < 4; ++j) {
                    Decode(Tets[parytet.tet].N[j], tetloop);
                    if (infected(tetloop)) continue;
                    tspivot(tetloop, checksh);
                    maketetrahedron(&hulltet);
                    const int pa = org(tetloop), pb = dest(tetloop), pc = apex(tetloop);
                    setvertices(hulltet, pb, pa, pc, DummyPoint);
                    bond(tetloop, hulltet);
                    sesymself(checksh);
                    tsbond(hulltet, checksh);
                    for (int k = 0; k < 3; ++k) {
                        if (issubseg(tetloop)) {
                            tsspivot1(tetloop, checkseg);
                            bond_seg(hulltet, checkseg);
                        }
                        enextself(tetloop);
                        eprevself(hulltet);
                    }
                    setpoint2tet(pa, Encode2(tetloop.tet, 0));
                    setpoint2tet(pb, Encode2(tetloop.tet, 0));
                    setpoint2tet(pc, Encode2(tetloop.tet, 0));
                    newhullfacearray.push_back(Triface{parytet.tet, j});
                }
            }

            for (size_t i = 0; i < newhullfacearray.size(); ++i) {
                Triface parytet = newhullfacearray[i];
                fsym(parytet, neightet);
                fsym(neightet, hulltet);
                for (int j = 0; j < 3; ++j) {
                    esym(hulltet, casface);
                    if (Tets[casface.tet].N[casface.ver & 3] == None) {
                        neightet = parytet;
                        while (true) {
                            fnextself(neightet);
                            if (!infected(neightet)) break;
                        }
                        if (!ishulltet(neightet)) {
                            fsymself(neightet);
                            esymself(neightet);
                        }
                        bond(casface, neightet);
                    }
                    enextself(hulltet);
                    enextself(parytet);
                }
            }

            if (!SubFaceStack.empty()) {
                Facet casingout, casingin;
                for (const int enc : SubFaceStack) {
                    Facet parysh;
                    SDecode(enc, parysh);
                    for (int j = 0; j < 3; ++j) {
                        spivot(parysh, casingout);
                        sspivot(parysh, checkseg);
                        if (casingout.sh != None) {
                            casingin = casingout;
                            while (true) {
                                spivot(casingin, checksh);
                                if (checksh.sh == parysh.sh) break;
                                casingin = checksh;
                            }
                            if (casingin.sh != casingout.sh) sbond1(casingin, casingout);
                            else sdissolve(casingout);
                            if (checkseg.sh != None) ssbond(casingout, checkseg);
                        } else if (checkseg.sh != None) {
                            subsegdealloc(checkseg.sh);
                        }
                        senextself(parysh);
                    }
                    subfacedealloc(parysh.sh);
                }
                SubFaceStack.clear();
            }

            if (!SubSegStack.empty()) {
                for (const int enc : SubSegStack) {
                    Facet paryseg;
                    SDecode(enc, paryseg);
                    if (paryseg.sh == None || isdeadsh(paryseg)) continue;
                    sstpivot1(paryseg, neightet);
                    if (neightet.tet != None && infected(neightet)) {
                        subsegdealloc(paryseg.sh);
                    }
                }
                SubSegStack.clear();
            }

            for (const int parypt : CaveTetVertList) {
                Decode(point2tet(parypt), neightet);
                if (neightet.tet == None || !infected(neightet)) continue;
                if (parypt > NumInputPoints) {
                    if (pointtype(parypt) == FreeSegVertex) --StSegRefCount;
                    else if (pointtype(parypt) == FreeFacetVertex) --StFacRefCount;
                    else --StVolRefCount;
                    if (SteinerLeft > 0) ++SteinerLeft;
                }
                setpointtype(parypt, UnusedVertex);
                ++UnuVerts;
            }
            CaveTetVertList.clear();

            HullSize += long(newhullfacearray.size()) - long(hullarray.size());

            for (auto &t : tetarray) tetrahedrondealloc(t.tet);
            for (auto &t : hullarray) tetrahedrondealloc(t.tet);
        }

        NonConvex = 1;

        // Peel the slivers off the hull.
        tetloop.ver = 11; // The face opposite dummypoint.
        for (int t = 0; t < int(Tets.size()); ++t) {
            if (Tets[t].V[0] == None) continue;
            if (Tets[t].V[3] != DummyPoint) continue;
            tetloop.tet = t;
            fsym(tetloop, neightet);
            flippush(&neightet);
        }
        FlipConstraints fc;
        fc.enqflag = 2;
        lawsonflip3d(&fc);
        UnflipQueue.clear();
    }

    int search_face(int pi, int pj, int pk, Triface &tetloop) {
        if (!getedge(pi, pj, &tetloop)) return 0;
        Triface spintet = tetloop;
        while (true) {
            if (apex(spintet) == pk) {
                tetloop = spintet;
                return 1;
            }
            fnextself(spintet);
            if (spintet.tet == tetloop.tet) break;
        }
        return 0;
    }

    // The flip stack and the unflippable queue share a size budget.
    std::vector<BadFace> UnflipQueue, LaterUnflipQueue;
    size_t flippool_items() const { return FlipStack.size() + UnflipQueue.size(); }
    double TetPrismVolSum{0};
    long RecoverDelaunayCount{0};

    // Bank the prism volume the last flip changed and reset the running total.
    void bank_prism_vol(FlipConstraints *fc) {
        if (!fc->remove_ndelaunay_edge) return;
        TetPrismVolSum += fc->tetprism_vol_sum;
        fc->tetprism_vol_sum = 0.0;
    }

    // Flip every stacked face that is not locally Delaunay.
    // A face no flip reaches is queued and retried once the round has made progress.
    long lawsonflip3d(FlipConstraints *fc) {
        Triface fliptets[5], neightet, hulltet;
        Facet checksh, casingout;
        long totalcount = 0, sliver_peels = 0;

        while (flippool_items() != 0) {
            long flipcount = 0;

            while (!FlipStack.empty()) {
                const BadFace popface = FlipStack.back();
                FlipStack.pop_back();
                fliptets[0] = popface.tt;
                if (isdeadtet(fliptets[0])) continue;
                if (!facemarked(fliptets[0])) continue;
                unmarkface(fliptets[0]);
                if (ishulltet(fliptets[0])) continue;

                fsym(fliptets[0], fliptets[1]);
                if (ishulltet(fliptets[1])) {
                    if (NonConvex) {
                        // The tet may be a hull sliver: two of its faces on the hull at one edge.
                        tspivot(fliptets[0], checksh);
                        if (checksh.sh == None) continue;
                        for (int i = 0; i < 3; ++i) {
                            if (!isshsubseg(checksh)) {
                                spivot(checksh, casingout);
                                if (casingout.sh != None) {
                                    if (sorg(checksh) != sdest(casingout)) sesymself(casingout);
                                    stpivot(casingout, neightet);
                                    if (neightet.tet == fliptets[0].tet) {
                                        edestoppo(neightet, hulltet); // [a,b,e,d]
                                        fsymself(hulltet);
                                        if (oppo(hulltet) == DummyPoint) {
                                            const int pe = org(neightet);
                                            if (pointtype(pe) == FreeFacetVertex || pointtype(pe) == FreeSegVertex) removevertexbyflips(pe);
                                        } else {
                                            eorgoppo(neightet, hulltet); // [b,a,d,e]
                                            fsymself(hulltet);
                                            if (oppo(hulltet) == DummyPoint) {
                                                const int pd = dest(neightet);
                                                if (pointtype(pd) == FreeFacetVertex || pointtype(pd) == FreeSegVertex) removevertexbyflips(pd);
                                            } else {
                                                // Peel the sliver with a 3-2 flip, unless the two new subfaces would point opposite ways.
                                                const int chk_pe = org(neightet), chk_pd = dest(neightet);
                                                const int chk_pa = apex(neightet), chk_pb = oppo(neightet);
                                                dvec3 n1, n2;
                                                facenormal(P(chk_pa), P(chk_pb), P(chk_pe), n1, 1, nullptr);
                                                facenormal(P(chk_pb), P(chk_pa), P(chk_pd), n2, 1, nullptr);
                                                if (dot(n1, n2) > 0.0) {
                                                    fliptets[0] = neightet;
                                                    fnext(fliptets[0], fliptets[1]);
                                                    fnext(fliptets[1], fliptets[2]);
                                                    flip32(fliptets, 1, fc);
                                                    --Flip32Count;
                                                    --Flip22Count;
                                                    ++sliver_peels;
                                                    bank_prism_vol(fc);
                                                }
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                            senextself(checksh);
                        }
                    }
                    continue;
                }

                if (CheckSubfaceFlag && issubface(fliptets[0])) continue;

                const Tet &e1 = Tets[fliptets[1].tet];
                if (insphere_s(e1.V[0], e1.V[1], e1.V[2], e1.V[3], oppo(fliptets[0])) >= 0) continue;

                const int pd = oppo(fliptets[0]), pe = oppo(fliptets[1]);
                double len3 = distance(P(pd), P(pe));
                len3 = len3 * len3 * len3;
                int round_flag = 0;
                double ori = 0;
                int i = 0;
                for (; i < 3; ++i) {
                    ori = orient3d(P(org(fliptets[0])), P(dest(fliptets[0])), P(pd), P(pe));
                    if (ori > 0) {
                        // Refuse a nearly degenerate new tet against the boundary.
                        esym(fliptets[0], fliptets[2]);
                        esym(fliptets[1], fliptets[3]);
                        if (issubface(fliptets[2]) || issubface(fliptets[3])) {
                            const double vol = orient3dfast(P(org(fliptets[0])), P(dest(fliptets[0])), P(pd), P(pe));
                            if (std::abs(vol) / len3 < Epsilon) {
                                ori = 0.0;
                                round_flag = 1;
                            }
                        }
                    }
                    if (ori <= 0) break;
                    enextself(fliptets[0]);
                    eprevself(fliptets[1]);
                }

                if (ori > 0) {
                    flip23(fliptets, 0, fc);
                    ++flipcount;
                    bank_prism_vol(fc);
                    continue;
                }

                if (CheckSubsegFlag && issubseg(fliptets[0])) continue;

                int scount = 0;
                esymself(fliptets[0]); // [b,a,d,c]
                for (int k = 0; k < 3; ++k) {
                    if (issubface(fliptets[k])) ++scount;
                    fnext(fliptets[k], fliptets[k + 1]);
                }
                if (fliptets[3].tet == fliptets[0].tet) {
                    if (scount == 1) continue; // the neighbouring subface is not back yet
                    if (scount == 2) {
                        int k = 0;
                        for (; k < 3; ++k) {
                            if (!issubface(fliptets[k])) break;
                        }
                        Triface face1, face2;
                        neightet = fliptets[(k + 1) % 3];
                        enext(neightet, face1);
                        esymself(face1);
                        eprev(neightet, face2);
                        esymself(face2);
                        if (issubface(face1) || issubface(face2)) continue;
                    }
                    flip32(fliptets, 0, fc);
                    ++flipcount;
                    bank_prism_vol(fc);
                    continue;
                }

                fnext(fliptets[3], fliptets[4]);
                if (fliptets[4].tet == fliptets[0].tet) {
                    if (ori != 0.0 && NonConvex && apex(fliptets[3]) == DummyPoint) {
                        ori = 0;
                        round_flag = 1;
                    }
                    if (ori == 0) {
                        // A 4-4 flip: the edge [a,b] becomes [d,e].
                        if (issubface(fliptets[0])) {
                            if (!issubface(fliptets[2])) continue;
                            if (issubface(fliptets[1]) || issubface(fliptets[3])) continue;
                        } else {
                            if (issubface(fliptets[1]) || issubface(fliptets[2]) || issubface(fliptets[3])) continue;
                        }
                        if (round_flag == 1) {
                            // The edge is only nearly coplanar.
                            // Take it only when every new tet is valid and every new face is locally Delaunay, or this will not stop.
                            const int pb = org(fliptets[0]), pa = dest(fliptets[0]);
                            const int pc = apex(fliptets[1]), pf = apex(fliptets[3]);
                            if (is_collinear_at(pa, pd, pe) || is_collinear_at(pb, pd, pe)) continue;
                            if (orient3d(P(pe), P(pd), P(pc), P(pa)) >= 0.0) continue;
                            if (orient3d(P(pe), P(pd), P(pb), P(pc)) >= 0.0) continue;
                            if (pf != DummyPoint) {
                                if (orient3d(P(pe), P(pd), P(pa), P(pf)) >= 0.0) continue;
                                if (orient3d(P(pe), P(pd), P(pf), P(pb)) >= 0.0) continue;
                            }
                            if (insphere_s(pe, pd, pc, pa, pb) < 0) continue;
                            if (pf != DummyPoint && insphere_s(pe, pd, pf, pb, pa) < 0) continue;
                        }
                        esymself(fliptets[0]); // [a,b,c,d]
                        flip23(fliptets, 0, fc);
                        fnext(fliptets[3], fliptets[1]);
                        fnext(fliptets[1], fliptets[2]);
                        flip32(&fliptets[1], apex(fliptets[3]) == DummyPoint, fc);
                        ++flipcount;
                        --Flip23Count;
                        --Flip32Count;
                        ++Flip44Count;
                        bank_prism_vol(fc);
                        continue;
                    }
                }

                // Nothing flips this face.
                // Keep it for the next round.
                {
                    BadFace bface;
                    esymself(fliptets[0]);
                    bface.tt = fliptets[0];
                    bface.forg = org(fliptets[0]);
                    bface.fdest = dest(fliptets[0]);
                    bface.fapex = apex(fliptets[0]);
                    UnflipQueue.push_back(bface);
                }
            }

            totalcount += flipcount;
            if (flippool_items() == 0) break;
            if (flipcount == 0) break;

            std::vector<BadFace> retry;
            retry.swap(UnflipQueue);
            for (auto &bface : retry) {
                if (!isdeadtet(bface.tt) && org(bface.tt) == bface.forg && dest(bface.tt) == bface.fdest && apex(bface.tt) == bface.fapex) {
                    flippush(&bface.tt);
                }
            }
        }

        if (!UnflipQueue.empty()) {
            for (auto &bface : UnflipQueue) {
                if (!isdeadtet(bface.tt) && org(bface.tt) == bface.forg && dest(bface.tt) == bface.fdest && apex(bface.tt) == bface.fapex) {
                    LaterUnflipQueue.push_back(bface);
                }
            }
            UnflipQueue.clear();
            FlipStack.clear();
        }
        return totalcount + sliver_peels;
    }

    // A Lawson sweep, then an edge-removal pass at climbing link level over what the sweep could not flip.
    // Both are gated on the lifted volume falling.
    void recoverdelaunay() {
        FlipConstraints fc;
        TetPrismVolSum = 0.0;

        if (!LaterUnflipQueue.empty()) {
            for (auto &bface : LaterUnflipQueue) {
                if (!isdeadtet(bface.tt) && org(bface.tt) == bface.forg && dest(bface.tt) == bface.fdest && apex(bface.tt) == bface.fapex) {
                    flippush(&bface.tt);
                }
            }
            LaterUnflipQueue.clear();
            if (flippool_items() == 0) return;
        } else if (flippool_items() == 0) {
            Triface tetloop, neightet;
            for (int t = 0; t < int(Tets.size()); ++t) {
                if (Tets[t].V[0] == None || Tets[t].V[3] == DummyPoint) continue;
                tetloop.tet = t;
                for (tetloop.ver = 0; tetloop.ver < 4; ++tetloop.ver) {
                    Decode(Tets[t].N[tetloop.ver], neightet);
                    if (neightet.tet != None && !facemarked(neightet)) flippush(&tetloop);
                }
                const Tet &e = Tets[t];
                TetPrismVolSum += tetprismvol(P(e.V[0]), P(e.V[1]), P(e.V[2]), P(e.V[3]));
            }
        }
        ++RecoverDelaunayCount;

        // A floor below which a volume change is only rounding.
        fc.bak_tetprism_vol = TetPrismVolSum * Epsilon * 1e-3;

        fc.remove_ndelaunay_edge = 1;
        fc.enqflag = 2;
        lawsonflip3d(&fc);
        if (LaterUnflipQueue.empty()) return;

        fc.unflip = 0;
        fc.collectnewtets = 1;
        fc.enqflag = 0;

        const int bak_autofliplinklevel = AutoFlipLinkLevel;
        const int bak_fliplinklevel = FlipLinkLevel;
        AutoFlipLinkLevel = 1;
        FlipLinkLevel = -1;

        while (!LaterUnflipQueue.empty() && AutoFlipLinkLevel < 4) {
            std::vector<BadFace> bfarray;
            bfarray.swap(LaterUnflipQueue);
            for (auto &bface : bfarray) {
                if (!getedge(bface.forg, bface.fdest, &bface.tt)) continue;
                if (removeedgebyflips(&bface.tt, &fc) == 2) {
                    TetPrismVolSum += fc.tetprism_vol_sum;
                } else {
                    LaterUnflipQueue.push_back(bface);
                }
                fc.tetprism_vol_sum = 0.0;
                if (!CaveTetList.empty()) {
                    Triface neightet;
                    for (auto &parytet : CaveTetList) {
                        if (isdeadtet(parytet)) continue;
                        for (parytet.ver = 0; parytet.ver < 4; ++parytet.ver) {
                            Decode(Tets[parytet.tet].N[parytet.ver], neightet);
                            if (neightet.tet != None && !facemarked(neightet)) flippush(&parytet);
                        }
                    }
                    CaveTetList.clear();
                }
            }
            ++AutoFlipLinkLevel;
        }

        if (flippool_items() > 0) {
            fc.remove_ndelaunay_edge = 1;
            fc.enqflag = 2;
            lawsonflip3d(&fc);
        }
        LaterUnflipQueue.clear();
        AutoFlipLinkLevel = bak_autofliplinklevel;
        FlipLinkLevel = bak_fliplinklevel;
    }

    //=== Element quality, encroachment and the small queue helpers ===

    double CosLargeDihed{0};

    // These maps are only built for refinement; with no map, nothing is adjacent.
    int facet_ridge_vertex_adjacent(Facet *chkfac, int chkpt) {
        if (IdxRidgeVertexFacetList.empty()) return 0;
        const int facidx = getfacetindex(*chkfac);
        for (int i = IdxRidgeVertexFacetList[chkpt]; i < IdxRidgeVertexFacetList[chkpt + 1]; ++i) {
            if (RidgeVertexFacetList[i] == facidx) return 1;
        }
        return 0;
    }
    int segfacetadjacent(Facet *subseg, Facet *subsh) {
        if (IdxSegmentFacetList.empty()) return 0;
        const int segidx = getfacetindex(*subseg);
        const int facidx = getfacetindex(*subsh);
        for (int i = IdxSegmentFacetList[segidx]; i < IdxSegmentFacetList[segidx + 1]; ++i) {
            if (SegmentFacetList[i] == facidx) return 1;
        }
        return 0;
    }

    // May a Steiner point be placed this close to an existing vertex?
    // Only when the two do not already share the segment or facet they sit on.
    bool create_a_shorter_edge(int steinerpt, int nearpt) {
        const VertType nearpt_type = pointtype(nearpt);
        const VertType steiner_type = pointtype(steinerpt);
        if (nearpt_type == RidgeVertex) {
            if (steiner_type == FreeSegVertex) {
                Facet parentseg;
                SDecode(point2sh(steinerpt), parentseg);
                const int segidx = getfacetindex(parentseg);
                const int pa = SegmentEndpointsList[segidx * 2], pb = SegmentEndpointsList[segidx * 2 + 1];
                return pa != nearpt && pb != nearpt;
            }
            if (steiner_type == FreeFacetVertex) {
                Facet parentsh;
                SDecode(point2sh(steinerpt), parentsh);
                return !facet_ridge_vertex_adjacent(&parentsh, nearpt);
            }
        } else if (nearpt_type == FreeSegVertex) {
            if (steiner_type == FreeSegVertex) {
                Facet seg1, seg2;
                SDecode(point2sh(steinerpt), seg1);
                SDecode(point2sh(nearpt), seg2);
                return getfacetindex(seg1) != getfacetindex(seg2);
            }
            if (steiner_type == FreeFacetVertex) {
                Facet parentseg, parentsh;
                SDecode(point2sh(steinerpt), parentsh);
                SDecode(point2sh(nearpt), parentseg);
                return !segfacetadjacent(&parentseg, &parentsh);
            }
        } else if (nearpt_type == FreeFacetVertex) {
            if (steiner_type == FreeSegVertex) {
                Facet parentseg, parentsh;
                SDecode(point2sh(nearpt), parentsh);
                SDecode(point2sh(steinerpt), parentseg);
                return !segfacetadjacent(&parentseg, &parentsh);
            }
            if (steiner_type == FreeFacetVertex) {
                Facet sh1, sh2;
                SDecode(point2sh(nearpt), sh1);
                SDecode(point2sh(steinerpt), sh2);
                return getfacetindex(sh1) != getfacetindex(sh2);
            }
        }
        return false;
    }

    void enqueuesubface(std::vector<BadFace> &pool, Facet *chkface) {
        if (smarktest2ed(*chkface)) return;
        smarktest2(*chkface);
        BadFace bf;
        bf.ss = *chkface;
        pool.push_back(bf);
    }
    void enqueuetetrahedron(Triface *chktet) {
        if (marktest2ed(*chktet)) return;
        marktest2(*chktet);
        BadFace bf;
        bf.tt = *chktet;
        BadTets.push_back(bf);
    }
    std::vector<BadFace> BadTets;

    // Does checkpt see the segment (pa, pb) at an obtuse angle, that is, does it fall inside the segment's diametral sphere?
    bool check_encroachment(int pa, int pb, int checkpt) const {
        return dot(P(pa) - P(checkpt), P(pb) - P(checkpt)) < 0.0;
    }

    bool get_subface_ccent(Facet *chkfac, dvec3 &pos) {
        const int p = Shells[chkfac->sh].V[0], q = Shells[chkfac->sh].V[1], r = Shells[chkfac->sh].V[2];
        if (circumsphere(P(p), P(q), P(r), nullptr, &pos, nullptr)) return true;
        bail(2);
    }

    // Is the subface's diametral ball violated, either by a given point or by one of the two apexes across it?
    bool check_enc_subface(Facet *chkfac, int *pencpt, const dvec3 &ccent, double *radius) {
        Triface adjtet;
        int encpt = None;
        dvec3 prjpt;
        double minprjdist = 0., prjdist;

        const double rd = distance(ccent, P(sorg(*chkfac)));
        *radius = rd;

        if (*pencpt != None) {
            double len = distance(ccent, P(*pencpt));
            if (std::abs(len - rd) / rd < 1e-3) len = rd;
            return len < rd;
        }

        stpivot(*chkfac, adjtet);
        if (adjtet.tet == None) return false;
        for (int i = 0; i < 2; ++i) {
            const int toppo = oppo(adjtet);
            if (toppo != DummyPoint) {
                double len = distance(ccent, P(toppo));
                if (std::abs(len - rd) / rd < 1e-3) len = rd;
                if (len < rd) {
                    int adjacent = 0;
                    if (pointtype(toppo) == RidgeVertex) {
                        adjacent = facet_ridge_vertex_adjacent(chkfac, toppo);
                    } else if (pointtype(toppo) == FreeSegVertex) {
                        Facet parentseg;
                        SDecode(point2sh(toppo), parentseg);
                        adjacent = segfacetadjacent(&parentseg, chkfac);
                    } else if (pointtype(toppo) == FreeFacetVertex) {
                        Facet parentsh;
                        SDecode(point2sh(toppo), parentsh);
                        if (getfacetindex(parentsh) == getfacetindex(*chkfac)) adjacent = 1;
                    }
                    if (adjacent) {
                        flippush(&adjtet);
                        return false;
                    }
                    const int pa = org(adjtet), pb = dest(adjtet), pc = apex(adjtet);
                    projpt2face(P(toppo), P(pa), P(pb), P(pc), prjpt);
                    if (orient3d(P(pa), P(pb), P(toppo), prjpt) >= 0 && orient3d(P(pb), P(pc), P(toppo), prjpt) >= 0 && orient3d(P(pc), P(pa), P(toppo), prjpt) >= 0) {
                        prjdist = distance(P(toppo), prjpt);
                        if (encpt == None || prjdist < minprjdist) {
                            encpt = toppo;
                            minprjdist = prjdist;
                        }
                    }
                }
            }
            fsymself(adjtet);
        }
        if (encpt != None) {
            *pencpt = encpt;
            return true;
        }
        return false;
    }

    // No background mesh is read, so no size is interpolated and the point keeps the default.
    double getpointmeshsize(int, Triface *, int) const { return 0.0; }

    // The aspect ratio, the extreme dihedral angles and the edge lengths of one tet, all read off the four inward face normals.
    bool get_tetqual_at(const dvec3 &pa, const dvec3 &pb, const dvec3 &pc, const dvec3 &pd, BadFace *bf) {
        double A[4][4]{}, D;
        int indx[4]{};

        const dvec3 da = pa - pd, db = pb - pd, dc = pc - pd;
        set_rows(A, da, db, dc);

        const dvec3 Vab = pb - pa, Vbc = pc - pb, Vca = pa - pc;
        // The six squared edge lengths, in the order the queue records them.
        const double L[6]{dot(dc, dc), dot(da, da), dot(Vab, Vab), dot(Vbc, Vbc), dot(db, db), dot(Vca, Vca)};
        const auto [shortest, longest] = std::ranges::minmax_element(L);
        const double Lmax = std::sqrt(*longest), Lmin = std::sqrt(*shortest);
        bf->cent[2] = Lmax / Lmin;
        bf->cent[3] = Lmin;

        dvec3 N[4];
        double H[4];
        bool flat_flag = false;
        if (lu_decmp(A, 3, indx, &D, 0)) {
            bf->cent[4] = std::abs(A[indx[0]][0] * A[indx[1]][1] * A[indx[2]][2]);
            if (bf->cent[4] > 0.0) {
                face_normals(A, indx, N);
            } else {
                flat_flag = true;
            }
        } else {
            flat_flag = true;
        }

        if (flat_flag) {
            bf->cent[4] = orient3d(pb, pa, pc, pd);
            if (bf->cent[4] <= 0.0) return false; // degenerate or inverted
            facenormal(pc, pb, pd, N[0], 1, nullptr);
            facenormal(pa, pc, pd, N[1], 1, nullptr);
            facenormal(pb, pa, pd, N[2], 1, nullptr);
            facenormal(pa, pb, pc, N[3], 1, nullptr);
        }

        for (int i = 0; i < 4; ++i) {
            H[i] = std::sqrt(dot(N[i], N[i]));
            if (H[i] <= 0.0) return false;
            N[i] /= H[i];
        }

        if (!flat_flag) {
            bf->key = Lmax * std::ranges::max(H);
        } else {
            bf->key = 1.e+30;
        }

        double cosmaxd = 1.0, cosmind = -1.0;
        int idx = 0;
        bf->ss.shver = 0;
        for (int i = 0; i < 6; ++i) {
            const auto [f1, f2] = DihedralFaces[i];
            double cosd = -dot(N[f1], N[f2]);
            if (cosd < -1.0) cosd = -1.0;
            if (cosd > 1.0) cosd = 1.0;
            if (cosd < cosmaxd) {
                cosmaxd = cosd;
                idx = i;
            }
            cosmind = std::max(cosd, cosmind);
            if (cosd < CosLargeDihed) ++bf->ss.shver;
        }
        bf->cent[0] = cosmaxd;
        bf->cent[1] = cosmind;
        bf->tt.ver = edge2ver[idx];
        bf->cent[5] = 0.0;
        return true;
    }
    bool get_tetqual(int pa, int pb, int pc, int pd, BadFace *bf) {
        *bf = BadFace{};
        bf->forg = pa;
        bf->fdest = pb;
        bf->fapex = pc;
        bf->foppo = pd;
        return get_tetqual_at(P(pa), P(pb), P(pc), P(pd), bf);
    }
    // True when the tet's worst dihedral is an improvement on cosdihed_in, the angle a flip starts from.
    // A relative change under Epsilon leaves the angle where it was, and a tet with no measurable quality improves nothing.
    bool improves_dihedral(int pa, int pb, int pc, int pd, double cosdihed_in, BadFace *bf) {
        if (!get_tetqual(pa, pb, pc, pd, bf)) return false;
        const double diff = bf->cent[0] - cosdihed_in;
        return diff > 0 && std::abs(diff / cosdihed_in) >= Epsilon;
    }
    bool get_tetqual(Triface *chktet, int oppo_pt, BadFace *bf) {
        if (chktet != nullptr) {
            *bf = BadFace{};
            if (oppo_pt == None) {
                const Tet &e = Tets[chktet->tet];
                bf->forg = e.V[0];
                bf->fdest = e.V[1];
                bf->fapex = e.V[2];
                bf->foppo = e.V[3];
            } else {
                bf->forg = org(*chktet);
                bf->fdest = dest(*chktet);
                bf->fapex = apex(*chktet);
                bf->foppo = oppo_pt;
            }
        }
        const bool ok = get_tetqual_at(P(bf->forg), P(bf->fdest), P(bf->fapex), P(bf->foppo), bf);
        if (chktet) bf->tt.tet = chktet->tet;
        return ok;
    }

    bool get_tet(int pa, int pb, int pc, int pd, Triface *searchtet) {
        if (!search_face(pa, pb, pc, *searchtet)) return false;
        if (oppo(*searchtet) == pd) return true;
        fsymself(*searchtet);
        return oppo(*searchtet) == pd;
    }

    //=== Removing a vertex from the surface mesh ===

    // Take delpt out of the surface triangulation by a sequence of 2-2 flips ending in a 3-1 flip.
    // With parentseg given, delpt is a Steiner point on a segment and the two half-segments merge back into one.
    // A degenerate subface holds the merged segment while the face ring is rebuilt around it.
    int sremovevertex(int delpt, Facet *parentsh, Facet *parentseg, int lawson) {
        Facet flipfaces[4], spinsh, fakesh;

        if (parentseg != nullptr) {
            // delpt (p) is a Steiner point in a segment [a,b] and parentseg is [p,b].
            Facet startsh, neighsh, nextsh;
            Facet abseg, prevseg, checkseg, adjseg1, adjseg2;
            senext2(*parentseg, prevseg);
            spivotself(prevseg);
            prevseg.shver = 0;
            const int pa = sorg(prevseg), pb = sdest(*parentseg);
            makesubseg(&abseg);
            setshvertices(abseg, pa, pb, None);
            setshellmark(abseg, shellmark(*parentseg));
            if (CheckConstraints) setareabound(abseg, areabound(*parentseg));
            if (UseInsertRadius) setfacetindex(abseg, getfacetindex(*parentseg));
            // Connect [#, a] to [a, b].
            senext2(prevseg, adjseg1);
            spivotself(adjseg1);
            if (adjseg1.sh != None) {
                adjseg1.shver = 0;
                senextself(adjseg1);
                senext2(abseg, adjseg2);
                sbond(adjseg1, adjseg2);
            }
            // Connect [a, b] to [b, #].
            senext(*parentseg, adjseg1);
            spivotself(adjseg1);
            if (adjseg1.sh != None) {
                adjseg1.shver = 0;
                senext2self(adjseg1);
                senext(abseg, adjseg2);
                sbond(adjseg1, adjseg2);
            }
            setpoint2sh(pa, SEncode(abseg));
            setpoint2sh(pb, SEncode(abseg));

            // The faces in the face ring at segment [p, b].
            spivot(*parentseg, *parentsh);
            if (parentsh->sh != None) {
                spinsh = *parentsh;
                while (true) {
                    CaveShList.push_back(spinsh);
                    spivotself(spinsh);
                    if (spinsh.sh == None) break; // Only one facet.
                    if (spinsh.sh == parentsh->sh) break;
                }
            }

            // The face ring of the new segment [a,b].
            // Each face in it is the degenerate [a,b,p], which the flips below remove.
            for (size_t i = 0; i < CaveShList.size(); ++i) {
                startsh = CaveShList[i];
                if (sorg(startsh) != delpt) sesymself(startsh);
                // startsh is [p, b, #1].
                // Find the subface [a, p, #2].
                neighsh = startsh;
                while (true) {
                    senext2self(neighsh);
                    sspivot(neighsh, checkseg);
                    if (checkseg.sh != None) break; // The segment [a, p].
                    spivotself(neighsh);
                    if (sorg(neighsh) != delpt) sesymself(neighsh);
                }
                // neighsh is [a, p, #2].
                if (neighsh.sh != startsh.sh) {
                    ssdissolve(startsh);
                    ssdissolve(neighsh);
                    // A degenerate subface [a,b,p] holds the new segment and joins [p,b,#1] to [a,p,#2].
                    makesubface(&fakesh);
                    setshvertices(fakesh, pa, pb, delpt);
                    setshellmark(fakesh, shellmark(startsh));
                    ssbond(fakesh, abseg);
                    senext(fakesh, nextsh);
                    sbond(nextsh, startsh);
                    senext2(fakesh, nextsh);
                    sbond(nextsh, neighsh);
                    smarktest(fakesh); // Faked.
                } else {
                    // A degenerate face [a,b,p] is already there.
                    senext2self(neighsh); // [a,b,p]
                    spivot(neighsh, startsh); // Its adjacent subface, which joins the new ring.
                    if (sorg(startsh) != pa) sesymself(startsh);
                    sdissolve(startsh);
                    ssbond(startsh, abseg);
                    fakesh = startsh; // Not marked.
                    subfacedealloc(neighsh.sh);
                }
                CaveSegShList.push_back(fakesh);
            }
            CaveShList.clear();

            // Re-create the face ring.
            if (CaveSegShList.size() > 1) {
                for (size_t i = 0; i < CaveSegShList.size(); ++i) {
                    fakesh = CaveSegShList[i];
                    nextsh = CaveSegShList[(i + 1) % CaveSegShList.size()];
                    sbond1(fakesh, nextsh);
                }
            }

            subsegdealloc(parentseg->sh);
            subsegdealloc(prevseg.sh);
            *parentseg = abseg;
        } else {
            // p is inside the surface.
            // Let delpt be the apex.
            senextself(*parentsh);
            CaveSegShList.push_back(*parentsh);
        }

        // Remove the point (p).
        for (size_t it = 0; it < CaveSegShList.size(); ++it) {
            Facet cur = CaveSegShList[it]; // [a,b,p]
            senextself(cur); // [b,p,a]
            spivotself(cur);
            if (sorg(cur) != delpt) sesymself(cur);
            // cur is [p,b,#].
            if (sorg(cur) != delpt) continue; // Already removed by the special case above.

            while (true) {
                // The flip edge list at p.
                spinsh = cur; // [p, b, #]
                while (true) {
                    CaveShList.push_back(spinsh);
                    senext2self(spinsh);
                    spivotself(spinsh);
                    if (spinsh.sh == cur.sh) break;
                    if (sorg(spinsh) != delpt) sesymself(spinsh);
                }

                if (CaveShList.size() == 3) {
                    for (int i = 0; i < 3; ++i) flipfaces[i] = CaveShList[i];
                    flip31(flipfaces, lawson);
                    for (int i = 0; i < 3; ++i) subfacedealloc(flipfaces[i].sh);
                    CaveShList.clear();
                    CaveShBdList.push_back(flipfaces[3]);
                    break; // The vertex is removed.
                }

                size_t i = 0;
                for (; i < CaveShList.size(); ++i) {
                    flipfaces[0] = CaveShList[i];
                    spivot(flipfaces[0], flipfaces[1]);
                    if (sorg(flipfaces[0]) != sdest(flipfaces[1])) sesymself(flipfaces[1]);
                    // Skip an edge of a faked subface.
                    if (!smarktested(flipfaces[0]) && !smarktested(flipfaces[1])) {
                        const int pa = sorg(flipfaces[0]), pb = sdest(flipfaces[0]);
                        const int pc = sapex(flipfaces[0]), pd = sapex(flipfaces[1]);
                        calculateabovepoint4(P(pa), P(pb), P(pc), P(pd));
                        const double ori1 = orient3d(P(pc), P(pd), P(DummyPoint), P(pa));
                        const double ori2 = orient3d(P(pc), P(pd), P(DummyPoint), P(pb));
                        if (ori1 * ori2 < 0) {
                            flip22(flipfaces, lawson, 0);
                            // flipfaces[1] holds p as its apex.
                            senext2(flipfaces[1], cur);
                            CaveShBdList.push_back(flipfaces[0]);
                            break;
                        }
                    }
                }

                if (i == CaveShList.size()) {
                    flipfaces[0] = CaveShList[0];
                    spivot(flipfaces[0], flipfaces[1]);
                    if (sorg(flipfaces[0]) != sdest(flipfaces[1])) sesymself(flipfaces[1]);
                    flip22(flipfaces, lawson, 0);
                    senext2(flipfaces[1], cur);
                    CaveShBdList.push_back(flipfaces[0]);
                }

                CaveShList.clear();
            }
        }
        CaveSegShList.clear();

        if (lawson) lawsonflip();
        return 0;
    }

    //=== Vertex smoothing ===

    bool BadTetsEnabled{false};

    // The midpoint of the two segment neighbours of a Steiner point sitting on a segment.
    bool get_seg_laplacian_center(int mesh_vert, dvec3 &target) {
        if (pointtype(mesh_vert) == UnusedVertex) return false;

        Facet leftseg, rightseg;
        SDecode(point2sh(mesh_vert), leftseg);
        leftseg.shver = 0;
        if (sdest(leftseg) == mesh_vert) {
            senext(leftseg, rightseg);
            spivotself(rightseg);
            rightseg.shver = 0;
            if (sorg(rightseg) != mesh_vert) sesymself(rightseg);
            if (sorg(rightseg) != mesh_vert) bail(2);
        } else {
            rightseg = leftseg;
            senext2(rightseg, leftseg);
            spivotself(leftseg);
            leftseg.shver = 0;
            if (sdest(leftseg) != mesh_vert) sesymself(leftseg);
            if (sdest(leftseg) != mesh_vert) bail(2);
        }
        target = 0.5 * (P(sorg(leftseg)) + P(sdest(rightseg)));
        return true;
    }

    // The centroid of the link vertices in the surface star of a Steiner point sitting in a facet.
    bool get_surf_laplacian_center(int mesh_vert, dvec3 &target) {
        if (pointtype(mesh_vert) == UnusedVertex) return false;

        getvertexstar(1, mesh_vert, &CaveOldTetList, nullptr, &CaveShList);
        // As many link vertices as link edges, each counted twice.
        const int npt = int(CaveShList.size());
        target = dvec3{0, 0, 0};
        // One endpoint at a time, which is the order the sum is accumulated in.
        for (const auto &cavesh : CaveShList) target += P(sorg(cavesh));
        for (const auto &cavesh : CaveShList) target += P(sdest(cavesh));
        target /= double(npt * 2);
        CaveOldTetList.clear();
        CaveShList.clear();
        return true;
    }

    // The centroid of the link vertices of an interior Steiner point.
    bool get_laplacian_center(int mesh_vert, dvec3 &target) {
        if (pointtype(mesh_vert) == UnusedVertex) return false;

        getvertexstar(1, mesh_vert, &CaveOldTetList, &CaveTetVertList, nullptr);
        const int npt = int(CaveTetVertList.size());
        target = dvec3{0, 0, 0};
        for (const int pt : CaveTetVertList) target += P(pt);
        target /= double(npt);
        CaveTetVertList.clear();
        return true;
    }

    // Step the vertex toward the target, halving the step until no tet of its star turns over.
    // Flips then restore the Delaunay property.
    bool move_vertex(int mesh_vert, const dvec3 &target) {
        if (pointtype(mesh_vert) == UnusedVertex) {
            CaveOldTetList.clear();
            return false;
        }
        // Stay put when the target is already within the resolvable distance.
        if (distance(P(mesh_vert), target) < MinEdgeLength) {
            CaveOldTetList.clear();
            return false;
        }

        double alpha = SmoothAlpha;
        const dvec3 dir = target - P(mesh_vert);
        // Per component, which is the form the stepped positions are compared in.
        dvec3 newpos;
        for (int j = 0; j < 3; ++j) newpos[j] = P(mesh_vert)[j] + alpha * dir[j];

        if (CaveOldTetList.empty()) getvertexstar(1, mesh_vert, &CaveOldTetList, nullptr, nullptr);

        // The vertex moves only when the full step keeps every tet of its star right side up.
        // Once a step is rejected the shortened steps are tried but can no longer be accepted.
        bool moveflag = true;
        for (int iter = 0; iter < 3; ++iter) {
            for (const auto &cavetet : CaveOldTetList) {
                if (ishulltet(cavetet)) continue;
                if (orient3d(P(org(cavetet)), P(dest(cavetet)), P(apex(cavetet)), newpos) >= 0) {
                    moveflag = false; // This tet would turn over.
                    break;
                }
            }
            if (moveflag) break;
            alpha /= 2.0;
            for (int j = 0; j < 3; ++j) newpos[j] = P(mesh_vert)[j] + alpha * dir[j];
        }

        if (moveflag) {
            Pts[mesh_vert].Pos = newpos;

            // Push every face of the star and of its link.
            Triface checkface, neightet;
            for (auto cavetet : CaveOldTetList) {
                if (ishulltet(cavetet)) continue;
                flippush(&cavetet);
                for (int j = 0; j < 3; ++j) {
                    esym(cavetet, checkface);
                    fsym(checkface, neightet);
                    if (!facemarked(neightet)) flippush(&checkface);
                    enextself(cavetet);
                }
            }

            if (BadTetsEnabled) {
                for (auto cavetet : CaveOldTetList) {
                    if (ishulltet(cavetet)) continue;
                    enqueuetetrahedron(&cavetet);
                }
            }

            FlipConstraints fc;
            fc.enqflag = 2;
            if (BadTetsEnabled) fc.chkencflag = 4;
            lawsonflip3d(&fc);
        }

        CaveOldTetList.clear();
        return moveflag;
    }

    // Laplacian smoothing of the Steiner points, on segments, in facets and inside the volume, for a fixed number of sweeps.
    void smooth_vertices() {
        std::vector<int> seg_smpt_list, surf_smpt_list, smpt_list;
        for (int p = 1; p < int(Pts.size()); ++p) {
            switch (pointtype(p)) {
                case FreeVolVertex: smpt_list.push_back(p); break;
                case FreeFacetVertex: surf_smpt_list.push_back(p); break;
                case FreeSegVertex: seg_smpt_list.push_back(p); break;
                default: break;
            }
        }
        if (long(smpt_list.size()) != StVolRefCount || long(surf_smpt_list.size()) != StFacRefCount ||
            long(seg_smpt_list.size()) != StSegRefCount) {
            bail(2);
        }

        std::vector<dvec3> seg_target_list(seg_smpt_list.size());
        std::vector<dvec3> surf_target_list(surf_smpt_list.size());
        std::vector<dvec3> target_list(smpt_list.size());

        for (int iter = 0; iter < SmoothMaxIter; ++iter) {
            int movedcount = 0;

            if (SmoothCriterion & 4) {
                for (size_t i = 0; i < seg_smpt_list.size(); ++i) get_seg_laplacian_center(seg_smpt_list[i], seg_target_list[i]);
                for (size_t i = 0; i < seg_smpt_list.size(); ++i) {
                    if (move_vertex(seg_smpt_list[i], seg_target_list[i])) {
                        if (int(LaterUnflipQueue.size()) > UnflipQueueLimit) recoverdelaunay();
                        ++movedcount;
                    }
                }
            }
            if (SmoothCriterion & 2) {
                for (size_t i = 0; i < surf_smpt_list.size(); ++i) get_surf_laplacian_center(surf_smpt_list[i], surf_target_list[i]);
                for (size_t i = 0; i < surf_smpt_list.size(); ++i) {
                    if (move_vertex(surf_smpt_list[i], surf_target_list[i])) {
                        if (int(LaterUnflipQueue.size()) > UnflipQueueLimit) recoverdelaunay();
                        ++movedcount;
                    }
                }
            }
            if (SmoothCriterion & 1) {
                for (size_t i = 0; i < smpt_list.size(); ++i) {
                    get_laplacian_center(smpt_list[i], target_list[i]);
                    CaveOldTetList.clear();
                }
                for (size_t i = 0; i < smpt_list.size(); ++i) {
                    if (move_vertex(smpt_list[i], target_list[i])) {
                        if (int(LaterUnflipQueue.size()) > UnflipQueueLimit) recoverdelaunay();
                        ++movedcount;
                    }
                }
            }

            if (movedcount == 0) break;
            if (!LaterUnflipQueue.empty()) recoverdelaunay();
        }
    }

    //=== Mesh improvement ===

    double CosSmtDihed{0}, CosSliDihed{0}, OptMaxSliverAspRatio{0};
    long OptFlipsCount{0}, OptSmoothCount{0};

    void clear_badtet_queue() {
        for (int i = 0; i < 64; ++i) BtQueueFront[i] = BtQueueTail[i] = None;
        BtFirstNonEmptyQ = -1;
        BtRecentQ = -1;
        BadTetStore.clear();
        DeadBadTets.clear();
        BadTetItems = 0;
    }

    // Push a bad tet into the bucket its quality key selects, 64 buckets from worst to best, each a first-in first-out list.
    void enqueue_badtet(const BadFace *bf) {
        int bt;
        if (!DeadBadTets.empty()) {
            bt = DeadBadTets.back();
            DeadBadTets.pop_back();
        } else {
            bt = int(BadTetStore.size());
            BadTetStore.emplace_back();
        }
        ++BadTetItems;
        BadTetStore[bt] = *bf;
        BadTetStore[bt].nextitem = None; // Marks the last item of a queue.

        const double qual = 1.0 / std::log(bf->key);
        int queuenumber = 0;
        if (qual < 1.0) {
            queuenumber = int(64.0 * (1.0 - qual));
            if (queuenumber > 63) queuenumber = 63;
        }

        if (BtQueueFront[queuenumber] == None) {
            if (queuenumber > BtFirstNonEmptyQ) {
                BtNextNonEmptyQ[queuenumber] = BtFirstNonEmptyQ;
                BtFirstNonEmptyQ = queuenumber;
            } else {
                int i = queuenumber + 1;
                while (BtQueueFront[i] == None) ++i;
                BtNextNonEmptyQ[queuenumber] = BtNextNonEmptyQ[i];
                BtNextNonEmptyQ[i] = queuenumber;
            }
            BtQueueFront[queuenumber] = bt;
        } else {
            BadTetStore[BtQueueTail[queuenumber]].nextitem = bt;
        }
        BtQueueTail[queuenumber] = bt;
    }

    // Queue a tet the improvement pass should look at, named by its four vertices so a later flip cannot lose it.
    void enqueue_if_bad(BadFace *bf) {
        if (bf->key > OptMaxAspRatio || bf->cent[0] < CosOptMaxDihedral) {
            bf->forg = org(bf->tt);
            bf->fdest = dest(bf->tt);
            bf->fapex = apex(bf->tt);
            bf->foppo = oppo(bf->tt);
            enqueue_badtet(bf);
        }
    }

    int top_badtet() {
        BtRecentQ = BtFirstNonEmptyQ;
        if (BtFirstNonEmptyQ < 0) return None;
        return BtQueueFront[BtFirstNonEmptyQ];
    }

    void dequeue_badtet() {
        if (BtRecentQ < 0) return;
        const int bt = BtQueueFront[BtRecentQ];
        BtQueueFront[BtRecentQ] = BadTetStore[bt].nextitem;
        if (bt == BtQueueTail[BtRecentQ]) {
            if (BtFirstNonEmptyQ == BtRecentQ) {
                BtFirstNonEmptyQ = BtNextNonEmptyQ[BtFirstNonEmptyQ];
            } else {
                int i = BtRecentQ + 1;
                while (BtQueueFront[i] == None) ++i;
                BtNextNonEmptyQ[i] = BtNextNonEmptyQ[BtRecentQ];
            }
        }
        DeadBadTets.push_back(bt);
        --BadTetItems;
    }

    // Split the fattest tet around a sliver's edge with its barycentre, then requeue whatever the insertion made worse.
    // The short edge a skinny tet holds, collapsed when it is shorter than the mesh can resolve.
    // The edge is found by matching the recorded minimum length, and one of its ends has to be a Steiner point to move.
    void shorten_skinny_edge(const BadFace *bf) {
        const double Lmin = bf->cent[3];
        Triface short_edge = bf->tt;
        int i = 0;
        for (; i < 6; ++i) {
            short_edge.ver = edge2ver[i];
            const double dd = distance(P(org(short_edge)), P(dest(short_edge)));
            if ((std::abs(Lmin - dd) / Lmin) < 1e-4) break;
        }
        if (i == 6) bail(2);

        if (Lmin <= MinEdgeLength) {
            const int e1 = org(short_edge), e2 = dest(short_edge);
            if (issteinerpoint(e1)) {
                if (!create_a_shorter_edge(e1, e2)) bail(2);
            } else if (issteinerpoint(e2)) {
                if (!create_a_shorter_edge(e2, e1)) bail(2);
            }
        }
    }

    bool add_steinerpt_to_repair(BadFace *bf, bool bSmooth) {
        const double cosmaxd = bf->cent[0];
        const double eta = bf->cent[2];
        const int lcount = bf->ss.shver; // The number of large dihedral angles.

        Triface splittet;
        splittet.tet = None;

        if (cosmaxd < CosSliDihed) {
            // A sliver (flat), which may also hold a short edge.
            char shape = 0;
            Triface sliver_edge;
            if (lcount == 2) {
                shape = 'S'; // A square. Remove the edge [a,b].
                sliver_edge = bf->tt;
            } else if (lcount == 3) {
                shape = 'T'; // A triangle. Remove the edge [c,d].
                edestoppo(bf->tt, sliver_edge);
            }

            if (shape == 'S') {
                double max_vol = 0.0;
                Triface check_sliver = sliver_edge;
                for (int i = 0; i < 2; ++i) {
                    bool is_bdry = false;
                    if (issubseg(check_sliver)) {
                        is_bdry = true;
                    } else {
                        Triface spintet = check_sliver;
                        do {
                            if (issubface(spintet)) {
                                is_bdry = true;
                                break;
                            }
                            fnextself(spintet);
                        } while (spintet.tet != check_sliver.tet);
                    }
                    if (!is_bdry) {
                        Triface spintet = check_sliver;
                        do {
                            const Tet &e = Tets[spintet.tet];
                            const double vol = orient3d(P(e.V[1]), P(e.V[0]), P(e.V[2]), P(e.V[3]));
                            if (vol > max_vol) {
                                max_vol = vol;
                                splittet = spintet;
                            }
                            fnextself(spintet);
                        } while (spintet.tet != check_sliver.tet);
                    }
                    edestoppoself(check_sliver); // The opposite edge.
                }
            }
        } else if (eta > OptMaxEdgeRatio) {
            shorten_skinny_edge(bf);
        }

        if (splittet.tet == None) return false;

        // Leave it alone when the tet to split is itself a bad one.
        BadFace tmpbf;
        if (get_tetqual(&splittet, None, &tmpbf)) {
            if (tmpbf.cent[0] < CosSliDihed) return false;
        } else {
            return false;
        }

        const int steinerpt = makepoint(FreeVolVertex);
        {
            const Tet &e = Tets[splittet.tet];
            Pts[steinerpt].Pos = (P(e.V[0]) + P(e.V[1]) + P(e.V[2]) + P(e.V[3])) / 4.0;
        }

        InsertFlags ivf{
            .iloc = Outside,
            .bowywat = 3,
            .lawson = 2,
            .splitbdflag = 0,
            .validflag = 1,
            .respectbdflag = 1,
            .rejflag = 0,
            .chkencflag = BadTetsEnabled ? 4 : 0,
            .sloc = 0,
            .sbowywat = 0,
            .smlenflag = 1, // Avoid creating a very short edge.
        };

        if (insertpoint(steinerpt, &splittet, nullptr, nullptr, &ivf)) {
            ++StVolRefCount;
            if (!FlipStack.empty()) {
                FlipConstraints fc;
                fc.enqflag = 2;
                if (BadTetsEnabled) fc.chkencflag = 4;
                lawsonflip3d(&fc);
            }
            if (int(LaterUnflipQueue.size()) > UnflipQueueLimit) LaterUnflipQueue.clear();
        } else {
            pointdealloc(steinerpt);
            return false;
        }

        if (bSmooth) {
            dvec3 ccent;
            get_laplacian_center(steinerpt, ccent);
            if (move_vertex(steinerpt, ccent)) ++OptSmoothCount;
        }

        if (!BadTets.empty()) {
            // Queue the tets the insertion made bad.
            BadFace nbf;
            for (auto &bface : BadTets) {
                if (isdeadtet(bface.tt)) continue;
                if (!marktest2ed(bface.tt)) continue; // Already processed.
                unmarktest2(bface.tt);
                if (ishulltet(bface.tt)) continue;
                get_tetqual(&bface.tt, None, &nbf);
                if (nbf.key > OptMaxAspRatio || nbf.cent[0] < CosOptMaxDihedral) {
                    nbf.forg = org(nbf.tt);
                    nbf.fdest = dest(nbf.tt);
                    nbf.fapex = apex(nbf.tt);
                    nbf.foppo = oppo(nbf.tt);
                    enqueue_badtet(&nbf);
                }
            }
            BadTets.clear();
        }

        // Has the bad tet gone?
        if (get_tet(bf->forg, bf->fdest, bf->fapex, bf->foppo, &bf->tt)) {
            if (repair_tet(bf, true, false, false)) return true;
        } else {
            return true; // Removed.
        }
        return false;
    }

    // Remove an edge of a bad tet by flips, accepting only a flip that improves the largest dihedral angle.
    bool flip_edge_to_improve(Triface *sliver_edge, double improved_cosmaxd) {
        if (issubseg(*sliver_edge)) return false;

        FlipConstraints fc;
        fc.noflip_in_surface = 1; // The input surface is fixed, so no flip may run inside it.
        fc.remove_large_angle = 1;
        fc.unflip = 1;
        fc.collectnewtets = 1;
        fc.checkflipeligibility = 1;
        fc.cosdihed_in = improved_cosmaxd;
        fc.cosdihed_out = 0.0; // 90 degrees.
        fc.max_asp_out = 0.0;

        if (removeedgebyflips(sliver_edge, &fc) == 2) {
            if (fc.cosdihed_out < CosOptMaxDihedral || fc.max_asp_out > OptMaxAspRatio) {
                // Queue the new bad tets for further improvement.
                BadFace bf;
                for (auto &parytet : CaveTetList) {
                    if (isdeadtet(parytet) || ishulltet(parytet)) continue;
                    if (!get_tetqual(&parytet, None, &bf)) bail(2);
                    enqueue_if_bad(&bf);
                }
            }
            CaveTetList.clear();
            return true;
        }
        return false;
    }

    // Try to remove one bad tet, by flipping the edge its shape names, and failing that by splitting it with a Steiner point.
    bool repair_tet(BadFace *bf, bool bFlips, bool bSmooth, bool bSteiners) {
        const double cosmaxd = bf->cent[0];
        const double eta = bf->cent[2];
        const int lcount = bf->ss.shver; // The number of large dihedral angles.

        if (cosmaxd < CosSmtDihed) {
            // A sliver (flat), which may also hold a short edge.
            char shape = '0';
            if (lcount == 2) shape = 'S'; // A square. Remove the edge [a,b].
            else if (lcount == 3) shape = 'T'; // A triangle. Remove the edge [c,d].

            if (bFlips) {
                if (shape == 'S') {
                    Triface sliver_edge = bf->tt;
                    if (flip_edge_to_improve(&sliver_edge, cosmaxd)) {
                        ++OptFlipsCount;
                        return true;
                    }
                    // An unflipped flip may have moved the sliver, so find it again.
                    if (get_tet(bf->forg, bf->fdest, bf->fapex, bf->foppo, &bf->tt)) {
                        edestoppo(bf->tt, sliver_edge); // Face [c,d,a].
                        if (flip_edge_to_improve(&sliver_edge, cosmaxd)) {
                            ++OptFlipsCount;
                            return true;
                        }
                    }
                } else if (shape == 'T') {
                    Triface sliver_edge;
                    edestoppo(bf->tt, sliver_edge); // Face [c,d,a].
                    if (flip_edge_to_improve(&sliver_edge, cosmaxd)) {
                        ++OptFlipsCount;
                        return true;
                    }
                }
            }
        } else if (eta > OptMaxEdgeRatio) {
            shorten_skinny_edge(bf);
        }

        if (bSteiners && (bf->key > OptMaxSliverAspRatio || cosmaxd < CosSliDihed)) {
            // The sliver is still there, and an unflipped flip may have moved it.
            if (get_tet(bf->forg, bf->fdest, bf->fapex, bf->foppo, &bf->tt)) {
                if (add_steinerpt_to_repair(bf, bSmooth)) return true;
            }
        }
        return false;
    }

    // Drain the priority queue, requeueing what could not be repaired so a later pass at a higher flip level sees it again.
    long repair_badqual_tets(bool bFlips, bool bSmooth, bool bSteiners) {
        long repaired_count = 0;

        while (BadTetItems > 0) {
            const int bt = top_badtet();
            BadFace work = BadTetStore[bt];
            if (get_tet(work.forg, work.fdest, work.fapex, work.foppo, &work.tt)) {
                if (repair_tet(&work, bFlips, bSmooth, bSteiners)) {
                    ++repaired_count;
                } else {
                    UnsplitBadTets.push_back(work);
                }
            }
            dequeue_badtet();
        }

        if (!UnsplitBadTets.empty()) {
            for (int i = 0; i < 64; ++i) BtQueueFront[i] = BtQueueTail[i] = None;
            BtFirstNonEmptyQ = -1;
            BtRecentQ = -1;
            std::vector<BadFace> pending;
            pending.swap(UnsplitBadTets);
            for (const auto &bt : pending) enqueue_badtet(&bt);
        }
        return repaired_count;
    }

    // Repair every tet whose aspect ratio or largest dihedral angle is out of range.
    // First by flips alone at climbing flip level, then by flips, smoothing and Steiner points together, then by flips once more.
    void improve_mesh() {
        clear_badtet_queue();
        BadTetsEnabled = true;
        BadTets.clear();
        UnsplitBadTets.clear();

        CosLargeDihed = std::cos(135.0 / 180.0 * PI); // Read by get_tetqual.
        CosOptMaxDihedral = std::cos(OptMaxDihedral / 180.0 * PI);

        // The smallest dihedral angle that names a sliver.
        const double sliver_ang_tol = std::max(OptMaxDihedral - 5.0, 172.0);
        CosSmtDihed = std::cos(sliver_ang_tol / 180.0 * PI);

        // The smallest dihedral angle that gets a sliver split.
        double split_sliver_ang_tol = OptMaxDihedral + 10.0;
        if (split_sliver_ang_tol < 179.0) split_sliver_ang_tol = 179.0;
        else if (split_sliver_ang_tol > 180.0) split_sliver_ang_tol = 179.9;
        CosSliDihed = std::cos(split_sliver_ang_tol / 180.0 * PI);

        OptMaxSliverAspRatio = OptMaxAspRatio * 10.0;

        // Queue every bad tet.
        {
            Triface checktet;
            BadFace bf;
            for (int t = 0; t < int(Tets.size()); ++t) {
                if (Tets[t].V[0] == None || Tets[t].V[3] == DummyPoint) continue;
                checktet = {t, 11};
                if (!get_tetqual(&checktet, None, &bf)) bail(2); // A degenerate tet.
                enqueue_if_bad(&bf);
            }
        }

        const int bak_autofliplinklevel = AutoFlipLinkLevel;
        const int bak_fliplinklevel = FlipLinkLevel;
        const int bak_maxflipstarsize = FlipStarSize;
        FlipLinkLevel = 1;
        FlipStarSize = 10;

        // Flips alone.
        while (BadTetItems > 0) {
            repair_badqual_tets(true, false, false);
            if (FlipLinkLevel < OptMaxFlipLevel) ++FlipLinkLevel;
            else break;
        }

        long bak_st_count = StVolRefCount;
        for (int iter = 0; iter < OptIterations && BadTetItems > 0; ++iter) {
            const long repaired_count = repair_badqual_tets(true, true, true);
            // Stop when nothing was repaired and no Steiner point was added.
            if (repaired_count == 0 && bak_st_count == StVolRefCount) break;
            bak_st_count = StVolRefCount;
        }

        if (BadTetItems > 0) repair_badqual_tets(true, false, false);

        if (int(LaterUnflipQueue.size()) > UnflipQueueLimit) LaterUnflipQueue.clear();

        AutoFlipLinkLevel = bak_autofliplinklevel;
        FlipLinkLevel = bak_fliplinklevel;
        FlipStarSize = bak_maxflipstarsize;

        BadTetsEnabled = false;
        BadTets.clear();
        clear_badtet_queue();
    }

    //=== Delaunay refinement, reached only when quality is asked for ===

    std::vector<Triface> CheckTetsList;
    double SmallestInsRadius{1.0e+30};

    // Is this tet badly shaped?
    // Fills param with the circumcentre, shortest edge length, radius-edge ratio and edge ratio.
    // The handle turns to name the shortest edge.
    bool checktet4split(Triface *chktet, double *param, int &qflag) {
        qflag = 0;
        for (int i = 0; i < 6; ++i) param[i] = 0.0;

        const Tet &e = Tets[chktet->tet];
        const int pd = e.V[3];
        if (pd == DummyPoint) return false; // Never split a hull tet.
        const int pa = e.V[0], pb = e.V[1], pc = e.V[2];

        // The matrix A = [d->a, d->b, d->c]^T.
        const dvec3 vda = P(pa) - P(pd), vdb = P(pb) - P(pd), vdc = P(pc) - P(pd);
        const dvec3 vab = P(pb) - P(pa), vbc = P(pc) - P(pb), vca = P(pa) - P(pc);
        double A[4][4]{}, rhs[4]{}, D;
        int indx[4]{};
        set_rows(A, vda, vdb, vdc);

        if (!lu_decmp(A, 3, indx, &D, 0)) {
            if (orient3d(P(pa), P(pb), P(pc), P(pd)) >= 0.0) bail(2); // A degenerate tet.
            // Leave it to mesh improvement.
            return false;
        }

        // The circumcentre and circumradius.
        rhs[0] = 0.5 * dot(vda, vda);
        rhs[1] = 0.5 * dot(vdb, vdb);
        rhs[2] = 0.5 * dot(vdc, vdc);
        lu_solve(A, 3, indx, rhs, 0);
        for (int i = 0; i < 3; ++i) param[i] = P(pd)[i] + rhs[i];
        const double rd = std::sqrt(rhs[0] * rhs[0] + rhs[1] * rhs[1] + rhs[2] * rhs[2]);

        // The volume bound.
        // The factorisation above already carries the determinant, whose pivots multiply to six times the volume.
        if (MaxVolume > 0) {
            const double vol = std::abs(A[indx[0]][0] * A[indx[1]][1] * A[indx[2]][2]) / 6.0;
            if (vol > MaxVolume) {
                qflag = 1;
                return true;
            }
        }

        // The radius-edge ratio.
        if (MinRatio > 0) {
            double elen[6];
            elen[0] = dot(vdc, vdc);
            elen[1] = dot(vda, vda);
            elen[2] = dot(vab, vab);
            elen[3] = dot(vbc, vbc);
            elen[4] = dot(vdb, vdb);
            elen[5] = dot(vca, vca);

            double Lmax = elen[0], Lmin = elen[0];
            int eidx = 0;
            for (int i = 1; i < 6; ++i) {
                Lmax = std::max(Lmax, elen[i]);
                if (Lmin > elen[i]) {
                    Lmin = elen[i];
                    eidx = i;
                }
            }
            chktet->ver = edge2ver[eidx];
            Lmin = std::sqrt(Lmin);
            const double ratio = rd / Lmin;
            if (ratio > MinRatio) {
                param[3] = Lmin;
                param[4] = ratio;
                param[5] = std::sqrt(Lmax) / Lmin;
                return true;
            }
        }

        // The smallest dihedral angle.
        if (MinDihedral > 0) {
            dvec3 N[4];
            double L[4];
            face_normals(A, indx, N);
            for (int i = 0; i < 4; ++i) {
                L[i] = std::sqrt(dot(N[i], N[i]));
                if (L[i] == 0) bail(2);
                N[i] /= L[i];
            }
            // Every pair of faces, so every one of the six edges: cd, bd, bc, then ad, ac, then ab.
            const double cosd[6]{
                -dot(N[0], N[1]), -dot(N[0], N[2]), -dot(N[0], N[3]),
                -dot(N[1], N[2]), -dot(N[1], N[3]), -dot(N[2], N[3])
            };
            if (std::ranges::max(cosd) > CosMinDihedral) return true; // A bad dihedral angle.
        }

        return false;
    }

    // Walk from the centre of searchtet toward searchpt, stopping at the first subface crossed or when the walk leaves the mesh.
    int locate_point_walk(int searchpt, Triface *searchtet, int chkencflag) {
        const dvec3 startpt = [&] {
            const Tet &e = Tets[searchtet->tet];
            return (P(e.V[0]) + P(e.V[1]) + P(e.V[2]) + P(e.V[3])) / 4.0;
        }();

        int loc = Outside;
        WalkMove nextmove = OrgMove;

        face_toward(searchpt, searchtet);
        int torg = org(*searchtet), tdest = dest(*searchtet), tapex = apex(*searchtet);

        int max_visited_tets = 10000;
        while (max_visited_tets > 0) {
            const int toppo = oppo(*searchtet);
            if (toppo == searchpt) {
                esymself(*searchtet);
                eprevself(*searchtet);
                loc = OnVertex;
                break;
            }

            // Which face does the line (startpt -> searchpt) leave through?
            const double oriorg = orient3d(P(tdest), P(tapex), P(toppo), P(searchpt));
            const double oridest = orient3d(P(tapex), P(torg), P(toppo), P(searchpt));
            const double oriapex = orient3d(P(torg), P(tdest), P(toppo), P(searchpt));

            if (oriorg < 0) {
                if (oridest < 0) {
                    if (oriapex < 0) {
                        // Any of the three faces is possible.
                        if (tri_edge_test(P(tdest), P(tapex), P(toppo), startpt, P(searchpt), nullptr, 0, nullptr, nullptr)) {
                            nextmove = OrgMove;
                        } else if (tri_edge_test(P(tapex), P(torg), P(toppo), startpt, P(searchpt), nullptr, 0, nullptr, nullptr)) {
                            nextmove = DestMove;
                        } else if (tri_edge_test(P(torg), P(tdest), P(toppo), startpt, P(searchpt), nullptr, 0, nullptr, nullptr)) {
                            nextmove = ApexMove;
                        } else {
                            const unsigned s = randomnation(3);
                            nextmove = s == 0 ? OrgMove : (s == 1 ? DestMove : ApexMove);
                        }
                    } else {
                        // The faces opposite the origin and the destination are possible.
                        if (tri_edge_test(P(tdest), P(tapex), P(toppo), startpt, P(searchpt), nullptr, 0, nullptr, nullptr)) {
                            nextmove = OrgMove;
                        } else if (tri_edge_test(P(tapex), P(torg), P(toppo), startpt, P(searchpt), nullptr, 0, nullptr, nullptr)) {
                            nextmove = DestMove;
                        } else {
                            nextmove = randomnation(2) ? OrgMove : DestMove;
                        }
                    }
                } else {
                    if (oriapex < 0) {
                        // The faces opposite the origin and the apex are possible.
                        if (tri_edge_test(P(tdest), P(tapex), P(toppo), startpt, P(searchpt), nullptr, 0, nullptr, nullptr)) {
                            nextmove = OrgMove;
                        } else if (tri_edge_test(P(torg), P(tdest), P(toppo), startpt, P(searchpt), nullptr, 0, nullptr, nullptr)) {
                            nextmove = ApexMove;
                        } else {
                            nextmove = randomnation(2) ? OrgMove : ApexMove;
                        }
                    } else {
                        nextmove = OrgMove; // Only the face opposite the origin.
                    }
                }
            } else {
                if (oridest < 0) {
                    if (oriapex < 0) {
                        // The faces opposite the destination and the apex are possible.
                        if (tri_edge_test(P(tapex), P(torg), P(toppo), startpt, P(searchpt), nullptr, 0, nullptr, nullptr)) {
                            nextmove = DestMove;
                        } else if (tri_edge_test(P(torg), P(tdest), P(toppo), startpt, P(searchpt), nullptr, 0, nullptr, nullptr)) {
                            nextmove = ApexMove;
                        } else {
                            nextmove = randomnation(2) ? DestMove : ApexMove;
                        }
                    } else {
                        nextmove = DestMove; // Only the face opposite the destination.
                    }
                } else if (oriapex < 0) {
                    nextmove = ApexMove; // Only the face opposite the apex.
                } else {
                    // The point is on the boundary of, or inside, this tet.
                    loc = locate_feature(searchtet, oriorg, oridest, oriapex);
                    break;
                }
            }

            if (const int stop = step_across(searchtet, nextmove, chkencflag); stop != LocUnknown) {
                loc = stop;
                break;
            }
            --max_visited_tets;

            torg = org(*searchtet);
            tdest = dest(*searchtet);
            tapex = apex(*searchtet);
        }

        return loc;
    }

    // Put a Steiner point at the circumcentre of a bad tet.
    // It is rejected when it is not visible from inside the tet or encroaches on the boundary.
    // The boundary is never split, so a rejected point is dropped.
    bool split_tetrahedron(Triface *splittet, double *param, int qflag, int chkencflag, InsertFlags &ivf) {
        const int newpt = makepoint(FreeVolVertex);
        Pts[newpt].Pos = dvec3{param[0], param[1], param[2]};

        // Walk from inside splittet to the new point, stopping at a subface or outside.
        Triface searchtet = *splittet;
        ivf.iloc = locate_point_walk(newpt, &searchtet, 1);

        if (ivf.iloc == EncSubface || ivf.iloc == Outside) {
            // The circumcentre is not visible from the inside of the tet.
            pointdealloc(newpt);
            ivf.iloc = FencedIn;
            return false;
        }

        ivf.bowywat = 3; // Bowyer-Watson, preserving subsegments and subfaces.
        ivf.lawson = 2;
        ivf.rejflag = 3; // Check for encroached segments and subfaces.
        ivf.chkencflag = chkencflag & ~3;
        ivf.sloc = ivf.sbowywat = 0;
        ivf.splitbdflag = 0;
        ivf.validflag = 1;
        ivf.respectbdflag = 1;
        ivf.refineflag = 1;
        ivf.refinetet = *splittet;
        ivf.smlenflag = UseInsertRadius;
        ivf.check_insert_radius = qflag ? 0 : UseInsertRadius;
        ivf.parentpt = None;

        if (insertpoint(newpt, &searchtet, nullptr, nullptr, &ivf)) {
            ++StVolRefCount;
            if (SteinerLeft > 0) --SteinerLeft;
            if (UseInsertRadius) {
                // The shorter of this tet's smallest edge and the distance to the nearest vertex.
                const double rv = param[3] > 0.0 ? std::min(param[3], ivf.smlen) : 0.0;
                setpointinsradius(newpt, rv);
                setpoint2ppt(newpt, ivf.parentpt);
                SmallestInsRadius = std::min(SmallestInsRadius, ivf.smlen);
            }
            if (!FlipStack.empty()) {
                FlipConstraints fc{.enqflag = 2, .chkencflag = chkencflag & ~3};
                lawsonflip3d(&fc);
            }
            if (int(LaterUnflipQueue.size()) > UnflipQueueLimit) recoverdelaunay();
            return true;
        }

        // The point is not inserted.
        // The encroached boundary is left alone, since it is never split.
        pointdealloc(newpt);
        if (ivf.iloc == EncSegment) EncSegList.clear();
        else if (ivf.iloc == EncSubface) EncShList.clear();
        return false;
    }

    // Drain the queue of tets to check, splitting the badly shaped ones.
    // The ones that resisted are kept for the caller's next round.
    void repairbadtets(double queratio, int chkencflag) {
        double param[6]{};
        int qflag = 0;

        while (!BadTets.empty() || !CheckTetsList.empty()) {
            if (!BadTets.empty()) {
                for (const auto &bface : BadTets) CheckTetsList.push_back(bface.tt);
                BadTets.clear();
            }
            if (SteinerLeft == 0) break;

            // Take a tet at random and fill its place with the last one.
            const size_t i = Rng() % CheckTetsList.size();
            const Triface checktet = CheckTetsList[i];
            CheckTetsList[i] = CheckTetsList.back();
            CheckTetsList.pop_back();

            if (isdeadtet(checktet) || !marktest2ed(checktet)) continue;
            unmarktest2(checktet);
            Triface worktet = checktet;
            if (!checktet4split(&worktet, param, qflag)) continue;

            InsertFlags ivf;
            if (!split_tetrahedron(&worktet, param, qflag, chkencflag, ivf)) {
                if (qflag || param[4] > queratio) {
                    BadFace bt{
                        .tt = worktet,
                        .key = double(qflag),
                        .forg = org(worktet),
                        .fdest = dest(worktet),
                        .fapex = apex(worktet),
                        .foppo = oppo(worktet),
                    };
                    for (int j = 0; j < 6; ++j) bt.cent[j] = param[j];
                    UnsplitBadTets.push_back(bt);
                }
            }
        }

        for (const auto &quetet : CheckTetsList) {
            if (!isdeadtet(quetet)) unmarktest2(quetet);
        }
        CheckTetsList.clear();
    }

    // The boundary is fixed, so refinement only adds interior points.
    // Bad-quality tets are split until none is left that can be.
    void delaunayrefinement() {
        CosFacetSepAngTol = std::cos(FacetSepAngTol / 180.0 * PI);
        CosCollinearAngTol = std::cos(CollinearAngTol / 180.0 * PI);
        CosMinDihedral = std::cos(MinDihedral / 180.0 * PI);

        BadTetsEnabled = true;
        BadTets.clear();
        CheckTetsList.clear();
        UnsplitBadTets.clear();

        for (int t = 0; t < int(Tets.size()); ++t) {
            if (Tets[t].V[0] == None || Tets[t].V[3] == DummyPoint) continue;
            const Triface checktet{t, 11};
            marktest2(checktet);
            CheckTetsList.push_back(checktet);
        }

        const int chkencflag = 4; // Check bad tetrahedra.
        const double queratio = (MinRatio > 2.0 ? MinRatio : 2.0) * 2.0;
        double param[6]{};

        for (int iter = 0; iter < 3; ++iter) {
            repairbadtets(queratio, chkencflag);
            if (!LaterUnflipQueue.empty()) recoverdelaunay();
            if (UnsplitBadTets.empty()) break;

            // Split the tets that resisted, at their barycentre.
            long splitcount = 0;
            std::vector<BadFace> pending;
            pending.swap(UnsplitBadTets);
            for (auto &bt : pending) {
                if (bt.tt.tet == None || isdeadtet(bt.tt)) continue;
                if (org(bt.tt) != bt.forg || dest(bt.tt) != bt.fdest || apex(bt.tt) != bt.fapex || oppo(bt.tt) != bt.foppo) continue;
                if (SteinerLeft == 0) break;

                InsertFlags ivf;
                const int qflag = int(bt.key);
                const Tet &e = Tets[bt.tt.tet];
                const dvec3 bary = (P(e.V[0]) + P(e.V[1]) + P(e.V[2]) + P(e.V[3])) / 4.0;
                param[0] = bary.x;
                param[1] = bary.y;
                param[2] = bary.z;
                for (int j = 3; j < 6; ++j) param[j] = bt.cent[j];
                if (split_tetrahedron(&bt.tt, param, qflag, chkencflag, ivf)) ++splitcount;

                if (!BadTets.empty()) {
                    for (const auto &bface : BadTets) CheckTetsList.push_back(bface.tt);
                    BadTets.clear();
                }
            }
            if (splitcount == 0) break;
        }

        for (const auto &quetet : CheckTetsList) {
            if (!isdeadtet(quetet)) unmarktest2(quetet);
        }
        CheckTetsList.clear();
        UnsplitBadTets.clear();
        BadTets.clear();
        BadTetsEnabled = false;
    }

    //=== Input, the driver and output ===

    // Copy the input points into the point pool, measure the bounding box and derive every length and angle tolerance from it.
    void transfernodes(std::span<const dvec3> points) {
        NumInputPoints = int(points.size());
        Pts.clear();
        // The input plus room for the Steiner points recovery and refinement add.
        // The pool is laid down once rather than copied forward as it grows.
        Pts.reserve(points.size() + points.size() / 4 + 16);
        makepoint(UnusedVertex);
        for (const dvec3 &pos : points) {
            const int p = makepoint(UnusedVertex);
            Pts[p].Pos = pos;
        }

        for (int i = 0; i < NumInputPoints; ++i) {
            const dvec3 &q = points[size_t(i)];
            if (i == 0) {
                BoxMin = BoxMax = q;
            } else {
                BoxMin = glm::min(BoxMin, q);
                BoxMax = glm::max(BoxMax, q);
            }
        }

        const double dx = BoxMax.x - BoxMin.x, dy = BoxMax.y - BoxMin.y, dz = BoxMax.z - BoxMin.z;
        // The largest edge two input vertices can span.
        LongEst = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (LongEst == 0.0) bail(10); // A trivial point set.
        // Two points closer than this are the same point.
        MinEdgeLength = LongEst * Epsilon;

        Rng.seed(unsigned(NumInputPoints));

        // A piecewise-linear input keeps the insertion radius of every Steiner point.
        UseInsertRadius = 1;

        CosMinDihedral = std::cos(MinDihedral / 180.0 * PI);
        CosOptMaxDihedral = std::cos(OptMaxDihedral / 180.0 * PI);
        CosLargeDihed = std::cos(135.0 / 180.0 * PI);
        CosCollinearAngTol = std::cos(CollinearAngTol / 180.0 * PI);
        CosFacetSepAngTol = std::cos(FacetSepAngTol / 180.0 * PI);
        CosSmallAngTol = std::cos(FacetSmallAngTol / 180.0 * PI);
    }

    // The whole run path, in order, once.
    void build(std::span<const dvec3> points, std::span<const uint32_t> triangle_indices, tetra::Profile &prof) {
        using clock = std::chrono::steady_clock;
        const auto secs = [](clock::time_point a, clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };

        InTris.resize(triangle_indices.size() / 3);
        for (size_t i = 0; i < InTris.size(); ++i) {
            InTris[i] = {int(triangle_indices[i * 3]) + 1, int(triangle_indices[i * 3 + 1]) + 1, int(triangle_indices[i * 3 + 2]) + 1};
        }

        transfernodes(points);
        // A Delaunay triangulation of n points runs to about 7n tets.
        // The surface contributes one subface per input triangle, plus its subsegments.
        // Both pools are laid down at that size so a run does not spend its time copying them forward.
        Tets.reserve(points.size() * 7 + 32);
        Shells.reserve(InTris.size() * 2 + 32);

        const auto t0 = clock::now();
        if (!incrementaldelaunay()) bail(10);
        const auto t1 = clock::now();
        for (const auto &e : Tets) {
            if (e.V[0] != None && e.V[3] != DummyPoint) ++prof.DelaunayTetCount;
        }

        meshsurface();
        const auto t2 = clock::now();

        recoverboundary();
        if (SkippedFacet) bail(3);
        const auto t3 = clock::now();

        carveholes();
        const auto t4 = clock::now();

        if (SupSteinerLevel > 0 && !SubVertStack.empty()) suppresssteinerpoints();
        const auto t5 = clock::now();

        recoverdelaunay();

        if (Quality) delaunayrefinement();

        if (SmoothMaxIter > 0 && (StVolRefCount > 0 || StFacRefCount > 0)) smooth_vertices();
        improve_mesh();
        const auto t7 = clock::now();

        prof.DelaunaySeconds = secs(t0, t1);
        prof.RecoverSeconds = secs(t2, t3);
        prof.SegmentSeconds = secs(t2, t3);
        prof.CarveSeconds = secs(t3, t4);
        prof.SuppressSeconds = secs(t4, t5);
        prof.FaceSeconds = secs(t1, t2);
        prof.RefineSeconds = secs(t5, t7);
        prof.Builds = 1;
        prof.FlipCount = u32(Flip23Count + Flip32Count + Flip44Count + Flip41Count + Flip22Count + Flip31Count);
        prof.MissingEdgeCount = u32(MissingEdgeCount);
        prof.MissingFaceCount = u32(MissingFaceCount);
        prof.BdrySteinerCount = u32(StSegRefCount + StFacRefCount);
        prof.VolSteinerCount = u32(StVolRefCount);
        prof.SplitCount = u32(StSegRefCount + StFacRefCount + StVolRefCount);
    }

    // The live interior tets, with input point i mapped back to i - 1 and the Steiner points that survived appended after them.
    // A tet is stored with Orient3D < 0, so one pair of vertices is swapped to meet TetMesh's positive convention.
    TetMesh out(tetra::Profile &prof) const {
        TetMesh mesh;
        mesh.Points.reserve(size_t(NumInputPoints));
        std::vector<int> vert_out(Pts.size(), None);
        for (int i = 0; i < NumInputPoints; ++i) {
            vert_out[size_t(i) + 1] = i;
            mesh.Points.push_back(Pts[size_t(i) + 1].Pos);
        }

        for (int t = 0; t < int(Tets.size()); ++t) {
            const Tet &e = Tets[t];
            if (e.V[0] == None || e.V[3] == DummyPoint) continue;
            std::array<uint32_t, 4> tv{};
            for (int c = 0; c < 4; ++c) {
                const int v = e.V[c];
                if (vert_out[size_t(v)] == None) {
                    vert_out[size_t(v)] = int(mesh.Points.size());
                    mesh.Points.push_back(Pts[size_t(v)].Pos);
                }
                tv[size_t(c)] = uint32_t(vert_out[size_t(v)]);
            }
            std::swap(tv[0], tv[1]);
            mesh.Tets.push_back(tv);
        }

        prof.TetCount = u32(mesh.Tets.size());
        prof.SteinerCount = u32(mesh.Points.size() - size_t(NumInputPoints));
        return mesh;
    }
};

} // namespace

std::expected<Result, std::string> Tetrahedralize(std::span<const dvec3> points, std::span<const uint32_t> triangle_indices, Options options) {
    Mesh m;
    m.MaxVolume = options.MaxVolume;
    // A volume bound turns the quality pass on, so the radius-edge and dihedral targets apply alongside it.
    m.Quality = options.Quality || options.MaxVolume > 0;
    // Without a quality bound the dihedral target relaxes.
    if (!m.Quality) m.OptMaxDihedral = 179.9;

    Profile prof;
    try {
        m.build(points, triangle_indices, prof);
    } catch (const TetError &e) {
        switch (e.Code) {
            case 3: return std::unexpected("the input surface intersects itself");
            case 4: return std::unexpected("an input feature is smaller than the resolvable size");
            case 10: return std::unexpected("the input points are degenerate");
            default: return std::unexpected("the tetrahedralizer failed on this surface");
        }
    }
    return Result{m.out(prof), prof};
}
} // namespace tetra
