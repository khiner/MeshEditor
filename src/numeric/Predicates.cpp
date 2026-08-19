#include "numeric/Predicates.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

// Exact evaluation uses floating-point expansions: nonoverlapping doubles in increasing magnitude, so the last component carries the sign.
// Filter bounds are conservative multiples of the machine epsilon, so the expansion path runs only near degeneracy.
namespace {
constexpr double Eps = 0x1p-53;

struct Two {
    double Hi, Lo; // Hi + Lo exactly, |Lo| <= ulp(Hi)/2
};

Two TwoSum(double a, double b) {
    const double s = a + b, bv = s - a, av = s - bv;
    return {s, (a - av) + (b - bv)};
}
Two TwoDiff(double a, double b) { return TwoSum(a, -b); }
Two TwoProd(double a, double b) {
    const double p = a * b;
    return {p, std::fma(a, b, -p)};
}
// Requires |a| >= |b| or a == 0.
Two FastTwoSum(double a, double b) {
    const double s = a + b;
    return {s, b - (s - a)};
}

// h = e + f, all nonoverlapping increasing-magnitude expansions with zeros eliminated. h needs capacity elen + flen.
// Returns the count written (>= 1, h = {0} for a zero sum).
int ExpSum(int elen, const double *e, int flen, const double *f, double *h) {
    double enow = e[0], fnow = f[0], Q, hh;
    int ei = 0, fi = 0, hn = 0;
    const auto take_e = [&] { return (fnow > enow) == (fnow > -enow); }; // |enow| <= |fnow|
    if (take_e()) {
        Q = enow;
        if (++ei < elen) enow = e[ei];
    } else {
        Q = fnow;
        if (++fi < flen) fnow = f[fi];
    }
    if (ei < elen && fi < flen) {
        Two first;
        if (take_e()) {
            first = FastTwoSum(enow, Q);
            if (++ei < elen) enow = e[ei];
        } else {
            first = FastTwoSum(fnow, Q);
            if (++fi < flen) fnow = f[fi];
        }
        Q = first.Hi;
        hh = first.Lo;
        if (hh != 0) h[hn++] = hh;
        while (ei < elen && fi < flen) {
            Two s;
            if (take_e()) {
                s = TwoSum(Q, enow);
                if (++ei < elen) enow = e[ei];
            } else {
                s = TwoSum(Q, fnow);
                if (++fi < flen) fnow = f[fi];
            }
            Q = s.Hi;
            if (s.Lo != 0) h[hn++] = s.Lo;
        }
    }
    while (ei < elen) {
        const auto s = TwoSum(Q, enow);
        if (++ei < elen) enow = e[ei];
        Q = s.Hi;
        if (s.Lo != 0) h[hn++] = s.Lo;
    }
    while (fi < flen) {
        const auto s = TwoSum(Q, fnow);
        if (++fi < flen) fnow = f[fi];
        Q = s.Hi;
        if (s.Lo != 0) h[hn++] = s.Lo;
    }
    if (Q != 0 || hn == 0) h[hn++] = Q;
    return hn;
}

// h = e * b. h needs capacity 2 * elen.
int ExpScale(int elen, const double *e, double b, double *h) {
    int hn = 0;
    auto [Q, hh] = TwoProd(e[0], b);
    if (hh != 0) h[hn++] = hh;
    for (int i = 1; i < elen; ++i) {
        const auto p = TwoProd(e[i], b);
        const auto s = TwoSum(Q, p.Lo);
        if (s.Lo != 0) h[hn++] = s.Lo;
        const auto q = FastTwoSum(p.Hi, s.Hi);
        if (q.Lo != 0) h[hn++] = q.Lo;
        Q = q.Hi;
    }
    if (Q != 0 || hn == 0) h[hn++] = Q;
    return hn;
}

void ExpNegate(int n, double *e) {
    for (int i = 0; i < n; ++i) e[i] = -e[i];
}

// The expansion's value rounded to one double, by summing the components as they are stored.
double Estimate(int n, const double *e) {
    double q = e[0];
    for (int i = 1; i < n; ++i) q += e[i];
    return q;
}

// out = u * v for two-component expansions (capacity 8).
int MulTwoTwo(Two u, Two v, double *out) {
    const double ue[2]{u.Lo, u.Hi};
    double t1[4], t2[4];
    const int n1 = ExpScale(2, ue, v.Lo, t1);
    const int n2 = ExpScale(2, ue, v.Hi, t2);
    return ExpSum(n1, t1, n2, t2, out);
}

// out = e * (v.Hi + v.Lo) for an elen-component expansion (capacity 4 * elen).
int MulExpTwo(int elen, const double *e, Two v, double *out, double *scratch) {
    double *t1 = scratch, *t2 = scratch + 2 * elen;
    const int n1 = ExpScale(elen, e, v.Lo, t1);
    const int n2 = ExpScale(elen, e, v.Hi, t2);
    return ExpSum(n1, t1, n2, t2, out);
}

// Exact det[a-d, b-d, c-d] via exact coordinate differences.
// Result capacity 192.
double Orient3DExact(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    const Two adx = TwoDiff(a.x, d.x), ady = TwoDiff(a.y, d.y), adz = TwoDiff(a.z, d.z);
    const Two bdx = TwoDiff(b.x, d.x), bdy = TwoDiff(b.y, d.y), bdz = TwoDiff(b.z, d.z);
    const Two cdx = TwoDiff(c.x, d.x), cdy = TwoDiff(c.y, d.y), cdz = TwoDiff(c.z, d.z);

    // Coordinates from quantized data usually subtract exactly.
    // The determinant of the nine plain differences then needs only a 24-component pipeline.
    if (adx.Lo == 0 && ady.Lo == 0 && adz.Lo == 0 && bdx.Lo == 0 && bdy.Lo == 0 && bdz.Lo == 0 && cdx.Lo == 0 && cdy.Lo == 0 && cdz.Lo == 0) {
        const auto cross = [](double u, double v, double w, double x, double *out4) {
            const auto m1 = TwoProd(u, v);
            const auto m2 = TwoProd(w, x);
            const double e1[2]{m1.Lo, m1.Hi}, e2[2]{-m2.Lo, -m2.Hi};
            return ExpSum(2, e1, 2, e2, out4);
        };
        double c1[4], c2[4], c3[4], t1[8], t2[8], t3[8], t12[16], det[24];
        const int m1 = cross(bdy.Hi, cdz.Hi, bdz.Hi, cdy.Hi, c1);
        const int m2 = cross(bdz.Hi, cdx.Hi, bdx.Hi, cdz.Hi, c2);
        const int m3 = cross(bdx.Hi, cdy.Hi, bdy.Hi, cdx.Hi, c3);
        const int n1 = ExpScale(m1, c1, adx.Hi, t1);
        const int n2 = ExpScale(m2, c2, ady.Hi, t2);
        const int n3 = ExpScale(m3, c3, adz.Hi, t3);
        const int n12 = ExpSum(n1, t1, n2, t2, t12);
        const int n = ExpSum(n12, t12, n3, t3, det);
        return det[n - 1];
    }

    // term = s * (u * v - w * x), capacity 64.
    double det[192], scratch[64];
    const auto cross_term = [&scratch](Two s, Two u, Two v, Two w, Two x, double *out) {
        double m1[8], m2[8], diff[16];
        const int n1 = MulTwoTwo(u, v, m1);
        const int n2 = MulTwoTwo(w, x, m2);
        ExpNegate(n2, m2);
        const int nd = ExpSum(n1, m1, n2, m2, diff);
        return MulExpTwo(nd, diff, s, out, scratch);
    };
    double t1[64], t2[64], t3[64], t12[128];
    const int n1 = cross_term(adx, bdy, cdz, bdz, cdy, t1);
    const int n2 = cross_term(ady, bdz, cdx, bdx, cdz, t2);
    const int n3 = cross_term(adz, bdx, cdy, bdy, cdx, t3);
    const int n12 = ExpSum(n1, t1, n2, t2, t12);
    const int n = ExpSum(n12, t12, n3, t3, det);
    return det[n - 1];
}

// Scratch for the exact insphere determinant in raw 5x5 form.
// 2x2 minors over (x, y), 3x3 minors with z, 4x4 minors over (x, y, z, 1), then lift products summed.
struct SphereScratch {
    double Pairs[10][4];
    int PairN[10];
    double Triples[10][24];
    int TripleN[10];
    double Quad[96], Term[1152], Run[5760], RunNext[5760];
    double Tmp[1152], Tmp2[192];
};

constexpr int PairIndex[5][5]{
    {-1, 0, 1, 2, 3},
    {0, -1, 4, 5, 6},
    {1, 4, -1, 7, 8},
    {2, 5, 7, -1, 9},
    {3, 6, 8, 9, -1},
};
constexpr auto TripleIndex = [] {
    std::array<std::array<std::array<int, 5>, 5>, 5> idx{};
    int n = 0;
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            for (int k = j + 1; k < 5; ++k) idx[i][j][k] = n++;
    return idx;
}();

// Exact 5x5 insphere determinant over rows (x, y, z, |p|^2, 1).
// Equal to the difference-form 4x4 determinant the fast path computes.
// Sign is exact.
double InSphereExact(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e) {
    thread_local auto scratch = std::make_unique<SphereScratch>();
    auto *s = scratch.get();
    const dvec3 p[5]{a, b, c, d, e};

    // Pairs: P(i, j) = x_i * y_j - x_j * y_i for i < j.
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            const int pi = PairIndex[i][j];
            const auto m1 = TwoProd(p[i].x, p[j].y);
            auto m2 = TwoProd(p[j].x, p[i].y);
            double e1[2]{m1.Lo, m1.Hi}, e2[2]{-m2.Lo, -m2.Hi};
            s->PairN[pi] = ExpSum(2, e1, 2, e2, s->Pairs[pi]);
        }
    }
    // Signed pair lookup: P(i, j) = -P(j, i).
    const auto pair = [&s](int i, int j, double *out) {
        const int pi = PairIndex[i][j];
        const int n = s->PairN[pi];
        std::copy_n(s->Pairs[pi], n, out);
        if (i > j) ExpNegate(n, out);
        return n;
    };
    // Triples: T(i, j, k) = det3 over (x, y, z) of rows i, j, k for i < j < k.
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            for (int k = j + 1; k < 5; ++k) {
                double pjk[4], pik[4], pij[4], t1[8], t2[8], t12[16];
                const int njk = pair(j, k, pjk), nik = pair(i, k, pik), nij = pair(i, j, pij);
                const int n1 = ExpScale(njk, pjk, p[i].z, t1);
                const int n2 = ExpScale(nik, pik, p[j].z, t2);
                ExpNegate(n2, t2);
                const int n12 = ExpSum(n1, t1, n2, t2, t12);
                const int n3 = ExpScale(nij, pij, p[k].z, t1);
                const int ti = TripleIndex[i][j][k];
                s->TripleN[ti] = ExpSum(n12, t12, n3, t1, s->Triples[ti]);
            }
        }
    }
    // Quad over rows (r0 < r1 < r2 < r3), columns (x, y, z, 1): -T(r1, r2, r3) + T(r0, r2, r3) - T(r0, r1, r3) + T(r0, r1, r2).
    const auto quad = [&s](const std::array<int, 4> &r, double *out96) {
        double q1[48], q2[48];
        const auto triple = [&s](int i, int j, int k, double *out, bool negate) {
            const int ti = TripleIndex[i][j][k];
            const int n = s->TripleN[ti];
            std::copy_n(s->Triples[ti], n, out);
            if (negate) ExpNegate(n, out);
            return n;
        };
        double t1[24], t2[24];
        int n1 = triple(r[1], r[2], r[3], t1, true);
        int n2 = triple(r[0], r[2], r[3], t2, false);
        const int nq1 = ExpSum(n1, t1, n2, t2, q1);
        n1 = triple(r[0], r[1], r[3], t1, true);
        n2 = triple(r[0], r[1], r[2], t2, false);
        const int nq2 = ExpSum(n1, t1, n2, t2, q2);
        return ExpSum(nq1, q1, nq2, q2, out96);
    };
    // det = sum over rows r of (-1)^(r + 1) * lift_r * quad(other rows).
    int run_n = 0;
    double *run = s->Run, *run_next = s->RunNext;
    for (int r = 0; r < 5; ++r) {
        std::array<int, 4> others;
        for (int i = 0, o = 0; i < 5; ++i)
            if (i != r) others[o++] = i;
        const int nq = quad(others, s->Quad);
        // lift_r = x^2 + y^2 + z^2 as an exact 6-expansion.
        const auto sqx = TwoProd(p[r].x, p[r].x), sqy = TwoProd(p[r].y, p[r].y), sqz = TwoProd(p[r].z, p[r].z);
        double lx[2]{sqx.Lo, sqx.Hi}, ly[2]{sqy.Lo, sqy.Hi}, lz[2]{sqz.Lo, sqz.Hi};
        double lxy[4], lift[6];
        const int nxy = ExpSum(2, lx, 2, ly, lxy);
        const int nl = ExpSum(nxy, lxy, 2, lz, lift);
        // term = lift * quad, accumulated by scaling quad with each lift component.
        int term_n = 0;
        for (int i = 0; i < nl; ++i) {
            const int ns = ExpScale(nq, s->Quad, lift[i], s->Tmp2);
            if (i == 0) {
                std::copy_n(s->Tmp2, ns, s->Term);
                term_n = ns;
            } else {
                term_n = ExpSum(term_n, s->Term, ns, s->Tmp2, s->Tmp);
                std::copy_n(s->Tmp, term_n, s->Term);
            }
        }
        if (r % 2 == 0) ExpNegate(term_n, s->Term);
        if (r == 0) {
            std::copy_n(s->Term, term_n, run);
            run_n = term_n;
        } else {
            run_n = ExpSum(run_n, run, term_n, s->Term, run_next);
            std::swap(run, run_next);
        }
    }
    return run[run_n - 1];
}
// x = (a.Hi + a.Lo) - (b.Hi + b.Lo) as a four-component expansion.
void TwoTwoDiff(Two a, Two b, double *x) {
    const Two lo = TwoDiff(a.Lo, b.Lo);
    x[0] = lo.Lo;
    const Two mid = TwoSum(a.Hi, lo.Hi);
    const Two hi = TwoDiff(mid.Lo, b.Hi);
    x[1] = hi.Lo;
    const Two top = TwoSum(mid.Hi, hi.Hi);
    x[2] = top.Lo;
    x[3] = top.Hi;
}

// The two refinement stages between the filter and the exact determinant.
// Stage B evaluates the determinant exactly in the rounded differences, and stage C adds what the difference tails contribute.
// The permanent is the one the filter already computed.
double Orient3DAdapt(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, double permanent) {
    const double adx = a.x - d.x, bdx = b.x - d.x, cdx = c.x - d.x;
    const double ady = a.y - d.y, bdy = b.y - d.y, cdy = c.y - d.y;
    const double adz = a.z - d.z, bdz = b.z - d.z, cdz = c.z - d.z;

    double bc[4], ca[4], ab[4], adet[8], bdet[8], cdet[8];
    TwoTwoDiff(TwoProd(bdx, cdy), TwoProd(cdx, bdy), bc);
    const int alen = ExpScale(4, bc, adz, adet);
    TwoTwoDiff(TwoProd(cdx, ady), TwoProd(adx, cdy), ca);
    const int blen = ExpScale(4, ca, bdz, bdet);
    TwoTwoDiff(TwoProd(adx, bdy), TwoProd(bdx, ady), ab);
    const int clen = ExpScale(4, ab, cdz, cdet);

    double abdet[16], fin[24];
    const int ablen = ExpSum(alen, adet, blen, bdet, abdet);
    const int finlen = ExpSum(ablen, abdet, clen, cdet, fin);

    double det = Estimate(finlen, fin);
    constexpr double BoundB = (3.0 + 28.0 * Eps) * Eps;
    double errbound = BoundB * permanent;
    if (det >= errbound || -det >= errbound) return det;

    const double adxtail = TwoDiff(a.x, d.x).Lo, bdxtail = TwoDiff(b.x, d.x).Lo, cdxtail = TwoDiff(c.x, d.x).Lo;
    const double adytail = TwoDiff(a.y, d.y).Lo, bdytail = TwoDiff(b.y, d.y).Lo, cdytail = TwoDiff(c.y, d.y).Lo;
    const double adztail = TwoDiff(a.z, d.z).Lo, bdztail = TwoDiff(b.z, d.z).Lo, cdztail = TwoDiff(c.z, d.z).Lo;
    // Every difference was exact, so stage B already held the whole determinant.
    if (adxtail == 0 && bdxtail == 0 && cdxtail == 0 && adytail == 0 && bdytail == 0 && cdytail == 0 &&
        adztail == 0 && bdztail == 0 && cdztail == 0) {
        return det;
    }

    constexpr double BoundC = (26.0 + 288.0 * Eps) * Eps * Eps;
    constexpr double ResultBound = (3.0 + 8.0 * Eps) * Eps;
    errbound = BoundC * permanent + ResultBound * std::abs(det);
    det += (adz * ((bdx * cdytail + cdy * bdxtail) - (bdy * cdxtail + cdx * bdytail)) + adztail * (bdx * cdy - bdy * cdx)) +
        (bdz * ((cdx * adytail + ady * cdxtail) - (cdy * adxtail + adx * cdytail)) + bdztail * (cdx * ady - cdy * adx)) +
        (cdz * ((adx * bdytail + bdy * adxtail) - (ady * bdxtail + bdx * adytail)) + cdztail * (adx * bdy - ady * bdx));
    if (det >= errbound || -det >= errbound) return det;

    return Orient3DExact(a, b, c, d);
}

// The two refinement stages between the filter and the exact determinant.
// Stage B is exact in the rounded differences, and stage C adds what the difference tails contribute.
// Each returns as soon as its own error bound clears, so the exact expansion runs only for points that really are cospherical.
// The permanent is the one the filter already computed.
double InSphereAdapt(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e, double permanent) {
    const double aex = a.x - e.x, bex = b.x - e.x, cex = c.x - e.x, dex = d.x - e.x;
    const double aey = a.y - e.y, bey = b.y - e.y, cey = c.y - e.y, dey = d.y - e.y;
    const double aez = a.z - e.z, bez = b.z - e.z, cez = c.z - e.z, dez = d.z - e.z;

    // The six 2x2 minors over (x, y), each exact in four components.
    double ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
    TwoTwoDiff(TwoProd(aex, bey), TwoProd(bex, aey), ab);
    TwoTwoDiff(TwoProd(bex, cey), TwoProd(cex, bey), bc);
    TwoTwoDiff(TwoProd(cex, dey), TwoProd(dex, cey), cd);
    TwoTwoDiff(TwoProd(dex, aey), TwoProd(aex, dey), da);
    TwoTwoDiff(TwoProd(aex, cey), TwoProd(cex, aey), ac);
    TwoTwoDiff(TwoProd(bex, dey), TwoProd(dex, bey), bd);

    double temp8a[8], temp8b[8], temp8c[8], temp16[16], temp24[24], temp48[48];
    double xdet[96], ydet[96], zdet[96], xydet[192];
    // One row's contribution: the 3x3 minor of the other three rows, times that row's lift.
    // The lift is a pair of scalings by each of the row's own coordinates, so the product stays a single expansion.
    // The lift carries the cofactor sign, alternating down the rows.
    const auto row = [&](const double *m1, double s1, const double *m2, double s2, const double *m3, double s3,
                         double x, double y, double z, double lift_sign, double *out) {
        const int n8a = ExpScale(4, m1, s1, temp8a);
        const int n8b = ExpScale(4, m2, s2, temp8b);
        const int n8c = ExpScale(4, m3, s3, temp8c);
        const int n16 = ExpSum(n8a, temp8a, n8b, temp8b, temp16);
        const int n24 = ExpSum(n8c, temp8c, n16, temp16, temp24);
        const int nx = ExpScale(ExpScale(n24, temp24, x, temp48), temp48, lift_sign * x, xdet);
        const int ny = ExpScale(ExpScale(n24, temp24, y, temp48), temp48, lift_sign * y, ydet);
        const int nz = ExpScale(ExpScale(n24, temp24, z, temp48), temp48, lift_sign * z, zdet);
        const int nxy = ExpSum(nx, xdet, ny, ydet, xydet);
        return ExpSum(nxy, xydet, nz, zdet, out);
    };
    double adet[288], bdet[288], cdet[288], ddet[288];
    const int alen = row(cd, bez, bd, -cez, bc, dez, aex, aey, aez, -1.0, adet);
    const int blen = row(da, cez, ac, dez, cd, aez, bex, bey, bez, 1.0, bdet);
    const int clen = row(ab, dez, bd, aez, da, bez, cex, cey, cez, -1.0, cdet);
    const int dlen = row(bc, aez, ac, -bez, ab, cez, dex, dey, dez, 1.0, ddet);

    double abdet[576], cddet[576], fin[1152];
    const int ablen = ExpSum(alen, adet, blen, bdet, abdet);
    const int cdlen = ExpSum(clen, cdet, dlen, ddet, cddet);
    const int finlen = ExpSum(ablen, abdet, cdlen, cddet, fin);

    double det = Estimate(finlen, fin);
    constexpr double BoundB = (5.0 + 72.0 * Eps) * Eps;
    double errbound = BoundB * permanent;
    if (det >= errbound || -det >= errbound) return det;

    const double aextail = TwoDiff(a.x, e.x).Lo, aeytail = TwoDiff(a.y, e.y).Lo, aeztail = TwoDiff(a.z, e.z).Lo;
    const double bextail = TwoDiff(b.x, e.x).Lo, beytail = TwoDiff(b.y, e.y).Lo, beztail = TwoDiff(b.z, e.z).Lo;
    const double cextail = TwoDiff(c.x, e.x).Lo, ceytail = TwoDiff(c.y, e.y).Lo, ceztail = TwoDiff(c.z, e.z).Lo;
    const double dextail = TwoDiff(d.x, e.x).Lo, deytail = TwoDiff(d.y, e.y).Lo, deztail = TwoDiff(d.z, e.z).Lo;
    // Every difference was exact, so stage B already held the whole determinant.
    if (aextail == 0 && aeytail == 0 && aeztail == 0 && bextail == 0 && beytail == 0 && beztail == 0 &&
        cextail == 0 && ceytail == 0 && ceztail == 0 && dextail == 0 && deytail == 0 && deztail == 0) {
        return det;
    }

    constexpr double BoundC = (71.0 + 1408.0 * Eps) * Eps * Eps;
    constexpr double ResultBound = (3.0 + 8.0 * Eps) * Eps;
    errbound = BoundC * permanent + ResultBound * std::abs(det);
    const double abeps = (aex * beytail + bey * aextail) - (aey * bextail + bex * aeytail);
    const double bceps = (bex * ceytail + cey * bextail) - (bey * cextail + cex * beytail);
    const double cdeps = (cex * deytail + dey * cextail) - (cey * dextail + dex * ceytail);
    const double daeps = (dex * aeytail + aey * dextail) - (dey * aextail + aex * deytail);
    const double aceps = (aex * ceytail + cey * aextail) - (aey * cextail + cex * aeytail);
    const double bdeps = (bex * deytail + dey * bextail) - (bey * dextail + dex * beytail);
    const double ab3 = ab[3], bc3 = bc[3], cd3 = cd[3], da3 = da[3], ac3 = ac[3], bd3 = bd[3];
    det += (((bex * bex + bey * bey + bez * bez) * ((cez * daeps + dez * aceps + aez * cdeps) + (ceztail * da3 + deztail * ac3 + aeztail * cd3)) +
             (dex * dex + dey * dey + dez * dez) * ((aez * bceps - bez * aceps + cez * abeps) + (aeztail * bc3 - beztail * ac3 + ceztail * ab3))) -
            ((aex * aex + aey * aey + aez * aez) * ((bez * cdeps - cez * bdeps + dez * bceps) + (beztail * cd3 - ceztail * bd3 + deztail * bc3)) +
             (cex * cex + cey * cey + cez * cez) * ((dez * abeps + aez * bdeps + bez * daeps) + (deztail * ab3 + aeztail * bd3 + beztail * da3)))) +
        2.0 * (((bex * bextail + bey * beytail + bez * beztail) * (cez * da3 + dez * ac3 + aez * cd3) + (dex * dextail + dey * deytail + dez * deztail) * (aez * bc3 - bez * ac3 + cez * ab3)) - ((aex * aextail + aey * aeytail + aez * aeztail) * (bez * cd3 - cez * bd3 + dez * bc3) + (cex * cextail + cey * ceytail + cez * ceztail) * (dez * ab3 + aez * bd3 + bez * da3)));
    if (det >= errbound || -det >= errbound) return det;

    return InSphereExact(a, b, c, d, e);
}

// The 3x3 determinant, built from the six 2x2 minors of x and y and scaled by z.
// The value is the expansion's leading component, not the rounded sum, so the pipeline order is part of the result.
double Orient3DCofactor(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    double ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
    TwoTwoDiff(TwoProd(a.x, b.y), TwoProd(b.x, a.y), ab);
    TwoTwoDiff(TwoProd(b.x, c.y), TwoProd(c.x, b.y), bc);
    TwoTwoDiff(TwoProd(c.x, d.y), TwoProd(d.x, c.y), cd);
    TwoTwoDiff(TwoProd(d.x, a.y), TwoProd(a.x, d.y), da);
    TwoTwoDiff(TwoProd(a.x, c.y), TwoProd(c.x, a.y), ac);
    TwoTwoDiff(TwoProd(b.x, d.y), TwoProd(d.x, b.y), bd);

    double temp8[8], abc[12], bcd[12], cda[12], dab[12];
    int len = ExpSum(4, cd, 4, da, temp8);
    const int cdalen = ExpSum(len, temp8, 4, ac, cda);
    len = ExpSum(4, da, 4, ab, temp8);
    const int dablen = ExpSum(len, temp8, 4, bd, dab);
    ExpNegate(4, bd);
    ExpNegate(4, ac);
    len = ExpSum(4, ab, 4, bc, temp8);
    const int abclen = ExpSum(len, temp8, 4, ac, abc);
    len = ExpSum(4, bc, 4, cd, temp8);
    const int bcdlen = ExpSum(len, temp8, 4, bd, bcd);

    double adet[24], bdet[24], cdet[24], ddet[24];
    const int alen = ExpScale(bcdlen, bcd, a.z, adet);
    const int blen = ExpScale(cdalen, cda, -b.z, bdet);
    const int clen = ExpScale(dablen, dab, c.z, cdet);
    const int dlen = ExpScale(abclen, abc, -d.z, ddet);

    double abdet[48], cddet[48], deter[96];
    const int ablen = ExpSum(alen, adet, blen, bdet, abdet);
    const int cdlen = ExpSum(clen, cdet, dlen, ddet, cddet);
    const int deterlen = ExpSum(ablen, abdet, cdlen, cddet, deter);
    return deter[deterlen - 1];
}

// The 4x4 determinant of five lifted points, through the same cofactor pipeline.
// Ten 2x2 minors of x and y scale by z into ten triples, which combine into five quadruples and scale by the heights.
double Orient4DCofactor(
    const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e,
    double aheight, double bheight, double cheight, double dheight, double eheight
) {
    double ab[4], bc[4], cd[4], de[4], ea[4], ac[4], bd[4], ce[4], da[4], eb[4];
    TwoTwoDiff(TwoProd(a.x, b.y), TwoProd(b.x, a.y), ab);
    TwoTwoDiff(TwoProd(b.x, c.y), TwoProd(c.x, b.y), bc);
    TwoTwoDiff(TwoProd(c.x, d.y), TwoProd(d.x, c.y), cd);
    TwoTwoDiff(TwoProd(d.x, e.y), TwoProd(e.x, d.y), de);
    TwoTwoDiff(TwoProd(e.x, a.y), TwoProd(a.x, e.y), ea);
    TwoTwoDiff(TwoProd(a.x, c.y), TwoProd(c.x, a.y), ac);
    TwoTwoDiff(TwoProd(b.x, d.y), TwoProd(d.x, b.y), bd);
    TwoTwoDiff(TwoProd(c.x, e.y), TwoProd(e.x, c.y), ce);
    TwoTwoDiff(TwoProd(d.x, a.y), TwoProd(a.x, d.y), da);
    TwoTwoDiff(TwoProd(e.x, b.y), TwoProd(b.x, e.y), eb);

    double temp8a[8], temp8b[8], temp16[16];
    double abc[24], bcd[24], cde[24], dea[24], eab[24], abd[24], bce[24], cda[24], deb[24], eac[24];
    // Each triple is one minor scaled by a z, less another, plus a third.
    const auto triple = [&](const double *x, double zx, const double *y, double zy, const double *z, double zz, double *out) {
        const int alen = ExpScale(4, x, zx, temp8a);
        const int blen = ExpScale(4, y, zy, temp8b);
        const int mlen = ExpSum(alen, temp8a, blen, temp8b, temp16);
        const int clen = ExpScale(4, z, zz, temp8a);
        return ExpSum(clen, temp8a, mlen, temp16, out);
    };
    const int abclen = triple(bc, a.z, ac, -b.z, ab, c.z, abc);
    const int bcdlen = triple(cd, b.z, bd, -c.z, bc, d.z, bcd);
    const int cdelen = triple(de, c.z, ce, -d.z, cd, e.z, cde);
    const int dealen = triple(ea, d.z, da, -e.z, de, a.z, dea);
    const int eablen = triple(ab, e.z, eb, -a.z, ea, b.z, eab);
    const int abdlen = triple(bd, a.z, da, b.z, ab, d.z, abd);
    const int bcelen = triple(ce, b.z, eb, c.z, bc, e.z, bce);
    const int cdalen = triple(da, c.z, ac, d.z, cd, a.z, cda);
    const int deblen = triple(eb, d.z, bd, e.z, de, b.z, deb);
    const int eaclen = triple(ac, e.z, ce, a.z, ea, c.z, eac);

    double temp48a[48], temp48b[48];
    double abcd[96], bcde[96], cdea[96], deab[96], eabc[96];
    double adet[192], bdet[192], cdet[192], ddet[192], edet[192];
    // Each quadruple is one pair of triples less another, scaled by a height.
    const auto quad = [&](const double *p, int plen, const double *q, int qlen, const double *r, int rlen, const double *t, int tlen, double *sum, double height, double *out) {
        const int alen = ExpSum(plen, p, qlen, q, temp48a);
        const int blen = ExpSum(rlen, r, tlen, t, temp48b);
        ExpNegate(blen, temp48b);
        const int slen = ExpSum(alen, temp48a, blen, temp48b, sum);
        return ExpScale(slen, sum, height, out);
    };
    const int alen = quad(cde, cdelen, bce, bcelen, deb, deblen, bcd, bcdlen, bcde, aheight, adet);
    const int blen = quad(dea, dealen, cda, cdalen, eac, eaclen, cde, cdelen, cdea, bheight, bdet);
    const int clen = quad(eab, eablen, deb, deblen, abd, abdlen, dea, dealen, deab, cheight, cdet);
    const int dlen = quad(abc, abclen, eac, eaclen, bce, bcelen, eab, eablen, eabc, dheight, ddet);
    const int elen = quad(bcd, bcdlen, abd, abdlen, cda, cdalen, abc, abclen, abcd, eheight, edet);

    double abdet[384], cddet[384], cdedet[576], deter[960];
    const int ablen = ExpSum(alen, adet, blen, bdet, abdet);
    const int cdlen = ExpSum(clen, cdet, dlen, ddet, cddet);
    const int cdelen2 = ExpSum(cdlen, cddet, elen, edet, cdedet);
    const int deterlen = ExpSum(ablen, abdet, cdelen2, cdedet, deter);
    return deter[deterlen - 1];
}

// The two refinement stages between orient4d's filter and the exact determinant.
// Built like InSphereAdapt, with the heights in place of the lifts, so each row is a single scaling rather than a squared pair.
double Orient4DAdapt(
    const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e,
    double aheight, double bheight, double cheight, double dheight, double eheight, double permanent
) {
    const double aex = a.x - e.x, bex = b.x - e.x, cex = c.x - e.x, dex = d.x - e.x;
    const double aey = a.y - e.y, bey = b.y - e.y, cey = c.y - e.y, dey = d.y - e.y;
    const double aez = a.z - e.z, bez = b.z - e.z, cez = c.z - e.z, dez = d.z - e.z;
    const double aeheight = aheight - eheight, beheight = bheight - eheight;
    const double ceheight = cheight - eheight, deheight = dheight - eheight;

    double ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
    TwoTwoDiff(TwoProd(aex, bey), TwoProd(bex, aey), ab);
    TwoTwoDiff(TwoProd(bex, cey), TwoProd(cex, bey), bc);
    TwoTwoDiff(TwoProd(cex, dey), TwoProd(dex, cey), cd);
    TwoTwoDiff(TwoProd(dex, aey), TwoProd(aex, dey), da);
    TwoTwoDiff(TwoProd(aex, cey), TwoProd(cex, aey), ac);
    TwoTwoDiff(TwoProd(bex, dey), TwoProd(dex, bey), bd);

    double temp8a[8], temp8b[8], temp8c[8], temp16[16], temp24[24];
    const auto row = [&](const double *m1, double s1, const double *m2, double s2, const double *m3, double s3, double height, double *out) {
        const int n8a = ExpScale(4, m1, s1, temp8a);
        const int n8b = ExpScale(4, m2, s2, temp8b);
        const int n8c = ExpScale(4, m3, s3, temp8c);
        const int n16 = ExpSum(n8a, temp8a, n8b, temp8b, temp16);
        const int n24 = ExpSum(n8c, temp8c, n16, temp16, temp24);
        return ExpScale(n24, temp24, height, out);
    };
    double adet[48], bdet[48], cdet[48], ddet[48];
    const int alen = row(cd, bez, bd, -cez, bc, dez, -aeheight, adet);
    const int blen = row(da, cez, ac, dez, cd, aez, beheight, bdet);
    const int clen = row(ab, dez, bd, aez, da, bez, -ceheight, cdet);
    const int dlen = row(bc, aez, ac, -bez, ab, cez, deheight, ddet);

    double abdet[96], cddet[96], fin[192];
    const int ablen = ExpSum(alen, adet, blen, bdet, abdet);
    const int cdlen = ExpSum(clen, cdet, dlen, ddet, cddet);
    const int finlen = ExpSum(ablen, abdet, cdlen, cddet, fin);

    double det = Estimate(finlen, fin);
    constexpr double BoundB = (5.0 + 72.0 * Eps) * Eps;
    double errbound = BoundB * permanent;
    if (det >= errbound || -det >= errbound) return det;

    const double aextail = TwoDiff(a.x, e.x).Lo, aeytail = TwoDiff(a.y, e.y).Lo, aeztail = TwoDiff(a.z, e.z).Lo;
    const double bextail = TwoDiff(b.x, e.x).Lo, beytail = TwoDiff(b.y, e.y).Lo, beztail = TwoDiff(b.z, e.z).Lo;
    const double cextail = TwoDiff(c.x, e.x).Lo, ceytail = TwoDiff(c.y, e.y).Lo, ceztail = TwoDiff(c.z, e.z).Lo;
    const double dextail = TwoDiff(d.x, e.x).Lo, deytail = TwoDiff(d.y, e.y).Lo, deztail = TwoDiff(d.z, e.z).Lo;
    const double aeheighttail = TwoDiff(aheight, eheight).Lo, beheighttail = TwoDiff(bheight, eheight).Lo;
    const double ceheighttail = TwoDiff(cheight, eheight).Lo, deheighttail = TwoDiff(dheight, eheight).Lo;
    // Every difference was exact, so stage B already held the whole determinant.
    if (aextail == 0 && aeytail == 0 && aeztail == 0 && bextail == 0 && beytail == 0 && beztail == 0 &&
        cextail == 0 && ceytail == 0 && ceztail == 0 && dextail == 0 && deytail == 0 && deztail == 0 &&
        aeheighttail == 0 && beheighttail == 0 && ceheighttail == 0 && deheighttail == 0) {
        return det;
    }

    constexpr double BoundC = (71.0 + 1408.0 * Eps) * Eps * Eps;
    constexpr double ResultBound = (3.0 + 8.0 * Eps) * Eps;
    errbound = BoundC * permanent + ResultBound * std::abs(det);
    const double abeps = (aex * beytail + bey * aextail) - (aey * bextail + bex * aeytail);
    const double bceps = (bex * ceytail + cey * bextail) - (bey * cextail + cex * beytail);
    const double cdeps = (cex * deytail + dey * cextail) - (cey * dextail + dex * ceytail);
    const double daeps = (dex * aeytail + aey * dextail) - (dey * aextail + aex * deytail);
    const double aceps = (aex * ceytail + cey * aextail) - (aey * cextail + cex * aeytail);
    const double bdeps = (bex * deytail + dey * bextail) - (bey * dextail + dex * beytail);
    const double ab3 = ab[3], bc3 = bc[3], cd3 = cd[3], da3 = da[3], ac3 = ac[3], bd3 = bd[3];
    det += ((beheight * ((cez * daeps + dez * aceps + aez * cdeps) + (ceztail * da3 + deztail * ac3 + aeztail * cd3)) +
             deheight * ((aez * bceps - bez * aceps + cez * abeps) + (aeztail * bc3 - beztail * ac3 + ceztail * ab3))) -
            (aeheight * ((bez * cdeps - cez * bdeps + dez * bceps) + (beztail * cd3 - ceztail * bd3 + deztail * bc3)) +
             ceheight * ((dez * abeps + aez * bdeps + bez * daeps) + (deztail * ab3 + aeztail * bd3 + beztail * da3)))) +
        ((beheighttail * (cez * da3 + dez * ac3 + aez * cd3) + deheighttail * (aez * bc3 - bez * ac3 + cez * ab3)) -
         (aeheighttail * (bez * cd3 - cez * bd3 + dez * bc3) + ceheighttail * (dez * ab3 + aez * bd3 + bez * da3)));
    if (det >= errbound || -det >= errbound) return det;

    return Orient4DCofactor(a, b, c, d, e, aheight, bheight, cheight, dheight, eheight);
}

} // namespace

namespace geom {
double Orient3DExactCofactor(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) { return Orient3DCofactor(a, b, c, d); }
double Orient3DRefined(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, double permanent) { return Orient3DAdapt(a, b, c, d, permanent); }
double Orient4DExactCofactor(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e, double ah, double bh, double ch, double dh, double eh) {
    return Orient4DCofactor(a, b, c, d, e, ah, bh, ch, dh, eh);
}
double Orient4DRefined(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e, double ah, double bh, double ch, double dh, double eh, double permanent) {
    return Orient4DAdapt(a, b, c, d, e, ah, bh, ch, dh, eh, permanent);
}

double Orient3D(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    const double adx = a.x - d.x, ady = a.y - d.y, adz = a.z - d.z;
    const double bdx = b.x - d.x, bdy = b.y - d.y, bdz = b.z - d.z;
    const double cdx = c.x - d.x, cdy = c.y - d.y, cdz = c.z - d.z;

    const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
    const double cdxady = cdx * ady, adxcdy = adx * cdy;
    const double adxbdy = adx * bdy, bdxady = bdx * ady;
    const double det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);

    const double permanent = (std::abs(bdxcdy) + std::abs(cdxbdy)) * std::abs(adz) +
        (std::abs(cdxady) + std::abs(adxcdy)) * std::abs(bdz) +
        (std::abs(adxbdy) + std::abs(bdxady)) * std::abs(cdz);
    constexpr double BoundA = (7.0 + 56.0 * Eps) * Eps;
    const double errbound = BoundA * permanent;
    if (det > errbound || -det > errbound) return det;

    return Orient3DAdapt(a, b, c, d, permanent);
}

// The difference-form determinant in a fixed expression order.
// The approximate value is returned whenever it clears the first error bound.
// Anything below that bound goes to the refinement stages, which reach the exact expansion only near cospherical.
double InSphere(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e) {
    const double aex = a.x - e.x, bex = b.x - e.x, cex = c.x - e.x, dex = d.x - e.x;
    const double aey = a.y - e.y, bey = b.y - e.y, cey = c.y - e.y, dey = d.y - e.y;
    const double aez = a.z - e.z, bez = b.z - e.z, cez = c.z - e.z, dez = d.z - e.z;

    // 2x2 minors over (x, y) of the four difference rows.
    const double aexbey = aex * bey, bexaey = bex * aey, ab = aexbey - bexaey;
    const double bexcey = bex * cey, cexbey = cex * bey, bc = bexcey - cexbey;
    const double cexdey = cex * dey, dexcey = dex * cey, cd = cexdey - dexcey;
    const double dexaey = dex * aey, aexdey = aex * dey, da = dexaey - aexdey;
    const double aexcey = aex * cey, cexaey = cex * aey, ac = aexcey - cexaey;
    const double bexdey = bex * dey, dexbey = dex * bey, bd = bexdey - dexbey;
    // 3x3 minors over (x, y, z).
    const double abc = aez * bc - bez * ac + cez * ab;
    const double bcd = bez * cd - cez * bd + dez * bc;
    const double cda = cez * da + dez * ac + aez * cd;
    const double dab = dez * ab + aez * bd + bez * da;

    const double alift = aex * aex + aey * aey + aez * aez;
    const double blift = bex * bex + bey * bey + bez * bez;
    const double clift = cex * cex + cey * cey + cez * cez;
    const double dlift = dex * dex + dey * dey + dez * dez;
    const double det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);

    const double aezplus = std::abs(aez), bezplus = std::abs(bez), cezplus = std::abs(cez), dezplus = std::abs(dez);
    const double permanent =
        ((std::abs(cexdey) + std::abs(dexcey)) * bezplus + (std::abs(dexbey) + std::abs(bexdey)) * cezplus + (std::abs(bexcey) + std::abs(cexbey)) * dezplus) * alift +
        ((std::abs(dexaey) + std::abs(aexdey)) * cezplus + (std::abs(aexcey) + std::abs(cexaey)) * dezplus + (std::abs(cexdey) + std::abs(dexcey)) * aezplus) * blift +
        ((std::abs(aexbey) + std::abs(bexaey)) * dezplus + (std::abs(bexdey) + std::abs(dexbey)) * aezplus + (std::abs(dexaey) + std::abs(aexdey)) * bezplus) * clift +
        ((std::abs(bexcey) + std::abs(cexbey)) * aezplus + (std::abs(cexaey) + std::abs(aexcey)) * bezplus + (std::abs(aexbey) + std::abs(bexaey)) * cezplus) * dlift;
    constexpr double BoundA = (16.0 + 224.0 * Eps) * Eps;
    const double errbound = BoundA * permanent;
    if (det > errbound || -det > errbound) return det;

    return InSphereAdapt(a, b, c, d, e, permanent);
}

int InSphereSoS(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d, const dvec3 &e, uint32_t ia, uint32_t ib, uint32_t ic, uint32_t id, uint32_t ie) {
    if (const double det = InSphere(a, b, c, d, e); det != 0) return det > 0 ? 1 : -1;
    // Cospherical: perturb each lift by an amount that shrinks with the point's index, so the smallest index dominates.
    // The perturbation term of row r is (-1)^(r + 3) * orient3d of the remaining rows in order.
    const dvec3 pts[5]{a, b, c, d, e};
    std::array<uint32_t, 5> order{0, 1, 2, 3, 4};
    const uint32_t ids[5]{ia, ib, ic, id, ie};
    std::ranges::sort(order, [&ids](uint32_t l, uint32_t r) { return ids[l] < ids[r]; });
    for (const uint32_t r : order) {
        std::array<int, 4> others;
        for (int i = 0, o = 0; i < 5; ++i) {
            if (uint32_t(i) != r) others[o++] = i;
        }
        const double orient = Orient3D(pts[others[0]], pts[others[1]], pts[others[2]], pts[others[3]]);
        if (orient != 0) return (r % 2 == 1 ? 1 : -1) * (orient > 0 ? 1 : -1);
    }
    return 0; // All five points coplanar.
}
} // namespace geom
