#pragma once

#include "audio/ContactModel.h"
#include "audio/ModalAudio.h"
#include "audio/SurfaceContact.h"
#include "numeric/vec3.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

// The surface-contact audio model: how two bodies resting or sliding on one another sound.
// SURFACE_AUDIO=1 at configure time selects it, behind the audio/SurfaceContact.h interface.

/***** Roughness tracks: the height field under a contact *****/

// Samples in a roughness track, which is read cyclically.
constexpr uint32_t TrackSamples{131072};

// Distance a contact travels before a synthesized field repeats, m.
constexpr float SurfaceRepeatLength{0.1f};

// Thread count a synthesis splits its columns over.
// SURFACE_FFT_THREADS overrides it.
void SetTransformThreads(uint32_t threads);

uint64_t HashParams(uint64_t seed, auto... values) {
    const auto combine = [&seed](double v) { seed ^= std::hash<double>{}(v) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2); };
    (combine(double(values)), ...);
    return seed;
}

// Boxcar widths a track's mean height is measured at, as powers of two in samples.
constexpr uint32_t TrackWidthCount{18};

// Track samples one short-wavelength cutoff spans (Pastewka and Robbins arXiv:1508.02154).
constexpr float SurfaceSamplesPerCutoff{4};

// Sample spacing a track resolves `short_wavelength` at, m.
// SURFACE_SAMPLES_PER_CUTOFF refines or coarsens it against that cutoff.
inline float FinishTrackSpacing(float short_wavelength) {
    static const float per_cutoff = [] {
        const char *v = std::getenv("SURFACE_SAMPLES_PER_CUTOFF");
        return v ? float(std::atof(v)) : SurfaceSamplesPerCutoff;
    }();
    return std::clamp(short_wavelength / per_cutoff, 1e-9f, 1e-4f);
}

// Distance a contact travels over which a track's local mean is removed, m.
constexpr float ReliefDcLength{1e-2f};

// The band a track's heights hold: the wavelength its spectrum turns over at, and the exponent it falls with above that.
// Fit as flat below the corner and a power law above it, by least squares on the track's power spectrum.
struct MeasuredBand {
    float CorrelationLength{0}, SpectralSlope{0};
};

// Surface heights indexed by distance along the surface, traversed cyclically.
struct RoughnessTrack {
    std::vector<float> Heights; // Zero-mean, unit root-mean-square.
    std::vector<float> Sum; // Running integral, one entry longer than Heights.
    float Spacing{0}; // Distance along the surface between samples, m.
    float Cutoff{0}; // Shortest wavelength these heights hold, m.
    float Rms{1}; // Root-mean-square height of the source, m.
    MeasuredBand Band{}; // Band measured from these heights. Zero on a track synthesized from stated parameters.
    float HeightMax{0}; // The tallest height, over the unit root-mean-square heights.
    // Root-mean-square height difference between adjacent samples, over the unit root-mean-square heights.
    // Divided by the spacing and scaled by the height scale, this is the surface gradient.
    float SlopeRms{0};
    // Root-mean-square second difference across adjacent samples, over the unit root-mean-square heights.
    float CurvatureRms{0};
    // Root-mean-square of the boxcar mean height at width 2^i samples, over the unit root-mean-square heights.
    std::array<float, TrackWidthCount> WindowRms{};
    // Root-mean-square gradient per sample at boxcar width 2^i, over the unit root-mean-square heights.
    // The width-one entry is SlopeRms itself, and widths past what the track resolves repeat the widest measured one.
    std::array<float, TrackWidthCount> SlopeWindowRms{};
};

// A per-octave table read at an arbitrary boxcar width in samples, interpolated between the measured powers of two.
inline float TrackOctaveAt(const std::array<float, TrackWidthCount> &table, float width) {
    if (width <= 1) return table[0];
    const float octave = std::min(std::log2(width), float(TrackWidthCount - 1));
    const auto lo = uint32_t(octave);
    if (lo + 1 >= TrackWidthCount) return table[TrackWidthCount - 1];
    return table[lo] + (octave - float(lo)) * (table[lo + 1] - table[lo]);
}

// Root-mean-square boxcar-mean height at that width, and root-mean-square gradient per sample at it.
inline float TrackWindowRms(const RoughnessTrack &t, float width) { return TrackOctaveAt(t.WindowRms, width); }
inline float TrackSlopeAtWidth(const RoughnessTrack &t, float width) { return TrackOctaveAt(t.SlopeWindowRms, width); }

// Width in samples at which the boxcar-mean height retains half the track's height variance, interpolated in the measured table.
// It sits at the spectral corner, about one correlation length.
inline float TrackHalfVarianceWidth(const RoughnessTrack &t) {
    constexpr float target = 0.5f;
    if (t.WindowRms[0] <= target) return 1.f;
    for (uint32_t i = 1; i < TrackWidthCount; ++i) {
        if (t.WindowRms[i] < target) {
            const float frac = (t.WindowRms[i - 1] - target) / (t.WindowRms[i - 1] - t.WindowRms[i]);
            return std::exp2(float(i - 1) + frac);
        }
    }
    return std::exp2(float(TrackWidthCount - 1));
}

// Synthesize a self-affine roughness track: flat below the spatial frequency the correlation length sets, falling as q^p above it, and empty below the wavelength `short_wavelength`.
// Deterministic in its arguments, and a different spacing draws a different realization.
RoughnessTrack SynthesizeRoughness(float correlation_length, float spectral_slope, float short_wavelength, float spacing, uint32_t count);

// A patch of surface synthesized from the same spectrum as a track, row-major with `columns` along the sweep and `rows` across it, zero mean and unit root-mean-square.
// Rows and columns share one spacing.
struct RoughnessField {
    std::vector<float> Heights;
    uint32_t Columns{0}, Rows{0};
    float Spacing{0};
};

// Height of a field at a position in meters, bilinear between samples and cyclic on both axes.
inline float FieldHeightAt(const RoughnessField &f, double x, double y) {
    if (f.Columns == 0 || f.Rows == 0 || f.Spacing <= 0) return 0.f;
    const auto wrap = [](double v, uint32_t n) {
        const double f = v - std::floor(v / double(n)) * double(n);
        return std::min(f, std::nextafter(double(n), 0.0));
    };
    const double cx = wrap(x / double(f.Spacing), f.Columns), cy = wrap(y / double(f.Spacing), f.Rows);
    const auto x0 = uint32_t(cx), y0 = uint32_t(cy);
    const uint32_t x1 = x0 + 1 < f.Columns ? x0 + 1 : 0, y1 = y0 + 1 < f.Rows ? y0 + 1 : 0;
    const auto tx = float(cx - double(x0)), ty = float(cy - double(y0));
    const float h00 = f.Heights[size_t(x0) * f.Rows + y0], h01 = f.Heights[size_t(x0) * f.Rows + y1];
    const float h10 = f.Heights[size_t(x1) * f.Rows + y0], h11 = f.Heights[size_t(x1) * f.Rows + y1];
    const float at_y0 = h00 + tx * (h10 - h00), at_y1 = h01 + tx * (h11 - h01);
    return at_y0 + ty * (at_y1 - at_y0);
}

// Synthesize a patch whose radial spectrum is flat below the corner the correlation length sets, falls as q^slope above it, and stops at the same short-wavelength cutoff.
// Deterministic in its arguments, and `realization` draws an independent patch of the same surface.
RoughnessField SynthesizeRoughnessPatch(float correlation_length, float spectral_slope, float short_wavelength, float spacing, uint32_t columns, uint32_t rows, uint32_t realization);

// The covariance of a synthesized patch's field at lattice lags up to two samples either way, indexed [column lag + 2][row lag + 2] and normalized to unit variance.
// Every realization of one surface has the same covariance.
using LatticeCovariance = std::array<std::array<double, 5>, 5>;
LatticeCovariance RoughnessLatticeCovariance(float correlation_length, float spectral_slope, float short_wavelength, float spacing, uint32_t columns, uint32_t rows);

// Synthesize a white track, one independent unit-variance draw per sample, deterministic in its seed.
RoughnessTrack SynthesizeTurnover(float spacing, uint32_t count, uint64_t seed);

// Heights measured off a real surface, with Band measured from them.
RoughnessTrack MakeProfileTrack(std::span<const float> heights, float spacing);

// A patch built around a measured trace: the trace's own heights along the sweep, and across the sweep the surface an isotropic reading of its spectrum implies.
// The trace is row zero of the field exactly, and SURFACE_SEED redraws the across-sweep surface alone.
// The field spans the trace, so a measured surface repeats over the length it was measured on.
RoughnessField SynthesizeProfileField(std::span<const float> heights, float spacing, uint32_t rows, uint32_t realization);

// A read position wrapped into the track: the sample, the fraction past it, and whole traversals from the start.
struct TrackPos {
    size_t Index;
    float Frac;
    double Wraps;
};
inline TrackPos WrapTrackPos(const RoughnessTrack &t, double pos) {
    const double n = double(t.Heights.size());
    const double wraps = std::floor(pos / n);
    const double f = std::max(pos - wraps * n, 0.0);
    // Rounding can put the remainder just outside [0, n), which would index past the end of the track.
    const auto i = std::min(size_t(f), t.Heights.size() - 1);
    return {i, float(f - double(i)), wraps};
}

// The running integral at an already wrapped position, extended cyclically.
inline float TrackIntegral(const RoughnessTrack &t, const TrackPos &p) {
    return t.Sum[p.Index] + p.Frac * t.Heights[p.Index] + float(p.Wraps) * t.Sum.back();
}

// The height at a fractional sample position, extended cyclically.
inline float TrackHeight(const RoughnessTrack &t, const TrackPos &p) {
    const auto j = p.Index + 1 < t.Heights.size() ? p.Index + 1 : 0;
    return t.Heights[p.Index] + p.Frac * (t.Heights[j] - t.Heights[p.Index]);
}

// The filtered height under the contact and the slope of that height along the surface, per track sample.
struct TrackReading {
    float Height, Slope;
};

// Height and per-sample slope of the surface reconstructed through the four samples around `pos`, a Hermite cubic (Dang 2013 Eqs. 30-31).
// Continuous in both, so a read narrower than one sample crosses sample edges without a step.
inline TrackReading ReadTrackSmooth(const RoughnessTrack &t, double pos) {
    const auto p = WrapTrackPos(t, pos);
    const size_t n = t.Heights.size();
    const float h0 = t.Heights[(p.Index + n - 1) % n], h1 = t.Heights[p.Index];
    const float h2 = t.Heights[(p.Index + 1) % n], h3 = t.Heights[(p.Index + 2) % n];
    const float c1 = 0.5f * (h2 - h0);
    const float c2 = h0 - 2.5f * h1 + 2.f * h2 - 0.5f * h3;
    const float c3 = 1.5f * (h1 - h2) + 0.5f * (h3 - h0);
    const float x = p.Frac;
    return {h1 + x * (c1 + x * (c2 + x * c3)), c1 + x * (2.f * c2 + x * (3.f * c3))};
}

// The track's boxcar mean over `window` samples centred on `pos`, and the slope of that mean.
inline TrackReading TrackBoxcar(const RoughnessTrack &t, double pos, float window) {
    const double half = 0.5 * double(window);
    const auto lo = WrapTrackPos(t, pos - half), hi = WrapTrackPos(t, pos + half);
    return {(TrackIntegral(t, hi) - TrackIntegral(t, lo)) / window, (TrackHeight(t, hi) - TrackHeight(t, lo)) / window};
}

// The same mean, following the reconstructed surface where the window narrows below one sample.
inline TrackReading ReadTrackSloped(const RoughnessTrack &t, double pos, float window) {
    return window <= 1.f ? ReadTrackSmooth(t, pos) : TrackBoxcar(t, pos, window);
}

// The height under each of `spots` cells spaced evenly across the contact window, scaled by `sigma` and accumulated into `out`.
// Each cell reads the track over its own filter of `sub_window` samples, following the reconstructed surface below one sample.
// The returned height and slope are the whole window's.
inline TrackReading ReadTrackSpots(const RoughnessTrack &t, double pos, float window, float sub_window, uint32_t spots, float sigma, float *out) {
    const double half = 0.5 * double(window);
    const double stride = double(window) / double(spots);
    const double sub = double(sub_window);
    for (uint32_t k = 0; k < spots; ++k) {
        const double centre = pos - half + (double(k) + 0.5) * stride;
        if (sub <= 1.0) {
            out[k] += sigma * ReadTrackSmooth(t, centre).Height;
            continue;
        }
        const float below = TrackIntegral(t, WrapTrackPos(t, centre - 0.5 * sub));
        const float above = TrackIntegral(t, WrapTrackPos(t, centre + 0.5 * sub));
        out[k] += sigma * float((above - below) / sub);
    }
    return TrackBoxcar(t, pos, window);
}

/***** Element springs: a surface tiled into contact springs *****/

// A surface tiled into elements, each with the non-linear contact spring its sub-element roughness implies (Andersson and Kropp 2008, J. Sound Vib. 318:296-312, section 2.6).
// An element's force is the bearing area below its crest: zero at first touch, stiffening as more summits bear, approaching the bulk at saturation.
// Elements interact only through the bulk.

// The roughness below the shortest wavelength a surface states, present inside every bearing contact.
// It is the second elastic energy of Pastewka et al. 2013 Appendix B: the microcontacts inside a bearing patch hold the load over a mean separation, and their compliance adds in series with the contact's (their Eq. B27).
struct SubCutoffRoughness {
    // Root-mean-square height of the band inside a contact one meter wide, m.
    // A contact of width w holds Amplitude * w^Hurst of it (Pastewka et al. 2013 Eq. B19).
    double Amplitude{0};
    double Hurst{0}; // Self-affine exponent the height is extrapolated along.
    double MaxWidth{0}; // Widest contact the band is read at, m. Above it the field's summits resolve the roughness.
};

// The band a self-affine finish holds below the shortest wavelength it states (Pastewka et al. 2013 Eqs. B2 and B19).
// `slope` is the one-dimensional spectral exponent p, whose Hurst exponent is -(1 + p)/2.
// Zero for a finish whose cutoff falls outside its correlation length, or whose exponent falls outside the self-affine range.
SubCutoffRoughness UnresolvedBand(double roughness, double correlation_length, double short_wavelength, double slope);

// Mean separation that band holds open inside a contact of width `width`, m.
// Persson's interface separation at the contact pressure, gamma times the roughness inside it (Pastewka et al. 2013 Eq. B17).
inline double SubCutoffSeparation(const SubCutoffRoughness &s, double width) {
    constexpr double PerssonGamma{0.4};
    return s.Amplitude > 0 ? PerssonGamma * s.Amplitude * std::pow(std::min(width, s.MaxWidth), s.Hurst) : 0.0;
}

// Bins the oblique-flank spectrum splits a population into, each holding half the depth of the one above it.
constexpr uint32_t FlankSpectrumBins{16};

// Which bin a contact of depth `depth` falls in under an engagement of `engagement`, counting halvings down from the engagement.
inline uint32_t FlankSpectrumBin(double depth, double engagement) {
    if (!(depth > 0) || !(engagement > depth)) return 0;
    return uint32_t(std::min(double(FlankSpectrumBins - 1), std::floor(std::log2(engagement / depth))));
}

struct ElementSprings {
    float Width{0}; // Along-track extent of one element, m
    float EngagementMax{0}; // Deepest engagement the curves are tabulated to, m
    uint32_t Knots{0}; // Engagement knots per element
    // Distinct curve sets, which the crests tile over.
    // Equal to the element count where nothing tiles.
    uint32_t Curves{0};
    float CrestRange{0}; // Spread of the crests the curves were gathered from, m.
    std::vector<float> Engagement; // Knots shared by every element, ascending from 0.
    std::vector<float> Crest; // Per element: height of its highest point, m.
    std::vector<float> Force; // Per element: force at each knot, N. Element-major, Knots per element.
    std::vector<float> Slope; // Per element: the interpolant's slope at each knot, N/m.
    std::vector<float> Potential; // Per element: the force curve's own integral at each knot, J.
    // Per element: the sum of its bearing summits' squared forces at each knot, N^2.
    // Slip redraws each junction independently, so an element's force fluctuates about the curve with this variance.
    std::vector<float> ForceVariance;
    // Per element: the oblique-flank moment of the summits bearing at each knot, N/m^2, accumulated summit by summit.
    std::vector<float> FlankMoment;
    std::vector<float> BearingCount; // Per element: how many of its summits bear at each knot, fractional between knots.
    // Per element: the widths those bearing contacts span, summed, m.
    // A contact's width is its Hertz stiffness over the modulus, dF/dd = 2 a E*.
    std::vector<float> BearingWidth;
    // The population's flank moment at each knot, split by how deep each summit's own contact sits.
    // A summit's breakaway stretch is a fixed multiple of its normal interference (Zhan and Huang 2018), so a bin's moment is the stiffness at that slider strength, which is the junction's Iwan density.
    // Knots * FlankSpectrumBins entries, knot-major.
    std::vector<float> FlankSpectrum;
    // The population's normal force at each knot, split into the same bins.
    // A bin's breakaway strength is its Coulomb cone over its stiffness, so force and moment together fix both.
    std::vector<float> FlankSpectrumForce;

    // Tallest crest in each aligned run of CrestBlock elements, so reads can skip runs that cannot reach the body.
    // A final run past the array's end wraps into its start.
    // RefreshCrestBlocks rebuilds it after any change to the crests.
    static constexpr uint32_t CrestBlock{64};
    std::vector<float> BlockCrest;
    // Last knot below each power of two of engagement, from KnotExponentFloor up.
    // Empty leaves reads bisecting the knots for the same index.
    int32_t KnotExponentFloor{0};
    std::vector<uint8_t> KnotByExponent;

    uint32_t Count() const { return uint32_t(Crest.size()); }
    // The curve set an element reads, its own where the array does not tile.
    uint32_t CurveOf(uint32_t element) const { return Curves > 0 ? element % Curves : element; }
};

// The share of the population's flank moment and normal force in each strength bin, read at the knot nearest `engagement`.
// Both sum to one over the occupied bins.
struct FlankSpectrumShares {
    std::array<float, FlankSpectrumBins> Moment{};
    std::array<float, FlankSpectrumBins> Force{};
};
FlankSpectrumShares FlankSpectrumShareAt(const ElementSprings &, float engagement);

// Force and stored energy of one element's spring at one engagement.
struct SpringRead {
    float Force{0}; // N
    float Potential{0}; // J
};

// The spring of `element` compressed by `engagement` meters below its crest, zero at and above the crest.
// Between knots the force is a monotone cubic (Fritsch-Carlson) and the potential is that cubic's integral, so Force is exactly dPotential/dEngagement.
SpringRead ReadElementSpring(const ElementSprings &, uint32_t element, float engagement);

// One element's slip-turnover force variance at `engagement` below its crest, N^2, linear between knots.
// Zero at and above the crest, like the force it fluctuates about.
float ElementForceVariance(const ElementSprings &, uint32_t element, float engagement);

// Rebuild the block maxima reads skip on. Call after any change to the crests.
void RefreshCrestBlocks(ElementSprings &);

// How far either side of a footprint an element can still reach the body, in elements.
// A curved body lifts away as the square of the distance, so the reach is where that lift exceeds the tallest crest's height above the deepest engagement.
uint32_t ElementReach(const ElementSprings &, double combined_curvature);

// The support envelope of a body over the springs: the body's height at first touch above each element, from its gap.
// One entry per element, at that element's centre.
std::vector<float> ElementEnvelope(const ElementSprings &, double combined_curvature, uint32_t reach);

// The same envelope over a bare crest array.
std::vector<float> CrestEnvelope(std::span<const float> crest, float width, double combined_curvature, uint32_t reach);

// The height at which the elements under a contact bear `load`, one entry per element, indexed as the envelope is.
// The array holds the mean height at first touch plus the surface excitation about it.
// Solved every eighth of the reach and interpolated between, since a footprint averages away anything shorter than itself.
std::vector<float> BearingDatum(const ElementSprings &, double combined_curvature, uint32_t reach, double load);

// The envelope at a fractional position along the elements, in element units, cyclically.
// Linear between element centres.
float EnvelopeAt(std::span<const float> envelope, double position);

// One side's crest deviations read at its slip behind the other, for a contact whose two surfaces move at different rates.
// The composite array is swept at the faster side's rate, and the slower side's share of the datum is read where that side stands.
// The correction moves the support envelope and every element's engagement together, never one element against another.
struct SlidingRegistration {
    std::span<const float> SlowCrestDev; // The slower side's zero-mean crest deviation, m, one per element. Empty applies none.
    double Slip{0}; // How far that side stands behind the array, in elements.
};

// The height the slower side's crest deviation adds at a fractional element position, zero without a registration to apply.
float Reregister(const SlidingRegistration &, double at);

// Total force and stored energy of the elements a body bears on, with the body's along-track gap applied at each element's distance from `position` (fractional, in element units).
// `engagement` is how far the body has descended below the envelope at that position, so zero is first touch and a body at or above it bears no force.
SpringRead ReadContactSprings(const ElementSprings &, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach, const SlidingRegistration &slide = {});

// ReadContactSprings, additionally accumulating each position bin's share of the force into `bin_force`, the bins slicing the reach window evenly along the track.
// `bin_engagement` deepens each bin's elements by its own body deflection, modulating the bin sums alone (empty applies none).
SpringRead ReadContactSpringBins(
    const ElementSprings &, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach,
    std::span<float> bin_force, std::span<const float> bin_engagement = {}, const SlidingRegistration &slide = {}
);

// The bins as engagement channels: every element is read at its bin's modulated engagement, and the returned totals are the sums of the per-bin forces and potentials.
// A body's local deflection then moves local force instead of the whole footprint.
SpringRead ReadContactSpringChannels(
    const ElementSprings &, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach,
    std::span<float> bin_force, std::span<float> bin_potential, std::span<const float> bin_engagement, const SlidingRegistration &slide = {}
);

// The engagement at which the springs carry `normal_force`, averaged over every element the contact could sit at.
float SolveSpringEngagement(const ElementSprings &, std::span<const float> envelope, double normal_force, double combined_curvature, uint32_t reach);

// The moving-load sweep accumulation: at every element position, the window of `2 * reach + 1` anchored forces centred there projected through the footprint's mode shapes.
// `field` holds one anchored force per element, read cyclically.
// `phi` is window-major, one row of `modes` shape projections per window slot.
// `table` receives the sums mode-major, `modes` rows of one column per element position.
void SweepModeDrives(std::span<const float> field, std::span<const float> phi, uint32_t modes, uint32_t reach, std::span<float> table);

// The bearing population's oblique-flank moment and normal stiffness at one engagement.
struct SpringFlankRead {
    // Sum of slope^2 / force over the bearing contacts, N/m^2, the population's whole contribution to oblique-flank micro-slip.
    // Each bearing contact dissipates (4/3) kt^2 (a sin theta)^3 / (mu F) per cycle of normal approach amplitude a, and the per-contact sum collapses to this moment.
    double Modulus{0};
    // Sum of slope over the same elements, N/m: each bearing contact's own stiffness in series with the roughness inside it.
    double Stiffness{0};
    double Bearing{0}; // Summits carrying load, which the moment runs over.
    double Width{0}; // Sum of those contacts' own widths, m.
};
SpringFlankRead SpringFlankMoments(const ElementSprings &, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach);

// Width the mean bearing contact spans, m, over the summits carrying load.
// A contact's incremental stiffness is its radius, dF/dd = 2 a E* for any axisymmetric shape (Pastewka and Robbins 2016).
// This is the length the flank tilt is read over: components longer than a contact tilt it, components shorter act as roughness inside it (Pastewka et al. 2013 Eqs. B6 and B18).
inline double SpringBearingWidth(const SpringFlankRead &read) {
    return read.Bearing > 0 ? read.Width / read.Bearing : 0.0;
}

// Mark the summits of a gap-folded height field: samples at or above every neighbour in their 8-neighbourhood, with columns wrapping and rows clamped.
// `folded` is row-major with `columns` along the track and `rows` across it.
std::vector<uint8_t> MarkFieldSummits(std::span<const float> folded, uint32_t columns, uint32_t rows);

// One summit's combined curvature, 1/m, from the second differences around it in the same gap-folded field.
// At a maximum both principal curvatures are negative and Hertz adds them, so this is minus the Laplacian.
// Columns wrap and rows clamp, and an edge row is read along the track alone and taken isotropic.
double FieldSummitCurvature(std::span<const float> folded, uint32_t columns, uint32_t rows, float column_spacing, uint32_t column, uint32_t row);

// The summits a contact bears on, gathered into the elements they stand in.
// One strip narrow enough to synthesize holds too few bearing asperities for a statistical share of the load, so a ribbon gathers several independent realizations of the same surface across the sweep.
struct ElementSummits {
    float Width{0}; // Along-sweep extent of one element, m.
    double InvModulus{0}; // 1/E* of the pair the summits were gathered for, which sizes their contacts.
    std::vector<float> Crest; // Highest point of each element over every realization, m. One per element.
    std::vector<uint32_t> Element; // Element each summit stands in.
    std::vector<float> Height; // Summit height under the body's gap, m.
    std::vector<float> Stiffness; // Hertz constant the summit's own curvature gives it, N/m^(3/2).
};

// Gather one realization's summits into `out`, tiling the field into elements `element_columns` wide.
// `heights` is row-major with `columns` along the track and `rows` across it, in meters, and `transverse_gap` is the body's gap at each row.
// `element_lift` raises each element bodily, one entry per element, for bands too long to vary within an element (empty applies none).
void GatherElementSummits(
    ElementSummits &out, std::span<const float> heights, uint32_t columns, uint32_t rows,
    std::span<const float> transverse_gap, std::span<const float> element_lift,
    uint32_t element_columns, float column_spacing, double inv_effective_modulus
);

// Merge one gathered population into another: the summits append and each element keeps the taller crest.
void MergeElementSummits(ElementSummits &into, const ElementSummits &from);

// The crest of each element over one side's field alone, gathered and merged as GatherElementSummits does, growing `out` to the element count.
void GatherElementCrests(
    std::vector<float> &out, std::span<const float> heights, uint32_t columns, uint32_t rows,
    std::span<const float> transverse_gap, std::span<const float> element_lift, uint32_t element_columns
);

// Build the springs of a gathered summit population.
// Each element's curve is the Hertz sum over its summits, each at the constant its curvature gives it, and every summit's contact is in series with `sub`.
// `engagement_max` bounds the tabulated engagement and `knots` sets the resolution, spaced geometrically from a sub-asperity toe.
ElementSprings BuildElementSprings(const ElementSummits &, const SubCutoffRoughness &, double engagement_max, uint32_t knots);

// Tile a height field into elements and build each one's spring.
// `heights` is row-major with `columns` along the track and `rows` across it, in meters.
// `transverse_gap` is the body's gap at each row (meters, one entry per row), which folds into the springs, so a set of springs holds for one contact curvature.
// `column_spacing` is the surface distance one column spans, which sizes an element.
ElementSprings BuildElementSprings(
    std::span<const float> heights, uint32_t columns, uint32_t rows, std::span<const float> transverse_gap,
    uint32_t element_columns, float column_spacing, double inv_effective_modulus,
    const SubCutoffRoughness &, double engagement_max, uint32_t knots
);

// One Gaussian spectral band a strip's field sums, described for the ensemble law below.
struct EnsembleBand {
    float Correlation{0};
    float Slope{0};
    float Cutoff{0};
    float Sigma{0}; // Height the unit-variance draw is scaled by, m.
};

// Build the springs from the population law of the summed bands rather than from realized summits.
// The per-knot sums are quadrature over the joint law of a lattice sample and its eight neighbours, so every element reads the one resulting curve (Curves is one).
// The crests stay realized and the curves anchor to their mean: the crest summit enters at engagement zero with the population's expected curvature at that height.
// The body's transverse gap shifts each row's bearing plane and adds the body's curvature to every summit's, and the interior law also covers the strip's first and last rows.
// Measured-profile bands are not Gaussian, and take the realized build.
ElementSprings BuildEnsembleSprings(
    std::span<const EnsembleBand> bands, float spacing, uint32_t columns, uint32_t rows,
    std::span<const float> transverse_gap, double body_curvature, uint32_t element_columns,
    uint32_t strip_realizations, double inv_effective_modulus,
    const SubCutoffRoughness &, double engagement_max, uint32_t knots, std::vector<float> crests
);

/***** Interface junctions (Mindlin 1949, Mindlin and Deresiewicz 1953) *****/

// Tangential-to-normal stiffness of a contact patch, Mindlin's 2*(1-nu)/(2-nu) near nu = 0.3.
constexpr double MindlinShearRatio{0.82};

// Mean cube of the sine of a surface's flank tilt, for tilts arctan(x) over a Gaussian slope x of rms `slope`.
// The oblique-flank loop area depends on the tilt only through this moment.
double FlankSineCube(double slope);

// The oblique-flank junction's Dahl stiffness, N/m: one element for the whole bearing population's tilted flanks, at its slope^2/force moment and the load it bears.
inline double FlankJunctionStiffness(double sine_cube, double normal_force, double flank_modulus) {
    return MindlinShearRatio * std::sqrt(sine_cube * normal_force * flank_modulus);
}

// Jenkins elements the junction is discretised over, at most one per bin of the bearing population's strength spectrum.
constexpr uint32_t MaxFlankBins{16};

// One Jenkins element of that spread: the stiffness its share of the population presents and the share of the contact's Coulomb cone it breaks away at.
struct FlankJunctionBin {
    float Stiffness{0}; // N/m
    float ConeShare{0};
};

// Discretise the junction over the bearing population's spectrum, writing the occupied bins into `out` and returning how many.
// Each bin takes its share of the contact's normal force and flank moment through the same law the whole junction uses.
// The bins' slip forces sum to the whole cone exactly (Segalman 2005 Eqs. 51 to 53), and their stiffnesses sum to less than the single element's, so a spread junction is softer at full stick.
uint32_t FlankJunctionSpread(
    double sine_cube, double normal_force, double flank_modulus,
    std::span<const float> moment_share, std::span<const float> force_share, std::span<FlankJunctionBin> out
);

/***** Conformal contact (Persson 2007) *****/

// A face resting on a face bears its load over an area geometry fixes, so its stiffness comes from the roughness flattening against the counterface rather than from a radius.

// Pressure the asperities bear the load at, p = E*|grad h|_rms/2, in Pa.
// Independent of both load and area, which makes the real contact area proportional to load.
// `profile_slope_rms` is the gradient along one cut of the surface, and an isotropic surface's full gradient has twice that variance.
double AsperityPressure(double inv_effective_modulus, double profile_slope_rms);

// Area the asperities actually touch over, in m^2, across every regime a contact can be in (Pastewka and Robbins 2016, arXiv:1508.02154, Eq. 4):
//
//     A = A0 * erf(sqrt(pi) * N / (2 * A0 * p))
//
// `bound_area` is A0, the region the contact is confined to: the Hertz patch a point or an edge grows under load, or the polygon two faces share, whichever is smaller.
// Light load linearizes to A = N/p and full contact saturates at A0.
double RealContactArea(double normal_force, double bound_area, double inv_effective_modulus, double profile_slope_rms);

// Hurst exponent H = -(1 + p)/2 from the one-dimensional roughness power-spectrum exponent p, held inside the open interval the separation theory is defined on.
double HurstExponent(double spectral_slope);

// Roll-off wavevector q0 = 2*pi/l, in rad/m, from the correlation length in meters.
double RollOffWavevector(double correlation_length);

// Persson's dimensionless coefficients, set by the Hurst exponent and the roughness band ratio q1/q0.
// Alpha fixes the separation the pressure decays over, u0 = roughness / Alpha, and Beta its full-contact level.
struct SeparationCoefficients {
    double Alpha{2.5}, Beta{0.5};
};

// The one fitted constant in Persson 2007 Eq. 15, which scales Alpha as 1/gamma.
// His figures are drawn at gamma = 1, while fits against numerical contact solutions put the decay length near u0 = 0.4 * roughness.
inline constexpr double SeparationGamma = 0.4;

// Band ratio q1/q0 the coefficients are integrated over, standing for the decades of roughness above a finish's correlation length.
// Alpha moves under 2% across four decades of it, so it is a constant.
inline constexpr double RoughnessBandRatio = 1e3;

// Alpha and Beta from Persson 2007 Eqs. 15 and 17. Pass gamma = 1 to reproduce his Fig. 2.
SeparationCoefficients PerssonSeparationCoefficients(double hurst, double band_ratio, double gamma = SeparationGamma);

// Pressure a nominally flat rough contact bears at mean separation u, Persson 2007 Eq. 18:
// p = Beta*q0*h_rms*E* * exp(-Alpha*u/h_rms), in Pa.
double ConformalPressure(double separation, double roughness, double roll_off_wavevector, double inv_effective_modulus, const SeparationCoefficients &);

// Mean interfacial separation under pressure, Persson 2007 Eq. 19, in meters.
// Zero once the pressure reaches the full-contact value, where the surfaces have flattened.
double ConformalSeparation(double pressure, double roughness, double roll_off_wavevector, double inv_effective_modulus, const SeparationCoefficients &);

// Standard deviation of the Gaussian pressure profile a rough curved contact bears its load over once roughness dominates, in meters (Tiwari and Persson 2020, s^2 = R*u0 with u0 = roughness/Alpha).
// Zero curvature is a flat face, whose patch is its nominal area.
double ConformalPatchWidth(double combined_curvature, double roughness, const SeparationCoefficients &);

// Normal stiffness dN/du of a conformal contact under load N, in N/m.
// Eq. 18 is exponential in separation, so this is Alpha*N/h_rms, linear in load where Hertz is a 3/2 power.
double ConformalStiffness(double normal_force, double roughness, const SeparationCoefficients &);

/***** The asperity bed (Greenwood and Williamson 1966) *****/

// Two nominally flat rough faces bear on discrete asperities rather than over the whole patch they share.
// Persson's exponential pressure-separation relation is the mean of that, and holds where the patch spans many roll-off lengths.
// A patch approaching one roll-off length bears on the first asperities to touch and responds as Hertz does (Pastewka et al. 2013, arXiv:1210.4635), the regime these patches are in, so each spot takes Hertz's law.

// Most spots a bed resolves.
inline constexpr uint32_t MaxBedSpots = 32;

// Expected value of max(lambda + z, 0)^power over a standard normal z, Greenwood and Williamson's integral.
// Power 3/2 gives the load a bed of Hertzian spots bears at separation lambda (in units of the spread of spot heights), 1/2 its stiffness, and 1 the area it touches over, a Hertz spot's area being pi*R*delta.
double BedHeightIntegral(double lambda, double power);

// Separation, in units of the spread of spot heights, at which a bed of `spot_count` spots of stiffness `spot_stiffness` (N/m^(3/2)) and height spread `height_rms` (m) bears `normal_force`.
// Negative where the load rests on the peaks alone, which is the ordinary case at light load.
// `power` is the spot law's exponent in approach, 3/2 for Hertz and 1 for a linear spot.
double SolveBedSeparation(double normal_force, double spot_count, double spot_stiffness, double height_rms, double power = 1.5);

// BedHeightIntegral at the two powers a cell needs every sample, read off a table.
// Below the table nothing bears, and above it every spot bears and the integral is its argument's power.
float BedLoadFactor(float lambda);

// Variance of the load one asperity bears at separation `lambda`, in the same units BedLoadFactor gives its mean.
// The spots bear independently, so a sum of n of them has n times this variance.
float BedVarianceFactor(float lambda);

// The load a clamped linear spot bears, at exponent one instead of Hertz's 3/2.
float BedLinearLoadFactor(float lambda);

// The elastic potential a cell's mean law stores: the load factor integrated over lambda from the table's low edge, per spot law.
float BedPotentialFactor(float lambda);
float BedLinearPotentialFactor(float lambda);

// The Hertz mean law and its potential at absolute engagement `engagement` (m) over a cell whose asperity heights spread `spread` (m) about their mean: spread^p * Factor(engagement / spread), with the beyond-table branches evaluated in meters.
// Continuous down to and including spread = 0, where both are the bare Hertz cell's own law and potential.
float BedLoadAt(float engagement, float spread);
float BedPotentialAt(float engagement, float spread);

// The bed of Hertz spots two rough faces bear on across the area they share.
// The patch holds more asperities than the track resolves along its length, so a rendered spot is a cell: the asperities across the patch at one position along the sweep, sharing the local mean height the track gives and spread about it at their own scale.
struct AsperityBed {
    uint32_t SpotCount{0}; // Cells rendered along the track
    double TotalSpots{0}; // Asperities bearing across the whole patch, which grows with the load
    double SpotWeight{0}; // Asperities one rendered cell stands for, TotalSpots over SpotCount
    double SpotStiffness{0}; // k of one asperity, N/m^(3/2)
    double SpotRadius{0}; // Radius of curvature at one asperity, m
    double HeightRms{0}; // Spread of gap height at the asperity scale, m
    double CellSpread{0}; // Spread of asperity heights about their cell's local mean, m
    double Separation{0}; // Separation the bed carries its load at, in units of HeightRms
};

// Load the bed bears, N, averaged over the height distribution its spots are drawn from.
double BedLoad(const AsperityBed &);

// Normal stiffness dN/du of the bed at its separation, N/m. Its spots resist in parallel, so they add.
double BedStiffness(const AsperityBed &);

// Area the bed's spots touch over, m^2. A Hertz spot of radius R pressed to depth d touches over pi*R*d.
double BedContactArea(const AsperityBed &);

// Share of a surface excursion the bed takes, the bulk beneath it taking the rest.
// The asperities are a compliant layer over the bulk under the same force, so the two add in series (Pastewka et al. 2013).
// Near one where the geometry fixes a wide patch, and well below one where the load presses out a small one.
double BedSurfaceShare(double inv_effective_modulus, double bound_area, const AsperityBed &);

// Cells rendered along the region: one per asperity site the region holds, capped at MaxBedSpots.
inline uint32_t BedCellCount(double patch_width, double region_width, double peak_density) {
    return uint32_t(std::clamp(std::max(region_width, patch_width) * peak_density, 1.0, double(MaxBedSpots)));
}

// The width a cell's datum is measured over: the contact's bearing share of the sweep, floored at the pair's half-variance width, and never wider than the stretch the cell's asperities span.
// The cap lets a sparse cell follow geometry its asperities cannot bridge.
inline double BedCellWindow(double patch_width, double region_width, double peak_density, double half_variance_width) {
    const auto cells = BedCellCount(patch_width, region_width, peak_density);
    const double share = patch_width / cells;
    const double across = std::max(patch_width * peak_density, 1.0);
    const double population_span = peak_density > 0 ? across * across / (cells * peak_density) : half_variance_width;
    return std::max(share, std::min(half_variance_width, population_span));
}

// The bed a contact of `patch_width` meters bears its load on.
// `region_width` is the extent of the region confining the contact along the sweep, which the rendered cells sample, while the bearing population the patch holds divides among them as their weight.
// `peak_density` is how densely asperities stand along the profile (1/m), which fixes their number.
// `cell_window` is the width (m) a cell's local mean height is measured over: geometry wider than it moves the cell's datum, geometry narrower stays in the cell's spread for the force law to carry.
// `spot_curvature(width)` and `spot_height_rms(width)` give the gap's curvature (1/m) and height spread (m) under a patch that wide.
// A spot bears at the curvature its surface has at the shortest wavelength that surface resolves, where Pastewka and Robbins 2016 Eq. (6) reads the asperity radius.
AsperityBed ResolveAsperityBed(double normal_force, double patch_width, double region_width, double inv_effective_modulus, double peak_density, double cell_window, auto &&spot_curvature, auto &&spot_height_rms) {
    // Asperities stand across the patch as well as along it, so the patch holds the square of what one cut of it crosses.
    const double across = std::max(patch_width * peak_density, 1.0);
    const double total = across * across;
    // The cells sample the sites along the region, so a cell's weight can fall below one asperity.
    const auto cells = BedCellCount(patch_width, region_width, peak_density);
    // One asperity spans the peak spacing.
    const double asperity_width = peak_density > 0 ? 1 / peak_density : patch_width;
    const double curvature = std::max(spot_curvature(asperity_width), 1e-6);
    const double height_rms = spot_height_rms(asperity_width);
    const double cell_mean_rms = std::min(spot_height_rms(cell_window), height_rms);
    AsperityBed bed{
        .SpotCount = cells,
        .TotalSpots = total,
        .SpotWeight = total / cells,
        .SpotStiffness = ContactStiffness(inv_effective_modulus, curvature),
        .SpotRadius = 1 / curvature,
        .HeightRms = height_rms,
        // What a cell's asperities keep once the height shared by the whole cell is removed.
        .CellSpread = std::sqrt(std::max(height_rms * height_rms - cell_mean_rms * cell_mean_rms, 0.0)),
    };
    bed.Separation = SolveBedSeparation(normal_force, bed.TotalSpots, bed.SpotStiffness, bed.HeightRms);
    return bed;
}

// Regularizer on the approach speed an engagement's Hunt-Crossley constant is sized from.
// The divisor is floored at Chi * dt^2, which vanishes with the time step so the constant converges to the continuous law (van Walstijn et al. 2024, Eq 38).
inline constexpr float ImpactVelocityChi = 1e7f;

/***** Sustained contacts: what one voice carries and renders *****/

// Sample points a contact reads its mode shapes from, barycentric over a triangle of the sample surface.
struct SamplePointBlend {
    std::array<uint32_t, 3> Points{};
    vec3 Weights{1, 0, 0};
};

// Position bins a spring contact's footprint is sliced into, each driving the modes at its own position.
// Local forces fluctuate while their sum stays near constant, so distributed forcing excites modes the total cannot.
constexpr uint32_t MaxSpringBins{32};
// Where bin `bin` of `bins` sits along the sweep, m from the footprint's centre.
// Each bin sits at the centre of its even slice of the footprint, which is also the lever arm its force acts through.
inline float SpringBinOffset(uint32_t bin, uint32_t bins, float half_extent) {
    return bins > 0 ? ((float(bin) + 0.5f) / float(bins) - 0.5f) * 2 * half_extent : 0.f;
}
// Points the relaxation channel records the contact's growing edge with.
// At this depth the channel reads within 2 percent of its closed form from 100 Hz up.
constexpr uint32_t MaxRelaxEdges{64};

// One surface track under a contact.
struct ContactTrack {
    int32_t Index{-1}; // Surface track pool slot, -1 for an unused one.
    float Rate{0}; // Track samples advanced per output sample, from that surface's own sweep.
    float Sigma{0}; // Height scale applied to the track, m.
    float Window{0}; // Extent the contact reads over the track, in track samples.
    // Width of the contact filter one bed cell reads at its own position along the window, in track samples.
    // Zero for a track read as one mean over the whole window.
    float SubWindow{0};
    float Step{0}; // Distance along the surface per output sample, m.
    float Spacing{0}; // Distance along the surface between track samples, m.
};

// KHR_audio_rigid_bodies contact state a sustained voice renders, plus the constants derived for its force model.
struct SustainedState {
    SamplePointBlend Blend{};
    vec3 N{0}; // Node-local unit normal, directed into the object.
    vec3 SlipDir{0}; // Node-local unit slip direction. Zero when nothing slides.
    // Node-local direction each surface's geometric force drives this object, in the contact's surface order.
    // Signed so that the two objects of one contact are driven apart rather than together.
    std::array<vec3, 2> SweepDir{};
    float NormalForce{0}; // N, the load the excitation fluctuates about.
    float Friction{0}; // Combined friction coefficient.
    float SlipSpeed{0}; // m/s, the pair's relative tangential speed. Zero when nothing slides.
    // N, the tangential force the solver applied to this body along SlipDir, which the stiction channel's fluctuation is measured about.
    float SolverFriction{0};
    float Stiffness{0}; // k, N/m^(3/2). Zero for a contact two faces fix the area of, which bears on a bed of asperities.
    // The approach the contact bears its load at, m.
    // Hertz's delta0 for a point contact, and for a bed the separation its spots bear the load at, negative wherever the load rests on the peaks alone.
    float StaticPenetration{0};
    // Cells of the asperity bed rendered along the track, from the density of peaks over the contact's width.
    // Positive selects the bed over the single Hertz spring a point contact takes.
    uint32_t SpotCount{0};
    float SpotWeight{0}; // Asperities one cell stands for, the patch being wider than the track's one cut.
    float SpotStiffness{0}; // k of one asperity, N/m^(3/2), from the radius of curvature at one.
    float CellSpread{0}; // Spread of asperity heights about their cell's local mean, m.
    float SurfaceCompliance{1}; // Share of a surface excursion the bed takes, the bulk taking the rest.
    // Hunt-Crossley dissipation, dimensionless: 1.5 * (1 - restitution).
    // Read as the damping constant in s/m at a 1 m/s approach, and under ETA_LANDING as the numerator each engagement's landing speed divides.
    float DampingFactor{0};
    // N/m, the patch shear spring the frictional force acts through (interfacial stiffness at Mindlin's shear ratio).
    // Zero binds the junction to the contact point rigidly.
    float ShearStiffness{0};
    // N/m, the oblique-flank micro-slip spring the normal channel acts through: the bearing population's tilted flanks collapsed to one Dahl element on the approach.
    // Zero leaves the normal channel purely viscous.
    float FlankStiffness{0};
    // The same junction with the bearing population's spread of breakaway strengths, one Jenkins element per occupied bin of its spectrum.
    // A single strength dissipates as the cube of the amplitude whatever the drive, where the spread reads between 2.5 and 3 as measured joints do (Segalman 2005).
    // Zero bins run the single element above.
    uint32_t FlankBins{0};
    std::array<FlankJunctionBin, MaxFlankBins> FlankBin{};
    // N/m^2, the relaxation channel's coefficient: (G*/E*) / 2 times the contact's second force derivative d2F/dd2 (Popov, Popov and Pohrt 2015 Eqs. 11 and 12).
    // It scales the energy carried away by each annulus a shrinking contact drops.
    // Zero leaves the channel inert.
    float RelaxScale{0};
    // Contact-spring pool slot (CONTACT_SPRINGS), where the contact bears on per-element springs instead of the statistical bed's cells.
    // Negative leaves the bed path in place.
    int32_t SpringIndex{-1};
    float SpringRate{0}; // Elements the contact advances per output sample.
    // Elements the slower surface falls behind the array per output sample, and which side that is.
    // The array is swept at the faster surface's rate, so this is zero wherever the two advance together.
    float SlipRate{0};
    uint32_t SlowSide{0};
    float SpringScale{1}; // This side's force scale on the shared springs, from the stiffness cap.
    // The distributed-excitation bins, each slice of the footprint driving the modes at its own position as a zero-sum fluctuation about its mean share of the contact force.
    uint32_t SpringBins{0}; // Position bins across the footprint. Zero leaves the channel off.
    std::array<SamplePointBlend, MaxSpringBins> SpringBinBlend{};
    // Half the footprint's extent along the sweep, m: where the outermost bin centre sits and the lever arm the bins' forces act through.
    float SpringHalfExtent{0};
    // kg^-1 m^-2 about the contact's pitch axis, which the normal and the slip direction span.
    // The separation across the footprint varies linearly with the tilt (Hess and Soom 1992, Eq. 6c), and the bins' forces turn the body about this axis.
    // Zero holds the body flat, which turns the channel off.
    float InverseAngularInertia{0};
    // The moving-load sweep channel's pool slot, for a side whose footprint is fixed on its surface.
    // Negative leaves the channel off.
    int32_t SweepIndex{-1};
    // Root of the footprint's slip-turnover force variance at the anchor, N, before SpringScale.
    // Slip redraws each bearing junction independently, so this fluctuation is incoherent across the footprint.
    float NoiseRms{0};
    // The footprint's bearing stiffness at the same anchor, N/m after the side's scale, which converts the turnover pool between force and engagement units.
    float NoiseStiffness{0};
    // The turnover shot process's geometry: junctions stand a Rice peak spacing apart along the sweep, engage over their Hertz patch diameter, and persist for the footprint's length, all m.
    float JunctionSpacing{0};
    float JunctionTransit{0};
    float FootprintLength{0};
    // Each surface's microscale finish, then its mesoscale relief.
    // A track's surface is its index's low bit, matching SweepDir, and both sides of a contact list them in the same order.
    static constexpr uint32_t TrackCount{4};
    std::array<ContactTrack, TrackCount> Tracks{};
    // The pair's turnover noise track, the bed's asperity site population, whose samples the cells read as surface-fixed heights.
    ContactTrack Turnover{};
};

// The state a sustained voice keeps from one sample to the next.
struct SustainedCarry {
    std::array<double, SustainedState::TrackCount> Pos{};
    double TurnoverPos{0};
    double SpringPos{0}; // Position along the element springs, in elements. Pair-level, so both voices read one gap.
    double SlipPos{0}; // How far the slower surface stands behind the array, in elements.
    double NoisePos{0}; // Position along the fast side's finish track the slip-turnover noise reads, in track samples.
    std::array<float, MaxSpringBins> NoiseMean{}; // Each turnover read's sub-audio mean, the channel's DC blocker.
    // The turnover shot process per bin: the junction pool, its patch-transit smoothing, and a deterministic copy driven by the expected arrival rate, subtracted to leave the fluctuation zero-mean.
    // One RNG stream serves the voice, seeded so both sides of a pair draw one event train.
    std::array<float, MaxSpringBins> NoisePool{};
    std::array<float, MaxSpringBins> NoiseSmooth{};
    std::array<float, MaxSpringBins> NoiseExpectedPool{};
    std::array<float, MaxSpringBins> NoiseExpectedSmooth{};
    uint64_t NoiseRng{0};
    // The engagement injection each bin reads, m, held one sample so its work is metered with the rest.
    // The scalar pair serves an unbinned voice.
    std::array<float, MaxSpringBins> NoiseEngageBin{};
    float NoiseEngage{0};
    float NoiseEngagePrev{0};
    float ReliefMean{0};
    float SpringDatum{0}; // The envelope height the contact has settled about, m.
    double PrevSpringPos{0}; // The position along the springs at the previous sample, in elements.
    double PrevSlipPos{0}; // The slip the slower surface stood at the previous sample, in elements.
    // Each bin's slow mean share of the contact force, seeded from the first sample's shares so the shares sum to one and the fluctuations to zero.
    std::array<float, MaxSpringBins> SpringShareMean{};
    // The bin channels' exchange state: per-bin psi, the previous sample's deflection read at the bin's place, and the previous engagement for the damping's relative rate.
    std::array<float, MaxSpringBins> PsiBins{};
    std::array<float, MaxSpringBins> PrevBinDefl{};
    std::array<float, MaxSpringBins> RawApproachBins{};
    // The nominal gradient each channel last bore at, which is the rate a parted channel's residual psi drains at.
    // Index 0 holds the scalar channel's.
    std::array<float, MaxSpringBins> LastBearingG{};
    bool BinsPrimed{false}; // Set once the bin channels seed their psi, so a voice gaining bins mid-life seeds cleanly.
    // The body's rigid tilt about the contact's pitch axis, rad, and its rate.
    // The bins' solved reactions turn the body about its centre of mass, and the tilt deepens each bin by its own lever arm.
    float Tilt{0};
    float PrevTilt{0}; // The tilt at the previous sample, so its work is metered the same way.
    float TiltRate{0}; // rad/s
    float TiltMoment{0}; // N*m, the previous sample's reactions about the pitch axis.
    // The orientation the contact has settled about, rad.
    // The bins read their tilt about this mean, so a standing imbalance in their shares causes no lasting rotation.
    float TiltMean{0};
    // Local mean height under each spot of the bed, seeded per spot so a voice starting mid-surface sits at equilibrium on every spot at once.
    std::array<float, MaxBedSpots> SpotMean{};
    // Which cells bore at the previous sample, one bit each, and the largest load each carried while bearing.
    // Maintained only under CONTACT_REPORT.
    uint32_t SpotBearing{0};
    std::array<float, MaxBedSpots> SpotPeak{};
    // The approach at the previous sample, so the damping's relative rate measures one sample's true motion.
    float RawApproach{0};
    // The Hunt-Crossley damping constant of the current engagement, s/m, sized while the contact is open and held for as long as it bears.
    float EngagementDamping{0};
    float PrevDeflection{0}; // The modal deflection read at the previous sample, m, so the datum's motion is separable.
    // How far the body has travelled along this contact's normal, m.
    float RigidNormal{0};
    float PrevRigidNormal{0}; // The same, as of the previous sample, so the body's travel is separable.
    // The body's travel along the slip direction and the previous sample's tangential state, from which the stiction solve measures the contact point's motion relative to the sliding surface.
    float RigidTangent{0};
    float PrevRigidTangent{0};
    float PrevTangentDefl{0};
    float PatchShear{0}; // The patch shear spring's stored stretch, m, carried on half-steps.
    // The same tangential state along the transverse direction (normal cross slip), the second in-plane axis of a spring voice's interface junction.
    float RigidTransverse{0};
    float PrevRigidTransverse{0};
    float PrevTransverseDefl{0};
    float PatchShearTransverse{0};
    // The oblique-flank junction's stored stretch, m, on the normal channel's own displacement.
    float PatchFlank{0};
    // Each Jenkins element's stretch where the junction holds a spread of strengths, m.
    // They share one displacement and one contact compliance, so they differ only in when each breaks away.
    std::array<float, MaxFlankBins> PatchFlankBin{};
    // The relaxation channel's memory of the contact's growing edge, an ascending polyline of engagement points: the approach each point engaged at and the tangential position it engaged with, interpolated between neighbours.
    // A shrinking contact drops the annuli above its new depth, each taking away the shear it took on since it engaged.
    std::array<float, MaxRelaxEdges> RelaxApproach{};
    std::array<float, MaxRelaxEdges> RelaxTangent{};
    uint32_t RelaxEdges{0};
    // The approach and tangential position the channel has integrated to, m, both measured from the point the contact last shrank past everything on record.
    float RelaxDepth{0};
    float RelaxTangentPos{0};
    bool PrevStick{false}; // Whether the previous sample took the stick branch, for the transition count.
    bool Primed{false}; // Set once the local mean is seeded from the first sample's relief height.
    // Set when a published state moved this voice's anchors farther than its engagement range resolves.
    // The next sample re-seeds the means and anchors, keeping the track positions and the body's travel, so the swap emits no force.
    bool Rebase{false};
    float Psi{0}; // The voice's auxiliary variable sqrt(2 * stored contact energy).
    float PrevRelief{0};
    std::array<float, MaxBedSpots> PrevSpotHeight{};
    bool PrevContact{false};
};

// The energy a shrinking contact releases with the annuli it drops, J, over one step of the contact's approach and of its tangential position (Popov, Popov and Pohrt 2015).
float RelaxationRelease(const SustainedState &, SustainedCarry &, float approach_step, float tangent_step);
// Clear the channel's history, which no longer applies once a contact parts or moves its datum.
void ResetRelaxation(SustainedCarry &);

// The oblique-flank junction's force on the normal channel over one approach step, N, positive opposing the approach.
float FlankJunctionStep(const SustainedState &, SustainedCarry &, float load, float step, float cn);

/***** The pools a voice reads: surface tracks, element springs, sweep tables *****/

// Every sustained contact of one main-thread frame, published whole.
// A contact missing from the newest set has ended.
struct VoiceSet {
    struct Voice {
        uint64_t Id; // Contact identity, which carries a voice's state across frames.
        uint32_t Object; // Bank object slot, valid only against the bank live when this was published.
        SustainedState State;
    };

    uint64_t Frame{0}; // The frame this was built in.
    std::vector<Voice> Voices;
};

// One entry in a pool, shared by everything whose content hashes to the same key.
// The audio thread loads Live without synchronizing, and the main thread repoints a slot only once no voice reads it.
template<typename T> struct PoolSlot {
    std::atomic<const T *> Live{nullptr};
    std::shared_ptr<const T> Owned; // Main thread only. Shared with whatever else holds the value.
    uint64_t Key{0}; // Content key of Owned. Main thread only.
};

// One track in the pool, shared by every surface whose finish or relief hashes to the same content.
using SurfaceTrackSlot = PoolSlot<RoughnessTrack>;

// The element springs a contact reads, with the reach and envelope of the curvature they were built for.
struct ContactSpringSet {
    ElementSprings Springs;
    std::vector<float> Envelope; // Element crests under the contact's own gap, m. One entry per element.
    std::vector<float> AnchorForce; // Mean stack force over sampled contact centres at each engagement knot, N.
    double Curvature{0}; // The bucketed combined curvature the set was built for. Zero for a face.
    uint32_t Reach{0}; // Elements either side of centre the body can reach.
    // Across-sweep extent the elements were tiled over, m.
    // A contact wider than one strip scales these springs by the number of strips its bearing width holds.
    float StripWidth{0};
    // The band below the pair's stated cutoffs, in series with every contact these springs make.
    SubCutoffRoughness SubCutoff;
    // Each side's crest deviation from its own field alone, m, one entry per element, zero mean.
    // Empty leaves the composite crests unchanged.
    std::array<std::vector<float>, 2> SideCrestDev;
};

// One ribbon's strip-level product: the springs built from the gathered summits and each side's crests, before the long bands and the footprint's span are applied.
// Independent of the footprint and load buckets.
struct SpringStrips {
    ElementSprings Springs;
    std::array<std::vector<float>, 2> SideCrests;
};

// The bearing population's flank moments over a whole set seated at `anchor`, reading every `stride`th element.
// The moments are sums, so `Sampled` is the divisor for a per-element mean.
struct SpringFlankTotals {
    SpringFlankRead Moments;
    uint32_t Sampled{0};
};
inline SpringFlankTotals SpringFlankSum(const ContactSpringSet &set, float anchor, uint32_t stride) {
    SpringFlankTotals out;
    for (uint32_t e = 0; e < set.Springs.Count(); e += stride, ++out.Sampled) {
        const auto read = SpringFlankMoments(set.Springs, set.Envelope, double(e) + 0.5, anchor, set.Curvature, set.Reach);
        out.Moments.Modulus += read.Modulus;
        out.Moments.Stiffness += read.Stiffness;
        out.Moments.Bearing += read.Bearing;
        out.Moments.Width += read.Width;
    }
    return out;
}

// What a set seated at `engagement` bears, summed over every element.
// Each element is read at the engagement the anchor leaves it under its envelope, the stiffness is that read's slope, and the variance is the slip turnover of the bearing ones.
// Force and stiffness take the side's spring `scale`, and `field_out`, where it has room, takes each element's scaled force.
struct AnchoredSums {
    double Force{0}, Variance{0}, Stiffness{0};
};
inline AnchoredSums AnchoredElementSums(const ContactSpringSet &set, double engagement, double scale, std::span<float> field_out) {
    const auto &springs = set.Springs;
    const float slope_step = std::max(springs.EngagementMax / 4096.f, 1e-9f);
    AnchoredSums out;
    for (uint32_t e = 0; e < springs.Count(); ++e) {
        const float eng = float(engagement - (double(EnvelopeAt(set.Envelope, double(e) + 0.5)) - double(springs.Crest[e])));
        const float force = ReadElementSpring(springs, e, eng).Force;
        const float scaled = float(scale) * force;
        if (e < field_out.size()) field_out[e] = scaled;
        out.Force += double(scaled);
        if (force > 0) {
            out.Stiffness += scale * double(ReadElementSpring(springs, e, eng + slope_step).Force - force) / double(slope_step);
            out.Variance += double(ElementForceVariance(springs, e, eng));
        }
    }
    return out;
}

// The resting anchor: the stack force at each engagement knot averaged over sampled contact centres.
// Every `stride`th element stands for the population.
inline void FillAnchorForce(ContactSpringSet &set, uint32_t stride = 1) {
    const auto knots = uint32_t(set.Springs.Engagement.size());
    const auto count = set.Springs.Count();
    set.AnchorForce.resize(knots);
    for (uint32_t j = 0; j < knots; ++j) {
        double sum = 0;
        uint32_t n = 0;
        for (uint32_t e = 0; e < count; e += stride, ++n) {
            sum += ReadContactSprings(set.Springs, set.Envelope, double(e) + 0.5, set.Springs.Engagement[j], set.Curvature, set.Reach).Force;
        }
        set.AnchorForce[j] = n > 0 ? float(sum / n) : 0.f;
    }
}

// The engagement at which the set's anchored mean force reaches `normal_force`, interpolated in the anchor table and continued past its last knot at the final slope.
inline float SolveSpringAnchor(const ContactSpringSet &set, double normal_force) {
    const auto &z = set.Springs.Engagement;
    const auto &f = set.AnchorForce;
    const size_t n = f.size();
    if (n < 2 || normal_force <= 0) return 0.f;
    if (const auto it = std::lower_bound(f.begin(), f.end(), float(normal_force)); it != f.end()) {
        const auto hi = size_t(it - f.begin());
        if (hi == 0) return z[0];
        const float df = f[hi] - f[hi - 1];
        return df > 0 ? z[hi - 1] + (z[hi] - z[hi - 1]) * float((normal_force - f[hi - 1]) / df) : z[hi - 1];
    }
    const float slope = z[n - 1] > z[n - 2] ? (f[n - 1] - f[n - 2]) / (z[n - 1] - z[n - 2]) : 0.f;
    return slope > 0 ? z[n - 1] + float((normal_force - f[n - 1]) / slope) : z[n - 1];
}

// One spring set in the pool, shared by every contact whose pair spectra and curvature bucket hash to the same key.
using ContactSpringSlot = PoolSlot<ContactSpringSet>;

// The moving-load sweep channel of one (spring set, object) pair: each mode's drive from the anchored element forces swept through its shape, tabulated over the spring position.
// The element forces are static on the surface while their positions in the body's frame sweep at the sliding speed, so mode m at frequency w is driven by the force field's spatial component at wavenumber w over speed.
struct SweepTableSet {
    uint32_t Modes{0}; // Rows, the object's tuned mode count when the table was built.
    uint32_t Positions{0}; // Columns, one per element along the springs.
    std::vector<float> Table; // Mode-major, Modes rows of Positions, N per unit mass-normalized shape.
    float ForceTotal{0}; // The anchored footprint force the table was built at, N.
    // The footprint's slip-turnover force variance at the same anchor, N^2, before the side's spring scale, which enters it linearly as a count of independent strips.
    float NoiseVariance{0};
    // The footprint's bearing stiffness at the same anchor, N/m after the side's spring scale, which converts the turnover pool between force and engagement units.
    float NoiseStiffness{0};
    // Per row: the bearing elements' mean stiffness projected through the mode's squared shape across the footprint, 1/s^2 at the baked scale.
    // The elements the drive sweeps also push back on the locally deflected body, so the read scales each row by w^2 / (w^2 + projection), the conformity gain.
    // Applied at read with the current mode frequency and world scale.
    std::vector<float> ModeStiffness;
};

// One sweep table in the pool, one per (spring set, object, footprint) in use.
using SweepTableSlot = PoolSlot<SweepTableSet>;

/***** The model's own state *****/

// Everything the surface-contact model owns, held by ModalAudio for as long as the audio system lives.
// The audio thread reads the pools without synchronizing, and the main thread repoints a slot only once the published masks show no voice reads it.
struct SurfaceAudioState {
    std::atomic<float> SustainLevel{1}; // Level of the sustained-contact excitation
    std::atomic<float> AccelNoiseGain{1}; // Level of the acceleration noise a body's rigid recoil radiates
    // How much of the object's vibration modulates the contact separation.
    std::atomic<float> Coupling{1};
    // Silence one modal drive row each, for isolating a feedback loop.
    std::atomic<bool> MuteGeometricDrive{false}, MuteFrictionDrive{false};
    // What the contact meter has seen since the render began: shocks per second, the share of samples every cell stood free, and the share of cell-samples that bore.
    // All three read zero except under CONTACT_REPORT.
    std::atomic<float> ContactShockRate{0}, ContactFreeShare{0}, ContactBearingShare{0};

    uint64_t SurfaceTracksRefused{0}, VoicesRefused{0}; // Main thread only.

    // The live voices, one per contact side. Audio thread only, and cleared with the bank they address.
    std::vector<uint64_t> VoiceId;
    std::vector<uint32_t> VoiceObject;
    std::vector<SustainedState> VoiceState;
    std::vector<SustainedCarry> VoiceCarry;

    // Sustained contacts, republished whole each main-thread frame and adopted once per callback.
    // Three slots, since a frame-rate publish against shorter callbacks sometimes finds the previous one still being read.
    std::array<VoiceSet, 3> VoiceSets;
    std::atomic<const VoiceSet *> PublishedVoices{nullptr};
    uint32_t VoiceSetWrite{0}; // Main thread only.
    uint64_t VoiceFrame{0}; // Main thread only.
    uint64_t ContactStep{0}; // Main thread only. The simulation step the newest published set was built from.
    uint64_t AdoptedVoiceFrame{~0ull}; // Audio thread only.
    uint32_t VoiceSetIdleSamples{0}; // Audio thread only. Samples since the published frame last advanced.

    // Surface tracks sustained voices read, one slot per distinct track, addressed by a content key.
    // A slot the audio thread is done with keeps its track until another needs the slot.
    static constexpr uint32_t MaxSurfaceTracks{64}; // One bit per slot in VoiceTrackMask.
    std::array<SurfaceTrackSlot, MaxSurfaceTracks> SurfaceTracks;
    std::unordered_map<uint64_t, uint32_t> SurfaceTrackSlotByKey; // Content key to slot. Main thread only.
    // The slots this callback's voices read, so the main thread knows which it may repoint.
    std::atomic<uint64_t> VoiceTrackMask{0};
    uint64_t ReusableSlots{0}; // Slots free to repoint this frame, cleared as each is claimed. Main thread only.

    // Element springs sustained voices read (CONTACT_SPRINGS), pooled exactly as the tracks are.
    // A set is a few megabytes, so the pool holds a handful.
    static constexpr uint32_t MaxContactSprings{8};
    std::array<ContactSpringSlot, MaxContactSprings> ContactSprings;
    std::unordered_map<uint64_t, uint32_t> ContactSpringSlotByKey; // Content key to slot. Main thread only.
    std::atomic<uint64_t> VoiceSpringMask{0}; // The spring slots this callback's voices read.
    uint64_t ReusableSpringSlots{0}; // Spring slots free to repoint this frame. Main thread only.
    // Built spring surfaces keyed by everything but the load bucket, so a new load bucket re-seats a copy.
    // Main thread only.
    std::unordered_map<uint64_t, std::shared_ptr<const ContactSpringSet>> SpringSurfaceByKey;
    // Strip products keyed additionally without the footprint bucket, so a new footprint reuses already synthesized strips.
    // Main thread only.
    std::unordered_map<uint64_t, std::shared_ptr<const SpringStrips>> SpringStripsByKey;

    // Moving-load sweep tables, one per (spring set, object, footprint) in use, pooled the same way.
    static constexpr uint32_t MaxSweepTables{8};
    std::array<SweepTableSlot, MaxSweepTables> SweepTables;
    std::unordered_map<uint64_t, uint32_t> SweepTableSlotByKey; // Content key to slot. Main thread only.
    std::atomic<uint64_t> VoiceSweepMask{0}; // The sweep slots this callback's voices read.
    uint64_t ReusableSweepSlots{0}; // Sweep slots free to repoint this frame. Main thread only.
};

static_assert(SurfaceAudioState::MaxSurfaceTracks == 8 * sizeof(decltype(SurfaceAudioState::VoiceTrackMask)::value_type));
static_assert(SurfaceAudioState::MaxContactSprings <= 8 * sizeof(decltype(SurfaceAudioState::VoiceSpringMask)::value_type));
static_assert(SurfaceAudioState::MaxSweepTables <= 8 * sizeof(decltype(SurfaceAudioState::VoiceSweepMask)::value_type));

// One voice's block-fixed constants: the slip-turnover channel's refresh, cascade poles, arrival rate, event amplitude and mean-blocker coefficient, and the sampled bed's cell weight, site share, tail cutoff and taper.
struct VoiceBlock {
    double Refresh{0}, P1{1}, P2{1};
    float P{0}, A{0}, Dcn{0};
    bool Shot{false}, Moving{false};
    float LoadW{0}, CSite{0}, CutoffM{0}, PotCutM{0}, TaperS{1};
};

// One direction's deflection read-out gains, a row of `count` per voice or per bin.
// The pairs hold each mode's one- and two-step rotations.
struct ModeReadGains {
    std::vector<float> Im, Re, Im2, Re2;

    void Resize(size_t n) {
        Im.resize(n);
        Re.resize(n);
        Im2.resize(n);
        Re2.resize(n);
    }
    // One mode's gain, rotated one and two samples ahead.
    void Fill(size_t at, float read, float c_re, float c_im) {
        Im[at] = read * c_re;
        Re[at] = read * c_im;
        Im2[at] = read * (c_re * c_re - c_im * c_im);
        Re2[at] = read * (2 * c_re * c_im);
    }
};

// One renderer's working memory for the coupled kernel, kept across blocks so a steady state allocates nothing.
struct SurfaceRenderScratch {
    std::vector<uint32_t> Voices; // The object's own, gathered out of the flat voice lists.
    // Per-mode gain of each excitation, a row being a mode shape projected onto a contact direction and fixed for the block.
    // Five rows per voice (normal, each surface's geometric, frictional, transverse frictional), then one row per impact.
    std::vector<float> DriveGains;
    // Each voice's deflection read-out along the contact normal, at the contact point.
    ModeReadGains PointRead;
    std::vector<float> Forces; // This sample's force behind each drive row.
    std::vector<float> Excite; // This sample's excitation of each mode.
    // The moving-load sweep drive of each mode this sample, computed before the channel solve so every channel's free prediction includes it.
    std::vector<float> SweepExcite;
    std::vector<const RoughnessTrack *> Tracks; // Each voice's surface tracks, resolved once per block.
    std::vector<const RoughnessTrack *> TurnoverTracks; // Each voice's turnover noise track, likewise.
    std::vector<const ContactSpringSet *> SpringSets; // Each voice's element springs, likewise.
    std::vector<const SweepTableSet *> SweepSets; // Each voice's moving-load table, likewise.
    std::vector<float> SweepConformity; // Each voice's per-mode sweep conformity gain, fixed for the block.
    std::vector<VoiceBlock> VoiceBlocks; // Each voice's block-fixed constants, likewise.
    std::vector<uint32_t> BinRowBase; // Each voice's first distributed-excitation drive row.
    // Per-bin deflection read-out, indexed like the bin drive rows, so each bin's local conformity is read at its own position on the surface.
    ModeReadGains BinRead;
    // The exchange's channel layout: a binned spring voice exchanges through one channel per bin, anything else through one channel.
    // Channel-indexed arrays size by the total channel count.
    std::vector<uint32_t> ChannelBase, ChannelCount;
    std::vector<float> ChSupport; // Each channel's share of its voice's solver support force.
    std::vector<float> SolvedForce; // Each voice's solved contact force this sample, for the sweep read.
    // Quadratised-exchange scratch.
    std::vector<float> NormalShape; // Per-voice per-mode normal shape projection, for the cross terms.
    std::vector<float> QuadCross; // Pairwise step response per newton of contact force, w-units, V x V.
    std::vector<float> QuadFeDefl; // Per-voice modal free response to the block's constant support forces.
    std::vector<float> QuadPsi, QuadG, QuadGamma, QuadFree, QuadPrevW, QuadS, QuadHx, QuadGNominal;
    std::vector<float> QuadDefl, QuadSlope;
    // Coupled-solve scratch, in double because a channel at the passivity-cap gradient puts 4/float-eps on its column, past where a float row keeps the identity.
    std::vector<double> QuadMat, QuadRhs;
    std::vector<uint8_t> QuadContact;
    // The impulsive slam closure's per-voice state: the landing's engagement rate when the sample slams (zero otherwise) and the junction's normal displacement response per newton.
    std::vector<float> QuadSlam, QuadCn;
    // The stiction channel's per-voice state: the tangential deflection read, tangential and cross compliance, the constant free-part share, and the assembled free step.
    ModeReadGains TangentRead;
    std::vector<float> QuadCt, QuadCnt, QuadFeT, QuadRhsT, QuadDeflT;
    std::vector<uint8_t> QuadTangent;
    // The transverse junction's mirror of the stiction arrays, along normal cross slip. Spring voices only.
    ModeReadGains TransverseRead;
    std::vector<float> QuadCtTr, QuadCntTr, QuadRhsTr, QuadDeflTr;
    std::vector<vec3> TransverseDir; // Each voice's transverse direction, fixed for the block.
    std::vector<uint32_t> TransverseRow; // Each voice's appended transverse drive row, or none.
    std::vector<uint8_t> QuadTransverse;
};

inline SurfaceAudioState &Surface(ModalAudio &m) { return *m.Surface; }
inline const SurfaceAudioState &Surface(const ModalAudio &m) { return *m.Surface; }

// Mark the track and spring slots no voice reads, which this frame may repoint. Main thread only.
inline void BeginSurfaceTrackFrame(SurfaceAudioState &s) {
    auto named = s.VoiceTrackMask.load(std::memory_order_acquire);
    auto named_springs = s.VoiceSpringMask.load(std::memory_order_acquire);
    auto named_sweeps = s.VoiceSweepMask.load(std::memory_order_acquire);
    // The main thread cannot tell which of the three sets a callback still holds, so all three count.
    for (const auto &set : s.VoiceSets) {
        for (const auto &v : set.Voices) {
            for (const auto &t : v.State.Tracks) {
                if (t.Index >= 0) named |= 1ull << uint32_t(t.Index);
            }
            if (v.State.Turnover.Index >= 0) named |= 1ull << uint32_t(v.State.Turnover.Index);
            if (v.State.SpringIndex >= 0) named_springs |= 1ull << uint32_t(v.State.SpringIndex);
            if (v.State.SweepIndex >= 0) named_sweeps |= 1ull << uint32_t(v.State.SweepIndex);
        }
    }
    s.ReusableSlots = ~named;
    s.ReusableSpringSlots = ~named_springs;
    s.ReusableSweepSlots = ~named_sweeps;
}

// The pool slot holding `key`, or -1 when every slot is taken and a voice reads each. Main thread only.
// `make` returns a shared_ptr to the value.
template<typename T, size_t N>
int32_t AdoptPoolSlot(std::array<PoolSlot<T>, N> &slots, std::unordered_map<uint64_t, uint32_t> &by_key, uint64_t &reusable, uint64_t key, auto &&make) {
    static_assert(N <= 64); // One bit per slot in the reusable mask.
    if (const auto it = by_key.find(key); it != by_key.end()) {
        reusable &= ~(1ull << it->second);
        return int32_t(it->second);
    }

    uint32_t index = 0;
    while (index < N && slots[index].Owned) ++index;
    if (index == N) {
        // Every slot holds a value, so take over one no voice reads.
        constexpr uint64_t Occupiable = N >= 64 ? ~0ull : (1ull << (N & 63)) - 1;
        const auto free_slots = reusable & Occupiable;
        if (free_slots == 0) return -1;
        index = uint32_t(std::countr_zero(free_slots));
        by_key.erase(slots[index].Key);
    }
    auto &slot = slots[index];
    slot.Live.store(nullptr, std::memory_order_relaxed);
    slot.Owned = make();
    slot.Key = key;
    slot.Live.store(slot.Owned.get(), std::memory_order_release);
    by_key.emplace(key, index);
    reusable &= ~(1ull << index);
    return int32_t(index);
}

// Release every slot in a pool. Main thread only.
template<typename T, size_t N>
void ResetPool(std::array<PoolSlot<T>, N> &slots, std::unordered_map<uint64_t, uint32_t> &by_key, std::atomic<uint64_t> &voice_mask, uint64_t &reusable) {
    by_key.clear();
    for (auto &slot : slots) {
        slot.Live.store(nullptr, std::memory_order_relaxed);
        slot.Owned.reset();
        slot.Key = 0;
    }
    voice_mask.store(0, std::memory_order_relaxed);
    reusable = 0;
}

inline int32_t AdoptSurfaceTrack(SurfaceAudioState &s, uint64_t key, auto &&make) {
    const auto index = AdoptPoolSlot(s.SurfaceTracks, s.SurfaceTrackSlotByKey, s.ReusableSlots, key, make);
    if (index < 0) ++s.SurfaceTracksRefused;
    return index;
}

inline int32_t AdoptContactSprings(SurfaceAudioState &s, uint64_t key, auto &&make) {
    return AdoptPoolSlot(s.ContactSprings, s.ContactSpringSlotByKey, s.ReusableSpringSlots, key, make);
}

inline int32_t AdoptSweepTable(SurfaceAudioState &s, uint64_t key, auto &&make) {
    return AdoptPoolSlot(s.SweepTables, s.SweepTableSlotByKey, s.ReusableSweepSlots, key, make);
}

// An empty set to write this frame's contacts into, never one a callback may still be reading. Main thread only.
VoiceSet &NextVoiceSet(SurfaceAudioState &);
// Publish the set NextVoiceSet handed out, which ends every contact it omits. Main thread only.
void PublishVoiceSet(SurfaceAudioState &);
