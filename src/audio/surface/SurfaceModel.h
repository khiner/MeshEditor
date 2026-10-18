#pragma once

#include "audio/ContactModel.h"
#include "audio/ModalAudio.h"

#include <algorithm>
#include <cmath>

constexpr uint32_t TrackSamples{131072};

// Distance a contact travels before a synthesized field repeats, m.
constexpr float SurfaceRepeatLength{0.1f};

// SURFACE_FFT_THREADS overrides the synthesis thread count.
void SetTransformThreads(uint32_t threads);

uint64_t HashParams(uint64_t seed, auto... values) {
    const auto combine = [&seed](double v) { seed ^= std::hash<double>{}(v) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2); };
    (combine(double(values)), ...);
    return seed;
}

// Boxcar statistics use widths of 2^i samples.
constexpr uint32_t TrackWidthCount{18};

// Track samples one short-wavelength cutoff spans (Pastewka and Robbins arXiv:1508.02154).
constexpr float SurfaceSamplesPerCutoff{4};

inline float FinishTrackSpacing(float short_wavelength) {
    static const float per_cutoff = [] {
        const char *v = std::getenv("SURFACE_SAMPLES_PER_CUTOFF");
        return v ? float(std::atof(v)) : SurfaceSamplesPerCutoff;
    }();
    return std::clamp(short_wavelength / per_cutoff, 1e-9f, 1e-4f);
}

// Distance a contact travels over which a track's local mean is removed, m.
constexpr float ReliefDcLength{1e-2f};

// Least-squares fit of a flat spectrum below its corner and a power law above it.
struct MeasuredBand {
    float CorrelationLength{0}, SpectralSlope{0};
};

// Surface heights indexed by distance along the surface, traversed cyclically.
struct RoughnessTrack {
    std::vector<float> Heights; // Zero-mean, unit root-mean-square.
    std::vector<float> Sum; // Running integral, one entry longer than Heights.
    float Spacing{0}; // Distance along the surface between samples, m.
    float Cutoff{0};
    float Rms{1}; // Root-mean-square height of the source, m.
    MeasuredBand Band{}; // Zero for tracks synthesized from parameters.
    float HeightMax{0}; // The tallest height, over the unit root-mean-square heights.
    // RMS first difference of unit-RMS heights; multiply by height scale / spacing for gradient.
    float SlopeRms{0};
    // RMS second difference of unit-RMS heights.
    float CurvatureRms{0};
    // RMS boxcar mean of unit-RMS heights at widths of 2^i samples.
    std::array<float, TrackWidthCount> WindowRms{};
    // RMS gradient per sample at boxcar widths of 2^i, using unit-RMS heights.
    // Width one equals SlopeRms; unresolved widths repeat the widest measurement.
    std::array<float, TrackWidthCount> SlopeWindowRms{};
};

// Interpolate table values by log2 sample width.
inline float TrackOctaveAt(const std::array<float, TrackWidthCount> &table, float width) {
    if (width <= 1) return table[0];
    const float octave = std::min(std::log2(width), float(TrackWidthCount - 1));
    const auto lo = uint32_t(octave);
    if (lo + 1 >= TrackWidthCount) return table[TrackWidthCount - 1];
    return table[lo] + (octave - float(lo)) * (table[lo + 1] - table[lo]);
}

// Return RMS boxcar-mean height at the requested width.
inline float TrackWindowRms(const RoughnessTrack &t, float width) { return TrackOctaveAt(t.WindowRms, width); }
// Return RMS gradient per sample at the requested width.
inline float TrackSlopeAtWidth(const RoughnessTrack &t, float width) { return TrackOctaveAt(t.SlopeWindowRms, width); }

// Return the interpolated sample width at half the unit-RMS height.
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

// Returns a deterministic self-affine track with spectrum q^p above the correlation-length corner.
RoughnessTrack SynthesizeRoughness(float correlation_length, float spectral_slope, float short_wavelength, float spacing, uint32_t count);

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

// Returns a deterministic patch with radial spectrum q^slope above the correlation-length corner.
RoughnessField SynthesizeRoughnessPatch(float correlation_length, float spectral_slope, float short_wavelength, float spacing, uint32_t columns, uint32_t rows, uint32_t realization);

using LatticeCovariance = std::array<std::array<double, 5>, 5>;
LatticeCovariance RoughnessLatticeCovariance(float correlation_length, float spectral_slope, float short_wavelength, float spacing, uint32_t columns, uint32_t rows);

// Returns a deterministic unit-variance white-noise track.
RoughnessTrack SynthesizeTurnover(float spacing, uint32_t count, uint64_t seed);

RoughnessTrack MakeProfileTrack(std::span<const float> heights, float spacing);

// Synthesizes an isotropic patch whose first row exactly matches a measured trace.
RoughnessField SynthesizeProfileField(std::span<const float> heights, float spacing, uint32_t rows, uint32_t realization);

// Cyclic track position with sample index, fraction, and traversal count.
struct TrackPos {
    size_t Index;
    float Frac;
    double Wraps;
};
inline TrackPos WrapTrackPos(const RoughnessTrack &t, double pos) {
    const double n = double(t.Heights.size());
    const double wraps = std::floor(pos / n);
    const double f = std::max(pos - wraps * n, 0.0);
    // Clamp rounding errors at the track boundary.
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

// Filtered height and slope per track sample.
struct TrackReading {
    float Height, Slope;
};

// Return continuous height and slope using four samples.
inline TrackReading ReadTrackSmooth(const RoughnessTrack &t, double pos) {
    // Continuity across sample boundaries prevents steps when reading windows narrower than one sample.
    const auto p = WrapTrackPos(t, pos);
    const size_t n = t.Heights.size();
    const float h0 = t.Heights[(p.Index + n - 1) % n], h1 = t.Heights[p.Index];
    const float h2 = t.Heights[(p.Index + 1) % n], h3 = t.Heights[(p.Index + 2) % n];
    // Hermite reconstruction from Dang 2013, Eqs. 30-31.
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

// Uses smooth reconstruction for windows narrower than one sample.
inline TrackReading ReadTrackSloped(const RoughnessTrack &t, double pos, float window) {
    return window <= 1.f ? ReadTrackSmooth(t, pos) : TrackBoxcar(t, pos, window);
}

// Accumulate sigma-scaled cell heights into out using sub_window samples per cell.
// Return height and slope over the full window.
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

// Sub-cutoff roughness compliance from Pastewka et al. 2013 Appendix B.
struct SubCutoffRoughness {
    // RMS height at width w is Amplitude * w^Hurst, in meters (Pastewka et al. 2013, Eq. B19).
    double Amplitude{0};
    double Hurst{0}; // Self-affine exponent the height is extrapolated along.
    double MaxWidth{0}; // Maximum width in meters for applying this band.
};

// Return the unresolved band for one-dimensional exponent slope, with Hurst = -(1 + slope)/2.
// Return zero outside the self-affine range or when cutoff exceeds correlation length.
SubCutoffRoughness UnresolvedBand(double roughness, double correlation_length, double short_wavelength, double slope);

// Return mean separation at contact width in meters.
inline double SubCutoffSeparation(const SubCutoffRoughness &s, double width) {
    // Persson separation coefficient from Pastewka et al. 2013, Eq. B17.
    constexpr double PerssonGamma{0.4};
    return s.Amplitude > 0 ? PerssonGamma * s.Amplitude * std::pow(std::min(width, s.MaxWidth), s.Hurst) : 0.0;
}

constexpr uint32_t FlankSpectrumBins{16};

inline uint32_t FlankSpectrumBin(double depth, double engagement) {
    if (!(depth > 0) || !(engagement > depth)) return 0;
    return uint32_t(std::min(double(FlankSpectrumBins - 1), std::floor(std::log2(engagement / depth))));
}

struct ElementSprings {
    float Width{0}; // Along-track extent of one element, m
    float EngagementMax{0}; // Deepest engagement the curves are tabulated to, m
    uint32_t Knots{0}; // Engagement knots per element
    // Number of distinct curve sets tiled across Crest.
    uint32_t Curves{0};
    float CrestRange{0}; // Spread of the crests the curves were gathered from, m.
    std::vector<float> Engagement; // Knots shared by every element, ascending from 0.
    std::vector<float> Crest; // Per element: height of its highest point, m.
    std::vector<float> Force; // Per element: force at each knot, N. Element-major, Knots per element.
    std::vector<float> Slope; // Per element: the interpolant's slope at each knot, N/m.
    std::vector<float> Potential; // Per element: the force curve's own integral at each knot, J.
    // Sum of squared bearing-summit forces per knot, N^2; variance for independent slip redraws.
    std::vector<float> ForceVariance;
    // Sum of bearing-summit oblique-flank moments per knot, N/m^2.
    std::vector<float> FlankMoment;
    std::vector<float> BearingCount; // Per element: how many of its summits bear at each knot, fractional between knots.
    // Sum of bearing-contact widths, m; each width equals Hertz stiffness / modulus.
    std::vector<float> BearingWidth;
    std::vector<float> FlankSpectrum;
    // Normal force per strength bin.
    // Coulomb force divided by stiffness determines breakaway displacement; force and flank moment determine both.
    std::vector<float> FlankSpectrumForce;

    // Cache maximum crests per aligned block, wrapping the final partial block.
    // Call RefreshCrestBlocks after crest changes.
    static constexpr uint32_t CrestBlock{64};
    std::vector<float> BlockCrest;
    // Last knot below each power-of-two engagement from KnotExponentFloor.
    // An empty table selects bisection.
    int32_t KnotExponentFloor{0};
    std::vector<uint8_t> KnotByExponent;

    uint32_t Count() const { return uint32_t(Crest.size()); }
    uint32_t CurveOf(uint32_t element) const { return Curves > 0 ? element % Curves : element; }
};

// Each set of strength-bin fractions sums to one over occupied bins.
struct FlankSpectrumShares {
    std::array<float, FlankSpectrumBins> Moment{};
    std::array<float, FlankSpectrumBins> Force{};
};
// Return force and flank-moment fractions by strength bin at the knot nearest engagement.
FlankSpectrumShares FlankSpectrumShareAt(const ElementSprings &, float engagement);

// Force and stored energy of one element's spring at one engagement.
struct SpringRead {
    float Force{0}; // N
    float Potential{0}; // J
};

// Return force and potential at engagement meters below the crest; return zero at or above it.
SpringRead ReadElementSpring(const ElementSprings &, uint32_t element, float engagement);

// Return slip-turnover force variance in N^2, linearly interpolated between knots.
// Return zero at or above the crest.
float ElementForceVariance(const ElementSprings &, uint32_t element, float engagement);

void RefreshCrestBlocks(ElementSprings &);

// Return the reach in elements where quadratic body lift exceeds the highest crest above maximum engagement.
uint32_t ElementReach(const ElementSprings &, double combined_curvature);

// Returns the first-contact body height at each element center.
std::vector<float> ElementEnvelope(const ElementSprings &, double combined_curvature, uint32_t reach);

// Returns the first-contact envelope over a crest array.
std::vector<float> CrestEnvelope(std::span<const float> crest, float width, double combined_curvature, uint32_t reach);

// Returns the load-bearing height at each element, interpolated at one-eighth of the contact reach.
std::vector<float> BearingDatum(const ElementSprings &, double combined_curvature, uint32_t reach, double load);

// Returns the cyclic envelope at a fractional element position.
float EnvelopeAt(std::span<const float> envelope, double position);

// Registers the slower surface's crest deviation against the composite element array.
struct SlidingRegistration {
    std::span<const float> SlowCrestDev; // Slower-side zero-mean crest deviation per element, m; empty disables it.
    double Slip{0}; // Offset behind the array in elements.
};

// Returns the registered crest deviation at a fractional element position.
float Reregister(const SlidingRegistration &, double at);

SpringRead ReadContactSprings(const ElementSprings &, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach, const SlidingRegistration &slide = {});

SpringRead ReadContactSpringBins(
    const ElementSprings &, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach,
    std::span<float> bin_force, std::span<const float> bin_engagement = {}, const SlidingRegistration &slide = {}
);

SpringRead ReadContactSpringChannels(
    const ElementSprings &, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach,
    std::span<float> bin_force, std::span<float> bin_potential, std::span<const float> bin_engagement, const SlidingRegistration &slide = {}
);

// Returns the engagement whose position-averaged spring force equals `normal_force`.
float SolveSpringEngagement(const ElementSprings &, std::span<const float> envelope, double normal_force, double combined_curvature, uint32_t reach);

void SweepModeDrives(std::span<const float> field, std::span<const float> phi, uint32_t modes, uint32_t reach, std::span<float> table);

// The bearing population's oblique-flank moment and normal stiffness at one engagement.
struct SpringFlankRead {
    double Modulus{0};
    // Sum bearing-contact stiffnesses in N/m, with sub-contact roughness compliance in series at each contact.
    double Stiffness{0};
    double Bearing{0}; // Number of load-bearing summits.
    double Width{0}; // Sum of those contacts' own widths, m.
};
SpringFlankRead SpringFlankMoments(const ElementSprings &, std::span<const float> envelope, double position, float engagement, double combined_curvature, uint32_t reach);

inline double SpringBearingWidth(const SpringFlankRead &read) {
    return read.Bearing > 0 ? read.Width / read.Bearing : 0.0;
}

// Mark samples at or above all eight neighbors, wrapping columns and clamping rows.
// Require row-major heights with columns along the track.
std::vector<uint8_t> MarkFieldSummits(std::span<const float> folded, uint32_t columns, uint32_t rows);

// Return summit curvature in 1/m as the negative discrete Laplacian.
// Both principal curvatures are negative at a maximum; Hertz contact uses their negated sum.
// Wrap columns and clamp rows; use isotropic along-track curvature at edge rows.
double FieldSummitCurvature(std::span<const float> folded, uint32_t columns, uint32_t rows, float column_spacing, uint32_t column, uint32_t row);

// Summit population grouped by element across several independent surface realizations.
struct ElementSummits {
    float Width{0}; // Along-sweep extent of one element, m.
    double InvModulus{0}; // Pair compliance 1/E* used for contact sizing.
    std::vector<float> Crest; // Highest point of each element over every realization, m. One per element.
    std::vector<uint32_t> Element; // Element index for each summit.
    std::vector<float> Height; // Summit height under the body's gap, m.
    std::vector<float> Stiffness; // Hertz constant the summit's own curvature gives it, N/m^(3/2).
};

// Gather summits into elements of element_columns columns.
// Require row-major heights in meters, with columns along the track.
// Apply element_lift as a rigid offset per element.
void GatherElementSummits(
    ElementSummits &out, std::span<const float> heights, uint32_t columns, uint32_t rows,
    std::span<const float> transverse_gap, std::span<const float> element_lift,
    uint32_t element_columns, float column_spacing, double inv_effective_modulus
);

// Appends a population and retains the maximum crest per element.
void MergeElementSummits(ElementSummits &into, const ElementSummits &from);

void GatherElementCrests(
    std::vector<float> &out, std::span<const float> heights, uint32_t columns, uint32_t rows,
    std::span<const float> transverse_gap, std::span<const float> element_lift, uint32_t element_columns
);

// Build per-element force curves with sub-cutoff compliance in series per summit.
// Tabulate knots geometrically up to engagement_max.
ElementSprings BuildElementSprings(const ElementSummits &, const SubCutoffRoughness &, double engagement_max, uint32_t knots);

ElementSprings BuildElementSprings(
    std::span<const float> heights, uint32_t columns, uint32_t rows, std::span<const float> transverse_gap,
    uint32_t element_columns, float column_spacing, double inv_effective_modulus,
    const SubCutoffRoughness &, double engagement_max, uint32_t knots
);

// One Gaussian spectral band of the strip field.
struct EnsembleBand {
    float Correlation{0};
    float Slope{0};
    float Cutoff{0};
    float Sigma{0}; // Height the unit-variance draw is scaled by, m.
};

ElementSprings BuildEnsembleSprings(
    std::span<const EnsembleBand> bands, float spacing, uint32_t columns, uint32_t rows,
    std::span<const float> transverse_gap, double body_curvature, uint32_t element_columns,
    uint32_t strip_realizations, double inv_effective_modulus,
    const SubCutoffRoughness &, double engagement_max, uint32_t knots, std::vector<float> crests
);

// Tangential-to-normal stiffness of a contact patch, Mindlin's 2*(1-nu)/(2-nu) near nu = 0.3.
constexpr double MindlinShearRatio{0.82};

double FlankSineCube(double slope);

inline double FlankJunctionStiffness(double sine_cube, double normal_force, double flank_modulus) {
    return MindlinShearRatio * std::sqrt(sine_cube * normal_force * flank_modulus);
}

constexpr uint32_t MaxFlankBins{16};

// One Jenkins element with stiffness and Coulomb breakaway force.
struct FlankJunctionBin {
    float Stiffness{0}; // N/m
    float ConeShare{0};
};

uint32_t FlankJunctionSpread(
    double sine_cube, double normal_force, double flank_modulus,
    std::span<const float> moment_share, std::span<const float> force_share, std::span<FlankJunctionBin> out
);

// Return asperity pressure p = E*|grad h|_rms/2 in Pa.
// Pressure is independent of load and area, making real contact area proportional to load.
// Assume isotropy to double profile-gradient variance for the full surface gradient.
double AsperityPressure(double inv_effective_modulus, double profile_slope_rms);

double RealContactArea(double normal_force, double bound_area, double inv_effective_modulus, double profile_slope_rms);

double HurstExponent(double spectral_slope);

// Return q0 = 2*pi/l in rad/m from correlation length in meters.
double RollOffWavevector(double correlation_length);

// Persson coefficients: u0 = roughness / Alpha; Beta sets full-contact pressure.
struct SeparationCoefficients {
    double Alpha{2.5}, Beta{0.5};
};

// Persson 2007 Eq. 15 fit: gamma = 0.4 matches numerical contact decay lengths; gamma = 1 reproduces the figures.
inline constexpr double SeparationGamma = 0.4;

// Integrate q1/q0 over three decades; Alpha varies under 2% across four decades.
inline constexpr double RoughnessBandRatio = 1e3;

SeparationCoefficients PerssonSeparationCoefficients(double hurst, double band_ratio, double gamma = SeparationGamma);

// Return nominally flat rough-contact pressure in Pa.
double ConformalPressure(double separation, double roughness, double roll_off_wavevector, double inv_effective_modulus, const SeparationCoefficients &);

// Return mean separation in meters, or zero at full-contact pressure.
double ConformalSeparation(double pressure, double roughness, double roll_off_wavevector, double inv_effective_modulus, const SeparationCoefficients &);

double ConformalPatchWidth(double combined_curvature, double roughness, const SeparationCoefficients &);

// Return conformal stiffness dN/du in N/m.
double ConformalStiffness(double normal_force, double roughness, const SeparationCoefficients &);

inline constexpr uint32_t MaxBedSpots = 32;

double BedHeightIntegral(double lambda, double power);

double SolveBedSeparation(double normal_force, double spot_count, double spot_stiffness, double height_rms, double power = 1.5);

// Return tabulated load and stiffness factors, with unloaded and full-contact limits outside the table.
float BedLoadFactor(float lambda);

// Return per-asperity load variance in BedLoadFactor units.
// Independent asperities add variances.
float BedVarianceFactor(float lambda);

// Return the clamped linear load factor.
float BedLinearLoadFactor(float lambda);

// Return the load factor integrated from the table lower bound.
float BedPotentialFactor(float lambda);
float BedLinearPotentialFactor(float lambda);

float BedLoadAt(float engagement, float spread);
float BedPotentialAt(float engagement, float spread);

struct AsperityBed {
    uint32_t SpotCount{0}; // Cells rendered along the track
    double TotalSpots{0}; // Asperities bearing across the whole patch, which grows with the load
    double SpotWeight{0};
    double SpotStiffness{0}; // k of one asperity, N/m^(3/2)
    double SpotRadius{0}; // Radius of curvature at one asperity, m
    double HeightRms{0}; // Spread of gap height at the asperity scale, m
    double CellSpread{0}; // Spread of asperity heights about their cell's local mean, m
    double Separation{0};
};

double BedLoad(const AsperityBed &);

// Return summed asperity stiffness in N/m.
double BedStiffness(const AsperityBed &);

// Return total Hertz contact area in m^2.
double BedContactArea(const AsperityBed &);

// Return the fraction of surface excursion absorbed by asperities in series with bulk compliance.
// The fraction approaches one for wide geometry-fixed patches and decreases for small load-created patches.
double BedSurfaceShare(double inv_effective_modulus, double bound_area, const AsperityBed &);

inline uint32_t BedCellCount(double patch_width, double region_width, double peak_density) {
    return uint32_t(std::clamp(std::max(region_width, patch_width) * peak_density, 1.0, double(MaxBedSpots)));
}

inline double BedCellWindow(double patch_width, double region_width, double peak_density, double half_variance_width) {
    const auto cells = BedCellCount(patch_width, region_width, peak_density);
    const double share = patch_width / cells;
    const double across = std::max(patch_width * peak_density, 1.0);
    const double population_span = peak_density > 0 ? across * across / (cells * peak_density) : half_variance_width;
    return std::max(share, std::min(half_variance_width, population_span));
}

AsperityBed ResolveAsperityBed(double normal_force, double patch_width, double region_width, double inv_effective_modulus, double peak_density, double cell_window, auto &&spot_curvature, auto &&spot_height_rms) {
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
        .CellSpread = std::sqrt(std::max(height_rms * height_rms - cell_mean_rms * cell_mean_rms, 0.0)),
    };
    bed.Separation = SolveBedSeparation(normal_force, bed.TotalSpots, bed.SpotStiffness, bed.HeightRms);
    return bed;
}

// Floor the approach-speed divisor at Chi*dt^2 for the small-step limit of van Walstijn et al. 2024 Eq. 38.
inline constexpr float ImpactVelocityChi = 1e7f;

// Barycentric sample-point weights on the modal surface.
struct SamplePointBlend {
    std::array<uint32_t, 3> Points{};
    vec3 Weights{1, 0, 0};
};

// Resolve local force variation across the footprint even when total force is nearly constant.
// Each position bin drives modes at its own contact position, retaining excitation lost by summing forces first.
constexpr uint32_t MaxSpringBins{32};
// Return the bin center offset from footprint center in meters.
inline float SpringBinOffset(uint32_t bin, uint32_t bins, float half_extent) {
    return bins > 0 ? ((float(bin) + 0.5f) / float(bins) - 0.5f) * 2 * half_extent : 0.f;
}
// At 64 edge samples, relaxation agrees with the closed form within 2% above 100 Hz.
constexpr uint32_t MaxRelaxEdges{64};

struct ContactTrack {
    int32_t Index{-1}; // Surface track pool slot, -1 for an unused one.
    float Rate{0}; // Track samples advanced per output sample, from that surface's own sweep.
    float Sigma{0}; // Height scale applied to the track, m.
    float Window{0}; // Extent the contact reads over the track, in track samples.
    // Cell filter width in samples; zero uses the whole-window mean.
    float SubWindow{0};
    float Step{0}; // Distance along the surface per output sample, m.
    float Spacing{0}; // Distance along the surface between track samples, m.
};

// Store sustained-contact state from KHR_audio_rigid_bodies and derived force-model constants.
struct SustainedState {
    SamplePointBlend Blend{};
    vec3 N{0}; // Node-local unit normal, directed into the object.
    vec3 SlipDir{0}; // Node-local unit slip direction; zero at rest.
    // Node-local geometric force directions, signed for equal and opposite body excitation.
    std::array<vec3, 2> SweepDir{};
    float NormalForce{0}; // N, the load the excitation fluctuates about.
    float Friction{0}; // Combined friction coefficient.
    float SlipSpeed{0}; // Relative tangential speed in m/s.
    // Applied tangential force along SlipDir in newtons, used as the stiction fluctuation baseline.
    float SolverFriction{0};
    float Stiffness{0}; // k, N/m^(3/2). Zero for a contact two faces fix the area of, which bears on a bed of asperities.
    // Equilibrium approach in meters: Hertz penetration or loaded-spot separation.
    float StaticPenetration{0};
    // Positive SpotCount selects distributed asperity cells instead of a single Hertz spring.
    uint32_t SpotCount{0};
    float SpotWeight{0};
    float SpotStiffness{0}; // k of one asperity, N/m^(3/2), from the radius of curvature at one.
    float CellSpread{0}; // Spread of asperity heights about their cell's local mean, m.
    float SurfaceCompliance{1}; // Share of a surface excursion the bed takes, the bulk taking the rest.
    // Dimensionless Hunt-Crossley coefficient, 1.5 * (1 - restitution).
    // Convert to s/m using approach speed.
    float DampingFactor{0};
    // Shear stiffness in N/m, using Mindlin's shear ratio for the interfacial patch.
    // Zero rigidly binds the junction to the contact point.
    float ShearStiffness{0};
    float FlankStiffness{0};
    uint32_t FlankBins{0};
    std::array<FlankJunctionBin, MaxFlankBins> FlankBin{};
    float RelaxScale{0};
    // Spring pool slot; negative selects the bed model.
    int32_t SpringIndex{-1};
    float SpringRate{0}; // Elements the contact advances per output sample.
    // Slower-surface lag in elements per output sample; zero for equal sweep rates.
    float SlipRate{0};
    uint32_t SlowSide{0};
    float SpringScale{1}; // This side's force scale on the shared springs, from the stiffness cap.
    uint32_t SpringBins{0}; // Position bins across the footprint; zero disables the channel.
    std::array<SamplePointBlend, MaxSpringBins> SpringBinBlend{};
    // Half the sweep footprint in metres, used for outer-bin centers and force moment arms.
    float SpringHalfExtent{0};
    float InverseAngularInertia{0};
    // Moving-load sweep pool slot for a footprint fixed on this side's surface.
    // Negative disables the channel.
    int32_t SweepIndex{-1};
    // Footprint turnover-force RMS in N before SpringScale, assuming independent junction redraws.
    float NoiseRms{0};
    // Scaled bearing stiffness at the anchor in N/m for turnover force-engagement conversion.
    float NoiseStiffness{0};
    float JunctionSpacing{0};
    float JunctionTransit{0};
    float FootprintLength{0};
    // Order tracks as both microscale finishes, then both mesoscale reliefs.
    // The low index bit identifies the surface, matching SweepDir on both contact sides.
    static constexpr uint32_t TrackCount{4};
    std::array<ContactTrack, TrackCount> Tracks{};
    // Surface-fixed asperity heights for turnover noise.
    ContactTrack Turnover{};
};

struct SustainedCarry {
    std::array<double, SustainedState::TrackCount> Pos{};
    double TurnoverPos{0};
    double SpringPos{0}; // Shared pair position in elements; both voices read the same gap.
    double SlipPos{0};
    double NoisePos{0}; // Position along the fast side's finish track the slip-turnover noise reads, in track samples.
    std::array<float, MaxSpringBins> NoiseMean{}; // Each turnover read's sub-audio mean, the channel's DC blocker.
    std::array<float, MaxSpringBins> NoisePool{};
    std::array<float, MaxSpringBins> NoiseSmooth{};
    std::array<float, MaxSpringBins> NoiseExpectedPool{};
    std::array<float, MaxSpringBins> NoiseExpectedSmooth{};
    uint64_t NoiseRng{0};
    std::array<float, MaxSpringBins> NoiseEngageBin{};
    float NoiseEngage{0};
    float NoiseEngagePrev{0};
    float ReliefMean{0};
    float SpringDatum{0};
    double PrevSpringPos{0}; // The position along the springs at the previous sample, in elements.
    double PrevSlipPos{0}; // The slip the slower surface stood at the previous sample, in elements.
    // Slow mean contact-force share per bin, initialized to sum to one with zero fluctuation.
    std::array<float, MaxSpringBins> SpringShareMean{};
    std::array<float, MaxSpringBins> PsiBins{};
    std::array<float, MaxSpringBins> PrevBinDefl{};
    std::array<float, MaxSpringBins> RawApproachBins{};
    std::array<float, MaxSpringBins> LastBearingG{};
    bool BinsPrimed{false}; // Set once the bin channels seed their psi, so a voice gaining bins mid-life seeds cleanly.
    // Rigid contact-axis tilt in radians and its rate; bin lever arms convert tilt to engagement.
    float Tilt{0};
    float PrevTilt{0}; // The tilt at the previous sample, so its work is metered the same way.
    float TiltRate{0}; // rad/s
    float TiltMoment{0}; // N*m, the previous sample's reactions about the pitch axis.
    float TiltMean{0};
    // Per-spot local mean height initialized for equilibrium when a voice starts mid-surface.
    std::array<float, MaxBedSpots> SpotMean{};
    uint32_t SpotBearing{0};
    std::array<float, MaxBedSpots> SpotPeak{};
    // Previous raw approach for sample-to-sample damping velocity.
    float RawApproach{0};
    float EngagementDamping{0};
    float PrevDeflection{0}; // The modal deflection read at the previous sample, m, so the datum's motion is separable.
    // How far the body has travelled along this contact's normal, m.
    float RigidNormal{0};
    float PrevRigidNormal{0}; // The same, as of the previous sample, so the body's travel is separable.
    float RigidTangent{0};
    float PrevRigidTangent{0};
    float PrevTangentDefl{0};
    float PatchShear{0};
    // Tangential state along the transverse in-plane axis.
    float RigidTransverse{0};
    float PrevRigidTransverse{0};
    float PrevTransverseDefl{0};
    float PatchShearTransverse{0};
    // The oblique-flank junction's stored stretch, m, on the normal channel's own displacement.
    float PatchFlank{0};
    std::array<float, MaxFlankBins> PatchFlankBin{};
    std::array<float, MaxRelaxEdges> RelaxApproach{};
    std::array<float, MaxRelaxEdges> RelaxTangent{};
    uint32_t RelaxEdges{0};
    // Integrated approach and tangential position in metres from the last fully released state.
    float RelaxDepth{0};
    float RelaxTangentPos{0};
    bool PrevStick{false}; // Whether the previous sample took the stick branch, for the transition count.
    bool Primed{false}; // Set once the local mean is seeded from the first sample's relief height.
    // Reseed anchors and means after changes beyond the engagement range.
    // Preserve traversal and body displacement to suppress swap-induced force.
    bool Rebase{false};
    float Psi{0}; // The voice's auxiliary variable sqrt(2 * stored contact energy).
    float PrevRelief{0};
    std::array<float, MaxBedSpots> PrevSpotHeight{};
    bool PrevContact{false};
};

float RelaxationRelease(const SustainedState &, SustainedCarry &, float approach_step, float tangent_step);
// Reset relaxation after contact separation or datum changes.
void ResetRelaxation(SustainedCarry &);

// The oblique-flank junction's force on the normal channel over one approach step, N, positive opposing the approach.
float FlankJunctionStep(const SustainedState &, SustainedCarry &, float load, float step, float cn);

// Publish all contacts each frame; omitted contacts end.
struct VoiceSet {
    struct Voice {
        uint64_t Id;
        uint32_t Object; // Bank object slot, valid only against the bank live when this was published.
        SustainedState State;
    };

    uint64_t Frame{0}; // The frame this was built in.
    std::vector<Voice> Voices;
};

// Share immutable values by content key.
// The audio thread reads Live without synchronization; the main thread replaces slots only when no audio-thread readers remain.
template<typename T> struct PoolSlot {
    std::atomic<const T *> Live{nullptr};
    std::shared_ptr<const T> Owned;
    uint64_t Key{0}; // Content key of Owned. Main thread only.
};

using SurfaceTrackSlot = PoolSlot<RoughnessTrack>;

// Store element springs with the curvature reach and envelope used to construct them.
struct ContactSpringSet {
    ElementSprings Springs;
    std::vector<float> Envelope; // Element crests under the contact's own gap, m. One entry per element.
    std::vector<float> AnchorForce; // Mean stack force over sampled contact centres at each engagement knot, N.
    double Curvature{0}; // Combined curvature bucket used to build the set; zero for a face.
    uint32_t Reach{0}; // Elements either side of centre the body can reach.
    float StripWidth{0};
    // The band below the pair's stated cutoffs, in series with every contact these springs make.
    SubCutoffRoughness SubCutoff;
    // Zero-mean per-element crest deviations in meters; empty arrays preserve composite crests.
    std::array<std::vector<float>, 2> SideCrestDev;
};

struct SpringStrips {
    ElementSprings Springs;
    std::array<std::vector<float>, 2> SideCrests;
};

// Summed flank moments at anchor, sampling every stride elements.
// Divide by Sampled for per-element means.
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

inline float SolveSpringAnchor(const ContactSpringSet &set, double normal_force) {
    const auto &z = set.Springs.Engagement;
    const auto &f = set.AnchorForce;
    const size_t n = f.size();
    if (n < 2 || normal_force <= 0) return 0.f;
    if (const auto it = std::ranges::lower_bound(f, float(normal_force)); it != f.end()) {
        const auto hi = size_t(it - f.begin());
        if (hi == 0) return z[0];
        const float df = f[hi] - f[hi - 1];
        return df > 0 ? z[hi - 1] + (z[hi] - z[hi - 1]) * float((normal_force - f[hi - 1]) / df) : z[hi - 1];
    }
    const float slope = z[n - 1] > z[n - 2] ? (f[n - 1] - f[n - 2]) / (z[n - 1] - z[n - 2]) : 0.f;
    return slope > 0 ? z[n - 1] + float((normal_force - f[n - 1]) / slope) : z[n - 1];
}

// Share spring sets between contacts with matching pair spectra and curvature buckets.
using ContactSpringSlot = PoolSlot<ContactSpringSet>;

struct SweepTableSet {
    uint32_t Modes{0}; // Rows, the object's tuned mode count when the table was built.
    uint32_t Positions{0}; // Columns, one per element along the springs.
    std::vector<float> Table; // Mode-major, Modes rows of Positions, N per unit mass-normalized shape.
    float ForceTotal{0}; // The anchored footprint force the table was built at, N.
    float NoiseVariance{0};
    float NoiseStiffness{0};
    // Per-row mean stiffness projected through squared mode shapes, in s^-2 at baked scale.
    // Apply w^2/(w^2 + projection) using current frequency and world scale.
    std::vector<float> ModeStiffness;
};

// One sweep table in the pool, one per (spring set, object, footprint) in use.
using SweepTableSlot = PoolSlot<SweepTableSet>;

struct SurfaceAudioState {
    std::atomic<float> SustainLevel{1}; // Level of the sustained-contact excitation
    std::atomic<float> AccelNoiseGain{1}; // Level of the acceleration noise a body's rigid recoil radiates
    std::atomic<float> Coupling{1};
    // Silence one modal drive row each, for isolating a feedback loop.
    std::atomic<bool> MuteGeometricDrive{false}, MuteFrictionDrive{false};
    std::atomic<float> ContactShockRate{0}, ContactFreeShare{0}, ContactBearingShare{0};

    uint64_t SurfaceTracksRefused{0}, VoicesRefused{0}; // Main thread only.

    // The live voices, one per contact side.
    // Audio thread only, and cleared with the bank they address.
    std::vector<uint64_t> VoiceId;
    std::vector<uint32_t> VoiceObject;
    std::vector<SustainedState> VoiceState;
    std::vector<SustainedCarry> VoiceCarry;

    // Publish once per main-thread frame and adopt once per callback.
    // Three slots allow a callback to retain the preceding frame.
    std::array<VoiceSet, 3> VoiceSets;
    std::atomic<const VoiceSet *> PublishedVoices{nullptr};
    uint32_t VoiceSetWrite{0}; // Main thread only.
    uint64_t VoiceFrame{0}; // Main thread only.
    uint64_t ContactStep{0}; // Simulation step used to build the newest published set; main thread only.
    uint64_t AdoptedVoiceFrame{~0ull}; // Audio thread only.
    uint32_t VoiceSetIdleSamples{0}; // Samples since the published frame last advanced; audio thread only.

    // Cache tracks by content key until an unreferenced slot is reused.
    static constexpr uint32_t MaxSurfaceTracks{64}; // One bit per slot in VoiceTrackMask.
    std::array<SurfaceTrackSlot, MaxSurfaceTracks> SurfaceTracks;
    std::unordered_map<uint64_t, uint32_t> SurfaceTrackSlotByKey; // Content key to slot; main thread only.
    // Audio-thread slot references; the main thread may replace unreferenced slots.
    std::atomic<uint64_t> VoiceTrackMask{0};
    uint64_t ReusableSlots{0}; // Reusable slots this frame, cleared on claim; main thread only.

    static constexpr uint32_t MaxContactSprings{8};
    std::array<ContactSpringSlot, MaxContactSprings> ContactSprings;
    std::unordered_map<uint64_t, uint32_t> ContactSpringSlotByKey; // Content key to slot; main thread only.
    std::atomic<uint64_t> VoiceSpringMask{0}; // The spring slots this callback's voices read.
    uint64_t ReusableSpringSlots{0}; // Reusable spring slots this frame; main thread only.
    // Main-thread spring cache excluding load, allowing a copied surface to be reseated.
    std::unordered_map<uint64_t, std::shared_ptr<const ContactSpringSet>> SpringSurfaceByKey;
    // Main-thread strip cache excluding footprint, allowing reuse across footprint changes.
    std::unordered_map<uint64_t, std::shared_ptr<const SpringStrips>> SpringStripsByKey;

    // Share moving-load sweep tables by (spring set, object, footprint).
    static constexpr uint32_t MaxSweepTables{8};
    std::array<SweepTableSlot, MaxSweepTables> SweepTables;
    std::unordered_map<uint64_t, uint32_t> SweepTableSlotByKey; // Content key to slot; main thread only.
    std::atomic<uint64_t> VoiceSweepMask{0}; // The sweep slots this callback's voices read.
    uint64_t ReusableSweepSlots{0}; // Reusable sweep slots this frame; main thread only.
};

static_assert(SurfaceAudioState::MaxSurfaceTracks == 8 * sizeof(decltype(SurfaceAudioState::VoiceTrackMask)::value_type));
static_assert(SurfaceAudioState::MaxContactSprings <= 8 * sizeof(decltype(SurfaceAudioState::VoiceSpringMask)::value_type));
static_assert(SurfaceAudioState::MaxSweepTables <= 8 * sizeof(decltype(SurfaceAudioState::VoiceSweepMask)::value_type));

struct VoiceBlock {
    double Refresh{0}, P1{1}, P2{1};
    float P{0}, A{0}, Dcn{0};
    bool Shot{false}, Moving{false};
    float LoadW{0}, CSite{0}, CutoffM{0}, PotCutM{0}, TaperS{1};
};

struct ModeReadGains {
    std::vector<float> Im, Re, Im2, Re2;

    void Resize(size_t n) {
        Im.resize(n);
        Re.resize(n);
        Im2.resize(n);
        Re2.resize(n);
    }
    void Fill(size_t at, float read, float c_re, float c_im) {
        Im[at] = read * c_re;
        Re[at] = read * c_im;
        Im2[at] = read * (c_re * c_re - c_im * c_im);
        Re2[at] = read * (2 * c_re * c_im);
    }
};

// Reuse renderer scratch across blocks to avoid steady-state allocation.
struct SurfaceRenderScratch {
    std::vector<uint32_t> Voices; // The object's own, gathered out of the flat voice lists.
    // Five direction rows per voice: normal, two geometric, frictional, and transverse frictional; then one row per impact.
    std::vector<float> DriveGains;
    ModeReadGains PointRead;
    std::vector<float> Forces; // This sample's force behind each drive row.
    std::vector<float> Excite; // This sample's excitation of each mode.
    std::vector<float> SweepExcite;
    std::vector<const RoughnessTrack *> Tracks; // Each voice's surface tracks, resolved once per block.
    std::vector<const RoughnessTrack *> TurnoverTracks; // Each voice's turnover noise track, likewise.
    std::vector<const ContactSpringSet *> SpringSets; // Each voice's element springs, likewise.
    std::vector<const SweepTableSet *> SweepSets; // Each voice's moving-load table, likewise.
    std::vector<float> SweepConformity; // Each voice's per-mode sweep conformity gain, fixed for the block.
    std::vector<VoiceBlock> VoiceBlocks; // Each voice's block-fixed constants, likewise.
    std::vector<uint32_t> BinRowBase; // Each voice's first distributed-excitation drive row.
    ModeReadGains BinRead;
    // Use one exchange channel per spring bin or one per unbinned voice.
    // Size channel arrays by the total count.
    std::vector<uint32_t> ChannelBase, ChannelCount;
    std::vector<float> ChSupport; // Each channel's share of its voice's solver support force.
    std::vector<float> SolvedForce; // Each voice's solved contact force this sample, for the sweep read.
    std::vector<float> NormalShape; // Per-voice per-mode normal shape projection, for the cross terms.
    std::vector<float> QuadCross; // Pairwise step response per newton of contact force, w-units, V x V.
    std::vector<float> QuadFeDefl; // Per-voice modal free response to the block's constant support forces.
    std::vector<float> QuadPsi, QuadG, QuadGamma, QuadFree, QuadPrevW, QuadS, QuadHx, QuadGNominal;
    std::vector<float> QuadDefl, QuadSlope;
    std::vector<double> QuadMat, QuadRhs;
    std::vector<uint8_t> QuadContact;
    std::vector<float> QuadSlam, QuadCn;
    ModeReadGains TangentRead;
    std::vector<float> QuadCt, QuadCnt, QuadFeT, QuadRhsT, QuadDeflT;
    std::vector<uint8_t> QuadTangent;
    // Transverse stiction state along normal cross slip for spring voices.
    ModeReadGains TransverseRead;
    std::vector<float> QuadCtTr, QuadCntTr, QuadRhsTr, QuadDeflTr;
    std::vector<vec3> TransverseDir; // Each voice's transverse direction, fixed for the block.
    std::vector<uint32_t> TransverseRow; // Each voice's appended transverse drive row, or none.
    std::vector<uint8_t> QuadTransverse;
};

inline SurfaceAudioState &Surface(ModalAudio &m) { return *m.Surface; }
inline const SurfaceAudioState &Surface(const ModalAudio &m) { return *m.Surface; }

// Mark unreferenced slots reusable on the main thread.
inline void BeginSurfaceTrackFrame(SurfaceAudioState &s) {
    auto named = s.VoiceTrackMask.load(std::memory_order_acquire);
    auto named_springs = s.VoiceSpringMask.load(std::memory_order_acquire);
    auto named_sweeps = s.VoiceSweepMask.load(std::memory_order_acquire);
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

// Return a main-thread pool slot for key, or -1 when all slots are referenced.
// make must return a shared_ptr.
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

// Release all pool slots on the main thread.
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

// Return an empty contact set with no callback readers, on the main thread.
VoiceSet &NextVoiceSet(SurfaceAudioState &);
// Publish the NextVoiceSet result on the main thread; omitted contacts end.
void PublishVoiceSet(SurfaceAudioState &);
