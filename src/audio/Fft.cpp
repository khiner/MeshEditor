#include "Fft.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <memory>
#include <mutex>
#include <new>
#include <numbers>

namespace {
// vDSP supports complex lengths 2^k and {3, 5, 15}*2^k when the power-of-two factor is at least eight.
bool VdspComplexLength(uint32_t n) {
    if (n == 0) return false;
    uint32_t rest = n, twos = 0;
    for (; rest % 2 == 0; rest /= 2) ++twos;
    return rest == 1 || ((rest == 3 || rest == 5 || rest == 15) && twos >= 3);
}
// Real transforms require even lengths and a power-of-two factor of at least sixteen.
bool VdspRealLength(uint32_t n) {
    if (n == 0 || n % 2 != 0) return false;
    uint32_t rest = n, twos = 0;
    for (; rest % 2 == 0; rest /= 2) ++twos;
    return rest == 1 || ((rest == 3 || rest == 5 || rest == 15) && twos >= 4);
}

std::mutex SetupMutex;
vDSP_DFT_Setup MakeComplexSetup(uint32_t n, vDSP_DFT_Direction direction) {
    const std::lock_guard lock{SetupMutex};
    return vDSP_DFT_zop_CreateSetup(nullptr, n, direction);
}
vDSP_DFT_Setup MakeRealSetup(uint32_t n, vDSP_DFT_Direction direction) {
    const std::lock_guard lock{SetupMutex};
    return vDSP_DFT_zrop_CreateSetup(nullptr, n, direction);
}
void DestroySetup(vDSP_DFT_Setup setup) {
    const std::lock_guard lock{SetupMutex};
    vDSP_DFT_DestroySetup(setup);
}

template<class T> struct TransformAllocator {
    using value_type = T;
    static constexpr std::align_val_t Alignment{64};

    TransformAllocator() = default;
    template<class U> constexpr TransformAllocator(const TransformAllocator<U> &) {}

    T *allocate(size_t n) { return static_cast<T *>(::operator new(n * sizeof(T), Alignment)); }
    void deallocate(T *p, size_t) { ::operator delete(p, Alignment); }
    bool operator==(const TransformAllocator &) const { return true; }
};

// vDSP reads and writes the real and imaginary halves of a complex array separately.
struct Split {
    explicit Split(uint32_t n) : Re(n, 0.f), Im(n, 0.f) {}
    std::vector<float, TransformAllocator<float>> Re, Im;
};

// Performs a complex transform directly or through Bluestein convolution for unsupported vDSP lengths.
struct ComplexDft {
    ComplexDft(uint32_t n, vDSP_DFT_Direction direction) : N(n) {
        if (VdspComplexLength(n)) Direct = MakeComplexSetup(n, direction);
        if (Direct) return;

        M = std::max(8u, std::bit_ceil(2 * n - 1));
        Forward = MakeComplexSetup(M, vDSP_DFT_FORWARD);
        Inverse = MakeComplexSetup(M, vDSP_DFT_INVERSE);
        // Compute e^(sign*i*pi*j^2/n) with j^2 modulo 2n to preserve angle precision.
        const double sign = direction == vDSP_DFT_FORWARD ? -1. : 1.;
        Chirp = std::make_unique<Split>(n);
        Split cyclic{M};
        for (uint32_t j = 0; j < n; ++j) {
            const double angle = sign * std::numbers::pi * double(uint64_t(j) * j % (2 * uint64_t(n))) / double(n);
            Chirp->Re[j] = float(std::cos(angle));
            Chirp->Im[j] = float(std::sin(angle));
            cyclic.Re[j] = Chirp->Re[j];
            cyclic.Im[j] = -Chirp->Im[j];
            // The kernel is even in j, so its negative half wraps to the top of the convolution.
            if (j > 0) {
                cyclic.Re[M - j] = cyclic.Re[j];
                cyclic.Im[M - j] = cyclic.Im[j];
            }
        }
        Kernel = std::make_unique<Split>(M);
        vDSP_DFT_Execute(Forward, cyclic.Re.data(), cyclic.Im.data(), Kernel->Re.data(), Kernel->Im.data());
    }
    ~ComplexDft() {
        DestroySetup(Direct);
        DestroySetup(Forward);
        DestroySetup(Inverse);
    }
    ComplexDft(const ComplexDft &) = delete;
    ComplexDft &operator=(const ComplexDft &) = delete;

    void operator()(const float *in_re, const float *in_im, float *out_re, float *out_im) const {
        if (Direct) return vDSP_DFT_Execute(Direct, in_re, in_im, out_re, out_im);

        Split chirped{M}, convolved{M};
        for (uint32_t j = 0; j < N; ++j) {
            chirped.Re[j] = in_re[j] * Chirp->Re[j] - in_im[j] * Chirp->Im[j];
            chirped.Im[j] = in_re[j] * Chirp->Im[j] + in_im[j] * Chirp->Re[j];
        }
        vDSP_DFT_Execute(Forward, chirped.Re.data(), chirped.Im.data(), convolved.Re.data(), convolved.Im.data());
        for (uint32_t m = 0; m < M; ++m) {
            const float re = convolved.Re[m] * Kernel->Re[m] - convolved.Im[m] * Kernel->Im[m];
            convolved.Im[m] = convolved.Re[m] * Kernel->Im[m] + convolved.Im[m] * Kernel->Re[m];
            convolved.Re[m] = re;
        }
        vDSP_DFT_Execute(Inverse, convolved.Re.data(), convolved.Im.data(), chirped.Re.data(), chirped.Im.data());
        const float scale = 1.f / float(M);
        for (uint32_t k = 0; k < N; ++k) {
            const float re = chirped.Re[k] * scale, im = chirped.Im[k] * scale;
            out_re[k] = re * Chirp->Re[k] - im * Chirp->Im[k];
            out_im[k] = re * Chirp->Im[k] + im * Chirp->Re[k];
        }
    }

    uint32_t N;
    vDSP_DFT_Setup Direct{};
    uint32_t M{}; // Convolution length of the chirp path.
    vDSP_DFT_Setup Forward{}, Inverse{};
    std::unique_ptr<Split> Chirp, Kernel;
};

struct RealDft {
    RealDft(uint32_t n, vDSP_DFT_Direction direction) : N(n) {
        if (VdspRealLength(n)) Packed = MakeRealSetup(n, direction);
        if (!Packed) Full = std::make_unique<ComplexDft>(n, direction);
    }
    ~RealDft() { DestroySetup(Packed); }
    RealDft(const RealDft &) = delete;
    RealDft &operator=(const RealDft &) = delete;

    void Forward(std::span<const float> in, std::span<std::complex<float>> out) const {
        const uint32_t bins = N / 2 + 1;
        if (Packed) {
            Split signal{N / 2}, spectrum{N / 2};
            for (uint32_t j = 0; j < N / 2; ++j) {
                signal.Re[j] = in[2 * j];
                signal.Im[j] = in[2 * j + 1];
            }
            vDSP_DFT_Execute(Packed, signal.Re.data(), signal.Im.data(), spectrum.Re.data(), spectrum.Im.data());
            out[0] = {spectrum.Re[0] * 0.5f, 0.f};
            out[bins - 1] = {spectrum.Im[0] * 0.5f, 0.f};
            for (uint32_t k = 1; k < N / 2; ++k) out[k] = {spectrum.Re[k] * 0.5f, spectrum.Im[k] * 0.5f};
            return;
        }
        Split signal{N}, spectrum{N};
        std::ranges::copy(in, signal.Re.begin());
        (*Full)(signal.Re.data(), signal.Im.data(), spectrum.Re.data(), spectrum.Im.data());
        for (uint32_t k = 0; k < bins; ++k) out[k] = {spectrum.Re[k], spectrum.Im[k]};
    }

    void Inverse(std::span<const std::complex<float>> in, std::span<float> out) const {
        const uint32_t bins = N / 2 + 1;
        if (Packed) {
            Split spectrum{N / 2}, signal{N / 2};
            spectrum.Re[0] = in[0].real();
            spectrum.Im[0] = in[bins - 1].real();
            for (uint32_t k = 1; k < N / 2; ++k) {
                spectrum.Re[k] = in[k].real();
                spectrum.Im[k] = in[k].imag();
            }
            vDSP_DFT_Execute(Packed, spectrum.Re.data(), spectrum.Im.data(), signal.Re.data(), signal.Im.data());
            for (uint32_t j = 0; j < N / 2; ++j) {
                out[2 * j] = signal.Re[j];
                out[2 * j + 1] = signal.Im[j];
            }
            return;
        }
        Split spectrum{N}, signal{N};
        spectrum.Re[0] = in[0].real();
        for (uint32_t k = 1; k < bins; ++k) {
            spectrum.Re[k] = in[k].real();
            spectrum.Im[k] = in[k].imag();
        }
        if (N % 2 == 0) spectrum.Im[bins - 1] = 0.f;
        for (uint32_t k = bins; k < N; ++k) {
            spectrum.Re[k] = spectrum.Re[N - k];
            spectrum.Im[k] = -spectrum.Im[N - k];
        }
        (*Full)(spectrum.Re.data(), spectrum.Im.data(), signal.Re.data(), signal.Im.data());
        std::ranges::copy(signal.Re, out.begin());
    }

    uint32_t N;
    vDSP_DFT_Setup Packed{};
    std::unique_ptr<ComplexDft> Full;
};
} // namespace

namespace fft {
std::vector<std::complex<float>> RealToComplex(std::span<const float> in) {
    if (in.empty()) return {};

    const auto n = uint32_t(in.size());
    std::vector<std::complex<float>> out(n / 2 + 1);
    RealDft{n, vDSP_DFT_FORWARD}.Forward(in, out);
    return out;
}

void ComplexToReal(std::span<const std::complex<float>> in, std::span<float> out) {
    if (out.empty()) return;

    RealDft{uint32_t(out.size()), vDSP_DFT_INVERSE}.Inverse(in, out);
}

uint32_t DirectLength(uint32_t n) {
    uint32_t length = std::max(n, 2u);
    while (!VdspRealLength(length)) ++length;
    return length;
}

void ComplexToReal2d(std::span<const std::complex<float>> in, uint32_t columns, uint32_t rows, std::span<float> out) {
    if (columns == 0 || rows == 0) return;

    // Transform columns first so each row contains one real signal's bins.
    const uint32_t bins = rows / 2 + 1;
    std::vector<std::complex<float>> mixed(size_t(columns) * bins);
    {
        const ComplexDft along_columns{columns, vDSP_DFT_INVERSE};
        Split column{columns}, transformed{columns};
        for (uint32_t j = 0; j < bins; ++j) {
            for (uint32_t i = 0; i < columns; ++i) {
                column.Re[i] = in[size_t(i) * bins + j].real();
                column.Im[i] = in[size_t(i) * bins + j].imag();
            }
            along_columns(column.Re.data(), column.Im.data(), transformed.Re.data(), transformed.Im.data());
            for (uint32_t i = 0; i < columns; ++i) mixed[size_t(i) * bins + j] = {transformed.Re[i], transformed.Im[i]};
        }
    }
    const RealDft along_rows{rows, vDSP_DFT_INVERSE};
    for (uint32_t i = 0; i < columns; ++i) {
        along_rows.Inverse({mixed.data() + size_t(i) * bins, bins}, {out.data() + size_t(i) * rows, rows});
    }
}
} // namespace fft
