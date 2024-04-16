#pragma once

#include <fftw3.h>

struct FFTData {
    fftwf_plan Plan;
    fftwf_complex *Complex;

    FFTData(float *data, const uint size) {
        Complex = fftwf_alloc_complex(size / 2 + 1);
        Plan = fftwf_plan_dft_r2c_1d(size, data, Complex, FFTW_MEASURE);
        fftwf_execute(Plan);
    }

    ~FFTData() {
        fftwf_destroy_plan(Plan);
        fftwf_free(Complex);
    }
};
