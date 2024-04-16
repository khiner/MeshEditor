#pragma once

#include <vector>

#include <fftw3.h>

struct FFTData {
    fftwf_plan Plan;
    fftwf_complex *Complex;

    FFTData(std::vector<float> &data) {
        Complex = fftwf_alloc_complex(data.size() / 2 + 1);
        Plan = fftwf_plan_dft_r2c_1d(data.size(), data.data(), Complex, FFTW_MEASURE);
        fftwf_execute(Plan);
    }

    ~FFTData() {
        fftwf_destroy_plan(Plan);
        fftwf_free(Complex);
    }
};
