#include "CholeskyShiftInvert.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>

namespace {
double SecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
} // namespace

struct CholeskyShiftInvert::Factorization {
    SparseOpaqueFactorization_Double Opaque;
    ~Factorization() { SparseCleanup(Opaque); }
};

CholeskyShiftInvert::CholeskyShiftInvert(const Eigen::SparseMatrix<double> &k, const Eigen::SparseMatrix<double> &m, double &factorize_seconds, double &solve_seconds)
    : K(k), M(m), FactorizeSeconds(factorize_seconds), SolveSeconds(solve_seconds) {}

CholeskyShiftInvert::~CholeskyShiftInvert() = default;

void CholeskyShiftInvert::set_shift(const Scalar &sigma) {
    const auto start = std::chrono::steady_clock::now();
    Eigen::SparseMatrix<double> shifted = K - sigma * M;
    // Accelerate reads CSC arrays, matching Eigen's layout apart from the long column starts.
    std::vector<long> column_starts(shifted.cols() + 1);
    std::copy_n(shifted.outerIndexPtr(), column_starts.size(), column_starts.begin());
    const SparseMatrix_Double shifted_matrix{
        .structure = {
            .rowCount = int(shifted.rows()),
            .columnCount = int(shifted.cols()),
            .columnStarts = column_starts.data(),
            .rowIndices = shifted.innerIndexPtr(),
            .attributes = {.triangle = SparseLowerTriangle, .kind = SparseSymmetric},
            .blockSize = 1,
        },
        .data = shifted.valuePtr(),
    };
    Factor = std::make_unique<Factorization>(SparseFactor(SparseFactorizationCholesky, shifted_matrix));
    if (Factor->Opaque.status != SparseStatusOK) throw std::runtime_error("Modal shift-invert factorization failed.");
    FactorizeSeconds += SecondsSince(start);
}

void CholeskyShiftInvert::perform_op(const Scalar *x_in, Scalar *y_out) const {
    const auto start = std::chrono::steady_clock::now();
    const int n = int(K.rows());
    SparseSolve(Factor->Opaque, DenseVector_Double{n, const_cast<double *>(x_in)}, DenseVector_Double{n, y_out});
    SolveSeconds += SecondsSince(start);
}
