#pragma once

#include <Eigen/SparseCore>

#include <memory>

// Shift-invert operator for Spectra, y = (K - sigma*M)^-1 x, backed by Accelerate's sparse Cholesky.
// The shift must be negative: K is positive semidefinite and M is positive definite, so
// K - sigma*M is then positive definite as Cholesky requires. Reads the lower triangles of K and M.
// Accumulates factorization and solve wall-clock time into the provided references.
class CholeskyShiftInvert {
public:
    using Scalar = double;

    CholeskyShiftInvert(const Eigen::SparseMatrix<double> &k, const Eigen::SparseMatrix<double> &m, double &factorize_seconds, double &solve_seconds);
    ~CholeskyShiftInvert();

    Eigen::Index rows() const { return K.rows(); }
    Eigen::Index cols() const { return K.cols(); }
    void set_shift(const Scalar &sigma);
    void perform_op(const Scalar *x_in, Scalar *y_out) const;
    // Solve across a column-major panel of `width` right-hand sides in one pass over the factor.
    void solve_panel(const Scalar *b_in, Scalar *x_out, int width) const;

private:
    const Eigen::SparseMatrix<double> &K, &M;
    double &FactorizeSeconds, &SolveSeconds;
    struct Factorization;
    std::unique_ptr<Factorization> Factor;
};
