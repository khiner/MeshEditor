#pragma once

#include <Eigen/SparseCore>

#include <memory>

// Applies y = (K - sigma*M)^-1 x for Spectra through Accelerate sparse Cholesky.
// Requires negative sigma because K is positive semidefinite and M is positive definite.
// Reads the lower triangles and accumulates factorization and solve times into the supplied references.
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
