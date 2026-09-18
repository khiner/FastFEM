#pragma once

#include "numeric/Vector.h"

#include "audio/FiniteCell.h"

namespace modal::finite_cell {
using numeric::Matrix, numeric::Vector;

struct EigenpairCertification {
    Vector<double> RelativeResiduals;
    double MassOrthogonalityError{};
};

EigenpairCertification CertifyEigenpairs(
    const FiniteCellOperator &, const Vector<double> &eigenvalues, const Matrix<double> &eigenvectors
);
} // namespace modal::finite_cell
