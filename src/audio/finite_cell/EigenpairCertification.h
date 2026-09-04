#pragma once

#include "audio/FiniteCell.h"

namespace modal::finite_cell {
struct EigenpairCertification {
    numeric::Vector<double> RelativeResiduals;
    double MassOrthogonalityError{};
};

EigenpairCertification CertifyEigenpairs(
    const FiniteCellOperator &, const numeric::Vector<double> &eigenvalues, const numeric::Matrix<double> &eigenvectors
);
} // namespace modal::finite_cell
