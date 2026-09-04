#pragma once

#include "audio/FiniteCell.h"

#include <Eigen/Core>

namespace modal::finite_cell {
struct EigenpairCertification {
    Eigen::VectorXd RelativeResiduals;
    double MassOrthogonalityError{};
};

EigenpairCertification CertifyEigenpairs(
    const FiniteCellOperator &, const Eigen::VectorXd &eigenvalues, const Eigen::MatrixXd &eigenvectors
);
} // namespace modal::finite_cell
