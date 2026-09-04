#include "EigenpairCertification.h"

#include <cmath>

modal::finite_cell::EigenpairCertification modal::finite_cell::CertifyEigenpairs(
    const FiniteCellOperator &operation, const Eigen::VectorXd &eigenvalues, const Eigen::MatrixXd &eigenvectors
) {
    EigenpairCertification result;
    if (eigenvectors.rows() != operation.Dofs() || eigenvectors.cols() != eigenvalues.size()) return result;
    Eigen::MatrixXd mass(operation.Dofs(), eigenvectors.cols()), stiffness(operation.Dofs(), eigenvectors.cols());
    // Alpha zero computes independent mass and stiffness actions in one traversal.
    operation.ApplyMassShifted(eigenvectors.data(), mass.data(), stiffness.data(), uint32_t(eigenvectors.cols()), 0);
    const Eigen::MatrixXd residual = stiffness - mass * eigenvalues.asDiagonal();
    result.RelativeResiduals.resize(eigenvalues.size());
    for (Eigen::Index mode = 0; mode < eigenvalues.size(); ++mode) {
        const double scale = stiffness.col(mode).norm() + std::abs(eigenvalues[mode]) * mass.col(mode).norm();
        result.RelativeResiduals[mode] = scale == 0 ? residual.col(mode).norm() : residual.col(mode).norm() / scale;
    }
    result.MassOrthogonalityError =
        (eigenvectors.transpose() * mass - Eigen::MatrixXd::Identity(eigenvalues.size(), eigenvalues.size())).norm();
    return result;
}
