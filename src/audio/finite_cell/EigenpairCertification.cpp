#include "numeric/Vector.h"

#include "EigenpairCertification.h"

#include <cmath>

using numeric::Matrix, numeric::Vector;

modal::finite_cell::EigenpairCertification modal::finite_cell::CertifyEigenpairs(
    const FiniteCellOperator &operation, const Vector<double> &eigenvalues, const Matrix<double> &eigenvectors
) {
    EigenpairCertification result;
    if (eigenvectors.rows() != operation.Dofs() || eigenvectors.cols() != eigenvalues.size()) return result;
    Matrix<double> mass(operation.Dofs(), eigenvectors.cols()), stiffness(operation.Dofs(), eigenvectors.cols());
    // Alpha zero computes independent mass and stiffness actions in one traversal.
    operation.ApplyMassShifted(eigenvectors.data(), mass.data(), stiffness.data(), uint32_t(eigenvectors.cols()), 0);
    const Matrix<double> residual = ColumnScaledDifference(stiffness.View(), mass.View(), eigenvalues.View());
    result.RelativeResiduals.Resize(eigenvalues.size());
    for (size_t mode = 0; mode < eigenvalues.size(); ++mode) {
        const double scale = Norm(stiffness.Column(mode)) + std::abs(eigenvalues[mode]) * Norm(mass.Column(mode));
        const double residual_norm = Norm(residual.Column(mode));
        result.RelativeResiduals[mode] = scale == 0 ? residual_norm : residual_norm / scale;
    }
    Matrix<double> gram = TransposeMultiply(eigenvectors.View(), mass.View());
    for (size_t mode = 0; mode < eigenvalues.size(); ++mode) gram(mode, mode) -= 1;
    result.MassOrthogonalityError = Norm(gram.View());
    return result;
}
