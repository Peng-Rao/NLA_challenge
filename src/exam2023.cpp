#include "gmres.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>

#include <iostream>

int main(int argc, char *argv[]) {
    constexpr int n{1000};
    Eigen::SparseMatrix<double> A(n, n), B(n, n), C(n, n);

    for (int i = 0; i < n; i++) {
        B.coeffRef(i, i) = 2 * (i + 1);
        if (i > 0)
            B.coeffRef(i, i - 1) = i;
        if (i < n - 1)
            B.coeffRef(i, i + 1) = i + 1;

        C.coeffRef(i, n - i - 1) = -(i + 1);
    }

    A = B + C;

    Eigen::VectorXd b = A * Eigen::VectorXd::Ones(n);
    Eigen::VectorXd e = Eigen::VectorXd::Ones(n);
    Eigen::DiagonalPreconditioner<double> D(A);

    int max_iter{1000};
    double tol{1.e-8};
    int restart = 1000;
    Eigen::VectorXd x(n);

    LinearAlgebra::GMRES(A, x, b, D, restart, max_iter, tol);
    std::cout << "iterations performed: " << max_iter << std::endl;
    std::cout << "tolerance achieved  : " << tol << std::endl;
    std::cout << "Error:                " << (x - e).norm() << std::endl;

    return 0;
}
