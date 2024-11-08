#include "jacobi.hpp"

#include <Eigen/SparseCore>
#include <unsupported/Eigen/SparseExtra>

#include <iostream>


int main(int argc, char *argv[]) {
    int n = 100;
    Eigen::SparseMatrix<double> A(n, n);

    A.reserve(298);

    for (int i = 0; i < n; i++) {
        A.coeffRef(i, i) = 2.0;
        if (i > 0)
            A.coeffRef(i, i - 1) = -1.0;
        if (i < n - 1)
            A.coeffRef(i, i + 1) = -1.0;
    }
    // The norm of A
    std::cout << A.norm() << '\n';

    // check if the matrix A is symmetric
    std::cout << (A - Eigen::SparseMatrix<double>(A.transpose())).norm() << '\n';

    // define the vector b
    Eigen::VectorXd b = Eigen::VectorXd::Ones(A.rows());

    // define preconditioner
    Eigen::DiagonalPreconditioner<double> D(A);

    // Jacobi method
    int result, max_iter{10000};
    double tol{1.e-5};
    Eigen::VectorXd x(A.rows());
    LinearAlgebra::Jacobi(A, x, b, D, max_iter, tol);
    std::cout << "Jacobi method " << '\n';
    std::cout << "iterations performed " << max_iter << '\n';
    std::cout << "tolerance achieved " << tol << '\n';
    std::cout << "norm of computed sol " << x.norm() << '\n';
    std::cout << "residual " << (b - A * x).norm() << '\n';


    // QR Factorization
    A.makeCompressed();
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cout << "cannot factorize the matrix" << std::endl;
        return 0;
    }
    x = solver.solve(b);
    std::cout << "Solution with Eigen QR:" << std::endl;
    return 0;
}
