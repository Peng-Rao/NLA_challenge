#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>


int main(int argc, char *argv[]) {
    constexpr int n{1000};
    Eigen::SparseMatrix<double> A(n, n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A.coeffRef(i, j) = 1.0 / (i + j + 1);

    // check if symmetric
    std::cout << "Norm of A-A.t: " << (A - Eigen::SparseMatrix<double>(A.transpose())).norm() << std::endl;

    Eigen::VectorXd b = A * Eigen::VectorXd::Ones(n);
    Eigen::VectorXd e = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd x(n);

    // Direct Cholesky
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_ldlt;
    solver_ldlt.compute(A);
    if (solver_ldlt.info() != Eigen::Success) {
        std::cout << "cannot factorize the matrix using LLT" << std::endl;
    } else {
        x = solver_ldlt.solve(b);
        std::cout << "effective error direct LDLT: " << (x - e).norm() << std::endl;
    }


    // Direct LU
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solvelu;
    solvelu.compute(A);
    if (solvelu.info() != Eigen::Success) {
        std::cout << "cannot factorize the matrix using LU" << std::endl;
    } else {
        x = solvelu.solve(b);
        std::cout << "effective error direct LU factorization: " << (x - e).norm() << std::endl;
    }


    // Conjugate Gradient
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
    Eigen::DiagonalPreconditioner<double> D(A);
    cg.setMaxIterations(1000);
    cg.setTolerance(1.e-8);
    cg.compute(A);
    if (cg.info() != Eigen::Success) {
        std::cout << "cannot factorize the matrix using cg" << std::endl;
    } else {
        x = cg.solve(b);
        std::cout << "#iterations:     " << cg.iterations() << std::endl;
        std::cout << "relative residual: " << cg.error() << std::endl;
        std::cout << "effective error cg: " << (x - e).norm() << std::endl;
    }

    saveMarket(A, "../exams_results/202301/A_1.mtx");

    return 0;
}
