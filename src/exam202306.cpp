#include "cg.hpp"

#include <Eigen/SparseCore>

#include <Eigen/IterativeLinearSolvers>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>

int main(int argc, char *argv[]) {
    constexpr int n{800};
    Eigen::SparseMatrix<double> A(n, n);

    for (int i = 0; i < n; ++i) {
        A.coeffRef(i, i) = 4;
        if (i > 0)
            A.coeffRef(i, i - 1) = -1;

        if (i > 1)
            A.coeffRef(i, i - 2) = -1;

        if (i < n - 1)
            A.coeffRef(i, i + 1) = -1;

        if (i < n - 2)
            A.coeffRef(i, i + 2) = -1;
    }

    Eigen::VectorXd v = Eigen::VectorXd::Ones(n);
    std::cout << (v.transpose() * A * v).norm() << '\n';

    Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 1)
            b(i) = 0;
    }
    std::cout << b.norm() << '\n';

    int max_iter{1000};
    double tol{1.e-12};
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::DiagonalPreconditioner<double> D(A);

    LinearAlgebra::CG(A, x, b, D, max_iter, tol);
    std::cout << "iterations performed: " << max_iter << '\n';
    std::cout << "tolerance achieved  : " << tol << '\n';

    saveMarket(A, "../exams_results/202306/matA.mtx");
    FILE *out = fopen("../exams_results/202306/vecB.mtx", "w");
    fprintf(out, "%%%%MatrixMarket vector coordinate real general\n");
    fprintf(out, "%d\n", n);
    for (int i = 0; i < n; i++) {
        fprintf(out, "%d %f\n", i, b(i));
    }
    fclose(out);
    return 0;
}
