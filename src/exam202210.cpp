#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/SparseExtra>

#include <cmath>
#include <iostream>

int main(int argc, char *argv[]) {
    int n{99};
    Eigen::SparseMatrix<double> A(n, n);

    for (int i = 0; i < n; i++) {
        A.coeffRef(i, i) = std::abs((n + 1) / 2 - i - 1) + 1.0;

        if (i < n - 1)
            A.coeffRef(i, i + 1) = 0.5;

        if (i > 0)
            A.coeffRef(i, i - 1) = 0.5;
    }

    std::cout << A.coeffRef(0, 0) << ' ' << A.coeffRef(n - 1, n - 1) << '\n';

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success)
        abort();

    auto eigenvalues = solver.eigenvalues().real();
    std::cout << eigenvalues(0) << " " << eigenvalues(n - 1);

    saveMarket(A, "../exams_results/202210/exer2.mtx");

    return 0;
}
