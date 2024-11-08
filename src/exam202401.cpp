#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include <iostream>

int main(int argc, char *argv[]) {
    int n{100};
    Eigen::SparseMatrix<double> A(n, n);

    for (int i = 0; i < n; i++) {
        A.coeffRef(i, i) = 8;
        if (i > 0)
            A.coeffRef(i, i - 1) = -2;
        if (i < n - 1)
            A.coeffRef(i, i + 1) = -4;
        if (i < n - 2)
            A.coeffRef(i, i + 2) = -1;
    }

    std::cout << A.norm() << std::endl; // Euclidean norm

    Eigen::MatrixXd Af;
    Af = Eigen::MatrixXd(A);
    Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver(Af);
    if (eigen_solver.info() != Eigen::Success) {
        abort();
    }
    // sort the eigenvalues
    Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues().real();
    std::sort(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());
    std::cout << eigenvalues(eigenvalues.size() - 1) << std::endl;

    saveMarket(A, "../exams_results/202401/Aex2.mtx");
    return 0;
}
