#include <Eigen/SVD>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/SparseExtra>

#include <iostream>


int main(int argc, char *argv[]) {
    constexpr int n{100};
    Eigen::SparseMatrix<double> A(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A.coeffRef(i, j) = 1.0 / (i + j + 1);

    // SVD decomposition
    Eigen::BDCSVD<Eigen::MatrixXd> svd;
    svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd Sigma = svd.singularValues();
    std::cout << Sigma.nonZeros() << '\n';

    return 0;
}
