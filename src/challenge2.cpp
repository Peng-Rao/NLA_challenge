#include "utils.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>
#include <string>
#include <unsupported/Eigen/SparseExtra>

int main(int argc, char *argv[]) {

    // Initialize the logger
    plog::init(plog::debug, "../ch2_result/log.txt");

    // Load the iamge as matrix A with size m times n
    int width, height, channels;
    auto *image_input_path = "/Users/raopend/Workspace/NLA_challenge/photos/256px-Albert_Einstein_Head.jpg";
    Eigen::MatrixXd red(height, width), green(height, width), blue(height, width);

    if (loadImage(image_input_path, red, green, blue, width, height, channels)) {
        PLOG_INFO << "Image loaded successfully.";
    } else {
        PLOG_ERROR << "Failed to load the image.";
    }

    // build the grayscale image matrix
    const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> grayscale_image_matrix =
            convertToGrayscale(red, green, blue).unaryExpr([](const double val) -> unsigned char {
                return static_cast<unsigned char>(val * 255.0);
            });

    // Report the size of the matrix
    PLOG_INFO << "The size of the original image matrix is: " + std::to_string(height) + " x " + std::to_string(width);

    Eigen::MatrixXd A = grayscale_image_matrix.cast<double>();

    // compute Gram matrix
    const Eigen::MatrixXd gram_matrix = A.transpose() * A;

    // Report the euclidean norm of the Gram matrix
    const double euclidean_norm = gram_matrix.norm();
    PLOG_INFO << "The Euclidean norm of the Gram matrix is: " + std::to_string(euclidean_norm);

    // Solve the eigenvalues of the Gram matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(gram_matrix);
    const Eigen::VectorXd eigenvalues = es.eigenvalues().real();
    const Eigen::MatrixXd eigenvectors = es.eigenvectors().real();
    // Report the two largest eigenvalues
    // Sort the eigenvalues in descending order

    // Report the two largest eigenvalues
    PLOG_INFO << "The two largest eigenvalues are: " + std::to_string(eigenvalues(eigenvalues.size() - 1)) + " and " +
                         std::to_string(eigenvalues(eigenvalues.size() - 2));

    // Export gram matrix to a .mtx file
    saveMarket(gram_matrix, "../ch2_result/gram_matrix.mtx");

    // perform a singular value decomposition of thematrix A.
    Eigen::JacobiSVD<Eigen::MatrixXd> jcb_svd;
    jcb_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    PLOG_INFO << "The largest singular values of the matrix A are: " + std::to_string(jcb_svd.singularValues()(0)) +
                         " and " + std::to_string(jcb_svd.singularValues()(1)) + " using Jacobi SVD.";
    Eigen::BDCSVD<Eigen::MatrixXd> bdcs_svd;
    bdcs_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    PLOG_INFO << "The largest singular values of the matrix A are: " + std::to_string(bdcs_svd.singularValues()(0)) +
                         " and " + std::to_string(bdcs_svd.singularValues()(1)) + " using BDC SVD.";
    // Get the diagonal matrix \Sigma
    const Eigen::MatrixXd sigma = bdcs_svd.singularValues().asDiagonal();
    // Report the norm of \Sigma
    PLOG_INFO << "The norm of the diagonal matrix Sigma is: " + std::to_string(sigma.norm());
    return 0;
}
