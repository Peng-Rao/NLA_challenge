#pragma once
#include <Eigen/Sparse>

Eigen::MatrixXd convertToGrayscale(const Eigen::MatrixXd &red, const Eigen::MatrixXd &green,
                                   const Eigen::MatrixXd &blue);

bool loadImage(const char *imagePath, Eigen::MatrixXd &image_matrix, int &width, int &height, int &channels);


Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
convertToUnsignedChar(const Eigen::MatrixXd &matrix);

void exportMatrixMarketExtended(const Eigen::SparseMatrix<double> &mat, const Eigen::VectorXd &vec,
                                const std::string &filename);

bool saveMatrixMarketToImage(const std::string &inputFilePath, const std::string &outputFilePath, int height,
                             int width);

bool saveImage(const std::string &outputFilePath, const Eigen::MatrixXd &data, int height, int width);
