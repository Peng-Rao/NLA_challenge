#pragma once
#include <Eigen/Sparse>

Eigen::MatrixXd convertToGrayscale(const Eigen::MatrixXd &red, const Eigen::MatrixXd &green,
                                   const Eigen::MatrixXd &blue);

bool loadImage(const char *imagePath, Eigen::MatrixXd &red, Eigen::MatrixXd &green, Eigen::MatrixXd &blue, int &width,
               int &height, int &channels);


Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> convertToUnsignedChar(const Eigen::MatrixXd &matrix);