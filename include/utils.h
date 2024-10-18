#pragma once


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <Eigen/Dense>
#include "stb_image.h"

using Eigen::MatrixXd;


MatrixXd convertToGrayscale(const MatrixXd &red, const MatrixXd &green, const MatrixXd &blue);

bool loadImage(const char *imagePath, MatrixXd &red, MatrixXd &green, MatrixXd &blue, int &width, int &height);
