#include "utils.h"
#include <Eigen/src/Core/util/Constants.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

// Function to convert RGB to grayscale
Eigen::MatrixXd convertToGrayscale(const Eigen::MatrixXd &red, const Eigen::MatrixXd &green,
                                   const Eigen::MatrixXd &blue) {
    return 0.299 * red + 0.587 * green + 0.114 * blue;
}

// Function to load an image
bool loadImage(const char *imagePath, Eigen::MatrixXd &red, Eigen::MatrixXd &green, Eigen::MatrixXd &blue, int &width,
               int &height, int &channels) {
    unsigned char *image_data = stbi_load(imagePath, &width, &height, &channels, 3);

    if (!image_data) {
        return false;
    }

    red = Eigen::MatrixXd(height, width);
    green = Eigen::MatrixXd(height, width);
    blue = Eigen::MatrixXd(height, width);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            const int index = (i * width + j) * 3;
            red(i, j) = static_cast<double>(image_data[index]) / 255.0;
            green(i, j) = static_cast<double>(image_data[index + 1]) / 255.0;
            blue(i, j) = static_cast<double>(image_data[index + 2]) / 255.0;
        }
    }

    stbi_image_free(image_data);
    return true;
}

// FUnction to cast Eigen::MatrixXd to unsigned char
Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> convertToUnsignedChar(const Eigen::MatrixXd &matrix) {
    return matrix.unaryExpr([](const double val) -> unsigned char {
        return static_cast<unsigned char>(std::min(255.0, std::max(0.0, val))); // Clip values between 0 and 255
    });
}
