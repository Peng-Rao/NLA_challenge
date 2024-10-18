#include "utils.h"


// Function to convert RGB to grayscale
MatrixXd convertToGrayscale(const MatrixXd &red, const MatrixXd &green, const MatrixXd &blue) {
    return 0.299 * red + 0.587 * green + 0.114 * blue;
}
// Function to load an image
bool loadImage(const char *imagePath, MatrixXd &red, MatrixXd &green, MatrixXd &blue, int &width, int &height) {
    int channels;
    unsigned char *image_data = stbi_load(imagePath, &width, &height, &channels, 3);

    if (!image_data) {
        return false;
    }

    red = MatrixXd(height, width);
    green = MatrixXd(height, width);
    blue = MatrixXd(height, width);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = (i * width + j) * 3;
            red(i, j) = static_cast<double>(image_data[index]) / 255.0;
            green(i, j) = static_cast<double>(image_data[index + 1]) / 255.0;
            blue(i, j) = static_cast<double>(image_data[index + 2]) / 255.0;
        }
    }

    stbi_image_free(image_data);
    return true;
}
