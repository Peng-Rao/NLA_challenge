#include "utils.h"

#include <Eigen/Core>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>
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
    // Create a grayscale matrix
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> grayscale_image_matrix =
            convertToGrayscale(red, green, blue).unaryExpr([](const double val) -> unsigned char {
                return static_cast<unsigned char>(val * 255.0);
            });

    // Report the size of the matrix
    PLOG_INFO << "The size of the original image matrix is: " + std::to_string(height) + " x " + std::to_string(width);
    return 0;
}
