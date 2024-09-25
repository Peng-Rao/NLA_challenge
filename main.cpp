#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

// Utility function to convert and clip values to the range [0, 255]
Matrix<unsigned char, Dynamic, Dynamic, RowMajor> convertToUnsignedChar(const MatrixXd &matrix) {
    return matrix.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(std::min(255.0, std::max(0.0, val))); // Clip values between 0 and 255
    });
}

// Function to convert RGB to grayscale
MatrixXd convertToGrayscale(const MatrixXd &red, const MatrixXd &green, const MatrixXd &blue) {
    return 0.299 * red + 0.587 * green + 0.114 * blue;
}

// Function to covert a image vector to a matrix
MatrixXd vectorToMatrix(const VectorXd &v, int height, int width) {
    return Map<const MatrixXd>(v.data(), height, width);
}

// Function to create a sparse matrix representing the A_avg 2 smoothing kernel
SparseMatrix<double> createAAvg2Matrix(int height, int width) {
    int size = height * width;
    std::vector<Triplet<double>> triplets;

    // 遍历每个像素，构造卷积矩阵
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j; // 当前像素在一维向量中的位置

            // 遍历3x3滤波器，向卷积矩阵中添加相应的值
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int ni = i + di; // 邻域的行
                    int nj = j + dj; // 邻域的列

                    // 确保邻域像素在图像范围内
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        int neighborIndex = ni * width + nj;
                        triplets.emplace_back(index, neighborIndex, 1.0 / 9.0);
                    }
                }
            }
        }
    }

    SparseMatrix<double> convMatrix(size, size);
    convMatrix.setFromTriplets(triplets.begin(), triplets.end());
    return convMatrix;
}

int main() {

    /**1.
     * Load the image as an Eigen matrix with size m × n.
     * Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere between 0 (black) and
     * 255 (white). Report the size of the matrix.
     */

    // Load the image as an Eigen matrix with size m × n.
    int width, height, channels;
    auto *input_image_path = "/Users/raopend/Workspace/NLA_ch1/Albert_Einstein_Head.jpg";
    unsigned char *image_data = stbi_load(input_image_path, &width, &height, &channels, 3); // Force load as grayscale

    if (!image_data) {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }
    // Prepare Eigen matrices for each RGB channel
    MatrixXd red(height, width), green(height, width), blue(height, width);
    // build the grayscale image matrix
    const Matrix<unsigned char, Dynamic, Dynamic> image_matrix(height, width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            const int index = (i * width + j) * 3; // 3 channels (RGB)
            red(i, j) = static_cast<double>(image_data[index]) / 255.0;
            green(i, j) = static_cast<double>(image_data[index + 1]) / 255.0;
            blue(i, j) = static_cast<double>(image_data[index + 2]) / 255.0;
        }
    }
    // Free memory!!!
    stbi_image_free(image_data);

    // Create a grayscale matrix
    const MatrixXd gray = convertToGrayscale(red, green, blue);
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image_matrix(height, width);
    // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
    grayscale_image_matrix =
            gray.unaryExpr([](const double val) -> unsigned char { return static_cast<unsigned char>(val * 255.0); });
    /**2.
     *Introduce a noise signal into the loaded image by adding random fluctuations of color
     *ranging between [−50, 50] to each pixel. Export the resulting image in .png and upload it.
     */
    // generate the random matrix, color ranging between -50 and 50
    MatrixXd noise_matrix = MatrixXd::Random(image_matrix.rows(), image_matrix.cols());
    noise_matrix = 50 * noise_matrix;
    // add the noise to the image matrix
    MatrixXd noisy_image_matrix = grayscale_image_matrix.cast<double>() + noise_matrix;
    // Save the grayscale image using stb_image_write
    const std::string output_image_path = "output_noisy.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1, convertToUnsignedChar(noisy_image_matrix).data(),
                       width) == 0) {
        std::cerr << "Error: Could not save grayscale image" << std::endl;
        return 1;
    }

    /**3.
     * Reshape the original image matrix and nosiy image matrix into vectors \vec{v} and \vec{w} respectively.
     * Verify that each vector has mn components. Report here the Euclidean norm of \vec{v}.
     */
    // Reshape the original image matrix and noisy  image matrix into vectors
    auto v = VectorXd{grayscale_image_matrix.cast<double>().reshaped()};
    // Verify that each vector has mn components
    assert(v.size() == grayscale_image_matrix.size());
    auto w = VectorXd{noisy_image_matrix.reshaped()};
    // Verify that each vector has mn components
    assert(w.size() == noisy_image_matrix.size());

    // Report here the Euclidean norm of \vec{v}
    std::cout << "The Euclidean norm of v is: " << v.norm() << std::endl;
    // Report here the Euclidean norm of \vec{w}
    std::cout << "The Euclidean norm of w is: " << w.norm() << std::endl;


    /**4.
     * Write the convolution operation corresponding to the smoothing kernel Hav2
     * as a matrix vector multiplication between a matrix A1 having size mn × mn and the image vector.
     * Report the number of non-zero entries in A1.
     */

    // Define the smoothing kernel Hav2
    // Define the matrix A1
    auto A1 = createAAvg2Matrix(width, height);
    std::cout << "The number of non-zero entries in A1 is: " << A1.nonZeros() << std::endl;

    /**5.
     * Apply the previous smoothing filter to the noisy image by performing the matrix vector multiplication A1w.
     * Export and upload the resulting image.
     */

    // Apply the smoothing filter to the noisy image
    auto smoothed_image = A1 * w;
    // Reshape the smoothed image vector to a matrix
    auto smoothed_image_matrix = vectorToMatrix(smoothed_image, height, width);
    // Save the smoothed image using stb_image_write
    const std::string smoothed_image_path = "output_smoothed.png";
    if (stbi_write_png(smoothed_image_path.c_str(), width, height, 1,
                       convertToUnsignedChar(smoothed_image_matrix).data(), width) == 0) {
        std::cerr << "Error: Could not save smoothed image" << std::endl;
        return 1;
    }

    /**6.
     * Write the convolution operation corresponding to the sharpening kernel Hsh2
     * as a matrix vector multiplication by a matrix A2 having size mn × mn. Report the number of non-zero
     * entries in A2. Is A2 symmetric?
     */

    return 0;
}
