#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

enum LogLevel { DEBUG, INFO, WARNING, ERROR };

class Logger {
public:
    // Constructor: Opens the log file in append mode
    explicit Logger(const std::string &filename) {
        logFile.open(filename, std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "Error opening log file." << std::endl;
        }
    }

    // Destructor: Closes the log file
    ~Logger() { logFile.close(); }

    // Logs a message with a given log level
    void log(LogLevel level, const std::string &message) {
        // Get current timestamp
        const time_t now = time(nullptr);
        const tm *timeinfo = localtime(&now);
        char timestamp[20];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", timeinfo);

        // Create log entry
        std::ostringstream logEntry;
        logEntry << "[" << timestamp << "] " << levelToString(level) << ": " << message << std::endl;

        // Output to console
        std::cout << logEntry.str();

        // Output to log file
        if (logFile.is_open()) {
            logFile << logEntry.str();
            logFile.flush(); // Ensure immediate write to file
        }
    }

private:
    std::ofstream logFile; // File stream for the log file

    // Converts log level to a string for output
    static std::string levelToString(LogLevel level) {
        switch (level) {
            case DEBUG:
                return "DEBUG";
            case INFO:
                return "INFO";
            case WARNING:
                return "WARNING";
            case ERROR:
                return "ERROR";
            default:
                return "UNKNOWN";
        }
    }
};

// Utility function to convert and clip values to the range [0, 255]
Matrix<unsigned char, Dynamic, Dynamic, RowMajor> convertToUnsignedChar(const MatrixXd &matrix) {
    return matrix.unaryExpr([](const double val) -> unsigned char {
        return static_cast<unsigned char>(std::min(255.0, std::max(0.0, val))); // Clip values between 0 and 255
    });
}

// Function to convert RGB to grayscale
MatrixXd convertToGrayscale(const MatrixXd &red, const MatrixXd &green, const MatrixXd &blue) {
    return 0.299 * red + 0.587 * green + 0.114 * blue;
}

// Function to create a sparse matrix representing the A_avg 2 smoothing kernel
SparseMatrix<double> createAAvg2Matrix(const int height, const int width) {
    const int size = height * width; // 图像的总像素数
    SparseMatrix<double> S(size, size);
    std::vector<Triplet<double>> tripletList;

    // 遍历每个像素
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int currentIndex = i * width + j;
            int neighbors = 0;

            // 添加邻域内像素的权重 (3x3 窗口)
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int ni = i + di; // 邻域像素的行索引
                    if (const int nj = j + dj; ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        int neighborIndex = ni * width + nj;
                        tripletList.emplace_back(currentIndex, neighborIndex, 1.0 / 9.0);
                        ++neighbors;
                    }
                }
            }
        }
    }

    // 构建稀疏矩阵
    S.setFromTriplets(tripletList.begin(), tripletList.end());

    return S;
}

// Function to create a sparse matrix representing the H_sh2 sharpening kernel
SparseMatrix<double> createHsh2Matrix(const int height, const int width) {
    const int size = height * width; // Total number of pixels in the image
    SparseMatrix<double> S(size, size);
    std::vector<Triplet<double>> tripletList;

    // Define the sharpening filter H_sh2
    constexpr int filter[3][3] = {{0, -3, 0}, {-1, 9, -3}, {0, -1, 0}};

    // Iterate over every pixel in the image
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int currentIndex = i * width + j;

            // Add weights of neighboring pixels (3x3 window)
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int ni = i + di; // Neighbor row index
                    int nj = j + dj; // Neighbor column index
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        int neighborIndex = ni * width + nj;
                        double weight = filter[di + 1][dj + 1]; // Adjust index to filter space
                        tripletList.emplace_back(currentIndex, neighborIndex, weight);
                    }
                }
            }
        }
    }

    // Build the sparse matrix from the triplet list
    S.setFromTriplets(tripletList.begin(), tripletList.end());

    return S;
}

int main() {
    // Initialize the logger
    Logger logger("log.txt");


    /**
     * Load the image as an Eigen matrix with size m × n.
     * Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere between 0 (black) and
     * 255 (white). Report the size of the matrix.
     */

    // Load the image as an Eigen matrix with size m × n.
    int width, height, channels;
    auto *input_image_path = "/Users/raopend/Workspace/NLA_ch1/photos/180px-Albert_Einstein_Head.jpg";
    unsigned char *image_data = stbi_load(input_image_path, &width, &height, &channels, 3); // Force load as grayscale

    if (!image_data) {
        logger.log(ERROR, "Could not load image");
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

    // Report the size of the matrix
    logger.log(INFO,
               "The size of the original image matrix is: " + std::to_string(height) + " x " + std::to_string(width));
    /**
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
        logger.log(ERROR, "Could not save noisy image");
        return 1;
    }
    logger.log(INFO, "Noisy image saved to: " + output_image_path);


    /**
     * Reshape the original image matrix and nosiy image matrix into vectors \vec{v} and \vec{w} respectively.
     * Verify that each vector has mn components. Report here the Euclidean norm of \vec{v}.
     */

    // Reshape the original image matrix and noisy  image matrix into vectors
    VectorXd v = grayscale_image_matrix.cast<double>().reshaped<RowMajor>().transpose();
    // Verify that each vector has mn components
    assert(v.size() == grayscale_image_matrix.size());
    // Verify that each vector has mn components
    VectorXd w = noisy_image_matrix.reshaped<RowMajor>().transpose();
    assert(w.size() == noisy_image_matrix.size());


    // Report here the Euclidean norm of \vec{v}
    logger.log(INFO, "The Euclidean norm of v is: " + std::to_string(v.norm()));
    // Report here the Euclidean norm of \vec{w}
    logger.log(INFO, "The Euclidean norm of w is: " + std::to_string(w.norm()));


    /**
     * Write the convolution operation corresponding to the smoothing kernel Hav2
     * as a matrix vector multiplication between a matrix A1 having size mn × mn and the image vector.
     * Report the number of non-zero entries in A1.
     */

    // Define the smoothing kernel Hav2
    // Define the matrix A1
    auto A1 = createAAvg2Matrix(height, width);
    logger.log(INFO, "The number of non-zero entries in A1 is: " + std::to_string(A1.nonZeros()));

    /**
     * Apply the previous smoothing filter to the noisy image by performing the matrix vector multiplication A1w.
     * Export and upload the resulting image.
     */

    // Apply the smoothing filter to the noisy image
    auto smoothed_image = A1 * w;
    // Reshape the smoothed image vector to a matrix
    auto smoothed_image_matrix = smoothed_image.reshaped<RowMajor>(height, width);
    // Save the smoothed image using stb_image_write
    const std::string smoothed_image_path = "output_smoothed.png";
    if (stbi_write_png(smoothed_image_path.c_str(), width, height, 1,
                       convertToUnsignedChar(smoothed_image_matrix).data(), width) == 0) {
        logger.log(ERROR, "Could not save smoothed image");
        return 1;
    }
    logger.log(INFO, "Smoothed image saved to: " + smoothed_image_path);

    /**
     * Write the convolution operation corresponding to the sharpening kernel Hsh2
     * as a matrix vector multiplication by a matrix A2 having size mn × mn. Report the number of non-zero
     * entries in A2. Is A2 symmetric?
     */
    auto A2 = createHsh2Matrix(height, width);
    logger.log(INFO, "The number of non-zero entries in A2 is: " + std::to_string(A2.nonZeros()));
    // apply the sharpening filter to the original image
    auto sharpened_image = A2 * v;
    // Reshape the sharpened image vector to a matrix
    auto sharpened_image_matrix = sharpened_image.reshaped<RowMajor>(height, width);
    // Save the sharpened image using stb_image_write
    const std::string sharpened_image_path = "output_sharpened.png";
    if (stbi_write_png(sharpened_image_path.c_str(), width, height, 1,
                       convertToUnsignedChar(sharpened_image_matrix).data(), width) == 0) {
        logger.log(ERROR, "Could not save sharpened image");
        return 1;
    }
    logger.log(INFO, "Sharpened image saved to: " + sharpened_image_path);
    return 0;
}
