#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lis.h"
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

// Function to count only the truly non-zero elements in a sparse matrix
template<typename T>
int countNonZeroElements(const SparseMatrix<T> &matrix) {
    int count = 0;
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (typename SparseMatrix<T>::InnerIterator it(matrix, k); it; ++it) {
            if (it.value() != 0.0) { // Only count true non-zero elements
                ++count;
            }
        }
    }
    return count;
}

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
    const long size = height * width;
    SparseMatrix<double> S(size, size);
    std::vector<Triplet<double>> tripletList;
    tripletList.reserve(size * 9); // Each pixel has up to 9 neighbors

    // Iterate over every pixel in the image
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            long currentIndex = i * width + j;

            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int ni = i + di;
                    if (const int nj = j + dj; ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        int neighborIndex = ni * width + nj;
                        tripletList.emplace_back(currentIndex, neighborIndex, 1.0 / 9.0);
                    }
                }
            }
        }
    }

    S.setFromTriplets(tripletList.begin(), tripletList.end());

    return S;
}

// Function to create a sparse matrix representing the A_avg 2 smoothing kernel using matrix shifts
SparseMatrix<double> createAAvg2MatrixOptimized(const int height, const int width) {
    const int size = height * width;
    SparseMatrix<double> S(size, size);

    // Identity matrix to represent the base image (center pixel of the kernel)
    SparseMatrix<double> I(size, size);
    I.setIdentity();

    // Define shifts in 8 different directions (left, right, up, down, diagonals)
    SparseMatrix<double> shiftUp(size, size), shiftDown(size, size);
    SparseMatrix<double> shiftLeft(size, size), shiftRight(size, size);
    SparseMatrix<double> shiftUpLeft(size, size), shiftUpRight(size, size);
    SparseMatrix<double> shiftDownLeft(size, size), shiftDownRight(size, size);

    // Shift by one row (up and down)
    shiftUp.reserve(size);
    shiftDown.reserve(size);
    for (int i = width; i < size; ++i) {
        shiftUp.insert(i, i - width) = 1.0;
        shiftDown.insert(i - width, i) = 1.0;
    }

    // Shift by one column (left and right)
    shiftLeft.reserve(size);
    shiftRight.reserve(size);
    for (int i = 1; i < size; ++i) {
        if (i % width != 0) { // avoid crossing row boundaries
            shiftLeft.insert(i, i - 1) = 1.0;
            shiftRight.insert(i - 1, i) = 1.0;
        }
    }

    // Diagonal shifts (up-left, up-right, down-left, down-right)
    shiftUpLeft.reserve(size);
    shiftUpRight.reserve(size);
    shiftDownLeft.reserve(size);
    shiftDownRight.reserve(size);
    for (int i = width; i < size; ++i) {
        if (i % width != 0) { // avoid crossing row boundaries
            shiftUpLeft.insert(i - width - 1, i) = 1.0;
            shiftDownRight.insert(i, i - width - 1) = 1.0;
        }
        if (i % width != width - 1) { // avoid crossing row boundaries
            shiftUpRight.insert(i - width + 1, i) = 1.0;
            shiftDownLeft.insert(i, i - width + 1) = 1.0;
        }
    }

    // Combine all shifted matrices with equal weights (1/9 for average smoothing)
    S = (I + shiftUp + shiftDown + shiftLeft + shiftRight + shiftUpLeft + shiftUpRight + shiftDownLeft +
         shiftDownRight) /
        (1.0 / 9.0);

    return S;
}

// Function to create a sparse matrix representing the H_sh2 sharpening kernel
SparseMatrix<double> createHsh2Matrix(const int height, const int width) {
    const long size = height * width; // Total number of pixels in the image
    SparseMatrix<double> S(size, size);
    std::vector<Triplet<double>> tripletList;
    tripletList.reserve(size * 9); // Each pixel has up to 9 neighbors

    // Define the sharpening filter H_sh2
    constexpr int filter[3][3] = {{0, -3, 0}, {-1, 9, -3}, {0, -1, 0}};

    // Iterate over every pixel in the image
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            long currentIndex = i * width + j;

            // Add weights of neighboring pixels (3x3 window)
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int ni = i + di; // Neighbor row index
                    if (const int nj = j + dj; ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        long neighborIndex = ni * width + nj;
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

// Function to create a sparse matrix representing the H_sh2 sharpening kernel using matrix shifts
SparseMatrix<double> createHsh2MatrixOptimized(const int height, const int width) {
    const int size = height * width; // Total number of pixels
    SparseMatrix<double> S(size, size);

    // Identity matrix to represent the base image (center element in the filter)
    SparseMatrix<double> I(size, size);
    I.setIdentity();

    // Define shifts in 8 different directions (up, down, left, right, diagonals)
    SparseMatrix<double> shiftUp(size, size), shiftDown(size, size);
    SparseMatrix<double> shiftLeft(size, size), shiftRight(size, size);

    // Shift by one row (up and down)
    shiftUp.reserve(size);
    shiftDown.reserve(size);
    for (int i = width; i < size; ++i) {
        shiftUp.insert(i, i - width) = 1.0;
        shiftDown.insert(i - width, i) = 1.0;
    }

    // Shift by one column (left and right)
    shiftLeft.reserve(size);
    shiftRight.reserve(size);
    for (int i = 1; i < size; ++i) {
        if (i % width != 0) {
            shiftLeft.insert(i, i - 1) = 1.0;
            shiftRight.insert(i - 1, i) = 1.0;
        }
    }


    // Apply weights from the sharpening filter H_sh2
    S = I * 9.0 // center pixel weight
        + shiftUp * (-1.0) + shiftDown * (-1.0) + shiftLeft * (-3.0) + shiftRight * (-3.0);

    return S;
}

// Function to create a sparse matrix representing the Laplacian filter (0, -1, 0; -1, 4, -1; 0, -1, 0)
SparseMatrix<double> createLaplacianMatrixOptimized(const int height, const int width) {
    const int size = height * width; // Total number of pixels in the image
    SparseMatrix<double> S(size, size);

    // Identity matrix to represent the base image (center element in the filter)
    SparseMatrix<double> I(size, size);
    I.setIdentity();

    // Define shifts in 4 directions (up, down, left, right)
    SparseMatrix<double> shiftUp(size, size), shiftDown(size, size);
    SparseMatrix<double> shiftLeft(size, size), shiftRight(size, size);

    // Shift by one row (up and down)
    shiftUp.reserve(size);
    shiftDown.reserve(size);
    for (int i = width; i < size; ++i) {
        shiftUp.insert(i, i - width) = 1.0;
        shiftDown.insert(i - width, i) = 1.0;
    }

    // Shift by one column (left and right)
    shiftLeft.reserve(size);
    shiftRight.reserve(size);
    for (int i = 1; i < size; ++i) {
        if (i % width != 0) {
            shiftLeft.insert(i, i - 1) = 1.0;
            shiftRight.insert(i - 1, i) = 1.0;
        }
    }

    // Apply weights from the Laplacian filter
    S = I * 4.0 // center pixel weight (4)
        + shiftUp * -1.0 // up
        + shiftDown * -1.0 // down
        + shiftLeft * -1.0 // left
        + shiftRight * -1.0 // right
            ;

    return S;
}

// Function to save a sparse matrix in MatrixMarket format
void exportMatrixMarketExtended(const SparseMatrix<double> &mat, const VectorXd &vec, const std::string &filename) {
    std::ofstream file(filename);

    // Matrix Market header with additional vector information
    file << "%%MatrixMarket matrix coordinate real general\n";

    // Write dimensions and non-zero count for the matrix and vector
    file << mat.rows() << " " << mat.cols() << " " << mat.nonZeros() << " "
         << "1"
         << " 0\n";

    // Write the matrix in coordinate format (row, col, value)
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
            file << (it.row() + 1) << " " << (it.col() + 1) << " " << it.value() << "\n";
        }
    }

    // Write the vector data (row, value)
    for (int i = 0; i < vec.size(); ++i) {
        file << (i + 1) << " " << vec(i) << "\n";
    }

    file.close();
}

// Function to read a MatrixMarket file, reshape it, and save as an image
bool saveMatrixMarketToImage(const std::string &inputFilePath, const std::string &outputFilePath, const int height,
                             const int width) {
    VectorXd imgVector(height * width);

    // Read the MatrixMarket file
    std::ifstream file(inputFilePath);
    if (!file) {
        std::cerr << "无法打开文件: " << inputFilePath << std::endl;
        return false;
    }

    std::string line;
    getline(file, line); // skip the first line
    getline(file, line); // skip the second line

    int index;
    double real, imag;

    for (int i = 0; i < height * width; ++i) {
        file >> index >> real >> imag;
        imgVector(i) = real;
    }

    file.close();

    if (const auto imgMatrix = imgVector.reshaped<RowMajor>(height, width);
        stbi_write_png(outputFilePath.c_str(), width, height, 1, convertToUnsignedChar(imgMatrix).data(), width) == 0) {
        std::cerr << "保存图像失败: " << outputFilePath << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char *argv[]) {
    // Initialize the logger
    Logger logger("log.txt");
    /**
     * Load the image as an Eigen matrix with size m × n.
     * Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere between 0 (black) and
     * 255 (white). Report the size of the matrix.
     */

    // Load the image as an Eigen matrix with size m × n.
    int width, height, channels;
    auto *input_image_path = "/Users/raopend/Workspace/NLA_ch1/photos/256px-Albert_Einstein_Head.jpg";
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


    /**
     * Write the convolution operation corresponding to the smoothing kernel Hav2
     * as a matrix vector multiplication between a matrix A1 having size mn × mn and the image vector.
     * Report the number of non-zero entries in A1.
     */

    // Define the smoothing kernel Hav2
    // Define the matrix A1
    auto A1 = createAAvg2Matrix(height, width);
    logger.log(INFO, "The number of non-zero entries in A1 is: " + std::to_string(countNonZeroElements(A1)));

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
    logger.log(INFO, "The number of non-zero entries in A2 is: " + std::to_string(countNonZeroElements(A2)));
    // Report if matrix A2 is symmetric
    logger.log(INFO, "Matrix A2 is symmetric: " + std::to_string(A2.isApprox(A2.transpose())));


    /**
     * Apply the previous sharpening filter to the original image by performing the matrix vector multiplication A2v.
     * Export and upload the resulting image.
     */

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

    /**
     * Export the Eigen matrix A2 and vector w in the .mtx format.
     * Using a suitable iterative solver and preconditioner technique available in the LIS library compute the
     * approximate solution to the linear system A2x = w prescribing a tolerance of 10−9.
     * Report here the iteration count and the final residual.
     */
    exportMatrixMarketExtended(A2, w, "A2_w.mtx");

    LIS_MATRIX A;
    LIS_VECTOR x, b;
    LIS_SOLVER solver;
    LIS_INT iter;
    LIS_REAL resid;
    double time;
    auto tol = 1.0e-9;

    // List of solvers to try
    std::vector<std::string> solvers = {"cg", "bicg", "jacobi", "gs", "bicgstab"};

    LIS_DEBUG_FUNC_IN;

    lis_initialize(&argc, &argv);

    // Matrix and vectors setup
    lis_matrix_create(LIS_COMM_WORLD, &A);
    lis_vector_create(LIS_COMM_WORLD, &b);
    lis_vector_create(LIS_COMM_WORLD, &x);
    lis_matrix_set_type(A, LIS_MATRIX_CSR);

    const std::string input_file = "A2_w.mtx";
    lis_input(A, b, x, const_cast<char *>(input_file.c_str()));
    lis_vector_duplicate(A, &x);

    for (const auto &solver_name: solvers) {
        lis_solver_create(&solver);
        // Set solver options dynamically based on the current solver_name
        std::string solver_option = std::format("-i {} -p none -tol {}", solver_name, tol);
        lis_solver_set_option(const_cast<char *>(solver_option.c_str()), solver);

        // Solve the system
        lis_solve(A, b, x, solver);

        // Get and log results
        lis_solver_get_iter(solver, &iter);
        lis_solver_get_time(solver, &time);
        lis_solver_get_residualnorm(solver, &resid);

        // Log details
        logger.log(INFO, "Solver: " + solver_name);
        logger.log(INFO, "Iterations: " + std::to_string(iter));
        logger.log(INFO, "Residual: " + std::format("{}", resid));
        logger.log(INFO, "Elapsed time: " + std::to_string(time) + " seconds");

        // Output results to .mtx and .png
        std::string output_file_mtx = solver_name + "_result.mtx";
        std::string output_file_png = solver_name + "_result.png";
        lis_output_vector(x, LIS_FMT_MM, const_cast<char *>(output_file_mtx.c_str()));

        /**
         * Import the previous approximate solution vector x in Eigen and then convert it into a .png image.
         * Upload the resulting file here
         */
        saveMatrixMarketToImage(output_file_mtx, output_file_png, height, width);

        // Clean up solver
        lis_solver_destroy(solver);
    }

    // Cleanup matrix and vectors
    lis_matrix_destroy(A);
    lis_vector_destroy(b);
    lis_vector_destroy(x);

    lis_finalize();

    /**
     * Write the convolution operation corresponding to the detection kernel Hlab as a matrix vector multiplication
     * by a matrix A3 having size mn × mn. Is matrix A3 symmetric?
     */
    // Define the detection kernel Hlab
    auto A3 = createLaplacianMatrixOptimized(height, width);
    logger.log(INFO, "The number of non-zero entries in A3 is: " + std::to_string(countNonZeroElements(A3)));
    // Report if matrix A3 is symmetric
    logger.log(INFO, "Matrix A3 is symmetric: " + std::to_string(A3.isApprox(A3.transpose())));

    /**
     * Apply the previous edge detection filter to the original image by performing the matrix
     * vector multiplication A3v. Export and upload the resulting image.
     */
    // Apply the edge detection filter to the original image
    auto laplacian_image = A3 * v;
    // Reshape the edge detection image vector to a matrix
    auto laplacian_image_matrix = laplacian_image.reshaped<RowMajor>(height, width);
    // Save the edge detection image using stb_image_write
    const std::string laplacian_image_path = "output_laplacian.png";
    if (stbi_write_png(laplacian_image_path.c_str(), width, height, 1,
                       convertToUnsignedChar(laplacian_image_matrix).data(), width) == 0) {
        logger.log(ERROR, "Could not save edge detection image");
        return 1;
    }
    logger.log(INFO, "Edge detection image saved to: " + laplacian_image_path);

    /**
     * Using a suitable iterative solver available in the Eigen library compute the approximate
     * solutionofthelinearsystem(I+A3)y= w,where I denotes the identity matrix,
     * prescribing a tolerance of 10−10. Report here the iteration count and the final residual.
     */
    // Define the identity matrix
    SparseMatrix<double> I(height * width, height * width);
    I.setIdentity();
    // Define the matrix A3
    auto A3_I = I + A3;
    // List of solver names and corresponding solver objects
    for (std::vector<std::string> solver_names = {"ConjugateGradient", "BiCGSTAB", "SparseLU"};
         const auto &solver_name: solver_names) {
        VectorXd y;

        if (solver_name == "ConjugateGradient") {
            // Conjugate Gradient solver
            ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
            cg.setTolerance(1.0e-10);
            cg.compute(A3_I);

            if (cg.info() != Eigen::Success) {
                logger.log(ERROR, solver_name + " solver failed to converge");
                continue;
            }
            y = cg.solve(w);

            // Logging iteration count and residual
            logger.log(INFO, "Solver: " + solver_name);
            logger.log(INFO, "Iterations: " + std::to_string(cg.iterations()));
            logger.log(INFO, "Final residual: " + std::format("{}", cg.error()));

        } else if (solver_name == "BiCGSTAB") {
            // BiCGSTAB solver
            BiCGSTAB<SparseMatrix<double>> bicgstab;
            bicgstab.setTolerance(1.0e-10);
            bicgstab.compute(A3_I);

            if (bicgstab.info() != Success) {
                logger.log(ERROR, solver_name + " solver failed to converge");
                continue;
            }
            y = bicgstab.solve(w);

            // Logging iteration count and residual
            logger.log(INFO, "Solver: " + solver_name);
            logger.log(INFO, "Iterations: " + std::to_string(bicgstab.iterations()));
            logger.log(INFO, "Final residual: " + std::format("{}", bicgstab.error()));

        } else if (solver_name == "SparseLU") {
            // SparseLU solver
            SparseLU<SparseMatrix<double>> sparse_lu;
            sparse_lu.compute(A3_I);

            if (sparse_lu.info() != Success) {
                logger.log(ERROR, solver_name + " solver failed to converge");
                continue;
            }
            y = sparse_lu.solve(w);

            // the final residual for the linear system (I + A3)y = w
            // Logging iteration count and residual (SparseLU doesn't iterate)
            logger.log(INFO, "Solver: " + solver_name);
            logger.log(INFO, "Iterations: 1 (direct solver)");
            logger.log(INFO, "Final residual: n/a (direct solver)");
        }

        // Reshape the solution vector to a matrix
        auto y_matrix = y.reshaped<RowMajor>(height, width);

        /**
         * Convert the image stored in the vector y into a .png image and upload it.
         */
        // Save the resulting image as a PNG file using stb_image_write
        const std::string y_image_path = solver_name + "_y.png";
        if (stbi_write_png(y_image_path.c_str(), width, height, 1, convertToUnsignedChar(y_matrix).data(), width) ==
            0) {
            logger.log(ERROR, "Could not save image for solver: " + solver_name);
            continue;
        }
        logger.log(INFO,
                   std::string("Image saved for solver: ").append(solver_name).append(" to ").append(y_image_path));
    }

    /**
     * Comment the obtained results.
     */

    /*
    Compared to direct methods (LU factorization), the Jacobi method, and the Gauss-Seidel method, the improved BiCG and
    BiCGSTAB algorithms for the Conjugate Gradient Method have shown good performance in solving this large sparse
    matrix. Both methods reduced the residual to very small values, indicating good convergence. Overall, the BiCGSTAB
    algorithm converged the fastest compared to BiCG, though its accuracy was slightly lower. All iterative methods were
    able to converge, whereas the direct method was unsuitable for this system (no solution).
     */
    return 0;
}
