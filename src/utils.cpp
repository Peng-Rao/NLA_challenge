#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <fstream>

// Function to convert RGB to grayscale
Eigen::MatrixXd convertToGrayscale(const Eigen::MatrixXd &red, const Eigen::MatrixXd &green,
                                   const Eigen::MatrixXd &blue) {
    return 0.299 * red + 0.587 * green + 0.114 * blue;
}

// Function to load an image
bool loadImage(const char *imagePath, Eigen::MatrixXd &image_matrix, int &width, int &height, int &channels) {
    unsigned char *image_data = stbi_load(imagePath, &width, &height, &channels, 1);

    if (!image_data) {
        return false;
    }

    image_matrix.resize(height, width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image_matrix(i, j) = static_cast<double>(image_data[i * width + j]);
        }
    }

    stbi_image_free(image_data);
    return true;
}


// FUnction to cast Eigen::MatrixXd to unsigned char
Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
convertToUnsignedChar(const Eigen::MatrixXd &matrix) {
    return matrix.unaryExpr([](const double val) -> unsigned char {
        return static_cast<unsigned char>(std::min(255.0, std::max(0.0, val))); // Clip values between 0 and 255
    });
}

// Function to export a matrix to Matrix Market format
void exportMatrixMarketExtended(const Eigen::SparseMatrix<double> &mat, const Eigen::VectorXd &vec,
                                const std::string &filename) {
    std::ofstream file(filename);

    // Matrix Market header with additional vector information
    file << "%%MatrixMarket matrix coordinate real general\n";

    // Write dimensions and non-zero count for the matrix and vector
    file << mat.rows() << " " << mat.cols() << " " << mat.nonZeros() << " "
         << "1"
         << " 0\n";

    // Write the matrix in coordinate format (row, col, value)
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
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
    Eigen::VectorXd imgVector(height * width);

    // Read the MatrixMarket file
    std::ifstream file(inputFilePath);
    if (!file) {
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

    if (const auto imgMatrix = imgVector.reshaped<Eigen::RowMajor>(height, width);
        stbi_write_png(outputFilePath.c_str(), width, height, 1, convertToUnsignedChar(imgMatrix).data(), width) == 0) {
        return false;
    }

    return true;
}

// Function to save an image
bool saveImage(const std::string &outputFilePath, const Eigen::MatrixXd &data, const int height, const int width) {
    if (stbi_write_png(outputFilePath.c_str(), width, height, 1, convertToUnsignedChar(data).data(), width) == 0) {
        return false;
    }
    return true;
}
