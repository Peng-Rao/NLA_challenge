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
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;

// Utility function to convert and clip values to the range [0, 255]
Matrix<unsigned char, Dynamic, Dynamic, RowMajor> convertToUnsignedChar(const MatrixXd &matrix) {
    return matrix.unaryExpr([](const double val) -> unsigned char {
        return static_cast<unsigned char>(std::min(255.0, std::max(0.0, val))); // Clip values between 0 and 255
    });
}

int main(int argc, char *argv[]) {
    int height = 768; // 替换为图像的实际高度
    int width = 576; // 替换为图像的实际宽度

    VectorXd imgVector(height * width); // 图像向量，长度应该是height * width

    // 打开MatrixMarket文件
    ifstream file("/Users/raopend/Workspace/NLA_ch1/result.mtx");
    if (!file) {
        cerr << "无法打开文件" << endl;
        return -1;
    }

    string line;
    getline(file, line); // 跳过第一行（%%MatrixMarket头信息）
    getline(file, line); // 跳过第二行（向量大小信息）

    int index;
    double real, imag;

    for (int i = 0; i < height * width; ++i) {
        file >> index >> real >> imag; // 读取行索引、实部、虚部
        imgVector(i) = real; // 只存储实部
    }

    file.close();

    // 将向量重新映射为矩阵
    auto imgMatrix = imgVector.reshaped<RowMajor>(height, width);

    // save to image
    const char *filename = "result.png";
    if (stbi_write_png(filename, width, height, 1, convertToUnsignedChar(imgMatrix).data(), width) == 0) {
        return 1;
    }
    return 0;
}
