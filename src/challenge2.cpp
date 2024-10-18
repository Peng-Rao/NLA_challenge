#include "utils.h"


#include <plog/Initializers/RollingFileInitializer.h>


int main(int argc, char *argv[]) {

    // Initialize the logger
    plog::init(plog::debug, "./ch2_result/log.txt");

    // Load the iamge as matrix A with size m times n

    return 0;
}
