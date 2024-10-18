cmake_minimum_required(VERSION 3.28)
project(NLA_challenge)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# include lis library
include_directories(extern/lis/include)
link_directories(extern/lis/lib)
file(
        GLOB_RECURSE mains RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/challenge*.cpp"
)

find_package(Stb REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(plog CONFIG REQUIRED)


foreach (mainfile IN LISTS mains)
    # Get file name without directory
    get_filename_component(mainname ${mainfile} NAME_WE)
    add_executable(${mainname} ${mainfile}
            src/utils.cpp)
    target_include_directories(${mainname} PRIVATE include)
    target_include_directories(${mainname} PRIVATE ${Stb_INCLUDE_DIR})
    target_link_libraries(${mainname} PRIVATE Eigen3::Eigen)
    target_link_libraries(${mainname} PRIVATE plog::plog)
    # Link lis library
    target_link_libraries(${mainname} PRIVATE lis)
endforeach ()
