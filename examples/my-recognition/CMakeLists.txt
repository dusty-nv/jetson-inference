
cmake_minimum_required(VERSION 2.8)
project(my-recognition)

find_package(jetson-utils)
find_package(jetson-inference)

find_package(CUDA)
find_package(Qt4)

include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

cuda_add_executable(my-recognition my-recognition.cpp)
target_link_libraries(my-recognition jetson-inference)
