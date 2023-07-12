
# require CMake 2.8 or greater
cmake_minimum_required(VERSION 2.8)

# declare my-recognition project
project(my-recognition)

# import jetson-inference and jetson-utils packages.
# note that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(jetson-utils)
find_package(jetson-inference)

# find VPI package (optional)
find_package(VPI 2.0)

# CUDA is required
find_package(CUDA)

# add directory for libnvbuf-utils to program
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

# compile the my-recognition program
cuda_add_executable(my-recognition my-recognition.cpp)

# link my-recognition to jetson-inference library
target_link_libraries(my-recognition jetson-inference)
