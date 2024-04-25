# require CMake 2.8 or greater
cmake_minimum_required(VERSION 2.8)

# declare my-recognition project
project(camera-test)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations")
endif()


# import jetson-inference and jetson-utils packages.
# note that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(jetson-utils)
find_package(jetson-inference)

# find VPI package (optional)
#find_package(VPI 2.0)

# CUDA is required
find_package(CUDA)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# add directory for libnvbuf-utils to program
link_directories(/usr/lib/aarch64-linux-gnu/tegra)
include_directories(util yolo fastscnn)

cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)

# compile the my-recognition program
cuda_add_executable(camera-test camera-test.cpp util/logger.cpp ipm.cu argmax.cu yolo/yolov3.cpp util/utils.cpp yololayer.cu yolo/kernel.cu fastscnn/fastscnn.cpp)

# link my-recognition to jetson-inference library
target_link_libraries(camera-test jetson-inference ${OpenCV_LIBS} yololayer nvinfer cudart)


