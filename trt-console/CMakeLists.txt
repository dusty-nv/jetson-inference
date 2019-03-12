
# OpenCV used for findHomography() and decomposeHomography()
# OpenCV version >= 3.0.0 required for decomposeHomography()
find_package(OpenCV 3.0.0 COMPONENTS core calib3d REQUIRED)

file(GLOB trtConsoleSources *.cpp)
file(GLOB trtConsoleIncludes *.h )

cuda_add_executable(trt-console ${trtConsoleSources})
target_link_libraries(trt-console nvcaffe_parser nvinfer jetson-inference opencv_core opencv_calib3d)
