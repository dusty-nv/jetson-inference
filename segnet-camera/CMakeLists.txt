
file(GLOB segnetCameraSources *.cpp)
file(GLOB segnetCameraIncludes *.h )

cuda_add_executable(segnet-camera ${segnetCameraSources})
target_link_libraries(segnet-camera nvcaffe_parser nvinfer jetson-inference)
