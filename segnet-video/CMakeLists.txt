
file(GLOB segnetVideoSources *.cpp)
file(GLOB segnetVideoIncludes *.h )

cuda_add_executable(segnet-video ${segnetVideoSources})
target_link_libraries(segnet-video nvcaffe_parser nvinfer jetson-inference ${OpenCV_LIBS})
