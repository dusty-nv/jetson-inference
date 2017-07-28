
file(GLOB detectnetVideoSources *.cpp)
file(GLOB detectnetVideoIncludes *.h )

cuda_add_executable(detectnet-video ${detectnetVideoSources})
target_link_libraries(detectnet-video nvcaffe_parser nvinfer jetson-inference ${OpenCV_LIBS})
