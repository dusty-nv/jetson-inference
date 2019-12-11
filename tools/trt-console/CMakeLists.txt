
file(GLOB trtConsoleSources *.cpp)
file(GLOB trtConsoleIncludes *.h )

cuda_add_executable(trt-console ${trtConsoleSources})
target_link_libraries(trt-console nvcaffe_parser nvinfer jetson-inference)
