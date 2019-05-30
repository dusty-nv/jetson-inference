
file(GLOB trtBenchSources *.cpp)
file(GLOB trtBenchIncludes *.h )

cuda_add_executable(trt-bench ${trtBenchSources})
target_link_libraries(trt-bench nvcaffe_parser nvinfer jetson-inference)
