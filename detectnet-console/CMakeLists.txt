
file(GLOB detectnetConsoleSources *.cpp)
file(GLOB detectnetConsoleIncludes *.h )

cuda_add_executable(detectnet-console ${detectnetConsoleSources})
target_link_libraries(detectnet-console nvcaffe_parser nvinfer jetson-inference)
