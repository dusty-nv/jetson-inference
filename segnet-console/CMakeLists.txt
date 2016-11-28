
file(GLOB segnetConsoleSources *.cpp)
file(GLOB segnetConsoleIncludes *.h )

cuda_add_executable(segnet-console ${segnetConsoleSources})
target_link_libraries(segnet-console nvcaffe_parser nvinfer jetson-inference)
