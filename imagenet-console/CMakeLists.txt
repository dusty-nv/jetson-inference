
file(GLOB imagenetConsoleSources *.cpp)
file(GLOB imagenetConsoleIncludes *.h )

cuda_add_executable(imagenet-console ${imagenetConsoleSources})
target_link_libraries(imagenet-console nvcaffe_parser nvinfer jetson-inference)
