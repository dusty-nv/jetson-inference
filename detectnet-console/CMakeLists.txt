
file(GLOB detectnetConsoleSources *.cpp)
file(GLOB detectnetConsoleIncludes *.h )

# includes cmake/FindSDL2.cmake
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(SDL2 REQUIRED)
find_package(SDL2TTF REQUIRED)
include_directories(${SDL2_INCLUDE_DIR})

cuda_add_executable(detectnet-console ${detectnetConsoleSources})
target_link_libraries(detectnet-console nvcaffe_parser nvinfer jetson-inference)
target_link_libraries(detectnet-console ${SDL2_LIBRARY} ${SDL2TTF_LIBRARY})
