
file(GLOB imagenetCameraSources *.cpp)
file(GLOB imagenetCameraIncludes *.h )


# includes cmake/FindSDL2.cmake
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(SDL2 REQUIRED)
find_package(SDL2TTF REQUIRED)
include_directories(${SDL2_INCLUDE_DIR})


cuda_add_executable(imagenet-camera ${imagenetCameraSources})
target_link_libraries(imagenet-camera nvcaffe_parser nvinfer jetson-inference)
target_link_libraries(imagenet-camera ${SDL2_LIBRARY} ${SDL2TTF_LIBRARY})

