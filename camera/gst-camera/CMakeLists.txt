
file(GLOB gstCameraSources *.cpp)
file(GLOB gstCameraIncludes *.h )

# includes cmake/FindSDL2.cmake
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(SDL2 REQUIRED)
find_package(SDL2TTF REQUIRED)
include_directories(${SDL2_INCLUDE_DIR})

add_executable(gst-camera ${gstCameraSources})
target_link_libraries(gst-camera jetson-inference)
target_link_libraries(gst-camera ${SDL2_LIBRARY} ${SDL2TTF_LIBRARY})
