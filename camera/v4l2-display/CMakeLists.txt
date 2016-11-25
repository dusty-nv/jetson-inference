
file(GLOB v4l2DisplaySources *.cpp)
file(GLOB v4l2DisplayIncludes *.h )

# includes cmake/FindSDL2.cmake
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(SDL2 REQUIRED)
find_package(SDL2TTF REQUIRED)
include_directories(${SDL2_INCLUDE_DIR})

add_executable(v4l2-display ${v4l2DisplaySources})
target_link_libraries(v4l2-display jetson-inference)
target_link_libraries(v4l2-display ${SDL2_LIBRARY} ${SDL2TTF_LIBRARY})
