
file(GLOB depthnetCameraSources *.cpp)
file(GLOB depthnetCameraIncludes *.h )

cuda_add_executable(depthnet-camera ${depthnetCameraSources})

target_link_libraries(depthnet-camera jetson-inference)

install(TARGETS depthnet-camera DESTINATION bin)
