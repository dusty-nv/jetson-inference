
file(GLOB stereonetCameraSources *.cpp)
file(GLOB stereonetCameraIncludes *.h )

cuda_add_executable(stereonet-camera ${stereonetCameraSources})

target_link_libraries(stereonet-camera jetson-inference)

install(TARGETS stereonet-camera DESTINATION bin)
