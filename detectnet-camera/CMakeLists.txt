
file(GLOB detectnetCameraSources *.cpp)
file(GLOB detectnetCameraIncludes *.h )

cuda_add_executable(detectnet-camera ${detectnetCameraSources})

target_link_libraries(detectnet-camera jetson-inference)

install(TARGETS detectnet-camera DESTINATION bin)
