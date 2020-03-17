
file(GLOB flownetCameraSources *.cpp)
file(GLOB flownetCameraIncludes *.h )

cuda_add_executable(flownet-camera ${flownetCameraSources})

target_link_libraries(flownet-camera jetson-inference)

install(TARGETS flownet-camera DESTINATION bin)
