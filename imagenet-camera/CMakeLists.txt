
file(GLOB imagenetCameraSources *.cpp)
file(GLOB imagenetCameraIncludes *.h )

cuda_add_executable(imagenet-camera ${imagenetCameraSources})

target_link_libraries(imagenet-camera jetson-inference)

install(TARGETS imagenet-camera DESTINATION bin)
