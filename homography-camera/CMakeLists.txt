
file(GLOB homographyCameraSources *.cpp)
file(GLOB homographyCameraIncludes *.h )

cuda_add_executable(homography-camera ${homographyCameraSources})

target_link_libraries(homography-camera jetson-inference)

install(TARGETS homography-camera DESTINATION bin)
