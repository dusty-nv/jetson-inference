
file(GLOB gstCameraSources *.cpp)
file(GLOB gstCameraIncludes *.h )

add_executable(gst-camera ${gstCameraSources})
target_link_libraries(gst-camera jetson-inference)
