
file(GLOB v4l2DisplaySources *.cpp)
file(GLOB v4l2DisplayIncludes *.h )

add_executable(v4l2-display ${v4l2DisplaySources})
target_link_libraries(v4l2-display jetson-inference)
