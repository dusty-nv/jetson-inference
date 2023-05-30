
file(GLOB segnetSources *.cpp)
file(GLOB segnetIncludes *.h )

cuda_add_executable(segnet ${segnetSources})
target_link_libraries(segnet jetson-inference)
install(TARGETS segnet DESTINATION bin)
