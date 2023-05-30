
file(GLOB detectnetSources *.cpp)
file(GLOB detectnetIncludes *.h )

cuda_add_executable(detectnet ${detectnetSources})
target_link_libraries(detectnet jetson-inference)
install(TARGETS detectnet DESTINATION bin)
