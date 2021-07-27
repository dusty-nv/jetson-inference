
file(GLOB depthnetSources *.cpp)
file(GLOB depthnetIncludes *.h )

cuda_add_executable(depthnet ${depthnetSources})

target_link_libraries(depthnet jetson-inference)

install(TARGETS depthnet DESTINATION bin)
