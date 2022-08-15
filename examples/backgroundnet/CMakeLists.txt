
file(GLOB backgroundnetSources *.cpp)
file(GLOB backgroundnetIncludes *.h )

cuda_add_executable(backgroundnet ${backgroundnetSources})
target_link_libraries(backgroundnet jetson-inference)
install(TARGETS backgroundnet DESTINATION bin)
