
file(GLOB posenetSources *.cpp)
file(GLOB posenetIncludes *.h )

cuda_add_executable(posenet ${posenetSources})
target_link_libraries(posenet jetson-inference)
install(TARGETS posenet DESTINATION bin)

