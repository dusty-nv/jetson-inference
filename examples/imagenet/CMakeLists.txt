
file(GLOB imagenetSources *.cpp)
file(GLOB imagenetIncludes *.h )

cuda_add_executable(imagenet ${imagenetSources})
target_link_libraries(imagenet jetson-inference)
install(TARGETS imagenet DESTINATION bin)
