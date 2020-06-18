
file(GLOB imagenetSources *.cpp)
file(GLOB imagenetIncludes *.h )

cuda_add_executable(imagenet ${imagenetSources})
target_link_libraries(imagenet jetson-inference)
install(TARGETS imagenet DESTINATION bin)

# for legacy users of imagenet-camera (will be deprecated)
cuda_add_executable(imagenet-camera ${imagenetSources})
target_link_libraries(imagenet-camera jetson-inference)
install(TARGETS imagenet-camera DESTINATION bin)

# for legacy users of imagenet-console (will be deprecated)
cuda_add_executable(imagenet-console ${imagenetSources})
set_target_properties(imagenet-console PROPERTIES COMPILE_FLAGS "-DHEADLESS" )
target_link_libraries(imagenet-console jetson-inference)
install(TARGETS imagenet-console DESTINATION bin)
