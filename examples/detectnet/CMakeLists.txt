
file(GLOB detectnetSources *.cpp)
file(GLOB detectnetIncludes *.h )

cuda_add_executable(detectnet ${detectnetSources})
target_link_libraries(detectnet jetson-inference)
install(TARGETS detectnet DESTINATION bin)

# for legacy users of detectnet-camera (will be deprecated)
cuda_add_executable(detectnet-camera ${detectnetSources})
target_link_libraries(detectnet-camera jetson-inference)
install(TARGETS detectnet-camera DESTINATION bin)

# for legacy users of detectnet-console (will be deprecated)
cuda_add_executable(detectnet-console ${detectnetSources})
set_target_properties(detectnet-console PROPERTIES COMPILE_FLAGS "-DHEADLESS" )
target_link_libraries(detectnet-console jetson-inference)
install(TARGETS detectnet-console DESTINATION bin)
