
file(GLOB detectnetVideoSources *.cpp)
file(GLOB detectnetVideoIncludes *.h )

cuda_add_executable(detectnet-video ${detectnetVideoSources})

target_link_libraries(detectnet-video jetson-inference)

install(TARGETS detectnet-video DESTINATION bin)
