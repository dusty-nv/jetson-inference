
file(GLOB actionnetSources *.cpp)
file(GLOB actionnetIncludes *.h )

cuda_add_executable(actionnet ${actionnetSources})
target_link_libraries(actionnet jetson-inference)
install(TARGETS actionnet DESTINATION bin)
