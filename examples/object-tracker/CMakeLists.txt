
file(GLOB objectTrackerSources *.cpp)
file(GLOB objectTrackerIncludes *.h )

cuda_add_executable(object-tracker ${objectTrackerSources})
target_link_libraries(object-tracker jetson-inference)
install(TARGETS object-tracker DESTINATION bin)
