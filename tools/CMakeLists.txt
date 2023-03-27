
# build subdirectories
add_subdirectory(camera-capture)
add_subdirectory(depth-viewer)

#add_subdirectory(trt-bench)
#add_subdirectory(trt-console)

# copy tools
file(COPY "test-models.py" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY "tao-model-downloader.sh" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY "l4t_version.sh" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

install(PROGRAMS "test-models.py" DESTINATION bin)
install(PROGRAMS "tao-model-downloader.sh" DESTINATION bin)
install(PROGRAMS "l4t_version.sh" DESTINATION bin)

#file(GLOB scripts *.sh)
#foreach(script ${scripts})
#	message("-- Copying ${script}")
#	configure_file(${script} ${PROJECT_BINARY_DIR} COPYONLY)
#endforeach()

