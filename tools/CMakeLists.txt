
# build subdirectories
add_subdirectory(camera-capture)
add_subdirectory(depth-viewer)

#add_subdirectory(trt-bench)
#add_subdirectory(trt-console)

# copy tools
file(COPY "segnet-batch.sh" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

#file(GLOB scripts *.sh)
#foreach(script ${scripts})
#	message("-- Copying ${script}")
#	configure_file(${script} ${PROJECT_BINARY_DIR} COPYONLY)
#endforeach()

