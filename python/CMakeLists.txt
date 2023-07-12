
#
# This list contains the versions of Python that will be detected 
# and the bindings built against (if that version is installed).
#
# If you want to support another version of Python, add it here.
#
if(LSB_RELEASE_CODENAME MATCHES "focal")
	set(PYTHON_BINDING_VERSIONS 3.8)
else()
	set(PYTHON_BINDING_VERSIONS 2.7 3.6 3.7)
endif()

message("-- trying to build Python bindings for Python versions:  ${PYTHON_BINDING_VERSIONS}")

foreach(PYTHON_BINDING_VERSION ${PYTHON_BINDING_VERSIONS})
	add_subdirectory(bindings bindings_python_${PYTHON_BINDING_VERSION})
endforeach()


#
# install Python examples
#
file(GLOB pyInferenceExamples examples/*.py)

foreach(pyExample ${pyInferenceExamples})
	message("-- Copying ${pyExample}")
	file(COPY ${pyExample} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	install(FILES "${pyExample}" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endforeach()


#
# replicate legacy Python examples (these will be deprecated)
#
function(copy_install file_in file_out)
    message("-- Copying ${file_in} -> ${file_out}")
    configure_file(${file_in} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${file_out} COPYONLY)
    install(FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${file_out}" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endfunction()

