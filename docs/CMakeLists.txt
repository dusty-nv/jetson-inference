# 
# doxygen
#
find_package(Doxygen)

if( DOXYGEN_FOUND )

	configure_file( Doxyfile.in 
				 ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY )

	add_custom_target(docs ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
					WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
					COMMENT "Generating API documentation" VERBATIM )

	add_custom_command(TARGET "docs" POST_BUILD 
				    COMMAND pydoc3 -w jetson
				    COMMAND pydoc3 -w jetson.inference
				    COMMAND pydoc3 -w jetson.utils
				    COMMAND pydoc3 -w jetson_inference
				    COMMAND pydoc3 -w jetson_utils
				    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/jetson*.html ${PROJECT_OUTPUT_DIR}/docs/html/python
				    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_OUTPUT_DIR}/docs/html/group__deepVision.html ${PROJECT_OUTPUT_DIR}/docs/html/index.html 
				    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_OUTPUT_DIR}/docs/html ${CMAKE_CURRENT_SOURCE_DIR}/html )

	#COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/navtree.js ${PROJECT_OUTPUT_DIR}/docs/html
	
	file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/docs)
	file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/docs/html)
	file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/docs/html/python)

	#file(GLOB docIndex *.html)
	#file(INSTALL ${docIndex} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/doc)

	message("-- Doxygen detected, to build API reference, run `make docs`")

endif( DOXYGEN_FOUND )
