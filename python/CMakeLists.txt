
#
# This list contains the versions of Python that will be detected 
# and the bindings built against (if that version is installed).
#
# If you want to support another version of Python, add it here.
#
set(PYTHON_BINDING_VERSIONS 2.7 3.6 3.7)

message("-- trying to build Python bindings for Python versions:  ${PYTHON_BINDING_VERSIONS}")

foreach(PYTHON_BINDING_VERSION ${PYTHON_BINDING_VERSIONS})
	add_subdirectory(bindings bindings_python_${PYTHON_BINDING_VERSION})
endforeach()
