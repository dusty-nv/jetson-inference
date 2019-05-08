# clear CMakeCache of Python version
unset(PYTHONINTERP_FOUND CACHE)
unset(PYTHON_EXECUTABLE CACHE)
unset(PYTHON_VERSION_STRING CACHE)
unset(PYTHON_VERSION_MAJOR CACHE)
unset(PYTHON_VERSION_MINOR CACHE)
unset(PYTHON_VERSION_PATCH CACHE)

unset(PYTHON_INCLUDE_PATH CACHE)
unset(PYTHON_INCLUDE_DIRS CACHE)
unset(PYTHON_INCLUDE_DIR CACHE)
unset(PYTHON_LIBRARY CACHE)
unset(PYTHON_LIBRARIES CACHE)
unset(PYTHON_DEBUG_LIBRARIES CACHE)
unset(PYTHON_MODULE_PREFIX CACHE)
unset(PYTHON_MODULE_EXTENSION CACHE)

unset(PYTHONLIBS_FOUND CACHE)
unset(PYTHONLIBS_VERSION_STRING CACHE)

# locate requested python version
message("-- detecting Python ${PYTHON_BINDING_VERSION}...")

find_package(PythonInterp ${PYTHON_BINDING_VERSION} QUIET)
find_package(PythonLibs ${PYTHON_BINDING_VERSION} QUIET)

if(NOT ${PYTHONLIBS_FOUND})
	message("-- Python ${PYTHON_BINDING_VERSION} wasn't found")
	return()
endif()

message("-- found Python version:  ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} (${PYTHONLIBS_VERSION_STRING})")
message("-- found Python include:  ${PYTHON_INCLUDE_DIRS}")
message("-- found Python library:  ${PYTHON_LIBRARIES}") 

include_directories(${PYTHON_INCLUDE_DIRS})

file(GLOB pythonExtSources *.cpp)

cuda_add_library(jetson-inference-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} SHARED ${pythonExtSources})

target_link_libraries(jetson-inference-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} jetson-inference ${PYTHON_LIBRARIES})

set_target_properties(jetson-inference-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
				  PROPERTIES
					PREFIX ""
					OUTPUT_NAME "jetson_inference_python"
					LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

