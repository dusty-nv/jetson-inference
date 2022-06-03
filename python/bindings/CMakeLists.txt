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

# attempt to find numpy
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../utils/python/bindings")
message("-- CMake module path:  ${CMAKE_MODULE_PATH}")
find_package(NumPy QUIET)

if(${NUMPY_FOUND})
	message("-- found NumPy version:  ${NUMPY_VERSION}")
	message("-- found NumPy include:  ${NUMPY_INCLUDE_DIR}")

	add_definitions(-DHAS_NUMPY)
	include_directories(${NUMPY_INCLUDE_DIR})
	link_directories(${NUMPY_INCLUDE_DIR}/../lib)
else()
	message("-- NumPy not found")
endif()

# build the bindings
file(GLOB pythonExtSources *.cpp)

cuda_add_library(jetson-inference-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} SHARED ${pythonExtSources})

target_link_libraries(jetson-inference-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} jetson-inference ${PYTHON_LIBRARIES} jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})

if(${NUMPY_FOUND})
	target_link_libraries(jetson-inference-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} npymath)
endif()

set_target_properties(jetson-inference-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
				  PROPERTIES
					PREFIX ""
					OUTPUT_NAME "jetson_inference_python"
					LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")

# on x86, install under /opt/conda/lib/pythonX.X/site-packages
# otherwise, install under /usr/lib/pythonX.X/dist-packages
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
	set(PYTHON_BINDING_INSTALL_DIR /opt/conda/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)
else()
	set(PYTHON_BINDING_INSTALL_DIR /usr/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages)
endif()

install(TARGETS jetson-inference-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} DESTINATION ${PYTHON_BINDING_INSTALL_DIR})

install(DIRECTORY ../python/Jetson DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
install(DIRECTORY ../python/jetson DESTINATION ${PYTHON_BINDING_INSTALL_DIR})

