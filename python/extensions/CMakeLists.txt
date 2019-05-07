
# locate Python for Python extension
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

message("-- found Python version:  ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} (${PYTHONLIBS_VERSION_STRING})")
message("-- found Python include:  ${PYTHON_INCLUDE_DIRS}")
message("-- found Python library:  ${PYTHON_LIBRARIES}") 

include_directories(${PYTHON_INCLUDE_DIRS})

file(GLOB pythonExtSources *.cpp)

cuda_add_library(jetson-inference-python SHARED ${pythonExtSources})

target_link_libraries(jetson-inference-python jetson-inference python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})

set_target_properties(jetson-inference-python
				  PROPERTIES
					PREFIX ""
					OUTPUT_NAME "jetson_inference_python")

