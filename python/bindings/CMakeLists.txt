# --- Clear legacy Python cache variables (avoid stale cache conflicts) ---
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

message("-- detecting Python ${PYTHON_BINDING_VERSION}...")

# --- Modern path (CMake >= 3.18): FindPython3 with Interpreter/Development/NumPy ---
find_package(Python3 ${PYTHON_BINDING_VERSION} QUIET COMPONENTS Interpreter Development NumPy)

if(Python3_Interpreter_FOUND AND Python3_Development_FOUND)
  # Normalize variables so the rest of the script can remain similar
  set(PYTHON_VERSION_MAJOR ${Python3_VERSION_MAJOR})
  set(PYTHON_VERSION_MINOR ${Python3_VERSION_MINOR})
  set(PYTHONLIBS_VERSION_STRING ${Python3_VERSION})
  set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
  set(PYTHON_LIBRARIES Python3::Python)

  message("-- found Python version:  ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} (${PYTHONLIBS_VERSION_STRING})")
  message("-- found Python include:  ${PYTHON_INCLUDE_DIRS}")
  message("-- using Python library target:  Python3::Python")

  # NumPy (modern)
  set(NUMPY_FOUND OFF)
  if(Python3_NumPy_FOUND)
    set(NUMPY_FOUND ON)
    set(NUMPY_VERSION ${Python3_NumPy_VERSION})
    set(NUMPY_INCLUDE_DIR ${Python3_NumPy_INCLUDE_DIRS})
    message("-- found NumPy version:  ${NUMPY_VERSION}")
    message("-- found NumPy include:  ${NUMPY_INCLUDE_DIR}")
    add_definitions(-DHAS_NUMPY)
  else()
    message("-- NumPy not found (modern path)")
  endif()

else()
  # --- Legacy fallback (older CMake / systems) ---
  find_package(PythonInterp ${PYTHON_BINDING_VERSION} QUIET)
  find_package(PythonLibs  ${PYTHON_BINDING_VERSION} QUIET)

  if(NOT PYTHONLIBS_FOUND)
    message("-- Python ${PYTHON_BINDING_VERSION} wasn't found")
    return()
  endif()

  message("-- found Python version:  ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} (${PYTHONLIBS_VERSION_STRING})")
  message("-- found Python include:  ${PYTHON_INCLUDE_DIRS}")
  message("-- found Python library:  ${PYTHON_LIBRARIES}")

  # Extend module path to pick up your custom FindNumPy.cmake if present
  set(CMAKE_MODULE_PATH
      ${CMAKE_MODULE_PATH}
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../utils/python/bindings"
  )
  message("-- CMake module path:  ${CMAKE_MODULE_PATH}")

  find_package(NumPy QUIET)

  if(NUMPY_FOUND)
    message("-- found NumPy version:  ${NUMPY_VERSION}")
    message("-- found NumPy include:  ${NUMPY_INCLUDE_DIR}")
    add_definitions(-DHAS_NUMPY)
  else()
    message("-- NumPy not found (legacy path)")
  endif()
endif()

# --- Sources & target ---
file(GLOB pythonExtSources *.cpp)

set(_PYMOD jetson-inference-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
cuda_add_library(${_PYMOD} SHARED ${pythonExtSources})

# Per-target includes and links (avoid global include/link directories)
target_include_directories(${_PYMOD} PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(${_PYMOD}
  PRIVATE
    jetson-inference
    ${PYTHON_LIBRARIES}
    jetson-utils-python-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
)

# Add NumPy include directories if available
if(NUMPY_FOUND)
  target_include_directories(${_PYMOD} PRIVATE ${NUMPY_INCLUDE_DIR})

  # Only for NumPy < 2.0: attempt to link npymath if it exists
  if(DEFINED NUMPY_VERSION AND NUMPY_VERSION VERSION_LESS "2.0")
    find_library(NPYMATH_LIB NAMES npymath)
    if(NPYMATH_LIB)
      target_link_libraries(${_PYMOD} PRIVATE ${NPYMATH_LIB})
      message(STATUS "NumPy ${NUMPY_VERSION} (< 2.0): linking npymath -> ${NPYMATH_LIB}")
    else()
      message(WARNING "NumPy ${NUMPY_VERSION} (< 2.0) but 'npymath' not found: continuing without it")
    endif()
  else()
    message(STATUS "NumPy ${NUMPY_VERSION} (>= 2.0): skipping npymath linkage")
  endif()
endif()

set_target_properties(${_PYMOD}
  PROPERTIES
    PREFIX ""
    OUTPUT_NAME "jetson_inference_python"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}"
)

# --- Install paths (x86_64 vs Jetson) ---
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(PYTHON_BINDING_INSTALL_DIR /opt/conda/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)
else()
  set(PYTHON_BINDING_INSTALL_DIR /usr/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages)
endif()

install(TARGETS ${_PYMOD} DESTINATION ${PYTHON_BINDING_INSTALL_DIR})

install(DIRECTORY ../python/Jetson           DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
install(DIRECTORY ../python/jetson           DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
install(DIRECTORY ../python/jetson_inference DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
