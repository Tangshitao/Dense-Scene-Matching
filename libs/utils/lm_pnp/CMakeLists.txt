cmake_minimum_required(VERSION 3.10)
project(lm_pnp)

set(CUDA_SDK_ROOT_DIR /usr/local/cuda)

# Compiler Options -----------------------------------------------------------------------------------------------------
SET(CMAKE_CONFIGURATION_TYPES RELEASE; DEBUG)
SET(CMAKE_BUILD_TYPE RELEASE CACHE STRING "Build Type: Release" FORCE)

if (CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
            OUTPUT_VARIABLE GCC_VERSION)
    string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
    list(GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR)
endif()
if(CMAKE_COMPILER_IS_GNUCXX)
    if (GCC_VERSION VERSION_GREATER 4.7)
        add_definitions(-std=c++11)
    elseif(GCC_VERSION VERSION_GREATER 5.1)
        add_definitions(-std=c++14)
    endif()
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    # extend release-profile with fast-math
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -ffast-math -march=native")
endif()
set(PYBIND11_CPP_STANDARD -std=c++11)

find_package(OpenMP)
if (OPENMP_FOUND)
    message(STATUS, "Found OpenMP")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Third-party libs -----------------------------------------------------------------------------------------------------
# OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    # make FIND_PACKAGE friendly
    if(NOT OpenCV_FIND_QUIETLY)
        if(OpenCV_FIND_REQUIRED)
            message(FATAL_ERROR "OpenCV required but some headers or libs not found. ${ERR_MSG}")
        else(OpenCV_FIND_REQUIRED)
            message(STATUS "WARNING: OpenCV was not found. ${ERR_MSG}")
        endif(OpenCV_FIND_REQUIRED)
    endif(NOT OpenCV_FIND_QUIETLY)
else(NOT OpenCV_FOUND)
    message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV Link Libraries: ${OpenCV_LIBS}")
endif(NOT OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS} )

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# Python Interp Options ------------------------------------------------------------------------------------------------

# Soft inlier PnP  -----------------------------------------------------------------------------------------------------
find_package(pybind11 REQUIRED)
pybind11_add_module(lm_pnp lm_pnp_interface.cpp
        cnn.h
        util.h
        util.cpp
        maxloss.h
        types.h
        )
target_link_libraries(lm_pnp PUBLIC ${OpenCV_LIBS})
