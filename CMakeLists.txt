cmake_minimum_required (VERSION 3.10)

project (pfld_ncnn)

find_package(OpenCV REQUIRED)

find_package(facedetection REQUIRED)

find_package(ncnn REQUIRED)
if (${ncnn_FOUND})
  message("-- NCNN found.")
  message("-- NCNN_VULKAN flag is ${NCNN_VULKAN}")
  if (${NCNN_VULKAN})
    message("-- AUTO-ENABLING GPU_SUPPORT")
    set(GPU_SUPPORT ON)
  endif()
  include_directories(${ncnn_INCLUDE})
endif()

if (GPU_SUPPORT)
  message("-- GPU support is ENABLED")
  find_package(Vulkan) # REQUIRES ncnn to be built with vulkan
  if (${VULKAN_FOUND})
    message("-- Vulkan found.")
  else()
    message("-- ERROR: AUTO-DISABLING GPU_SUPPORT, because Vulkan was not found")
    set(GPU_SUPPORT OFF)
  endif()
else()
  message("-- GPU support is DISABLED")
endif()

configure_file("include/ncnn_config.h.in" "ncnn_config.h")
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(include)

add_executable(detect src/detect.cpp)
set_target_properties(detect PROPERTIES CXX_STANDARD 11)
target_link_libraries(detect ncnn facedetection ${OpenCV_LIBS})
