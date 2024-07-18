cmake_minimum_required(VERSION 3.18)
project(PixelFX LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)

add_subdirectory(third_party/imgui)

include_directories(
    gui
    src/filters
    third_party/stb
    third_party/imgui
    third_party/imgui/backends
)

file(GLOB_RECURSE GUI_SRC
    gui/*.cpp
)
file(GLOB_RECURSE FILTERS_SRC
    src/filters/*.cu
)

add_executable(pixelfx_gui
    ${GUI_SRC}
    ${FILTERS_SRC}
)

set_target_properties(pixelfx_gui PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

target_link_libraries(pixelfx_gui
    imgui
    OpenGL::GL
    glfw
)
