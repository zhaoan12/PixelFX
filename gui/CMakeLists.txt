cmake_minimum_required(VERSION 3.10)
project(PixelFX_GUI)

set(CMAKE_CXX_STANDARD 17)

# Define source files
set(SOURCES
    main_gui.cpp
    gui_app.cpp
)

# ImGui includes (assumes ImGui is vendored in a subdirectory)
# You can replace this with your actual ImGui path
add_subdirectory(../third_party/imgui imgui_build)

# GLFW and OpenGL
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

include_directories(
    ../third_party/imgui
    ../third_party/imgui/backends
)

add_executable(PixelFX_GUI ${SOURCES})

target_include_directories(PixelFX_GUI PRIVATE
    ../third_party/imgui
    ../third_party/imgui/backends
)

target_link_libraries(PixelFX_GUI
    imgui
    OpenGL::GL
    glfw
)
