#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include "gui_app.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>
#include <iostream>

bool GuiApp::init(const char* title, int w, int h) {
    width = w;
    height = h;

    if (!glfwInit()) return false;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    GLFWwindow* win = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!win) return false;
    window = win;
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    return true;
}

void GuiApp::run() {
    GLFWwindow* win = static_cast<GLFWwindow*>(window);
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        updateCudaTexture();

        renderUI();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(win, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }
}

void GuiApp::renderUI() {
    ImGui::Begin("PixelFX - Filters");

    if (ImGui::Button("Open Image")) {
        loadImage("assets/sample.png");  
    }

    ImGui::Separator();
    ImGui::Combo("Select Filter", &selectedFilterIdx, filterNames, IM_ARRAYSIZE(filterNames));

    ImGui::Text("Select a filter:");
    if (ImGui::Combo("##filter_combo", &selectedFilterIdx, filterNames, IM_ARRAYSIZE(filterNames))) {
        switch (selectedFilterIdx) {
            case 0: currentFilter = FilterType::None; break;
            case 1: currentFilter = FilterType::Sobel; break;
            case 2: currentFilter = FilterType::Prewitt; break;
            case 3: currentFilter = FilterType::Blur; break;
            case 4: currentFilter = FilterType::Laplacian; break;
            case 5: currentFilter = FilterType::GrayscaleInvert; break;
            case 6: currentFilter = FilterType::Invert; break;
            case 7: currentFilter = FilterType::Sharpen; break;
            case 8: currentFilter = FilterType::Sepia; break;
            case 9: currentFilter = FilterType::Mean; break;
            case 10: currentFilter = FilterType::Emboss; break;
        }

    }

    ImGui::Separator();
    ImGui::Text("Live Filter Output:");
    ImGui::Image((void*)(intptr_t)glTex, ImVec2((float)texWidth, (float)texHeight));

    ImGui::End();
}

void GuiApp::updateCudaTexture() {
    cudaGraphicsMapResources(1, &cudaTexResource);
    cudaArray_t cudaArray;
    cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaTexResource, 0, 0);

    applyFilter(currentFilter, cudaArray, texWidth, texHeight);  // real filter dispatch

    cudaGraphicsUnmapResources(1, &cudaTexResource);
}


void GuiApp::shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(static_cast<GLFWwindow*>(window));
    glfwTerminate();
}

void GuiApp::loadImage(const std::string& path) {
    int channels;
    unsigned char* data = stbi_load(path.c_str(), &texWidth, &texHeight, &channels, STBI_rgb_alpha);
    if (!data) return;

    glBindTexture(GL_TEXTURE_2D, glTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texWidth, texHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glBindTexture(GL_TEXTURE_2D, 0);

    stbi_image_free(data);
}

///////////////////////////////////////////////////////////////////////////////

// Create OpenGL texture
glGenTextures(1, &glTex);
glBindTexture(GL_TEXTURE_2D, glTex);
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texWidth, texHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
glBindTexture(GL_TEXTURE_2D, 0);

// Register with CUDA
cudaGraphicsGLRegisterImage(&cudaTexResource, glTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

///////////////////////////////////////////////////////////////////////////////
