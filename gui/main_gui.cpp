#include "gui_app.h"

int main(int argc, char** argv) {
    GuiApp app;
    if (!app.init("PixelFX - Image Filter GUI", 1280, 720)) {
        return -1;
    }
    app.run();
    app.shutdown();
    return 0;
}
