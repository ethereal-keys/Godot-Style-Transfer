#include "my_custom_node.hpp"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/image_texture.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <stdexcept>
#include <chrono>

using namespace godot;

void MyCustomNode::_bind_methods() {
    ClassDB::bind_method(D_METHOD("say_hello"), &MyCustomNode::say_hello);
    ClassDB::bind_method(D_METHOD("set_viewport", "path"), &MyCustomNode::set_viewport);
    ClassDB::bind_method(D_METHOD("get_viewport"), &MyCustomNode::get_viewport);
    ClassDB::bind_method(D_METHOD("set_display_sprite", "path"), &MyCustomNode::set_display_sprite);
    ClassDB::bind_method(D_METHOD("get_display_sprite"), &MyCustomNode::get_display_sprite);

    ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "viewport_path"), "set_viewport", "get_viewport");
    ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "display_sprite_path"), "set_display_sprite", "get_display_sprite");
}

MyCustomNode::MyCustomNode() {
    if (!Engine::get_singleton()->is_editor_hint()) {
        UtilityFunctions::print("MyCustomNode created!");
    }
    viewport = nullptr;
    display_sprite = nullptr;
    display_texture = Ref<ImageTexture>();
    set_process(true);

    try {
        model = torch::jit::load("/home/sushanth/Dev/SML-Godot/ArtSynapse/model.pth", torch::kCPU);
        model.eval();
        if (torch::cuda::is_available()) {
            UtilityFunctions::print("CUDA available, moving model to GPU...");
            model.to(torch::kCUDA);
        } else {
            UtilityFunctions::print("CUDA not available, using CPU.");
        }
    } catch (const std::exception& e) {
        UtilityFunctions::print("Error loading model: ", e.what());
    }
}

MyCustomNode::~MyCustomNode() {
    if (!Engine::get_singleton()->is_editor_hint()) {
        UtilityFunctions::print("MyCustomNode destroyed!");
    }
}

String MyCustomNode::say_hello() const {
    if (!Engine::get_singleton()->is_editor_hint()) {
        UtilityFunctions::print("say_hello called!");
    }
    return "Hello from GDExtension!";
}

void MyCustomNode::set_viewport(NodePath path) {
    viewport = Object::cast_to<SubViewport>(get_node_or_null(path));
    if (!Engine::get_singleton()->is_editor_hint()) {
        UtilityFunctions::print("Viewport set to: ", viewport ? path : "null");
    }
}

NodePath MyCustomNode::get_viewport() const {
    return viewport ? viewport->get_path() : NodePath();
}

void MyCustomNode::set_display_sprite(NodePath path) {
    display_sprite = Object::cast_to<Sprite2D>(get_node_or_null(path));
    if (!Engine::get_singleton()->is_editor_hint()) {
        UtilityFunctions::print("DisplaySprite set to: ", display_sprite ? path : "null");
    }
}

NodePath MyCustomNode::get_display_sprite() const {
    return display_sprite ? display_sprite->get_path() : NodePath();
}

void MyCustomNode::_ready() {
    if (!Engine::get_singleton()->is_editor_hint()) {
        UtilityFunctions::print("MyCustomNode _ready called!");
    }
}

void MyCustomNode::_process(double delta) {
    if (Engine::get_singleton()->is_editor_hint()) {
        return;
    }

    if (!viewport || !display_sprite) {
        UtilityFunctions::print("Viewport or DisplaySprite not set!");
        return;
    }

    Ref<ViewportTexture> viewport_tex = viewport->get_texture();
    if (!viewport_tex.is_valid()) {
        UtilityFunctions::print("Viewport texture invalid!");
        return;
    }

    Ref<Image> image = viewport_tex->get_image();
    if (!image.is_valid() || image->is_empty()) {
        UtilityFunctions::print("Image invalid or empty!");
        return;
    }

    Ref<Image> processed_image = process_image(image);
    if (!processed_image.is_valid()) {
        UtilityFunctions::print("Processed image invalid!");
        return;
    }

    if (!display_texture.is_valid()) {
        display_texture = ImageTexture::create_from_image(processed_image);
        display_sprite->set_texture(display_texture);
    } else {
        display_texture->update(processed_image);
    }
    UtilityFunctions::print("Texture updated with style transfer!");
}

Ref<Image> MyCustomNode::process_image(const Ref<Image>& input_image) {
    auto start = std::chrono::high_resolution_clock::now();

    UtilityFunctions::print("Starting process_image...");
    Ref<Image> processed_image = input_image->duplicate();

    // Convert to RGB if the image is in RGBA format
    if (processed_image->get_format() == Image::FORMAT_RGBA8) {
        processed_image->convert(Image::FORMAT_RGB8);
        UtilityFunctions::print("Converted image from RGBA to RGB");
    }

    // Get the image dimensions dynamically
    int width = processed_image->get_width();
    int height = processed_image->get_height();
    UtilityFunctions::print("Image dimensions: ", width, "x", height);

    PackedByteArray data = processed_image->get_data();
    UtilityFunctions::print("Got image data, size: ", data.size());

    // Check data size for RGB format (3 bytes per pixel)
    if (data.size() != width * height * 3) {
        UtilityFunctions::print("Unexpected image data size: ", data.size(), ", expected: ", width * height * 3);
        return Ref<Image>();
    }

    // Create tensor with dynamic dimensions
    torch::Tensor tensor = torch::from_blob(const_cast<void*>(static_cast<const void*>(data.ptr())), {height, width, 3}, torch::kUInt8).to(torch::kFloat32);
    tensor = tensor / 255.0f;
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0);  // To 1x3xHxW (CHW)
    UtilityFunctions::print("Created tensor: ", tensor.sizes()[0], "x", tensor.sizes()[1], "x", tensor.sizes()[2], "x", tensor.sizes()[3]);

    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    torch::Tensor mean_tensor = torch::from_blob(mean, {3}, torch::kFloat32);
    torch::Tensor std_tensor = torch::from_blob(std, {3}, torch::kFloat32);
    tensor = (tensor - mean_tensor.view({1, 3, 1, 1})) / std_tensor.view({1, 3, 1, 1});
    UtilityFunctions::print("Tensor filled with image data");

    if (torch::cuda::is_available()) {
        tensor = tensor.to(torch::kCUDA);
    }

    try {
        std::vector<torch::jit::IValue> inputs = {tensor};
        torch::Tensor output = model.forward(inputs).toTensor();
        UtilityFunctions::print("Model inference done, output size: ", output.sizes()[0], "x", output.sizes()[1], "x", output.sizes()[2], "x", output.sizes()[3]);

        if (torch::cuda::is_available()) {
            output = output.to(torch::kCPU);
        }

        output = output.squeeze(0);  // 3xHxW (CHW)
        UtilityFunctions::print("Output squeezed, size: ", output.sizes()[0], "x", output.sizes()[1], "x", output.sizes()[2]);

        // Verify output dimensions match input
        if (output.sizes()[1] != height || output.sizes()[2] != width) {
            UtilityFunctions::print("Output dimensions do not match input: ", output.sizes()[1], "x", output.sizes()[2], ", expected: ", height, "x", width);
            return Ref<Image>();
        }

        output = output * std_tensor.view({3, 1, 1}) + mean_tensor.view({3, 1, 1});
        output = output.clamp(0, 1) * 255.0f;
        output = output.to(torch::kUInt8);
        output = output.permute({1, 2, 0}).contiguous();  // To HxWx3 (HWC)
        UtilityFunctions::print("Permuted output shape: ", output.sizes()[0], "x", output.sizes()[1], "x", output.sizes()[2]);

        Ref<Image> output_image = Image::create(width, height, false, Image::FORMAT_RGB8);
        PackedByteArray out_data = output_image->get_data();
        UtilityFunctions::print("Created output image, data size: ", out_data.size());

        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                UtilityFunctions::print("Pixel (", y, ",", x, "): R=", output[y][x][0].item<uint8_t>(),
                                        " G=", output[y][x][1].item<uint8_t>(),
                                        " B=", output[y][x][2].item<uint8_t>());
            }
        }

        memcpy(out_data.ptrw(), output.data_ptr<uint8_t>(), width * height * 3);
        UtilityFunctions::print("Filled output image data");

        output_image->set_data(width, height, false, Image::FORMAT_RGB8, out_data);
        UtilityFunctions::print("Set output image data");

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float fps = 1000.0f / duration;
        UtilityFunctions::print("Process time: ", duration, " ms, FPS: ", fps);

        return output_image;
    } catch (const std::exception& e) {
        UtilityFunctions::print("Inference error: ", e.what());
        return Ref<Image>();
    }
}

// GDExtension initialization
#include <godot_cpp/godot.hpp>

void my_extension_init(godot::ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    if (!Engine::get_singleton()->is_editor_hint()) {
        UtilityFunctions::print("GDExtension initializing!");
    }
    ClassDB::register_class<MyCustomNode>();
}

void my_extension_terminate(godot::ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
}

extern "C" {
GDExtensionBool GDE_EXPORT my_extension_library_init(
    GDExtensionInterfaceGetProcAddress p_interface,
    GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization *r_initialization) {
    godot::GDExtensionBinding::InitObject init_obj(p_interface, p_library, r_initialization);
    init_obj.register_initializer(my_extension_init);
    init_obj.register_terminator(my_extension_terminate);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
    return init_obj.init();
}
}
