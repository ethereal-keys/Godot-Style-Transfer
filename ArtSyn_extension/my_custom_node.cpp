#include "my_custom_node.hpp"
#include "style_transfer_backend_factory.hpp"

#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/image_texture.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

namespace {

String make_backend_hint_string(const PackedStringArray &backend_ids) {
    String hint;

    for (int index = 0; index < backend_ids.size(); ++index) {
        if (index > 0) {
            hint += ",";
        }
        hint += backend_ids[index];
    }

    return hint;
}

} // namespace

void MyCustomNode::_bind_methods() {
    String available_backend_hint = make_backend_hint_string(artsyn::get_available_style_transfer_backend_ids());

    ClassDB::bind_method(D_METHOD("say_hello"), &MyCustomNode::say_hello);
    ClassDB::bind_method(D_METHOD("set_viewport", "path"), &MyCustomNode::set_viewport);
    ClassDB::bind_method(D_METHOD("get_viewport"), &MyCustomNode::get_viewport);
    ClassDB::bind_method(D_METHOD("set_display_sprite", "path"), &MyCustomNode::set_display_sprite);
    ClassDB::bind_method(D_METHOD("get_display_sprite"), &MyCustomNode::get_display_sprite);
    ClassDB::bind_method(D_METHOD("set_backend_id", "backend_id"), &MyCustomNode::set_backend_id);
    ClassDB::bind_method(D_METHOD("get_backend_id"), &MyCustomNode::get_backend_id);
    ClassDB::bind_method(D_METHOD("get_available_backend_ids"), &MyCustomNode::get_available_backend_ids);
    ClassDB::bind_method(D_METHOD("set_model_path", "path"), &MyCustomNode::set_model_path);
    ClassDB::bind_method(D_METHOD("get_model_path"), &MyCustomNode::get_model_path);
    ClassDB::bind_method(D_METHOD("set_benchmark_logging_enabled", "enabled"), &MyCustomNode::set_benchmark_logging_enabled);
    ClassDB::bind_method(D_METHOD("is_benchmark_logging_enabled"), &MyCustomNode::is_benchmark_logging_enabled);
    ClassDB::bind_method(D_METHOD("set_verbose_logging_enabled", "enabled"), &MyCustomNode::set_verbose_logging_enabled);
    ClassDB::bind_method(D_METHOD("is_verbose_logging_enabled"), &MyCustomNode::is_verbose_logging_enabled);
    ClassDB::bind_method(D_METHOD("set_benchmark_log_interval_seconds", "seconds"), &MyCustomNode::set_benchmark_log_interval_seconds);
    ClassDB::bind_method(D_METHOD("get_benchmark_log_interval_seconds"), &MyCustomNode::get_benchmark_log_interval_seconds);
    ClassDB::bind_method(D_METHOD("set_inference_scale", "scale"), &MyCustomNode::set_inference_scale);
    ClassDB::bind_method(D_METHOD("get_inference_scale"), &MyCustomNode::get_inference_scale);
    ClassDB::bind_method(D_METHOD("get_backend_name"), &MyCustomNode::get_backend_name);
    ClassDB::bind_method(D_METHOD("get_backend_last_error"), &MyCustomNode::get_backend_last_error);
    ClassDB::bind_method(D_METHOD("is_model_loaded"), &MyCustomNode::is_model_loaded);
    ClassDB::bind_method(D_METHOD("reload_model"), &MyCustomNode::reload_model);

    ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "viewport_path"), "set_viewport", "get_viewport");
    ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "display_sprite_path"), "set_display_sprite", "get_display_sprite");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "backend_id", PROPERTY_HINT_ENUM, available_backend_hint), "set_backend_id", "get_backend_id");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "*.pth,*.onnx"), "set_model_path", "get_model_path");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "benchmark_logging_enabled"), "set_benchmark_logging_enabled", "is_benchmark_logging_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "verbose_logging_enabled"), "set_verbose_logging_enabled", "is_verbose_logging_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "benchmark_log_interval_seconds", PROPERTY_HINT_RANGE, "0.1,60.0,0.1,or_greater"), "set_benchmark_log_interval_seconds", "get_benchmark_log_interval_seconds");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inference_scale", PROPERTY_HINT_RANGE, "0.1,1.0,0.05"), "set_inference_scale", "get_inference_scale");
}

MyCustomNode::MyCustomNode() {
    if (!Engine::get_singleton()->is_editor_hint() && verbose_logging_enabled) {
        UtilityFunctions::print("MyCustomNode created!");
    }

    viewport = nullptr;
    display_sprite = nullptr;
    display_texture = Ref<ImageTexture>();
    configure_backend(false);
    set_process(true);
}

MyCustomNode::~MyCustomNode() {
    if (!Engine::get_singleton()->is_editor_hint() && verbose_logging_enabled) {
        UtilityFunctions::print("MyCustomNode destroyed!");
    }
}

String MyCustomNode::say_hello() const {
    if (!Engine::get_singleton()->is_editor_hint() && verbose_logging_enabled) {
        UtilityFunctions::print("say_hello called!");
    }

    return "Hello from GDExtension!";
}

void MyCustomNode::set_viewport(NodePath path) {
    viewport_path = path;
    viewport = Object::cast_to<SubViewport>(get_node_or_null(path));
    if (!Engine::get_singleton()->is_editor_hint() && verbose_logging_enabled) {
        UtilityFunctions::print("Viewport set to: ", viewport ? path : "null");
    }
}

NodePath MyCustomNode::get_viewport() const {
    return viewport_path;
}

void MyCustomNode::set_display_sprite(NodePath path) {
    display_sprite_path = path;
    display_sprite = Object::cast_to<Sprite2D>(get_node_or_null(path));
    if (!Engine::get_singleton()->is_editor_hint() && verbose_logging_enabled) {
        UtilityFunctions::print("DisplaySprite set to: ", display_sprite ? path : "null");
    }
}

NodePath MyCustomNode::get_display_sprite() const {
    return display_sprite_path;
}

void MyCustomNode::set_backend_id(const String &new_backend_id) {
    String normalized_backend_id = new_backend_id.strip_edges().to_lower();
    if (normalized_backend_id.is_empty()) {
        normalized_backend_id = artsyn::get_default_style_transfer_backend_id();
    }

    if (backend_id == normalized_backend_id && backend) {
        return;
    }

    backend_id = normalized_backend_id;
    configure_backend(is_inside_tree() && !Engine::get_singleton()->is_editor_hint());
}

String MyCustomNode::get_backend_id() const {
    return backend_id;
}

PackedStringArray MyCustomNode::get_available_backend_ids() const {
    return artsyn::get_available_style_transfer_backend_ids();
}

void MyCustomNode::set_model_path(const String &path) {
    model_path = path;
    model_path_uses_backend_default = false;
    model_load_pending = true;

    if (is_inside_tree() && !Engine::get_singleton()->is_editor_hint()) {
        load_model();
    }
}

String MyCustomNode::get_model_path() const {
    return model_path;
}

void MyCustomNode::set_benchmark_logging_enabled(bool enabled) {
    benchmark_logging_enabled = enabled;

    if (!benchmark_logging_enabled) {
        reset_benchmark_counters();
    }
}

bool MyCustomNode::is_benchmark_logging_enabled() const {
    return benchmark_logging_enabled;
}

void MyCustomNode::set_verbose_logging_enabled(bool enabled) {
    verbose_logging_enabled = enabled;
}

bool MyCustomNode::is_verbose_logging_enabled() const {
    return verbose_logging_enabled;
}

void MyCustomNode::set_benchmark_log_interval_seconds(double seconds) {
    benchmark_log_interval_seconds = seconds > 0.0 ? seconds : 0.1;
}

double MyCustomNode::get_benchmark_log_interval_seconds() const {
    return benchmark_log_interval_seconds;
}

void MyCustomNode::set_inference_scale(double scale) {
    if (scale < 0.1) {
        inference_scale = 0.1;
    } else if (scale > 1.0) {
        inference_scale = 1.0;
    } else {
        inference_scale = scale;
    }
}

double MyCustomNode::get_inference_scale() const {
    return inference_scale;
}

void MyCustomNode::configure_backend(bool reload_model_if_needed) {
    backend_setup_error = "";
    model_loaded = false;
    model_load_pending = true;
    if (backend_id.is_empty()) {
        backend_id = artsyn::get_default_style_transfer_backend_id();
    }

    if (model_path_uses_backend_default || model_path.is_empty()) {
        model_path = artsyn::get_default_style_transfer_model_path(backend_id);
        model_path_uses_backend_default = true;
    }

    backend = artsyn::create_style_transfer_backend(backend_id);

    if (!backend) {
        backend_setup_error = "Unsupported backend id: " + backend_id;
        if (!Engine::get_singleton()->is_editor_hint()) {
            UtilityFunctions::print(backend_setup_error);
        }
        return;
    }

    if (reload_model_if_needed) {
        load_model();
    }
}

String MyCustomNode::get_backend_name() const {
    if (!backend) {
        return "";
    }

    return backend->get_backend_name();
}

String MyCustomNode::get_backend_last_error() const {
    if (!backend) {
        if (!backend_setup_error.is_empty()) {
            return backend_setup_error;
        }
        return "No style transfer backend is configured.";
    }

    return backend->get_last_error();
}

bool MyCustomNode::is_model_loaded() const {
    return model_loaded;
}

void MyCustomNode::reload_model() {
    if (!Engine::get_singleton()->is_editor_hint()) {
        model_load_pending = true;
        load_model();
    }
}

void MyCustomNode::reset_benchmark_counters() {
    accumulated_time = 0.0;
    frame_count = 0;
}

void MyCustomNode::resolve_node_references() {
    if (!viewport_path.is_empty()) {
        viewport = Object::cast_to<SubViewport>(get_node_or_null(viewport_path));
    }

    if (!display_sprite_path.is_empty()) {
        display_sprite = Object::cast_to<Sprite2D>(get_node_or_null(display_sprite_path));
    }
}

void MyCustomNode::set_runtime_warning(const String &warning) {
    if (warning == last_runtime_warning) {
        return;
    }

    last_runtime_warning = warning;
    if (!warning.is_empty()) {
        UtilityFunctions::push_warning(warning);
    }
}

void MyCustomNode::clear_runtime_warning() {
    last_runtime_warning = "";
}

void MyCustomNode::load_model() {
    model_loaded = false;
    model_load_pending = false;
    clear_runtime_warning();

    if (!backend) {
        UtilityFunctions::print(get_backend_last_error());
        return;
    }

    if (model_path.is_empty()) {
        UtilityFunctions::print("Model path is empty; skipping model load.");
        return;
    }

    String resolved_model_path = ProjectSettings::get_singleton()->globalize_path(model_path);
    if (!FileAccess::file_exists(model_path) && !FileAccess::file_exists(resolved_model_path)) {
        UtilityFunctions::print("Model file not found: ", model_path, " (resolved to ", resolved_model_path, ")");
        return;
    }

    if (backend->load_model(resolved_model_path)) {
        model_loaded = backend->is_model_loaded();
        String backend_details = backend->get_backend_details();
        if (backend_details.is_empty()) {
            UtilityFunctions::print("Loaded model from ", model_path, " using backend ", backend->get_backend_name());
        } else {
            UtilityFunctions::print(
                    "Loaded model from ",
                    model_path,
                    " using backend ",
                    backend->get_backend_name(),
                    " (",
                    backend_details,
                    ")");
        }
    } else {
        UtilityFunctions::print(
                "Error loading model from ",
                model_path,
                " using backend ",
                backend->get_backend_name(),
                ": ",
                backend->get_last_error());
    }
}

void MyCustomNode::_ready() {
    if (!Engine::get_singleton()->is_editor_hint()) {
        if (verbose_logging_enabled) {
            UtilityFunctions::print("MyCustomNode _ready called!");
        }
        resolve_node_references();
        load_model();
    }
}

void MyCustomNode::_process(double delta) {
    if (Engine::get_singleton()->is_editor_hint()) {
        return;
    }

    if (model_load_pending) {
        load_model();
    }

    if (!viewport || !display_sprite) {
        resolve_node_references();
    }

    if (!model_loaded) {
        set_runtime_warning("ArtSyn model is not loaded. Check backend_id, model_path, and runtime dependencies.");
        return;
    }

    if (!viewport || !display_sprite) {
        set_runtime_warning("ArtSyn viewport_path or display_sprite_path is not set.");
        return;
    }

    Ref<ViewportTexture> viewport_tex = viewport->get_texture();
    if (!viewport_tex.is_valid()) {
        set_runtime_warning("ArtSyn viewport texture is invalid.");
        return;
    }

    Ref<Image> image = viewport_tex->get_image();
    if (!image.is_valid() || image->is_empty()) {
        set_runtime_warning("ArtSyn input image is invalid or empty.");
        return;
    }

    Ref<Image> backend_input_image = image;
    int original_width = image->get_width();
    int original_height = image->get_height();
    if (inference_scale < 0.999) {
        int scaled_width = static_cast<int>(original_width * inference_scale);
        int scaled_height = static_cast<int>(original_height * inference_scale);
        if (scaled_width < 1) {
            scaled_width = 1;
        }
        if (scaled_height < 1) {
            scaled_height = 1;
        }

        backend_input_image = image->duplicate();
        backend_input_image->resize(scaled_width, scaled_height, Image::INTERPOLATE_LANCZOS);
    }

    Ref<Image> processed_image = process_image(backend_input_image);
    if (!processed_image.is_valid()) {
        String backend_error = get_backend_last_error();
        if (backend_error.is_empty()) {
            set_runtime_warning("ArtSyn backend returned an invalid image.");
        } else {
            set_runtime_warning("ArtSyn backend failed: " + backend_error);
        }
        return;
    }

    clear_runtime_warning();

    if (processed_image->get_width() != original_width || processed_image->get_height() != original_height) {
        processed_image->resize(original_width, original_height, Image::INTERPOLATE_LANCZOS);
    }

    if (!display_texture.is_valid()) {
        display_texture = ImageTexture::create_from_image(processed_image);
        display_sprite->set_texture(display_texture);
    } else {
        display_texture->update(processed_image);
    }

    if (benchmark_logging_enabled) {
        accumulated_time += delta;
        frame_count++;

        if (accumulated_time >= benchmark_log_interval_seconds) {
            float average_fps = frame_count / accumulated_time;
            UtilityFunctions::print("Average FPS over ", accumulated_time, " seconds: ", average_fps);
            reset_benchmark_counters();
        }
    }
}

Ref<Image> MyCustomNode::process_image(const Ref<Image> &input_image) {
    if (!backend) {
        return Ref<Image>();
    }

    return backend->process_image(input_image);
}

void my_extension_init(godot::ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
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
