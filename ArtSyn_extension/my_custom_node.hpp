#ifndef MY_CUSTOM_NODE_HPP
#define MY_CUSTOM_NODE_HPP

#include <memory>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/sub_viewport.hpp>
#include <godot_cpp/classes/viewport_texture.hpp>
#include <godot_cpp/classes/sprite2d.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/image_texture.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

namespace artsyn {
class StyleTransferBackend;
}

namespace godot {

class MyCustomNode : public Node {
    GDCLASS(MyCustomNode, Node);

private:
    NodePath viewport_path;
    NodePath display_sprite_path;
    SubViewport *viewport;
    Sprite2D *display_sprite;
    Ref<ImageTexture> display_texture;
    String backend_id;
    String backend_setup_error;
    String last_runtime_warning;
    String model_path = "res://addons/artsyn/models/model.pth";
    bool model_path_uses_backend_default = true;
    bool model_loaded = false;
    bool model_load_pending = true;
    bool benchmark_logging_enabled = false;
    bool verbose_logging_enabled = false;
    double benchmark_log_interval_seconds = 5.0;
    double inference_scale = 1.0;
    double accumulated_time = 0.0;
    int frame_count = 0;
    std::unique_ptr<artsyn::StyleTransferBackend> backend;

    void configure_backend(bool reload_model_if_needed);
    void load_model();
    void reset_benchmark_counters();
    void resolve_node_references();
    void set_runtime_warning(const String &warning);
    void clear_runtime_warning();

protected:
    static void _bind_methods();

public:
    MyCustomNode();
    ~MyCustomNode();
    void _ready() override;
    void _process(double delta) override;
    void set_viewport(NodePath path);
    NodePath get_viewport() const;
    void set_display_sprite(NodePath path);
    NodePath get_display_sprite() const;
    void set_backend_id(const String &backend_id);
    String get_backend_id() const;
    PackedStringArray get_available_backend_ids() const;
    void set_model_path(const String &path);
    String get_model_path() const;
    void set_benchmark_logging_enabled(bool enabled);
    bool is_benchmark_logging_enabled() const;
    void set_verbose_logging_enabled(bool enabled);
    bool is_verbose_logging_enabled() const;
    void set_benchmark_log_interval_seconds(double seconds);
    double get_benchmark_log_interval_seconds() const;
    void set_inference_scale(double scale);
    double get_inference_scale() const;
    String get_backend_name() const;
    String get_backend_last_error() const;
    bool is_model_loaded() const;
    void reload_model();
    String say_hello() const;
    Ref<Image> process_image(const Ref<Image> &input_image);
};

} // namespace godot

#endif // MY_CUSTOM_NODE_HPP
