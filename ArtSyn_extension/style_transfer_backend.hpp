#ifndef STYLE_TRANSFER_BACKEND_HPP
#define STYLE_TRANSFER_BACKEND_HPP

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/string.hpp>

namespace artsyn {

class StyleTransferBackend {
public:
    virtual ~StyleTransferBackend() = default;

    virtual godot::String get_backend_name() const = 0;
    virtual godot::String get_backend_details() const { return ""; }
    virtual bool load_model(const godot::String &model_path) = 0;
    virtual bool is_model_loaded() const = 0;
    virtual godot::String get_last_error() const = 0;
    virtual godot::Ref<godot::Image> process_image(const godot::Ref<godot::Image> &input_image) = 0;
};

} // namespace artsyn

#endif // STYLE_TRANSFER_BACKEND_HPP
