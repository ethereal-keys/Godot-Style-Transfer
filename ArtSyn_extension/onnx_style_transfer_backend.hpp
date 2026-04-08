#ifndef ONNX_STYLE_TRANSFER_BACKEND_HPP
#define ONNX_STYLE_TRANSFER_BACKEND_HPP

#include "style_transfer_backend.hpp"

#include <memory>

namespace artsyn {

class OnnxStyleTransferBackend : public StyleTransferBackend {
public:
    OnnxStyleTransferBackend();
    ~OnnxStyleTransferBackend() override;

    godot::String get_backend_name() const override;
    godot::String get_backend_details() const override;
    bool load_model(const godot::String &model_path) override;
    bool is_model_loaded() const override;
    godot::String get_last_error() const override;
    godot::Ref<godot::Image> process_image(const godot::Ref<godot::Image> &input_image) override;

private:
#ifdef ARTSYN_WITH_ONNX_RUNTIME
    class Impl;
    std::unique_ptr<Impl> impl;
#endif
    godot::String last_error;
};

} // namespace artsyn

#endif // ONNX_STYLE_TRANSFER_BACKEND_HPP
