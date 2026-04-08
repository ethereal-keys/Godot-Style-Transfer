#ifndef LIBTORCH_STYLE_TRANSFER_BACKEND_HPP
#define LIBTORCH_STYLE_TRANSFER_BACKEND_HPP

#include "style_transfer_backend.hpp"
#include <memory>

namespace artsyn {

class LibTorchStyleTransferBackend : public StyleTransferBackend {
public:
    LibTorchStyleTransferBackend();
    ~LibTorchStyleTransferBackend() override;

    godot::String get_backend_name() const override;
    bool load_model(const godot::String &model_path) override;
    bool is_model_loaded() const override;
    godot::String get_last_error() const override;
    godot::Ref<godot::Image> process_image(const godot::Ref<godot::Image> &input_image) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace artsyn

#endif // LIBTORCH_STYLE_TRANSFER_BACKEND_HPP
