#include "style_transfer_backend_factory.hpp"

#ifdef ARTSYN_WITH_LIBTORCH_BACKEND
#include "libtorch_style_transfer_backend.hpp"
#endif
#include "onnx_style_transfer_backend.hpp"

namespace artsyn {

std::unique_ptr<StyleTransferBackend> create_style_transfer_backend(const godot::String &backend_id) {
#ifdef ARTSYN_WITH_LIBTORCH_BACKEND
    if (backend_id == "libtorch") {
        return std::make_unique<LibTorchStyleTransferBackend>();
    }
#endif

    if (backend_id == "onnx") {
        return std::make_unique<OnnxStyleTransferBackend>();
    }

    return nullptr;
}

godot::PackedStringArray get_available_style_transfer_backend_ids() {
    godot::PackedStringArray backend_ids;
#ifdef ARTSYN_WITH_LIBTORCH_BACKEND
    backend_ids.push_back("libtorch");
#endif
    backend_ids.push_back("onnx");
    return backend_ids;
}

godot::String get_default_style_transfer_backend_id() {
    godot::PackedStringArray backend_ids = get_available_style_transfer_backend_ids();
    if (backend_ids.is_empty()) {
        return "";
    }

    return backend_ids[0];
}

godot::String get_default_style_transfer_model_path(const godot::String &backend_id) {
    if (backend_id == "onnx") {
        return "res://addons/artsyn/models/model.onnx";
    }

    return "res://addons/artsyn/models/model.pth";
}

} // namespace artsyn
