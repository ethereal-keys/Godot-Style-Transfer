#ifndef STYLE_TRANSFER_BACKEND_FACTORY_HPP
#define STYLE_TRANSFER_BACKEND_FACTORY_HPP

#include "style_transfer_backend.hpp"

#include <memory>

#include <godot_cpp/variant/packed_string_array.hpp>

namespace artsyn {

std::unique_ptr<StyleTransferBackend> create_style_transfer_backend(const godot::String &backend_id);
godot::PackedStringArray get_available_style_transfer_backend_ids();
godot::String get_default_style_transfer_backend_id();
godot::String get_default_style_transfer_model_path(const godot::String &backend_id);

} // namespace artsyn

#endif // STYLE_TRANSFER_BACKEND_FACTORY_HPP
