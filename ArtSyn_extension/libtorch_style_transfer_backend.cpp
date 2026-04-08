#include "libtorch_style_transfer_backend.hpp"

#include <cstdint>
#include <cstring>
#include <vector>
#include <torch/script.h>
#include <torch/torch.h>

namespace artsyn {

class LibTorchStyleTransferBackend::Impl {
public:
    torch::jit::script::Module model;
    bool model_loaded = false;
    godot::String last_error;
};

LibTorchStyleTransferBackend::LibTorchStyleTransferBackend() :
        impl(std::make_unique<Impl>()) {}

LibTorchStyleTransferBackend::~LibTorchStyleTransferBackend() = default;

godot::String LibTorchStyleTransferBackend::get_backend_name() const {
    return "LibTorch";
}

bool LibTorchStyleTransferBackend::load_model(const godot::String &model_path) {
    impl->model_loaded = false;
    impl->last_error = "";

    try {
        auto utf8_model_path = model_path.utf8();
        impl->model = torch::jit::load(utf8_model_path.get_data(), torch::kCPU);
        impl->model.eval();

        if (torch::cuda::is_available()) {
            impl->model.to(torch::kCUDA);
        }

        impl->model_loaded = true;
        return true;
    } catch (const std::exception &e) {
        impl->last_error = e.what();
        return false;
    }
}

bool LibTorchStyleTransferBackend::is_model_loaded() const {
    return impl->model_loaded;
}

godot::String LibTorchStyleTransferBackend::get_last_error() const {
    return impl->last_error;
}

godot::Ref<godot::Image> LibTorchStyleTransferBackend::process_image(const godot::Ref<godot::Image> &input_image) {
    impl->last_error = "";

    if (!impl->model_loaded) {
        impl->last_error = "Model is not loaded.";
        return godot::Ref<godot::Image>();
    }

    godot::Ref<godot::Image> processed_image = input_image->duplicate();
    if (processed_image->get_format() == godot::Image::FORMAT_RGBA8) {
        processed_image->convert(godot::Image::FORMAT_RGB8);
    }

    int width = processed_image->get_width();
    int height = processed_image->get_height();
    godot::PackedByteArray data = processed_image->get_data();

    if (data.size() != width * height * 3) {
        impl->last_error = "Unexpected image data size for RGB8 image.";
        return godot::Ref<godot::Image>();
    }

    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    torch::Tensor tensor = torch::from_blob(
            const_cast<void *>(static_cast<const void *>(data.ptr())),
            {height, width, 3},
            torch::kUInt8)
                                   .to(torch::kFloat32)
                                   .to(device);

    tensor = tensor / 255.0f;
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0);

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor mean_tensor = torch::tensor({0.485f, 0.456f, 0.406f}, tensor_options).view({1, 3, 1, 1});
    torch::Tensor std_tensor = torch::tensor({0.229f, 0.224f, 0.225f}, tensor_options).view({1, 3, 1, 1});
    tensor = (tensor - mean_tensor) / std_tensor;

    try {
        torch::InferenceMode inference_mode_guard;
        std::vector<torch::jit::IValue> inputs = {tensor};
        torch::Tensor output = impl->model.forward(inputs).toTensor();

        if (output.sizes()[2] != height || output.sizes()[3] != width) {
            impl->last_error = "Output dimensions do not match input dimensions.";
            return godot::Ref<godot::Image>();
        }

        output = output * std_tensor + mean_tensor;
        output = output.clamp(0, 1).mul(255.0f).to(torch::kUInt8);

        if (output.is_cuda()) {
            output = output.to(torch::kCPU);
        }

        output = output.squeeze(0).permute({1, 2, 0}).contiguous();

        godot::Ref<godot::Image> output_image = godot::Image::create(width, height, false, godot::Image::FORMAT_RGB8);
        godot::PackedByteArray out_data = output_image->get_data();
        std::memcpy(out_data.ptrw(), output.data_ptr<uint8_t>(), width * height * 3);
        output_image->set_data(width, height, false, godot::Image::FORMAT_RGB8, out_data);
        return output_image;
    } catch (const std::exception &e) {
        impl->last_error = e.what();
        return godot::Ref<godot::Image>();
    }
}

} // namespace artsyn
