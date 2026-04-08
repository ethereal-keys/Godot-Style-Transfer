#include "onnx_style_transfer_backend.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef ARTSYN_WITH_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace {

constexpr float kChannelMean[] = {0.485f, 0.456f, 0.406f};
constexpr float kChannelStd[] = {0.229f, 0.224f, 0.225f};
constexpr int kDefaultOnnxCudaDeviceId = 0;

const char *kOnnxBackendUnavailableMessage =
        "ONNX Runtime backend selection is wired up, but ONNX Runtime support was not compiled into this build.";
const char *kOnnxModelExtensionMessage =
        "ONNX backend expects a .onnx model file.";

bool prepare_rgb_image(
        const godot::Ref<godot::Image> &input_image,
        godot::Ref<godot::Image> &prepared_image,
        godot::String &error_message) {
    if (!input_image.is_valid() || input_image->is_empty()) {
        error_message = "Input image is invalid or empty.";
        return false;
    }

    prepared_image = input_image->duplicate();
    if (prepared_image->get_format() != godot::Image::FORMAT_RGB8) {
        prepared_image->convert(godot::Image::FORMAT_RGB8);
    }

    int width = prepared_image->get_width();
    int height = prepared_image->get_height();
    godot::PackedByteArray data = prepared_image->get_data();
    if (data.size() != width * height * 3) {
        error_message = "Unexpected image data size for RGB8 image.";
        return false;
    }

    return true;
}

#ifdef ARTSYN_WITH_ONNX_RUNTIME
bool has_execution_provider(const std::vector<std::string> &provider_names, const char *provider_name) {
    return std::find(provider_names.begin(), provider_names.end(), provider_name) != provider_names.end();
}

std::string join_provider_names(const std::vector<std::string> &provider_names) {
    if (provider_names.empty()) {
        return "<none>";
    }

    std::string joined;
    for (size_t index = 0; index < provider_names.size(); ++index) {
        if (index > 0) {
            joined += ", ";
        }
        joined += provider_names[index];
    }

    return joined;
}

bool try_enable_cuda_execution_provider(
        Ort::SessionOptions &session_options,
        int device_id,
        godot::String &error_message) {
    try {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = device_id;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        return true;
    } catch (const Ort::Exception &e) {
        error_message = e.what();
        return false;
    }
}

bool validate_nchw_shape(
        const std::vector<int64_t> &shape,
        godot::String &error_message) {
    if (shape.size() != 4) {
        error_message = "Expected a 4D NCHW tensor.";
        return false;
    }

    if (shape[0] > 0 && shape[0] != 1) {
        error_message = "Expected batch dimension 1.";
        return false;
    }

    if (shape[1] > 0 && shape[1] != 3) {
        error_message = "Expected 3 input channels.";
        return false;
    }

    return true;
}

void resize_image_for_model_input(
        const std::vector<int64_t> &shape,
        godot::Ref<godot::Image> &prepared_image) {
    if (!prepared_image.is_valid()) {
        return;
    }

    int target_height = prepared_image->get_height();
    int target_width = prepared_image->get_width();

    if (shape.size() >= 4) {
        if (shape[2] > 0) {
            target_height = static_cast<int>(shape[2]);
        }

        if (shape[3] > 0) {
            target_width = static_cast<int>(shape[3]);
        }
    }

    if (target_width < 1 || target_height < 1) {
        return;
    }

    if (prepared_image->get_width() != target_width || prepared_image->get_height() != target_height) {
        prepared_image->resize(target_width, target_height, godot::Image::INTERPOLATE_LANCZOS);
    }
}

void image_to_nchw_tensor(const godot::Ref<godot::Image> &image, std::vector<float> &tensor_values) {
    int width = image->get_width();
    int height = image->get_height();
    const uint8_t *data = image->get_data().ptr();
    size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);

    tensor_values.assign(pixel_count * 3, 0.0f);
    for (size_t pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
        size_t src_index = pixel_index * 3;
        for (int channel = 0; channel < 3; ++channel) {
            float pixel_value = static_cast<float>(data[src_index + channel]) / 255.0f;
            tensor_values[static_cast<size_t>(channel) * pixel_count + pixel_index] =
                    (pixel_value - kChannelMean[channel]) / kChannelStd[channel];
        }
    }
}

godot::Ref<godot::Image> nchw_tensor_to_image(
        const float *tensor_data,
        int width,
        int height,
        godot::String &error_message) {
    if (!tensor_data) {
        error_message = "ONNX output tensor data is null.";
        return godot::Ref<godot::Image>();
    }

    size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    godot::PackedByteArray out_data;
    out_data.resize(static_cast<int>(pixel_count * 3));
    uint8_t *out_ptr = out_data.ptrw();

    for (size_t pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
        size_t dst_index = pixel_index * 3;
        for (int channel = 0; channel < 3; ++channel) {
            float denormalized =
                    tensor_data[static_cast<size_t>(channel) * pixel_count + pixel_index] * kChannelStd[channel] +
                    kChannelMean[channel];
            float scaled = denormalized * 255.0f;
            float clamped = std::clamp(scaled, 0.0f, 255.0f);
            out_ptr[dst_index + channel] = static_cast<uint8_t>(clamped + 0.5f);
        }
    }

    godot::Ref<godot::Image> output_image = godot::Image::create(width, height, false, godot::Image::FORMAT_RGB8);
    output_image->set_data(width, height, false, godot::Image::FORMAT_RGB8, out_data);
    return output_image;
}
#endif

} // namespace

namespace artsyn {

#ifdef ARTSYN_WITH_ONNX_RUNTIME
class OnnxStyleTransferBackend::Impl {
public:
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ArtSynOnnx"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;
    std::vector<std::string> available_execution_providers;
    std::string active_execution_provider;
    std::string execution_provider_details;
    bool model_loaded = false;

    Impl() {
        reset_session_options();
    }

    void reset_session_options() {
        session_options = Ort::SessionOptions{};
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }
};
#endif

OnnxStyleTransferBackend::OnnxStyleTransferBackend() {
#ifdef ARTSYN_WITH_ONNX_RUNTIME
    impl = std::make_unique<Impl>();
#endif
}

OnnxStyleTransferBackend::~OnnxStyleTransferBackend() = default;

godot::String OnnxStyleTransferBackend::get_backend_name() const {
#ifdef ARTSYN_WITH_ONNX_RUNTIME
    if (impl && !impl->active_execution_provider.empty()) {
        return "ONNX Runtime [" + godot::String(impl->active_execution_provider.c_str()) + "]";
    }
#endif
    return "ONNX Runtime";
}

godot::String OnnxStyleTransferBackend::get_backend_details() const {
#ifdef ARTSYN_WITH_ONNX_RUNTIME
    if (impl && !impl->execution_provider_details.empty()) {
        return godot::String(impl->execution_provider_details.c_str());
    }
#endif
    return "";
}

bool OnnxStyleTransferBackend::load_model(const godot::String &model_path) {
    last_error = "";

    if (model_path.get_extension().to_lower() != "onnx") {
        last_error = kOnnxModelExtensionMessage;
        return false;
    }

#ifdef ARTSYN_WITH_ONNX_RUNTIME
    impl->session.reset();
    impl->input_name.clear();
    impl->output_name.clear();
    impl->input_shape.clear();
    impl->available_execution_providers.clear();
    impl->active_execution_provider.clear();
    impl->execution_provider_details.clear();
    impl->model_loaded = false;
    impl->reset_session_options();

    try {
        impl->available_execution_providers = Ort::GetAvailableProviders();
        std::string available_providers_string = join_provider_names(impl->available_execution_providers);

        if (has_execution_provider(impl->available_execution_providers, "CUDAExecutionProvider")) {
            godot::String cuda_provider_error;
            if (try_enable_cuda_execution_provider(impl->session_options, kDefaultOnnxCudaDeviceId, cuda_provider_error)) {
                impl->active_execution_provider = "CUDAExecutionProvider";
                impl->execution_provider_details =
                        "Execution provider: CUDAExecutionProvider; available providers: " + available_providers_string;
            } else {
                impl->active_execution_provider = "CPUExecutionProvider";
                impl->execution_provider_details =
                        "Execution provider: CPUExecutionProvider; CUDA provider setup failed: " +
                        std::string(cuda_provider_error.utf8().get_data()) +
                        "; available providers: " + available_providers_string;
            }
        } else {
            impl->active_execution_provider = "CPUExecutionProvider";
            impl->execution_provider_details =
                    "Execution provider: CPUExecutionProvider; available providers: " + available_providers_string;
        }

        godot::String session_error;
        try {
            auto utf8_model_path = model_path.utf8();
            impl->session = std::make_unique<Ort::Session>(impl->env, utf8_model_path.get_data(), impl->session_options);
        } catch (const Ort::Exception &e) {
            session_error = e.what();
            impl->session.reset();
        }

        if (!impl->session) {
            if (impl->active_execution_provider == "CUDAExecutionProvider") {
                std::string cuda_session_error = std::string(session_error.utf8().get_data());

                impl->reset_session_options();
                impl->active_execution_provider = "CPUExecutionProvider";
                impl->execution_provider_details =
                        "Execution provider: CPUExecutionProvider; CUDA session initialization failed: " +
                        cuda_session_error + "; available providers: " + available_providers_string;

                try {
                    auto utf8_model_path = model_path.utf8();
                    impl->session = std::make_unique<Ort::Session>(impl->env, utf8_model_path.get_data(), impl->session_options);
                } catch (const Ort::Exception &e) {
                    last_error = e.what();
                    impl->session.reset();
                    return false;
                }
            } else {
                last_error = session_error;
                return false;
            }
        }

        if (impl->session->GetInputCount() < 1 || impl->session->GetOutputCount() < 1) {
            last_error = "ONNX model must expose at least one input and one output.";
            impl->session.reset();
            return false;
        }

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = impl->session->GetInputNameAllocated(0, allocator);
        auto output_name = impl->session->GetOutputNameAllocated(0, allocator);
        impl->input_name = input_name.get() ? input_name.get() : "";
        impl->output_name = output_name.get() ? output_name.get() : "";

        if (impl->input_name.empty() || impl->output_name.empty()) {
            last_error = "Could not resolve ONNX model input or output names.";
            impl->session.reset();
            return false;
        }

        Ort::TypeInfo input_type_info = impl->session->GetInputTypeInfo(0);
        impl->input_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        impl->model_loaded = true;
        last_error = "";
        return true;
    } catch (const Ort::Exception &e) {
        last_error = e.what();
        impl->session.reset();
        return false;
    }
#else
    last_error = kOnnxBackendUnavailableMessage;
    return false;
#endif
}

bool OnnxStyleTransferBackend::is_model_loaded() const {
#ifdef ARTSYN_WITH_ONNX_RUNTIME
    return impl && impl->model_loaded;
#else
    return false;
#endif
}

godot::String OnnxStyleTransferBackend::get_last_error() const {
    if (!last_error.is_empty()) {
        return last_error;
    }

#ifdef ARTSYN_WITH_ONNX_RUNTIME
    if (!is_model_loaded()) {
        return "Model is not loaded.";
    }
#else
    if (!is_model_loaded()) {
        return kOnnxBackendUnavailableMessage;
    }
#endif

    return "";
}

godot::Ref<godot::Image> OnnxStyleTransferBackend::process_image(const godot::Ref<godot::Image> &input_image) {
    last_error = "";

#ifdef ARTSYN_WITH_ONNX_RUNTIME
    if (!impl || !impl->model_loaded || !impl->session) {
        last_error = "Model is not loaded.";
        return godot::Ref<godot::Image>();
    }

    godot::Ref<godot::Image> prepared_image;
    if (!prepare_rgb_image(input_image, prepared_image, last_error)) {
        return godot::Ref<godot::Image>();
    }

    if (!validate_nchw_shape(impl->input_shape, last_error)) {
        return godot::Ref<godot::Image>();
    }

    resize_image_for_model_input(impl->input_shape, prepared_image);

    int width = prepared_image->get_width();
    int height = prepared_image->get_height();
    std::vector<float> input_tensor_values;
    image_to_nchw_tensor(prepared_image, input_tensor_values);

    std::vector<int64_t> input_shape = {1, 3, height, width};
    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                input_tensor_values.size(),
                input_shape.data(),
                input_shape.size());

        const char *input_names[] = {impl->input_name.c_str()};
        const char *output_names[] = {impl->output_name.c_str()};
        Ort::RunOptions run_options;
        std::vector<Ort::Value> outputs = impl->session->Run(
                run_options,
                input_names,
                &input_tensor,
                1,
                output_names,
                1);

        if (outputs.empty() || !outputs[0].IsTensor()) {
            last_error = "ONNX inference did not return a tensor output.";
            return godot::Ref<godot::Image>();
        }

        Ort::TensorTypeAndShapeInfo output_info = outputs[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = output_info.GetShape();
        if (output_shape.size() != 4) {
            last_error = "Expected a 4D tensor output from the ONNX model.";
            return godot::Ref<godot::Image>();
        }

        if (output_shape[0] != 1 || output_shape[1] != 3) {
            last_error = "Expected ONNX model output shape to begin with [1, 3, ...].";
            return godot::Ref<godot::Image>();
        }

        int output_height = static_cast<int>(output_shape[2]);
        int output_width = static_cast<int>(output_shape[3]);
        if (output_height < 1 || output_width < 1) {
            last_error = "ONNX model returned invalid output dimensions.";
            return godot::Ref<godot::Image>();
        }

        const float *output_tensor = outputs[0].GetTensorData<float>();
        return nchw_tensor_to_image(output_tensor, output_width, output_height, last_error);
    } catch (const Ort::Exception &e) {
        last_error = e.what();
        return godot::Ref<godot::Image>();
    }
#else
    (void)input_image;
    last_error = kOnnxBackendUnavailableMessage;
    return godot::Ref<godot::Image>();
#endif
}

} // namespace artsyn
