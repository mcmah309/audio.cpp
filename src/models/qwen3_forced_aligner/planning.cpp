#include "engine/models/qwen3_forced_aligner/planning.h"

#include "engine/framework/runtime/artifacts.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/qwen3_forced_aligner/processor.h"
#include "engine/models/qwen3_asr/tokenizer_text.h"

#include <algorithm>
#include <limits>

namespace engine::models::qwen3_forced_aligner {
namespace {
namespace json = engine::io::json;

uint64_t add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a)
        throw runtime::CapacityError("Qwen3 alignment memory accounting overflow");
    return a + b;
}

uint64_t multiply(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw runtime::CapacityError("Qwen3 alignment memory accounting overflow");
    return a * b;
}

uint64_t dimension(int64_t value) {
    if (value <= 0) throw runtime::CapacityError("Invalid Qwen3 alignment model dimension");
    return static_cast<uint64_t>(value);
}

uint64_t weight_bytes(const engine::models::qwen3_asr::Qwen3ASRAssets & assets) {
    uint64_t bytes = 0;
    for (const auto & tensor : assets.model_weights->tensors()) {
        if (tensor.shape.empty()) throw runtime::CapacityError("Alignment tensor has no shape");
        uint64_t rows = 1;
        for (size_t i = 0; i + 1 < tensor.shape.size(); ++i)
            rows = multiply(rows, dimension(tensor.shape[i]));
        const auto width = dimension(tensor.shape.back());
        // Biases/norms are uploaded as F32. Bound nonmatrix convolution weights
        // by F32 as graph execution can materialize converted copies.
        const auto type = tensor.shape.size() == 2
            ? assets::ggml_type_for_tensor_dtype(tensor.dtype) : GGML_TYPE_F32;
        if (width % ggml_blck_size(type)) throw runtime::CapacityError("Alignment tensor is not block aligned");
        const auto storage = multiply(rows, ggml_row_size(type, width));
        bytes = add(bytes, multiply(add(storage, 255) / 256, 256));
    }
    // The encoder creates a F32 sinusoidal position table rather than using a
    // checkpoint tensor. Source-only/tied tensors above are conservative extras.
    const auto & e = assets.config.audio_encoder;
    return add(bytes, multiply(4, multiply(dimension(e.max_source_positions), dimension(e.d_model))));
}

class PlanningSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    PlanningSession(const runtime::SessionOptions & options,
        std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> assets)
        : RuntimeSessionBase(options), assets_(std::move(assets)), tokenizer_(assets_), processor_(*assets_, tokenizer_) {
        if (options.backend.type != core::BackendType::Cpu)
            throw runtime::InvalidRequestError("Alignment planning_only requires CPU backend");
    }

    std::string family() const override { return "qwen3_forced_aligner"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Alignment; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Qwen3 alignment inspection");
        if (!request.text_input || request.text_input->text.empty() || request.text_input->language.empty())
            throw runtime::InvalidRequestError("Alignment inspection requires text and language");
        const auto samples = runtime::parse_i64_option(request.options, {"inspect_audio_samples"}).value_or(0);
        const auto rate = runtime::parse_i64_option(request.options, {"inspect_sample_rate"}).value_or(0);
        const auto device = runtime::parse_i64_option(request.options, {"inspect_memory_device"}).value_or(-1);
        if (samples <= 0 || rate <= 0 || rate > std::numeric_limits<int>::max() || device < 0 ||
            device > std::numeric_limits<int>::max() || static_cast<uint64_t>(samples) > multiply(rate, 300))
            throw runtime::InvalidRequestError("Alignment inspection requires a device and audio bounds within five minutes");
        const auto & config = assets_->config;
        const auto & encoder = config.audio_encoder;
        const auto & decoder = config.text_decoder;
        // Round resampling upward and retain two extra feature frames to cover
        // frontend endpoint rounding. Minimum frontend input is half a second.
        const auto resampled = std::max<uint64_t>(8000,
            add(multiply(samples, dimension(config.frontend.sample_rate)), rate - 1) / rate);
        const auto frames = add(resampled / dimension(config.frontend.hop_length), 2);
        const auto chunk = multiply(2, dimension(encoder.n_window));
        const auto padded = multiply(add(frames, chunk - 1) / chunk, chunk);
        const auto tokens = modules::qwen3_audio_encoder_token_count(frames);
        if (tokens > encoder.max_source_positions)
            throw runtime::CapacityError("Alignment audio bound exceeds encoder positions");
        const auto prompt = processor_.build_prompt(request.text_input->text, request.text_input->language, tokens);
        const uint64_t rows = prompt.prompt.input_ids.size();
        if (rows > dimension(decoder.max_position_embeddings))
            throw runtime::CapacityError("Alignment prompt bound exceeds decoder positions");
        // Live-tensor envelopes, not allocator traces: padded convolution and
        // im2col/cast/activation buffers; transformer residual/QKV/MLP copies;
        // F32 attention scores, softmax and masks; output projections/logits.
        // Encoder and classification graphs remain resident together, but graph
        // replacement releases the old graph before constructing its successor.
        const auto conv = multiply(padded, add(multiply(8, multiply(dimension(encoder.num_mel_bins),
            dimension(encoder.downsample_hidden_size))), multiply(8, dimension(encoder.num_mel_bins))));
        const auto encoder_row = add(add(multiply(12, dimension(encoder.d_model)),
            multiply(8, dimension(encoder.encoder_ffn_dim))), multiply(4, dimension(encoder.output_dim)));
        const auto encoder_workspace = add(conv, add(multiply(4, multiply(tokens, encoder_row)),
            multiply(8, multiply(multiply(tokens, tokens), add(dimension(encoder.encoder_attention_heads), 1)))));
        const auto query = multiply(dimension(decoder.num_attention_heads), dimension(decoder.head_dim));
        const auto kv = multiply(dimension(decoder.num_key_value_heads), dimension(decoder.head_dim));
        const auto decoder_row = add(add(add(multiply(12, dimension(decoder.hidden_size)), multiply(8, query)),
            add(multiply(8, kv), multiply(8, dimension(decoder.intermediate_size)))), dimension(decoder.output_size));
        const auto classification_workspace = add(multiply(4, multiply(rows, decoder_row)),
            multiply(8, multiply(multiply(rows, rows), add(dimension(decoder.num_attention_heads), 1))));
        const auto weights = weight_bytes(*assets_);
        const auto workspace = add(encoder_workspace, classification_workspace);
        const auto profile = json::Value::make_object({
            {"device", json::Value::make_number(device)},
            {"maximum_audio_samples", json::Value::make_number(samples)},
            {"sample_rate", json::Value::make_number(rate)},
            {"prompt_rows", json::Value::make_number(rows)},
            {"weight_storage_bytes", json::Value::make_number(weights)},
            {"encoder_workspace_bytes", json::Value::make_number(encoder_workspace)},
            {"classification_workspace_bytes", json::Value::make_number(classification_workspace)},
            {"total_bytes", json::Value::make_number(add(weights, workspace))}});
        runtime::TaskResult result;
        result.output_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::Custom,
            "qwen3_alignment_memory", json::stringify(profile), {{"family", family()}}));
        return result;
    }
private:
    std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> assets_;
    engine::models::qwen3_asr::Qwen3ASRTextTokenizer tokenizer_;
    Qwen3ForcedAlignProcessor processor_;
};

}  // namespace

std::unique_ptr<runtime::IVoiceTaskSession> make_planning_session(
    const runtime::SessionOptions & options,
    std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> assets) {
    return std::make_unique<PlanningSession>(options, std::move(assets));
}

}  // namespace engine::models::qwen3_forced_aligner
