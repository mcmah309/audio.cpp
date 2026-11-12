#include "engine/community_models/fish_drama3/session.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace engine::models::fish_drama3 {
namespace {
namespace json = engine::io::json;

// Inspect residency without reading tensor payloads or loading inference weights.
json::Value memory_profile(const Drama3Assets & assets, int device) {
    const auto add = [](uint64_t left, uint64_t right) {
        if (right > std::numeric_limits<uint64_t>::max() - left)
            throw runtime::CapacityError("Drama 3 memory accounting overflow");
        return left + right;
    };
    const auto multiply = [](uint64_t left, uint64_t right) {
        if (right && left > std::numeric_limits<uint64_t>::max() / right)
            throw runtime::CapacityError("Drama 3 memory accounting overflow");
        return left * right;
    };
    uint64_t weights = 0;
    for (const auto & source : {assets.model_weights, assets.codec_weights}) {
        for (const auto & tensor : source->tensors()) {
            const std::string packed_suffix = ".attention.wqkv.weight";
            if (source == assets.codec_weights && tensor.name.size() >= packed_suffix.size() &&
                tensor.name.compare(tensor.name.size() - packed_suffix.size(), packed_suffix.size(), packed_suffix) == 0) {
                const auto prefix = tensor.name.substr(0, tensor.name.size() - packed_suffix.size()) + ".attention.";
                if (source->has_tensor(prefix + "q_proj.weight") && source->has_tensor(prefix + "k_proj.weight") &&
                    source->has_tensor(prefix + "v_proj.weight")) continue;
            }
            if (tensor.shape.empty()) throw runtime::CapacityError("Drama 3 tensor has no shape");
            uint64_t rows = 1;
            for (size_t i = 0; i < tensor.shape.size(); ++i) {
                if (tensor.shape[i] <= 0) throw runtime::CapacityError("Drama 3 tensor has an invalid dimension");
                if (i + 1 < tensor.shape.size()) rows = multiply(rows, tensor.shape[i]);
            }
            // Vectors are uploaded as F32. Count codec QKV split views instead of
            // their unused packed source; this is not an allocator trace.
            const auto type = tensor.shape.size() == 1 ? GGML_TYPE_F32 : assets::ggml_type_for_tensor_dtype(tensor.dtype);
            if (tensor.shape.back() % ggml_blck_size(type))
                throw runtime::CapacityError("Drama 3 tensor row is not block aligned");
            weights = add(weights, multiply(rows, ggml_row_size(type, tensor.shape.back())));
        }
    }
    const auto & b = assets.backbone;
    const auto & t = assets.tower;
    const auto full_layers = b.layers / b.full_attention_interval;
    const auto linear_layers = b.layers - full_layers;
    const auto tower_layers = t.shared_layers + t.residual_layers;
    const uint64_t backbone_row = multiply(multiply(full_layers, b.kv_heads), b.head_dim);
    const uint64_t tower_row = multiply(multiply(tower_layers, t.kv_heads), t.head_dim);
    // Key and value, F16, plus the backbone F16 attention mask.
    const uint64_t bytes_per_row = add(multiply(add(backbone_row, tower_row), 4), 2);
    const auto state_values = multiply(multiply(b.linear_value_head_dim, b.linear_value_head_dim), b.linear_num_value_heads);
    const auto conv_width = add(multiply(multiply(b.linear_key_head_dim, b.linear_num_key_heads), 2),
        multiply(b.linear_value_head_dim, b.linear_num_value_heads));
    const auto state_bytes = multiply(add(multiply(linear_layers + 1, state_values),
        multiply(multiply(linear_layers, conv_width), b.linear_conv_kernel_dim - 1)), 4);
    core::ensure_backends_loaded();
    const auto snapshot = core::query_backend_memory(core::BackendConfig{core::BackendType::Cuda, device, 1});
    if (!snapshot.available) throw runtime::CapacityError("CUDA device memory is unavailable for context selection");
    return json::Value::make_object({
        {"device", json::Value::make_number(device)},
        {"total_bytes", json::Value::make_number(snapshot.total_bytes)},
        {"free_bytes", json::Value::make_number(snapshot.free_bytes)},
        {"weight_storage_bytes", json::Value::make_number(weights)},
        {"fixed_state_bytes", json::Value::make_number(state_bytes)},
        {"cache_bytes_per_row", json::Value::make_number(bytes_per_row)},
        {"maximum_context_rows", json::Value::make_number(262144)}});
}

json::Value parse_conditioning_json(std::string_view text) {
    try {
        return json::parse(text);
    } catch (const std::runtime_error & error) {
        throw runtime::InvalidRequestError(error.what());
    }
}

codecs::FishDacCodes parse_codes(const json::Value & value, int64_t codebook_size) {
    codecs::FishDacCodes codes;
    const auto & books = value.as_array();
    codes.codebooks = static_cast<int64_t>(books.size());
    if (books.size() != 10) throw std::runtime_error("Drama 3 vq_codes needs 10 codebooks");
    codes.frames = static_cast<int64_t>(books.front().as_array().size());
    for (const auto & book : books) {
        if (static_cast<int64_t>(book.as_array().size()) != codes.frames) throw std::runtime_error("Drama 3 vq_codes rows must have equal length");
        for (const auto & code : book.as_array()) {
            const int64_t integer = code.as_i64();
            if (integer < 0 || integer >= codebook_size || code.as_number() != static_cast<double>(integer)) {
                throw std::runtime_error("Drama 3 vq_codes requires integers in the codebook range");
            }
            codes.codes.push_back(static_cast<int32_t>(integer));
        }
    }
    validate_drama3_codes(codes, codebook_size);
    return codes;
}

runtime::AudioBuffer read_reference(const std::string & path) {
    auto wav = engine::audio::read_wav_f32(std::filesystem::path(path));
    return {wav.sample_rate, wav.channels, std::move(wav.samples)};
}

json::Value take_json(const Drama3Take & take) {
    json::Value::Array books;
    for (int64_t book = 0; book < 10; ++book) {
        json::Value::Array frames;
        for (int64_t frame = 0; frame < take.codes.frames; ++frame) {
            frames.push_back(json::Value::make_number(take.codes.codes[book * take.codes.frames + frame]));
        }
        books.push_back(json::Value::make_array(std::move(frames)));
    }
    return json::Value::make_object({{"text", json::Value::make_string(take.text)},
                                    {"vq_codes", json::Value::make_array(std::move(books))}});
}

runtime::VoiceArtifact joint_pcm_artifact(const runtime::AudioBuffer & audio) {
    if (audio.samples.size() > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::runtime_error("Drama 3 joint codec artifact size overflow");
    }
    static_assert(sizeof(float) == sizeof(uint32_t), "PCM comparison requires 32-bit floats");
    std::vector<std::byte> payload(audio.samples.size() * sizeof(float));
    for (size_t index = 0; index < audio.samples.size(); ++index) {
        uint32_t bits;
        std::memcpy(&bits, &audio.samples[index], sizeof(bits));
        for (size_t byte = 0; byte < sizeof(bits); ++byte) {
            payload[index * sizeof(bits) + byte] = static_cast<std::byte>((bits >> (8 * byte)) & 0xffU);
        }
    }
    return runtime::make_voice_artifact(runtime::ArtifactKind::Custom, "drama3_joint_pcm",
        std::move(payload), {{"family", "fish_drama3"}, {"schema", "drama3_pcm_f32le_v1"},
            {"format", "pcm_f32le"}, {"decode_policy", "continuous_causal"},
            {"sample_rate", std::to_string(audio.sample_rate)},
            {"channels", std::to_string(audio.channels)}, {"samples", std::to_string(audio.samples.size())}});
}

codecs::FishDacCodes join_codes(const std::vector<Drama3Take> & takes) {
    codecs::FishDacCodes merged;
    merged.codebooks = 10;
    for (const auto & take : takes) merged.frames += take.codes.frames;
    merged.codes.reserve(static_cast<size_t>(merged.frames * 10));
    for (int64_t book = 0; book < 10; ++book) {
        for (const auto & take : takes) {
            const auto begin = take.codes.codes.begin() + book * take.codes.frames;
            merged.codes.insert(merged.codes.end(), begin, begin + take.codes.frames);
        }
    }
    return merged;
}

}  // namespace

Drama3Session::Drama3Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
                             std::shared_ptr<const Drama3Assets> assets)
    : runtime::RuntimeSessionBase(options), task_(task), assets_(std::move(assets)) {
    if (!assets_ || (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning)) {
        throw std::runtime_error("Fish Drama 3 supports TTS and voice cloning");
    }
    const auto planning = runtime::find_option(options.options, {"fish_drama3.planning_only"});
    if (planning && runtime::parse_bool_option(*planning, "fish_drama3.planning_only")) {
        planning_tokenizer_ = std::make_unique<Drama3TextTokenizer>(assets_->resources);
    } else {
        generator_ = std::make_unique<Drama3Generator>(assets_, execution_context(), options,
            task_.mode == runtime::RunMode::Streaming);
    }
}
Drama3Session::~Drama3Session() = default;
std::string Drama3Session::family() const { return "fish_drama3"; }
runtime::VoiceTaskKind Drama3Session::task_kind() const { return task_.task; }
runtime::RunMode Drama3Session::run_mode() const { return task_.mode; }

void Drama3Session::prepare(const runtime::SessionPreparationRequest & request) {
    defaults_ = {};
    defaults_.text_input = request.text;
    defaults_.voice = request.voice;
    defaults_.options = request.options;
    mark_prepared();
}

Drama3Take Drama3Session::conditioning_take(const json::Value & value, bool require_codes) {
    Drama3Take take;
    std::optional<std::string> audio_path;
    try {
        take.text = json::require_string(value, "text");
        validate_drama3_text(take.text, "conditioning text");
        if (const auto * codes = value.find("vq_codes")) {
            take.codes = parse_codes(*codes, assets_->tower.codebook_size);
        } else if (const auto * audio = value.find("audio_path")) audio_path = audio->as_string();
        else if (const auto * audio = value.find("audio")) audio_path = audio->as_string();
        else throw runtime::InvalidRequestError("Drama 3 conditioning requires vq_codes, audio_path, or audio");
    } catch (const std::runtime_error & error) {
        throw runtime::InvalidRequestError(error.what());
    }
    // Only parsing above translates client errors. I/O and codec failures retain
    // their original classification.
    if (audio_path) {
        if (require_codes || !generator_) {
            throw runtime::InvalidRequestError("Drama 3 prompt inspection requires pre-encoded vq_codes");
        }
        take.codes = generator_->encode_reference(read_reference(*audio_path));
    }
    return take;
}

runtime::TaskResult Drama3Session::run(const runtime::TaskRequest & request) { return synthesize(request, false); }
runtime::TaskResult Drama3Session::generate_stream(const runtime::TaskRequest & request) { return synthesize(request, true); }

runtime::TaskResult Drama3Session::synthesize(const runtime::TaskRequest & request, bool streaming) {
    require_prepared("Fish Drama 3 generation");
    auto effective = defaults_;
    if (request.text_input) effective.text_input = request.text_input;
    if (request.voice) effective.voice = request.voice;
    for (const auto & option : request.options) effective.options[option.first] = option.second;
    effective.input_artifacts = request.input_artifacts;
    if (!effective.text_input) throw runtime::InvalidRequestError("Drama 3 requires input text");
    const auto inspect = runtime::find_option(effective.options, {"inspect_prompt"});
    const bool inspection = !generator_ || (inspect && runtime::parse_bool_option(*inspect, "inspect_prompt"));
    std::optional<std::vector<std::string>> explicit_segments;
    if (const auto segments = runtime::find_option(effective.options, {"text_segments"})) {
        explicit_segments = parse_drama3_text_segments(*segments);
    }
    Drama3Sampling sampling;
    std::optional<uint32_t> seed;
    try {
        sampling.temperature = runtime::parse_finite_float_option(effective.options, {"temperature"}).value_or(1.0F);
        sampling.top_p = runtime::parse_finite_float_option(effective.options, {"top_p"}).value_or(0.9F);
        sampling.top_k = runtime::parse_int_option(effective.options, {"top_k"}).value_or(4097);
        sampling.max_frames = runtime::parse_positive_i64_option(effective.options, {"max_tokens", "max_new_tokens"}, 1024);
        if (const auto require_recent = runtime::find_option(effective.options, {"require_recent_history"})) {
            sampling.require_recent_history = runtime::parse_bool_option(*require_recent, "require_recent_history");
        }
        seed = runtime::parse_u32_option(effective.options, {"seed"});
    } catch (const std::runtime_error & error) {
        throw runtime::InvalidRequestError(error.what());
    } catch (const std::logic_error & error) {
        throw runtime::InvalidRequestError(error.what());
    }
    sampling.seed = seed.value_or(runtime::random_u32_seed());
    std::vector<Drama3Take> references;
    if (effective.voice && effective.voice->speaker && effective.voice->speaker->cached_voice_id &&
        !effective.voice->speaker->audio) {
        throw runtime::InvalidRequestError("Drama 3 requires reference audio or a reference descriptor with VQ codes");
    }
    if (effective.voice && effective.voice->speaker && effective.voice->speaker->audio) {
        if (inspection) throw runtime::InvalidRequestError("Drama 3 prompt inspection requires pre-encoded vq_codes");
        const auto transcript = runtime::find_option(effective.options, {"reference_text"});
        if (!transcript) throw runtime::InvalidRequestError("Drama 3 reference audio requires reference_text");
        references.push_back({*transcript, generator_->encode_reference(*effective.voice->speaker->audio)});
    }
    if (const auto ordered = runtime::find_option(effective.options, {"multi_reference_cond", "references"})) {
        const auto descriptors = parse_conditioning_json(*ordered);
        if (!descriptors.is_array()) throw runtime::InvalidRequestError("Drama 3 references must be a JSON array");
        for (const auto & value : descriptors.as_array()) references.push_back(conditioning_take(value, inspection));
    }
    if (task_.task == runtime::VoiceTaskKind::VoiceCloning && references.empty()) {
        throw runtime::InvalidRequestError("Drama 3 cloning requires at least one reference");
    }
    if (const auto prepare = runtime::find_option(effective.options, {"encode_reference_only"});
        !inspection && prepare && runtime::parse_bool_option(*prepare, "encode_reference_only")) {
        if (references.empty()) throw runtime::InvalidRequestError("Reference preparation requires reference audio or codes");
        if (task_.mode == runtime::RunMode::Streaming) generator_->warm_streaming_codec(references.front().codes);
        runtime::TaskResult result;
        for (size_t index = 0; index < references.size(); ++index) {
            result.output_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::AcousticTokens,
                "drama3_reference_" + std::to_string(index), json::stringify(take_json(references[index])),
                {{"family", "fish_drama3"}, {"schema", "drama3_reference_v1"}}));
        }
        return result;
    }
    std::vector<Drama3Take> history;
    for (const auto & artifact : effective.input_artifacts) {
        if (artifact.kind != runtime::ArtifactKind::AcousticTokens) continue;
        const auto artifact_family = artifact.meta.find("family");
        if (artifact_family != artifact.meta.end() && artifact_family->second != "fish_drama3") {
            throw runtime::InvalidRequestError("Drama 3 continuation cannot consume acoustic tokens from another model family");
        }
        const std::string payload(reinterpret_cast<const char *>(artifact.payload.data()), artifact.payload.size());
        history.push_back(conditioning_take(parse_conditioning_json(payload), inspection));
    }
    const auto takes = runtime::find_option(effective.options, {"previous_takes"});
    const auto takes_file = runtime::find_option(effective.options, {"previous_takes_file"});
    if (takes && takes_file) throw runtime::InvalidRequestError("choose previous_takes or previous_takes_file");
    if (takes || takes_file) {
        const auto descriptors = parse_conditioning_json(takes_file ? engine::io::read_text_file(*takes_file) : *takes);
        const auto append_take = [&](const json::Value & value) {
            if (value.is_string()) {
                if (inspection) throw runtime::InvalidRequestError("Drama 3 prompt inspection requires pre-encoded history takes");
                auto generated = generator_->generate(value.as_string(), references, history, sampling, false);
                history.push_back({value.as_string(), std::move(generated.codes)});
            } else history.push_back(conditioning_take(value, inspection));
        };
        if (descriptors.is_array()) {
            for (const auto & value : descriptors.as_array()) append_take(value);
        } else {
            append_take(descriptors);
        }
    }
    if (inspection) {
        if (explicit_segments && (explicit_segments->size() != 1 ||
            explicit_segments->front() != effective.text_input->text)) {
            throw runtime::InvalidRequestError("Drama 3 prompt inspection requires one explicit input unit");
        }
        if (!planning_tokenizer_) planning_tokenizer_ = std::make_unique<Drama3TextTokenizer>(assets_->resources);
        const auto counted = inspect_drama3_prompt(*assets_, *planning_tokenizer_,
            effective.text_input->text, references, history);
        runtime::TaskResult result;
        result.output_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::Custom,
            "drama3_prompt", json::stringify(json::Value::make_object({
                {"prompt_rows", json::Value::make_number(counted.prompt_rows)},
                {"current_text_rows", json::Value::make_number(counted.current_text_rows)},
                {"history_turn_rows", json::Value::make_number(counted.history_turn_rows)}})),
            {{"family", "fish_drama3"}, {"schema", "drama3_prompt_v1"}}));
        if (const auto device = runtime::parse_i64_option(effective.options, {"inspect_memory_device"})) {
            if (*device < 0 || *device > std::numeric_limits<int>::max())
                throw runtime::InvalidRequestError("CUDA memory device must be a nonnegative index");
            result.output_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::Custom,
                "drama3_memory", json::stringify(memory_profile(*assets_, static_cast<int>(*device))),
                {{"family", "fish_drama3"}, {"schema", "drama3_memory_v1"}}));
        }
        return result;
    }
    const auto skip_empty = runtime::find_option(effective.options, {"skip_empty_chunks"});
    const bool skip_empty_chunks = skip_empty && runtime::parse_bool_option(*skip_empty, "skip_empty_chunks");
    const auto compare_joint = runtime::find_option(effective.options, {"compare_joint_codec"});
    const bool compare_joint_codec = compare_joint && runtime::parse_bool_option(*compare_joint, "compare_joint_codec");
    if (compare_joint_codec && !streaming) {
        throw runtime::InvalidRequestError("Drama 3 joint codec comparison requires streaming mode");
    }
    std::vector<std::string> texts;
    if (explicit_segments) {
        texts = std::move(*explicit_segments);
    } else {
        const int64_t chunk_size = runtime::parse_positive_i64_option(effective.options, {"text_chunk_size"}, 128);
        const auto mode = engine::text::parse_text_chunk_mode_override(effective.options).value_or(engine::text::TextChunkMode::TagAware);
        const auto chunks = runtime::chunk_text_request(effective, chunk_size, mode);
        for (const auto & chunk : chunks) texts.push_back(chunk.text_input->text);
    }
    runtime::TaskResult result;
    runtime::AudioBuffer merged;
    std::vector<Drama3Take> generated_takes;
    json::Value::Array segments;
    const auto started = std::chrono::steady_clock::now();
    bool limited = false;
    for (size_t index = 0; index < texts.size(); ++index) {
        auto chunk_sampling = sampling;
        chunk_sampling.seed += static_cast<uint32_t>(index);
        std::function<void(const runtime::AudioBuffer &)> callback;
        if (streaming && stream_event_sink()) {
            callback = [&](const runtime::AudioBuffer & audio) {
                runtime::StreamEvent event;
                event.audio_output = audio;
                stream_event_sink()(event);
            };
        }
        const auto & text = texts[index];
        auto generated = generator_->generate(text, references, history, chunk_sampling, streaming, callback, index > 0, generated_takes, skip_empty_chunks);
        const auto start_sample = merged.samples.size();
        if (generated.codes.frames != 0) runtime::append_audio_buffer(merged, generated.audio);
        segments.push_back(json::Value::make_object({
            {"text", json::Value::make_string(text)},
            {"start_sample", json::Value::make_number(start_sample)},
            {"end_sample", json::Value::make_number(merged.samples.size())},
            {"frames", json::Value::make_number(generated.codes.frames)},
            {"generation_budget", json::Value::make_number(generated.generation_budget)},
            {"prompt_rows", json::Value::make_number(generated.prompt_rows)},
            {"tower_prefill_block_rows", json::Value::make_number(Drama3Tower::prefill_block_rows)},
            {"retained_generated_history_turns", json::Value::make_number(generated.retention.retained_generated_history_turns)},
            {"retained_first_generated_history", json::Value::make_bool(generated.retention.retained_first_generated_history)},
            {"retained_latest_generated_history", json::Value::make_bool(generated.retention.retained_latest_generated_history)},
            {"explicit_history_turns", json::Value::make_number(generated.retention.explicit_history_turns)},
            {"reached_limit", json::Value::make_bool(generated.reached_limit)}}));
        if (generated.codes.frames == 0) continue;
        limited = limited || generated.reached_limit;
        generated_takes.push_back({text, std::move(generated.codes)});
    }
    if (generated_takes.empty()) {
        throw std::runtime_error("Drama 3 produced no audio: all text chunks were empty");
    }
    Drama3Take take{effective.text_input->text, join_codes(generated_takes)};
    if (compare_joint_codec) {
        auto continuous = generator_->decode_continuously(generated_takes);
        if (continuous.sample_rate != merged.sample_rate || continuous.channels != merged.channels ||
            continuous.samples.size() != merged.samples.size()) {
            throw std::runtime_error("Drama 3 joint codec comparison changed the recording length or format");
        }
        result.output_artifacts.push_back(joint_pcm_artifact(continuous));
    }
    result.output_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::AcousticTokens,
        "drama3_take", json::stringify(take_json(take)), {{"family", "fish_drama3"}, {"schema", "drama3_take_v1"}}));
    result.output_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::Custom,
        "drama3_segments", json::stringify(json::Value::make_array(std::move(segments))),
        {{"family", "fish_drama3"}, {"schema", "drama3_segments_v1"}}));
    result.audio_output = std::move(merged);
    const auto wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    engine::debug::timing_log_scalar("session.wall_ms", wall_ms);
    result.custom_schema_output = runtime::CustomSchemaOutput{"drama3_generation_v1", json::Value::make_object({
        {"frames", json::Value::make_number(take.codes.frames)},
        {"reached_limit", json::Value::make_bool(limited)},
        {"wall_ms", json::Value::make_number(wall_ms)}})};
    return result;
}

}  // namespace engine::models::fish_drama3
