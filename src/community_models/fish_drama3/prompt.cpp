#include "engine/community_models/fish_drama3/prompt.h"

#include "engine/framework/runtime/errors.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"

#include <utf8proc.h>

#include <algorithm>
#include <array>
#include <regex>
#include <stdexcept>

namespace engine::models::fish_drama3 {
namespace {

const std::regex speaker_pattern(R"(<\|speaker:([0-9]+)\|>)");

// Python's prompt builder uses Unicode White_Space, rather than ASCII trim.
size_t leading_space_bytes(const std::string & text) {
    size_t offset = 0;
    while (offset < text.size()) {
        const auto first = static_cast<unsigned char>(text[offset]);
        uint32_t code = first;
        size_t width = 1;
        if ((first & 0xe0) == 0xc0) { code = first & 0x1f; width = 2; }
        else if ((first & 0xf0) == 0xe0) { code = first & 0x0f; width = 3; }
        else if ((first & 0xf8) == 0xf0) { code = first & 7; width = 4; }
        if (offset + width > text.size()) break;
        for (size_t part = 1; part < width; ++part) code = (code << 6) | (static_cast<unsigned char>(text[offset + part]) & 0x3f);
        const bool whitespace = (code >= 9 && code <= 13) || code == 0x20 || code == 0x85 || code == 0xa0 ||
            code == 0x1680 || (code >= 0x2000 && code <= 0x200a) || code == 0x2028 || code == 0x2029 ||
            code == 0x202f || code == 0x205f || code == 0x3000;
        if (!whitespace) break;
        offset += width;
    }
    return offset;
}

std::string tagged(const std::string & text, bool reference) {
    const size_t leading = leading_space_bytes(text);
    const std::string body = text.substr(leading);
    std::smatch match;
    if (!std::regex_search(body, match, speaker_pattern, std::regex_constants::match_continuous)) {
        return "<|speaker:0|> " + text;
    }
    if (!reference) return text;
    return match.str() + text.substr(0, leading) + body.substr(match.length());
}

}  // namespace

void validate_drama3_text(const std::string & text, const char * location) {
    size_t position = 0;
    while (position < text.size()) {
        utf8proc_int32_t code;
        const auto width = utf8proc_iterate(
            reinterpret_cast<const utf8proc_uint8_t *>(text.data() + position),
            static_cast<utf8proc_ssize_t>(text.size() - position), &code);
        if (width < 0 || code == 0) {
            throw runtime::InvalidRequestError(std::string(location) + ": invalid UTF-8 text");
        }
        position += static_cast<size_t>(width);
    }
    if (leading_space_bytes(text) == text.size()) throw runtime::InvalidRequestError(std::string(location) + ": text is empty");
    for (const auto * token : {"<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"}) {
        if (text.find(token) != std::string::npos) throw runtime::InvalidRequestError(std::string(location) + ": structural tokens are reserved for the prompt builder");
    }
    size_t offset = 0;
    while ((offset = text.find("<|speaker", offset)) != std::string::npos) {
        const auto body = text.substr(offset);
        std::smatch match;
        if (!std::regex_search(body, match, speaker_pattern, std::regex_constants::match_continuous) ||
            match[1].length() > 2 || (match[1].length() > 1 && match[1].str()[0] == '0') ||
            std::stoi(match[1].str()) > 15) {
            throw runtime::InvalidRequestError(std::string(location) + ": speaker tags require N in 0..15 without leading zeroes");
        }
        offset += match.length();
    }
}

std::vector<std::string> parse_drama3_text_segments(std::string_view value) {
    try {
        const auto parsed = io::json::parse(value);
        if (!parsed.is_array() || parsed.as_array().empty()) {
            throw runtime::InvalidRequestError("Drama 3 text_segments requires a nonempty JSON array of strings");
        }
        std::vector<std::string> segments;
        for (const auto & item : parsed.as_array()) {
            if (!item.is_string()) {
                throw runtime::InvalidRequestError("Drama 3 text_segments entries must be strings");
            }
            validate_drama3_text(item.as_string(), "text_segments");
            segments.push_back(item.as_string());
        }
        return segments;
    } catch (const std::runtime_error & error) {
        throw runtime::InvalidRequestError(error.what());
    }
}

void validate_drama3_codes(const codecs::FishDacCodes & codes, int64_t size) {
    if (codes.codebooks != 10 || codes.frames <= 0 ||
        codes.codes.size() != static_cast<size_t>(codes.frames * codes.codebooks)) {
        throw std::runtime_error("Drama 3 codes require a nonempty [10, frames] grid");
    }
    for (const auto code : codes.codes) {
        if (code < 0 || code >= size) throw std::runtime_error("Drama 3 audio code is outside the codebook");
    }
}

std::vector<Drama3Frame> build_drama3_prompt(const Drama3Assets & assets,
    const Drama3TextTokenizer & tokenizer, const std::string & text,
    const std::vector<Drama3Take> & references, const std::vector<Drama3Take> & history,
    const std::vector<Drama3Take> & generated_history, size_t max_rows,
    Drama3HistoryRetention * retention) {
    std::vector<Drama3Frame> frames;
    const auto append_text = [&](const std::string & segment) {
        for (const auto token : tokenizer.encode(segment)) {
            Drama3Frame row{};
            row[0] = token;
            frames.push_back(row);
        }
    };
    const auto append_codes = [&](const codecs::FishDacCodes & codes) {
        validate_drama3_codes(codes, assets.tower.codebook_size);
        for (int64_t frame = 0; frame < codes.frames; ++frame) {
            Drama3Frame row{};
            for (int64_t book = 0; book < 10; ++book) row[1 + book] = codes.codes[book * codes.frames + frame];
            row[0] = row[1] + assets.tower.semantic_begin;
            frames.push_back(row);
        }
    };
    append_text("<|im_start|>system\n");
    append_text(references.empty() ? "convert the provided text to speech" : "convert the provided text to speech reference to the following:");
    for (size_t index = 0; index < references.size(); ++index) {
        validate_drama3_text(references[index].text, "reference");
        append_text("\n\n[Reference " + std::to_string(index + 1) + "]\nText: " + tagged(references[index].text, true) + "\nSpeech: ");
        append_codes(references[index].codes);
    }
    append_text("<|im_end|>\n");
    const auto user_turn = [&](const std::string & value) {
        validate_drama3_text(value, "input");
        append_text("<|im_start|>user\n");
        append_text(tagged(value, false));
        append_text("<|im_end|>\n");
        append_text("<|im_start|>assistant\n<|voice|>");
    };
    const auto append_take = [&](const Drama3Take & take) {
        user_turn(take.text);
        append_codes(take.codes);
        append_text("<|im_end|>\n");
    };
    for (const auto & take : history) append_take(take);
    const auto first_begin = frames.size();
    if (!generated_history.empty()) append_take(generated_history.front());
    const auto latest_begin = frames.size();
    if (generated_history.size() > 1) append_take(generated_history.back());
    auto input_begin = frames.size();
    user_turn(text);
    size_t retained = std::min<size_t>(generated_history.size(), 2);
    bool retained_first = !generated_history.empty();
    bool retained_latest = !generated_history.empty();
    if (frames.size() > max_rows && generated_history.size() > 1) {
        frames.erase(frames.begin() + latest_begin, frames.begin() + input_begin);
        input_begin = latest_begin;
        --retained;
        retained_latest = false;
    }
    if (frames.size() > max_rows && !generated_history.empty()) {
        frames.erase(frames.begin() + first_begin, frames.begin() + input_begin);
        retained = 0;
        retained_first = retained_latest = false;
    }
    if (frames.size() > max_rows) {
        throw runtime::CapacityError("Drama 3 required prompt leaves no room to generate within fish_drama3.context_length");
    }
    debug::trace_log_scalar("drama3.history.generated_turns", retained);
    if (retention != nullptr) {
        *retention = {retained, retained_first, retained_latest, history.size()};
    }
    return frames;
}

Drama3PromptInspection inspect_drama3_prompt(const Drama3Assets & assets,
    const Drama3TextTokenizer & tokenizer, const std::string & text,
    const std::vector<Drama3Take> & references, const std::vector<Drama3Take> & history) {
    const auto prompt = build_drama3_prompt(assets, tokenizer, text, references, history);
    auto extended_history = history;
    extended_history.push_back({text, {std::vector<int32_t>(10, 0), 10, 1}});
    const auto extended = build_drama3_prompt(assets, tokenizer, text, references, extended_history);
    return {prompt.size(), tokenizer.encode(tagged(text, false)).size(),
        extended.size() - prompt.size() - 1};
}

Drama3GenerationPrompt build_drama3_generation_prompt(const Drama3Assets & assets,
    const Drama3TextTokenizer & tokenizer, const std::string & text,
    const std::vector<Drama3Take> & references, const std::vector<Drama3Take> & history,
    const std::vector<Drama3Take> & generated_history, int64_t context_length,
    int64_t requested_frames, bool require_recent_history) {
    if (context_length <= 1 || requested_frames <= 0) {
        throw runtime::CapacityError("Drama 3 requires context capacity for both input and generation");
    }
    Drama3HistoryRetention retention;
    auto required = build_drama3_prompt(assets, tokenizer, text, references, history, {},
        static_cast<size_t>(context_length - 1), &retention);
    const auto budget = std::min(requested_frames,
        context_length - static_cast<int64_t>(required.size()));
    if (generated_history.empty()) return {std::move(required), budget, retention};
    auto frames = build_drama3_prompt(assets, tokenizer, text, references, history,
        generated_history, static_cast<size_t>(context_length - budget), &retention);
    if (require_recent_history && !retention.retained_latest_generated_history) {
        throw runtime::CapacityError("Drama 3 latest generated history does not fit together with the requested generation budget");
    }
    return {std::move(frames), budget, retention};
}

}  // namespace engine::models::fish_drama3
