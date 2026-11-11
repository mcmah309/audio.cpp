#pragma once

#include "engine/community_models/fish_drama3/assets.h"
#include "engine/community_models/fish_drama3/tokenizer_text.h"

#include <limits>
#include <string_view>

namespace engine::models::fish_drama3 {

struct Drama3Take {
    std::string text;
    codecs::FishDacCodes codes;
};

struct Drama3HistoryRetention {
    size_t retained_generated_history_turns = 0;
    bool retained_first_generated_history = false;
    bool retained_latest_generated_history = false;
    size_t explicit_history_turns = 0;
};

struct Drama3GenerationPrompt {
    std::vector<Drama3Frame> frames;
    int64_t generation_budget;
    Drama3HistoryRetention retention;
};

void validate_drama3_text(const std::string & text, const char * location);
void validate_drama3_codes(const codecs::FishDacCodes & codes, int64_t codebook_size);
std::vector<std::string> parse_drama3_text_segments(std::string_view value);
// Explicit references/history are required. Automatic history keeps the first and
// latest turns, dropping the latest then the first only when max_rows requires it.
std::vector<Drama3Frame> build_drama3_prompt(const Drama3Assets & assets,
    const Drama3TextTokenizer & tokenizer, const std::string & text,
    const std::vector<Drama3Take> & references, const std::vector<Drama3Take> & history,
    const std::vector<Drama3Take> & generated_history = {},
    size_t max_rows = std::numeric_limits<size_t>::max(),
    Drama3HistoryRetention * retention = nullptr);

// Reserve output capacity using required conditioning before fitting optional
// generated history. Caller-supplied history and references remain mandatory.
Drama3GenerationPrompt build_drama3_generation_prompt(const Drama3Assets & assets,
    const Drama3TextTokenizer & tokenizer, const std::string & text,
    const std::vector<Drama3Take> & references, const std::vector<Drama3Take> & history,
    const std::vector<Drama3Take> & generated_history, int64_t context_length,
    int64_t requested_frames, bool require_recent_history = false);

}  // namespace engine::models::fish_drama3
