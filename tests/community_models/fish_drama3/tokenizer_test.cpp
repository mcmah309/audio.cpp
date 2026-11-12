#include "engine/community_models/fish_drama3/tokenizer_text.h"
#include "engine/community_models/fish_drama3/prompt.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/errors.h"

#include <iostream>
#include <stdexcept>

namespace drama = engine::models::fish_drama3;
namespace json = engine::io::json;

int main(int argc, char ** argv) {
    try {
        if (argc != 3 && !(argc == 4 && std::string(argv[3]) == "--log"))
            throw std::runtime_error("usage: tokenizer_test SIDECAR_DIRECTORY_OR_GGUF VECTORS_JSON [--log]");
        if (argc == 4) engine::debug::configure_logging({true, std::nullopt});
        engine::assets::ResourceBundle resources;
        if (std::filesystem::is_regular_file(argv[1])) {
            resources = engine::model_spec::load_resource_bundle_for_family(argv[1], "fish_drama3");
        } else {
            resources = engine::assets::ResourceBundle(argv[1]);
            resources.add_model_files({{"tokenizer_json", "tokenizer.json"},
                                       {"tokenizer_config", "tokenizer_config.json"}});
        }
        drama::Drama3TextTokenizer tokenizer(resources);
        const auto vectors = json::parse_file(argv[2]);
        int passed = 0;
        for (const auto & item : vectors.require("cases").as_array()) {
            auto text = item.require("text").as_string();
            if (const auto * bytes = item.find("utf8")) {
                text.clear();
                for (const auto & byte : bytes->as_array()) text.push_back(static_cast<char>(byte.as_i64()));
            }
            const auto tokens = tokenizer.encode(text);
            const auto & expected = item.require("tokens").as_array();
            bool equal = tokens.size() == expected.size();
            for (size_t i = 0; equal && i < tokens.size(); ++i) equal = tokens[i] == expected[i].as_i64();
            if (!equal) throw std::runtime_error("token mismatch: " + item.require("id").as_string());
            ++passed;
        }
        // Compare actual prompt rows, including fixed references and explicit takes.
        drama::Drama3Assets assets;
        const drama::Drama3Take reference{"A reference.", {std::vector<int32_t>(20, 1), 10, 2}};
        const drama::Drama3Take explicit_take{"Caller history.", {std::vector<int32_t>(30, 2), 10, 3}};
        const drama::Drama3Take first_take{"First chunk.", {std::vector<int32_t>(40, 3), 10, 4}};
        const drama::Drama3Take middle_take{"Middle chunk.", {std::vector<int32_t>(50, 4), 10, 5}};
        const drama::Drama3Take latest_take{"Latest chunk.", {std::vector<int32_t>(60, 5), 10, 6}};
        const std::string next_text = "The next chunk.";
        const auto mandatory = drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take});
        const auto anchor = drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take, first_take});
        const auto full = drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take, first_take, latest_take});
        const auto counted = drama::inspect_drama3_prompt(assets, tokenizer, next_text,
            {reference}, {explicit_take});
        if (counted.prompt_rows != mandatory.size())
            throw std::runtime_error("inspection must count the exact generation prompt");
        const drama::Drama3Take appended_take{next_text, {std::vector<int32_t>(170, 1), 10, 17}};
        const auto appended = drama::build_drama3_prompt(assets, tokenizer, next_text,
            {reference}, {explicit_take, appended_take});
        if (appended.size() != counted.prompt_rows + counted.history_turn_rows + 17)
            throw std::runtime_error("completed-turn accounting must add exact text/template rows and acoustic frames");
        const auto plain = drama::inspect_drama3_prompt(assets, tokenizer, next_text, {}, {});
        const auto bare = drama::build_drama3_prompt(assets, tokenizer, next_text, {}, {});
        if (plain.prompt_rows != bare.size() || plain.history_turn_rows != counted.history_turn_rows)
            throw std::runtime_error("completed-turn size must be independent of reference and prior history");
        if (counted.current_text_rows != tokenizer.encode("<|speaker:0|> " + next_text).size())
            throw std::runtime_error("current text accounting must include the actual automatic speaker tag");
        const auto tagged = drama::inspect_drama3_prompt(assets, tokenizer, "<|speaker:1|> ¡Árbol!", {}, {});
        if (tagged.current_text_rows != tokenizer.encode("<|speaker:1|> ¡Árbol!").size())
            throw std::runtime_error("inspection must preserve explicit speaker tags and Unicode tokenization");
        if (drama::inspect_drama3_prompt(assets, tokenizer, "a\xcc\x81", {}, {}).prompt_rows !=
            drama::inspect_drama3_prompt(assets, tokenizer, "á", {}, {}).prompt_rows)
            throw std::runtime_error("inspection must apply the same Unicode normalization as generation");
        const std::vector<drama::Drama3Take> generated{first_take, middle_take, latest_take};
        if (drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, generated) != full)
            throw std::runtime_error("automatic history must retain first and latest, excluding middle");
        if (drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, {first_take}) != anchor)
            throw std::runtime_error("the first generated turn must not be duplicated");
        if (drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, generated, full.size()) != full)
            throw std::runtime_error("an exactly fitting full history must remain unchanged");
        if (drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, generated, anchor.size()) != anchor)
            throw std::runtime_error("drop the latest automatic turn before the anchor");
        if (drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, generated, mandatory.size()) != mandatory)
            throw std::runtime_error("drop only automatic history when necessary");
        bool overflow_rejected = false;
        try { drama::build_drama3_prompt(assets, tokenizer, next_text, {reference}, {explicit_take}, generated, mandatory.size() - 1); }
        catch (const engine::runtime::CapacityError &) { overflow_rejected = true; }
        if (!overflow_rejected) throw std::runtime_error("required references, caller history and input must not be dropped");
        bool invalid_rejected = false;
        try { drama::build_drama3_prompt(assets, tokenizer, "<|im_start|>", {reference}, {explicit_take}, generated, 1); }
        catch (const engine::runtime::InvalidRequestError &) { invalid_rejected = true; }
        if (!invalid_rejected) throw std::runtime_error("prompt fitting must not hide malformed input");
        const auto context = static_cast<int64_t>(full.size() + 100);
        const auto requested = context - static_cast<int64_t>(anchor.size());
        const auto fitted = drama::build_drama3_generation_prompt(assets, tokenizer, next_text,
            {reference}, {explicit_take}, generated, context, requested);
        if (fitted.frames != anchor || fitted.generation_budget != requested)
            throw std::runtime_error("automatic latest history must yield to the requested output budget");
        if (fitted.retention.retained_generated_history_turns != 1 ||
            !fitted.retention.retained_first_generated_history ||
            fitted.retention.retained_latest_generated_history || fitted.retention.explicit_history_turns != 1)
            throw std::runtime_error("retention metadata must describe the fitted first-only history");
        bool recent_history_rejected = false;
        try {
            drama::build_drama3_generation_prompt(assets, tokenizer, next_text,
                {reference}, {explicit_take}, generated, context, requested, true);
        } catch (const engine::runtime::CapacityError &) { recent_history_rejected = true; }
        if (!recent_history_rejected)
            throw std::runtime_error("requiring recent history must reject a prompt that drops the latest take");
        const auto full_fit = drama::build_drama3_generation_prompt(assets, tokenizer, next_text,
            {reference}, {explicit_take}, generated, context, 100, true);
        if (full_fit.frames != full || full_fit.retention.retained_generated_history_turns != 2 ||
            !full_fit.retention.retained_first_generated_history || !full_fit.retention.retained_latest_generated_history)
            throw std::runtime_error("retention metadata must report both distinct retained automatic turns");
        const auto one_prior = drama::build_drama3_generation_prompt(assets, tokenizer, next_text,
            {reference}, {explicit_take}, {first_take}, context, requested, true);
        if (one_prior.frames != anchor || one_prior.retention.retained_generated_history_turns != 1 ||
            !one_prior.retention.retained_first_generated_history || !one_prior.retention.retained_latest_generated_history)
            throw std::runtime_error("one prior generated take is both first and latest");
        const auto first_segment = drama::build_drama3_generation_prompt(assets, tokenizer, next_text,
            {reference}, {explicit_take}, {}, context, requested, true);
        if (first_segment.frames != mandatory || first_segment.retention.retained_generated_history_turns != 0 ||
            first_segment.retention.retained_first_generated_history || first_segment.retention.retained_latest_generated_history ||
            first_segment.retention.explicit_history_turns != 1)
            throw std::runtime_error("a first segment has no automatic history requirement");
        const auto all_output = context - static_cast<int64_t>(mandatory.size());
        const auto no_auto = drama::build_drama3_generation_prompt(assets, tokenizer, next_text,
            {reference}, {explicit_take}, generated, context, all_output);
        if (no_auto.frames != mandatory || no_auto.generation_budget != all_output)
            throw std::runtime_error("all automatic history must yield before reducing requested output");
        if (no_auto.retention.retained_generated_history_turns != 0 ||
            no_auto.retention.retained_first_generated_history || no_auto.retention.retained_latest_generated_history)
            throw std::runtime_error("retention metadata must report that all automatic history was removed");
        const auto capped = drama::build_drama3_generation_prompt(assets, tokenizer, next_text,
            {reference}, {explicit_take}, generated, context, context);
        if (capped.frames != mandatory || capped.generation_budget != all_output)
            throw std::runtime_error("only mandatory prompt rows may reduce requested output capacity");
        bool required_overflow = false;
        try {
            drama::build_drama3_generation_prompt(assets, tokenizer, next_text,
                {reference}, {explicit_take}, generated, mandatory.size(), 1);
        } catch (const engine::runtime::CapacityError &) { required_overflow = true; }
        if (!required_overflow) throw std::runtime_error("mandatory prompt must leave at least one generation row");
        const std::vector<std::string> expected_segments{"<|speaker:1|> One paragraph.", " Otro párrafo. "};
        if (drama::parse_drama3_text_segments("[\"<|speaker:1|> One paragraph.\",\" Otro párrafo. \"]") != expected_segments)
            throw std::runtime_error("explicit segments must preserve order and exact text");
        const std::vector<std::string> invalid_segments = {
            "[]", "{}", "null", "[1]", "[\"valid\",null]", "[\"\"]", "[\" \\n\\u2003\"]",
            "[\"valid\",\"<|im_start|>\"]", "[\"<|speaker:16|> invalid\"]", "[\"unfinished]",
            std::string("[\"") + std::string("\xc0\xaf", 2) + "\"]",
        };
        for (const auto & value : invalid_segments) {
            bool rejected = false;
            try { drama::parse_drama3_text_segments(value); }
            catch (const engine::runtime::InvalidRequestError &) { rejected = true; }
            if (!rejected) throw std::runtime_error("malformed explicit segments accepted");
        }
        const std::vector<std::string> invalid = {
            std::string("\xc0\xaf", 2), std::string("\xed\xa0\x80", 3),
            std::string("\xf4\x90\x80\x80", 4), std::string("\xe2\x82", 2),
            std::string("\x80", 1),
        };
        for (const auto & text : invalid) {
            bool rejected = false;
            try { tokenizer.encode(text); }
            catch (const engine::runtime::InvalidRequestError &) { rejected = true; }
            if (!rejected) throw std::runtime_error("malformed UTF-8 accepted");
        }
        // A rejected input must not poison the persistent tokenizer.
        const auto & first = vectors.require("cases").as_array().front();
        const auto recovered = tokenizer.encode(first.require("text").as_string());
        const auto & expected = first.require("tokens").as_array();
        bool recovered_exact = recovered.size() == expected.size();
        for (size_t i = 0; recovered_exact && i < recovered.size(); ++i) {
            recovered_exact = recovered[i] == expected[i].as_i64();
        }
        if (!recovered_exact) {
            throw std::runtime_error("tokenizer recovery failed");
        }
        std::cout << "reference_vectors_exact=" << passed
                  << " malformed_utf8_rejected=" << invalid.size() << " recovery=true\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
