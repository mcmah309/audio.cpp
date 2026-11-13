#pragma once

#include "engine/models/qwen3_asr/assets.h"
#include "engine/framework/runtime/session.h"

#include <memory>

namespace engine::models::qwen3_forced_aligner {

// CPU-only metadata/token inspection; no inference weights or graphs are loaded.
std::unique_ptr<runtime::IVoiceTaskSession> make_planning_session(
    const runtime::SessionOptions & options,
    std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> assets);

}  // namespace engine::models::qwen3_forced_aligner
