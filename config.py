import os
import tempfile
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RUNTIME_DIR = Path(tempfile.gettempdir()) / "draw2learn"
RUNTIME_DIR = Path(os.getenv("DRAW2LEARN_RUNTIME_DIR", DEFAULT_RUNTIME_DIR)).resolve()
OUTPUTS_DIR = RUNTIME_DIR / "outputs"
PROCESSED_DIR = OUTPUTS_DIR / "processed_drawings"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
STATE_DIR = OUTPUTS_DIR / "state"
GENERATED_DIR = OUTPUTS_DIR / "generated_assets"
GENERATED_CHARACTERS_DIR = GENERATED_DIR / "characters"
AUDIO_DIR = OUTPUTS_DIR / "audio"
AI_CACHE_DIR = RUNTIME_DIR / "ai_generated"
AI_CHARACTERS_DIR = AI_CACHE_DIR / "characters"
AI_PROMPTS_DIR = AI_CACHE_DIR / "prompts"
DEFAULT_CHARACTERS_DIR = BASE_DIR / "default_characters"
EXPRESSIONS_DIR = BASE_DIR / "expressions"
BACKGROUNDS_DIR = BASE_DIR / "backgrounds"
CHAPTER_BACKGROUNDS_DIR = BACKGROUNDS_DIR / "chapters"
PROJECT_STATE_FILE = STATE_DIR / "project_state.json"
CAPTION_FONT_SIZE_MIN = 44
CAPTION_FONT_SIZE_MAX = 72
DEFAULT_CAPTION_FONT_SIZE = max(
    CAPTION_FONT_SIZE_MIN,
    min(CAPTION_FONT_SIZE_MAX, _env_int("DRAW2LEARN_CAPTION_FONT_SIZE", 57)),
)


for directory in (
    RUNTIME_DIR,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    VIDEOS_DIR,
    STATE_DIR,
    GENERATED_DIR,
    GENERATED_CHARACTERS_DIR,
    AUDIO_DIR,
    AI_CACHE_DIR,
    AI_CHARACTERS_DIR,
    AI_PROMPTS_DIR,
    DEFAULT_CHARACTERS_DIR,
    BACKGROUNDS_DIR,
    CHAPTER_BACKGROUNDS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
