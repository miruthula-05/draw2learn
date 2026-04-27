from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
PROCESSED_DIR = OUTPUTS_DIR / "processed_drawings"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
STATE_DIR = OUTPUTS_DIR / "state"
GENERATED_DIR = OUTPUTS_DIR / "generated_assets"
GENERATED_CHARACTERS_DIR = GENERATED_DIR / "characters"
AUDIO_DIR = OUTPUTS_DIR / "audio"
AI_CACHE_DIR = BASE_DIR / "ai_generated"
AI_CHARACTERS_DIR = AI_CACHE_DIR / "characters"
AI_PROMPTS_DIR = AI_CACHE_DIR / "prompts"
DEFAULT_CHARACTERS_DIR = BASE_DIR / "default_characters"
EXPRESSIONS_DIR = BASE_DIR / "expressions"
BACKGROUNDS_DIR = BASE_DIR / "backgrounds"
CHAPTER_BACKGROUNDS_DIR = BACKGROUNDS_DIR / "chapters"
PROJECT_STATE_FILE = STATE_DIR / "project_state.json"


for directory in (
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
