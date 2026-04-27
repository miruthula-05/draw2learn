import base64
import os
from pathlib import Path

from config import AI_CHARACTERS_DIR, AI_PROMPTS_DIR


OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")


def _write_prompt_record(kind: str, slug: str, prompt: str) -> None:
    prompt_path = AI_PROMPTS_DIR / f"{kind}_{slug}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")


def build_character_prompt(lesson_title: str, lesson_text: str, object_name: str) -> str:
    return (
        f"Create a clean children's storybook character illustration for '{object_name}'. "
        f"Lesson title: {lesson_title}. "
        "Use a bright classroom-friendly visual style, full body pose, simple readable silhouette, "
        "transparent or plain light background, and keep the design suitable for compositing into a 2D lesson video. "
        f"Story context: {lesson_text[:700]}"
    )


def _try_openai_image(prompt: str, output_path: Path, size: str = "1024x1024") -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False

    try:
        from openai import OpenAI
    except ImportError:
        return False

    try:
        client = OpenAI(api_key=api_key)
        response = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size=size,
        )
        image_b64 = response.data[0].b64_json
        output_path.write_bytes(base64.b64decode(image_b64))
        return True
    except Exception:
        return False


def ensure_ai_character_art(lesson_title: str, lesson_text: str, object_name: str, slug: str) -> str | None:
    output_path = AI_CHARACTERS_DIR / f"{slug}.png"
    if output_path.exists():
        return str(output_path)

    prompt = build_character_prompt(lesson_title, lesson_text, object_name)
    _write_prompt_record("character", slug, prompt)
    if _try_openai_image(prompt, output_path):
        return str(output_path)
    return None


