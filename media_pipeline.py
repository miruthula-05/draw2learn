from __future__ import annotations

import io
import math
from datetime import datetime
from pathlib import Path

import numpy as np
try:
    from moviepy import ImageClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.compositing.concatenate import concatenate_videoclips
except ImportError:
    from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont
from rembg import remove

from ai_generation import ensure_ai_character_art
from config import CHAPTER_BACKGROUNDS_DIR, EXPRESSIONS_DIR, GENERATED_CHARACTERS_DIR, PROCESSED_DIR, VIDEOS_DIR
from lesson_parser import should_apply_expression
from narration import generate_narration_audio

FRAME_SIZE = (1280, 720)
DEFAULT_OVERLAY = {"x": 0, "y": 0, "size": 22}
GROUND_Y = 610
SUPPORTED_BACKGROUND_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
CHARACTER_HEIGHTS = {
    "mother": 360,
    "father": 360,
    "teacher": 360,
    "badal": 310,
    "boy": 310,
    "girl": 300,
    "moti": 220,
    "dog": 220,
    "puppy": 220,
    "rope": 220,
    "neighbours": 300,
    "neighbor": 300,
    "neighbour": 300,
}


def available_expression_names(expressions_dir: Path) -> list[str]:
    return sorted(path.stem for path in expressions_dir.glob("*.png")) or ["happy"]


def slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in cleaned.split("_") if part) or "item"


def _seed(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(value)) % 1000003


def _load_background_from_assets(lesson_title: str, scene_index: int) -> Image.Image | None:
    chapter_dir = CHAPTER_BACKGROUNDS_DIR / slugify(lesson_title)
    if not chapter_dir.exists():
        return None
    candidates = []
    for pattern in (f"{scene_index + 1:02d}.*", f"{scene_index + 1}.*"):
        candidates.extend(path for path in chapter_dir.glob(pattern) if path.suffix.lower() in SUPPORTED_BACKGROUND_SUFFIXES)
    if not candidates:
        return None
    selected = sorted(candidates, key=lambda path: (len(path.stem), path.name.lower()))[0]
    return Image.open(selected).convert("RGBA")


def _load_expression(expression_name: str) -> Image.Image | None:
    fallback_names = {
        "worried": "sad",
        "afraid": "surprised",
        "scared": "surprised",
    }
    expression_path = EXPRESSIONS_DIR / f"{expression_name}.png"
    if not expression_path.exists() and expression_name in fallback_names:
        expression_path = EXPRESSIONS_DIR / f"{fallback_names[expression_name]}.png"
    if not expression_path.exists():
        expression_path = EXPRESSIONS_DIR / "happy.png"
        if not expression_path.exists():
            return None
    return Image.open(expression_path).convert("RGBA")


def _detect_head_positions(base_img: Image.Image) -> list[tuple[int, int, int]]:
    alpha = np.array(base_img.getchannel("A"))
    if alpha.size == 0:
        return []
    mask = alpha > 20
    rows, cols = mask.shape
    top_band = mask[: max(1, int(rows * 0.48)), :]
    column_strength = top_band.sum(axis=0)
    threshold = max(6, int(top_band.shape[0] * 0.08))
    peaks = []
    start = None
    for x, value in enumerate(column_strength):
        if value >= threshold and start is None:
            start = x
        elif value < threshold and start is not None:
            if x - start >= max(18, cols // 14):
                peaks.append((start, x - 1))
            start = None
    if start is not None and cols - start >= max(18, cols // 14):
        peaks.append((start, cols - 1))

    heads = []
    for left, right in peaks[:6]:
        width = right - left + 1
        center_x = (left + right) // 2
        band = top_band[:, left : right + 1]
        ys = np.where(band.any(axis=1))[0]
        center_y = int(ys[0] + max(12, len(ys) * 0.3)) if len(ys) else int(rows * 0.22)
        heads.append((center_x, center_y, width))
    if len(heads) >= 2:
        return heads

    total_width = np.where(column_strength >= threshold)[0]
    if len(total_width) == 0:
        return []
    left = int(total_width[0])
    right = int(total_width[-1])
    span = max(1, right - left)
    estimated = 3 if span > cols * 0.52 else (2 if span > cols * 0.3 else 1)
    spacing = span / max(estimated, 1)
    fallback = []
    for index in range(estimated):
        center_x = int(left + spacing * (index + 0.5))
        fallback.append((center_x, int(rows * 0.24), int(spacing * 0.72)))
    return fallback


def _overlay_expression(base_img: Image.Image, expression_name: str, overlay_position: dict, object_name: str = "") -> Image.Image:
    frame = base_img.convert("RGBA").copy()
    expression = _load_expression(expression_name)
    if expression is None:
        return frame
    width, height = frame.size
    lowered_name = object_name.lower()
    if any(token in lowered_name for token in ("neigh", "neighbor", "students", "friends")):
        head_positions = _detect_head_positions(frame)
        if head_positions:
            for center_x, center_y, size_hint in head_positions:
                new_size = max(22, min(72, int(size_hint * 0.68)))
                face = expression.resize((new_size, new_size), Image.Resampling.LANCZOS)
                overlay_x = max(0, min(center_x - new_size // 2, width - new_size))
                overlay_y = max(0, min(center_y - new_size // 2, height - new_size))
                frame.alpha_composite(face, dest=(overlay_x, overlay_y))
            return frame
    new_size = max(16, int(width * (overlay_position.get("size", 22) / 100)))
    face = expression.resize((new_size, new_size), Image.Resampling.LANCZOS)
    x_offset = overlay_position.get("x", 0)
    y_offset = overlay_position.get("y", 0)
    overlay_x = max(0, min((width // 2 - new_size // 2) + x_offset, width - new_size))
    overlay_y = max(0, min((height // 3 - new_size // 2) + y_offset, height - new_size))
    frame.alpha_composite(face, dest=(overlay_x, overlay_y))
    return frame


def process_uploaded_drawing(uploaded_file, object_name: str) -> str:
    image = Image.open(uploaded_file).convert("RGBA")
    cleaned = remove(image)
    cleaned_image = Image.open(io.BytesIO(cleaned)).convert("RGBA") if isinstance(cleaned, bytes) else cleaned.convert("RGBA")
    output_path = PROCESSED_DIR / f"{slugify(object_name)}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_image.save(output_path)
    return str(output_path)


def render_overlay_preview(image_path: str, expression_name: str, overlay_position: dict) -> Image.Image:
    return _overlay_expression(Image.open(image_path).convert("RGBA"), expression_name, overlay_position)


def _character_kind(object_name: str) -> str:
    lowered = object_name.lower()
    if any(token in lowered for token in ("dog", "puppy", "moti")):
        return "dog"
    if any(token in lowered for token in ("cat", "rabbit", "lion", "tiger", "fox", "frog", "owl", "cow", "elephant", "monkey")):
        return "animal"
    if any(token in lowered for token in ("neighbour", "neighbor", "friends", "students")):
        return "group"
    if any(token in lowered for token in ("mother", "father", "maa", "mom", "mummy", "mum", "teacher", "king", "queen")):
        return "adult"
    if any(token in lowered for token in ("boy", "girl", "badal", "child", "student", "prince", "princess")):
        return "child"
    if any(token in lowered for token in ("circle", "square", "triangle", "rectangle", "star")):
        return "shape"
    return "child"


def _draw_person(draw: ImageDraw.ImageDraw, kind: str, seed_value: int) -> None:
    skin_tones = [(255, 224, 189), (233, 190, 152), (201, 149, 108), (153, 102, 69)]
    hair_colors = [(66, 46, 36), (94, 60, 48), (34, 34, 46), (121, 77, 42)]
    outfit_palettes = {
        "child": ((88, 154, 255), (247, 201, 72)),
        "adult": ((171, 109, 224), (78, 166, 123)),
        "group": ((245, 135, 158), (96, 176, 255)),
    }
    skin = skin_tones[seed_value % len(skin_tones)]
    hair = hair_colors[(seed_value // 3) % len(hair_colors)]
    shirt, pants = outfit_palettes.get(kind, outfit_palettes["child"])

    if kind == "group":
        x_positions = [170, 350, 530]
        scales = [0.78, 0.88, 0.78]
        for x, scale in zip(x_positions, scales):
            head_w = int(120 * scale)
            body_w = int(96 * scale)
            body_h = int(170 * scale)
            leg_h = int(112 * scale)
            head_left = x - head_w // 2
            head_top = int(150 - 18 * scale)
            draw.ellipse((head_left, head_top, head_left + head_w, head_top + head_w), fill=skin, outline=(110, 78, 58), width=4)
            draw.arc((head_left - 6, head_top - 8, head_left + head_w + 6, head_top + head_w // 2), start=180, end=360, fill=hair, width=12)
            body_left = x - body_w // 2
            body_top = head_top + head_w - 12
            draw.rounded_rectangle((body_left, body_top, body_left + body_w, body_top + body_h), radius=24, fill=shirt, outline=(55, 65, 90), width=4)
            draw.line((body_left, body_top + 35, body_left - int(46 * scale), body_top + 95), fill=skin, width=max(8, int(12 * scale)))
            draw.line((body_left + body_w, body_top + 35, body_left + body_w + int(46 * scale), body_top + 95), fill=skin, width=max(8, int(12 * scale)))
            draw.line((x - int(18 * scale), body_top + body_h, x - int(18 * scale), body_top + body_h + leg_h), fill=pants, width=max(10, int(18 * scale)))
            draw.line((x + int(18 * scale), body_top + body_h, x + int(18 * scale), body_top + body_h + leg_h), fill=pants, width=max(10, int(18 * scale)))
        return

    head_box = (224, 124, 476, 376) if kind == "adult" else (236, 136, 464, 364)
    draw.ellipse(head_box, fill=skin, outline=(120, 86, 70), width=5)
    hair_top = 104 if kind == "adult" else 118
    draw.arc((head_box[0] - 8, hair_top, head_box[2] + 8, head_box[1] + 70), start=180, end=360, fill=hair, width=18)
    if kind == "adult":
        draw.rounded_rectangle((250, 352, 450, 592), radius=44, fill=shirt, outline=(71, 82, 130), width=4)
        draw.line((280, 388, 208, 492), fill=skin, width=18)
        draw.line((420, 388, 492, 492), fill=skin, width=18)
        draw.line((316, 590, 300, 666), fill=pants, width=22)
        draw.line((384, 590, 400, 666), fill=pants, width=22)
    else:
        draw.rounded_rectangle((268, 360, 432, 570), radius=38, fill=shirt, outline=(71, 82, 130), width=4)
        draw.line((288, 392, 220, 482), fill=skin, width=16)
        draw.line((412, 392, 480, 482), fill=skin, width=16)
        draw.line((324, 568, 308, 652), fill=pants, width=20)
        draw.line((376, 568, 392, 652), fill=pants, width=20)

    eye_y = 232 if kind == "adult" else 236
    for x in (300, 382):
        draw.ellipse((x, eye_y, x + 36, eye_y + 36), fill=(52, 58, 80))
        draw.ellipse((x + 12, eye_y + 12, x + 20, eye_y + 20), fill=(255, 255, 255))
    draw.arc((300, 278, 400, 334), start=15, end=165, fill=(160, 74, 67), width=5)

def _draw_dog(draw: ImageDraw.ImageDraw, seed_value: int) -> None:
    fur_colors = [(199, 145, 92), (235, 220, 192), (92, 78, 68), (223, 172, 92)]
    patch_colors = [(255, 245, 232), (80, 60, 54), (130, 90, 58), (250, 227, 200)]
    fur = fur_colors[seed_value % len(fur_colors)]
    patch = patch_colors[(seed_value // 2) % len(patch_colors)]
    draw.ellipse((180, 250, 520, 500), fill=fur, outline=(94, 72, 52), width=5)
    draw.ellipse((390, 138, 610, 330), fill=fur, outline=(94, 72, 52), width=5)
    draw.ellipse((420, 205, 505, 280), fill=patch)
    draw.ellipse((448, 230, 490, 272), fill=(38, 38, 38))
    draw.ellipse((445, 192, 470, 217), fill=(40, 40, 52))
    draw.ellipse((515, 192, 540, 217), fill=(40, 40, 52))
    draw.pieslice((380, 152, 450, 250), start=160, end=330, fill=(110, 78, 56), outline=(94, 72, 52))
    draw.pieslice((542, 152, 612, 250), start=210, end=20, fill=(110, 78, 56), outline=(94, 72, 52))
    draw.arc((445, 246, 535, 292), start=15, end=165, fill=(126, 72, 64), width=4)
    for x in (240, 320, 392, 470):
        draw.line((x, 470, x - 10, 650), fill=fur, width=24)
    draw.arc((100, 245, 248, 420), start=300, end=80, fill=(94, 72, 52), width=16)
    draw.ellipse((225, 286, 300, 360), fill=patch)


def _draw_animal(draw: ImageDraw.ImageDraw, object_name: str, seed_value: int) -> None:
    lowered = object_name.lower()
    if "rabbit" in lowered:
        fur = (244, 244, 244)
        inner = (255, 190, 212)
        draw.ellipse((205, 235, 505, 520), fill=fur, outline=(118, 118, 130), width=5)
        draw.ellipse((270, 145, 360, 310), fill=fur, outline=(118, 118, 130), width=5)
        draw.ellipse((350, 145, 440, 310), fill=fur, outline=(118, 118, 130), width=5)
        draw.ellipse((294, 168, 336, 286), fill=inner)
        draw.ellipse((374, 168, 416, 286), fill=inner)
        draw.ellipse((260, 168, 450, 360), fill=fur, outline=(118, 118, 130), width=5)
        draw.ellipse((314, 230, 338, 254), fill=(52, 58, 80))
        draw.ellipse((374, 230, 398, 254), fill=(52, 58, 80))
        draw.polygon([(350, 270), (370, 298), (330, 298)], fill=(255, 148, 162))
        draw.arc((310, 292, 390, 338), start=12, end=168, fill=(142, 88, 104), width=4)
        return
    if "lion" in lowered or "tiger" in lowered:
        body = (224, 169, 84) if "lion" in lowered else (249, 171, 84)
        mane = (126, 72, 40) if "lion" in lowered else (78, 54, 38)
        draw.ellipse((180, 250, 520, 500), fill=body, outline=(94, 72, 52), width=5)
        draw.ellipse((220, 128, 520, 428), fill=mane)
        draw.ellipse((270, 178, 470, 378), fill=body, outline=(94, 72, 52), width=5)
        for x in (318, 394):
            draw.ellipse((x, 236, x + 28, 264), fill=(40, 40, 52))
        draw.polygon([(370, 274), (350, 300), (390, 300)], fill=(88, 58, 56))
        draw.arc((320, 296, 420, 344), start=15, end=165, fill=(120, 72, 52), width=4)
        return
    _draw_dog(draw, seed_value)


def _draw_star(draw: ImageDraw.ImageDraw, center_x: int, center_y: int, radius: int, fill: tuple[int, int, int]) -> None:
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        current_radius = radius if index % 2 == 0 else radius * 0.42
        points.append((center_x + current_radius * math.cos(angle), center_y + current_radius * math.sin(angle)))
    draw.polygon(points, fill=fill, outline=(96, 84, 78))


def _draw_shape_character(draw: ImageDraw.ImageDraw, object_name: str) -> None:
    lowered = object_name.lower()
    face_fill = (255, 221, 116)
    outline = (96, 84, 78)
    if "circle" in lowered:
        draw.ellipse((190, 160, 510, 480), fill=face_fill, outline=outline, width=6)
    elif "square" in lowered or "rectangle" in lowered:
        draw.rounded_rectangle((190, 160, 510, 480), radius=28, fill=face_fill, outline=outline, width=6)
    elif "triangle" in lowered:
        draw.polygon([(350, 130), (160, 500), (540, 500)], fill=face_fill, outline=outline)
    else:
        _draw_star(draw, 350, 320, 180, face_fill)
        draw.line((350, 486, 350, 610), fill=(84, 72, 68), width=20)
        draw.line((350, 520, 270, 620), fill=(84, 72, 68), width=16)
        draw.line((350, 520, 430, 620), fill=(84, 72, 68), width=16)
    for x in (295, 385):
        draw.ellipse((x, 255, x + 30, 285), fill=(52, 58, 80))
    draw.arc((292, 315, 408, 375), start=18, end=162, fill=(160, 74, 67), width=5)


def generate_character_fallback(object_name: str) -> str:
    output_path = GENERATED_CHARACTERS_DIR / f"{slugify(object_name)}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_value = _seed(object_name)
    palettes = [
        ((255, 216, 178), (245, 146, 110)),
        ((192, 223, 255), (109, 171, 240)),
        ((210, 236, 180), (126, 190, 104)),
        ((248, 218, 240), (217, 145, 198)),
        ((255, 224, 140), (255, 117, 117)),
    ]
    top_color, bottom_color = palettes[seed_value % len(palettes)]
    image = Image.new("RGBA", (700, 700), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for y in range(700):
        mix = y / 699
        color = tuple(int(top + (bottom - top) * mix) for top, bottom in zip(top_color, bottom_color)) + (255,)
        draw.line((0, y, 700, y), fill=color)
    draw.rounded_rectangle((88, 66, 612, 654), radius=74, fill=(255, 255, 255, 150), outline=(255, 255, 255, 215), width=5)

    kind = _character_kind(object_name)
    if kind in {"child", "adult", "group"}:
        _draw_person(draw, kind, seed_value)
    elif kind == "dog":
        _draw_dog(draw, seed_value)
    elif kind == "animal":
        _draw_animal(draw, object_name, seed_value)
    else:
        _draw_shape_character(draw, object_name)
    image.save(output_path)
    return str(output_path)


def _ensure_character_asset(
    lesson_title: str,
    lesson_text: str,
    object_name: str,
    asset_map: dict[str, str],
    overlay_positions: dict[str, dict],
    use_ai_generated_characters: bool,
    auto_generate_missing_drawings: bool,
) -> None:
    existing_path = asset_map.get(object_name)
    if existing_path and Path(existing_path).exists():
        return
    if not should_apply_expression(object_name):
        return
    if use_ai_generated_characters:
        ai_path = ensure_ai_character_art(lesson_title, lesson_text, object_name, slugify(object_name))
        if ai_path:
            asset_map[object_name] = ai_path
            overlay_positions.setdefault(object_name, DEFAULT_OVERLAY.copy())
            return
    if auto_generate_missing_drawings:
        asset_map[object_name] = generate_character_fallback(object_name)
        overlay_positions.setdefault(object_name, DEFAULT_OVERLAY.copy())


def ensure_character_assets(
    lesson_title: str,
    lesson_text: str,
    selected_objects: list[str],
    processed_drawings: dict[str, str],
    overlay_positions: dict[str, dict],
    use_ai_generated_characters: bool,
    auto_generate_missing_drawings: bool,
) -> dict[str, str]:
    asset_map = dict(processed_drawings)
    for object_name in selected_objects:
        _ensure_character_asset(
            lesson_title,
            lesson_text,
            object_name,
            asset_map,
            overlay_positions,
            use_ai_generated_characters,
            auto_generate_missing_drawings,
        )
    return asset_map


def _target_character_height(object_name: str) -> int:
    lowered = object_name.lower()
    for key, height in CHARACTER_HEIGHTS.items():
        if key in lowered:
            return height
    if any(token in lowered for token in ("cat", "rabbit", "lion", "tiger", "fox", "frog", "owl", "cow", "elephant", "monkey")):
        return 240
    if should_apply_expression(object_name):
        return 280
    return 220


def _clip_with_duration(clip: ImageClip, duration: float) -> ImageClip:
    return clip.with_duration(duration) if hasattr(clip, "with_duration") else clip.set_duration(duration)


def _clip_with_audio(clip: ImageClip, audio_clip: AudioFileClip | None):
    if audio_clip is None:
        return clip
    return clip.with_audio(audio_clip) if hasattr(clip, "with_audio") else clip.set_audio(audio_clip)


def _write_videofile(clip, output_path: str) -> None:
    kwargs = {
        "fps": 16,
        "codec": "libx264",
        "audio_codec": "aac",
        "preset": "ultrafast",
        "threads": 4,
        "logger": None,
    }
    try:
        clip.write_videofile(output_path, **kwargs)
    except TypeError:
        kwargs.pop("preset", None)
        kwargs.pop("threads", None)
        kwargs.pop("logger", None)
        clip.write_videofile(output_path, **kwargs)

def _contains_any(text: str, keywords: tuple[str, ...] | list[str] | set[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _pit_phase(sentence: str) -> str | None:
    lowered = sentence.lower()
    if "pit" not in lowered and not _contains_any(lowered, ("rope", "pull", "rescued")):
        return None
    if _contains_any(lowered, ("pull", "rope", "rescued", "save")):
        return "rescue"
    if _contains_any(lowered, ("could not come out", "tried hard", "come out of the pit")):
        return "struggle"
    if _contains_any(lowered, ("fell into", "slipped", "fell")):
        return "fall"
    return "pit"


def _is_best_friends_story(selected_objects: list[str]) -> bool:
    core = {"Circle", "Square", "Triangle", "Rectangle"}
    return core.issubset(set(selected_objects))


CHAPTER2_SPEAKER_RANGES = {
    "Circle": range(3, 8),
    "Square": range(9, 12),
    "Triangle": range(12, 18),
    "Rectangle": range(18, 27),
    "Star": range(29, 44),
}


CHAPTER2_ALL_GROUP_SCENES = {0, 1, 2, 8, 27, 28, 44}
CHAPTER2_CORE_ORDER = ["Circle", "Square", "Triangle", "Rectangle"]


def _chapter2_star_present(scene_index: int, sentence: str) -> bool:
    return scene_index >= 29 or "star" in sentence.lower()


def _chapter2_present_objects(selected_objects: list[str], scene_index: int, sentence: str) -> list[str]:
    present = [name for name in CHAPTER2_CORE_ORDER if name in selected_objects]
    if "Star" in selected_objects and _chapter2_star_present(scene_index, sentence):
        present.append("Star")
    return present


def _chapter2_speaker(scene_index: int) -> str | None:
    for name, index_range in CHAPTER2_SPEAKER_RANGES.items():
        if scene_index in index_range:
            return name
    return None


def _filter_scene_objects(sentence: str, scene_objects: list[str], selected_objects: list[str], lesson_title: str = "", scene_index: int = 0) -> list[str]:
    lowered = sentence.lower()
    if _is_best_friends_story(selected_objects):
        present = _chapter2_present_objects(selected_objects, scene_index, sentence)
        extras = []
        for name in ("Mountains", "Patterns"):
            if name in selected_objects and name.lower() in lowered:
                extras.append(name)
        return present + [name for name in extras if name not in present]

    objects = list(dict.fromkeys(scene_objects))

    def add_if_present(name: str) -> None:
        if name in selected_objects and name not in objects:
            objects.append(name)

    if _contains_any(lowered, ("play", "played", "games", "best friends")):
        add_if_present("Badal")
        add_if_present("Moti")
    if _contains_any(lowered, ("hugged", "thanked")):
        objects = [name for name in objects if name not in {"Mother", "Neighbours"}]
        add_if_present("Badal")
        add_if_present("Moti")
    if _contains_any(lowered, ("brought him home", "stay with us", "promise to take care", "named the puppy")):
        objects = [name for name in objects if name != "Neighbours"]
        add_if_present("Badal")
        add_if_present("Moti")
        add_if_present("Mother")
    if _contains_any(lowered, ("scent", "barked", "barking")):
        objects = [name for name in objects if name != "Badal"]
        add_if_present("Moti")
        if "pit" in lowered:
            add_if_present("Pit")
    if _contains_any(lowered, ("worried",)):
        objects = [name for name in objects if name not in {"Badal", "Neighbours"}]
        add_if_present("Mother")
    if _contains_any(lowered, ("waiting at the gate",)):
        objects = [name for name in objects if name not in {"Badal", "Neighbours", "Mother"}]
        add_if_present("Moti")
    if _contains_any(lowered, ("search", "gathered the neighbours", "accompanied")):
        objects = [name for name in objects if name != "Badal"]
        add_if_present("Mother")
        add_if_present("Neighbours")
        if _contains_any(lowered, ("moti", "accompanied")):
            add_if_present("Moti")
    pit_phase = _pit_phase(sentence)
    if pit_phase == "fall":
        objects = [name for name in objects if name in {"Badal", "Pit"}]
        add_if_present("Badal")
        add_if_present("Pit")
    elif pit_phase == "struggle":
        objects = [name for name in objects if name in {"Badal", "Pit"}]
        add_if_present("Badal")
        add_if_present("Pit")
    elif pit_phase == "rescue":
        objects = [name for name in objects if name not in {"School"}]
        for name in ("Pit", "Badal", "Mother", "Neighbours", "Moti", "Rope"):
            add_if_present(name)
    elif "pit" in lowered and "Badal" in objects and not _contains_any(lowered, ("pull", "rope", "rescued", "save", "could not come out", "tried hard", "fell", "slipped")):
        objects = [name for name in objects if name != "Badal"]

    ordered = []
    for name in objects:
        if name in selected_objects and name not in ordered:
            ordered.append(name)
    return ordered


def _placement(name: str, center_x: int, baseline_y: int, *, scale: float = 1.0, z: int = 2, angle: int = 0, flip: bool = False, pose: str = "stand", is_speaker: bool = False) -> dict:
    return {
        "name": name,
        "center_x": center_x,
        "baseline_y": baseline_y,
        "scale": scale,
        "z": z,
        "angle": angle,
        "flip": flip,
        "pose": pose,
        "is_speaker": is_speaker,
    }


def _detect_pit_anchor(background: Image.Image) -> tuple[int, int, int]:
    rgb = np.array(background.convert("RGB"))
    if rgb.size == 0:
        return (930, 612, 240)
    height, width, _ = rgb.shape
    start_y = int(height * 0.45)
    end_y = int(height * 0.96)
    region = rgb[start_y:end_y, :]
    if region.size == 0:
        return (930, 612, 240)
    gray = region.mean(axis=2)
    darkness = 255 - gray
    x_bias = np.linspace(0.92, 1.12, width)[None, :]
    y_bias = np.linspace(0.9, 1.16, region.shape[0])[:, None]
    score = darkness * x_bias * y_bias
    threshold = np.percentile(score, 91)
    mask = score >= threshold
    ys, xs = np.where(mask)
    if len(xs) < 180:
        return (930, 612, 240)
    weights = score[ys, xs]
    center_x = int(np.average(xs, weights=weights))
    center_y = int(np.average(ys, weights=weights)) + start_y
    left = int(np.percentile(xs, 12))
    right = int(np.percentile(xs, 88))
    span = max(180, min(360, right - left))
    return (
        max(180, min(width - 120, center_x)),
        max(540, min(height - 40, center_y + 55)),
        span,
    )


def _build_scene_layout(sentence: str, scene_objects: list[str], pit_anchor: tuple[int, int, int] | None = None, lesson_title: str = "", scene_index: int = 0) -> list[dict]:
    lowered = sentence.lower()
    pit_phase = _pit_phase(sentence)
    layout = []
    pit_x, pit_y, pit_span = pit_anchor or (930, 612, 260)

    if _is_best_friends_story(scene_objects):
        speaker = _chapter2_speaker(scene_index)
        present = [name for name in scene_objects if name in CHAPTER2_CORE_ORDER or name == "Star"]
        if scene_index in CHAPTER2_ALL_GROUP_SCENES or not speaker or speaker not in present:
            positions = [(250, 610), (470, 610), (690, 610), (910, 610)]
            if "Star" in present:
                positions = [(190, 610), (390, 610), (590, 610), (790, 610), (1010, 610)]
            for (x_pos, y_pos), name in zip(positions, present):
                scale = 1.04 if name == "Star" else 1.0
                layout.append(_placement(name, x_pos, y_pos, scale=scale, z=3, flip=False, pose="stand", is_speaker=False))
            return layout

        listener_names = [name for name in present if name != speaker]
        speaker_scale = 1.08 if speaker == "Star" else 1.02
        layout.append(_placement(speaker, 320, 600, scale=max(1.18, speaker_scale + 0.12), z=4, flip=False, pose="talk", is_speaker=True))
        if len(listener_names) == 3:
            listener_positions = [(760, 570), (930, 620), (1080, 570)]
        elif len(listener_names) == 4:
            listener_positions = [(710, 555), (845, 615), (980, 615), (1110, 555)]
        else:
            listener_positions = [(860, 590)] * max(1, len(listener_names))
        for (x_pos, y_pos), name in zip(listener_positions, listener_names):
            scale = 0.92 if name == "Star" else 0.9
            layout.append(_placement(name, x_pos, y_pos, scale=scale, z=3, flip=True, pose="stand", is_speaker=False))
        return layout

    if pit_phase == "fall":
        if "Pit" in scene_objects:
            layout.append(_placement("Pit", pit_x, pit_y, scale=1.0, z=1))
        if "Badal" in scene_objects:
            layout.append(_placement("Badal", pit_x - max(20, pit_span // 8), pit_y - 52, scale=0.82, z=3, angle=-8, pose="fall"))
        return layout

    if pit_phase == "struggle":
        if "Pit" in scene_objects:
            layout.append(_placement("Pit", pit_x, pit_y, scale=1.0, z=1))
        if "Badal" in scene_objects:
            layout.append(_placement("Badal", pit_x - max(14, pit_span // 10), pit_y - 30, scale=0.62, z=3, angle=-6, pose="climb"))
        return layout

    if pit_phase == "rescue":
        left_rim_x = max(210, pit_x - max(185, int(pit_span * 0.78)))
        rim_y = max(330, pit_y - max(210, int(pit_span * 0.58)))
        neighbour_x = left_rim_x
        mother_x = left_rim_x + 150
        moti_x = max(170, left_rim_x - 135)
        moti_y = min(FRAME_SIZE[1] - 28, pit_y + 78)
        badal_x = min(FRAME_SIZE[0] - 135, pit_x + max(42, int(pit_span * 0.14)))
        badal_y = pit_y - 42
        rope_start_x = mother_x + 35
        rope_start_y = rim_y - 38
        rope_target_x = pit_x - max(12, int(pit_span * 0.06))
        rope_target_y = pit_y + 120
        rope_x = int((rope_start_x + rope_target_x) / 2)
        rope_y = int((rope_start_y + rope_target_y) / 2) + 30
        rope_angle = int(math.degrees(math.atan2(rope_target_y - rope_start_y, rope_target_x - rope_start_x)))

        layout.append(_placement("Pit", pit_x, pit_y, scale=1.0, z=1))
        if "Badal" in scene_objects:
            layout.append(_placement("Badal", badal_x, badal_y, scale=0.56, z=2, angle=-3, pose="rescue_lift"))
        if "Neighbours" in scene_objects:
            layout.append(_placement("Neighbours", neighbour_x, rim_y, scale=0.74, z=4, flip=False, pose="reach"))
        if "Mother" in scene_objects:
            layout.append(_placement("Mother", mother_x, rim_y - 10, scale=0.76, z=4, flip=False, pose="reach"))
        if "Rope" in scene_objects:
            layout.append(_placement("Rope", rope_x, rope_y, scale=0.84, z=5, angle=rope_angle, flip=False, pose="reach"))
        if "Moti" in scene_objects:
            layout.append(_placement("Moti", moti_x, moti_y, scale=0.70, z=3, flip=False, pose="stand"))
        return layout

    if _contains_any(lowered, ("hugged", "thanked")):
        if "Badal" in scene_objects:
            layout.append(_placement("Badal", 560, GROUND_Y, scale=1.0, z=3, flip=False, pose="hug"))
        if "Moti" in scene_objects:
            layout.append(_placement("Moti", 700, 628, scale=0.98, z=3, flip=True, pose="hug"))
        return layout

    if _contains_any(lowered, ("play", "played", "games")):
        if "Badal" in scene_objects:
            layout.append(_placement("Badal", 470, GROUND_Y, scale=1.0, z=3, flip=False, pose="play"))
        if "Moti" in scene_objects:
            layout.append(_placement("Moti", 760, 628, scale=1.0, z=3, flip=True, pose="play"))
        return layout

    if _contains_any(lowered, ("scent", "barked", "barking")):
        if "Pit" in scene_objects:
            layout.append(_placement("Pit", 930, 612, scale=1.0, z=1))
        if "Moti" in scene_objects:
            layout.append(_placement("Moti", 420, 628, scale=1.0, z=3, flip=False, pose="sniff"))
        return layout

    if _contains_any(lowered, ("brought him home", "stay with us", "promise to take care", "named the puppy")):
        if "Badal" in scene_objects:
            layout.append(_placement("Badal", 370, 632, scale=1.0, z=3, flip=False, pose="stand"))
        if "Mother" in scene_objects:
            layout.append(_placement("Mother", 840, 626, scale=0.98, z=3, flip=True, pose="stand"))
        if "Moti" in scene_objects:
            layout.append(_placement("Moti", 620, 646, scale=0.96, z=3, flip=False, pose="stand"))
        return layout

    expressive = [name for name in scene_objects if should_apply_expression(name)]
    if len(expressive) == 1:
        name = expressive[0]
        pose = "run" if _contains_any(lowered, ("ran", "returning", "followed", "went")) and name == "Moti" else "stand"
        layout.append(_placement(name, 640, 626 if name != "Moti" else 636, scale=1.0, z=3, pose=pose))
    elif len(expressive) == 2:
        left_name, right_name = expressive[:2]
        layout.append(_placement(left_name, 430, 626 if left_name != "Moti" else 636, scale=1.0, z=3, flip=False, pose="stand"))
        layout.append(_placement(right_name, 840, 626 if right_name != "Moti" else 636, scale=1.0, z=3, flip=True, pose="stand"))
    elif len(expressive) >= 3:
        x_positions = [290, 640, 980]
        for index, name in enumerate(expressive[:3]):
            layout.append(_placement(name, x_positions[index], 626 if name != "Moti" else 636, scale=1.0, z=3, flip=index > 0, pose="stand"))

    if not layout and "Pit" in scene_objects:
        layout.append(_placement("Pit", 640, 612, scale=1.0, z=1))
    return layout


def _animated_placement(placement: dict, phase: float) -> dict:
    animated = dict(placement)
    pose = animated.get("pose", "stand")
    sway = -1 if animated.get("flip") else 1
    if pose == "run":
        animated["center_x"] += int(36 * phase * sway)
        animated["baseline_y"] += int(6 * math.sin(phase * math.pi))
        animated["angle"] += int(6 * phase * sway)
    elif pose == "reach":
        animated["center_x"] += int(8 * phase * sway)
        animated["angle"] += int(4 * phase * sway)
    elif pose == "hug":
        animated["center_x"] += int(10 * math.sin(phase * math.pi))
        animated["baseline_y"] -= int(5 * math.sin(phase * math.pi))
    elif pose == "play":
        animated["center_x"] += int(16 * math.sin(phase * math.pi)) * sway
        animated["baseline_y"] -= int(12 * math.sin(phase * math.pi))
        animated["angle"] += int(8 * math.sin(phase * math.pi)) * sway
    elif pose == "sniff":
        animated["center_x"] += int(250 * phase)
        animated["baseline_y"] += int(4 * math.sin(phase * math.pi))
        animated["angle"] -= int(8 * phase)
    elif pose == "rescue_lift":
        animated["baseline_y"] -= int(18 * phase)
        animated["center_x"] += int(3 * phase)
        animated["angle"] += int(4 * phase)
    elif pose == "climb":
        animated["baseline_y"] -= int(18 * math.sin(phase * math.pi))
        animated["center_x"] += int(12 * math.sin(phase * math.pi))
        animated["angle"] += int(8 * math.sin(phase * math.pi))
    elif pose == "fall":
        animated["baseline_y"] += int(10 * phase)
        animated["angle"] -= int(6 * phase)
    else:
        animated["baseline_y"] += int(3 * math.sin(phase * math.pi))
    return animated


def _prepare_character_asset(object_name: str, asset_map: dict[str, str], overlay_positions: dict[str, dict], expression_name: str) -> Image.Image | None:
    asset_path = asset_map.get(object_name)
    if not asset_path or not Path(asset_path).exists():
        return None
    image = Image.open(asset_path).convert("RGBA")
    if should_apply_expression(object_name):
        image = _overlay_expression(image, expression_name, overlay_positions.get(object_name, DEFAULT_OVERLAY), object_name)
    return image


def _transformed_asset(image: Image.Image, placement: dict, object_name: str) -> Image.Image:
    target_height = max(40, int(_target_character_height(object_name) * placement.get("scale", 1.0)))
    ratio = target_height / max(image.height, 1)
    resized = image.resize((max(32, int(image.width * ratio)), target_height), Image.Resampling.LANCZOS)
    if placement.get("flip"):
        resized = resized.transpose(Image.FLIP_LEFT_RIGHT)
    angle = placement.get("angle", 0)
    if angle:
        resized = resized.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    return resized


def _paste_with_shadow(frame: Image.Image, asset: Image.Image, center_x: int, baseline_y: int, z: int) -> None:
    width, height = asset.size
    left = int(center_x - width / 2)
    top = int(baseline_y - height)
    shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_width = max(36, int(width * 0.46))
    shadow_height = max(12, int(height * 0.08))
    shadow_alpha = 58 if z >= 3 else 36
    shadow_draw.ellipse(
        (center_x - shadow_width // 2, baseline_y - shadow_height // 2, center_x + shadow_width // 2, baseline_y + shadow_height // 2),
        fill=(0, 0, 0, shadow_alpha),
    )
    frame.alpha_composite(shadow)
    frame.alpha_composite(asset, dest=(left, top))


def _paste_speaker_glow(frame: Image.Image, asset: Image.Image, center_x: int, baseline_y: int) -> None:
    width, height = asset.size
    left = int(center_x - width / 2)
    top = int(baseline_y - height)
    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    for expand, alpha in ((26, 34), (16, 52), (8, 78)):
        gdraw.rounded_rectangle(
            (left - expand, top - expand, left + width + expand, top + height + expand),
            radius=40,
            fill=(255, 225, 92, alpha),
            outline=(255, 255, 180, min(120, alpha + 20)),
            width=4,
        )
    frame.alpha_composite(glow)


def _speaker_bubble_text(sentence: str, speaker: str) -> str:
    text = sentence.replace("\n", " ").strip().strip("\"'")
    prefixes = [
        f"{speaker} said,",
        f"{speaker} says,",
        f"The {speaker.lower()} said,",
        f"The {speaker.lower()} says,",
    ]
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix.lower()):
            text = text[len(prefix):].strip(" \"'.,")
            break
    if len(text) > 72:
        cut = text.rfind(" ", 0, 72)
        text = text[:cut if cut > 30 else 72].rstrip(" ,.") + "..."
    return text

def _draw_speech_bubble(frame: Image.Image, speaker_text: str, center_x: int, top_y: int) -> None:
    draw = ImageDraw.Draw(frame)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    words = speaker_text
    if len(words) > 36:
        split_at = words.rfind(" ", 0, len(words) // 2 + 8)
        if split_at > 12:
            words = words[:split_at] + "\n" + words[split_at + 1:]
    bbox = draw.multiline_textbbox((0, 0), words, font=font, align="center", spacing=6)
    bubble_w = (bbox[2] - bbox[0]) + 34
    bubble_h = (bbox[3] - bbox[1]) + 24
    bubble_x = max(24, min(FRAME_SIZE[0] - bubble_w - 24, int(center_x - bubble_w / 2)))
    bubble_y = max(18, top_y - bubble_h - 26)
    draw.rounded_rectangle(
        (bubble_x, bubble_y, bubble_x + bubble_w, bubble_y + bubble_h),
        radius=24,
        fill=(255, 255, 255, 232),
        outline=(255, 193, 87, 255),
        width=4,
    )
    tail = [
        (center_x - 12, bubble_y + bubble_h - 2),
        (center_x + 6, bubble_y + bubble_h - 2),
        (center_x - 3, bubble_y + bubble_h + 22),
    ]
    draw.polygon(tail, fill=(255, 255, 255, 232), outline=(255, 193, 87, 255))
    draw.multiline_text((bubble_x + 17, bubble_y + 10), words, font=font, fill=(58, 48, 42), align="center", spacing=6)

def _draw_caption(frame: Image.Image, sentence: str) -> None:
    draw = ImageDraw.Draw(frame)
    words = sentence.strip()
    if len(words) > 110:
        midpoint = len(words) // 2
        split_at = words.rfind(" ", 0, midpoint)
        if split_at > 20:
            words = words[:split_at] + "\n" + words[split_at + 1 :]
    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.multiline_textbbox((0, 0), words, font=font, align="center", spacing=10)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (FRAME_SIZE[0] - text_width) / 2
    y = int(FRAME_SIZE[1] * 0.79 - text_height / 2)
    draw.multiline_text(
        (x, y),
        words,
        font=font,
        fill=(255, 255, 255),
        align="center",
        spacing=10,
        stroke_width=4,
        stroke_fill=(0, 0, 0),
    )


def _scene_expression(sentence: str, default_expression: str) -> str:
    lowered = sentence.lower()
    if _contains_any(lowered, ("fell into", "slipped", "could not come out", "worried", "waiting at the gate", "search", "scent", "barked", "alerted")):
        return "sad" if _contains_any(lowered, ("fell into", "slipped", "could not come out")) else "worried"
    if _contains_any(lowered, ("pull", "rope", "rescued", "relieved", "hugged", "thanked")):
        return "happy"
    return default_expression


def compose_scene_frame(
    lesson_title: str,
    sentence: str,
    expression_name: str,
    scene_index: int,
    scene_objects: list[str],
    asset_map: dict[str, str],
    overlay_positions: dict[str, dict],
    selected_objects: list[str],
    phase: float,
) -> Image.Image:
    background = _load_background_from_assets(lesson_title, scene_index)
    if background is None:
        raise ValueError(f"Missing uploaded background for scene {scene_index + 1} in {lesson_title}.")
    frame = background.resize(FRAME_SIZE, Image.Resampling.LANCZOS).convert("RGBA")
    scene_objects = _filter_scene_objects(sentence, scene_objects, selected_objects, lesson_title, scene_index)
    pit_anchor = _detect_pit_anchor(frame) if "Pit" in scene_objects or _pit_phase(sentence) else None
    layout = _build_scene_layout(sentence, scene_objects, pit_anchor, lesson_title, scene_index)

    for placement in sorted(layout, key=lambda item: item["z"]):
        name = placement["name"]
        if name in {"Pit", "School"}:
            continue
        asset = _prepare_character_asset(name, asset_map, overlay_positions, expression_name)
        if asset is None:
            continue
        animated = _animated_placement(placement, phase)
        transformed = _transformed_asset(asset, animated, name)
        if name == "Rope":
            left = int(animated["center_x"] - transformed.width / 2)
            top = int(animated["baseline_y"] - transformed.height)
            frame.alpha_composite(transformed, dest=(left, top))
        else:
            if animated.get("is_speaker"):
                _paste_speaker_glow(frame, transformed, animated["center_x"], animated["baseline_y"])
            _paste_with_shadow(frame, transformed, animated["center_x"], animated["baseline_y"], animated["z"])
            if animated.get("is_speaker"):
                speaker_text = _speaker_bubble_text(sentence, name)
                asset_top = int(animated["baseline_y"] - transformed.height)
                _draw_speech_bubble(frame, speaker_text, animated["center_x"], asset_top)
    _draw_caption(frame, sentence)
    return frame


def generate_lesson_video(
    lesson_title: str,
    lesson_text: str,
    scenes: list[dict],
    selected_objects: list[str],
    processed_drawings: dict[str, str],
    overlay_positions: dict[str, dict],
    auto_generate_missing_drawings: bool,
    use_ai_generated_characters: bool,
    add_narration_audio: bool,
    progress_callback=None,
) -> tuple[str, dict[str, str]]:
    asset_map = ensure_character_assets(
        lesson_title=lesson_title,
        lesson_text=lesson_text,
        selected_objects=selected_objects,
        processed_drawings=processed_drawings,
        overlay_positions=overlay_positions,
        use_ai_generated_characters=use_ai_generated_characters,
        auto_generate_missing_drawings=auto_generate_missing_drawings,
    )

    clips = []
    total_scenes = max(1, len(scenes))
    for index, scene in enumerate(scenes):
        if progress_callback:
            progress_callback(10 + int(70 * index / total_scenes), f"Rendering scene {index + 1} of {total_scenes}")
        scene_expression = _scene_expression(scene["sentence"], scene.get("expression", "happy"))
        start_frame = compose_scene_frame(
            lesson_title,
            scene["sentence"],
            scene_expression,
            index,
            scene.get("objects", []),
            asset_map,
            overlay_positions,
            selected_objects,
            0.0,
        )
        end_frame = compose_scene_frame(
            lesson_title,
            scene["sentence"],
            scene_expression,
            index,
            scene.get("objects", []),
            asset_map,
            overlay_positions,
            selected_objects,
            1.0,
        )
        audio_clip = None
        scene_duration = 2.6
        if add_narration_audio:
            audio_path = generate_narration_audio(index, scene["sentence"])
            if audio_path and Path(audio_path).exists():
                try:
                    audio_clip = AudioFileClip(audio_path)
                    scene_duration = max(scene_duration, float(audio_clip.duration) + 0.35)
                except Exception:
                    audio_clip = None
        first = _clip_with_duration(ImageClip(np.array(start_frame.convert("RGB"))), scene_duration / 2)
        second = _clip_with_duration(ImageClip(np.array(end_frame.convert("RGB"))), scene_duration / 2)
        scene_clip = concatenate_videoclips([first, second], method="chain")
        scene_clip = _clip_with_audio(scene_clip, audio_clip)
        clips.append(scene_clip)

    if not clips:
        raise ValueError("No scenes were available to render.")

    if progress_callback:
        progress_callback(84, "Stitching video")
    final_clip = concatenate_videoclips(clips, method="chain")
    output_path = VIDEOS_DIR / f"{slugify(lesson_title)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    _write_videofile(final_clip, str(output_path))

    if progress_callback:
        progress_callback(96, "Cleaning temporary clips")
    for clip in clips:
        try:
            clip.close()
        except Exception:
            pass
    try:
        final_clip.close()
    except Exception:
        pass

    return str(output_path), asset_map
