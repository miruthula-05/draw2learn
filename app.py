import base64
import io
from pathlib import Path
import shutil

import streamlit as st
import streamlit.elements.image as st_image
from streamlit.elements.lib.image_utils import image_to_url as _streamlit_image_to_url
from streamlit.elements.lib.layout_utils import LayoutConfig
from PIL import Image

if not hasattr(st_image, "image_to_url"):
    def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
        return _streamlit_image_to_url(
            image=image,
            layout_config=LayoutConfig(width=width),
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )

    st_image.image_to_url = _image_to_url_compat

from streamlit_drawable_canvas import st_canvas

from config import AUDIO_DIR, EXPRESSIONS_DIR, GENERATED_DIR, PROJECT_STATE_FILE, PROCESSED_DIR, VIDEOS_DIR
from lesson_parser import build_story_scenes, should_apply_expression, should_request_child_drawing, split_sentences
from lessons import PREDEFINED_LESSONS
from media_pipeline import (
    available_expression_names,
    ensure_character_assets,
    generate_lesson_video,
    process_uploaded_drawing,
    render_overlay_preview,
)
from storage import load_project_state, save_project_state


DEFAULT_POSITION = {"x": 0, "y": 0, "size": 22}
DEFAULT_LESSON_NAME = next(iter(PREDEFINED_LESSONS))
DISPLAY_WIDTH = 460
PAGE_ORDER = ["lesson_select", "lesson_details", "drawing_stage", "video_generation"]


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #fff7ef 0%, #fef5fb 45%, #eef8ff 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            max-width: 1180px;
        }
        .page-shell {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(255,196,214,0.85);
            border-radius: 28px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            box-shadow: 0 18px 44px rgba(235, 164, 190, 0.16);
        }
        .hero-title {
            font-size: 2.3rem;
            line-height: 1.1;
            font-weight: 800;
            color: #000000;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            color: #000000;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        .lesson-card {
            background: linear-gradient(180deg, #fffdf6 0%, #fff2f8 100%);
            border: 2px solid #ffd7e6;
            border-radius: 24px;
            padding: 1rem;
            min-height: 220px;
            box-shadow: 0 12px 26px rgba(255, 192, 214, 0.16);
        }
        .lesson-card h4 {
            color: #000000;
            margin: 0 0 0.45rem 0;
            font-size: 1.15rem;
        }
        .lesson-card p {
            color: #000000;
            font-size: 0.95rem;
            margin-bottom: 0.8rem;
        }
        .soft-chip {
            display: inline-block;
            padding: 0.32rem 0.8rem;
            margin: 0.18rem 0.25rem 0.18rem 0;
            border-radius: 999px;
            background: #fff0b8;
            color: #4f420f;
            font-size: 0.88rem;
            font-weight: 600;
        }
        .progress-strip {
            display: flex;
            gap: 0.6rem;
            margin: 0.5rem 0 1rem 0;
            flex-wrap: wrap;
        }
        .progress-pill {
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            background: #f4effa;
            color: #000000;
            font-weight: 700;
            font-size: 0.9rem;
            border: 1px solid #e7dcef;
        }
        .progress-pill.active {
            background: linear-gradient(90deg, #ffd5e5 0%, #ffe8ad 100%);
            color: #000000;
            border-color: #f8c7da;
        }
        .summary-card {
            background: linear-gradient(180deg, #eefbff 0%, #fff8ed 100%);
            border: 1px solid #d7eef9;
            border-radius: 22px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .summary-card, .summary-card p, .summary-card h3, .summary-card strong {
            color: #000000 !important;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption, .stText, label, .stSubheader,
        h1, h2, h3, h4, h5, h6,
        div[data-testid="stHeading"] {
            color: #000000 !important;
        }
        div[data-testid="stCheckbox"] label p,
        div[data-testid="stCheckbox"] label span,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stMultiSelect"] span,
        div[data-testid="stMultiSelect"] input,
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"] span,
        div[data-testid="stMultiSelect"] div[role="combobox"] {
            color: #000000 !important;
        }
        div[data-testid="stRadio"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clear_directory(directory: Path) -> None:
    if not directory.exists():
        return
    for item in list(directory.iterdir()):
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        except PermissionError:
            continue
        except OSError:
            continue


def clear_temporary_media() -> None:
    for directory in (PROCESSED_DIR, GENERATED_DIR, AUDIO_DIR, VIDEOS_DIR):
        _clear_directory(directory)
        directory.mkdir(parents=True, exist_ok=True)


def _apply_selected_lesson() -> None:
    lesson = PREDEFINED_LESSONS[st.session_state.lesson_name]
    st.session_state.lesson_title = lesson["title"]
    st.session_state.lesson_text = lesson["text"]


def _lesson_summary(text: str, count: int = 3) -> str:
    sentences = split_sentences(text)
    return " ".join(sentences[:count]) if sentences else text


def initialize_state() -> None:
    saved_state = load_project_state(PROJECT_STATE_FILE)
    if "session_media_cleared" not in st.session_state:
        clear_temporary_media()
        st.session_state.session_media_cleared = True

    st.session_state.setdefault("current_page", "lesson_select")
    st.session_state.setdefault("lesson_name", saved_state.get("lesson_name", DEFAULT_LESSON_NAME))
    _apply_selected_lesson()
    saved_objects = saved_state.get("selected_objects")
    default_objects = PREDEFINED_LESSONS[st.session_state.lesson_name]["objects"]
    st.session_state.setdefault("selected_objects", saved_objects or default_objects)
    st.session_state.setdefault("processed_drawings", {})
    st.session_state.setdefault("overlay_positions", {})
    st.session_state.setdefault("canvas_drawings", {})
    st.session_state.setdefault("canvas_seed_drawings", {})
    st.session_state.setdefault("canvas_revisions", {})
    st.session_state.setdefault("pending_overlay_positions", {})
    st.session_state.setdefault("final_video_path", None)
    st.session_state.auto_generate_missing_drawings = True
    st.session_state.use_ai_generated_characters = False
    st.session_state.add_narration_audio = True


def persist_project_state() -> None:
    save_project_state(
        PROJECT_STATE_FILE,
        {
            "lesson_name": st.session_state.lesson_name,
            "selected_objects": st.session_state.selected_objects,
            "auto_generate_missing_drawings": st.session_state.auto_generate_missing_drawings,
            "use_ai_generated_characters": st.session_state.use_ai_generated_characters,
            "add_narration_audio": st.session_state.add_narration_audio,
        },
    )


def _go_to(page_name: str) -> None:
    st.session_state.current_page = page_name


def _step_progress() -> None:
    labels = {
        "lesson_select": "1. Lessons",
        "lesson_details": "2. Summary",
        "drawing_stage": "3. Drawings",
        "video_generation": "4. Video",
    }
    current_index = PAGE_ORDER.index(st.session_state.current_page)
    pills = []
    for index, page in enumerate(PAGE_ORDER):
        active_class = "active" if index <= current_index else ""
        pills.append(f'<div class="progress-pill {active_class}">{labels[page]}</div>')
    st.markdown(f'<div class="progress-strip">{"".join(pills)}</div>', unsafe_allow_html=True)


def _generate_missing_assets_into_state() -> None:
    updated_assets = ensure_character_assets(
        lesson_title=st.session_state.lesson_title,
        lesson_text=st.session_state.lesson_text,
        selected_objects=st.session_state.selected_objects,
        processed_drawings=st.session_state.processed_drawings,
        overlay_positions=st.session_state.overlay_positions,
        use_ai_generated_characters=st.session_state.use_ai_generated_characters,
        auto_generate_missing_drawings=st.session_state.auto_generate_missing_drawings,
    )
    st.session_state.processed_drawings = updated_assets
    persist_project_state()


def _select_lesson(lesson_name: str) -> None:
    st.session_state.lesson_name = lesson_name
    _apply_selected_lesson()
    st.session_state.selected_objects = PREDEFINED_LESSONS[lesson_name]["objects"]
    st.session_state.processed_drawings = {}
    st.session_state.overlay_positions = {}
    st.session_state.canvas_drawings = {}
    st.session_state.canvas_seed_drawings = {}
    st.session_state.canvas_revisions = {}
    st.session_state.pending_overlay_positions = {}
    st.session_state.final_video_path = None
    clear_temporary_media()
    persist_project_state()
    _go_to("lesson_details")


def lesson_cards_page() -> None:
    st.markdown('<div class="hero-title">Pick a Lesson Adventure</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Choose a lesson card to start building a playful classroom video.</div>', unsafe_allow_html=True)

    lesson_names = list(PREDEFINED_LESSONS.keys())
    columns = st.columns(len(lesson_names))
    for column, lesson_name in zip(columns, lesson_names):
        lesson = PREDEFINED_LESSONS[lesson_name]
        with column:
            st.markdown(
                f"""
                <div class="lesson-card">
                    <h4>{lesson['title']}</h4>
                    <p>{_lesson_summary(lesson['text'], 2)}</p>
                    <div>{''.join(f'<span class="soft-chip">{obj}</span>' for obj in lesson['objects'][:4])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Open {lesson['title']}", key=f"open_{lesson_name}", width="stretch"):
                _select_lesson(lesson_name)


def lesson_details_page() -> None:
    _step_progress()
    lesson = PREDEFINED_LESSONS[st.session_state.lesson_name]
    st.markdown('<div class="hero-title">Lesson Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Review the story and choose which characters children will draw.</div>', unsafe_allow_html=True)
    st.markdown(f"### {lesson['title']}")

    st.subheader("Lesson Summary")
    st.write(_lesson_summary(lesson['text'], 5))

    st.subheader("Lesson Characters and Objects")
    st.markdown("".join(f'<span class="soft-chip">{obj}</span>' for obj in lesson["objects"]), unsafe_allow_html=True)

    st.subheader("Choose characters to draw")
    selected_defaults = set(
        [obj for obj in st.session_state.selected_objects if obj in lesson["objects"]] or lesson["objects"]
    )
    selectable_objects = [obj for obj in lesson["objects"] if should_request_child_drawing(obj)]
    auto_background_objects = [obj for obj in lesson["objects"] if not should_request_child_drawing(obj)]
    selected_objects = []
    checkbox_columns = st.columns(3)
    for index, obj in enumerate(selectable_objects):
        with checkbox_columns[index % 3]:
            if st.checkbox(obj, value=obj in selected_defaults, key=f"character_{st.session_state.lesson_name}_{obj}"):
                selected_objects.append(obj)
    selected_objects.extend([obj for obj in auto_background_objects if obj not in selected_objects])
    st.session_state.selected_objects = selected_objects

    if selectable_objects:
        st.caption(f"Children draw: {', '.join(selectable_objects)}")
    if auto_background_objects:
        st.caption(f"Auto-generated with background: {', '.join(auto_background_objects)}")
    if selected_objects:
        st.success(f"Ready for drawing stage: {', '.join(selected_objects)}")

    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        if st.button("Back to Lessons", width="stretch"):
            _go_to("lesson_select")
    with nav_right:
        if st.button("Next: Drawings", type="primary", width="stretch"):
            persist_project_state()
            _go_to("drawing_stage")


def _canvas_initial_drawing(image_path: str, overlay_position: dict) -> tuple[dict, int]:
    base_image = Image.open(image_path)
    scale = DISPLAY_WIDTH / base_image.width
    display_height = int(base_image.height * scale)
    new_size = max(24, int(base_image.width * (overlay_position.get("size", 22) / 100.0)))
    overlay_x = max(0, min((base_image.width // 2 - new_size // 2) + overlay_position.get("x", 0), base_image.width - new_size))
    overlay_y = max(0, min((base_image.height // 3 - new_size // 2) + overlay_position.get("y", 0), base_image.height - new_size))
    rect = {
        "version": "4.4.0",
        "objects": [
            {
                "type": "rect",
                "left": overlay_x * scale,
                "top": overlay_y * scale,
                "width": new_size * scale,
                "height": new_size * scale,
                "scaleX": 1,
                "scaleY": 1,
                "fill": "rgba(255, 212, 120, 0.22)",
                "stroke": "#ff8eb1",
                "strokeWidth": 4,
                "transparentCorners": False,
                "cornerColor": "#ff8eb1",
                "cornerStrokeColor": "#ff8eb1",
                "borderColor": "#ff8eb1",
            }
        ],
    }
    return rect, display_height


def _prepared_expression_image(expression_name: str) -> Image.Image | None:
    expression_path = EXPRESSIONS_DIR / f"{expression_name}.png"
    if not expression_path.exists():
        expression_path = EXPRESSIONS_DIR / "happy.png"
        if not expression_path.exists():
            return None
    expression = Image.open(expression_path).convert("RGBA")
    bbox = expression.getbbox()
    if not bbox:
        return expression
    expression = expression.crop(bbox)
    padded = Image.new("RGBA", (expression.width + 16, expression.height + 16), (0, 0, 0, 0))
    padded.alpha_composite(expression, dest=(8, 8))
    return padded


def _expression_data_url(expression_name: str, display_width: int, display_height: int) -> tuple[str | None, int, int]:
    expression = _prepared_expression_image(expression_name)
    if expression is None:
        return None, 0, 0
    display_width = max(20, display_width)
    display_height = max(20, display_height)
    expression = expression.resize((display_width, display_height))
    buffer = io.BytesIO()
    expression.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}", display_width, display_height


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _canvas_background_preview(image_path: str, canvas_height: int) -> Image.Image:
    base_image = Image.open(image_path).convert("RGBA")
    preview = Image.new("RGBA", base_image.size, (255, 252, 246, 255))
    preview.alpha_composite(base_image)
    return preview.resize((DISPLAY_WIDTH, canvas_height))


def _canvas_initial_expression(image_path: str, overlay_position: dict, expression_name: str) -> tuple[dict, int]:
    base_image = Image.open(image_path)
    scale = DISPLAY_WIDTH / base_image.width
    display_height = int(base_image.height * scale)
    preview_image = _canvas_background_preview(image_path, display_height)
    expression_image = _prepared_expression_image(expression_name)
    if expression_image is None:
        return _canvas_initial_drawing(image_path, overlay_position)
    overlay_x, overlay_y, new_width, new_height = _resolve_expression_geometry(base_image.width, base_image.height, overlay_position, expression_name)
    display_width = max(20, int(new_width * scale))
    display_height_target = max(20, int(new_height * scale))
    expression_src, display_width, display_height_px = _expression_data_url(expression_name, display_width, display_height_target)
    if not expression_src:
        return _canvas_initial_drawing(image_path, overlay_position)
    sticker = {
        "version": "4.4.0",
        "objects": [
            {
                "type": "image",
                "left": 0,
                "top": 0,
                "width": DISPLAY_WIDTH,
                "height": display_height,
                "scaleX": 1,
                "scaleY": 1,
                "src": _image_data_url(preview_image),
                "selectable": False,
                "evented": False,
                "hasControls": False,
                "hasBorders": False,
            },
            {
                "type": "image",
                "left": overlay_x * scale,
                "top": overlay_y * scale,
                "width": display_width,
                "height": display_height_px,
                "scaleX": 1,
                "scaleY": 1,
                "src": expression_src,
                "transparentCorners": False,
                "cornerColor": "#ff8eb1",
                "cornerStrokeColor": "#ff8eb1",
                "borderColor": "#ff8eb1",
                "cornerSize": 12,
                "hasControls": True,
                "hasBorders": True,
                "lockUniScaling": False,
            }
        ],
    }
    return sticker, display_height

def _canvas_display_height(image_path: str) -> int:
    base_image = Image.open(image_path)
    return int(base_image.height * (DISPLAY_WIDTH / base_image.width))


def _is_uploaded_processed_drawing(image_path: str) -> bool:
    try:
        return Path(image_path).resolve().parent == PROCESSED_DIR.resolve()
    except OSError:
        return False


def _resolve_expression_geometry(base_width: int, base_height: int, overlay_position: dict, expression_name: str) -> tuple[int, int, int, int]:
    expression_image = _prepared_expression_image(expression_name)
    default_aspect_ratio = (expression_image.height / expression_image.width) if expression_image else 1.0

    if "width_pct" in overlay_position:
        width_pct = overlay_position.get("width_pct", overlay_position.get("size", 22))
        height_pct = overlay_position.get("height_pct")
        width_px = max(16, int(base_width * (width_pct / 100.0)))
        if height_pct is not None:
            height_px = max(16, int(base_height * (height_pct / 100.0)))
        else:
            height_px = max(16, int(width_px * default_aspect_ratio))
        left_px = int(base_width * (overlay_position.get("left_pct", 50) / 100.0))
        top_px = int(base_height * (overlay_position.get("top_pct", 33) / 100.0))
    else:
        width_px = max(16, int(base_width * (overlay_position.get("size", 22) / 100.0)))
        height_px = max(16, int(width_px * default_aspect_ratio))
        left_px = int((base_width // 2 - width_px // 2) + overlay_position.get("x", 0))
        top_px = int((base_height // 3 - height_px // 2) + overlay_position.get("y", 0))

    left_px = max(0, min(left_px, max(0, base_width - width_px)))
    top_px = max(0, min(top_px, max(0, base_height - height_px)))
    return left_px, top_px, width_px, height_px


def _extract_overlay_from_canvas(obj: str, image_path: str, expression_name: str, canvas_result) -> dict | None:
    if not canvas_result or not canvas_result.json_data:
        return None
    objects = canvas_result.json_data.get("objects") or []
    if not objects:
        return None
    rect = next((item for item in objects if item.get("selectable", True)), None)
    if not rect:
        return None
    st.session_state.canvas_drawings[obj] = canvas_result.json_data
    base_image = Image.open(image_path)
    scale = DISPLAY_WIDTH / base_image.width
    rect_width = max(1, rect.get("width", 1) * rect.get("scaleX", 1)) / scale
    rect_height = max(1, rect.get("height", 1) * rect.get("scaleY", 1)) / scale
    rect_left = rect.get("left", 0) / scale
    rect_top = rect.get("top", 0) / scale
    width_pct = max(5, min(60, (rect_width / base_image.width) * 100))
    height_pct = max(5, min(60, (rect_height / base_image.height) * 100))
    return {
        "left_pct": (rect_left / base_image.width) * 100,
        "top_pct": (rect_top / base_image.height) * 100,
        "width_pct": width_pct,
        "height_pct": height_pct,
        "size": int(width_pct),
    }


def drawing_stage_page() -> None:
    _step_progress()
    st.markdown('<div class="hero-title">Upload and Place Expressions</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Upload child drawings, then place expressions only on characters. If a character drawing is missing, the app will create a temporary story-specific stand-in just for the video.</div>', unsafe_allow_html=True)

    if not st.session_state.selected_objects:
        st.warning("Choose lesson characters first.")
        return

    uploads = {}
    drawable_objects = [obj for obj in st.session_state.selected_objects if should_request_child_drawing(obj)]
    upload_columns = st.columns(2)
    for index, obj in enumerate(drawable_objects):
        with upload_columns[index % 2]:
            uploaded = st.file_uploader(f"Drawing for {obj}", type=["png", "jpg", "jpeg"], key=f"upload_{obj}")
            if uploaded is not None:
                uploads[obj] = uploaded
    background_objects = [obj for obj in st.session_state.selected_objects if not should_request_child_drawing(obj)]
    if background_objects:
        st.info(f"These are part of the scene background, so children do not need to draw them: {', '.join(background_objects)}")

    if st.button("Process Drawings", type="primary", width="stretch"):
        if not uploads:
            st.warning("Upload at least one drawing before processing.")
        else:
            processed_any = False
            for obj, uploaded in uploads.items():
                try:
                    processed_path = process_uploaded_drawing(uploaded, obj)
                    st.session_state.processed_drawings[obj] = processed_path
                    st.session_state.overlay_positions.setdefault(obj, DEFAULT_POSITION.copy())
                    st.session_state.canvas_drawings.pop(obj, None)
                    st.session_state.pending_overlay_positions.pop(obj, None)
                    processed_any = True
                except Exception as exc:
                    st.error(f"Could not process {obj}: {exc}")
            if processed_any:
                st.success("Drawings are ready for placement.")

    expression_names = available_expression_names(EXPRESSIONS_DIR)
    preview_expression = st.selectbox(
        "Expression style for placement preview",
        expression_names,
        index=expression_names.index("happy") if "happy" in expression_names else 0,
    )
    if st.session_state.get("canvas_expression_name") != preview_expression:
        st.session_state.canvas_drawings = {}
        st.session_state.canvas_seed_drawings = {}
        st.session_state.canvas_expression_name = preview_expression

    preview_objects = [
        obj for obj in st.session_state.selected_objects
        if obj in st.session_state.processed_drawings
        and _is_uploaded_processed_drawing(st.session_state.processed_drawings[obj])
        and should_apply_expression(obj)
    ]
    non_expression_objects = [
        obj for obj in st.session_state.selected_objects
        if obj in st.session_state.processed_drawings
        and _is_uploaded_processed_drawing(st.session_state.processed_drawings[obj])
        and not should_apply_expression(obj)
    ]
    if non_expression_objects:
        st.info(f"Expressions are skipped for objects: {', '.join(non_expression_objects)}")

    if preview_objects:
        active_object = st.selectbox(
            "Choose character to place expression on",
            preview_objects,
            key="active_expression_object",
        )
        image_path = st.session_state.processed_drawings[active_object]
        st.subheader(active_object)
        st.caption("Drag and resize the expression sticker. The preview updates as you move it, then click OK to save the current position.")

        current_position = st.session_state.pending_overlay_positions.get(
            active_object,
            st.session_state.overlay_positions.setdefault(active_object, DEFAULT_POSITION.copy()),
        )
        canvas_height = _canvas_display_height(image_path)
        canvas_version = int(Path(image_path).stat().st_mtime)
        canvas_revision = st.session_state.canvas_revisions.get(active_object, 0)
        canvas_key = f"canvas_{active_object}_{preview_expression}_{canvas_version}_{canvas_revision}"
        initial_drawing = st.session_state.canvas_seed_drawings.get(canvas_key)
        if not initial_drawing:
            initial_drawing = st.session_state.canvas_drawings.get(active_object)
            if not initial_drawing:
                initial_drawing, canvas_height = _canvas_initial_expression(image_path, current_position, preview_expression)
            st.session_state.canvas_seed_drawings[canvas_key] = initial_drawing

        left_col, right_col = st.columns([1.2, 1])
        with left_col:
            canvas_result = st_canvas(
                fill_color="rgba(255, 212, 120, 0.18)",
                stroke_width=4,
                stroke_color="#ff8eb1",
                background_color="#fffcf6",
                update_streamlit=True,
                height=canvas_height,
                width=DISPLAY_WIDTH,
                drawing_mode="transform",
                initial_drawing=initial_drawing,
                display_toolbar=False,
                key=canvas_key,
            )
            live_position = _extract_overlay_from_canvas(active_object, image_path, preview_expression, canvas_result)
            if live_position:
                st.session_state.pending_overlay_positions[active_object] = live_position

            save_disabled = active_object not in st.session_state.pending_overlay_positions
            if st.button(f"OK for {active_object}", key=f"save_{active_object}", width="stretch", disabled=save_disabled):
                confirmed_position = st.session_state.pending_overlay_positions.get(active_object)
                if confirmed_position:
                    st.session_state.overlay_positions[active_object] = confirmed_position
                    st.session_state.canvas_drawings.pop(active_object, None)
                    st.session_state.canvas_revisions[active_object] = st.session_state.canvas_revisions.get(active_object, 0) + 1
                    st.rerun()
                else:
                    st.info("Move the expression sticker first, then click OK.")

            if st.button(f"Reset {active_object}", key=f"reset_{active_object}", width="stretch"):
                st.session_state.overlay_positions[active_object] = DEFAULT_POSITION.copy()
                st.session_state.pending_overlay_positions.pop(active_object, None)
                st.session_state.canvas_drawings.pop(active_object, None)
                st.session_state.canvas_revisions[active_object] = st.session_state.canvas_revisions.get(active_object, 0) + 1
                st.rerun()

        with right_col:
            preview_position = st.session_state.pending_overlay_positions.get(
                active_object,
                st.session_state.overlay_positions.get(active_object, current_position),
            )
            preview_image = render_overlay_preview(
                image_path=image_path,
                expression_name=preview_expression,
                overlay_position=preview_position,
            )
            st.image(preview_image, caption=f"{active_object} live preview", width="stretch")
    elif st.session_state.processed_drawings:
        st.info("The processed drawings on this lesson do not need expression stickers.")
    else:
        st.info("Upload and process a child drawing to start placing expressions. Missing characters will be auto-created later during video generation only.")

    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        if st.button("Back to Summary", width="stretch"):
            persist_project_state()
            _go_to("lesson_details")
    with nav_right:
        if st.button("Next: Generate Video", type="primary", width="stretch"):
            persist_project_state()
            _go_to("video_generation")


def video_generation_page() -> None:
    _step_progress()
    st.markdown('<div class="hero-title">Generate the Lesson Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Create the final video and watch the progress as each scene is prepared.</div>', unsafe_allow_html=True)

    scenes = build_story_scenes(st.session_state.lesson_text, st.session_state.selected_objects)
    st.markdown(
        f"""
        <div class="summary-card">
            <strong style="color:#000000;">Lesson:</strong> {st.session_state.lesson_title}<br>
            <strong style="color:#000000;">Characters:</strong> {', '.join(st.session_state.selected_objects)}<br>
            <strong style="color:#000000;">Scenes:</strong> {len(scenes)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress_box = st.container()
    if st.button("Generate Final Video", type="primary", width="stretch"):
        progress_bar = progress_box.progress(0)
        progress_text = progress_box.empty()

        def update_progress(percent: int, message: str) -> None:
            progress_bar.progress(max(0, min(100, percent)))
            progress_text.info(f"{percent}% - {message}")

        available_drawings = {
            obj: path
            for obj, path in st.session_state.processed_drawings.items()
            if Path(path).exists()
        }
        update_progress(5, "Preparing assets")

        try:
            final_video_path, asset_map = generate_lesson_video(
                lesson_title=st.session_state.lesson_title,
                lesson_text=st.session_state.lesson_text,
                scenes=scenes,
                selected_objects=st.session_state.selected_objects,
                processed_drawings=available_drawings,
                overlay_positions=st.session_state.overlay_positions,
                auto_generate_missing_drawings=st.session_state.auto_generate_missing_drawings,
                use_ai_generated_characters=st.session_state.use_ai_generated_characters,
                add_narration_audio=st.session_state.add_narration_audio,
                progress_callback=update_progress,
            )
        except Exception as exc:
            progress_text.error(f"Video generation failed: {exc}")
            return

        st.session_state.processed_drawings.update(asset_map)
        st.session_state.final_video_path = final_video_path
        persist_project_state()
        update_progress(100, "Video ready")
        st.success("Lesson video generated successfully.")

    if st.session_state.final_video_path and Path(st.session_state.final_video_path).exists():
        st.video(st.session_state.final_video_path)
        st.caption(st.session_state.final_video_path)

    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        if st.button("Back to Placement", width="stretch"):
            _go_to("drawing_stage")
    with nav_right:
        if st.button("Start Over", width="stretch"):
            clear_temporary_media()
            st.session_state.processed_drawings = {}
            st.session_state.overlay_positions = {}
            st.session_state.canvas_drawings = {}
            st.session_state.canvas_seed_drawings = {}
            st.session_state.canvas_revisions = {}
            st.session_state.pending_overlay_positions = {}
            st.session_state.final_video_path = None
            _go_to("lesson_select")


def main() -> None:
    st.set_page_config(page_title="AI Lesson Video Generator", layout="wide")
    apply_theme()
    initialize_state()

    st.markdown('<div class="page-shell">', unsafe_allow_html=True)
    if st.session_state.current_page == "lesson_select":
        lesson_cards_page()
    elif st.session_state.current_page == "lesson_details":
        lesson_details_page()
    elif st.session_state.current_page == "drawing_stage":
        drawing_stage_page()
    else:
        video_generation_page()
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()




