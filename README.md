# AI-Powered Lesson Video Generator

Turn predefined lessons into classroom-ready videos by using children's drawings as story characters, with optional AI image hooks and narration.

## What the app does
- Shows lesson names as selection cards on the first page
- Opens a lesson summary page with fixed lesson content and character selection
- Lets the teacher upload drawings on a dedicated page
- Supports drag-and-place expression positioning with a canvas tool instead of sliders
- Uses a dedicated video generation page with progress percentage feedback
- Uses a pastel, child-friendly interface style

## Project structure
- `app.py`: Streamlit UI and workflow
- `lessons.py`: predefined lesson content and default lesson objects
- `lesson_parser.py`: sentence splitting, smarter object matching, emotion and setting detection
- `media_pipeline.py`: character/background generation, scene composition, narration attachment, and video export
- `ai_generation.py`: optional AI image hooks and prompt creation
- `narration.py`: narration audio generation with multiple fallbacks
- `storage.py`: project save/load helpers
- `config.py`: shared paths for outputs and caches

## Setup
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## UI flow
1. Lesson selection cards
2. Lesson summary and character selection
3. Drawing upload and drag placement
4. Video generation with progress
