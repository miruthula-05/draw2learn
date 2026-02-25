# AI-Powered Cinematic Lesson Video Generator

**Internship project** – Miruthula Sri A
**Company:** curiosense pvt ltd 
**Period:** February - May 2026
**Location:** Bengaluru

Turn 2nd/3rd grade lessons into short cinematic videos using children's own drawings as main animated characters.

## Current Features (Week 1)
- Teacher pastes lesson text
- Basic AI suggestion of characters/objects (using NLTK)
- Simple web interface with Streamlit

## Planned Features
- Select which characters children draw
- Upload drawings → background removal
- Animate drawings (Meta Animated Drawings)
- Generate missing assets (Stable Diffusion in Colab)
- Create video script + narration (gTTS)
- Stitch clips into final MP4 (MoviePy)

## Tech Stack (so far)
- Python 3.10+
- Streamlit (web UI)
- NLTK (noun extraction)
- rembg (background removal)
- gTTS (text-to-speech)
- MoviePy (video editing)

Heavy tasks (image/video generation) will run in Google Colab (free GPU).

## How to Run (Local)
```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux

# 2. Install libraries
pip install streamlit pillow opencv-python nltk rembg gtts moviepy

# 3. Download NLTK data (run once)
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# 4. Start the app
streamlit run app.py