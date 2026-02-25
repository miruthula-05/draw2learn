import streamlit as st
import nltk
from nltk import pos_tag, word_tokenize
import json
import os
from rembg import remove
from PIL import Image
import io

# Assume NLTK and rembg are pre-downloaded (run pre-load scripts above once)

st.set_page_config(page_title="AI Lesson Video Generator", layout="wide")

st.title("AI Lesson Video Generator for Kids")
st.markdown("**2nd/3rd Grade Lessons – Using Children's Own Drawings!** 🎨✨")

# Session state
if 'suggested_characters' not in st.session_state:
    st.session_state.suggested_characters = []
if 'selected_for_drawing' not in st.session_state:
    st.session_state.selected_for_drawing = []
if 'processed_drawings' not in st.session_state:
    st.session_state.processed_drawings = {}

# Step 1: Lesson Input
st.header("Step 1: Enter the Lesson")
lesson_text = st.text_area(
    "Paste your lesson text here:",
    height=150,
    placeholder="Example: The knight saves the dragon from the castle."
)

# Step 2: Suggest
if st.button("Suggest Characters & Objects"):
    if lesson_text.strip():
        try:
            tokens = word_tokenize(lesson_text.lower())
            tagged = pos_tag(tokens)
            
            stop_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by', 'from', 'about'}
            
            suggested = [word for word, pos in tagged if pos.startswith('NN') and len(word) > 3 and word not in stop_words]
            suggested = list(set(suggested))
            suggested.sort(key=str.lower)
            
            st.session_state.suggested_characters = suggested
            
            st.success(f"Found {len(suggested)} suggestions!")
        except Exception as e:
            st.error(f"Suggestion failed: {e}")
    else:
        st.warning("Enter lesson text first!")

# Show suggestions + checkboxes
if st.session_state.suggested_characters:
    st.header("Step 2: Select for Children to Draw")
    selected = []
    for char in st.session_state.suggested_characters:
        if st.checkbox(char, key=f"char_{char}"):
            selected.append(char)
    
    st.session_state.selected_for_drawing = selected
    
    if selected:
        st.info(f"Children draw: {', '.join(selected)}")
    else:
        st.warning("Select at least one item.")

# Step 3: Upload & Process
if st.session_state.selected_for_drawing:
    st.header("Step 3: Upload Drawings")
    uploaded_files = {}
    for char in st.session_state.selected_for_drawing:
        file = st.file_uploader(f"Drawing for {char}", type=['jpg', 'png'], key=f"upload_{char}")
        if file:
            uploaded_files[char] = file

    if st.button("Process Drawings"):
        if uploaded_files:
            for char, file in uploaded_files.items():
                try:
                    img = Image.open(file)
                    cleaned = remove(img)
                    
                    buf = io.BytesIO()
                    cleaned.save(buf, format="PNG")
                    buf.seek(0)
                    
                    st.session_state.processed_drawings[char] = buf
                except Exception as e:
                    st.error(f"Error on {char}: {e}")
            
            st.success("Processing done!")
        else:
            st.warning("Upload drawings first.")

    # Previews
    if st.session_state.processed_drawings:
        st.subheader("Cleaned Drawings")
        cols = st.columns(3)
        for i, (char, buf) in enumerate(st.session_state.processed_drawings.items()):
            with cols[i % 3]:
                st.image(buf, caption=char)

# Save button
if st.button("Save Progress"):
    save_data = {
        "lesson_text": lesson_text,
        "selected": st.session_state.selected_for_drawing
    }
    with open("saved_lesson.json", "w") as f:
        json.dump(save_data, f, indent=4)
    st.success("Saved!")