import streamlit as st
import nltk
from nltk import pos_tag, word_tokenize
import json
import os

# Download NLTK data (run once outside the app if needed, or here with check)
if not os.path.exists("nltk_data"):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

st.set_page_config(page_title="AI Lesson Video Generator", layout="wide")

st.title("AI Lesson Video Generator for Kids")
st.markdown("**2nd/3rd Grade Lessons – Using Children's Own Drawings!** 🎨✨")

st.header("Step 1: Enter the Lesson")

# Text area for lesson
lesson_text = st.text_area(
    "Paste your lesson text here (story, science, moral, etc.):",
    height=180,
    placeholder="Example: The brave knight saves the friendly dragon from the tall castle in the dark forest."
)

# Session state to store suggestions and selections
if 'suggested_characters' not in st.session_state:
    st.session_state.suggested_characters = []
if 'selected_for_drawing' not in st.session_state:
    st.session_state.selected_for_drawing = []

# Button to suggest characters
# Button to suggest characters
if st.button("Suggest Characters & Objects"):
    if lesson_text.strip():
        tokens = word_tokenize(lesson_text.lower())
        tagged = pos_tag(tokens)
        
        # Improved filter: only concrete nouns (NN, NNS, NNP, NNPS)
        # Exclude very short words and common stop-words that are not drawable
        stop_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by', 'from', 'about'}
        
        suggested = []
        for word, pos in tagged:
            if pos.startswith('NN') and len(word) > 3 and word not in stop_words:
                suggested.append(word)
        
        suggested = list(set(suggested))  # remove duplicates
        suggested.sort(key=str.lower)     # sort alphabetically
        
        st.session_state.suggested_characters = suggested
        
        if suggested:
            st.success(f"Found {len(suggested)} drawable suggestions!")
        else:
            st.info("No clear drawable objects found. Try a more descriptive lesson with people, animals, things.")
    else:
        st.warning("Please enter some lesson text first!")

# Show suggestions and checkboxes if we have them
if st.session_state.suggested_characters:
    st.header("Step 2: Select Characters / Objects")
    st.write("Check the ones **children will draw** themselves. Unchecked = AI will generate.")
    
    selected = []
    for char in st.session_state.suggested_characters:
        # Optional: show example hint for drawable items
        hint = " (good for drawing!)" if len(char) > 4 else ""
        if st.checkbox(f"{char}", key=f"char_{char}"):
            selected.append(char)
    
    st.session_state.selected_for_drawing = selected
    
    # Show summary
    if selected:
        st.info(f"**Children will draw:** {', '.join(selected)}")
        ai_generated = [c for c in st.session_state.suggested_characters if c not in selected]
        if ai_generated:
            st.info(f"**AI will generate:** {', '.join(ai_generated)}")
        else:
            st.info("All suggested items will be drawn by children – great!")
    else:
        st.warning("Select at least one item for children to draw.")

# Save lesson button
st.header("Step 3: Save Lesson Progress")
if st.button("Save Lesson (for later use)"):
    if lesson_text.strip() and st.session_state.suggested_characters:
        save_data = {
            "lesson_text": lesson_text,
            "suggested_characters": st.session_state.suggested_characters,
            "selected_for_drawing": st.session_state.selected_for_drawing,
            "timestamp": str(st.session_state.get('last_run', 'now'))
        }
        
        save_filename = "saved_lesson.json"
        with open(save_filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        
        st.success(f"Lesson saved to **{save_filename}**!")
        st.download_button(
            label="Download saved lesson",
            data=json.dumps(save_data, indent=4),
            file_name="saved_lesson.json",
            mime="application/json"
        )
    else:
        st.warning("Enter lesson text and suggest characters first.")

# Future steps placeholder
st.markdown("---")
st.subheader("Coming next...")
st.write("- Upload drawings")
st.write("- Generate missing assets")
st.write("- Animate & create video")