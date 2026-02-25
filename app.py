import streamlit as st
import nltk
from nltk import pos_tag, word_tokenize

# Download NLTK data (run once outside the app if needed)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

st.set_page_config(page_title="AI Lesson Video Generator", layout="wide")

st.title("AI Lesson Video Generator for Kids")
st.markdown("**2nd/3rd Grade Lessons – Using Children's Drawings!**")

st.header("Step 1: Enter the Lesson")

lesson_text = st.text_area(
    "Paste your lesson text here:",
    height=150,
    placeholder="Example: The brave knight saves the dragon from the tall castle."
)

if st.button("Suggest Characters"):
    if lesson_text.strip():
        tokens = word_tokenize(lesson_text.lower())
        tagged = pos_tag(tokens)
        suggested = [word for word, pos in tagged if pos in ('NN', 'NNP') and len(word) > 2]
        suggested = list(set(suggested))  # remove duplicates
        
        if suggested:
            st.success(f"Found {len(suggested)} suggestions!")
            st.write("Suggested characters / objects:")
            st.write(", ".join(suggested))
        else:
            st.info("No clear nouns found. Try a more detailed lesson text.")
    else:
        st.warning("Please enter some lesson text first!")