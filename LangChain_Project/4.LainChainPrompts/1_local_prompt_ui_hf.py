import asyncio
import torch
import streamlit as st
from transformers import pipeline

# Ensure the event loop is set correctly
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set Streamlit app title
st.title("LLM Prompt UI with Hugging Face")

# Load model
@st.cache_resource
def load_pipeline():
    repo_id = "EleutherAI/gpt-neo-1.3B"  # Change this to your model
    return pipeline(
        "text-generation",
        model=repo_id,
        device=0 if torch.cuda.is_available() else -1,  # Auto-detect GPU/CPU
        truncation=True,
        max_length=512  # Ensures output is controlled
    )

prompt_pipeline = load_pipeline()

# Input box for user prompt
user_input = st.text_area("Enter your prompt:", "Once upon a time...")

if st.button("Generate"):
    with st.spinner("Generating response..."):
        response = prompt_pipeline(user_input, max_length=512, truncation=True)[0]['generated_text']
        st.write("### AI Response")
        st.write(response)
