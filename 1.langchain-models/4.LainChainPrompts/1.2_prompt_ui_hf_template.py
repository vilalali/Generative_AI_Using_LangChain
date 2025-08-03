import os
import streamlit as st
from dotenv import load_dotenv  # Load environment variables
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Retrieve API token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    st.error("Hugging Face API token is missing. Make sure `.env` contains `HUGGINGFACEHUB_API_TOKEN` and restart the app.")
    st.stop()

# Initialize Hugging Face Inference Client
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token=HUGGINGFACEHUB_API_TOKEN
)

# Define Prompt Template
prompt_template = PromptTemplate.from_template(
    """Summarize the following research paper in a {style_input} way, limited to {length_input} sentences:
    {paper_input}"""
)

# Streamlit UI
st.title("LangChain Hugging Face Chatbot")

# Inputs
paper_input = st.text_area("Enter the research paper content:")
style_input = st.selectbox("Select summary style:", ["Concise", "Detailed", "Bullet Points"])
length_input = st.slider("Summary Length (in sentences):", 1, 10, 5)

# Button to summarize
if st.button("Summarize"):
    with st.spinner("Generating summary..."):
        try:
            # Format prompt using LangChain
            formatted_prompt = prompt_template.format(
                paper_input=paper_input,
                style_input=style_input,
                length_input=length_input
            )

            # Call Hugging Face Inference API
            result = client.text_generation(formatted_prompt, max_new_tokens=150)

            # Display Result
            st.write("### Summary:")
            st.write(result)

        except Exception as e:
            st.error(f"Error: {e}")
